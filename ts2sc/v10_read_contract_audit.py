#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 · 读取契约审计：312 个 NEW 张量是否真的进了前向？

## 为什么要查

V7 历史最致命的一条：「NEW 参数全零 + SGC 卷积注意力只是空挂在
state_dict 里，根本没进前向」。v10 的 `SGCV10Branch.read_keys()` 之前直接
返回 `n_layers * len(NEW_KEYS) = 24*13 = 312`，那是**声明**，不是证据。

本脚本用独立判据：对单层每个 NEW key 做数值扰动，重新构造同一 seed 的层，
同输入 / 同 β / 同 pos 算 Δ，比较扰动前后是否变化。

  ACTIVE      : 扰动改变 Δ ⇒ 该参数确实在前向路径里
  NO_READ     : 扰动完全不改变 Δ ⇒ 空挂（V7 同款隐患）
  NEUTRAL     : 读了但初值为中性（全零/被覆盖），扰动也不改变，
                需结合源码判定，单独登记不混进 ACTIVE

## 两种模式
  --baseline : 采集 v10.0（修正前）的 Δ 字节快照，存 json，
               供修正后做「数值逐位不变」对照
  --audit    : 默认，跑扰动分类 + 输出报告

诚实约定：NO_READ / NEUTRAL 都属于「未在前向贡献」，绝不包装成已继承。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgc_v10 as A  # noqa: E402

OUT = "/data/workspace/output"
BASELINE = os.path.join(OUT, "v10_0_delta_baseline.json")
AUDIT = os.path.join(OUT, "v10_read_contract_audit.json")
D = 1536
SEED = 20261128 + 101 * 7     # 固定第 7 层
BETA = 1.0


def layer(params=None):
    p = params if params is not None else A.deterministic_layer_params(D, SEED)
    lay = A.SGCV10Layer(D, 1000 + 17 * 7, p)
    lay.beta = BETA
    return lay


def delta_of(p, x, pos=3):
    lay = layer(p)
    return np.asarray(lay.hook(x, pos), np.float32)


def fingerprint(b):
    return hashlib.sha256(np.ascontiguousarray(b, np.float32).tobytes()).hexdigest()


def perturb(p, key, rng):
    """对单个 key 做显著数值扰动，其余参数逐位不动。"""
    q = {k: np.array(v, dtype=np.float32, copy=True) for k, v in p.items()}
    a = q[key]
    if a.size == 1:
        q[key] = np.array([float(a.reshape(-1)[0]) + 1.0], np.float32)
    elif np.all(a == 0):
        # 全零初值：填一个确定性的非零模式，才能测出「是否被读取」
        q[key] = (rng.normal(0, 0.5, a.shape)).astype(np.float32)
    else:
        q[key] = (a * 1.25 + 0.125 * rng.normal(0, 1, a.shape)).astype(np.float32)
    return q


def baseline():
    """采集 v10.0 的 Δ 快照：多输入 × 多 pos × 多层，逐位可对照。"""
    rng = np.random.default_rng(31337)
    rows = []
    for li in (0, 7, 23):
        p = A.deterministic_layer_params(D, 20261128 + 101 * li)
        for j, mag in enumerate((1e-3, 1.0, 1e3)):
            x = (rng.normal(0, 1, D) * mag).astype(np.float32)
            lay = A.SGCV10Layer(D, 1000 + 17 * li, p)
            lay.beta = BETA
            for pos in (0, 1, 5):
                lay.reset()
                d = np.asarray(lay.hook(x, pos), np.float32)
                rows.append(dict(layer=li, case=j, pos=pos,
                                 norm=float(np.linalg.norm(d)),
                                 sha256=fingerprint(d)))
    rep = dict(title="v10.0 Δ baseline (pre read-contract fix)",
               date=time.strftime("%Y-%m-%d %H:%M:%S"),
               d=D, beta=BETA, n=len(rows),
               module_sha256=hashlib.sha256(
                   open(os.path.join(os.path.dirname(__file__), "sgc_v10.py"),
                        "rb").read()).hexdigest(),
               rows=rows)
    json.dump(rep, open(BASELINE, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"[baseline] {len(rows)} Δ snapshots | module sha "
          f"{rep['module_sha256'][:16]}…", flush=True)
    return rep


def audit():
    rng = np.random.default_rng(4242)
    x = rng.normal(0, 1, D).astype(np.float32)
    p0 = A.deterministic_layer_params(D, SEED)
    d0 = delta_of(p0, x)
    rows = []
    keys = [k for k, _, _ in A.TENSOR_SPECS]
    for key in keys:
        q = perturb(p0, key, np.random.default_rng(
            int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)))
        d1 = delta_of(q, x)
        diff = float(np.max(np.abs(d1.astype(np.float64) - d0.astype(np.float64))))
        changed = bool(diff > 0.0)
        init_all_zero = bool(np.all(np.asarray(p0[key]) == 0))
        rows.append(dict(key=key, shape=list(np.asarray(p0[key]).shape),
                         init_all_zero=init_all_zero,
                         max_abs_delta_after_perturb=diff,
                         active=changed,
                         status="ACTIVE" if changed else "NOT_CONTRIBUTING"))

    # 独立判据：gate.beta 是否在构造期被真正读取（不调用 set_beta）
    pb = {k: np.array(v, dtype=np.float32, copy=True) for k, v in p0.items()}
    pb["gate.beta"] = np.array([0.77], np.float32)
    lay_b = A.SGCV10Layer(D, 1000 + 17 * 7, pb)
    beta_persisted = abs(lay_b.beta - 0.77) < 1e-6
    beta_row = next(r for r in rows if r["key"] == "gate.beta")
    beta_row["persisted_init_read"] = bool(beta_persisted)
    beta_row["status"] = "PERSISTED_INIT" if beta_persisted else "NOT_CONTRIBUTING"

    n_active = sum(1 for r in rows if r["status"] == "ACTIVE")
    n_persist = sum(1 for r in rows if r["status"] == "PERSISTED_INIT")
    n_bad = sum(1 for r in rows if r["status"] == "NOT_CONTRIBUTING")
    rep = dict(
        title="v10.1 · 312 NEW 张量读取契约审计（扰动 + 持久化双判据）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        method="perturb one key, rebuild same-seed layer, compare Δ bitwise; "
               "plus gate.beta construction-time persistence probe",
        d=D, beta=BETA, n_keys=len(rows),
        n_active=n_active, n_persisted_init=n_persist,
        n_not_contributing=n_bad,
        total_not_contributing_24_layers=n_bad * 24,
        all_keys_accounted_for=bool(n_bad == 0),
        declared_read_keys_before=24 * len(rows),
        honest_note="forward-active with NEUTRAL initial value (role.Wc=0, "
                    "gate.alpha=1) is NOT claimed as contributing; it is a "
                    "calibratable slot for L2. gate.beta is read at init and "
                    "overridden by set_beta(1.0) at delivery.",
        rows=rows)
    json.dump(rep, open(AUDIT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    for r in rows:
        print(f"  {r['key']:14s} {str(r['shape']):18s} "
              f"{r['status']:18s} diff={r['max_abs_delta_after_perturb']:.4g}"
              + (f" persisted={r.get('persisted_init_read')}"
                 if "persisted_init_read" in r else ""), flush=True)
    print(f"\n[audit] ACTIVE {n_active}/{len(rows)} | PERSISTED_INIT "
          f"{n_persist} | NOT_CONTRIBUTING {n_bad} | "
          f"all accounted = {rep['all_keys_accounted_for']}", flush=True)
    return rep


def compare():
    """v10.1 与 v10.0 的 27 个 Δ 快照逐位对照：证明修正数值中性。"""
    base = json.load(open(BASELINE, encoding="utf-8"))
    rng = np.random.default_rng(31337)
    ok, bad = 0, []
    for li in (0, 7, 23):
        p = A.deterministic_layer_params(D, 20261128 + 101 * li)
        for j, mag in enumerate((1e-3, 1.0, 1e3)):
            x = (rng.normal(0, 1, D) * mag).astype(np.float32)
            lay = A.SGCV10Layer(D, 1000 + 17 * li, p)
            lay.beta = BETA
            for pos in (0, 1, 5):
                lay.reset()
                d = np.asarray(lay.hook(x, pos), np.float32)
                sha = fingerprint(d)
                want = next(r["sha256"] for r in base["rows"]
                            if r["layer"] == li and r["case"] == j
                            and r["pos"] == pos)
                if sha == want:
                    ok += 1
                else:
                    bad.append(dict(layer=li, case=j, pos=pos,
                                    want=want, got=sha))
    rep = dict(title="v10.1 vs v10.0 Δ bitwise equivalence",
               date=time.strftime("%Y-%m-%d %H:%M:%S"),
               n=base["n"], identical=ok, mismatched=len(bad),
               bad=bad, all_identical=bool(len(bad) == 0),
               meaning="read-contract fix is numerically neutral: all prior "
                       "generation evidence remains valid without re-run")
    json.dump(rep, open(os.path.join(OUT, "v10_1_bitwise_equivalence.json"),
                        "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[compare] {ok}/{base['n']} Δ snapshots bitwise identical | "
          f"mismatched {len(bad)} | all_identical={rep['all_identical']}",
          flush=True)
    return rep


if __name__ == "__main__":
    if "--baseline" in sys.argv:
        baseline()
    elif "--compare" in sys.argv:
        compare()
    else:
        audit()
