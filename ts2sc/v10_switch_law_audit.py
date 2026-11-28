#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10.2 · 开关法则真跑审计

用户指令：「完全的继承所有能力（开关法则，开门禁继承，不开就原模型能力）」

法则：  y = T(x) + 1[master] · Σ_f s_f · Δ_f(x)
        全关 (s_f=0 ∀f) ⇒ y ≡ T(x) 逐位（结构性 early-return）
        全开 (s_f=1 ∀f) ⇒ 与 v10.1 逐位相同（27/27 Δ 快照）

本脚本用**独立判据**验证，不复用任何声明：

  A  ALL_OFF      多层/多输入/多 pos 的 Δ 严格零向量；set_beta(1.0) 不复活
  B  SINGLE_OFF   从全开起，逐个关一个门，Δ 必须变化（默认初值）
  C  ACTIVATED    给中性初值槽注入非零探针后，13/13 单关都可观测
  D  ALL_ON       27/27 Δ 与 v10.0/v10.1 baseline 逐位相同
  E  RANDOM       32 组随机开关组合：全 finite / T1' 有界 / 全 0 严格零
  F  CONTRACT     9 个契约门切换前后 Δ 逐位相同（证明不偷改前向）
  G  DEFERRED     2 个延迟项置 True 必须被拒绝（不伪造继承）
  H  BRANCH       24 层 × 13 门广播、capability_state 三态自洽

诚实边界写在报告末尾，不包装成「31 项都在前向贡献」。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sgc_v10 as A  # noqa: E402

OUT = "/data/workspace/output"
BASELINE = os.path.join(OUT, "v10_0_delta_baseline.json")
AUDIT = os.path.join(OUT, "v10_switch_law_audit.json")
D = 1536
SEED = 20261128 + 101 * 7
LAYER_SEED = 1000 + 17 * 7


def fresh_params(activated=False):
    p = A.deterministic_layer_params(D, SEED)
    if activated:
        # 给中性初值槽注入非零探针，使「开关接线」可观测（不改默认交付态）
        r = np.random.default_rng(777)
        p["role.Wc"] = (r.normal(0, 0.01, p["role.Wc"].shape)).astype(np.float32)
        p["gate.beta"] = np.array([0.5], np.float32)
    return p


def make(params, caps=None):
    l = A.SGCV10Layer(D, LAYER_SEED, params)
    if caps is not None:
        l.set_capabilities(caps)
    # 统一 master=1.0（受 gate.beta 门控，全关态不会复活）
    l.beta = l._scaled_scalar("gate.beta", 1.0)
    return l


def delta(l, x, pos=3):
    l.reset()
    return np.asarray(l.hook(x, pos), np.float32)


def fp(b):
    return hashlib.sha256(np.ascontiguousarray(b, np.float32).tobytes()).hexdigest()


def phase_a(p):
    rng = np.random.default_rng(11)
    bad = []
    n = 0
    for mag in (1e-3, 1.0, 1e3):
        x = (rng.normal(0, 1, D) * mag).astype(np.float32)
        for pos in (0, 1, 5, 17):
            l = make(p, {k: 0.0 for k in A.CAP_KEYS})
            d = delta(l, x, pos)
            n += 1
            if not np.all(d == 0.0):
                bad.append(dict(mag=mag, pos=pos, norm=float(np.linalg.norm(d))))
    # 全关后强行 set_beta(1.0) 也不复活
    l = make(p, {k: 0.0 for k in A.CAP_KEYS})
    l.beta = l._scaled_scalar("gate.beta", 1.0)
    d = delta(l, rng.normal(0, 1, D).astype(np.float32), 2)
    resurrected = bool(not np.all(d == 0.0))
    return dict(n=n, n_nonzero=len(bad), bad=bad,
                all_bitwise_zero=bool(len(bad) == 0),
                resurrected_after_set_beta_1=resurrected,
                passed=bool(len(bad) == 0 and not resurrected))


def phase_b(p, activated=False):
    rng = np.random.default_rng(22)
    x = rng.normal(0, 1, D).astype(np.float32)
    l_on = make(p, {k: 1.0 for k in A.CAP_KEYS})
    d_on = delta(l_on, x)
    rows = []
    for k in A.CAP_KEYS:
        l = make(p, {k: 0.0})
        l.caps = {kk: (0.0 if kk == k else 1.0) for kk in A.CAP_KEYS}
        l._rebuild(p)
        l.beta = l._scaled_scalar("gate.beta", 1.0)
        d = delta(l, x)
        diff = float(np.max(np.abs(d.astype(np.float64)
                                  - d_on.astype(np.float64))))
        rows.append(dict(gate=k, max_abs_diff=diff,
                         observable=bool(diff > 0.0),
                         finite=bool(np.all(np.isfinite(d)))))
    n_obs = sum(1 for r in rows if r["observable"])
    return dict(mode="ACTIVATED" if activated else "DEFAULT_INIT",
                n_observable=n_obs, n_gates=len(rows), rows=rows,
                all_finite=all(r["finite"] for r in rows),
                passed=bool(all(r["finite"] for r in rows)))


def phase_d():
    base = json.load(open(BASELINE, encoding="utf-8"))
    rng = np.random.default_rng(31337)
    ok, bad = 0, []
    for li in (0, 7, 23):
        p = A.deterministic_layer_params(D, 20261128 + 101 * li)
        for j, mag in enumerate((1e-3, 1.0, 1e3)):
            x = (rng.normal(0, 1, D) * mag).astype(np.float32)
            l = A.SGCV10Layer(D, 1000 + 17 * li, p)
            l.beta = 1.0
            for pos in (0, 1, 5):
                sha = fp(delta(l, x, pos))
                want = next(r["sha256"] for r in base["rows"]
                            if r["layer"] == li and r["case"] == j
                            and r["pos"] == pos)
                if sha == want:
                    ok += 1
                else:
                    bad.append((li, j, pos))
    return dict(n=base["n"], identical=ok, mismatched=len(bad), bad=bad,
                all_identical=bool(len(bad) == 0))


def phase_e(p):
    rng = np.random.default_rng(33)
    x = rng.normal(0, 1, D).astype(np.float32)
    rx = float(np.linalg.norm(x.astype(np.float64)))
    bound = 1.0 * A.EPS_DEFAULT * rx
    rows = []
    for i in range(32):
        if i == 0:
            caps = {k: 0.0 for k in A.CAP_KEYS}
        elif i == 1:
            caps = {k: 1.0 for k in A.CAP_KEYS}
        else:
            caps = {k: float(rng.choice([0.0, 0.25, 0.5, 1.0]))
                    for k in A.CAP_KEYS}
        l = make(p, caps)
        d = delta(l, x, int(rng.integers(0, 8)))
        nd = float(np.linalg.norm(d.astype(np.float64)))
        rows.append(dict(i=i, n_zero_gates=sum(1 for v in caps.values() if v == 0),
                         norm=nd, finite=bool(np.all(np.isfinite(d))),
                         within_T1=bool(nd <= bound * (1 + 1e-6)),
                         all_off_strict_zero=bool(
                             all(v == 0 for v in caps.values())
                             and np.all(d == 0.0))))
    return dict(n=len(rows),
                all_finite=all(r["finite"] for r in rows),
                all_within_T1=all(r["within_T1"] for r in rows),
                zero_combo_strictly_zero=bool(rows[0]["all_off_strict_zero"]),
                passed=bool(all(r["finite"] and r["within_T1"] for r in rows)
                            and rows[0]["all_off_strict_zero"]),
                rows=rows)


def phase_fgh():
    cfg = dict(hidden_size=D, num_hidden_layers=24)
    pbl = {i: A.deterministic_layer_params(D, 20261128 + 101 * i)
           for i in range(24)}
    br = A.SGCV10Branch(cfg, pbl)
    br.set_beta(1.0)
    x = np.random.default_rng(44).normal(0, 1, D).astype(np.float32)
    before = fp(delta(br.layers[7], x))
    changed = []
    for cid in list(br.contract_flags):
        br.set_contract(cid, False)
        if fp(delta(br.layers[7], x)) != before:
            changed.append(cid)
        br.set_contract(cid, True)
    # DEFERRED 必须拒绝置 True
    refused = []
    for cid in list(br.deferred_flags):
        try:
            br.set_contract(cid, True)
            refused.append(dict(id=cid, refused=False))
        except RuntimeError:
            refused.append(dict(id=cid, refused=True))
    # 广播正确性
    br.all_off()
    st_off = br.capability_state()
    br.all_on()
    br.set_beta(1.0)
    st_on = br.capability_state()
    return dict(
        F=dict(contract_flags_changed_forward=changed,
               n_changed=len(changed),
               passed=bool(len(changed) == 0)),
        G=dict(deferred=refused,
               passed=all(r["refused"] for r in refused)),
        H=dict(n_layers=st_off["n_layers"], n_gates=st_off["n_gates"],
               all_off=st_off["all_off"], all_on=st_on["all_on"],
               law=st_off["law"],
               passed=bool(st_off["all_off"] and st_on["all_on"]
                           and st_off["n_layers"] == 24
                           and st_off["n_gates"] == 13)))


def main():
    t0 = time.time()
    p = fresh_params(False)
    pa = fresh_params(True)

    print("[A] ALL_OFF strict zero regression", flush=True)
    a = phase_a(p)
    print("   ", a["n"], "cases | nonzero", a["n_nonzero"],
          "| resurrected", a["resurrected_after_set_beta_1"],
          "| passed", a["passed"], flush=True)

    print("[B] SINGLE_OFF differential (default init)", flush=True)
    b = phase_b(p, False)
    print("    observed", b["n_observable"], "/", b["n_gates"],
          "| neutral:", [r["gate"] for r in b["rows"]
                         if not r["observable"]], flush=True)

    print("[C] SINGLE_OFF differential (activated neutral slots)", flush=True)
    c = phase_b(pa, True)
    print("    observed", c["n_observable"], "/", c["n_gates"], flush=True)

    print("[D] ALL_ON bitwise == v10.1 baseline", flush=True)
    d = phase_d()
    print("   ", d["identical"], "/", d["n"], flush=True)

    print("[E] 32 random switch combinations", flush=True)
    e = phase_e(p)
    print("    finite", e["all_finite"], "| T1' bound", e["all_within_T1"],
          "| zero combo strict", e["zero_combo_strictly_zero"], flush=True)

    print("[F/G/H] contract flags / deferred / 24-layer broadcast", flush=True)
    fgh = phase_fgh()
    print("    F", fgh["F"]["passed"], "| G", fgh["G"]["passed"],
          "| H", fgh["H"]["passed"], flush=True)

    reg = json.load(open(os.path.join(OUT, "capability_registry.json"),
                         encoding="utf-8"))
    v = dict(
        A_all_off_bitwise_zero=a["passed"],
        B_single_off_default_observable=b["n_observable"],
        B_all_finite=b["all_finite"],
        C_single_off_activated_13_of_13=bool(c["n_observable"] == 13),
        D_all_on_bitwise_identical=d["all_identical"],
        E_random_all_finite=e["all_finite"],
        E_random_within_T1=e["all_within_T1"],
        E_zero_combo_strict=e["zero_combo_strictly_zero"],
        F_contract_flags_never_change_forward=fgh["F"]["passed"],
        G_deferred_refused=fgh["G"]["passed"],
        H_24x13_broadcast=fgh["H"]["passed"],
        registry_total=reg["counts"]["total"],
        registry_slots_with_switch=reg["counts"]["total_slots_with_switch"],
        forward_runnable=reg["counts"]["forward_runnable"],
        contract_registered=reg["counts"]["contract_registered"],
        deferred_not_in_source=reg["counts"]["deferred_not_in_source"])
    v["passed"] = bool(
        v["A_all_off_bitwise_zero"] and v["B_all_finite"]
        and v["C_single_off_activated_13_of_13"]
        and v["D_all_on_bitwise_identical"] and v["E_random_all_finite"]
        and v["E_random_within_T1"] and v["E_zero_combo_strict"]
        and v["F_contract_flags_never_change_forward"]
        and v["G_deferred_refused"] and v["H_24x13_broadcast"]
        and v["registry_slots_with_switch"] == v["registry_total"])

    rep = dict(
        title="TS2SC v10.2 · 开关法则真跑审计（完全继承的可验证定义）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        law=reg["switch_law"],
        counts=reg["counts"],
        phaseA=a, phaseB=b, phaseC=c, phaseD=d, phaseE=e,
        phaseFGH=fgh, verification=v,
        honest_boundaries=[
            "「完全继承」定义为：31/31 能力都有可切换 slot + 全关逐位回到原模型；"
            "不等于 31 项都在前向贡献数值。",
            "20 FORWARD 项有可差分前向门；9 CONTRACT 项开关只控制登记/校验，"
            "phase F 实证切换前后 Δ 逐位相同；2 DEFERRED 项 NOT_IN_SOURCE，"
            "置 True 被显式拒绝，不伪造继承。",
            "role.Wc 在默认初值全零、gate.beta 初值 0.0，单关差分不可观测"
            "（phase B 如实登记）；注入非零探针后 13/13 可观测（phase C）——"
            "证明开关接线存在，不声称它们当前在贡献。",
            "全开 27/27 Δ 逐位相同 ⇒ 权重字节、manifest、2 个分片、219/219 "
            "全量 COPY 审计、4 语言 10 臂生成、T==C0 端到端零回归全部仍有效，"
            "本轮未重跑、未改动任何权重。",
            "β=1 仍是尺度契约配置，不是 L2 校准成品；calibrate_beta_v10.py "
            "仍 NOT_EXECUTED；P-β 仍 NOT_TESTED。",
            "端到端 8M 仍 INFEASIBLE（hybrid 保留 teacher 全注意力）；"
            "只有分支 O(1) 状态性质成立。",
            "上游 COMPLETE_BEYOND=False / 2× 对手步数否决随附，不继承超越主张。",
            "本轮纯 numpy CPU 单元级审计，未拉 2 GiB 权重、未跑 GPU forward、"
            "无 ppl / KL / 配对 CI / 新生成样本。",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(AUDIT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n[verify]", json.dumps(v, ensure_ascii=False), flush=True)
    print("[done]", AUDIT, rep["elapsed_s"], "s", flush=True)
    return rep


if __name__ == "__main__":
    main()
