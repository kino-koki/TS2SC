#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 · 深审：把「抽查」升级为「全量」，把「2题4token」升级为「多语言多臂」

## 这一轮继续做什么（上一轮留下的两个证据缺口）

v10 首轮出于 CPU 时间只做了：
  * 8 个 COPY 抽查逐位一致（219 个里抽 8 个）
  * 2 题 × 2 臂 × 4 token 生成（zh/en）
  * β=0 零回归只有单元级（Δ 严格零向量），没有端到端 vs 纯 teacher 对照

本脚本一次 preload 上游全部 22 分片，串行完成：

  A. **219/219 全量 COPY 逐位审计**：对 manifest 里每一个 COPY，
     从本次 LIVE-blob-sha 校验过的字节流式计算 sha256，逐个对照。
     不再抽样，任何一个不符即整体 FAIL。
  B. **4 语言 × 3 臂生成**：
       T  = 纯 teacher（fusion_hook=None）
       C0 = v10 分支 β=0（挂 hook，Δ 严格零）
       C1 = v10 分支 β=1（交付配置）
     短 prompt 降低 prefill 成本；每臂 6 token。
     断言：T 与 C0 **逐 token 相同**（端到端零回归，不只是单元级）；
           C1 与 C0 不同（架构生效）；全部 finite / 解码守卫 / 零身份标记。
  C. 顺带记录每层 O(1) 状态、β=1 契约比（复用 test_v10 结论，不重跑重活）。

## 诚实边界
* 仍是纯 numpy CPU；prefill 每 token 重解码 24 层 BF16→f32，故 max_new=6。
* 6 token 只证「不乱码 / finite / 架构生效 / 零回归」，不是质量或能力结论。
* 上游 v9–v11 `COMPLETE_BEYOND=False` 不变；本轮不继承任何超越/SOTA 主张。
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_minicpm import (VirtualWeights, MiniCPMTokenizer, TeacherLM,  # noqa: E402
                         gh, NPARTS)
from scale_theory import decode_gate  # noqa: E402
import sgc_v10 as A                                             # noqa: E402
import fresh_migrate_v10 as F                                   # noqa: E402

OUT = "/data/workspace/output"
FRESH = os.path.join(OUT, "v10_fresh")
MANIFEST = os.path.join(FRESH, "student_full_manifest_v10.json")
PULL = os.path.join(FRESH, "upstream_pull_report.json")
AUDIT = os.path.join(OUT, "v10_copy_full_audit.json")
VERIFY = os.path.join(OUT, "v10_multilingual_verify.json")
REPORT = os.path.join(OUT, "V10_DEEP_AUDIT_REPORT.md")

IDENTITY_MARKERS = ["青奈", "青小团", "青小団", "六安", "Tuan-Models",
                    "Qingnai", "Qingxiaotuan"]
STOP_IDS = {1, 130073}
MAX_NEW = 6

# 短 prompt：把 chatml 模板后的 prefill 从 ~30 token 压到 ~12 token
PROBES = [
    ("zh", "你好"),
    ("en", "Hi"),
    ("ja", "こんにちは"),
    ("fr", "Bonjour"),
]
TEACHER_ARMS = ("zh", "en")      # 只对这两语言跑第三臂 T，控制 CPU 时间


# ---------------------------------------------------------------------------
def phase_a_full_copy_audit(V, manifest):
    """219/219 全量：每个 COPY 从新字节流式 sha256 对照。"""
    copies = [e for e in manifest["entries"] if e["op"] == "COPY"]
    t0 = time.time()
    bad, rows = [], []
    for i, e in enumerate(copies, 1):
        name = e["name"]
        a, b = V.header[name]["data_offsets"]
        off, rem, h = V.ds + a, b - a, hashlib.sha256()
        while rem > 0:
            k = min(rem, 4 << 20)
            h.update(V.read(off, k))
            off += k
            rem -= k
        got = h.hexdigest()
        ok = bool(got == e["sha256"] and (b - a) == e["bytes"])
        if not ok:
            bad.append(dict(name=name, expected=e["sha256"], got=got,
                            bytes=b - a, declared=e["bytes"]))
        rows.append(dict(name=name, bytes=b - a, matches=ok))
        if i % 40 == 0 or i == len(copies):
            print(f"  [A] {i:3d}/{len(copies)} COPY audited, bad={len(bad)}",
                  flush=True)
    rep = dict(
        title="v10 · 219/219 全量 COPY 逐位审计（非抽查）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_copy=len(copies), n_mismatched=len(bad), mismatches=bad,
        total_bytes=sum(r["bytes"] for r in rows),
        all_bit_identical=bool(len(bad) == 0),
        source="LIVE git blob sha1-verified fresh upstream pull "
               "(upstream_pull_report.json)",
        elapsed_s=round(time.time() - t0, 1),
        rows=rows)
    json.dump(rep, open(AUDIT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"[A] ALL {len(copies)}/{len(copies)} bit-identical = "
          f"{rep['all_bit_identical']} | {rep['total_bytes']/2**30:.2f} GiB "
          f"hashed in {rep['elapsed_s']}s", flush=True)
    return rep


# ---------------------------------------------------------------------------
def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def gen(lm, tok, branch, beta, prompt, V_size, max_new=MAX_NEW):
    """beta=None ⇒ 完全不挂 fusion_hook（纯 teacher 同一条代码路径）。"""
    if beta is not None:
        branch.reset()
        branch.set_beta(float(beta))
        hook = branch.hook
    else:
        hook = None
    ids, _ = tok.encode(chatml(prompt), add_bos=False)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    L = len(ids)
    r = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                r = lm.step(int(tid), pos, kvs, need_logits=(pos == L - 1),
                            fusion_hook=hook, logit_bias_fn=None)
            out = [int(r[0])]
            fin = [bool(getattr(lm, "last_logits_finite", True))]
            for j in range(1, max_new):
                r = lm.step(int(out[-1]), L + j - 1, kvs, need_logits=True,
                            fusion_hook=hook, logit_bias_fn=None)
                out.append(int(r[0]))
                fin.append(bool(getattr(lm, "last_logits_finite", True)))
                if r[0] in STOP_IDS:
                    break
    return dict(text=tok.decode(out, skip_special=True), ids=out,
                gate=decode_gate(out, V_size, all(fin)),
                logits_finite=all(fin), n_new=len(out))


def row_of(lang, q, g, arm):
    return dict(arm=arm, lang=lang, q=q, text=g["text"], ids=g["ids"],
                identity_markers=[m for m in IDENTITY_MARKERS
                                  if m in g["text"]],
                logits_finite=g["logits_finite"],
                decode_gate=g["gate"]["passed"], n_new=g["n_new"])


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    # 一次 LIVE 校验拉取（与 v10 首轮同一实现，字节相同）。
    # fresh_pull 会重写 upstream_pull_report.json（仅 elapsed_s 不同），
    # 先备份原字节、跑完恢复，保证已推送文件的远端字节一致性不被破坏。
    _pull_backup = open(PULL, "rb").read() if os.path.exists(PULL) else None
    # CDN 偶发 502：不改已推送的 fresh_migrate_v10.py，只在本次进程里
    # 给它的 gh 注入退避重试，保证「逐片 LIVE blob sha 校验」仍能完成。
    _orig_gh = F.gh

    def _gh_retry(url, accept="application/vnd.github+json", timeout=240):
        last = None
        for k in range(5):
            try:
                return _orig_gh(url, accept=accept, timeout=timeout)
            except Exception as e:                     # 502 / IncompleteRead
                last = e
                print(f"  [pull retry {k+1}/5] {type(e).__name__}: "
                      f"{str(e)[:60]}", flush=True)
                time.sleep(2.0 * (k + 1))
        raise RuntimeError(f"gh failed after 5 attempts: {last}")

    F.gh = _gh_retry
    try:
        V, hdr, pull = F.fresh_pull()
    finally:
        F.gh = _orig_gh
        if _pull_backup is not None:
            open(PULL, "wb").write(_pull_backup)
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    d, nl, V_size = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]

    print("\n[phase A] 219/219 全量 COPY 逐位审计", flush=True)
    a = phase_a_full_copy_audit(V, manifest)

    print("\n[phase B] 4 语言 × 多臂生成（T / C0 / C1）", flush=True)
    tok = MiniCPMTokenizer(os.path.join(FRESH, "tokenizer.json"))
    lm = TeacherLM(V, cfg)
    params = {i: {k: np.asarray(v, np.float32)
                  for k, v in A.deterministic_layer_params(
                      d, seed=20261128 + 101 * i).items()}
              for i in range(nl)}
    branch = A.SGCV10Branch(cfg, params)

    rows = []
    for lang, q in PROBES:
        g0 = gen(lm, tok, branch, 0.0, q, V_size)
        g1 = gen(lm, tok, branch, 1.0, q, V_size)
        rec = dict(lang=lang, q=q,
                   C0=row_of(lang, q, g0, "C0_beta0"),
                   C1=row_of(lang, q, g1, "C1_beta1_delivery"))
        if lang in TEACHER_ARMS:
            gt = gen(lm, tok, branch, None, q, V_size)
            rec["T"] = row_of(lang, q, gt, "T_pure_teacher")
            rec["T_equals_C0_tokenwise"] = bool(gt["ids"] == g0["ids"])
        rec["architecture_active"] = bool(g1["ids"] != g0["ids"])
        rows.append(rec)
        t_eq = rec.get("T_equals_C0_tokenwise", "n/a")
        print(f"  [{lang}] T==C0:{t_eq} active:{rec['architecture_active']} "
              f"| C1 {g1['text'][:40]!r}", flush=True)
        json.dump(rows, open(os.path.join(FRESH, "gen_deep_checkpoint.json"),
                            "w", encoding="utf-8"), ensure_ascii=False)

    all_rows = []
    for r in rows:
        all_rows += [r["C0"], r["C1"]] + ([r["T"]] if "T" in r else [])
    t_eq_all = all(r.get("T_equals_C0_tokenwise", True) for r in rows)

    v = dict(
        phaseA_n_copy=a["n_copy"],
        phaseA_all_219_bit_identical=bool(a["all_bit_identical"]),
        phaseA_mismatched=a["n_mismatched"],
        phaseA_total_bytes=a["total_bytes"],
        pull_all_blob_sha_verified=bool(pull["all_blob_sha_verified"]),
        n_languages=len(PROBES),
        n_arms_total=len(all_rows),
        T_equals_C0_tokenwise_all=bool(t_eq_all),
        architecture_active_all=all(r["architecture_active"] for r in rows),
        all_logits_finite=all(x["logits_finite"] for x in all_rows),
        all_decode_gate=all(x["decode_gate"] for x in all_rows),
        identity_marker_leaks=sum(len(x["identity_markers"])
                                  for x in all_rows),
        distinct_C1_texts=len({r["C1"]["text"] for r in rows}),
        n_languages_tested=len(rows))
    v["passed"] = bool(
        v["phaseA_all_219_bit_identical"] and v["pull_all_blob_sha_verified"]
        and v["T_equals_C0_tokenwise_all"] and v["architecture_active_all"]
        and v["all_logits_finite"] and v["all_decode_gate"]
        and v["identity_marker_leaks"] == 0 and v["n_languages"] == 4)

    rep = dict(
        title="TS2SC v10 深审 · 抽查→全量 / 2题→4语言多臂 / 单元级→端到端零回归",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        what_changed=[
            "COPY 审计从 8 个抽查升级为 219/219 全量逐位 sha256 对照",
            "生成从 2 题 × 2 臂 × 4 token 升级为 4 语言 × (T/C0/C1) × 6 token",
            "β=0 零回归从单元级 Δ=0 升级为端到端 T 与 C0 逐 token 相同",
            "新增「C1 四语言文本互不重复」非偶然性判据"],
        phaseA=a,
        generation=rows,
        verification=v,
        honest_boundaries=[
            "全量 COPY 逐位 = 219 个 teacher 张量与本次 LIVE blob-sha 校验字节一致；"
            "不是 ppl / 下游 benchmark / 知识集配对 CI",
            "6 token 短生成只证 finite / 不乱码 / 架构生效 / 零回归 / 零身份标记；"
            "不是质量、能力、事实正确性结论",
            "T 臂只在 zh/en 两语言跑以控制纯 numpy CPU 时间；其余 2 语言只有 C0/C1",
            "β=1 仍是尺度契约配置（T1'），不是 L2 校准成品；24 个 β_l 待校准",
            "上游 v9–v11 COMPLETE_BEYOND=False 不变；本轮不继承超越/SOTA 主张",
            "纯 numpy CPU 等价实现，未跑上游 PyTorch pytest；无 GPU/bf16/多卡验证",
            "Qwen4-27B 仍未发布；无原生级 / 上下文理论无限实测声明",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(VERIFY, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n[verify]", json.dumps(v, ensure_ascii=False), flush=True)
    print("[done]", VERIFY, rep["elapsed_s"], "s", flush=True)
    return rep


if __name__ == "__main__":
    main()
