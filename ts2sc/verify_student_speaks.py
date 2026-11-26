#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.4 · 解决乱码理论问题后，迁移后的模型到底会不会说话？

Same real MiniCPM5-1B weights, SAME chatml prompt, SAME greedy decode:

  T  teacher baseline        : 上一轮真跑，从 minicpm5_teacher_verify.json 读回
  A  student β=0             : 交付默认态，预期逐字相同（零回归）
  C  student β=1 CONTRACTED  : v8.4 理论修复后的开启态，无量纲尺度契约
                               预期 logits finite、id 全合法、中文可读
  B  v8.3 β=1 RAW (历史)     : 未校准全开，24/24 越界 id → 空串
                               从旧 json 读回，作为理论解释的负对照（不重跑）

The decisive new arm is C. T3' decode gate replaces the old non-degenerate proxy.
No torch. Token from env only, never written to a file.
"""
from __future__ import annotations

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
                         shard_shas, BASE)
from run_full_migration import SGCFullBranch, derive_kv_bytes  # noqa: E402
from modeling_v8 import build_slot_map  # noqa: E402
from scale_theory import decode_gate, EPS_DEFAULT, SNR_DEFAULT  # noqa: E402

OUT = "/data/workspace/output"
PROMPT = ("<s><|im_start|>user\n请用一句话介绍你自己。<|im_end|>\n"
          "<|im_start|>assistant\n")
MAX_NEW = 24


def gen(lm, tok, hook, max_new=MAX_NEW):
    """Greedy decode, collecting finite flags + ids for T3' gate."""
    ids, _ = tok.encode(PROMPT, add_bos=False)
    cache = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
              np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    L = len(ids)
    r = None
    for pos, tid in enumerate(ids):
        r = lm.step(int(tid), pos, cache, need_logits=(pos == L - 1),
                    fusion_hook=hook)
    out = [int(r[0])]
    finite = [bool(getattr(lm, "last_logits_finite", True))]
    for j in range(1, max_new):
        r = lm.step(int(out[-1]), L + j - 1, cache, need_logits=True,
                    fusion_hook=hook)
        out.append(int(r[0]))
        finite.append(bool(getattr(lm, "last_logits_finite", True)))
        if r[0] in (1, 130073):
            break
    return dict(text=tok.decode(out, skip_special=True), ids=out,
                n_prompt=L, logits_finite=all(finite),
                finite_steps=sum(finite), n_steps=len(finite))


def readable(s):
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    bad = sum(1 for c in s if 0x00A0 <= ord(c) <= 0x024F)
    return dict(n_chars=len(s), cjk=cjk, latin_mojibake=bad,
                has_chinese=cjk > 0,
                readable=bool(len(s) >= 8 and cjk > 0 and bad <= 2))


def main():
    t0 = time.time()
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    d, nl = cfg["hidden_size"], cfg["num_hidden_layers"]
    V_size = cfg["vocab_size"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    slot = build_slot_map(d, kv_dim, cfg["num_attention_heads"])

    part0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    V = VirtualWeights(part0, shard_shas(), hdr, 8 + n)
    nf, nb = V.preload()
    print(f"[pull] {nf} shards / {nb/2**30:.2f} GiB", flush=True)

    tok = MiniCPMTokenizer(os.path.join(BASE, "tokenizer.json"))
    lm = TeacherLM(V, cfg)
    dv = derive_kv_bytes(V, cfg, slot)
    print(f"[derive] {len(dv)} SGC k/v tensors from REAL teacher rows", flush=True)

    # ---- T: baseline from the previous REAL teacher run ----
    prev = json.load(open(os.path.join(OUT, "minicpm5_teacher_verify.json"),
                          encoding="utf-8"))
    baseline = next(g for g in prev["D_teacher_generation"]["generations"]
                    if g["tag"] == "chatml")["generated"]
    print(f"[T] baseline = {baseline[:60]!r}", flush=True)

    branch = SGCFullBranch(cfg, V, slot, dv, beta=0.0, enabled=True,
                           eps=EPS_DEFAULT, role_snr=SNR_DEFAULT)

    # ---- A: β=0 (delivered default) ----
    branch.beta = 0.0; branch.reset()
    a = gen(lm, tok, branch.hook)
    a_gate = decode_gate(a["ids"], V_size, a["logits_finite"])
    a["identical_to_teacher"] = bool(a["text"] == baseline)
    a["readability"] = readable(a["text"])
    a["decode_gate"] = a_gate
    print(f"[A] beta=0 -> {a['text'][:60]!r} | identical={a['identical_to_teacher']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- C: β=1 under the PROVEN dimensionless scale contract ----
    branch.beta = 1.0; branch.reset()
    c_err = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore"):
                c = gen(lm, tok, branch.hook)
    except Exception as e:
        c_err = type(e).__name__ + ":" + str(e)[:160]
        c = dict(text="", ids=[], n_prompt=0, logits_finite=False,
                 finite_steps=0, n_steps=0)
    c_gate = decode_gate(c["ids"], V_size, c["logits_finite"] and c_err is None)
    c["readability"] = readable(c["text"])
    c["decode_gate"] = c_gate
    c["error"] = c_err
    c["contract"] = dict(beta=1.0, eps=EPS_DEFAULT, role_snr=SNR_DEFAULT,
                         inequality="‖Δ‖₂ ≤ βε‖x‖₂",
                         note="scale-contracted, NOT L2-calibrated; NOT training")
    print(f"[C] beta=1 contracted -> {c['text'][:60]!r} | "
          f"finite={c['logits_finite']} gate={c_gate['passed']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- B: historical v8.3 negative control (theory explains it) ----
    old = json.load(open(os.path.join(OUT, "student_speaks_verify.json"),
                         encoding="utf-8"))
    b = old["arms"]["B_student_uncalibrated"]
    b["theory_explanation"] = (
        "v8.3 raw β=1: 1e4~1e5 injection → RMSNorm overflow → NaN logits → "
        "`NaN > best_v` always False → best_i stayed at sentinel -1 → "
        "24/24 out-of-range ids → decode dropped every token → EMPTY STRING. "
        "Not UTF-8 mojibake; the worst form of '乱码' produces no character at all.")
    b["not_rerun"] = True

    beta0_ok = bool(a["identical_to_teacher"] and a["readability"]["readable"]
                    and a_gate["passed"])
    beta_pos_ok = bool(c_gate["passed"] and c["readability"]["readable"]
                       and c["logits_finite"] and c_err is None)

    rep = dict(
        title="解决乱码理论问题后 · 迁移后模型会不会说话（v8.4 重验证）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        prompt=PROMPT, max_new=MAX_NEW, vocab_size=V_size,
        engine="纯 numpy 教师前向 + 24 层 SGC 并联 hook（无 torch）",
        theory=dict(
            T1="dimensionless scale contract ‖Δ‖≤βε‖x‖ (ε=0.10)",
            T2="scale-invariant role bit k̂=k/(r_k+δ), k'=k̂+snr·e (snr=0.50)",
            T3="finite-and-in-range decode gate"),
        arms=dict(T_teacher_baseline=dict(text=baseline, source="上一轮真跑"),
                  A_student_beta0=a, C_student_beta1_contracted=c,
                  B_v83_raw_negative_control=b),
        answer=dict(
            beta0_speaks=beta0_ok,
            beta_positive_speaks=beta_pos_ok,
            beta_positive_text=c["text"],
            beta_positive_reason=(
                "β=1 在尺度契约下：logits 全 finite、生成 id 全在 [0,V)、"
                "中文可读、无拉丁扩展乱码 ⇒ 开着融合分支也能说话。"
                if beta_pos_ok else "仍有非 finite / 越界 / 不可读，见 C 臂"),
            remaining_caveat="β=1 是尺度契约配置，不是 L2 校准后的成品；"
                             "未做 logit KL / 配对 CI / ppl，不写能力或原生级结论。",
            strict_caveat="完整 2.6 GB student 未落盘；本实测是 "
                          "'真实权重 + 24 层融合 hook' 的前向，"
                          "不是对已组装大文件的直接加载。"),
        elapsed_s=round(time.time() - t0, 1),
        boundaries=["无 ppl / KL / 配对 CI / 能力结论",
                    "C 臂不是训练，是理论契约下的推理",
                    "token 仅从环境读取，未写入任何文件",
                    "未删除/修改/推送任何仓库条目"],
    )
    p = os.path.join(OUT, "student_speaks_verify_v84.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n=== ANSWER ===")
    print("beta0_speaks        =", beta0_ok)
    print("beta_positive_speaks=", beta_pos_ok)
    print("C decode_gate       =", c_gate["passed"],
          "| out_of_range =", c_gate["n_out_of_range"])
    print("[done]", p, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
