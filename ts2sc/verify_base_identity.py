#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补强「原本认同回归」：绕过 <think> 前缀，直接续写式 probe。

clean_state_verify.py 的 40-token deep probe 全落在 <think> 思考块里，
还没走到回答部分 ⇒ 没出现品牌词（已如实标注 false）。
本脚本用**不带 chatml 模板**的短前缀让基座直接续写自我描述，
3 题 × 16 token，零身份注入、零融合 hook（纯 teacher 干净态）。
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
from verify_clean_state import row_of, IDENTITY_MARKERS, mem_guard  # noqa: E402

OUT = "/data/workspace/output"
VERIFY = os.path.join(OUT, "base_identity_probe.json")
STOP_IDS = {1, 130073}
MAX_NEW = 16

PROMPTS = [
    ("zh", "我是"),
    ("zh", "我是由"),
    ("en", "I am"),
]


def run(lm, tok, prompt, V_size):
    ids, _ = tok.encode(prompt, add_bos=False)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    L = len(ids)
    r = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                r = lm.step(int(tid), pos, kvs, need_logits=(pos == L - 1),
                            fusion_hook=None, logit_bias_fn=None)
            out = [int(r[0])]
            fin = [bool(getattr(lm, "last_logits_finite", True))]
            for j in range(1, MAX_NEW):
                r = lm.step(int(out[-1]), L + j - 1, kvs, need_logits=True,
                            fusion_hook=None, logit_bias_fn=None)
                out.append(int(r[0]))
                fin.append(bool(getattr(lm, "last_logits_finite", True)))
                if r[0] in STOP_IDS:
                    break
    from scale_theory import decode_gate
    return dict(text=tok.decode(out, skip_special=True), ids=out,
                gate=decode_gate(out, V_size, all(fin)),
                logits_finite=all(fin), n_new=len(out))


def main():
    t0 = time.time()
    mem_guard(2.6)
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    V_size = cfg["vocab_size"]
    p0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", p0[:8])[0]
    V = VirtualWeights(p0, shard_shas(), json.loads(p0[8:8 + n].decode()), 8 + n)
    nf, nb = V.preload()
    print(f"[pull] {nf} shards / {nb/2**30:.2f} GiB", flush=True)
    tok = MiniCPMTokenizer(os.path.join(BASE, "tokenizer.json"))
    lm = TeacherLM(V, cfg)

    rows = []
    for lang, q in PROMPTS:
        g = run(lm, tok, q, V_size)
        r = row_of(lang, "base_identity_continuation", q, g)
        rows.append(r)
        print(f"  [{lang}] base={r['base_markers']} "
              f"identity={r['identity_markers']} -> {g['text'][:88]!r}",
              flush=True)
        json.dump(rows, open(VERIFY, "w", encoding="utf-8"),
                  ensure_ascii=False)   # 逐题断点

    base_hits = sum(1 for r in rows if r["n_base_markers"] >= 1)
    leaks = sum(r["n_identity_markers"] for r in rows)
    rep = dict(
        title="原本认同回归 · 直接续写补强 probe（干净态，零注入）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        why="40-token chatml deep probe 全落在 <think> 块内，"
            "未走到回答部分；本脚本绕过模板让基座直接续写自我描述。",
        config=dict(fusion_hook="none", logit_bias_fn="none",
                    identity_adapter="none", max_new=MAX_NEW,
                    chatml_template=False),
        probe_rows=rows,
        base_marker_hits=base_hits,
        identity_marker_leaks=leaks,
        all_finite=all(r["logits_finite"] for r in rows),
        all_decode_gate=all(r["decode_gate"] for r in rows),
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(VERIFY, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n[summary] base_hits", base_hits, "/", len(rows),
          "| identity leaks", leaks, "| elapsed", rep["elapsed_s"], "s",
          flush=True)
    print("[done]", VERIFY, flush=True)


if __name__ == "__main__":
    main()
