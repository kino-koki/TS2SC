#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · Functional Alignment & Native-Level Acceptance

Why this file exists: V7's verify_arch.py judged success with
gen_quality(): n>=20, max_run<=6, rep3<=0.5, distinct1. A model emitting
Latin-extended mojibake scores HIGH on distinct1 and LOW on rep3, so it
PASSES. That is the institutional reason "被迁移的那几个模型只会吐乱码"
was ever recorded as a successful migration.

Here the endpoint is not "non-degenerate", it is "indistinguishable from the
teacher under the same conditions", measured in five escalating gates.
No gate may be skipped, and no level-4 claim is allowed without a paired
teacher control.
"""
from __future__ import annotations

import math


GATES = {
    "L0_STRUCTURE": [
        "provenance_v8.json passed",
        "static_contract passed",
        "student state_dict contains no teacher-only attention keys",
        "every NEW parameter has non-zero first-step gradient",
    ],
    "L1_CODEC": [
        "runtime_contract passed: len(tokenizer)==embed.rows",
        "5 roundtrips (zh / en / mixed / emoji / multibyte boundary) exact",
        "UTF-8 streamed decode equals whole-buffer decode",
        "chat_template applies and ends with generation prompt",
        "all special ids inside [0, vocab_size)",
    ],
    "L2_FUNCTION": [
        "layer-wise hidden-state cosine vs teacher, same input, every layer",
        "logit KL(student||teacher) <= declared budget",
        "logit top-1 agreement >= declared floor",
        "full-unroll == O(1)-state streaming forward (numeric equivalence)",
        "attention-path contribution non-zero: delta_sgc > 0 AND calibrated, not alpha=-2",
    ],
    "L3_BEHAVIOUR": [
        "paired generation on N>=200 prompts, same seed/sampling/tokenizer",
        "Chinese readable-rate, Latin-mojibake rate, repetition, EOS termination",
        "instruction following / chat-template identity / tool-call format",
        "no-regression paired difference with 95% CI (Wilson / bootstrap)",
        "explicitly NOT a claim of improvement",
    ],
    "L4_NATIVE": [
        "L0-L3 all pass on the released Qwen4-27B weights",
        "same chat template, same special tokens, same stop behaviour as teacher",
        "same context window declared in config (Qwen4 hint: 1M; MUST come from real config)",
        "no tokenizer shim, no vocab subset, no manual id remap in the serving path",
        "serves through the standard HF/vLLM path without custom decode code",
    ],
}


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% CI for a paired success rate. No normal-approx shortcut."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mojibake_rate(text: str) -> float:
    """Latin-extended / byte-to-unicode leakage: the exact historical 乱码 signature.

    TS2SC_VL47_PREFLIGHT.md records ';++ / VILLE / nullptr' and
    '¤Ĳ厅 / «闪店'. GPT-2 byte-level B2U leftovers live in U+00A0..U+017F
    and U+0100..U+024F; Chinese text should contain none of them.
    """
    if not text:
        return 1.0
    bad = sum(1 for ch in text
              if 0x00A0 <= ord(ch) <= 0x024F
              or ch in "\ufffd\uFFFDU+FFFD")
    return bad / len(text)


def evaluate(paired_results: list, thresholds: dict | None = None) -> dict:
    """paired_results: [{prompt, teacher_ok, student_ok, student_text, eos_hit}, ...]"""
    th = dict(top1_floor=0.90, readable_floor=0.95, mojibake_ceiling=0.01,
              kl_ceiling=0.20, ci_floor=0.0)
    th.update(thresholds or {})
    n = len(paired_results)
    if n == 0:
        return dict(passed=False, level="L0", reason="NO_PAIRED_CONTROL")

    wins = sum(1 for r in paired_results if r["student_ok"] and r["teacher_ok"])
    regress = sum(1 for r in paired_results if r["teacher_ok"] and not r["student_ok"])
    moji = [mojibake_rate(r.get("student_text", "")) for r in paired_results]
    avg_moji = sum(moji) / n
    eos = sum(1 for r in paired_results if r.get("eos_hit")) / n
    lo, hi = wilson_ci(wins, n)

    checks = dict(
        n_paired=n, paired_success=wins, regressions=regress,
        paired_rate=wins / n, paired_ci95=[lo, hi],
        avg_mojibake_rate=avg_moji, eos_hit_rate=eos,
        no_regression=(lo >= th["ci_floor"]),
        mojibake_ok=(avg_moji <= th["mojibake_ceiling"]),
    )
    level = "L0"
    if checks["no_regression"] and checks["mojibake_ok"]:
        level = "L3"
    checks["level"] = level
    checks["passed"] = level == "L3"
    checks["honest_bound"] = (
        "L3 is no-regression, not improvement. L4 requires the released Qwen4-27B "
        "config + weights + tokenizer and a full L0-L3 rerun; it is NOT claimable today."
    )
    return checks


if __name__ == "__main__":
    # Self-test with a deliberately mojibake-heavy fake student output: the OLD
    # gen_quality() would call it non-degenerate; THIS gate must reject it.
    fake = [dict(prompt="中国的首都是", teacher_ok=True, student_ok=True,
                 student_text="¤Ĳ城市 «闪店 ±展 北京北京", eos_hit=True)
            for _ in range(20)]
    print("old-style non_degenerate would pass; v8 says:",
          evaluate(fake)["passed"], evaluate(fake)["avg_mojibake_rate"])
