#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · Config-driven migration executor

V7 hardcoded Qwen3 shapes and silently skipped files. Here:
  * every shape comes from TeacherSpec (spec.py)
  * every teacher tensor goes through provenance.classify (UNCLASSIFIED == 0)
  * COPY asserts shape + dtype + bit identity
  * lm_head follows the teacher's tie flag (R1)
  * GQA rows use build_slot_map, NEVER zero padding (R6)
  * QK-norm is folded, not skipped (R8)

`plan()` / `simulate()` are pure stdlib and run in the sandbox.
`execute()` needs safetensors.torch on the T+0 machine; it is deliberately
NOT callable here and says so instead of faking a result.
"""
from __future__ import annotations

import json
import os

from provenance import classify, write_ledger
from modeling_v8 import build_slot_map, fold_qk_norm, lm_head_plan


def plan(spec, teacher_keys: list, shapes: dict | None = None,
         dtypes: dict | None = None) -> dict:
    """Dry-run ledger. No weights are read."""
    led = classify(teacher_keys, spec)
    v = spec.values
    d, kv, hd = v.get("hidden_size"), v.get("num_key_value_heads"), v.get("head_dim")
    tie = v.get("tie_word_embeddings")
    has_lm = "lm_head.weight" in teacher_keys

    slot = None
    if all(isinstance(x, int) and x > 0 for x in (d, kv, hd)):
        try:
            slot = build_slot_map(d, kv * hd, v.get("num_attention_heads"))
        except ValueError as e:
            led["blockers"].append("SLOT_MAP_FAILED:" + str(e))

    head_plan = lm_head_plan(tie, has_lm)
    if not head_plan["passed"]:
        led["blockers"].append("LM_HEAD_PLAN_FAILED:" + str(head_plan))

    qk = None
    if all(isinstance(x, int) and x > 0 for x in (d, kv, hd)):
        qk = fold_qk_norm(kv * hd, hd, 1.0)

    # shape/dtype coverage: every COPY must be verifiable at execute time
    shapes = shapes or {}
    dtypes = dtypes or {}
    unverifiable = [e["src"] for e in led["entries"]
                    if e["op"] == "COPY" and (e["src"] not in shapes or e["src"] not in dtypes)]
    if shapes and unverifiable:
        led["blockers"].append("COPY_TENSORS_WITHOUT_SHAPE_DTYPE:" + ",".join(unverifiable[:4]))

    return dict(
        passed=bool(led["passed"] and head_plan["passed"] and not led["blockers"]),
        ledger=led,
        student_config=spec.student_config(),
        slot_map=slot,
        qk_fold=qk,
        lm_head_plan=head_plan,
        n_copy=led["counts"].get("COPY", 0),
        n_derive=led["counts"].get("DERIVE", 0),
        n_new=led["counts"].get("NEW", 0),
        blockers=led["blockers"],
        next="bind (tokenizer contract) -> execute (T+0) -> align_accept L2",
    )


def copy_assert(teacher_tensor_meta: dict, student_tensor_meta: dict) -> dict:
    """The atomic COPY invariant, expressed on metadata (no torch needed)."""
    checks = dict(
        shape_equal=teacher_tensor_meta.get("shape") == student_tensor_meta.get("shape"),
        dtype_equal=teacher_tensor_meta.get("dtype") == student_tensor_meta.get("dtype"),
        bit_identical=bool(teacher_tensor_meta.get("bytes") == student_tensor_meta.get("bytes")),
    )
    return dict(passed=all(checks.values()), checks=checks,
                note="necessary for L0 only; NOT sufficient for functional parity")


def simulate(spec, teacher_keys: list) -> dict:
    """Sandbox self-test: fake metadata dict, proves the ledger + gates fire.
    NOT real weights, NOT Qwen4 data.
    """
    shapes = {k: [4, 4] for k in teacher_keys}
    dtypes = {k: "BF16" for k in teacher_keys}
    p = plan(spec, teacher_keys, shapes, dtypes)
    bad = copy_assert(dict(shape=[4, 4], dtype="BF16", bytes=b"abcd"),
                      dict(shape=[4, 4], dtype="BF16", bytes=b"abce"))
    return dict(note="SIMULATED METADATA ONLY",
                plan_passed=p["passed"],
                n_copy=p["n_copy"], n_derive=p["n_derive"], n_new=p["n_new"],
                slot_slots=(p["slot_map"] or {}).get("n_slots"),
                zero_padded_rows=(p["slot_map"] or {}).get("zero_padded_rows"),
                lm_head_op=p["lm_head_plan"]["op"],
                qk_fold_transform=(p["qk_fold"] or {}).get("transform"),
                copy_assert_rejects_bitflip=not bad["passed"])


def execute(teacher_dir: str, out_dir: str, spec):
    """T+0 ONLY. Refuses to run without the real runtime."""
    missing = []
    try:
        import torch                                   # noqa: F401
        from safetensors.torch import load_file, save_file   # noqa: F401
    except Exception as e:
        missing.append(type(e).__name__)
    if missing or not spec.t0_ready:
        return dict(
            status="BLOCKED",
            reason="execute() requires torch + safetensors.torch + a T+0-ready TeacherSpec",
            missing=missing, spec_blockers=spec.blockers,
            honest="No student checkpoint was written. Nothing is claimed.",
        )
    raise NotImplementedError(
        "At T+0: load real safetensors index, run plan(), then for each entry "
        "apply COPY (with copy_assert) / DERIVE (slot_map, qk_fold, per-expert "
        "map) / NEW (initialised and grad-checked), then save. This stub is "
        "intentionally not a fake migration."
    )


if __name__ == "__main__":
    import sys
    from spec import TeacherSpec
    s = TeacherSpec.from_dir(sys.argv[1])
    keys = json.load(open(sys.argv[2], encoding="utf-8")) if len(sys.argv) > 2 else []
    print(json.dumps(plan(s, keys), indent=2, ensure_ascii=False, default=str))
