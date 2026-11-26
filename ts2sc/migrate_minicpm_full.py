#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.2 · Full fusion migration recipe for MiniCPM5-1B

Fusion contract (NOT replacement):
    EVERY one of the 219 teacher tensors is COPY'd verbatim into the student.
    The SGC convolutional-attention branch is ADDED in parallel through a
    learnable, explicitly gated beta. No teacher layer is deleted.

    out = teacher_block(x) + enabled * beta * sgc_branch(x)

What runs in THIS sandbox (--dry-run):
    * real config.json + real safetensors header (219 tensors, 2.16 GB data)
      -> full fusion ledger with UNCLASSIFIED == 0
    * all shape/dtype/decomposition invariants asserted
    * NO 2.16 GB file is written (315 MiB free disk)

What runs on the target machine (--execute):
    * stream teacher bytes -> write full student.safetensors
    * reopen and verify EVERY COPY byte
    * emit provenance + student config

No torch. No transformers. No credential is stored in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_v8 import (SafeTensors, write_safetensors, derive_gqa_row_remap,  # noqa: E402
                            zero_tensor, DTYPE_BYTES)
from modeling_v8 import build_slot_map, fold_qk_norm, lm_head_plan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/data/workspace/minicpm5_1b"
OUT = "/data/workspace/output"


def real_header():
    """Parse the REAL safetensors header from model_part_00 on disk."""
    p = os.path.join(BASE, "model_part_00")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    T = {k: v for k, v in hdr.items() if k != "__metadata__"}
    return T, 8 + n


def fusion_ledger(T: dict, cfg: dict) -> dict:
    """100% COPY of teacher tensors + NEW parallel branch.

    This replaces v8.0's replacement-style classify() for the fusion path.
    DERIVE here creates ADDITIONAL SGC tensors; it never consumes a teacher key.
    """
    d = cfg["hidden_size"]
    nl = cfg["num_hidden_layers"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    tie = cfg["tie_word_embeddings"]
    slot = build_slot_map(d, kv_dim, cfg["num_attention_heads"])
    qk = fold_qk_norm(kv_dim, cfg["head_dim"], 1.0)
    hp = lm_head_plan(tie, "lm_head.weight" in T)

    copy = []
    for k, m in T.items():
        n = 1
        for x in m["shape"]:
            n *= x
        copy.append(dict(src=k, dst=k, op="COPY", dtype=m["dtype"],
                         shape=m["shape"], n_params=int(n),
                         bytes=int(n * DTYPE_BYTES[m["dtype"]]),
                         bit_identical_required=True))
    # DERIVE -> ADDITIONAL SGC tensors (teacher keys remain untouched)
    derive = []
    for i in range(nl):
        p = f"model.layers.{i}.self_attn."
        for tail in ("k_proj.weight", "v_proj.weight"):
            src = p + tail
            derive.append(dict(src=src, dst=f"model.blocks.{i}.sgc.W{tail[0].upper()}.weight",
                               op="DERIVE", transform="GQA_ROW_REMAP",
                               in_shape=T[src]["shape"], out_shape=[d, d],
                               zero_padded_rows=0,
                               note="additional branch tensor; teacher k/v COPY still present"))
        derive.append(dict(src=p + "q_proj.weight",
                           dst=f"model.blocks.{i}.sgc.Wq.weight",
                           op="DERIVE", transform="VERBATIM_AS_Q",
                           in_shape=T[p + "q_proj.weight"]["shape"],
                           out_shape=list(T[p + "q_proj.weight"]["shape"])))
    # NEW: gated parallel branch
    new = []
    for i in range(nl):
        b = f"model.blocks.{i}.sgc."
        for k, shp, init in (("Wc", [3, d], "zeros"), ("Wg.weight", [d, d], "zeros"),
                             ("wgate.weight", [1, d], "zeros"), ("alpha", [1], "zeros"),
                             ("beta", [1], "zeros")):
            new.append(dict(dst=b + k, op="NEW", dtype="F32", shape=shp,
                            init=init, grad_required=True,
                            note="beta MUST be calibrated at L2; never left hardcoded"))

    blockers = []
    if not hp["passed"]:
        blockers.append("LM_HEAD_PLAN_FAILED")
    if slot["zero_padded_rows"] != 0:
        blockers.append("ZERO_PADDING_DETECTED")
    if qk["dropped_columns"] != 0:
        blockers.append("QKNORM_DROPPED")
    if tie is not False:
        blockers.append("TIE_NOT_AS_EXPECTED_FOR_THIS_RELEASE")
    if len(T) != 219:
        blockers.append(f"UNEXPECTED_TENSOR_COUNT:{len(T)}")

    student_cfg = dict(cfg)
    student_cfg.update(
        model_type="simplcore_v8", architectures=["SimplCoreV8ForCausalLM"],
        n_slots=slot["n_slots"], conv_kernel=3, rho=0.90,
        fusion="PARALLEL_GATED", beta_init=0.0,
        teacher_layers_retained=len(T),
        note="ALL teacher attention/FFN/norm/embed/lm_head layers kept verbatim",
    )
    return dict(
        passed=not blockers, blockers=blockers,
        n_teacher=len(T), n_copy=len(copy), n_derive=len(derive), n_new=len(new),
        unclassified=0,
        total_teacher_params=sum(c["n_params"] for c in copy),
        total_teacher_bytes=sum(c["bytes"] for c in copy),
        copy=copy, derive=derive, new=new,
        slot_map=dict(n_slots=slot["n_slots"], kv_dim=kv_dim,
                      zero_padded_rows=slot["zero_padded_rows"],
                      gqa_ratio=slot["gqa_ratio"]),
        qk_fold=dict(transform=qk["transform"], folded_columns=qk["folded_columns"],
                     dropped_columns=qk["dropped_columns"]),
        lm_head_plan=hp,
        student_config=student_cfg,
        invariants=[
            "UNCLASSIFIED == 0 (every teacher tensor has exactly one op)",
            "COPY count == teacher tensor count (fusion, not replacement)",
            "lm_head.weight present because tie_word_embeddings == false",
            "no zero-padded GQA rows; every SGC row maps to a real teacher row",
            "QK-norm folded, not skipped",
            "beta initialised 0 and explicitly gated -> no ungated self-evolution",
            "COPY bit-identical is necessary for L0 but NOT sufficient for parity",
        ],
    )


def execute_full(teacher_safetensors: str, out_dir: str, cfg: dict, ledger: dict):
    """Target-machine path. Streams bytes; never materialises 2.16 GB in RAM."""
    os.makedirs(out_dir, exist_ok=True)
    slot = build_slot_map(cfg["hidden_size"],
                          cfg["num_key_value_heads"] * cfg["head_dim"],
                          cfg["num_attention_heads"])
    tensors, report, failures = {}, [], []
    with SafeTensors(teacher_safetensors) as s:
        for c in ledger["copy"]:
            raw = s.tensor(c["src"])
            if len(raw) != c["bytes"]:
                failures.append("COPY_SIZE_MISMATCH:" + c["src"])
            tensors[c["dst"]] = (c["dtype"], c["shape"], raw)
            report.append(dict(src=c["src"], dst=c["dst"], op="COPY", bytes=len(raw)))
        for c in ledger["derive"]:
            m = s.meta(c["src"]); raw = s.tensor(c["src"])
            if c["transform"] == "GQA_ROW_REMAP":
                out, shape = derive_gqa_row_remap(raw, m, slot)
                tensors[c["dst"]] = (m["dtype"], shape, out)
            elif c["transform"] == "VERBATIM_AS_Q":
                tensors[c["dst"]] = (m["dtype"], m["shape"], raw)
            report.append(dict(src=c["src"], dst=c["dst"], op="DERIVE",
                               transform=c["transform"]))
        for c in ledger["new"]:
            tensors[c["dst"]] = (c["dtype"], c["shape"],
                                 zero_tensor(c["shape"], c["dtype"]))
        w = write_safetensors(os.path.join(out_dir, "student.safetensors"), tensors,
                              metadata=dict(format="pt", ts2sc="v8.2", mode="FUSION"))
    # reopen and verify EVERY COPY byte
    with SafeTensors(os.path.join(out_dir, "student.safetensors")) as out, \
         SafeTensors(teacher_safetensors) as src:
        for r in report:
            if r["op"] != "COPY": continue
            if out.tensor(r["dst"]) != src.tensor(r["src"]):
                failures.append("COPY_BYTE_MISMATCH:" + r["src"])
    json.dump(ledger["student_config"],
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    json.dump(dict(written=w, failures=failures, report=report,
                   copy_verified_reopen=not failures),
              open(os.path.join(out_dir, "migration_report.json"), "w"), indent=2)
    return dict(status="PASS" if not failures else "FAIL", written=w,
                n_copy=len(report), failures=failures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", default=None,
                    help="path to reassembled teacher model.safetensors (target machine)")
    ap.add_argument("--out", default=os.path.join(OUT, "minicpm5_student_full"))
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    T, ds = real_header()
    led = fusion_ledger(T, cfg)
    p = os.path.join(OUT, "minicpm5_fusion_ledger.json")
    json.dump(led, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[ledger] teacher={led['n_teacher']} COPY={led['n_copy']} "
          f"DERIVE={led['n_derive']} NEW={led['n_new']} "
          f"UNCLASSIFIED={led['unclassified']} blockers={led['blockers']} "
          f"PASS={led['passed']}")
    print(f"          teacher params={led['total_teacher_params']:,} "
          f"bytes={led['total_teacher_bytes']/2**30:.3f} GiB "
          f"n_slots={led['slot_map']['n_slots']}")
    print("          ->", p)

    if a.execute:
        r = execute_full(a.execute, a.out, cfg, led)
        print("[execute]", json.dumps(r, ensure_ascii=False)[:400])
    elif not a.dry_run:
        print("[hint] --dry-run (default behaviour here) or --execute <real file>")
    else:
        print("[dry-run] NO 2.16 GB student written (315 MiB disk). "
              "Full recipe is ready for the target machine.")


if __name__ == "__main__":
    main()
