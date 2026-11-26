#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · Weight Provenance Ledger

V7's ledger was the source of three concrete defects:
  (1) only embed_tokens.weight was COPY'd; teacher lm_head.weight was dropped
      while tie_word_embeddings was forced True  -> logits semantically wrong.
  (2) k_proj/v_proj were zero-PADDED from (kv_dim,d) to (d,d); with R=2 the
      second SGC slot was identically zero -> half the state was dead.
  (3) q_norm/k_norm were SKIP'd with no functional counterpart recorded.
Here every teacher tensor must land in exactly one op, every DERIVE must state
its transform, and every NEW must declare its init AND prove it is in the
forward graph. copy_bit_identical is necessary but NOT sufficient.
"""
from __future__ import annotations

import json
import os
from collections import Counter


def classify(teacher_keys, spec) -> dict:
    v = spec.values
    nl = v.get("num_hidden_layers", 0)
    tie = v.get("tie_word_embeddings")
    attn = any("self_attn" in k or "attention" in k for k in teacher_keys)
    entries, unknown = [], []

    def add(k, op, dst=None, transform=None, note=""):
        entries.append(dict(src=k, op=op, dst=dst or k,
                            transform=transform, note=note))

    for k in teacher_keys:
        if k in ("model.embed_tokens.weight", "model.norm.weight"):
            add(k, "COPY", note="bit-identical required")
        elif k == "lm_head.weight":
            add(k, "COPY", note="untied teacher: standalone head MUST survive")
        elif k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
            add(k, "COPY")
        elif any(t in k for t in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")):
            add(k, "COPY", note="teacher SwiGLU retained verbatim")
        elif any(t in k for t in ("q_norm.weight", "k_norm.weight", "q_layernorm", "k_layernorm")):
            add(k, "DERIVE", dst=None, transform="FOLD_INTO_WqWk",
                note="per-head QK norm folded into the Wq/Wk column scale, NOT skipped")
        elif any(t in k for t in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")):
            add(k, "DERIVE", transform="GQA_HEAD_EXPAND_OR_PROJECT",
                note="no zero padding; slot layout derived from num_key_value_heads*head_dim")
        elif ".experts." in k or "expert" in k.lower():
            add(k, "DERIVE", transform="PER_EXPERT_MAP_REQUIRED",
                note="MoE: declare before T+0; refuse to guess")
        elif any(t in k for t in ("rotary", "rope", "inv_freq", "positions")):
            add(k, "DERIVE", transform="POSITIONAL_INTERFACE_REVIEW",
                note="W14: RoPE absolute-position defect was a historical 乱码 root cause")
        elif k.endswith(".bias"):
            add(k, "DERIVE", transform="BIAS_CARRY_OR_DECLARED_DROP")
        else:
            unknown.append(k)

    # Student-only NEW parameters. Each must be in the forward graph.
    news = []
    for i in range(nl):
        b = f"model.blocks.{i}."
        for name, init, role in (
            ("Wc", "zeros", "bounded causal depthwise conv"),
            ("Wg.weight", "zeros", "pointwise local gate"),
            ("wgate.weight", "zeros", "calibrated SGC write gate"),
            ("alpha", "scalar(-2.0)", "blend gate; MUST be calibrated, see align_accept"),
        ):
            news.append(dict(dst=b + name, op="NEW", init=init, role=role,
                             grad_required=True))
    news.append(dict(dst="model.blocks.*.slot_map", op="NEW", init="identity",
                     role="derived GQA->slot index map", grad_required=False))

    counts = Counter(e["op"] for e in entries) + Counter({"NEW": len(news)})
    blockers = []
    if unknown:
        blockers.append("UNCLASSIFIED_TEACHER_TENSORS:" + ",".join(unknown[:6]))
    if tie is False and "lm_head.weight" not in teacher_keys:
        blockers.append("UNTIED_TEACHER_BUT_NO_LM_HEAD_KEY")
    if tie is True and "lm_head.weight" in teacher_keys:
        blockers.append("TIED_TEACHER_BUT_STANDALONE_LM_HEAD_PRESENT")
    if not attn:
        blockers.append("NO_ATTENTION_KEYS_FOUND: QSA/MoE layout needs an explicit adapter")
    if spec.derived.get("n_slots") is None:
        blockers.append("SLOT_LAYOUT_UNRESOLVED")

    return dict(
        n_teacher=len(teacher_keys), n_layers=nl,
        counts=dict(counts),
        entries=entries, unclassified=unknown, new_params=news,
        blockers=blockers,
        passed=not blockers,
        invariants=[
            "UNCLASSIFIED == 0",
            "COPY tensors bit-identical AND shape/dtype identical",
            "tie_word_embeddings inherited, never forced",
            "lm_head.weight present iff teacher is untied",
            "no zero-padded KV rows; slot map derived from real head layout",
            "every NEW tensor has non-zero gradient on the first optimizer step",
            "ALL OF THE ABOVE IS NECESSARY BUT NOT SUFFICIENT: align_accept.py is the real gate",
        ],
    )


def write_ledger(out_dir: str, ledger: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "provenance_v8.json")
    json.dump(ledger, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return p
