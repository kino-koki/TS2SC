#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · TeacherSpec

Design rule: NOTHING about the teacher architecture is hardcoded.
V7 failed on Qwen3-shaped constants (R=2, V=151936, bos=151643,
eos=151645, max_position=40960, self_attn.* key names). Qwen4-27B has no
published config yet, so every field is either read from the released
config.json or declared as a T0 blocker -- never guessed, never defaulted.

No torch / transformers import: this module is pure stdlib so that the
preflight can run on a machine without a GPU or a package index.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

# Fields that MUST be present in the released teacher config. Anything absent
# is a blocker, not a default. Defaults are how V7 silently corrupted lm_head.
REQUIRED = (
    "model_type", "vocab_size", "hidden_size", "num_hidden_layers",
    "intermediate_size", "tie_word_embeddings", "max_position_embeddings",
    "bos_token_id", "eos_token_id", "torch_dtype",
)
# Attention layout. On Qwen4 these may be MoE / QSA / sparse-attention names;
# if absent we do NOT assume GQA and we do NOT emit a migration plan.
ATTN_FIELDS = (
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "attention_bias", "qk_norm", "rope_theta", "sliding_window",
)
# Known-unknown at 2026-11-25. Filled at T+0 from the released model card.
QWEN4_27B_PENDING = dict(
    model="Qwen4-27B",
    status="ANNOUNCED_NOT_RELEASED",
    announced="2026-09-22 (云栖大会, 开放权重档)",
    released_members="Qwen4-Max 2026-11-16 (2.4T MoE / 71B active / QSA / 1M ctx)",
    open_weights_smaller="官方称『短期内发布』，无日期、无模型卡、无权重、无 config、无 tokenizer、无许可证",
    context_hint_only="Qwen Code PR #15762 给 qwen4* 标识 1M token 回退值；是线索不是规格",
)


@dataclass
class TeacherSpec:
    src: str
    values: dict = field(default_factory=dict)
    missing_required: list = field(default_factory=list)
    unknown_attn: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    derived: dict = field(default_factory=dict)

    @classmethod
    def from_dir(cls, teacher_dir: str) -> "TeacherSpec":
        p = os.path.join(teacher_dir, "config.json")
        if not os.path.exists(p):
            raise SystemExit(f"[spec] missing {p}; cannot migrate without a real config")
        cfg = json.load(open(p, encoding="utf-8"))
        # text_config / llm_config wrapper for VL-style releases
        inner = cfg.get("text_config") or cfg.get("llm_config") or cfg
        merged = dict(cfg)
        merged.update(inner)

        s = cls(src=teacher_dir, values=merged)
        s.missing_required = [k for k in REQUIRED if k not in merged]
        s.unknown_attn = [k for k in ATTN_FIELDS if k not in merged]

        if s.missing_required:
            s.blockers.append("REQUIRED_CONFIG_FIELDS_MISSING: " + ",".join(s.missing_required))
        if not any(k in merged for k in ("num_attention_heads", "n_head", "qsa_config")):
            s.blockers.append("ATTENTION_LAYOUT_UNKNOWN: refuse to assume GQA / QKV key names")
        if "num_experts" in merged or merged.get("model_type", "").lower().find("moe") >= 0:
            s.blockers.append("MOE_DETECTED: per-expert COPY/DERIVE mapping must be declared before T+0")
        if merged.get("tie_word_embeddings") is None:
            s.blockers.append("TIE_WORD_EMBEDDINGS_UNDECLARED: V7 forced True and dropped teacher lm_head")

        # Slot count derived, never hardcoded. V7 used R=2.
        d = merged.get("hidden_size")
        kv = merged.get("num_key_value_heads")
        hd = merged.get("head_dim")
        if all(isinstance(x, int) and x > 0 for x in (d, kv, hd)):
            kv_dim = kv * hd
            s.derived["kv_dim"] = kv_dim
            s.derived["n_slots"] = d // kv_dim if d % kv_dim == 0 else None
            s.derived["gqa_ratio"] = (merged.get("num_attention_heads", 0) or 0) // kv
            if s.derived["n_slots"] is None:
                s.blockers.append("HIDDEN_NOT_DIVISIBLE_BY_KV_DIM: cannot derive slot layout")
        else:
            s.derived["n_slots"] = None
            s.blockers.append("SLOT_LAYOUT_UNRESOLVED: hidden/kv/head_dim incomplete")

        # Qwen4-27B is not released: emit the pending record as an explicit T0 gate.
        s.derived["target_pending"] = QWEN4_27B_PENDING
        s.derived["t0_gate"] = (
            "PASS only after real config.json + tokenizer.json + safetensors index are all present"
        )
        return s

    @property
    def t0_ready(self) -> bool:
        return not self.blockers and not self.missing_required

    def to_dict(self):
        return asdict(self)

    def student_config(self) -> dict:
        """Student config inherits teacher identity fields verbatim.
        Crucially tie_word_embeddings is INHERITED, never forced.
        """
        v = self.values
        keys = ("vocab_size", "hidden_size", "num_hidden_layers",
                "intermediate_size", "tie_word_embeddings", "max_position_embeddings",
                "bos_token_id", "eos_token_id", "torch_dtype", "rms_norm_eps")
        out = {k: v[k] for k in keys if k in v}
        out.update(
            model_type="simplcore_v8",
            architectures=["SimplCoreV8ForCausalLM"],
            n_slots=self.derived.get("n_slots"),
            conv_kernel=3,
            rho=0.90,
            tokenizer_binding="REQUIRED",   # see contract.py
            teacher_spec_sha="computed at T+0",
        )
        return out


if __name__ == "__main__":
    import sys
    print(json.dumps(TeacherSpec.from_dir(sys.argv[1]).to_dict(), indent=2, ensure_ascii=False))
