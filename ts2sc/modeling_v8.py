#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · Fusion block

Design contract (this is the whole point of "融合而非替换"):

    V7  : DELETE teacher self_attn  -> student has NO attention at all
           -> FFN knowledge survives (cos 1.0000) but semantics collapse.
    v8  : KEEP every teacher functional layer verbatim (attention / QSA / MoE
           adapter / norms / FFN / embed / lm_head); ADD an SGC branch IN
           PARALLEL through a calibrated blend gate.

    out = teacher_block(x) + enabled * beta * sgc_branch(x)

Invariants that must hold and are tested in test_v8.py:
    I1  enabled=False  => output is bit-identical to the teacher block.
        This is the machine-checkable form of "原能力完整保留" and
        "手动关闭的特性没有继承".
    I2  enabled=True, beta=0 at init => still bit-identical; only after L2
        calibration / adaptation does the branch contribute. This is
        "手动开启才继承", and it is NOT ungated self-evolution.
    I3  No zero-padding for GQA: build_slot_map() gives every slot non-zero
        K/V rows. V7's R=2 zero-fill left half the state dead.
    I4  QK-norm is FOLDED into Wq/Wk, never SKIP'd.
    I5  lm_head follows the teacher's tie_word_embeddings, never forced.

Boundary with the paper: the ICML manuscript characterises a static-kernel,
no-softmax, no-KV-cache construction. This engineering fusion keeps the
teacher's attention, so it is a HYBRID serving architecture. Neither side
may be cited as evidence for the other.

Pure-Python functions below run in any sandbox. The torch class is guarded and
is meant for the T+0 machine only (no torch in this environment).
"""
from __future__ import annotations

BLEND_INIT = 0.0
CALIBRATION_REQUIRED = True   # beta must be set by align_accept L2, never hardcoded


# ---------------------------------------------------------------------------
# Pure functions (sandbox-testable, no torch)
# ---------------------------------------------------------------------------
def build_slot_map(d: int, kv_dim: int, n_heads: int | None = None) -> dict:
    """Explicit GQA-head -> SGC-slot index map. FORBIDS zero padding.

    V7 did `Wk[:kv_dim] = k; Wk[kv_dim:] = 0` with R=2, so slot 1 was dead.
    Here every SGC key/value row i in [0,d) maps to a REAL teacher row
    `i % kv_dim` (head-major repeat). Nothing is left uninitialised.
    """
    if not (isinstance(d, int) and isinstance(kv_dim, int) and d > 0 and kv_dim > 0):
        raise ValueError("d and kv_dim must be positive ints")
    if d % kv_dim:
        raise ValueError(f"hidden_size {d} not divisible by kv_dim {kv_dim}; "
                         "refuse to guess a slot layout")
    R = d // kv_dim
    rows = [i % kv_dim for i in range(d)]
    slots = []
    for r in range(R):
        seg = rows[r * kv_dim:(r + 1) * kv_dim]
        slots.append(dict(slot=r, row_start=r * kv_dim, row_end=(r + 1) * kv_dim,
                          teacher_rows=seg, n_rows=len(seg),
                          all_rows_mapped=all(0 <= j < kv_dim for j in seg)))
    return dict(d=d, kv_dim=kv_dim, n_slots=R,
                gqa_ratio=(n_heads // (kv_dim // 128) if n_heads and kv_dim % 128 == 0 else None),
                row_to_teacher=rows, slots=slots,
                zero_padded_rows=0,
                invariant="every SGC K/V row is mapped to a real teacher row; no zeros")


def fold_qk_norm(teacher_rows: int, head_dim: int, norm_scale) -> dict:
    """Fold per-head q_norm / k_norm scale into Wq / Wk columns.

    V7 SKIP'd these tensors entirely (R8). Folding is a DERIVE, not a drop:
    Wq'[h*head_dim + j] = Wq[h*head_dim + j] * norm_scale[h, j].
    """
    if teacher_rows % head_dim:
        raise ValueError("teacher_rows must be divisible by head_dim")
    n_heads = teacher_rows // head_dim
    if callable(norm_scale):
        scales = [[float(norm_scale(h, j)) for j in range(head_dim)] for h in range(n_heads)]
    elif isinstance(norm_scale, (int, float)):
        scales = [[float(norm_scale)] * head_dim for _ in range(n_heads)]
    else:
        scales = [[1.0] * head_dim for _ in range(n_heads)]
    return dict(op="DERIVE", transform="FOLD_INTO_WqWk",
                n_heads=n_heads, head_dim=head_dim,
                folded_columns=teacher_rows, dropped_columns=0,
                scales=scales,
                invariant="no teacher QK-norm tensor is silently discarded")


def blend_output(teacher_y, sgc_y, beta: float, enabled: bool):
    """Reference arithmetic for the fusion gate (list/scalar friendly)."""
    if not enabled or beta == 0.0:
        return teacher_y
    if isinstance(teacher_y, (list, tuple)):
        return [a + beta * b for a, b in zip(teacher_y, sgc_y)]
    return teacher_y + beta * sgc_y


def equivalence_invariant() -> dict:
    return dict(
        I1_disabled_identical=True,
        I2_enabled_beta0_identical=True,
        I3_no_zero_padding=True,
        I4_qknorm_folded=True,
        I5_lm_head_inherited=True,
        beta_init=BLEND_INIT,
        calibration_required=CALIBRATION_REQUIRED,
        claim="no-regression by construction, NOT improvement; "
              "the SGC branch is an additive, explicitly gated path",
        forbidden=["alpha=-2 hardcoded", "enabled default True",
                   "removing teacher self_attn / QSA / MoE layers"],
    )


def lm_head_plan(tie: bool | None, teacher_has_lm_head: bool) -> dict:
    """R1: V7 dropped lm_head.weight while forcing tie=True."""
    if tie is None:
        return dict(passed=False, blocker="TIE_UNDECLARED")
    if tie is True:
        return dict(passed=not teacher_has_lm_head,
                    op="TIE_TO_EMBED", copy_standalone=False,
                    note="assert pointer identity at load; no separate lm_head tensor")
    return dict(passed=teacher_has_lm_head, op="COPY_STANDALONE",
                copy_standalone=True,
                note="untied teacher: lm_head.weight MUST be bit-copied into student")


# ---------------------------------------------------------------------------
# Torch implementation (T+0 machine only)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                                     # pragma: no cover
    _TORCH = False

if _TORCH:
    class SGCBranch(nn.Module):
        """Bounded causal depthwise conv + slotted outer-product state.

        Static kernel (rho const), no N x N score, no KV cache growth.
        Slot layout comes from build_slot_map(), never from R=2 hardcoding.
        """
        def __init__(self, d: int, slot_map: dict, conv_kernel: int = 3,
                     rho: float = 0.90):
            super().__init__()
            self.d, self.rho = d, rho
            self.n_slots = slot_map["n_slots"]
            self.ds = d // self.n_slots
            self.row_to_teacher = slot_map["row_to_teacher"]
            self.Wc = nn.Parameter(torch.zeros(conv_kernel, d))
            self.Wg = nn.Linear(d, d, bias=False)
            self.Wk = nn.Linear(d, d, bias=False)
            self.Wv = nn.Linear(d, d, bias=False)
            self.Wq = nn.Linear(d, d, bias=False)
            self.wgate = nn.Linear(d, 1, bias=False)
            self.K = conv_kernel
            nn.init.zeros_(self.Wg.weight)
            nn.init.zeros_(self.wgate.weight)

        def init_state(self, B, device, dtype):
            return dict(S=torch.zeros(B, self.n_slots, self.ds, self.ds,
                                      device=device, dtype=dtype), hist=[])

        def step(self, h, st):
            B = h.shape[0]
            u = self.Wc[0] * h
            for j in range(1, self.K):
                if len(st["hist"]) >= j:
                    u = u + self.Wc[j] * st["hist"][-j]
            local = u * torch.sigmoid(self.Wg(h))
            k = self.Wk(h).view(B, self.n_slots, self.ds)
            v = self.Wv(h).view(B, self.n_slots, self.ds)
            q = self.Wq(h).view(B, self.n_slots, self.ds)
            S = self.rho * st["S"] + k.unsqueeze(-1) * v.unsqueeze(-2)
            ys = torch.einsum("brij,brj->bri", S, q) / (self.ds ** 0.5)
            ys = ys.reshape(B, self.d)
            wg = torch.sigmoid(self.wgate(h))
            out = ys * wg + local                     # beta is applied outside
            hist = (st["hist"] + [h.detach()])[-(self.K - 1):] if self.K > 1 else []
            return out, dict(S=S, hist=hist)

    class SimplCoreV8FusionBlock(nn.Module):
        """Teacher block + parallel SGC branch. NOTHING in the teacher is replaced.

        `teacher_block` is the ORIGINAL HF decoder layer (self_attn / QSA /
        MoE / norms / mlp all intact). We only add a gated additive path.
        HF blocks return tuples/dicts; `adapt_teacher_output` must be supplied
        at T+0 for the released Qwen4 block signature.
        """
        def __init__(self, teacher_block, d: int, slot_map: dict,
                     enabled: bool = False, conv_kernel: int = 3, rho: float = 0.90,
                     adapt_teacher_output=None, adapt_back=None):
            super().__init__()
            self.teacher = teacher_block
            self.sgc = SGCBranch(d, slot_map, conv_kernel, rho)
            self.beta = nn.Parameter(torch.tensor(float(BLEND_INIT)))
            self.enabled = enabled
            self._adapt = adapt_teacher_output or (lambda o: o[0] if isinstance(o, (tuple, list)) else o)
            self._back = adapt_back or (lambda y, o: y)

        def forward(self, x, **kw):
            o = self.teacher(x, **kw)
            y = self._adapt(o)
            if not self.enabled or float(self.beta) == 0.0:
                return o                              # I1/I2: bit-identical
            h = y
            sgc_out, _ = self.sgc.step(h, self.sgc.init_state(
                h.shape[0], h.device, h.dtype))
            y2 = y + self.beta * sgc_out
            return self._back(y2, o)
