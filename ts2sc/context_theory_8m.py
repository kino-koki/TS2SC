#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8M Token 上下文 · 理论验证（用真实 MiniCPM5-1B config，纯计算）

不吹「支持 8M」，而是分三层给出可核对的命题与反例条件：

  A  SimplCore 构造（SGC 分支）：8M 下状态是常数 → 理论成立
  B  当前 hybrid 迁移（保留 teacher 全注意力）：端到端 8M 不可行 → 给出确切数字
  C  要达到 8M 的可证条件 + 代价（会触碰「融合不替换」红线）

所有数字来自 sandbox 内真实 config.json，不猜架构参数。
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scale_theory import SNR_DEFAULT  # noqa: E402
from modeling_v8 import build_slot_map  # noqa: E402

BASE = "/data/workspace/minicpm5_1b"
OUT = "/data/workspace/output"
L = 8 * 1024 * 1024          # 8M tokens = 8,388,608


def main():
    c = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    d = c["hidden_size"]; nl = c["num_hidden_layers"]
    nq = c["num_attention_heads"]; nkv = c["num_key_value_heads"]
    hd = c["head_dim"]; V_size = c["vocab_size"]
    max_pos = c["max_position_embeddings"]; theta = c["rope_theta"]
    kv_dim = nkv * hd
    slot = build_slot_map(d, kv_dim, nq)
    R, ds = slot["n_slots"], d // slot["n_slots"]

    # ---------------- A: SimplCore 构造 ----------------
    # F05 实测过的单层状态：S ∈ R^{R×ds×ds} float32 + (K-1)×d hist
    s_bytes = R * ds * ds * 4
    hist_bytes = 2 * d * 4
    per_layer_state = s_bytes + hist_bytes
    total_state = per_layer_state * nl
    # FFT–IIR 对偶：O(L log L) 蝶形；IIR 递推每 token O(R·ds²) MAC
    fft_butterflies = L * math.log2(L)
    iir_mac_per_token_per_layer = R * ds * ds
    iir_total_mac = iir_mac_per_token_per_layer * nl * L
    # Theorem 3: 输入侧 bit 预算是 per-fibre 的，与 L 无关
    B_required = math.log2(R)          # role bit 提供 log2(R) bits
    A = dict(
        claim="静态 Toeplitz 核 + IIR 递推 ⇒ 状态与 L 无关",
        R=R, ds=ds,
        single_layer_state_bytes=per_layer_state,
        all_24_layers_state_bytes=total_state,
        all_24_layers_state_MiB=round(total_state / 2**20, 3),
        state_independent_of_L=True,
        fft_butterflies=round(fft_butterflies),
        iir_mac_per_token_per_layer=iir_mac_per_token_per_layer,
        iir_total_mac=round(iir_total_mac),
        streaming_needs_no_O_L_buffer=True,
        input_side_budget_bits=round(B_required, 4),
        budget_invariant_in_L=True,
        note="输入侧 B ≥ log₂|A(v)| 是 per-fibre 计数下界，不随 L 增长；"
             "role bit 的尺度可分辨性已由 F26 保证（距离 CV=0 跨 1e4×）。",
        theoretical_verdict="8M 下 SGC 分支状态恒定 36.28 MiB，理论成立。")

    # ---------------- B: hybrid 的 teacher 全注意力 ----------------
    kv_per_token_per_layer = 2 * nkv * hd * 2          # K,V × BF16
    kv_total = kv_per_token_per_layer * nl * L
    attn_pairs_per_layer_per_head = L * (L + 1) // 2
    attn_pairs_total = attn_pairs_per_layer_per_head * nq * nl
    rope_extrapolation_factor = L / max_pos
    # RoPE 在 8M 位置的最低频率相位（近似，用于说明混叠）
    rope_lowest_phase = L / (theta ** (2 * 0 / d)) if d else L
    B_ = dict(
        claim="当前迁移 = hybrid，teacher softmax attention 全部保留",
        max_position_embeddings=max_pos, rope_theta=theta,
        kv_bytes_per_token_per_layer=kv_per_token_per_layer,
        kv_total_bytes=kv_total,
        kv_total_GiB=round(kv_total / 2**30, 2),
        causal_attention_pairs_total=round(attn_pairs_total, 0),
        causal_attention_pairs_exponent=round(math.log10(attn_pairs_total), 2),
        rope_extrapolation_factor=round(rope_extrapolation_factor, 2),
        rope_aliasing=(L > max_pos),
        theoretical_verdict=(
            f"端到端 8M 不可行：KV cache ≈ {kv_total/2**30:.0f} GiB，"
            f"全因果 attention ≈ 10^{math.log10(attn_pairs_total):.1f} 对，"
            f"且 RoPE 需 {rope_extrapolation_factor:.0f}× 位置外推。"),
        why="红线：融合而非替换 ⇒ teacher attention 一层没删。"
            "SGC 的 O(1) 状态不能替 teacher 路径付 KV cache 的钱。")

    # ---------------- C: 达到 8M 的可证条件 ----------------
    rows = []
    for W in (1024, 4096, 8192, 32768, 131072):
        kv_w = kv_per_token_per_layer * nl * W
        rows.append(dict(window_W=W, kv_bytes=kv_w,
                         kv_MiB=round(kv_w / 2**20, 2),
                         feasible=bool(kv_w < 8 * 2**30)))
    C = dict(
        conditions=[
            "(1) teacher 路径改局部窗/滑窗 W，长程检索交给 SGC 分支（O(1) 状态）",
            "(2) 或分层路由：浅层 teacher 局部窗、深层 SGC 全局；"
            "代价是动了 teacher attention，与『融合不替换』红线冲突",
            "(3) RoPE 需 NTK/YaRN/位置重标定才能外推到 8M；"
            "SimplCore 静态核是平移不变的 Toeplitz，本身不依赖绝对位置",
        ],
        window_kv_table=rows,
        example="取 W=4096 ⇒ 24 层 KV ≈ 96 MiB，显存可行；"
                "但这是新架构，必须重跑 no-regression + L2/L3 配对验证",
        not_implemented=True,
        theoretical_verdict="8M 是 SGC 分支的理论性质，"
                            "不是当前 hybrid 迁移模型的已验证上下文长度。")

    # ---------------- 当前实测边界 ----------------
    mv = json.load(open(os.path.join(OUT, "multi_prompt_verify.json"),
                        encoding="utf-8"))
    measured = dict(
        n_questions=mv["non_accidental_stats"]["n_questions"],
        non_accidental=mv["non_accidental_stats"]["non_accidental"],
        beta1_all_finite=mv["non_accidental_stats"]["beta1_all_finite"],
        beta1_all_ids_in_range=mv["non_accidental_stats"]["beta1_all_ids_in_range"],
        max_new_tokens=mv["config"]["max_new"],
        longest_prompt_tokens=max(q["beta1_contract"]["n_prompt"]
                                  for q in mv["questions"]),
        total_tokens_tested=max(q["beta1_contract"]["n_prompt"]
                                + q["beta1_contract"]["n_new"]
                                for q in mv["questions"]),
        note="实测只覆盖短 prompt + 24 token 贪心生成；"
             "8M 属于理论推演，不是实测长度。")

    rep = dict(
        title="8M Token 上下文 · 理论验证（含反例条件）",
        date="2026-11-26",
        L_tokens=L,
        config_used=dict(hidden_size=d, num_hidden_layers=nl,
                         num_attention_heads=nq, num_key_value_heads=nkv,
                         head_dim=hd, vocab_size=V_size,
                         max_position_embeddings=max_pos, rope_theta=theta,
                         n_slots=R, slot_dim=ds,
                         source="minicpm5_1b/config.json 真实字段"),
        A_simplcore_construction=A,
        B_current_hybrid=B_,
        C_conditions_for_8M=C,
        D_measured_boundary=measured,
        final_answer=(
            "理论上：SimplCore 的静态 Toeplitz + IIR 递推在 8M 下状态仍是 "
            f"{total_state/2**20:.2f} MiB 常数，输入侧 bit 预算也不随 L 增长，"
            "这一层成立且有实测不变量支撑。"
            "但当前迁移是 hybrid、保留 teacher 全注意力，端到端 8M 需要 "
            f"≈{kv_total/2**30:.0f} GiB KV cache 与 10^{math.log10(attn_pairs_total):.1f} "
            "量级 attention 对，不可行。"
            "要真正做到 8M 必须动 teacher 路径（滑窗/路由/位置重标定），"
            "那是新架构，需要重新验证零回归。今天不声称支持 8M。"),
        boundaries=["无 8M 实测（无对应算力/数据/显存）",
                    "不写『原生级 / 能力更强 / SOTA』",
                    "Qwen4-27B 未发布，本轮与其无关"],
    )
    p = os.path.join(OUT, "context_8m_theory.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(dict(A=A, B=B_, C=C, D=measured),
                     indent=2, ensure_ascii=False))
    print("\n[done]", p)


if __name__ == "__main__":
    main()
