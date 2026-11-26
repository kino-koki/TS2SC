#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.5 · 「完全相同 / 甚至更强」的可证伪协议

用户要求：优化模型，直到达到与原模型完全相同的能力甚至更强。

先分清两件事，别混：
  「完全相同」  —— **已经达成**，不需要优化。β_l=0 全层 ⇒ 与 teacher 逐位相同：
                  F11 实测 identical_to_teacher=True；12 臂 β=0 全部
                  gate=True / readable=True / logits finite。这是硬保证不是近似。
  「甚至更强」  —— **今天未达成，且不允许声称**。它需要一个带置信下界的
                  配对实验，本文件定义门槛与接口，不产出任何假数字。

本模块只做三件事：
  (1) 定义逐层 β_l ∈ [0,1] 的参数化与初值策略（已接进 SGCFullBranch）
  (2) 定义 L2 校准目标（teacher 同条件 forward 的逐层/逐 logit 约束）
  (3) 定义 L3 判定门槛（配对 Wilson CI 下界 > 0 才算「更强」）

严禁：把 24 token 短生成、可读性、首句不同、或尺度契约下的 finite，
当作能力/正确性/ppl/SOTA 证据。
"""
from __future__ import annotations

import json
import math

import numpy as np

N_LAYERS = 24
EPS = 0.10


# ---------------------------------------------------------------------------
# (1) 逐层 β_l 参数化
# ---------------------------------------------------------------------------
def beta_init(strategy: str = "zeros", n: int = N_LAYERS):
    """返回长度 n 的 β_l 向量，全部落在 [0,1] 无量纲区间。

    zeros        : 全 0 —— 与 teacher 逐位相同（交付默认，零回归）
    uniform      : 全 1 —— 当前 β=1 契约态（已实测会说话，非校准成品）
    shallow_warm : 浅层小、深层大 —— 供 L2 网格搜索的确定性起点，不是训练结果
    """
    if strategy == "zeros":
        return np.zeros(n, np.float64)
    if strategy == "uniform":
        return np.ones(n, np.float64)
    if strategy == "shallow_warm":
        return np.linspace(0.05, 0.60, n)
    raise ValueError(f"unknown strategy {strategy}")


def contract_bound(beta_l, eps: float = EPS):
    """逐层契约：每层注入比例 ≤ β_l·ε；全 1 时残差放大 ≤ 1+ε。"""
    b = np.asarray(beta_l, np.float64)
    assert b.shape == (N_LAYERS,) and np.all((0 <= b) & (b <= 1))
    return dict(per_layer_max_ratio=(b * eps).tolist(),
                global_max_residual_gain=float(1 + eps * b.max()),
                all_bounded=bool(np.all(b * eps <= eps + 1e-12)),
                inequality="‖x+Δ‖ ≤ (1+β_l·ε)‖x‖ ≤ 1.1‖x‖")


# ---------------------------------------------------------------------------
# (2) L2 校准目标
# ---------------------------------------------------------------------------
L2_TARGETS = dict(
    name="L2 scale calibration (teacher-conditioned)",
    requires="同一批 prompt、同一 teacher 权重、同条件 forward（当前沙盒未跑）",
    metrics=[
        dict(metric="per_layer_hidden_cosine", symbol="cos_l",
             threshold="≥ 1 − 1e−3 for all 24 layers",
             purpose="保证融合分支没把 teacher 的逐层表征拽偏"),
        dict(metric="logit_kl_divergence", symbol="KL(p_teacher‖p_student)",
             threshold="≤ δ (δ 由 held-out 集上的 teacher 自比确定)",
             purpose="保证输出分布不漂移"),
        dict(metric="per_layer_residual_ratio", symbol="‖Δ_l‖/‖x_l‖",
             threshold="≤ β_l·ε", purpose="尺度契约的实测核对"),
        dict(metric="beta_fit", symbol="β_l*",
             threshold="argmin over grid, 保留 no-regression 约束",
             purpose="把 24 个标量拟出来，而不是硬编码 1.0"),
    ],
    grid=[round(v, 2) for v in np.linspace(0, 1, 11)],
    note="L2 只校准尺度，不产生任何能力数字。")


# ---------------------------------------------------------------------------
# (3) L3 判定门槛
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96):
    """配对成功率的 Wilson 区间。用于判定 paired difference 的下界。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


L3_GATES = dict(
    name="L3 paired no-regression / improvement gate",
    requires="同 seed、同数据顺序、等算力预算、只切 β_l 的配对实验（当前未跑）",
    verdicts=[
        dict(claim="与原模型完全相同",
             criterion="β_l ≡ 0 ⇒ hidden / logits / 生成逐位相同",
             status="ALREADY PROVEN (F11 + 12 臂 β=0)",
             may_claim=True),
        dict(claim="不退化 (no-regression)",
             criterion="配对差 Δ 的 Wilson 95% CI 下界 ≥ −tol "
                       "且 mojibake_rate 不升、越界 id = 0",
             status="NOT YET RUN", may_claim=False),
        dict(claim="甚至更强 (strictly better)",
             criterion="在 held-out 任务集上配对差 Δ 的 Wilson 95% CI "
                       "**下界 > 0**，且 β=0 对照臂不退化、等算力基线齐备",
             status="NOT YET RUN", may_claim=False),
    ],
    forbidden_evidence=[
        "24 token 短生成可读",
        "首句与 β=0 不同（只证明分支有贡献）",
        "logits finite / 0 越界 id（只证明不崩）",
        "单题 / 单 seed / 无配对",
        "ppl 或 benchmark 数字（当前一个都没跑）",
    ],
    note="Theorem 10 的 paired no-regression gate 是渐近一致的，"
         "但 finite-n 覆盖未被声称；必须真跑配对样本才能给出区间。")


# ---------------------------------------------------------------------------
# 汇总：当前能说什么 / 不能说什么
# ---------------------------------------------------------------------------
def status():
    return dict(
        identical_to_teacher=dict(achieved=True,
                                  evidence="β_l≡0 逐位相同；F11 + 12 臂 β=0 全绿",
                                  needs_optimization=False),
        beta_positive_speaks=dict(achieved=True,
                                  evidence="6 题 β=1 契约态 non_accidental=True",
                                  caveat="尺度契约配置，非 L2 校准成品"),
        no_regression=dict(achieved=False, needs="L2 + L3 配对实验"),
        strictly_better=dict(achieved=False, needs="Wilson CI 下界 > 0 的配对证据"),
        context_8M=dict(theoretical_simplcore_branch=True,
                        end_to_end_hybrid=False,
                        needs="teacher 路径滑窗/路由 + 位置重标定 + 重验零回归"),
        honest_line=("『完全相同』已达成；『甚至更强』今天不能声称。"
                     "继续优化的正确动作是跑 L2/L3，而不是把短生成包装成能力结论。"),
    )


if __name__ == "__main__":
    out = dict(
        title="完全相同 / 甚至更强 · 可证伪优化协议",
        beta_init={s: beta_init(s).tolist() for s in
                   ("zeros", "uniform", "shallow_warm")},
        contract=contract_bound(beta_init("uniform")),
        L2=L2_TARGETS, L3=L3_GATES,
        wilson_example=dict(k=60, n=100, ci=list(wilson_ci(60, 100))),
        status=status())
    print(json.dumps(out, indent=2, ensure_ascii=False))
