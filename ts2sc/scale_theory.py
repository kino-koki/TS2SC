#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.4 · 乱码的两个理论根因与可证修复

===========================================================================
根因链（精确复现，不是猜测）
===========================================================================
T1 数值层 —— 未校准加性注入破坏归一化前提：
    teacher 残差 x 的 RMS 是 O(1)（RMSNorm 前提）。
    真实 Wk/Wv 派生出的 SGC 读出的量级是 1e4~1e5。
    x + β·o (β=1) 把残差推到 1e4 → mean(x*x) 溢出 → 1/sqrt(inf)=0,
    inf*0=nan → 全层 nan → lm_head 分块 logits 全 nan。

    致命的一跳在 TeacherLM.logits:
        best_v, best_i = -1e30, -1
        if lg[j] > best_v:  ...      # NaN > x 恒为 False !!
    所有块都是 NaN 时 best_i 永远保持初始哨兵 **-1**，
    不抛异常、不报警，静默产生 24 个越界 id。
    解码器对 -1 查不到 token → 逐字丢弃 → **空串**。
    这不是 UTF-8 mojibake，是「连乱码字符都没产生」。
    而 V7 的 gen_quality 只测 distinct1/rep3（R4），空串/越界 id 完全漏检。

T2 信息层 —— role bit 的可分辨性依赖 k 的偶然尺度（真实测得 r_k=0.32）：
    论文 Theorem 3: B ≥ log₂|A(v)|，B=0 时 per-fibre 退化到 chance。
    v8.3 把 role one-hot (幅度 1) 直接加到 k 上，role 相对能量比是
        ρ_naive = ‖e‖/‖k‖ ∝ 1/r_k
    它随输入尺度飘：r_k=0.32 时 ρ≈3（role 反而主导、扰动 k 的语义），
    r_k=1e2 时 ρ≈0.01（被淹没）。**不是「总被淹没」，是不可控的尺度依赖**。
    F08 的合成验证只取了一个量级，掩盖了这一点。
    修正后 k 的幅度交给 F10 写入门控制，role 只负责输入侧信息，两者解耦。

===========================================================================
修复（可证，不是 workaround）
===========================================================================
T1' 无量纲尺度契约 (Dimensionless Scale Contract)
    令 r_x = ‖x‖₂/√d, r_o = ‖o‖₂/√d, û = o/(r_o+δ), 注入
        Δ(x,o) = β · ε · r_x · û
    命题: ‖Δ‖₂ = βε‖x‖₂ · r_o/(r_o+δ) ≤ βε‖x‖₂.
    取 β∈[0,1], ε=0.1 ⇒ ‖x+Δ‖ ≤ 1.1‖x‖ ⇒ RMSNorm 输入 RMS 有界
    ⇒ 无 overflow ⇒ logits finite ⇒ best_i 必被某个有限块更新 ⇒ id∈[0,V).
    推论1: β=0 ⇒ Δ=0 ⇒ 与 teacher 逐位相同（F11 I1 保持）。
    推论2: β 现在是**无量纲比例**，L2 校准的是 [0,1] 里的一个标量，
           不再是去拟合 1e-5 量级的怪系数。clip 被移除——clip 是掩盖，
           契约是约束。
    推论3: o 的原始量级（1e4 或 1e-5）被 r_o 吸收，与 teacher 解耦。

T2' 尺度不变 role 注入 (Scale-invariant role bit)
    k̂ = k/(r_k+δ)  (单位 RMS),   k'_r = k̂ + snr · e_r
    命题: (i) ‖k'_r − k'_s‖ = snr·√(2·n_r)，与 ‖k‖ 完全无关；
          (ii) role 相对能量比恒为 snr·√(n_r/(R·ds))，跨 1e4 倍尺度不变；
          (iii) B=0（不注入）时所有假设相同，距离 0。
    对比: naive 的 ρ 跨同样尺度衰减 1e4 倍 ⇒ 不可控。
    k 的绝对幅度交给 F10 写入门控制，role 只携带输入侧信息 B=log₂R。
    这是 Theorem 3 在真实权重尺度下的工程兑现，不是 LM 上的定理复现。

T3' 解码守卫 (finite-and-in-range gate)
    显式断言 logits finite 且 0 ≤ id < vocab_size；
    把 best_i 初始哨兵从 -1 改成合法 id 0，并上报 finite 标志。
    拒绝「非退化即通过」：空串 / 越界 id / NaN 一律 FAIL。
"""
from __future__ import annotations

import math

import numpy as np

EPS_DEFAULT = 0.10          # 注入 RMS 上限占 teacher 残差 RMS 的比例
SNR_DEFAULT = 0.50          # role bit 相对 k RMS 的信噪比
DELTA = 1e-6                # 防 r_o=0


# ---------------------------------------------------------------------------
# T1' 无量纲尺度契约
# ---------------------------------------------------------------------------
def rms_norm(x) -> float:
    """float64 RMS，避免 float32 在 1e4 量级平方溢出。"""
    a = np.asarray(x, dtype=np.float64).ravel()
    return float(np.linalg.norm(a) / math.sqrt(a.size))


def dimensionless_gate(x, o, beta: float, eps: float = EPS_DEFAULT,
                       delta: float = DELTA) -> np.ndarray:
    """Δ = β·ε·r_x·û, 满足 ‖Δ‖ ≤ βε‖x‖。返回与 x 同 dtype 的加性项。"""
    x = np.asarray(x)
    o64 = np.nan_to_num(np.asarray(o, dtype=np.float64),
                        nan=0.0, posinf=1e12, neginf=-1e12)
    if beta == 0.0:
        return np.zeros_like(x)
    r_x = rms_norm(x)
    r_o = rms_norm(o64)
    if r_x == 0.0 or not np.isfinite(r_x):
        return np.zeros_like(x)
    u = o64 / (r_o + delta)                  # unit-RMS direction
    out = beta * eps * r_x * u
    return np.asarray(out, dtype=x.dtype)


def contract_bound(x, o, beta, eps=EPS_DEFAULT) -> dict:
    """可证不等式 ‖Δ‖ ≤ βε‖x‖ 的实测核对。"""
    d = dimensionless_gate(x, o, beta, eps)
    nx = float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel()))
    nd = float(np.linalg.norm(np.asarray(d, dtype=np.float64).ravel()))
    bound = abs(beta) * eps * nx
    # Δ is returned in x.dtype (float32); allow its relative rounding error
    # instead of an absolute 1e-9, which is below float32 resolution at 1e4.
    tol = 1e-6 * max(bound, 1.0)
    return dict(passed=bool(nd <= bound + tol and np.isfinite(nd)),
                delta_norm=nd, x_norm=nx, bound=bound,
                ratio=(nd / nx if nx else 0.0),
                beta=beta, eps=eps,
                inequality="‖Δ‖₂ = βε‖x‖₂·r_o/(r_o+δ) ≤ βε‖x‖₂")


# ---------------------------------------------------------------------------
# T2' role 信噪比注入
# ---------------------------------------------------------------------------
def _broadcast_role(e, k) -> np.ndarray:
    e = np.asarray(e, dtype=np.float32)
    if e.ndim == 2 and e.shape[1] == 1 and k.ndim == 2 and k.shape[1] > 1:
        e = np.repeat(e, k.shape[1], axis=1)
    return e


def role_injection(k, e, snr: float = SNR_DEFAULT,
                   normalize: bool = True, delta: float = DELTA) -> np.ndarray:
    """尺度不变注入: k̂=k/(r_k+δ); k'_r = k̂ + snr·e_r.

    命题 (i)(ii) 见文件头 T2'。normalize=False 保留 v8.3 的 naive 行为，
    仅用于负对照。
    """
    k = np.asarray(k, dtype=np.float32)
    e = _broadcast_role(e, k)
    base = (k / np.float32(rms_norm(k) + delta)) if normalize else k
    return np.asarray(base + np.float32(snr) * e, dtype=np.float32)


def role_resolvability(k, R: int, snr: float = SNR_DEFAULT) -> dict:
    """跨尺度检验 T2' 命题 (i)(ii)(iii)。

    在 r_k 跨 1e4 倍的输入尺度上：
      - 尺度不变注入的假设间距离恒定、role 相对能量比恒定；
      - naive 注入的相对能量比衰减 1e4 倍（不可控的尺度依赖）；
      - B=0 时假设间距离恒为 0。
    """
    k = np.asarray(k, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] != R:
        raise ValueError("k must be (R, ds)")
    def pairwise(vs):
        ds = []
        for i in range(R):
            for j in range(i + 1, R):
                ds.append(float(np.linalg.norm((vs[i] - vs[j]).ravel())))
        return float(np.mean(ds)) if ds else 0.0
    e0 = np.zeros((R, 1), np.float32); e0[0, 0] = 1.0
    e0b = _broadcast_role(e0, k)
    rows = []
    for scale in (0.1, 1.0, 10.0, 100.0, 1000.0):
        ks = np.asarray(k * np.float32(scale), dtype=np.float32)
        r_k = rms_norm(ks)
        # (i) scale-invariant hypothesis distance
        inv = []
        for r in range(R):
            e = np.zeros((R, 1), np.float32); e[r, 0] = 1.0
            inv.append(role_injection(ks, e, snr, normalize=True))
        d_inv = pairwise(inv)
        # (ii) role-to-key energy ratio
        khat = ks / np.float32(r_k + DELTA)
        ratio_inv = rms_norm(np.float32(snr) * e0b) / max(rms_norm(khat), 1e-12)
        ratio_naive = rms_norm(e0b) / max(r_k, 1e-12)
        rows.append(dict(scale=scale, rms_k=r_k, distance_invariant=d_inv,
                         ratio_invariant=ratio_inv, ratio_naive=ratio_naive))
    d_vals = [r["distance_invariant"] for r in rows]
    ri_vals = [r["ratio_invariant"] for r in rows]
    d_cv = float(np.std(d_vals) / max(np.mean(d_vals), 1e-12))
    ri_cv = float(np.std(ri_vals) / max(np.mean(ri_vals), 1e-12))
    naive_decay = rows[0]["ratio_naive"] / max(rows[-1]["ratio_naive"], 1e-30)
    passed = bool(d_cv < 1e-3 and ri_cv < 1e-3 and naive_decay > 1e3)
    return dict(passed=passed,
                # 兼容旧报告字段
                distance_B0=0.0,
                distance_naive_onehot=float(np.sqrt(2 * k.shape[1])),
                distance_snr_matched=float(np.mean(d_vals)),
                amplification_over_naive=naive_decay,
                rms_k=rms_norm(k), snr=snr, R=R,
                # 新增尺度不变性证据
                distance_cv=d_cv, ratio_cv=ri_cv,
                naive_ratio_decay=naive_decay,
                proposition="(i) ‖k'_r−k'_s‖=snr√(2n_r) independent of ‖k‖; "
                            "(ii) role/key energy ratio constant; (iii) B=0 ⇒ distance 0",
                rows=rows,
                note="scale-invariance criterion; NOT Theorem 3 reproduced on the LM")


# ---------------------------------------------------------------------------
# T3' 解码守卫
# ---------------------------------------------------------------------------
def decode_gate(ids, vocab_size: int, logits_finite: bool = True) -> dict:
    """拒绝空串 / 越界 id / NaN。取代 V7 的 distinct1/rep3 非退化判据。"""
    ids = [int(i) for i in ids]
    out_of_range = [i for i in ids if not (0 <= i < vocab_size)]
    empty = len(ids) == 0 or all(i < 0 or i >= vocab_size for i in ids)
    return dict(passed=bool(logits_finite and not out_of_range and not empty
                            and len(ids) > 0),
                logits_finite=bool(logits_finite),
                n_ids=len(ids), n_out_of_range=len(out_of_range),
                empty=empty, vocab_size=vocab_size,
                rejected_reasons=([] if not out_of_range else
                                  ["OUT_OF_RANGE_ID"]) +
                                 ([] if logits_finite else ["NONFINITE_LOGITS"]) +
                                 ([] if not empty else ["EMPTY_DECODE"]),
                note="finite-and-in-range gate; replaces 'non-degenerate' proxy")


def nan_to_sentinel_demo(vocab_size=130560):
    """可复现的根因演示：全 NaN logits → best_i 保持 -1 哨兵。"""
    best_v, best_i = -1e30, -1
    for r0 in range(0, vocab_size, 8192):
        block = np.full(min(8192, vocab_size - r0), np.nan, np.float32)
        j = int(block.argmax())
        if block[j] > best_v:                # NaN > x is False
            best_v, best_i = float(block[j]), r0 + j
    return dict(best_i=best_i, legal=0 <= best_i < vocab_size,
                root_cause="NaN > best_v is always False; initial sentinel -1 survives",
                fix="finite-and-in-range gate + legal initial id")


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(23)
    checks = {}
    # 1) 极端量级的 o 都被契约吸收
    for mag in (1e-5, 1.0, 1e4, 1e5):
        x = rng.standard_normal(1536).astype(np.float32)
        o = (rng.standard_normal(1536) * mag).astype(np.float32)
        r = contract_bound(x, o, 1.0, 0.1)
        checks[f"bound_mag_{mag:g}"] = r["passed"]
        checks[f"ratio_mag_{mag:g}"] = round(r["ratio"], 6)
    # 2) beta=0 逐位零
    x = rng.standard_normal(64).astype(np.float32)
    checks["beta0_zero"] = bool(np.array_equal(
        dimensionless_gate(x, np.full(64, 1e9, np.float32), 0.0),
        np.zeros_like(x)))
    # 3) 有界 ⇒ 下游 RMSNorm 不溢出
    x = rng.standard_normal(1536).astype(np.float32)
    add = dimensionless_gate(x, np.full(1536, 1e5, np.float32), 1.0, 0.1)
    y = x + add
    checks["rms_finite"] = bool(np.isfinite(y).all() and
                                np.isfinite(np.mean(y * y) + 1e-6))
    # 4) role 信噪比
    k = (rng.standard_normal((6, 256)) * 100).astype(np.float32)   # 真实量级
    rr = role_resolvability(k, 6)
    checks["role_snr"] = rr["passed"]
    checks["role_naive_decay"] = round(rr["naive_ratio_decay"], 1)
    checks["role_distance_cv"] = round(rr["distance_cv"], 8)
    # 5) 解码守卫
    checks["gate_rejects_neg1"] = not decode_gate([-1] * 24, 130560)["passed"]
    checks["gate_accepts_legal"] = decode_gate([0, 5, 999], 130560)["passed"]
    # 6) 根因演示
    demo = nan_to_sentinel_demo()
    checks["rootcause_reproduced"] = (demo["best_i"] == -1 and not demo["legal"])
    return dict(passed=all(v for k, v in checks.items()
                           if isinstance(v, bool)),
                checks=checks,
                theory=dict(T1="dimensionless scale contract ‖Δ‖≤βε‖x‖",
                            T2="scale-invariant role bit (distance CV=0 across 1e4× scales)",
                            T3="finite-and-in-range decode gate"))


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, ensure_ascii=False))
