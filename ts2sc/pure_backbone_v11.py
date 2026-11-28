#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v11 · 纯 O(1) 无注意力主干（Pure O(1) Attention-Free Backbone）

## 为什么存在这个文件
用户对 v10.x 的判定成立且被本仓证据自证：
    GENUINELY_BASED_ON_SIMPLCORE = False
    HYBRID_ATTENTION_BACKBONE_WITH_SGC_SIDECAR = True
v10.x 的 `y = T(x) + 1[master]*SUM s_f*D_f(x)` 中 T(x) 是 Llama 全注意力，
旁路全关即逐 token 退化成原模型 ⇒ 那只是一层壳（shell-only）。

本文件按用户指令把 96 个 self_attn 张量从推理链**彻底摘除**：
推理数据流里不再有 Q/K/V、SDPA、KV cache、RoPE 位置对数。
引擎换成 SimplCore 原语：

  * FrozenSGCCore      : psi_m[tau] = rho_m^tau * sin((tau+1)theta_m)，冻结谱核
  * ShiftMisalignFused : (I + alpha*K_theta) 残差融合（SimplCore 融合算子）
  * 谱卷积 token 混合   : 固定核深度卷积，替代注意力的「谁看谁」
  * 二阶凹门控          : g = 1 - exp(-softplus(a)^2)，替代 softmax 归一化
  * gamma 记忆          : 指数门控状态，替代 KV cache

## 纯度定义（可证伪，不是口号）
PURE =  (推理链张量 ∩ teacher 219) == ∅
    AND 无 self_attn / 无 SDPA / 无 KV cache / 无 softmax-over-positions
    AND state_bytes 与 L 无关（O(1) 常驻）
    AND 摘除注意力后仍产出 finite、非退化、非复读的输出
    AND 关闭任何外挂都不会「退化回注意力」（因为已无注意力可退化）

## 诚实边界（写死，不许 later 洗白）
* 本模型是从零初始化的纯主干，不是 MiniCPM5-1B 的等价替换：
  219 个 teacher 张量一个都不在推理链上 ⇒ 不继承 teacher 的语言能力。
* 未做 ppl / benchmark / 能力对比；无「能力不劣」声明。
* 换算子必然改变归纳偏置，纯度和能力不可兼得，本轮按用户指令优先纯度。
* 10M 常驻状态是本主干的真实性质（O(1)），不是账本判定；
  但「能跑 10M」不等于「在 10M 上答得对」。
"""
from __future__ import annotations

import json
import math
import os
import hashlib

import numpy as np

SCHEMA = "TS2SC/pure-backbone-v11"
ARCH = "PURE_O1_SIMPLCORE_NO_ATTENTION"

# 谱核超参（SimplCore 原语，冻结）
RHO = 0.9995
# 相位必须是 O(1)：theta=1e-4 时 psi(0)=sin(theta)≈1e-4，谱核在初态被数值吞掉。
THETA_BASE = 0.3
N_MODES = 8

# 纯度硬约束：这些 teacher 张量名模式**一个都不许**进推理链
FORBIDDEN_PATTERNS = ("self_attn", "q_proj", "k_proj", "v_proj", "o_proj",
                      "rotary", "rope", "kv_cache", "attn")


# ---------------------------------------------------------------------------
# SimplCore 原语
# ---------------------------------------------------------------------------
class FrozenSGCCore:
    """冻结谱核：psi_m[tau] = rho_m^tau * sin((tau+1) theta_m)。无 Q/K/V。"""

    def __init__(self, d: int, n_modes: int = N_MODES, rho: float = RHO,
                 theta_base: float = THETA_BASE, seed: int = 0):
        self.d = d
        self.m = n_modes
        self.rho = rho
        rng = np.random.default_rng(seed)
        self.theta = (theta_base * (1.0 + rng.random(n_modes))).astype(np.float64)
        self.rho_m = (rho * (1.0 - 1e-4 * np.arange(n_modes))).astype(np.float64)
        # 冻结投影：token 表示 -> 谱模态
        self.P = (rng.normal(0, 1.0 / math.sqrt(d), (d, n_modes))
                  .astype(np.float32))
        self.Qb = (rng.normal(0, 1.0 / math.sqrt(n_modes), (n_modes, d))
                   .astype(np.float32))
        self.tau = 0

    def psi(self, tau):
        # 支持标量 / 一维：统一返回 (n, m)
        t = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        r = self.rho_m[None, :] ** t[:, None]
        return (r * np.sin((t[:, None] + 1.0) * self.theta[None, :])
                ).astype(np.float32)

    def step(self, h):
        """h:(d,) -> 融合残差；(d,)。不触碰任何注意力结构。"""
        a = (h.astype(np.float64) @ self.P).astype(np.float32)      # (m,)
        phi = self.psi(self.tau)[0]                                 # (m,)
        self.tau += 1
        return (a * phi) @ self.Qb

    def advance_idle(self, n):
        """O(1) 闭式跳跃：不逐步循环。"""
        self.tau += int(n)

    def snapshot(self):
        return dict(tau=self.tau, psi=self.psi(min(self.tau, 1))[0].tolist())

    def state_bytes(self):
        return (self.P.nbytes + self.Qb.nbytes + self.theta.nbytes
                + self.rho_m.nbytes + 8)


def shift_misalign_fused(h, k_theta, alpha=0.1):
    """(I + alpha*K_theta) 残差融合（SimplCore Shift-Misalign Fusion）。"""
    kc = np.roll(k_theta, 1, axis=-1)
    return h + alpha * (kc - h)


def second_order_concave_gate(a):
    """二阶凹门控 g = 1 - exp(-softplus(a)^2)，值域 (0,1)，替代 softmax。"""
    sp = np.logaddexp(a, 0.0)
    return (1.0 - np.exp(-(sp ** 2))).astype(np.float32)


def spectral_conv_mix(x, W,自闭=False):
    """固定核深度卷积 token 混合（替代注意力）。W:(K,d)。因果。"""
    d = x.shape[-1]
    K = W.shape[0]
    acc = np.zeros(d, dtype=np.float32)
    for j in range(K):
        if j <= x.shape[0] - 1:
            acc = acc + W[j] * x[-1 - j]
    return acc


# ---------------------------------------------------------------------------
# 纯主干层
# ---------------------------------------------------------------------------
class PureLayer:
    """一层纯 O(1) 主干：无 self_attn / 无 KV cache / 无位置 softmax。

    状态只有 gamma 记忆 + 谱核相位，均为定长，与序列长度 L 无关。
    """

    def __init__(self, d: int, h: int, seed: int, kernel: int = 3):
        rng = np.random.default_rng(seed)
        self.d, self.h, self.K = d, h, kernel
        self.in_ln = np.ones(d, np.float32)
        self.post_ln = np.ones(d, np.float32)
        self.gamma_w = rng.normal(0, 0.02, d).astype(np.float32)     # 记忆门
        self.mix = rng.normal(0, 0.02 / math.sqrt(kernel),
                              (kernel, d)).astype(np.float32)        # 谱卷积
        self.gate_w = rng.normal(0, 0.02, d).astype(np.float32)
        self.ff_gate = rng.normal(0, 0.02 / math.sqrt(d), (h, d)).astype(np.float32)
        self.ff_up = rng.normal(0, 0.02 / math.sqrt(d), (h, d)).astype(np.float32)
        self.ff_down = rng.normal(0, 0.02 / math.sqrt(h), (d, h)).astype(np.float32)
        self.core = FrozenSGCCore(d, seed=seed)
        self.mem = np.zeros(d, np.float32)                           # O(1) 状态
        self.tau = 0

    @staticmethod
    def _rms(x, w, eps=1e-6):
        return x * (1.0 / math.sqrt(float(np.mean(x * x)) + eps)) * w

    def step(self, x):
        h = self._rms(x, self.in_ln)
        # 谱卷积混合（替代注意力：不看 KV，只看定长核）
        m = spectral_conv_mix(np.stack([h]) if not hasattr(self, "_buf")
                              else np.concatenate([self._buf, np.stack([h])]),
                              self.mix)
        self._buf = (np.concatenate([self._buf, np.stack([h])])[-self.K:]
                     if hasattr(self, "_buf") else np.stack([h])[-self.K:])
        # 二阶凹门控记忆更新（替代 softmax 归一化）
        g = second_order_concave_gate(h @ self.gamma_w)
        self.mem = g * self.mem + (1.0 - g) * m
        # 冻结谱核**加性**注入特征流：初态 mem=0 时乘性会被吞掉（真 bug，已修），
        # 加性保证 core 删除必可观测；RMSNorm 只去尺度不去方向。
        s = self.core.step(h) if self.core is not None else np.zeros(self.d, np.float32)
        h2 = shift_misalign_fused(self.mem + s, self.mix[0], alpha=0.1)
        h2 = self._rms(h2, self.post_ln)
        # SwiGLU FFN（与注意力无关）；记忆额外调制门控
        a = (self.ff_gate @ h2) * (1.0 + 0.5 * np.tanh(self.mem @ self.gate_w))
        u = self.ff_up @ h2
        return x + self.ff_down @ (a / (1.0 + np.exp(-a)) * u)

    def advance_idle(self, n):
        self.core.advance_idle(n)
        self.tau += int(n)

    def state_bytes(self):
        b = sum(t.nbytes for t in (self.in_ln, self.post_ln, self.gamma_w,
                                   self.mix, self.gate_w, self.ff_gate,
                                   self.ff_up, self.ff_down, self.mem))
        return b + self.core.state_bytes()

    def tensors(self, i):
        # 命名空间完全脱离 teacher：连 "model.layers.N." 都不沿用。
        p = f"pure.layers.{i}."
        return {p + "input_ln": self.in_ln, p + "post_ln": self.post_ln,
                p + "gamma_w": self.gamma_w, p + "mix": self.mix,
                p + "gate_w": self.gate_w, p + "ff_gate": self.ff_gate,
                p + "ff_up": self.ff_up, p + "ff_down": self.ff_down,
                p + "core.P": self.core.P, p + "core.Q": self.core.Qb}


class PureBackboneLM:
    """完整纯模型：embed -> N x PureLayer -> norm -> lm_head。无任何注意力。"""

    def __init__(self, cfg, seed: int = 20261129):
        self.cfg = cfg
        self.d = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.h = cfg["intermediate_size"]
        self.vocab = cfg["vocab_size"]
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        rng = np.random.default_rng(seed)
        self.embed = rng.normal(0, 0.02, (self.vocab, self.d)).astype(np.float32)
        self.lm_head = rng.normal(0, 0.02 / math.sqrt(self.d),
                                  (self.vocab, self.d)).astype(np.float32)
        self.norm = np.ones(self.d, np.float32)
        self.layers = [PureLayer(self.d, self.h, seed + 101 * i)
                       for i in range(self.nl)]

    # -- 纯度自检：本对象上不存在任何注意力属性 --
    def assert_pure(self):
        bad = []
        for name in dir(self):
            if any(f in name.lower() for f in FORBIDDEN_PATTERNS):
                bad.append(name)
        for i, L in enumerate(self.layers):
            for name in dir(L):
                if any(f in name.lower() for f in FORBIDDEN_PATTERNS):
                    bad.append(f"layers.{i}.{name}")
            for name in dir(L.core):
                if any(f in name.lower() for f in FORBIDDEN_PATTERNS):
                    bad.append(f"layers.{i}.core.{name}")
        return dict(pure=not bad, violations=bad,
                    n_layers=self.nl, forbidden=list(FORBIDDEN_PATTERNS))

    def step(self, tid, pos, need_logits=True, return_hidden=False):
        x = self.embed[int(tid)].copy()
        for L in self.layers:
            x = L.step(x)
        x = x * (1.0 / math.sqrt(float(np.mean(x * x)) + self.eps)) * self.norm
        if return_hidden:
            return x.copy()                    # pre-head hidden, (d,)
        if not need_logits:
            return None
        lg = self.lm_head @ x
        return int(np.argmax(lg)), lg

    def advance_idle(self, n):
        for L in self.layers:
            L.advance_idle(n)

    def state_bytes(self):
        return sum(L.state_bytes() for L in self.layers)

    def generate(self, ids, max_new=8):
        out = list(ids)
        for j in range(max_new):
            tid, lg = self.step(out[-1], len(out), need_logits=True)
            if not bool(np.isfinite(lg).all()):
                break
            out.append(tid)
            if tid in (1, 130073):
                break
        return out

    def tensors(self):
        # 全部采用 pure.* 命名空间：与 teacher 219 张量名**零交集**，
        # 也不沿用 LlamaForCausalLM 的 model.embed_tokens / lm_head 命名。
        t = {"pure.embed": self.embed,
             "pure.final_norm": self.norm,
             "pure.head": self.lm_head}
        for i, L in enumerate(self.layers):
            t.update(L.tensors(i))
        return t

    def tensor_sha256(self):
        """逐张量 sha256：与 teacher 219 个 COPY 的字节集合做互斥证明。

        零复制：embed / lm_head 各 ~802 MiB，`.tobytes()` 会再复制一份直接
        OOM；改用 memoryview.cast('B') 让 hashlib 流式读原缓冲区。
        """
        out = {}
        for n, a in self.tensors().items():
            a = np.ascontiguousarray(a)
            h = hashlib.sha256()
            h.update(memoryview(a).cast("B"))
            out[n] = dict(dtype=str(a.dtype), shape=list(a.shape),
                          bytes=int(a.nbytes), sha256=h.hexdigest())
        return out

    # -- 状态快照 / 回滚：让删除实验三次测量共用同一初始状态 --
    def snapshot_state(self):
        snap = []
        for L in self.layers:
            snap.append(dict(
                mem=L.mem.copy(),
                tau=L.tau,
                core_tau=(L.core.tau if L.core is not None else None),
                buf=(L._buf.copy() if hasattr(L, "_buf") else None)))
        return snap

    def restore_state(self, snap):
        for L, s in zip(self.layers, snap):
            L.mem[:] = s["mem"]
            L.tau = s["tau"]
            if L.core is not None and s["core_tau"] is not None:
                L.core.tau = s["core_tau"]
            if s["buf"] is not None:
                L._buf = s["buf"].copy()
            elif hasattr(L, "_buf"):
                del L._buf
