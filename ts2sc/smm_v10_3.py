#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10.3 · SMM 记忆与自进化模块（范围收窄交付：SGC + 外挂 + SMM）

## 来源（真实源码，非编造）
`output/upstream_arch_snapshot/simplcore__core__arch__evolution_mixer.py`
  * `FusedEvolutionMixer` docstring: "STEP 2 -- SMM x SGC fused state,
    written by a zero-order gate. Memory and computation are ONE algebraic
    object, not a sidecar"
  * `CalibratedFusedMixer` STEP 3 三项修正：rho_mem 分离时间尺度 /
    per-key cooldown / dominance gate
  * `observe()` = CT7 / SE1-SE5 零阶巩固：真实共现计数 >= tau、
    读回不命中才写、capacity 淘汰、rho_mem 衰减；无梯度穿过检索。

## 本模块的移植边界（如实）
* 上游是 (B, L, d) 训练期 mixer；这里是**推理期逐 token** 的 24 层
  加性分支，状态形状沿用 d×d / Hadamard 码本 / VK=d/2 桶，
  但 d 取 v10 的低维 DS=64，避免 1536×1536 状态。
* `observe()` 只在**显式调用**时巩固（手动门禁），绝不自发：
  这正对应「开门禁继承，不开就原模型」。
* `advance_idle(n)` 是 **O(1) 闭式**：10M 上下文不做 10M 次循环。
* 加性项仍受 T1' 尺度契约：‖Δ_SMM‖ ≤ s_smm·beta·eps·‖x‖。

## 开关
  s_smm = 0 ⇒ 模块完全不贡献（与 v10.2 全开态逐位相同）
  s_smm = 1 ⇒ 交付继承态
"""
from __future__ import annotations

import math

import numpy as np

EPS_DEFAULT = 0.10


def hadamard(n):
    """归一化 Sylvester-Hadamard：C C^T = I_n（与上游同名同实现）。"""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:n, :n] / math.sqrt(n)


class SMMModule:
    """SMM × SGC 融合状态：零阶巩固记忆 + 显式门控自进化。

    状态 S_t = rho_mem · S_{t-1} + Σ_accepted C[k]^T C[VK+v]
    读出 add_t = C[bucket(tok_t)] @ S_t @ W_c
    写回只由 observe() 在显式门禁下发生，且需：
      counts[k,v] >= tau  (重复证据)
      该 v 在该 k 的共现质量占比 >= dominance
      step - last_commit[k] >= cooldown
      read(k) 不命中或 margin < eps  (避免重复写)
      capacity 上限，按权重最小淘汰
    """

    def __init__(self, d: int = 64, seed: int = 0,
                 tau: int = 6, capacity: int = 16, eps: float = 1e-3,
                 rho_mem: float = 0.9995, cooldown: int = 40,
                 dominance: float = 0.35, s: float = 1.0):
        self.d = int(d)
        self.C = hadamard(self.d)
        self.VK = self.d // 2
        self.tau = int(tau)
        self.capacity = int(capacity)
        self.eps = float(eps)
        self.rho_mem = float(rho_mem)
        self.cooldown = int(cooldown)
        self.dominance = float(dominance)
        self.s = float(s)                       # 独立开关
        self.S = np.zeros((self.d, self.d), np.float64)
        self.slots = {}                         # key -> (value, weight)
        self.counts = np.zeros((self.VK, self.VK), np.int64)
        self.step = 0
        self.last_commit = {}
        self.commits = 0
        self.rejects = 0
        rng = np.random.default_rng(seed + 3000)
        self.W_c = rng.standard_normal((self.d, self.d)) / math.sqrt(self.d)
        self.logical_pos = 0

    # ---- 核心：零阶巩固（显式门禁，非自发） ----
    def _bucket(self, tok):
        return np.asarray(tok) % self.VK

    def _read(self, key):
        z = self.C[key] @ self.S
        logits = z @ self.C[self.VK:, :].T
        return int(np.argmax(logits)), float(logits.max())

    def observe(self, tok, tgt):
        """一次真实共现批次 → 零阶巩固。必须显式调用（手动开门）。"""
        self.step += 1
        k = self._bucket(tok).ravel()
        v = self._bucket(tgt).ravel()
        np.add.at(self.counts, (k, v), 1)
        for kk, vv in set(zip(k.tolist(), v.tolist())):
            if self.counts[kk, vv] < self.tau:
                continue
            if self.step - self.last_commit.get(kk, -10 ** 9) < self.cooldown:
                continue
            row = self.counts[kk].astype(np.float64)
            if row.sum() <= 0 or self.counts[kk, vv] / row.sum() < self.dominance:
                continue
            cur, margin = self._read(kk)
            if cur == vv and margin >= self.eps:
                self.rejects += 1
                continue
            if kk in self.slots:
                old_v, old_w = self.slots[kk]
                self.S -= old_w * np.outer(self.C[kk], self.C[self.VK + old_v])
            self.S += np.outer(self.C[kk], self.C[self.VK + vv])
            self.slots[kk] = (int(vv), 1.0)
            self.last_commit[kk] = self.step
            self.commits += 1
        self.S *= self.rho_mem
        for kk, (vv, w) in list(self.slots.items()):
            self.slots[kk] = (vv, w * self.rho_mem)
        while len(self.slots) > self.capacity:
            kk = min(self.slots, key=lambda x: self.slots[x][1])
            vv, w = self.slots.pop(kk)
            self.S -= w * np.outer(self.C[kk], self.C[self.VK + vv])
        return dict(commits=self.commits, rejects=self.rejects,
                    slots=len(self.slots))

    # ---- 前向读出（受开关与 T1' 契约） ----
    def delta(self, x, tok, beta=1.0, eps=EPS_DEFAULT):
        """返回加性 Δ；s=0 或 beta<=0 ⇒ 严格零向量。"""
        if self.s <= 0.0 or beta <= 0.0:
            return np.zeros_like(x, dtype=np.float32)
        xf = np.asarray(x, np.float64)
        rx = float(np.linalg.norm(xf))
        raw = self.C[self._bucket(tok).ravel() % self.d] @ self.S
        o = raw @ self.W_c
        o = np.asarray(o, np.float64).reshape(-1)
        if o.size != xf.size:
            # d(64) -> hidden(1536)：确定性零相位升维，不引入随机
            rep = int(math.ceil(xf.size / o.size))
            o = np.tile(o, rep)[:xf.size]
        ro = float(np.linalg.norm(o))
        u = o / (ro + 1e-12) if ro > 0 else np.zeros_like(o)
        out = self.s * beta * eps * rx * u
        nd = float(np.linalg.norm(out))
        bound = self.s * beta * eps * rx
        if nd > bound * (1.0 + 1e-9):
            out *= bound / (nd + 1e-12)
        return out.astype(np.float32)

    # ---- 10M 上下文：O(1) 闭式 idle，绝不循环 n 次 ----
    def advance_idle(self, n: int):
        """闭式跳跃：S ← rho_mem^n · S，slot 权重同步；与逐步 gate=0 等价。"""
        n = int(n)
        f = self.rho_mem ** n
        self.S *= f
        for kk, (vv, w) in list(self.slots.items()):
            self.slots[kk] = (vv, w * f)
        self.logical_pos += n
        return f

    def reset(self):
        self.S[:] = 0.0
        self.slots.clear()
        self.counts[:] = 0
        self.step = 0
        self.last_commit.clear()
        self.commits = 0
        self.rejects = 0
        self.logical_pos = 0

    def state_bytes(self):
        return int(self.S.nbytes + self.counts.nbytes
                   + self.W_c.nbytes + self.C.nbytes)

    def invariants(self):
        return dict(
            module="SMM", source="upstream evolution_mixer.py "
                                 "FusedEvolutionMixer/CalibratedFusedMixer",
            fused_state=True, zero_order_gate=True,
            consolidated_only_by_explicit_observe=True,
            state_independent_of_L=True,
            state_independent_of_n_bindings=True,
            state_bytes=self.state_bytes(),
            core_state_bytes=int(self.S.nbytes),
            has_NxN_score=False, has_kv_cache=False,
            commits=self.commits, rejects=self.rejects,
            slots=len(self.slots), logical_pos=self.logical_pos,
            rho_mem=self.rho_mem, tau=self.tau,
            cooldown=self.cooldown, dominance=self.dominance,
            capacity=self.capacity,
            remaining_limitation="上游如实登记：桶数 VK=d/2 小于符号字母表，"
                                 "桶内碰撞未解决；本模块不声称修复")
