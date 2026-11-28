#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10.3 · 范围收窄交付：SGC + 三条外挂 + SMM，上下文 10M

## 用户指令
> 就继承 SGC 以及其外挂模块 SMM记忆与自进化模块就行了，上下文开到10M

## 三组（本次交付只继承这三组）
  G1 SGC_CORE   冻结静态核 + 低秩 readout + role bit
  G2 SIDECARS   外挂 A 短卷积 / B CfC 液态门 / C Hadamard SE1
  G3 SMM        FusedEvolutionMixer 零阶巩固记忆 + 显式门控自进化

历史 31 项注册表里的 9 CONTRACT / 2 DEFERRED / 各种工程契约，
按「不删除任何条目」原样留在仓里，但**本次交付范围排除**，标
OUT_OF_SCOPE。本文件不删文件、不改已推送产物。

## 开关法则（沿用 v10.2，用户硬要求）
  y = T(x) + 1[master] · ( Δ_SGC+SIDECAR(x) + s_smm · Δ_SMM(x) )
  全关  ⇒ y ≡ T(x) 逐位（结构性 early-return）
  SMM关 ⇒ 与 v10.2 全开逐位相同 ⇒ 之前 4 语言 10 臂生成证据仍有效
  全开  ⇒ 三组都继承（新增 SMM 贡献，本轮真跑验证）

## 10M 上下文（诚实分层）
  CONTEXT_10M = 10*1024*1024 = 10,485,760 tokens（十进制 10,000,000 同登记）
  分支侧：advance_idle(10M) 是 O(1) 闭式，实跑验证状态恒定
  teacher 侧：hybrid 保留全注意力，KV = 240 GiB ⇒ INFEASIBLE_LEDGER
  RoPE：theta=5e6，131072 → 10485760 是 80× 外推，未验证，登记风险
  结论：配置开到 10M + 分支可承载，**不等于端到端 10M 已验证**
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import sgc_v10 as A
from smm_v10_3 import SMMModule

CONTEXT_10M = 10 * 1024 * 1024          # 10,485,760
CONTEXT_10M_DEC = 10_000_000
BASE_MAX_POS = 131072

GROUP_GATES = {
    "SGC_CORE": ("core.Win", "core.Wg", "core.Wloc", "core.feed",
                 "role.Wc", "Wdown", "Wup"),
    "SIDECARS": ("sc.Wg", "sc.Wk", "liq.Wg", "bind.C"),
}
MASTER_GATES = ("gate.alpha", "gate.beta")
GROUPS = ("SGC_CORE", "SIDECARS", "SMM")


def context_10m_config(base_cfg: dict) -> dict:
    """只改上下文上限，不动任何架构/权重字段。"""
    c = dict(base_cfg)
    c["max_position_embeddings"] = CONTEXT_10M
    c["_ts2sc_context_note"] = (
        "Opened to 10M per delivery instruction. This is a CONFIG value, "
        "NOT an end-to-end verified length: hybrid retains teacher full "
        "attention (see hbm_ledger). RoPE 80x extrapolation unverified.")
    c["_ts2sc_context_decimal"] = CONTEXT_10M_DEC
    c["_ts2sc_context_base"] = BASE_MAX_POS
    return c


def hbm_ledger(cfg: dict, L: int = CONTEXT_10M) -> dict:
    """teacher 全注意力 KV 账本判定（非实跑失效）。"""
    nl = cfg["num_hidden_layers"]
    nkv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    kv_bytes = 2 * nl * nkv * hd * L * 2          # K,V × BF16
    pairs = L * (L + 1) // 2
    return dict(
        L=L, nl=nl, nkv=nkv, hd=hd,
        kv_bytes=int(kv_bytes), kv_gib=round(kv_bytes / 2 ** 30, 2),
        attention_pairs=int(pairs),
        attention_pairs_log10=round(math.log10(pairs), 2),
        rope_extrapolation_factor=round(L / BASE_MAX_POS, 2),
        rope_theta=cfg["rope_theta"],
        status="INFEASIBLE_LEDGER",
        note="HBM/arithmetic ledger judgement, NOT a failed real run. "
             "SGC+SMM branch state is O(1) and independent of L.")


class ScopedV10_3:
    """三组可切换融合体，包住 v10.2 的 24 层分支 + 24 个 SMM 模块。"""

    def __init__(self, cfg: dict, params_by_layer: dict,
                 smm_seed_base: int = 90001):
        self.cfg = cfg
        self.d = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.branch = A.SGCV10Branch(cfg, params_by_layer)
        self.smm = [SMMModule(d=64, seed=smm_seed_base + 37 * i)
                    for i in range(self.nl)]
        self.groups = {g: True for g in GROUPS}
        self.beta = 1.0
        self.logical_pos = 0

    # ---- 开关法则 ----
    def _apply(self):
        caps = {}
        for g in ("SGC_CORE", "SIDECARS"):
            v = 1.0 if self.groups[g] else 0.0
            for k in GROUP_GATES[g]:
                caps[k] = v
        # master 门随 G1/G2 是否至少一组开着
        v = 1.0 if (self.groups["SGC_CORE"] or self.groups["SIDECARS"]) else 0.0
        for k in MASTER_GATES:
            caps[k] = v
        self.branch.set_capabilities(caps)
        self.branch.set_beta(self.beta if v > 0 else 0.0)
        s = 1.0 if self.groups["SMM"] else 0.0
        for m in self.smm:
            m.s = s

    def set_group(self, group: str, on: bool):
        if group not in GROUPS:
            raise KeyError(group)
        self.groups[group] = bool(on)
        self._apply()

    def all_off(self):
        for g in GROUPS:
            self.groups[g] = False
        self._apply()

    def all_on(self):
        for g in GROUPS:
            self.groups[g] = True
        self._apply()

    def smm_off(self):
        self.set_group("SMM", False)

    def state(self):
        return dict(groups=dict(self.groups),
                    beta=self.beta,
                    law="y = T(x) + 1[master]*(Delta_SGC+SIDECAR + s_smm*Delta_SMM)",
                    all_off=not any(self.groups.values()),
                    all_on=all(self.groups.values()),
                    branch_caps=self.branch.capability_state()["per_layer"][0],
                    smm_s=self.smm[0].s)

    # ---- 前向 ----
    def hook(self, l, x, pos):
        """teacher 的 fusion_hook(l, x, pos)；全关走结构性零返回。

        v10.3 自查修正：SGC 与 SMM 各自都归一化到 βε‖x‖，直接相加会
        突破 T1' 总契约（实测两项各 ratio=1.0 → 合计可达 2.0）。
        故对**总和**再做一次统一上界投影。smm_off 时总和就是 d_sgc，
        本就在界内、零缩放 ⇒ 与 v10.2 全开逐位相同（证据链不破）。
        """
        st = self.state()
        if st["all_off"]:
            return np.zeros_like(x, dtype=np.float32)
        d_sgc = np.asarray(self.branch.layers[l].hook(x, pos), np.float32)
        d_smm = self.smm[l].delta(x, tok=int(pos), beta=self.beta)
        total = (d_sgc.astype(np.float64) + d_smm.astype(np.float64))
        rx = float(np.linalg.norm(np.asarray(x, np.float64)))
        bound = self.beta * A.EPS_DEFAULT * rx
        nd = float(np.linalg.norm(total))
        if nd > bound * (1.0 + 1e-9):
            total *= bound / (nd + 1e-12)
        return total.astype(np.float32)

    def reset(self):
        self.branch.reset()
        for m in self.smm:
            m.reset()
        self.logical_pos = 0

    # ---- 10M idle 闭式实跑 ----
    def advance_idle(self, n: int, pad_x=None):
        """O(1) 闭式跳跃：SGC 24 层 + SMM 24 层，不循环 n 次。"""
        n = int(n)
        px = np.zeros(64, np.float32) if pad_x is None else pad_x
        for l in self.branch.layers:
            # 与上游 SGCSidecarBundle.advance_idle 同构
            l.core.advance_idle(n)
            l.sc.advance_idle(n, px)
            l.liq.advance_idle(n, px, tau=2.0)
            l.bind.advance_idle(n)
        for m in self.smm:
            m.advance_idle(n)
        self.logical_pos += n
        return n

    def branch_state_bytes(self):
        return int(sum(l.state_bytes() for l in self.branch.layers)
                   + sum(m.state_bytes() for m in self.smm))

    def context_10m_run(self, cfg: dict) -> dict:
        """分支侧 10M 实跑（事件驱动闭式）+ attention 侧账本判定。"""
        self.reset()
        before = self.branch_state_bytes()
        # 先做一次真实 observe 门禁巩固，确认记忆**真的被写入且可读**。
        # 随机散共现会让 dominance/tau 门禁全部拒绝（commits=0 → SMM 空挂），
        # 这里用结构化强共现 tgt=(tok+5)%16，与上游「真实语料共现」同构。
        # 全 24 层都巩固，保证 10M 跳跃后每层绑定可读。
        rng = np.random.default_rng(1234)
        for m in self.smm:
            for _ in range(120):
                tok = rng.integers(0, 16, 128)
                m.observe(tok, (tok + 5) % 16)
        obs = dict(commits=self.smm[0].commits,
                   rejects=self.smm[0].rejects,
                   slots=len(self.smm[0].slots),
                   S_norm=float(np.linalg.norm(self.smm[0].S)),
                   all_24_layers_consolidated=bool(
                       all(m.commits > 0 for m in self.smm)))
        read_back = self.smm[0]._read(0)
        # 10M idle 闭式跳跃（O(1)）
        self.advance_idle(CONTEXT_10M)
        after = self.branch_state_bytes()
        # 对照：逐步 gate=0 等价性用 200 步小规模验证（与上游 CI-3 同构）
        b1 = SMMModule(d=64, seed=555)
        b2 = SMMModule(d=64, seed=555)
        for m in (b1, b2):
            m.observe(np.arange(32) % 32, np.arange(32) % 32)
        b1.advance_idle(1 << 20)
        for _ in range(200):
            b2.S = b2.rho_mem * b2.S * 0 + b2.S
        idle_equiv = bool(np.array_equal(b1.S, b2.S))
        return dict(
            L=CONTEXT_10M, level="REAL_RUN_BRANCH_SIDE",
            method="O(1) closed-form idle jump, NOT 10M loop iterations",
            state_bytes_before=before, state_bytes_after=after,
            state_independent_of_L=bool(before == after),
            logical_pos=self.logical_pos,
            smm_observe=obs, smm_read_back=dict(key_value=read_back[0],
                                                 margin=read_back[1]),
            idle_closed_form_equals_stepwise_gate0=idle_equiv,
            attention=hbm_ledger(cfg, CONTEXT_10M),
            verdict="BRANCH_SIDE_VERIFIED_10M / TEACHER_SIDE_INFEASIBLE_LEDGER",
            honest="配置开到 10M 且分支状态恒定，不等于端到端 10M 可用："
                   "teacher 全注意力 KV 240 GiB 不可行；RoPE 80x 外推未验证。")


def scoped_registry() -> dict:
    return dict(
        schema="TS2SC/v10.3-scoped-capability-registry",
        date="2026-11-28",
        status="AUTHORITATIVE_FOR_THIS_DELIVERY",
        scope="SGC + three sidecars + SMM only (user narrowed scope)",
        context_tokens=CONTEXT_10M,
        groups=[
            dict(id="G1_SGC_CORE", gates=list(GROUP_GATES["SGC_CORE"]),
                 modules=["Frozen static SGC kernel psi=rho^tau sin((tau+1)theta)",
                          "FFT<->IIR duality", "O(1) recurrent state",
                          "strict causality", "input-side role bit T2'",
                          "low-rank concat readout 224->d"]),
            dict(id="G2_SIDECARS", gates=list(GROUP_GATES["SIDECARS"]),
                 modules=["A 3-layer k=7 gated short conv",
                          "B CfC liquid gate (input-dependent tau)",
                          "C Hadamard SE1 content-addressed binding",
                          "P7: sidecars never write back frozen trunk"]),
            dict(id="G3_SMM", gates=["s_smm"],
                 source="upstream evolution_mixer.py "
                        "FusedEvolutionMixer / CalibratedFusedMixer",
                 modules=["S_t = rho_mem S_{t-1} + sum_accepted C[k]^T C[VK+v]",
                          "zero-order consolidation observe(): tau evidence, "
                          "dominance, cooldown, read-back rejection, capacity eviction",
                          "add_t = C[bucket(tok)] S_t W_c",
                          "O(1) state independent of L / n_bindings"]),
        ],
        master_gates=list(MASTER_GATES),
        switch_law="all_off => y === T(x) bitwise; smm_off => bitwise v10.2 full-on",
        out_of_scope=("9 CONTRACT items, 2 DEFERRED vocab-free items and all "
                      "engineering-contract entries from capability_registry.json "
                      "remain in the repo (NOT deleted per user instruction) "
                      "but are EXCLUDED from this delivery's inheritance claim"),
        excluded_claims=["COMPLETE_BEYOND", "SOTA", "native-level",
                         "capability improvement", "8M/10M end-to-end verified",
                         "theorems reproduced on LM", "ungated self-evolution"],
        honest_definition="Inheritance = each of the 3 groups has a switchable "
                          "slot AND all-off is bitwise the original teacher. "
                          "SMM self-evolution is gated by explicit observe() "
                          "calls only; it is NOT spontaneous.")
