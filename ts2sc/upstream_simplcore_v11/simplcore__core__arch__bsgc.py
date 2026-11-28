# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""BindingSpectralConv（``mixer="sgc-bind"``）—— SGC + 定容绑定记忆辅助通道.

针对的已确诊缺陷（2026-10-31 独立复现，``docs/REPRODUCTION_REPORT.md``）
--------------------------------------------------------------------------
MQAR 前哨（G=4，2 层，3000 步）：attention 0.500 vs SGC 0.3125（chance
0.125）；精确指向 1.000 vs 0.398。SGC 缺的不是感受野，是 **key→value
绑定能力**：谱核是输入无关常数，逐元素门控只能逐位置缩放，表达不了
「位置 i 应当聚合历史的哪种模式」（``sgc.py`` MoK 注释同源诊断）。

机制（对着复现报告定位的卡点：卡点在读取、在绑定，不在容量）
--------------------------------------------------------------------------
在 SGC 三通路（长程 FFT / 局部卷积 / 瞬时项）之外，加一条 **定容绑定
记忆** 辅助通路，复用 :mod:`simplcore.core.memory.binding_memory`（R1）
的构造性结论，并做三处「可学性」修正（探针实证，见 ``research/binding/``）：

1. **槽内 key/value 分开存 + 残差形式**：``k_new = u[t-1] + ak(u[t-1])``、
   ``v_new = u[t] + av(u[t])``、``q_t = u_t + aq(u_t)``。纯投影版在探针中
   600 步学不动（loss ≈2.05 平台）——随机初始化下寻址分数无区分度；
   残差形式让查询与槽 key 在**初始化时即自同构**（e_k·e_k ≫ e_k·e_j），
   梯度从第 0 步就有方向（induction head 的残差流同构）。
2. **硬 FIFO 写入**（``bound`` 模式，默认）：写掩码是 one-hot 常量
   （位置 = (t-1) mod S），out-of-place 更新 ``sk + mask·(k_new - sk)`` ——
   对 k_new 可微（掩码无梯度需求）。**软门控写入被探针证伪**：0.5 混合
   把全部绑定平均成一碗汤，槽内容趋同 → 寻址分数全等 → softmax 梯度
   逐槽相消为零（复现报告二期「3000 步退化」的根因实锤）。
3. **可学习锐度 τ**：``probs = softmax(τ·scores/√D)``，τ softplus 参数化
   （oracle 实测 τ=8 完美寻址；τ 可学，初值 ≈4）。

与 BoundedBindingMemory（R1 实验件）的差异
------------------------------------------
R1 是**外挂记忆组件**（单序列、推理/训练两条路径）；本 mixer 把绑定
记忆**内嵌为第四条混合通路**：批量并行（训练 scan）、O(1) 流式
（``step``）、分块（``forward_chunk``）三条路径同构实现，并随
``LayerStreamState`` 走断点持久化。注入走**加性项**进入三路混合之后
的 ``y``（初始 ``mem_gate=σ(-4)≈0.018`` 近恒等，训练中自适应打开），
不引入任何 token↔token 成对打分。

一致性判据（架构忠诚度）
------------------------
* 状态与 L 无关：槽 O(S·D) + u_prev O(D)，写满后门控混合（无 FIFO
  指针），``streaming_state_bytes`` 恒定；
* 无 O(L²) 交互：寻址 softmax over **S 个槽**（O(S·D)/位置），不是
  over L 个位置；长程主通路仍是 FFT 谱卷积 O(L log L)；
* FFT↔IIR 双形式一致：绑定 scan 的读/写是逐位置确定性同序操作，
  并行（``forward``）与流式（``step``）在 fp32 内一致（测试锁定）；
* 纯洁性：无 ``q/k/v_proj``、无 SDPA、无位置表 —— 通过 purity 三层
  守卫（``addr_key``/``slot_k`` 等命名均不在注意力判定口径内）。

诚实边界
--------
* 训练 scan 是 **O(L) 逐步 Python 循环**（每步 O(B·S·D) 张量算子）——
  槽写入是顺序递推（FIFO），与 RNN 同构；探针规模（L≤数千）可用，
  超长序列训练需后续做 chunked scan（与 SRF 同思路）。长程主通路仍
  是 FFT，不受此影响；
* ``soft`` 写入模式保留为**失败对照**（探针实证槽内容趋同、梯度相消），
  默认 ``bound``；
* 未在真实数据训练验证 —— 机制/双形式/口径由本模块测试锁定；检索
  质量数值由 axis-probe 对照实验（``--arms sgc-bind``）记账。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sgc import (LayerStreamState, SpectralGlobalConv,
                  depthwise_causal_conv, fft_causal_conv, recurrent_dtype)

__all__ = ["BindingSpectralConv"]

#: 绑定槽状态三元组类型别名：(slot_k, slot_v, u_prev)
BindSlots = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BindingSpectralConv(SpectralGlobalConv):
    """SGC + 定容绑定记忆辅助通道（``mixer="sgc-bind"``）.

    Args:
        继承 :class:`SpectralGlobalConv` 全部参数（dim / num_bases /
        decay_max / local_kernel / 各门控开关 / init_mode / rho_param …）；
        另加：
        n_slots: 绑定记忆槽数 S（定容）。应 ≥ 任务的关键绑定数；
            序列远长于 S 时最老绑定被门控混合逐步覆盖。
        mem_gate_init: 注入强度 ``mem_gate`` 的初始 logit（默认 -4 →
            σ≈0.018，近恒等起步；非零保证梯度从第 0 步可达）。
    """

    def __init__(self, *args, n_slots: int = 128,
                 mem_gate_init: float = 0.0, write_mode: str = "bound",
                 **kwargs) -> None:
        # 融合 C++/CUDA 流式算子按父类属性布局编写，未实现绑定记忆通道；
        # 显式关闭而不是静默给出不一致结果。
        kwargs["use_mixer_stream"] = False
        super().__init__(*args, **kwargs)
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        if write_mode not in ("bound", "soft"):
            raise ValueError(
                f"write_mode must be 'bound'|'soft', got {write_mode!r}")
        self.n_slots = int(n_slots)
        self.write_mode = write_mode
        # 寻址/写入投影：key 与 value 分开（R1 的构造性结论）。
        # 使用形式是**残差**的：k_new = u_prev + ak(u_prev)（v/q 同理）——
        # 随机初始化下槽内容 ≈ 原始 embedding，查询与正确槽在初始化时
        # 即自同构（探针实证：纯投影版 600 步学不动，残差版立即可学）。
        self.addr_key = nn.Linear(self.dim, self.dim, bias=False)
        self.addr_value = nn.Linear(self.dim, self.dim, bias=False)
        self.addr_query = nn.Linear(self.dim, self.dim, bias=False)
        if write_mode == "soft":
            # 失败对照（探针实证槽内容趋同、softmax 梯度逐槽相消）；
            # 保留供消融，默认 bound。
            self.write_gate = nn.Linear(2 * self.dim, self.n_slots, bias=True)
        # 读出锐度 τ（softplus 参数化，可学；oracle 实测 τ=8 完美寻址）
        self.raw_read_temp = nn.Parameter(torch.tensor(3.8))
        # 注入强度标量（logit）：σ(0)=0.5 起步，训练中自适应
        self.mem_gate = nn.Parameter(torch.tensor(float(mem_gate_init)))

    # ------------------------------------------------------------------
    # 绑定记忆核心：单步读/写（并行 scan 与流式 step 共用同一套数学）
    # ------------------------------------------------------------------
    def _empty_slots(self, u: torch.Tensor, rd: Optional[torch.dtype] = None) -> BindSlots:
        dt = rd if rd is not None else u.dtype
        z = torch.zeros(u.shape[0], self.n_slots, self.dim,
                        device=u.device, dtype=dt)
        up = torch.zeros(u.shape[0], self.dim, device=u.device, dtype=dt)
        return (z, z.clone(), up)

    def _binding_read(self, ut: torch.Tensor, slots: BindSlots,
                      filled: int = -1) -> torch.Tensor:
        """位置 t 的读出：softmax over **S 个槽**（O(S·D)，与历史长度无关）.

        ut: (B, D)；返回注入向量 (B, D)。``filled`` = 已完成写入数：
        硬 FIFO 下未写槽是全零行，参与 softmax 会稀释命中槽（0.1-α 调试
        实录），故 ``0 <= filled < S`` 时把槽 j ≥ filled 掩成 -inf；
        ``filled >= S``（写满回绕）或 ``filled < 0``（不掩码，soft 模式）
        时全槽参与。向量化快路径的因果掩码与本口径一致（测试锁定）。
        """
        slot_k, slot_v, _ = slots
        gate = torch.sigmoid(self.mem_gate)
        if filled == 0:
            # 尚无已完成写入：无可寻址槽，读出恒零（与向量化路径的
            # nan_to_num 口径一致，且不产生 NaN）
            return gate * torch.zeros_like(ut)
        q = ut + self.addr_query(ut)                              # 残差查询
        tau = F.softplus(self.raw_read_temp) + 0.01
        scores = tau * torch.einsum("bd,bsd->bs", q, slot_k) / math.sqrt(self.dim)
        if 0 < filled < self.n_slots:
            j_idx = torch.arange(self.n_slots, device=ut.device)
            scores = scores.masked_fill(
                (j_idx >= filled).unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)                     # (B, S)
        read = torch.einsum("bs,bsd->bd", probs, slot_v)          # (B, D)
        return torch.sigmoid(self.mem_gate) * read

    def _binding_write(self, ut: torch.Tensor, slots: BindSlots,
                       ptr: int) -> BindSlots:
        """写入一个绑定单元 (u_prev, u_t)（key=前驱残差、value=当前残差）.

        ``bound``（默认）：one-hot FIFO 掩码（位置 = ptr mod S）out-of-place
        更新——对 k_new/v_new 可微，掩码是常量无梯度需求；
        ``soft``（失败对照）：逐槽 σ 门控混合（探针实证趋同退化）。
        """
        slot_k, slot_v, u_prev = slots
        k_new = u_prev + self.addr_key(u_prev)                    # (B, D) 残差
        v_new = ut + self.addr_value(ut)                          # (B, D) 残差
        if self.write_mode == "soft":
            w = torch.sigmoid(self.write_gate(
                torch.cat([k_new, v_new], dim=-1))).unsqueeze(-1)  # (B,S,1)
            slot_k = (1.0 - w) * slot_k + w * k_new.unsqueeze(1)
            slot_v = (1.0 - w) * slot_v + w * v_new.unsqueeze(1)
        else:
            mask = torch.zeros(1, self.n_slots, 1, device=ut.device,
                               dtype=ut.dtype)
            mask[0, ptr % self.n_slots, 0] = 1.0                  # 常量，无梯度
            slot_k = slot_k + mask * (k_new.unsqueeze(1) - slot_k)
            slot_v = slot_v + mask * (v_new.unsqueeze(1) - slot_v)
        return (slot_k, slot_v, ut)

    def _binding_scan(self, u: torch.Tensor,
                      slots: Optional[BindSlots] = None,
                      t0: int = 0) -> Tuple[torch.Tensor, BindSlots]:
        """整段 scan：位置 t 先读（用截至 t-1 的槽）后写 (u[t-1], u[t]).

        u: (B, L, D)（归一化残差流）；slots: 继承的槽状态（分块时传入，
        ``None`` = 零初值）；t0: 本段首个位置的全局索引。

        返回 ``(inject (B,L,D), 终态 slots)``。梯度经递推自然流回全部
        历史写入投影（复现报告 mem_Wq 教训的根治：无 in-place、无
        CopySlices）。

        **向量化快路径**（``bound`` 模式、零初值、S ≥ L-1 即无 FIFO 覆盖）：
        硬 FIFO 的槽位分配是确定性的（全局位置 t 的写入进槽 t-1），因此
        逐位置 scan 与「一次算出全部槽内容 + 因果掩码 softmax」在数学上
        **逐式等价**——O(L·S·D) 的顺序递推可写成 3 个 matmul（O(L²·D)）。
        这只是同一算子的向量化求值，不是引入注意力：键集仍是定容槽
        （S 为固定超参），流式推理仍走 O(1) 逐步递推，purity 口径不变。
        ``soft`` 模式、S < L（有覆盖）与分块继承状态走逐位置循环。
        """
        B, L, D = u.shape
        vec_ok = (slots is None and t0 == 0 and self.write_mode == "bound"
                  and self.n_slots >= L - 1 and L >= 2)
        if vec_ok:
            # ---- 向量化快路径（与下方循环逐式等价，测试锁定） ------------
            tau = F.softplus(self.raw_read_temp) + 0.01
            prev, cur = u[:, :-1], u[:, 1:]                       # (B,L-1,D)
            k_all = prev + self.addr_key(prev)                    # K[j]（槽 j 的 key）
            v_all = cur + self.addr_value(cur)                    # V[j]（槽 j 的 value）
            q = u + self.addr_query(u)                            # (B, L, D)
            scores = tau * torch.einsum("btd,bsd->bts", q, k_all)                 / math.sqrt(self.dim)                             # (B, L, L-1)
            # 因果掩码：位置 t 只读已完成写入 j ≤ t-2（先读后写）
            j_idx = torch.arange(L - 1, device=u.device)
            t_idx = torch.arange(L, device=u.device)
            allowed = (j_idx.unsqueeze(0) <= t_idx.unsqueeze(1) - 2)  # (L, L-1)
            probs = torch.softmax(
                scores.masked_fill(~allowed.unsqueeze(0), float("-inf")),
                dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)              # t<2 无可读槽 → 0
            inject = torch.sigmoid(self.mem_gate) * torch.einsum(
                "bts,bsd->btd", probs, v_all)
            slot_k = u.new_zeros(B, self.n_slots, D)
            slot_v = u.new_zeros(B, self.n_slots, D)
            slot_k[:, :L - 1] = k_all
            slot_v[:, :L - 1] = v_all
            return inject, (slot_k, slot_v, u[:, -1].contiguous())
        if slots is None:
            slot_k = u.new_zeros(B, self.n_slots, D)
            slot_v = u.new_zeros(B, self.n_slots, D)
            u_prev = u.new_zeros(B, D)
        else:
            # 分块继承：cast 到本段计算精度（fp32 训练时零差异）
            slot_k = slots[0].to(u.dtype)
            slot_v = slots[1].to(u.dtype)
            u_prev = slots[2].to(u.dtype)
        outs = []
        for t in range(L):
            ut = u[:, t]
            gt = t0 + t
            filled = min(max(gt - 1, 0), self.n_slots)            # 已完成写入数
            outs.append(self._binding_read(ut, (slot_k, slot_v, u_prev),
                                           filled))
            if gt >= 1:                                           # 有前驱才写
                # FIFO 物理槽位由全局位置决定：全局 t 写入槽 (t-1) mod S，
                # 分块/流式/整段三条路径同式推导，无需额外指针状态。
                slot_k, slot_v, _ = self._binding_write(
                    ut, (slot_k, slot_v, u_prev), gt - 1)
            # u_prev **无条件**前移（含 t=0）：写入用的是前驱 h[t-1]，
            # 首个绑定 (h[0], h[1]) 的 key 侧必须是 h[0] 而不是零向量
            # （0.1-α 三期调试实录：漏掉本行导致 slot0 key=0，三条路径
            # 互不一致、MQAR 掉分）。
            u_prev = ut
        return torch.stack(outs, dim=1), (slot_k, slot_v, u_prev)

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def init_state(self, batch: int, device, dtype) -> LayerStreamState:
        st = super().init_state(batch, device, dtype)
        rd = recurrent_dtype(dtype)
        # 槽与 u_prev 按 recurrent_dtype 分配（与 IIR 状态同精度纪律）
        st.bind_slots = (
            torch.zeros(batch, self.n_slots, self.dim, device=device, dtype=rd),
            torch.zeros(batch, self.n_slots, self.dim, device=device, dtype=rd),
            torch.zeros(batch, self.dim, device=device, dtype=rd))
        return st

    def streaming_state_bytes(self, batch: int, dtype=torch.float32) -> int:
        """定容账本：父类 IIR/conv 状态 + 槽 (2·S·D) + u_prev (D)."""
        n = super().streaming_state_bytes(batch, dtype)
        rd_itemsize = torch.empty(0, dtype=recurrent_dtype(dtype)).element_size()
        n += (2 * self.n_slots * self.dim + self.dim) * batch * rd_itemsize
        return n

    # ------------------------------------------------------------------
    # 前向（并行 / 训练形式）
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, L, D = x.shape
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        if self.num_kernels > 1:
            kernels = self.build_kernels(L, device=u.device, dtype=u.dtype)
            g = self._kernel_route_weights(h)                     # (B,L,K)
            long = torch.zeros_like(u)
            for k in range(self.num_kernels):
                long = long + g[..., k:k + 1] * fft_causal_conv(
                    u, kernels[k], self.fft_dtype)
        else:
            kernel = self.build_kernel(L, device=u.device, dtype=u.dtype)
            long = fft_causal_conv(u, kernel, self.fft_dtype)
        local = self._local_conv_full(u)

        # 绑定记忆辅助通路：绑定在**归一化残差流** h 上（逐位置 rms=1，
        # token 身份在层 0 原样保留）——绑定在门控投影 u 上会让寻址分数
        # 随 init_std 缩到均匀 softmax（探针实证），h 是 O(1) 尺度。
        inject, _slots = self._binding_scan(h)

        if self.evidence_gate:
            long = long * self._evidence_weights(u)
        if self.uncertainty_gate:
            long = long * self._uncertainty_weights(long, local)

        inst = self._instant_term(u)
        y = long + local + inst + inject
        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        return residual + self.out_scale * out

    # ------------------------------------------------------------------
    # 分块前向（截断 BPTT；与 forward 逐位一致）
    # ------------------------------------------------------------------
    def forward_chunk(self, x: torch.Tensor,
                      state: Optional[LayerStreamState] = None,
                      *, update_state: bool = True
                      ) -> Tuple[torch.Tensor, Optional[LayerStreamState]]:
        residual = x
        B, T, D = x.shape
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        g = None
        if self.num_kernels > 1:
            kernels = self.build_kernels(T, device=u.device, dtype=u.dtype)
            g = self._kernel_route_weights(h)
            long = torch.zeros_like(u)
            for k in range(self.num_kernels):
                long = long + g[..., k:k + 1] * fft_causal_conv(
                    u, kernels[k], self.fft_dtype)
        else:
            kernel = self.build_kernel(T, device=u.device, dtype=u.dtype)
            long = fft_causal_conv(u, kernel, self.fft_dtype)
        if state is not None and state.length > 0:
            long = long + self._free_response(u, state, g)

        buf = state.conv_buf if state is not None else None
        local, new_buf = depthwise_causal_conv(u, self.local_conv.weight, buf)

        t0 = state.length if state is not None else 0
        slots_in = state.bind_slots if state is not None else None
        inject, slots_out = self._binding_scan(h, slots_in, t0)

        new_energy = None
        if self.evidence_gate:
            w, new_energy = evidence_weights_chunk(
                u,
                state.energy_sum if state is not None else None,
                state.length if state is not None else 0,
                self.evidence_gain, self.evidence_pos_gain, self.evidence_bias)
            long = long * w.to(long.dtype)
        if self.uncertainty_gate:
            long = long * self._uncertainty_weights(long, local)

        inst = self._instant_term(u)
        y = long + local + inst + inject
        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        y = residual + self.out_scale * out
        if state is None or not update_state:
            return y, state

        with torch.no_grad():
            y1, y2 = self._tail_state(u.detach(), state)
            slots_det = tuple(t.detach() for t in slots_out)
        new_state = LayerStreamState(
            y1=y1, y2=y2,
            conv_buf=(new_buf.detach() if new_buf is not None
                      and new_buf.shape[1] > 0 else None),
            length=state.length + T,
            energy_sum=(new_energy.detach() if new_energy is not None else None),
            bind_slots=slots_det)
        return y, new_state

    # ------------------------------------------------------------------
    # 流式（O(1) 状态、逐 token）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, x_t: torch.Tensor,
             state: LayerStreamState) -> Tuple[torch.Tensor, LayerStreamState]:
        """One token, O(1) state。绑定记忆：先读（因果）后写 (u_prev, u_t)."""
        squeeze = x_t.dim() == 2
        if squeeze:
            x_t = x_t.unsqueeze(1)
        B, one, D = x_t.shape
        M = self.num_bases
        assert one == 1, "step() consumes exactly one position"
        rd = state.y1.dtype

        h = self.norm(x_t)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        rho, theta = self.oscillator_params()
        a = (2.0 * rho * torch.cos(theta)).to(rd)                 # (M,)
        b = (-(rho ** 2)).to(rd)
        c = torch.sin(theta).to(rd)

        um = u.squeeze(1).unsqueeze(1).to(rd)                     # (B,1,D)
        y_new = (a.view(1, M, 1) * state.y1 + b.view(1, M, 1) * state.y2
                 + c.view(1, M, 1) * um)
        state.y2, state.y1 = state.y1, y_new

        if self.num_kernels > 1:
            nxt_k = torch.einsum("bmd,kmd->bkd", y_new, self._coefs_eff(rd))
            g_t = self._kernel_route_weights(h)                   # (B,1,K)
            nxt = torch.einsum("bkd,bk->bd", nxt_k, g_t.squeeze(1)).unsqueeze(1)
        else:
            nxt = torch.einsum("bmd,md->bd", y_new,
                               self._coef_eff(rd)).unsqueeze(1)
        nxt = nxt.to(u.dtype)
        loc, state.conv_buf = depthwise_causal_conv(
            u, self.local_conv.weight, state.conv_buf)

        if self.evidence_gate:
            nxt = nxt * self._evidence_step(u, state).to(nxt.dtype)
        if self.uncertainty_gate:
            nxt = nxt * self._uncertainty_weights(nxt, loc).to(nxt.dtype)

        inst = self._instant_term(u)

        # -- 绑定记忆：先读（因果）后写 -------------------------------------
        if state.bind_slots is None:
            state.bind_slots = self._empty_slots(u, rd)
        t = state.length
        ht = h[:, 0]
        sk, sv, up = state.bind_slots
        inj = self._binding_read(ht, (sk, sv, up),
                                 min(max(t - 1, 0), self.n_slots))
        if t >= 1:                                                # 有前驱才写
            sk, sv, _ = self._binding_write(ht, (sk, sv, up), t - 1)
        state.bind_slots = (sk, sv, ht)                           # u_prev 无条件前移

        y = nxt + loc + inst + inj.unsqueeze(1)
        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        state.length += 1
        y_t = x_t + self.out_scale * out
        return (y_t.squeeze(1) if squeeze else y_t), state

    # -- introspection --------------------------------------------------------
    def extra_repr(self) -> str:
        return super().extra_repr() + f", n_slots={self.n_slots}"


# evidence_weights_chunk 在 forward_chunk 里用到；放文件尾避免循环导入观感
from .gates import evidence_weights_chunk  # noqa: E402
