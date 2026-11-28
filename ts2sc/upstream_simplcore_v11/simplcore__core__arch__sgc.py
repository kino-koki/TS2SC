# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC mixer: content-gated causal spectral global convolution.

See package docstring in ``simplcore/core/__init__.py`` and ``docs/SGC_CORE.md``.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import (evidence_weight_step, evidence_weights,
                    evidence_weights_chunk, uncertainty_weights)

# θ（振荡频率）参数级钳制（P1，全 mixer 家族单一事实源）：命题 5 修正后的
# 误差界含 1/|sinθ_m|，θ→0/π 时界发散。
# **下界必须跟随 decay_max**（见 theta_clamp_floor）：logfreq 初始化把最慢基
# 设计在 θ_min = π/decay_max（振荡周期与衰减窗口尺度配对）——decay_max=4M 时
# θ_min≈7.9e-7，是合法设计值，固定 1e-3 下界会把它钳掉（低频基获得虚假幅度，
# 双形式误差超限，tuan-2b 档位直接受影响）。上界 π−1e-3 是绝对界。
THETA_CLAMP_LO = 1e-3
THETA_CLAMP_HI = math.pi - 1e-3


def theta_clamp_floor(decay_max: float) -> float:
    """θ 钳制下界：取 min(1e-3, π/decay_max)。

    恰好不低于 logfreq 设计的最慢基 θ_min = π/decay_max（合法初始化不被钳），
    同时防止训练后 raw_theta 无界漂移把 θ 推向 0（1/|sinθ| 发散、递推退化）。
    """
    if not decay_max or decay_max <= 0:
        return THETA_CLAMP_LO
    return min(THETA_CLAMP_LO, math.pi / float(decay_max))


def recurrent_dtype(compute_dtype) -> "torch.dtype":
    """递推状态的高精度 dtype：低精度计算时退化为 fp32。

    线上训练主用 bf16；IIR 振荡与 SSM 复数递推在 bf16/fp16 下会逐 token 累积
    舍入误差，使 O(1) 流式状态「数值可信」的前提失效。业界通行做法（Mamba /
    Mamba2）是把**递推状态**留在 fp32，只在 I/O 边界 cast。这里沿用同一策略：
    计算 dtype 为 ``bfloat16`` / ``float16`` 时递推状态恒为 ``float32``；
    ``float32`` 计算则保持 ``float32``（零行为变化）。所有 mixer 家族共用此单一
    事实源，避免各写一份导致策略漂移。

    非浮点 dtype（如调用方误传 token id 的 int64）一律落到 fp32——递推状态
    是浮点语义，整数域递推是数值垃圾（v1.1.2 勘误，longctx 一致性闸门实测）。
    """
    if compute_dtype in (torch.bfloat16, torch.float16):
        return torch.float32
    if not torch.tensor([], dtype=compute_dtype).is_floating_point():
        return torch.float32
    return compute_dtype


# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Cheaper than LayerNorm and the de-facto standard in modern LMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def make_norm(kind: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm kind: {kind!r}")


# ----------------------------------------------------------------------------
# Streaming state
# ----------------------------------------------------------------------------
@dataclass
class LayerStreamState:
    """Constant-size recurrent state of one SGC mixer (independent of sequence length).

    Attributes:
        y1: (B, M, D) most recent oscillator state.
        y2: (B, M, D) previous oscillator state.
        conv_buf: (B, local_kernel - 1, D) depthwise-conv history, may be None.
        length: number of tokens consumed so far.
        energy_sum: (B, 1) running sum of the per-position input energy,
            used by the L2 evidence gate. Constant size.
        spike_membrane: (B, D) optional LIF membrane state for the experimental
            SNN-SGC mixer.
        ssm_state: 仅 SRF 路径使用——持久保存的 :class:`SelectiveSpectralSSM`
            状态（含 step 的复用 scratch）。SGM/SGC 路径恒为 ``None``。用 ``Any``
            标型以避免与 ``ssm.py`` 的循环导入；其持有的 ``h_re``/``h_im`` 与
            ``y1``/``y2`` 是同一批张量（别名），所以递推态字节仍由 ``y1``/``y2`` 计，
            不重复计费；scratch 工作集不计入常驻递推态。
        bind_slots: 仅 ``sgc-bind`` mixer 使用——定容绑定记忆的
            ``(slot_k, slot_v, u_prev)`` 三元组（O(S·D) + O(D)，与 L 无关）。
            其它 mixer 恒为 ``None``。用 ``Any`` 标型避免与 ``bsgc.py`` 循环导入。
    """

    y1: torch.Tensor
    y2: torch.Tensor
    conv_buf: Optional[torch.Tensor]
    length: int = 0
    energy_sum: Optional[torch.Tensor] = None
    ssm_state: Optional[Any] = None
    spike_membrane: Optional[torch.Tensor] = None
    bind_slots: Optional[Any] = None


@dataclass
class SGCStreamState:
    """Per-layer streaming state bundle for a whole :class:`SimplCoreModel`.

    Attributes:
        layers: 每层的 mixer 状态（``LayerStreamState``；旧 checkpoint 反序列化
            时也可能出现 legacy dict 状态，``detach`` / ``num_bytes`` 会安全跳过）。
        length: 已消费的 token 数。
        memory: 投影后的 CMT 压缩记忆 (B, D)；``init_stream(memory=...)`` 注入，
            ``checkpoint_stream`` / ``restore_stream`` 随断点持久化。
        concept_state: 可选 NCM 概念递归状态 (B, concept_dim)，随会话状态独立保存。
    """

    layers: list
    length: int = 0
    memory: Optional[torch.Tensor] = None    # 投影后的 CMT 压缩记忆 (B, D)
    concept_state: Optional[torch.Tensor] = None

    def detach(self) -> "SGCStreamState":
        for st in self.layers:
            if st is None:
                continue
            # legacy dict 状态（注意力时代 checkpoint）没有 y1/y2，安全跳过。
            if isinstance(st, dict):
                continue
            st.y1 = st.y1.detach()
            st.y2 = st.y2.detach()
            if st.conv_buf is not None:
                st.conv_buf = st.conv_buf.detach()
            if st.spike_membrane is not None:
                st.spike_membrane = st.spike_membrane.detach()
            if st.bind_slots is not None:
                sk, sv, up = st.bind_slots
                st.bind_slots = (sk.detach(), sv.detach(), up.detach())
            # SRF 路径持久保存了 ssm_state，其 h_re/h_im 与 y1/y2 是别名；
            # detach 后必须重新锚定，否则持久状态会指向旧（未 detach）张量。
            if st.ssm_state is not None:
                st.ssm_state.h_re = st.y1
                st.ssm_state.h_im = st.y2
        if self.memory is not None:
            self.memory = self.memory.detach()
        if self.concept_state is not None:
            self.concept_state = self.concept_state.detach()
        return self

    def num_bytes(self) -> int:
        total = 0
        for st in self.layers:
            if st is None:
                continue
            if isinstance(st, dict):
                total += sum(v.numel() * v.element_size()
                             for v in st.values() if torch.is_tensor(v))
                continue
            total += st.y1.numel() * st.y1.element_size()
            total += st.y2.numel() * st.y2.element_size()
            if st.conv_buf is not None:
                total += st.conv_buf.numel() * st.conv_buf.element_size()
            if st.energy_sum is not None:
                total += st.energy_sum.numel() * st.energy_sum.element_size()
            if st.spike_membrane is not None:
                total += st.spike_membrane.numel() * st.spike_membrane.element_size()
            if st.bind_slots is not None:
                for t in st.bind_slots:
                    if isinstance(t, torch.Tensor):
                        total += t.numel() * t.element_size()
        if self.concept_state is not None:
            total += self.concept_state.numel() * self.concept_state.element_size()
        return total


# ----------------------------------------------------------------------------
# FFT helpers
# ----------------------------------------------------------------------------
def fft_causal_conv(u: torch.Tensor, kernel: torch.Tensor, fft_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Causal (depthwise) convolution of ``u`` with a length-L kernel bank.

    Computes ``y[b, t, d] = sum_{s <= t} kernel[t - s, d] * u[b, s, d]`` exactly,
    in ``O(L log L)``, by zero-padding to ``N = 2L`` and multiplying in the
    frequency domain (circular convolution of length 2L has no aliasing for a
    causal kernel of length <= L).

    Args:
        u: (B, L, D) input.
        kernel: (L, D) kernel bank (shared across the batch).
        fft_dtype: dtype used for the transform; ``float32`` is recommended.
    """
    B, L, D = u.shape
    N = 2 * L
    work_dtype = fft_dtype if fft_dtype is not None else u.dtype
    xw = u.to(work_dtype)
    kw = kernel.to(work_dtype)
    cX = torch.fft.rfft(xw, n=N, dim=1)
    cK = torch.fft.rfft(kw, n=N, dim=0).unsqueeze(0)
    cY = cX * cK
    y = torch.fft.irfft(cY, n=N, dim=1)[:, :L, :]
    return y.to(u.dtype)


def depthwise_causal_conv(u: torch.Tensor, weight: torch.Tensor, buf: Optional[torch.Tensor]):
    """Depthwise causal 1-D convolution with ``weight`` of shape (D, 1, K).

    Returns ``(y, new_buf)`` where ``new_buf`` holds the last ``K-1`` inputs, so a
    single implementation serves both the parallel path (``buf=None``) and the
    streaming path (one position at a time, with ``buf`` carried in the state).

    In the streaming case the convolution is evaluated over ``cat(buf, u)`` and
    only the outputs belonging to the **new** positions are returned; otherwise the
    buffered positions would be emitted again.
    """
    K = weight.shape[-1]
    if K == 1:
        return u * weight.view(1, 1, -1), u[:, -0:, :]
    if buf is not None and buf.shape[1] > 0:
        uu = torch.cat([buf, u], dim=1)
    else:
        uu = u
    n_new = u.shape[1]
    pad = K - 1
    up = F.pad(uu.transpose(1, 2), (pad, 0))          # (B, D, T+pad), left pad => causal
    y = F.conv1d(up, weight, groups=weight.shape[0])  # (B, D, T)
    y = y.transpose(1, 2)[:, -n_new:, :]              # keep only the new positions
    if uu.shape[1] >= pad:
        new_buf = uu[:, uu.shape[1] - pad:, :]
    else:
        new_buf = uu
    return y, new_buf


# ----------------------------------------------------------------------------
# The mixer
# ----------------------------------------------------------------------------
class SpectralGlobalConv(nn.Module):
    """SGC mixer: content-gated causal spectral global convolution.

    Forward (parallel / training form)::

        h  = norm(x)
        u  = in_proj(h) * sigmoid(gate_proj(h))        # content gating (selective)
        y  = fft_causal_conv(u, K)                     # long range, O(L log L)
           + depthwise_causal_conv(u, W_loc)           # short range, O(L*K)
           + feed * u                                  # per-channel instantaneous term
        out = out_proj(y)
        return x + out_scale * out

    where the global kernel bank is::

        K[tau, d] = sum_m coef[m, d] * psi_m[tau]
        psi_m[tau] = rho_m^tau * sin((tau + 1) * theta_m)

    ``step()`` evaluates the same operator one token at a time through the
    equivalent 2nd-order IIR recurrence, with O(1) state in the sequence length.

    Args:
        dim: channel width D.
        num_bases: number of damped-oscillator bases M (default 64).
        decay_max: longest decay time constant used to initialise rho (default 8192).
        local_kernel: depthwise causal conv width K (default 4).
        rank: rank r of the short-range depthwise conv parameterisation. The conv
            is depthwise so this is unused unless ``low_rank_local`` is set.
        content_gate: enable input-dependent gating (default True).
        out_gate: enable an output gate (default False).
        norm: "rmsnorm" or "layernorm".
        fft_dtype: dtype for the FFT path; ``torch.float32`` by default.
        eps: norm epsilon.
        kernel_norm: "none" (default, bitwise-compatible with all existing
            checkpoints) or "analytic_l2" — divide each oscillator's readout
            coefficient by its infinite-horizon L2 energy. Removes the
            length-dependent injection drift; no L-dependent term is
            introduced, so FFT/IIR duality and O(1) state are preserved.
    """

    def __init__(
        self,
        dim: int,
        num_bases: int = 64,
        decay_max: float = 8192.0,
        local_kernel: int = 4,
        content_gate: bool = True,
        out_gate: bool = False,
        evidence_gate: bool = False,      # P0-4: 与 SimplCoreConfig / 技术报告 §3.1 对齐
        uncertainty_gate: bool = False,
        norm: str = "rmsnorm",
        fft_dtype: Optional[torch.dtype] = torch.float32,
        eps: float = 1e-6,
        num_kernels: int = 1,
        use_mixer_stream: bool = False,
        path_gate: bool = False,
        inst_nonlin: bool = False,
        init_mode: str = "logfreq",
        rho_param: str = "sigmoid",
        linear_floor: float = 0.3,
        kernel_norm: str = "none",
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")
        if local_kernel < 1:
            raise ValueError("local_kernel must be >= 1")
        if kernel_norm not in ("none", "analytic_l2"):
            raise ValueError(
                f"unknown kernel_norm={kernel_norm!r} (none | analytic_l2)")
        if kernel_norm != "none" and use_mixer_stream:
            # 融合算子在 C++ 侧直接读 coef，未实现解析归一；必须回退 PyTorch
            # step 路径，否则 FFT/IIR 对偶静默破裂。显式报错而非静默降级。
            raise ValueError(
                "kernel_norm='analytic_l2' is incompatible with "
                "use_mixer_stream=True (fused op reads raw coef); "
                "set use_mixer_stream=False")

        self.kernel_norm = kernel_norm
        self.dim = dim
        self.num_bases = num_bases
        self.local_kernel = local_kernel
        self.content_gate = content_gate
        self.out_gate = out_gate
        self.fft_dtype = fft_dtype
        # 融合 C++/CUDA 热路径开关（默认关闭，保持原纯 PyTorch 行为不变）。
        # 仅在 fp32 + 扩展已构建时生效，否则 step() 自动回退。
        self.use_mixer_stream = bool(use_mixer_stream)

        # --- v1.2 增强（全部默认关，开启后不影响 FFT/IIR 对偶） --------------
        # path_gate：三路（长程/局部/瞬时）逐位置自适应混合。
        #   混合权重是**当前 token 的因果函数** w_t = softmax(W h_t)，与序列
        #   其余位置无关 → 并行形式与流式形式逐位一致，对偶性不破坏。
        #   比固定相加多了"按内容选路"的表达力：路由型任务可以让长程路静默。
        self.path_gate = path_gate
        if path_gate:
            self.path_gate_proj = nn.Linear(dim, 3, bias=True)
        # inst_nonlin：瞬时项非线性 feed·SiLU(u)（默认关，保持 feed·u）。
        #   逐点函数不经过卷积/递推，两种形式天然一致；给"当下"路径一个
        #   非线性响应，与长程线性卷积形成曲率对比。
        self.inst_nonlin = inst_nonlin
        # init_mode：振荡基网格初始化，见 _init_oscillator_grid。
        #   * "logfreq"（历史默认）：theta 对数分布 [pi/decay_max, 0.9*pi/2]。
        #     ★ 实测缺陷：低频端绝对间隔趋零，字典近似共线 ——
        #       仓库守护口径 (Psi=sin((t+1)theta), L=512, SVD) 下
        #       M=32 时数值秩 24/32、cond 1.92e16；含 rho 的真实字典
        #       条件数更差。表征容量塌缩，与命题5 的 1/|sin(theta)| 发散同源。
        #   * "hybrid"：线性均匀 + 黄金比扰动，修了秩缺失但仍保留
        #       theta_min=pi/decay_max（数值病态未根除）。
        #   * "linear"（推荐用于新训练）：theta 线性均匀
        #       [linear_floor, pi-linear_floor]，默认 floor=0.3。
        #       同口径 cond 1.04、秩 32/32；零参数零 FLOPs 零状态开销。
        #   * "legacy"：均匀网格有代数公共零点（tau=2M-1 等），显式拒绝。
        if init_mode not in ("legacy", "logfreq", "hybrid", "linear"):
            raise ValueError(
                f"unknown init_mode: {init_mode!r} "
                f"(legacy | logfreq | hybrid | linear)")
        if not (0.0 < float(linear_floor) < math.pi / 2):
            raise ValueError(
                f"linear_floor must be in (0, pi/2), got {linear_floor}")
        self.init_mode = init_mode
        self.linear_floor = float(linear_floor)
        # rho_param：衰减极点参数化。
        #   * "sigmoid"（默认）：rho = sigmoid(raw).clamp(max=1-1e-6)。
        #     torch fp32 sigmoid 在 raw ≳ 16.636（=24·ln2，exp(x)/(1+exp(x))
        #     实现路径实测；解析舍入充分界为 25·ln2≈17.33）时精确返回 1.0，
        #     夹子防单位圆极点；代价是最长记忆约 1e6 token，4M 所需的
        #     0.99999975 不可达。
        #   * "logspace"（opt-in）：rho = exp(-softplus(raw_rho_log))，
        #     可表达任意接近 1 的极点（受 fp32 分辨率与 1-2^-23 卫兵限制），
        #     无 sigmoid 饱和；state_dict 与旧参数化自动迁移。
        if rho_param not in ("sigmoid", "logspace"):
            raise ValueError(
                f"unknown rho_param: {rho_param!r} (sigmoid | logspace)")
        self.rho_param = rho_param

        self.norm = make_norm(norm, dim, eps=eps)

        # --- content / projections -------------------------------------------
        self.in_proj = nn.Linear(dim, dim, bias=False)
        if content_gate:
            self.gate_proj = nn.Linear(dim, dim, bias=False)
        if out_gate:
            self.out_gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # --- long-range oscillator bank --------------------------------------
        rho0, theta0 = _init_oscillator_grid(
            num_bases, decay_max, mode=self.init_mode,
            theta_floor=self.linear_floor)
        # θ 钳制下界在构造时冻结（跟随 decay_max，见 theta_clamp_floor）
        self._theta_floor = theta_clamp_floor(decay_max)
        if rho_param == "logspace":
            # softplus(x) = 1/lam  =>  x = log(expm1(1/lam))
            inv_lam = -torch.log(rho0)
            self.raw_rho_log = nn.Parameter(torch.log(torch.expm1(inv_lam)))
        else:
            self.raw_rho = nn.Parameter(_logit(rho0))
        self.raw_theta = nn.Parameter(_logit(theta0 / math.pi))
        # Small non-zero init so that d/d(rho), d/d(theta) are live from step 0
        # (with coef == 0 the kernel does not depend on rho/theta at all).
        self.coef = nn.Parameter(torch.randn(num_bases, dim) * 0.02)

        # --- MoK: Mixture-of-Kernels（逐位置内容路由，修复 pairwise 缺陷）----
        #
        # 架构缺陷（实测，非推测）
        # ----------------------
        # 注意力:  y[i] = Σ_j softmax(q_i·k_j/√d)·v_j
        #   权重 a[i,j] 是**位置对 (i,j) 的内容函数** —— 二阶（pairwise）交互。
        # SGC:     y[i] = Σ_τ K[τ]·u[i-τ]
        #   核 K 是**输入无关的常数**；输入相关部分只有逐元素门控
        #   （content_gate / evidence_gate / uncertainty_gate / path_gate），
        #   这些门控只能逐位置、逐通道地缩放，**无法表达
        #   "位置 i 应当聚合历史的哪种模式"**。
        #   → 余弦只能拟合"平均位置分布"，复现不了逐 token 的内容路由。
        #   这是 TS→SC 迁移里 mixer 余弦长期卡在 0.55~0.97 的结构性原因。
        #
        # 修复：核本身成为输入的函数 —— N 个核的**逐位置**加权混合
        #   y_long[t] = Σ_k g_k(h_t) · conv(u, K_k)[t],   g = softmax(W h_t)
        #
        # 为什么这修的是根因
        #   * 逐位置：每个 token 依据自身内容选择不同的历史聚合模式
        #     → 同一序列里不同位置可以有不同的"关注模式"，这是 pairwise
        #       路由在卷积框架下可达到的最强形式。
        #   * 仍然卷积：每个 conv(u, K_k) 是纯因果卷积 → FFT 仍 O(L log L)。
        #
        # 为什么核心契约不破（这是本设计的关键）
        #   所有核**共享同一组振荡基 psi**（rho/theta），只有 readout 系数
        #   coef 不同。于是 IIR 递推
        #       y_new = a·y1 + b·y2 + c·u
        #   与核无关 —— 状态 (y1,y2) 的形状/字节数与 num_kernels 完全无关，
        #   **O(1) 流式状态逐字节不变**。多核只体现在 readout：
        #       nxt_k = einsum("bmd,kmd->bkd", y_new, coefs)
        #       nxt   = Σ_k g_k · nxt_k
        #   代价是 readout 的 K 倍常数 FLOPs，与序列长度无关。
        #
        # 对偶性：g_k(h_t) 是 h_t 的因果函数，与序列其余位置无关 —— 与
        #   content_gate / path_gate 同构 → 并行(FFT) 与流式(IIR) 逐位一致。
        #
        # 兼容：num_kernels=1 时不创建任何新参数、走原代码路径，逐位等价。
        self.num_kernels = int(num_kernels)
        if self.num_kernels < 1:
            raise ValueError("num_kernels must be >= 1")
        if self.num_kernels > 1:
            # 多核 readout 系数；coef[0] 是原核，其余小随机初始化，
            # 保证 num_kernels=1 的参数语义不变。
            self.coefs = nn.Parameter(
                torch.randn(self.num_kernels, num_bases, dim) * 0.02)
            with torch.no_grad():
                self.coefs[0].copy_(self.coef)
            # 逐位置路由器：D → K。
            # ★ 不能 zero-init：若 g 恒为均匀 1/K 且所有核初始相同，
            #   ∂L/∂K_k ∝ g_k·G 对每个 k 完全相同 → 对称性锁定，
            #   各核永远收到同一梯度、无法分化（MoE 经典陷阱，见
            #   RoutedSpectralGlobalConv.route_proj 的同类处理）。
            #   用 0.01 小随机打破对称：g 仍近似均匀（≈1/K），
            #   起步接近"平均核"，但梯度已能分流。
            self.kernel_route = nn.Linear(dim, self.num_kernels, bias=True)
            with torch.no_grad():
                self.kernel_route.weight.normal_(0, 0.01)
                self.kernel_route.bias.zero_()

        # Per-channel instantaneous (diagonal) term; init 1 keeps the block close
        # to identity at the start of training.
        self.feed = nn.Parameter(torch.ones(dim))

        # --- short-range depthwise conv --------------------------------------
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=local_kernel,
                                    groups=dim, bias=False)
        with torch.no_grad():
            self.local_conv.weight.zero_()
            self.local_conv.weight[:, 0, -1] = 1.0   # identity at init

        self.out_scale = nn.Parameter(torch.ones(1))

        # --- four-layer gate: L2 evidence / L3 uncertainty -------------------
        # L2 证据门: per-position evidence derived from the *causal prefix* energy
        # of the gated input; damps the long-range path when the context carries
        # no usable signal. Because the statistic is a causal-prefix quantity,
        # the parallel (cumsum) form and the streaming (running sum) form agree,
        # which is what preserves the FFT / IIR dual form.
        self.evidence_gate = evidence_gate
        if evidence_gate:
            # 用标量参数而非 Linear：闸门响应形状是**结构性**的（证据能量越高越开），
            # 只有强度/偏置可学。这样做有两个好处：
            #   1) 避免全零权重导致闸门退化成常数（对输入不敏感，形同虚设）；
            #   2) 不受上层模型 `apply(self._init_weights)` 对所有 nn.Linear 重施
            #      normal(0, init_std) 的影响。
            # v1.2 闸门 opt-in：门关闭时**不创建**参数（analytic_params 同口径计数），
            # 避免"监督缺失即空转"的幽灵参数。
            self.evidence_gain = nn.Parameter(torch.tensor(1.0))       # 证据项强度
            self.evidence_pos_gain = nn.Parameter(torch.tensor(0.0))   # 位置计数项强度
            self.evidence_bias = nn.Parameter(torch.tensor(0.0))        # 偏置
        # L3 不确定门: the long-range and local paths are two independent
        # estimates of the same context; their disagreement is an
        # uncertainty proxy. High disagreement damps the long path so the
        # model falls back on local evidence.
        self.uncertainty_gate = uncertainty_gate
        if uncertainty_gate:
            self.uncertainty_logit = nn.Parameter(torch.zeros(1))
        self.gate_eps = 1e-6

    # -- kernel ---------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs) -> None:
        """在两种 rho 参数化之间迁移 checkpoint；并把单核 checkpoint 热启动到 MoK。

        MoK 热启动（重要，否则 TS2SC 已有全部 mixer 权重作废）
        ------------------------------------------------------
        单核 checkpoint 只有 ``coef``，没有 ``coefs`` / ``kernel_route``。
        映射规则：
          * ``coefs[k] = coef`` 对所有 k（所有核初始化为同一个核）
          * ``kernel_route`` 保持 zero-init（路由均匀 = 1/K）
        因为 Σ_k g_k = 1 且所有 K_k 相同，若 router 也置零则
            y_long = Σ_k g_k · conv(u, K) = conv(u, K)
        → 输出**逐位等价**（可显式把 router 置零来获得精确热启动）。
        但默认注入 0.01 小随机 router 以打破梯度对称，此时输出是
        **近似等价**（相对误差 ~1% 量级）：起步接近"平均核"，
        不丢已有知识、不引入初始化冲击，且训练能真正让各核分化。
        """
        if self.num_kernels > 1:
            ck = prefix + "coefs"
            c1 = prefix + "coef"
            if ck not in state_dict and c1 in state_dict:
                coef = state_dict[c1]
                state_dict[ck] = coef.unsqueeze(0).expand(
                    self.num_kernels, -1, -1).clone()
                # 注意：保留 state_dict[c1]（不删）。多核模块仍注册 ``coef``
                # 作为兼容占位参数（M×D，前向不走它），这样单核/多核
                # checkpoint 都能 strict=True 双向加载，不破坏任何既有权重。
                # 补上 router 参数，使 strict=True 也能直接加载。
                # 必须是**小随机而非零**：热启动时所有核被复制成同一个核，
                # 若 g 还恒等于 1/K，则梯度对称、多核永久退化为单核。
                state_dict[prefix + "kernel_route.weight"] = (
                    torch.randn(self.num_kernels, self.dim, dtype=coef.dtype,
                                device=coef.device) * 0.01)
                state_dict[prefix + "kernel_route.bias"] = torch.zeros(
                    self.num_kernels, dtype=coef.dtype, device=coef.device)

        old = prefix + "raw_rho"
        new = prefix + "raw_rho_log"
        if self.rho_param == "logspace" and old in state_dict and new not in state_dict:
            rho_old = torch.sigmoid(state_dict[old]).clamp(min=1e-6, max=1 - 1e-6)
            state_dict[new] = torch.log(torch.expm1(-torch.log(rho_old)))
            del state_dict[old]
        elif self.rho_param == "sigmoid" and new in state_dict and old not in state_dict:
            rho_new = torch.exp(-F.softplus(state_dict[new])) \
                .clamp(min=1e-6, max=1 - 1e-6)
            state_dict[old] = torch.log(rho_new) - torch.log1p(-rho_new)
            del state_dict[new]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def _psi_l2_inv(self, dtype) -> Optional[torch.Tensor]:
        """每个振荡基在**无限长**序列上能量的倒数平方根，形状 (M,)。

        闭式（严格与 L 无关，因此可安全用于 FFT 与 IIR 两条路径）::

            ||psi_m||^2_oo = SUM_{t>=0} rho^{2t} sin^2((t+1)theta)
                           = 1/2/(1-z) - 1/2 (c-z)/(1-2zc+z^2),
            z = rho^2, c = cos(2 theta).

        推导：sin^2(a)=(1-cos 2a)/2，几何级数 SUM z^t e^{2i(t+1)theta} 取实部。
        数值核对见 docs/kernel_norm_fix.md（m=31: 闭式 354.4743 =
        L=8192 数值 354.4710）。

        为什么需要它（实测架构缺陷）
        --------------------------------
        未归一化时 ``long[t] = SUM_tau K[tau] u[t-tau]`` 的累加项数随 t 增长，
        rho->1 的慢基几乎不衰减，注入幅度随序列长度单调膨胀；而注意力的
        softmax 是凸组合、幅度与 L 无关。实测 Qwen3-0.6B 迁移 ck 上
        L=8->256 时 SGC long 的 rms 增长 2.82x，teacher attn@v 反而降到 0.27x
        ——方向相反。生成时序列逐 token 变长，mixer 注入持续增大，正是
        「前言不搭后语」的结构来源。

        把 1/||psi_m||_oo 吸收进 readout 系数 coef 后，每个基对核的贡献
        变为单位能量，long 的统计幅度由 coef 决定、与 L 无关。
        递推 (y1,y2) 完全不变 —— 只是 readout 缩放，故 O(1) 状态、
        FFT/IIR 对偶、因果性全部保持。梯度仍可经 rho/theta 回流。
        """
        if self.kernel_norm == "none":
            return None
        rho, theta = self.oscillator_params()
        rho = rho.to(dtype); theta = theta.to(dtype)
        z = (rho * rho).clamp(max=1.0 - 1e-7)
        c = torch.cos(2.0 * theta)
        denom = (1.0 - 2.0 * z * c + z * z).clamp_min(1e-12)
        e2 = 0.5 / (1.0 - z) - 0.5 * (c - z) / denom
        return torch.rsqrt(e2.clamp_min(1e-8))                       # (M,)

    def _coef_eff(self, dtype) -> torch.Tensor:
        """readout 系数（单核）：归一化因子已吸收，形状 (M,D)。"""
        c = self.coef.to(dtype)
        inv = self._psi_l2_inv(dtype)
        return c if inv is None else c * inv.unsqueeze(1)

    def _coefs_eff(self, dtype) -> torch.Tensor:
        """readout 系数（MoK 多核）：形状 (K,M,D)。"""
        c = self.coefs.to(dtype) if self.num_kernels > 1 \
            else self.coef.to(dtype).unsqueeze(0)
        inv = self._psi_l2_inv(dtype)
        return c if inv is None else c * inv.view(1, -1, 1)

    def oscillator_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (rho, theta), both of shape (M,), with rho in (0,1), theta in (0,pi).

        两套极点参数化：

        * ``sigmoid``（默认）：``rho = sigmoid(raw).clamp(max=1-1e-6)``。
          torch fp32 sigmoid 在 raw ≳ 16.636（=24·ln2，exp(x)/(1+exp(x))
          实现路径实测；解析舍入充分界 25·ln2≈17.33）时精确返回 1.0，
          夹子防单位圆极点；代价是最长记忆约 1e6 token，4M 所需的
          ``0.99999975`` 不可达。
        * ``logspace``（opt-in）：``rho = exp(-softplus(raw_rho_log))``。
          无 sigmoid 饱和，可表达 4M 级极点；上限 ``1-2^-23`` 是 fp32 在 1.0
          附近的最小可表示间隔（≈8.4M token 时间常数），仍高于 4M 所需。
        """
        if self.rho_param == "logspace":
            rho = torch.exp(-F.softplus(self.raw_rho_log)) \
                .clamp(max=1.0 - 2.0 ** -23)
        else:
            rho = torch.sigmoid(self.raw_rho).clamp(max=1.0 - 1e-6)
        # θ 参数级钳制（P1）：命题 5 修正后的误差界含 1/|sinθ_m| 因子，θ→0/π
        # 时界发散。下界在构造时冻结（跟随 decay_max，保证 logfreq 最慢基的
        # 合法设计值不被钳）；上界 π−1e-3 是绝对界。raw_theta 可学、训练后
        # 无上界，钳制防其漂到数值退化区（θ 贴边时 a=2ρcosθ 退化为 ±2ρ，
        # 振荡语义消失）。
        theta = (torch.sigmoid(self.raw_theta) * math.pi
                 ).clamp(self._theta_floor, THETA_CLAMP_HI)
        return rho, theta

    def build_kernel(self, length: int, device=None, dtype=None) -> torch.Tensor:
        """Materialise the causal kernel bank K of shape (length, D).

        Defined for **any** ``length``: there is no ``max_len`` truncation. This
        removes the SGC v2 defect where sequences longer than ``max_len``
        silently degraded into a fixed-window convolution.
        """
        rho, theta = self.oscillator_params()
        device = device if device is not None else self.coef.device
        dtype = dtype if dtype is not None else self.coef.dtype
        tau = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)      # (1,L)
        log_rho = torch.log(rho.to(dtype)).unsqueeze(1)                          # (M,1)
        psi = torch.exp(log_rho * tau) * torch.sin((tau + 1.0) * theta.to(dtype).unsqueeze(1))
        c = self._coefs_eff(dtype)[0] if self.num_kernels > 1 \
            else self._coef_eff(dtype)
        kernel = c.t() @ psi                                                     # (D, M) @ (M, L)
        return kernel.t().contiguous()                                           # (L, D)

    def build_kernels(self, length: int, device=None, dtype=None) -> torch.Tensor:
        """物化 MoK 核栈，shape ``(K, length, D)``。

        所有核共享同一组振荡基 ``psi``（rho/theta），只有 readout 系数不同 ——
        这是"多核不增大 O(1) 状态"的结构原因（见 ``__init__`` 的 MoK 注释）。
        ``num_kernels == 1`` 时返回单核栈，与 :meth:`build_kernel` 一致。
        """
        rho, theta = self.oscillator_params()
        device = device if device is not None else self.coef.device
        dtype = dtype if dtype is not None else self.coef.dtype
        tau = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)      # (1,L)
        log_rho = torch.log(rho.to(dtype)).unsqueeze(1)                          # (M,1)
        psi = torch.exp(log_rho * tau) * torch.sin((tau + 1.0) * theta.to(dtype).unsqueeze(1))
        # (K, M, D) @ (M, L) -> (K, D, L) -> (K, L, D)
        coefs = self._coefs_eff(dtype)
        kernels = torch.matmul(coefs.transpose(-1, -2), psi)
        return kernels.transpose(-1, -2).contiguous()

    def _kernel_route_weights(self, h: torch.Tensor) -> torch.Tensor:
        """逐位置核路由权重 ``g``，shape ``(B, L, K)``，沿 K 归一。

        ``g[t] = softmax(W h_t)`` 是**当前 token 的因果函数**，与序列其余位置
        无关 —— 与 ``content_gate`` / ``path_gate`` 同构，因此并行(FFT)与
        流式(IIR)两种形式在 fp32 内逐位一致，FFT/IIR 对偶性不被破坏。
        """
        return torch.softmax(self.kernel_route(h), dim=-1)

    # -- forward --------------------------------------------------------------
    def _instant_term(self, u: torch.Tensor) -> torch.Tensor:
        """瞬时项：``feed * u``（默认）或 ``feed * SiLU(u)``（inst_nonlin）。"""
        if self.inst_nonlin:
            return self.feed * F.silu(u)
        return self.feed * u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, L, D = x.shape
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        if self.num_kernels > 1:
            # MoK：逐位置内容路由的核混合（修复 pairwise 表达力缺陷）
            kernels = self.build_kernels(L, device=u.device, dtype=u.dtype)   # (K,L,D)
            g = self._kernel_route_weights(h)                                  # (B,L,K)
            long = torch.zeros_like(u)
            for k in range(self.num_kernels):
                long = long + g[..., k:k + 1] * fft_causal_conv(
                    u, kernels[k], self.fft_dtype)
        else:
            kernel = self.build_kernel(L, device=u.device, dtype=u.dtype)
            long = fft_causal_conv(u, kernel, self.fft_dtype)
        local = self._local_conv_full(u)

        if self.evidence_gate:
            long = long * self._evidence_weights(u)
        if self.uncertainty_gate:
            long = long * self._uncertainty_weights(long, local)

        inst = self._instant_term(u)
        if self.path_gate:
            w = torch.softmax(self.path_gate_proj(h), dim=-1)   # (B,L,3)
            y = (w[..., 0:1] * long + w[..., 1:2] * local
                 + w[..., 2:3] * inst)
        else:
            y = long + local + inst

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        return residual + self.out_scale * out

    # -- chunked / stateful parallel form --------------------------------------
    def _homogeneous(self, state: LayerStreamState, rd: torch.dtype):
        """二阶 IIR 的齐次解系数（per oscillator × per channel）。

        ``step`` 用递推推进状态，chunk 形式需要在"新块的第一位置"处恢复
        历史贡献。给定上一块末状态 ``y1=y[-1], y2=y[-2]``，齐次解为::

            h[t] = rho^(t+1) * (A cos((t+1)theta) + B sin((t+1)theta))
            A = y1
            B = (A cos(theta) - rho * y2) / sin(theta)

        ``theta -> 0`` 时特征根退化为重根 ``rho``，上式除以 ``sin(theta)``
        会失去数值意义；此时用极限形式 ``h[t] = (C + D t) rho^t``，
        其中 ``C = 2 rho y1 - rho^2 y2``、``D = rho y1 - rho^2 y2``。
        两支都计算、用 ``where`` 选择 —— 对小 theta 的振荡支做 ``sin`` 下限
        保护，避免被丢弃的分支产生 inf/nan 污染梯度。
        """
        rho, theta = self.oscillator_params()
        rho = rho.to(rd)
        theta = theta.to(rd)
        a = state.y1.to(rd)
        y2 = state.y2.to(rd)
        rho_b = rho.view(1, -1, 1)
        cos_t = torch.cos(theta).view(1, -1, 1)
        sin_t = torch.sin(theta).clamp(min=1e-3).view(1, -1, 1)
        b_osc = (a * cos_t - rho_b * y2) / sin_t
        rho2 = rho_b * rho_b
        lin_d = rho_b * a - rho2 * y2
        lin_c = 2.0 * rho_b * a - rho2 * y2
        small = (theta < 1e-3).view(1, -1, 1)
        a_eff = torch.where(small, lin_c, a)
        b_eff = torch.where(small, lin_d, b_osc)
        return rho, theta, a_eff, b_eff, small

    def _free_response(self, u: torch.Tensor,
                        state: LayerStreamState,
                        g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """上一块状态在本块上的**齐次响应**，形状 (B, T, D)。

        与 ``fft_causal_conv(u, K)`` 的零初值部分相加，得到与整段并行/逐
        token 流式完全相同的 ``long`` —— 这是 chunk 间状态能"无损前向"的
        数学基础（梯度仍按块截断，见 ``forward_chunk``）。
        """
        B, T, D = u.shape
        rd = state.y1.dtype
        rho, theta, a_eff, b_eff, _ = self._homogeneous(state, rd)
        tau = torch.arange(T, device=u.device, dtype=rd).view(1, -1)
        log_rho = torch.log(rho.clamp(min=1e-12)).view(-1, 1)
        phase = (tau + 1.0) * theta.view(-1, 1)
        decay = torch.exp(log_rho * (tau + 1.0))
        p = decay * torch.cos(phase)
        q = decay * torch.sin(phase)
        # 重根极限支：h[t] = (A + B t) rho^t
        r = torch.exp(log_rho * tau)
        p_lin = r
        q_lin = r * tau
        small_m = (theta < 1e-3).view(-1, 1)
        p = torch.where(small_m, p_lin, p)
        q = torch.where(small_m, q_lin, q)
        # 状态 (y1,y2) 与核无关（共享 psi），但 readout 依赖核：
        # MoK 下自由响应必须按同一组逐位置路由权重 g 加权，
        # 否则 chunk 形式与 full forward 在跨块处不一致。
        if g is not None and self.num_kernels > 1:
            coefs = self._coefs_eff(rd)                               # (K,M,D)
            a_k = a_eff.unsqueeze(1) * coefs.unsqueeze(0)              # (B,K,M,D)
            b_k = b_eff.unsqueeze(1) * coefs.unsqueeze(0)
            h_k = torch.einsum("mt,bkmd->bktd", p, a_k)
            h_k = h_k + torch.einsum("mt,bkmd->bktd", q, b_k)
            h = torch.einsum("bktd,btk->btd", h_k, g.to(rd))
        else:
            coef = self._coef_eff(rd)
            h = torch.einsum("mt,bmd->btd", p, a_eff * coef.unsqueeze(0))
            h = h + torch.einsum("mt,bmd->btd", q, b_eff * coef.unsqueeze(0))
        return h.to(u.dtype)

    def _tail_state(self, u: torch.Tensor,
                    state: LayerStreamState) -> Tuple[torch.Tensor, torch.Tensor]:
        """块末的 ``(y1, y2)``，形状各 (B, M, D)。

        零初值卷积的末两点无法从混合后的 ``long`` 反解（``coef`` 把 M 个
        振荡器混进了 D 个通道），因此按 oscillator 维度做两次
        ``(M, T) @ (T, D)``：成本 ``O(M·T·D)``，与自由响应同阶，远小于
        FFT 卷积本身的 ``O(T log T · D)`` 常数倍。

        该计算在 ``forward_chunk`` 的 ``no_grad`` 内执行：块间状态按截断
        BPTT 语义 **detach**，不为它构建跨块计算图。
        """
        B, T, D = u.shape
        rd = state.y1.dtype
        M = self.num_bases
        rho, theta, a_eff, b_eff, small = self._homogeneous(state, rd)
        tau = torch.arange(T, device=u.device, dtype=rd)
        log_rho = torch.log(rho.clamp(min=1e-12)).view(-1, 1)
        psi = torch.exp(log_rho * tau.view(1, -1)) \
            * torch.sin((tau + 1.0).view(1, -1) * theta.view(-1, 1))
        flip1 = psi.flip(-1)                       # psi[T-1-s]
        if T >= 2:
            flip2 = torch.cat([flip1[:, 1:],
                               torch.zeros(M, 1, device=u.device, dtype=rd)], dim=1)
        else:
            flip2 = torch.zeros_like(flip1)
        u_rd = u.to(rd)
        z1 = torch.einsum("ms,bsd->bmd", flip1, u_rd)
        z2 = torch.einsum("ms,bsd->bmd", flip2, u_rd)

        def _h_at(t: int) -> torch.Tensor:
            # 振荡支：h[t] = rho^(t+1) (A cos((t+1)θ) + B sin((t+1)θ))
            ph = float(t + 1) * theta.view(-1, 1)
            decay = torch.exp(log_rho * float(t + 1)).view(1, -1, 1)
            osc = decay * (a_eff * torch.cos(ph).view(1, -1, 1)
                           + b_eff * torch.sin(ph).view(1, -1, 1))
            # 重根支：h[t] = (C + D t) rho^t
            lin = (a_eff + b_eff * float(t)) \
                * torch.exp(log_rho * float(t)).view(1, -1, 1)
            return torch.where(small, lin, osc)

        return z1 + _h_at(T - 1), z2 + _h_at(T - 2)

    def forward_chunk(self, x: torch.Tensor,
                      state: Optional[LayerStreamState] = None,
                      *, update_state: bool = True
                      ) -> Tuple[torch.Tensor, Optional[LayerStreamState]]:
        """带跨块状态的并行前向：``forward`` 的 chunk 形式。

        语义与 ``forward(x_full)`` 逐位一致（fp32 误差内），但显存是
        ``O(chunk · D)`` 而不是 ``O(L · D)``：每块只物化 chunk 长的核与激活，
        块间历史通过 ``state`` 的齐次响应注入。

        梯度语义是**截断 BPTT**：块内参数梯度完整；块间状态 detach
        （``_tail_state`` 在 ``no_grad`` 内计算）。这与
        ``trainer.StreamingTrainer`` 的口径一致，也是长上下文训练的标准取舍。
        """
        residual = x
        B, T, D = x.shape
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        g = None
        if self.num_kernels > 1:
            kernels = self.build_kernels(T, device=u.device, dtype=u.dtype)   # (K,T,D)
            g = self._kernel_route_weights(h)                                  # (B,T,K)
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
        if self.path_gate:
            w = torch.softmax(self.path_gate_proj(h), dim=-1)
            y = (w[..., 0:1] * long + w[..., 1:2] * local
                 + w[..., 2:3] * inst)
        else:
            y = long + local + inst

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        y = residual + self.out_scale * out
        if state is None or not update_state:
            return y, state

        with torch.no_grad():
            y1, y2 = self._tail_state(u.detach(), state)
        new_state = LayerStreamState(
            y1=y1, y2=y2,
            conv_buf=(new_buf.detach() if new_buf is not None
                      and new_buf.shape[1] > 0 else None),
            length=state.length + T,
            energy_sum=(new_energy.detach() if new_energy is not None else None))
        return y, new_state

    def _local_conv_full(self, u: torch.Tensor) -> torch.Tensor:
        K = self.local_kernel
        if K == 1:
            return u * self.local_conv.weight.view(1, 1, -1)
        up = F.pad(u.transpose(1, 2), (K - 1, 0))
        return F.conv1d(up, self.local_conv.weight, groups=self.dim).transpose(1, 2)

    # -- L2 evidence gate -----------------------------------------------------
    def _evidence_weights(self, u: torch.Tensor) -> torch.Tensor:
        """Per-position evidence weight in (0, 1), shape (B, L, 1).

        Evidence is the causal prefix average energy of the gated input::

            c_t = (1/t) * sum_{s<=t} ||u_s||^2

        ``log1p`` features of ``(c_t, t)`` feed a 2->1 linear squashed by a
        sigmoid, so a low-energy prefix (nothing usable in the context)
        shrinks the long-range contribution, while a high-energy prefix
        opens it.

        The statistic is a *causal prefix* quantity, hence the parallel
        (cumsum) form and the streaming (running sum) form agree -- this is
        what keeps the FFT / IIR dual form intact.
        """
        return evidence_weights(u, self.evidence_gain,
                                self.evidence_pos_gain, self.evidence_bias)

    # -- L3 uncertainty gate --------------------------------------------------
    def _uncertainty_weights(self, long: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Per-position damping factor in (0, 1], shape (B, L, 1).

        The long-range and local paths are two independent estimates of the
        same context. Their disagreement is a disagreement-style uncertainty
        proxy: when they diverge, the long-range contribution is damped and
        the model falls back on local evidence.

            d_t = ||long_t - local_t||_1 / (||long_t||_1 + ||local_t||_1 + eps)
            w_t = 1 - sigmoid(gamma) * d_t
        """
        return uncertainty_weights(long, local, self.uncertainty_logit,
                                   self.gate_eps)

    # -- L2 evidence gate, streaming form -------------------------------------
    def _evidence_step(self, u: torch.Tensor, state: LayerStreamState) -> torch.Tensor:
        """Streaming counterpart of :meth:`_evidence_weights` for one position.

        The running prefix energy sum lives in the (constant-size) stream
        state, so the streamed weight matches the parallel one.
        """
        w, state.energy_sum = evidence_weight_step(
            u, state.energy_sum, state.length,
            self.evidence_gain, self.evidence_pos_gain, self.evidence_bias)
        return w

    # -- streaming ------------------------------------------------------------
    def init_state(self, batch: int, device, dtype) -> LayerStreamState:
        M, D = self.num_bases, self.dim
        # 递推状态恒按 recurrent_dtype 分配：低精度计算时退化为 fp32，避免 bf16/fp16
        # 下 IIR 振荡递推逐 token 累积舍入误差。conv_buf 是短 FIR 历史、energy 是
        # fp32 标量，均无需升精度。
        rd = recurrent_dtype(dtype)
        y1 = torch.zeros(batch, M, D, device=device, dtype=rd)
        y2 = torch.zeros(batch, M, D, device=device, dtype=rd)
        buf = None
        if self.local_kernel > 1:
            buf = torch.zeros(batch, self.local_kernel - 1, D, device=device, dtype=dtype)
        # 证据门的 O(1) 运行能量和只在门开启时分配（num_bytes 对 None 自动跳过，
        # 流式字节账目与实际状态严格一致）。
        energy_sum = None
        if self.evidence_gate:
            energy_sum = torch.zeros(batch, 1, device=device, dtype=torch.float32)
        return LayerStreamState(y1=y1, y2=y2, conv_buf=buf, length=0,
                                 energy_sum=energy_sum)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: LayerStreamState) -> Tuple[torch.Tensor, LayerStreamState]:
        """One token, O(1) state, O(1) compute in the sequence length.

        Args:
            x_t: (B, 1, D) or (B, D) input for the next position.
            state: persistent state from :meth:`init_state` or a previous call.
        Returns:
            (y_t, new_state) with ``y_t`` of shape (B, 1, D).
        """
        squeeze = x_t.dim() == 2
        if squeeze:
            x_t = x_t.unsqueeze(1)
        # 融合热路径：在 fp32/bf16/fp16 + 扩展就绪 + 3D 输入（真实流式路径）时下沉到
        # C++/CUDA。非 fp32 精度由 ops.mixer_stream 在 Python 层归一化为 fp32 后再计算
        # （IIR 递推恒在 fp32 执行、输出 cast 回原精度），因此 bf16 的推理/服务
        # 也能用上融合算子；其余情况（扩展未构建、(B,D) 直调）回退纯 PyTorch。
        if (self.use_mixer_stream and not squeeze
                and x_t.dtype in (torch.float32, torch.bfloat16, torch.float16)):
            from ...ops import is_available, mixer_stream_step
            if is_available():
                return mixer_stream_step(self, x_t, state)
        B, one, D = x_t.shape
        M = self.num_bases
        assert one == 1, "step() consumes exactly one position"

        # 递推 dtype：状态已按 recurrent_dtype 分配（低精度计算时为 fp32）。
        # 递推在 rd 中执行，输出 nxt 再 cast 回计算精度与局部路径混合——
        # 既保证 O(1) 状态数值可信，又维持 I/O 与计算精度一致。fp32 计算时 rd=fp32。
        rd = state.y1.dtype

        h = self.norm(x_t)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        rho, theta = self.oscillator_params()
        a = (2.0 * rho * torch.cos(theta)).to(rd)                       # (M,)
        b = (-(rho ** 2)).to(rd)
        c = torch.sin(theta).to(rd)

        um = u.squeeze(1).unsqueeze(1).to(rd)                           # (B,1,D) in rd
        y_new = a.view(1, M, 1) * state.y1 + b.view(1, M, 1) * state.y2 + c.view(1, M, 1) * um
        state.y2, state.y1 = state.y1, y_new

        if self.num_kernels > 1:
            # MoK readout：状态递推与核无关（共享 psi），故 y1/y2 形状不变，
            # 多核只体现在 readout 的 K 路 einsum + 逐位置路由加权。
            nxt_k = torch.einsum("bmd,kmd->bkd", y_new, self._coefs_eff(rd))     # (B,K,D)
            g_t = self._kernel_route_weights(h)                                  # (B,1,K)
            nxt = torch.einsum("bkd,bk->bd", nxt_k, g_t.squeeze(1)).unsqueeze(1)
        else:
            nxt = torch.einsum("bmd,md->bd", y_new, self._coef_eff(rd)).unsqueeze(1)
        nxt = nxt.to(u.dtype)                                          # 回到计算精度再混合
        loc, state.conv_buf = depthwise_causal_conv(u, self.local_conv.weight, state.conv_buf)

        if self.evidence_gate:
            # 门控权重在 fp32 态里算（energy_sum 是 fp32 递推量），乘回 nxt 前
            # 必须 cast 回计算精度——否则低精度路径被提升回 fp32，out_proj 炸
            # （bf16 流式的结构性约束：递推态 fp32、I/O 与计算 bf16）
            nxt = nxt * self._evidence_step(u, state).to(nxt.dtype)
        if self.uncertainty_gate:
            nxt = nxt * self._uncertainty_weights(nxt, loc).to(nxt.dtype)

        inst = self._instant_term(u)
        if self.path_gate:
            w = torch.softmax(self.path_gate_proj(h), dim=-1)   # (B,1,3)
            y = (w[..., 0:1] * nxt + w[..., 1:2] * loc
                 + w[..., 2:3] * inst)
        else:
            y = nxt + loc + inst

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        state.length += 1
        y_t = x_t + self.out_scale * out
        return (y_t.squeeze(1) if squeeze else y_t), state

    # -- introspection --------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, num_bases={self.num_bases}, "
                f"local_kernel={self.local_kernel}, content_gate={self.content_gate}, "
                f"out_gate={self.out_gate}, evidence_gate={self.evidence_gate}, "
                f"uncertainty_gate={self.uncertainty_gate}, "
                f"path_gate={self.path_gate}, inst_nonlin={self.inst_nonlin}, "
                f"num_kernels={self.num_kernels}, init_mode={self.init_mode!r}, "
                f"kernel_norm={self.kernel_norm!r}")

    def streaming_state_bytes(self, batch: int, dtype=torch.float32) -> int:
        """Bytes of constant streaming state, independent of sequence length.

        递推状态（y1/y2）按 recurrent_dtype 计字节：低精度计算时退化为 fp32，
        因此按 fp32 计；conv_buf 为短 FIR 历史、按计算精度计。

        L2 证据门开启时，每条序列额外分配一个 fp32 前缀能量标量；
        关闭时该状态不存在，账目也不应虚报这部分常驻内存。
        """
        itemsize = torch.empty(0, dtype=dtype).element_size()
        rd_itemsize = torch.empty(0, dtype=recurrent_dtype(dtype)).element_size()
        n = 2 * self.num_bases * self.dim * batch * rd_itemsize
        n += max(self.local_kernel - 1, 0) * self.dim * batch * itemsize
        if self.evidence_gate:
            n += batch * torch.empty(0, dtype=torch.float32).element_size()
        return n

    def effective_kernel_rank(self, length: int) -> int:
        """Numerical rank of the kernel bank; shows the expressivity of the basis."""
        with torch.no_grad():
            # MoK：对 K 个核分别求秩，取最大（保守上界）
            if self.num_kernels > 1:
                Ks = self.build_kernels(length)
                return max(torch.linalg.matrix_rank(Ks[k]).item()
                           for k in range(self.num_kernels))
            K = self.build_kernel(length)
            return int(torch.linalg.matrix_rank(K.t().float()))


# ----------------------------------------------------------------------------
# SGC-R: routed spectral-basis variant (MiTA-style shared expert + routed
# experts, no token-pair scoring)
# ----------------------------------------------------------------------------
class RoutedSpectralGlobalConv(SpectralGlobalConv):
    """SGC-R：谱专家路由变体 —— 在不改变 SGC 时序算子身份的前提下，吸收
    MiTA（arXiv:2602.01219, NeurIPS 2026）的核心主张：**shared expert +
    deformable routed experts + reusable top-k**。

    定位（MiTA 快权重分类法）
    ------------------------
    SGC 的 FFT 全局通路本身就是 MiTA 意义下的 "narrow shared expert"：
    把 N 宽的输入诱导快权重 MLP 压缩到 M 个振荡基（低秩时间基）。SGC 缺的
    是 **routing 分支**。本类把它补齐：M 个基按频率连续分成 G 组"谱专家"，
    逐 token 路由器 top-k 选择注入哪几组。deformable = 激活基组随内容变化
    （不是刚性分块）；reusable top-k = 每步只做一次路由决策、被切换掉的
    专家记忆仍在 O(1) 状态里齐次衰减，不丢失。

    与注意力的边界（purity 契约的语义依据）
    --------------------------------------
    * 路由是 **逐 token → 固定专家集合** 的 MoE 选择（Linear + top-k），
      不存在 token↔token 成对打分、softmax 相似度矩阵或任何 KV 载体；
    * 未被路由的专家不接收输入注入，但递推照常推进 —— 门控只作用于
      **输入注入端**，不作用于状态演化；
    * O(1) 流式状态的形状与字节与 ``SpectralGlobalConv`` **逐字节相同**
      （(B, M, D)×2 + conv_buf + 证据门标量）。

    FFT/IIR 对偶性保持的证明要点
    ----------------------------
    路由权重 ``w_g(t)`` 是**当前 token 的因果函数**（``route_proj(h_t)``），
    与序列其余位置无关 —— 与 ``content_gate`` / ``path_gate`` 同构。专家 g
    的有效输入是 ``u_g(t) = w_g(t)·u(t)``，其 IIR 解为 齐次解 +
    ``Σ_{s≤t} c·psi(t-s)·w_g(s)·u(s)``，即以 ``u_g`` 为输入的标准谱卷积。
    因此并行形式（对各 ``u_g`` 做因果 FFT 卷积后求和）与流式形式（把
    ``w_g(t)·u(t)`` 注入第 g 组递推）是同一算子；块间延续沿用父类的
    ``_free_response``（纯齐次，不依赖 u）与 per-group 的 ``_tail_state``。
    由 ``tests/test_sgc_routed.py`` 按 ``test_mixer_math`` 同等强度守护。

    梯度与负载均衡
    --------------
    训练时 straight-through：前向 hard top-k、反向梯度流经 dense softmax
    （MoE 惯例）。另暴露 switch 风格负载均衡 aux（:meth:`router_aux_loss`），
    **由训练侧显式取用** —— model/train 主路径不自动加，保持现有训练行为
    逐位不变。

    显存口径
    --------
    训练核按组循环"即建即弃"（峰值 = 1×(T, D) 核 + G 份 ``u_g`` 激活），
    不做 G 组核批量物化；``train_window_bytes`` 的 O(L·D) 核预算口径对
    本类**不变**，G 倍只出现在激活内存（属于激活预算，不属核预算）。
    """

    def __init__(self, *args, route_groups: int = 4, route_topk: int = 2,
                 route_aux: float = 1.0, **kwargs) -> None:
        # 融合 C++/CUDA 流式算子按父类属性布局编写，未实现路由注入；
        # 显式关闭而不是静默给出错误结果。
        kwargs["use_mixer_stream"] = False
        super().__init__(*args, **kwargs)
        if self.num_kernels > 1:
            raise ValueError(
                "RoutedSpectralGlobalConv 使用自己的分组谱专家路由，"
                "与 MoK(num_kernels>1) 的逐位置核路由是两套不同的混入机制；"
                "叠加会静默忽略 MoK 而给出不一致结果。请二选一。")
        M = self.num_bases
        if route_groups < 1 or M % route_groups:
            raise ValueError(
                f"route_groups must divide num_bases (M={M}, G={route_groups})")
        if not (1 <= route_topk <= route_groups):
            raise ValueError(
                f"route_topk must be in [1, {route_groups}], got {route_topk}")
        self.route_groups = route_groups
        self.route_topk = route_topk
        self.route_aux = float(route_aux)
        per = M // route_groups
        #: 基 → 组 的连续频率段划分（rho 从快到慢，组 = 时间尺度特化先验）
        self._group_slices = [
            (g, slice(g * per, (g + 1) * per)) for g in range(route_groups)]
        self._group_of_base = torch.arange(M) // per
        # 路由器：D → G logits。小随机初始化打破并列（zero init 会让 top-k
        # 恒选中同一批组，未激活专家成为梯度死区）。
        self.route_proj = nn.Linear(self.dim, route_groups, bias=True)
        with torch.no_grad():
            self.route_proj.weight.normal_(0, 0.01)
            self.route_proj.bias.zero_()
        #: 最近一次前向的 switch 负载均衡损失（未乘 :attr:`route_aux` 权重）
        self.last_route_aux: Optional[torch.Tensor] = None

    # -- routing --------------------------------------------------------------
    def _route_weights(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逐位置路由：返回 ``(w, soft)``，均为 (B, L, G)。

        ``w`` 是注入权重：top-k 集合内为 renormalized softmax 值、集合外为 0。
        训练且有梯度时用 straight-through（前向数值 = hard，反向梯度 =
        dense softmax）；eval / no_grad（含流式 ``step``）恒为纯 hard ——
        保证"训练前向"与"O(1) 流式推理"在 fp32 精度内是同一决策。
        """
        logits = self.route_proj(h)
        soft = torch.softmax(logits, dim=-1)
        k = min(self.route_topk, soft.shape[-1])
        topv, topi = torch.topk(soft, k, dim=-1)
        topn = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        hard = torch.zeros_like(soft).scatter(-1, topi, topn)
        if self.training and torch.is_grad_enabled():
            return soft + (hard - soft).detach(), soft
        return hard, soft

    def _switch_aux(self, w: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        """switch 风格负载均衡：``aux = G · Σ_g f_g · P_g``。

        ``f_g`` = hard 分到的权重份额（w 逐位置均值），``P_g`` = dense softmax
        概率均值。均衡时 f = P = 1/G，aux = 1；失衡时 aux > 1。
        """
        G = self.route_groups
        f = w.mean(dim=(0, 1))
        p = soft.mean(dim=(0, 1))
        return G * (f * p).sum()

    def router_aux_loss(self) -> Optional[torch.Tensor]:
        """训练侧负载均衡辅助损失（``route_aux`` 权重已乘）；未路由时 None。"""
        if self.last_route_aux is None:
            return None
        return self.route_aux * self.last_route_aux

    # -- kernels --------------------------------------------------------------
    def _oscillator_psi(self, length: int, device=None, dtype=None) -> torch.Tensor:
        """全基时间基矩阵 psi，形状 (M, L)：``psi_m[tau] = rho^tau sin((tau+1)theta)``。"""
        rho, theta = self.oscillator_params()
        device = device if device is not None else self.coef.device
        dtype = dtype if dtype is not None else self.coef.dtype
        tau = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)   # (1,L)
        log_rho = torch.log(rho.to(dtype)).unsqueeze(1)                       # (M,1)
        return torch.exp(log_rho * tau) \
            * torch.sin((tau + 1.0) * theta.to(dtype).unsqueeze(1))

    def _group_kernel(self, sel: slice, length: int, device, dtype) -> torch.Tensor:
        """第 ``sel`` 组基的核 bank (L, D)：``coef[sel].T @ psi[sel]``。"""
        psi = self._oscillator_psi(length, device=device, dtype=dtype)
        return (self._coef_eff(dtype)[sel].t() @ psi[sel]).t().contiguous()

    def _grouped_long(self, u: torch.Tensor, w: torch.Tensor,
                      length: int) -> torch.Tensor:
        """长程通路：Σ_g fft_causal_conv(u·w_g, K_g)。核按组即建即弃。"""
        long = None
        for g, sel in self._group_slices:
            u_g = u * w[..., g: g + 1]
            k_g = self._group_kernel(sel, length, device=u.device, dtype=u.dtype)
            part = fft_causal_conv(u_g, k_g, self.fft_dtype)
            long = part if long is None else long + part
        return long

    # -- forward (parallel / training form) ------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, L, D = x.shape
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        w, soft = self._route_weights(h)
        long = self._grouped_long(u, w, L)
        local = self._local_conv_full(u)

        if self.evidence_gate:
            long = long * self._evidence_weights(u)
        if self.uncertainty_gate:
            long = long * self._uncertainty_weights(long, local)

        inst = self._instant_term(u)
        if self.path_gate:
            pw = torch.softmax(self.path_gate_proj(h), dim=-1)   # (B,L,3)
            y = (pw[..., 0:1] * long + pw[..., 1:2] * local
                 + pw[..., 2:3] * inst)
        else:
            y = long + local + inst

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        self.last_route_aux = self._switch_aux(w, soft)
        return residual + self.out_scale * out

    # -- chunked / stateful parallel form --------------------------------------
    def forward_chunk(self, x: torch.Tensor,
                      state: Optional[LayerStreamState] = None,
                      *, update_state: bool = True
                      ) -> Tuple[torch.Tensor, Optional[LayerStreamState]]:
        """带跨块状态的并行前向：与 :meth:`forward` 逐位一致（fp32 误差内），
        梯度语义为截断 BPTT（与父类 :meth:`SpectralGlobalConv.forward_chunk`
        相同口径）。长程通路按组卷积；块间齐次延续不依赖路由决策（路由门只
        作用于输入注入端）。"""
        residual = x
        B, T, D = x.shape
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        w, soft = self._route_weights(h)
        long = self._grouped_long(u, w, T)
        if state is not None and state.length > 0:
            long = long + self._free_response(u, state)

        buf = state.conv_buf if state is not None else None
        local, new_buf = depthwise_causal_conv(u, self.local_conv.weight, buf)

        new_energy = None
        if self.evidence_gate:
            ew, new_energy = evidence_weights_chunk(
                u,
                state.energy_sum if state is not None else None,
                state.length if state is not None else 0,
                self.evidence_gain, self.evidence_pos_gain, self.evidence_bias)
            long = long * ew.to(long.dtype)
        if self.uncertainty_gate:
            long = long * self._uncertainty_weights(long, local)

        inst = self._instant_term(u)
        if self.path_gate:
            pw = torch.softmax(self.path_gate_proj(h), dim=-1)
            y = (pw[..., 0:1] * long + pw[..., 1:2] * local
                 + pw[..., 2:3] * inst)
        else:
            y = long + local + inst

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        self.last_route_aux = self._switch_aux(w, soft)
        y = residual + self.out_scale * out
        if state is None or not update_state:
            return y, state

        with torch.no_grad():
            y1, y2 = self._tail_state(u.detach(), w.detach(), state)
        new_state = LayerStreamState(
            y1=y1, y2=y2,
            conv_buf=(new_buf.detach() if new_buf is not None
                      and new_buf.shape[1] > 0 else None),
            length=state.length + T,
            energy_sum=(new_energy.detach() if new_energy is not None else None))
        return y, new_state

    def _tail_state(self, u: torch.Tensor, w: torch.Tensor,
                    state: LayerStreamState) -> Tuple[torch.Tensor, torch.Tensor]:
        """块末 ``(y1, y2)`` 的 per-group 版本。

        与父类唯一的不同：第 m 基的历史贡献用其所属组的加权输入
        ``u_g = w_g·u`` 累计（父类用共享 u）。齐次部分 ``_h_at`` 与输入
        无关，逐字复用父类数学（振荡支 + 重根极限支）。
        """
        B, T, D = u.shape
        rd = state.y1.dtype
        M = self.num_bases
        rho, theta, a_eff, b_eff, small = self._homogeneous(state, rd)
        tau = torch.arange(T, device=u.device, dtype=rd)
        log_rho = torch.log(rho.clamp(min=1e-12)).view(-1, 1)
        psi = torch.exp(log_rho * tau.view(1, -1)) \
            * torch.sin((tau + 1.0).view(1, -1) * theta.view(-1, 1))   # (M, T)
        flip1 = psi.flip(-1)                       # psi[T-1-s]
        if T >= 2:
            flip2 = torch.cat([flip1[:, 1:],
                               torch.zeros(M, 1, device=u.device, dtype=rd)], dim=1)
        else:
            flip2 = torch.zeros_like(flip1)
        u_rd = u.to(rd)
        w_rd = w.to(rd)
        z1 = torch.zeros(B, M, D, device=u.device, dtype=rd)
        z2 = torch.zeros(B, M, D, device=u.device, dtype=rd)
        for g, sel in self._group_slices:
            u_g = u_rd * w_rd[..., g: g + 1]
            z1[:, sel, :] = torch.einsum("ms,bsd->bmd", flip1[sel], u_g)
            z2[:, sel, :] = torch.einsum("ms,bsd->bmd", flip2[sel], u_g)

        def _h_at(t: int) -> torch.Tensor:
            # 振荡支：h[t] = rho^(t+1) (A cos((t+1)θ) + B sin((t+1)θ))
            ph = float(t + 1) * theta.view(-1, 1)
            decay = torch.exp(log_rho * float(t + 1)).view(1, -1, 1)
            osc = decay * (a_eff * torch.cos(ph).view(1, -1, 1)
                           + b_eff * torch.sin(ph).view(1, -1, 1))
            # 重根支：h[t] = (C + D t) rho^t
            lin = (a_eff + b_eff * float(t)) \
                * torch.exp(log_rho * float(t)).view(1, -1, 1)
            return torch.where(small, lin, osc)

        return z1 + _h_at(T - 1), z2 + _h_at(T - 2)

    # -- streaming ------------------------------------------------------------
    @torch.no_grad()
    def step(self, x_t: torch.Tensor,
             state: LayerStreamState) -> Tuple[torch.Tensor, LayerStreamState]:
        """One token, O(1) state —— 与父类唯一的差别是输入注入端多一个
        per-group 路由门 ``w_g(t)``；未激活组的递推照常推进（纯齐次衰减），
        其历史记忆仍通过 einsum 全基输出贡献给当前位（reusable top-k 的
        SGC 对应物：切换专家不丢记忆）。"""
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

        w, _soft = self._route_weights(h)                     # (B, 1, G)
        # 每个基的注入权重 = 所属组的路由权重；(B, G) -> (B, M)
        w_base = w[:, 0, :].index_select(
            -1, self._group_of_base.to(x_t.device))           # (B, M)

        rho, theta = self.oscillator_params()
        a = (2.0 * rho * torch.cos(theta)).to(rd)             # (M,)
        b = (-(rho ** 2)).to(rd)
        c = torch.sin(theta).to(rd)

        # 注入项按基带各自组的权重：c_m · w_{g(m)} · u_d，形状 (B, M, D)
        u_inj = u.squeeze(1).to(rd).unsqueeze(1) \
            * w_base.to(rd).unsqueeze(-1)                    # (B,1,D)*(B,M,1)
        y_new = a.view(1, M, 1) * state.y1 + b.view(1, M, 1) * state.y2 \
            + c.view(1, M, 1) * u_inj
        state.y2, state.y1 = state.y1, y_new

        nxt = torch.einsum("bmd,md->bd", y_new, self._coef_eff(rd)).unsqueeze(1)
        nxt = nxt.to(u.dtype)
        loc, state.conv_buf = depthwise_causal_conv(
            u, self.local_conv.weight, state.conv_buf)

        if self.evidence_gate:
            nxt = nxt * self._evidence_step(u, state).to(nxt.dtype)
        if self.uncertainty_gate:
            nxt = nxt * self._uncertainty_weights(nxt, loc).to(nxt.dtype)

        inst = self._instant_term(u)
        if self.path_gate:
            pw = torch.softmax(self.path_gate_proj(h), dim=-1)   # (B,1,3)
            y = (pw[..., 0:1] * nxt + pw[..., 1:2] * loc
                 + pw[..., 2:3] * inst)
        else:
            y = nxt + loc + inst

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        state.length += 1
        y_t = x_t + self.out_scale * out
        return (y_t.squeeze(1) if squeeze else y_t), state

    # -- introspection --------------------------------------------------------
    def extra_repr(self) -> str:
        return (super().extra_repr()
                + f", route_groups={self.route_groups}, route_topk={self.route_topk}")



# ----------------------------------------------------------------------------
# Initialisation helpers
# ----------------------------------------------------------------------------
def _assert_no_common_zero(theta: torch.Tensor, num_bases: int) -> None:
    """theta 网格公共零点的**全 τ 值级**守护（fp64 提升后检查）。

    判据（对齐命题 5 的比值口径）：对每个 τ 计算
    ``r_τ = ‖sin((τ+1)·θ)‖₂ / √M``。因为 ρ_m^τ > 0 恒成立，核基向量
    Ψ[τ] = 0 ⟺ sin 因子向量为零——``min_τ ‖Ψ[τ]‖/‖ρ^τ‖`` 的零点结构与
    ``min_τ ‖sin((τ+1)θ)‖`` 完全一致（ρ^τ 只是逐基衰减归一化）。

    扫描 **全部** τ ∈ [0, 4M+16)：legacy 均匀网格 θ_m = π(m+0.5)/M 的真实
    公共零点在 **τ = 2Mk−1**（首个 τ=2M−1，周期 2M）——旧实现只扫
    τ+1 = j(M+1)，与真实零点位置不对齐（仅 gcd 巧合时部分命中，首个零点
    τ=2M−1 永远扫不到），非模式化网格在其它 τ 处的近零会静默漏检。

    口径边界：fp32 参数化误差（τ·ulp(θ) ~ 1.5e-3）会把 legacy 网格的
    **代数**零点在值级抬到 ~1.5e-3，值级扫描看不到——它由
    ``_init_oscillator_grid`` 的 legacy 分支公式级判定直接拒绝（双保险）。
    值级守护负责的是公式覆盖不到的情形：非模式化网格（logfreq、训练后
    漂移、将来新增的 init_mode）数值上真正撞出的近零。
    """
    M = num_bases
    T = 4 * M + 16
    theta_d = theta.to(torch.float64).reshape(1, M)
    tau = torch.arange(1, T + 1, dtype=torch.float64).reshape(T, 1)
    sins = torch.sin(tau * theta_d)                       # (T, M)
    r = sins.norm(dim=1) / math.sqrt(M)                   # ‖sin‖₂/√M，逐 τ
    r_min, idx = r.min(dim=0)
    if r_min.item() < 1e-6:
        raise ValueError(
            f"theta grid has a (near-)common zero at tau={int(idx.item())} "
            f"(min_tau ||Psi[tau]||/||rho^tau|| = {r_min.item():.3e} < 1e-6): "
            f"the kernel is (nearly) identically zero there for every coef "
            f"(unfixable time blind spot). Use init_mode='logfreq' or a "
            f"different grid.")


def _init_oscillator_grid(num_bases: int, decay_max: float,
                          mode: str = "logfreq",
                          theta_floor: float = 0.3):
    """Initialise the (rho, theta) grid of the oscillator bank.

    ``mode="logfreq"``（默认）：theta 对数分布，覆盖
    ``[pi/decay_max, pi/2]`` —— 让"振荡周期"与"衰减窗口"在尺度上配对：
    最慢的基周期≈decay_max（衰减窗口内完整振荡一次），最快的基贴近
    每两个 token 翻转一次（局部细节）。无公共零点（theta 无均匀网格的
    代数结构），rho 网格不变。

    ``mode="legacy"``：decay 时间常数几何分布，theta **均匀**分布
    ——短周期（高频）基的密度与长周期一样，长上下文任务下高频段欠采样；
    且存在公共零点（**首个 τ = 2M−1，周期 2M**，即 τ = 2Mk−1 全体）
    ——legacy 分支做公式级判定，显式选择时命中即拒绝构造。

    ``mode="linear"``（推荐用于新训练）：theta **线性均匀**分布
    ``[theta_floor, pi - theta_floor]``，默认 ``theta_floor=0.3``。

    依据（独立复现，见 docs/frequency_grid_fix.md）：周期/连续信号轴上，
    logfreq 的低频端 θ_min=pi/decay_max≈3.8e-4 使相邻基近似共线
    （M=32 时数值秩 24/32、cond 1.92e16）；线性均匀网格把 cond 降到
    1.04、秩满 32/32。同一模型同一参数量下收敛速度显著提升，且
    **不增加任何参数、FLOPs 或流式状态字节**。

    ★ 归因修正：消融显示换成**纯随机正交基**同样能赢，而振荡基反而较慢
      —— 起作用的是**基的正交性/可分性**，不是"振荡归纳偏置"。正是
      "把最慢基周期配到 decay_max"这个设计把 θ 推向 0，制造了塌缩。

    ★ 边界：该改动只在周期/连续信号轴兑现；key→value 绑定轴
      （MQAR 类任务）无收益甚至回退 —— 共享叠加状态不可分离是独立的
      结构限制，正交网格救不了。

    Returns (rho0, theta0) both of shape (M,), with rho0 in (0,1) and
    theta0 in (0,pi).
    """
    lo, hi = math.log(1.0), math.log(max(decay_max, 2.0))
    lam = torch.exp(torch.linspace(lo, hi, num_bases))
    rho0 = torch.exp(-1.0 / lam)
    if mode == "legacy":
        theta0 = math.pi * (torch.arange(num_bases, dtype=torch.float32) + 0.5) / num_bases
        # 公式级代数判定：theta_m = pi(m+0.5)/M 型均匀网格在
        # j = 2M/gcd(M+1,2)（M 偶即 j=2M，M 奇即 j=M）处全体基同时为零
        # ——sin(整数·pi)=0 对任何 coef 都成立，是修不了的时间盲点。
        # 不能用 fp32 值级检查：tau·ulp(theta) ~ 1.5e-3（M=64, j=2M）
        # 的参数化误差会把代数零点掩盖成"近零"。
        j0 = 2 * num_bases // math.gcd(num_bases + 1, 2)
        theta_exact = math.pi * (torch.arange(num_bases, dtype=torch.float64)
                                 + 0.5) / num_bases
        if torch.sin(float(j0 * (num_bases + 1)) * theta_exact) \
                .abs().max().item() < 1e-6:
            raise ValueError(
                f"legacy uniform theta grid has an algebraic common zero "
                f"(first at tau=2M-1, period 2M; hit here at "
                f"tau={j0 * (num_bases + 1) - 1}, j={j0}): the kernel is "
                f"identically zero there for every coef (unfixable time "
                f"blind spot). Use init_mode='logfreq'.")
    elif mode == "logfreq":
        theta_lo = math.pi / max(decay_max, 2.0)
        theta_hi = math.pi / 2.0
        # 下端点即 pi/decay_max（最慢基的周期≈decay_max，衰减窗口内恰一次
        # 完整振荡）；上端点内缩到 0.9·pi/2，避免贴着 Nyquist 的对称退化
        theta0 = torch.exp(torch.linspace(
            math.log(theta_lo), math.log(theta_hi * 0.9), num_bases,
            dtype=torch.float32))
    elif mode == "hybrid":
        # **R2（0.1-α）：混合网格——均匀分辨力 + 打破等差零点结构**。
        # 动机（2026-10-31 复现报告实测）：logfreq 的低频端绝对间隔趋零
        # （decay=8192 时 θ_lo≈3.8e-4），有限 τ 窗口内相邻低频列近共线——
        # 实测 Ψ 秩缺失：M=16 rank 14/16、M=64 rank 39/64、M=128 rank
        # 63/128，条件数 1e14~1e19（与命题 5 的 |sinθ|→0 病态同源）。
        # hybrid：在 [θ_lo, θ_hi] 上**线性均匀** (m+0.5) 栅格（分辨力 ≈
        # DFT 频点栅格，低频端绝对间隔 = (θ_hi−θ_lo)/M，比 logfreq 大
        # 数量级）+ **黄金比例小扰动**（打破等差的有理比结构——legacy
        # 均匀网格的公共零点正源于此）；最慢基 θ_lo（decay_max 配对）
        # 与最快基均保留。
        theta_lo_ = math.pi / max(decay_max, 2.0)
        theta_hi_ = math.pi / 2.0 * 0.9
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        # **含端点栅格**：m=0 精确落在 θ_lo（最慢基 = π/decay_max 的配对
        # 设计保留），m=M-1 到 θ_hi；(M-1) 等分 + 黄金比扰动打破等差。
        m_idx = torch.arange(num_bases, dtype=torch.float64)
        span = theta_hi_ - theta_lo_
        step = span / max(num_bases - 1, 1)
        base = theta_lo_ + m_idx * step
        # 黄金比扰动幅度乘 sin 窗（首尾基锚定：m=0 精确 θ_lo、m=M-1 精确
        # θ_hi——decay_max 配对的最慢基与 Nyquist 上界都不被扰动推走）
        taper = torch.sin(m_idx / max(num_bases - 1, 1) * math.pi)
        jitter = (torch.tensor(
            [((m_idx[i] + golden) % 1.0) - 0.5 for i in range(num_bases)],
            dtype=torch.float64) * step * 0.5 * taper)
        theta0 = (base + jitter).clamp(theta_lo_, theta_hi_).to(torch.float32)
    elif mode == "linear":
        # 线性均匀 θ ∈ [f, π−f]：把低频端从 π/decay_max 抬到 f，
        # 同时放开上界到 π−f（覆盖整个 (0,π) 而非只到 Nyquist 一半以下）。
        # (m+0.5)/M 栅格避开端点代数结构；_assert_no_common_zero 值级守护。
        f = float(theta_floor)
        u = (torch.arange(num_bases, dtype=torch.float64) + 0.5) / num_bases
        theta0 = (f + (math.pi - 2.0 * f) * u).to(torch.float32)
    else:
        raise ValueError(f"unknown init_mode: {mode!r} "
                         f"(legacy | logfreq | hybrid | linear)")
    _assert_no_common_zero(theta0, num_bases)
    if mode in ("hybrid", "linear"):
        # hybrid / linear 必须过数值质量门（logfreq 的秩缺失是它们存在的理由）
        _assert_numerical_rank(theta0, num_bases)
    elif mode == "logfreq":
        # P0-3（上游待修清单）：logfreq 是历史默认，不能改成硬拒绝（会破坏
        # 既有 checkpoint 的可复现性），但也不能静默放行一个 cond~1e16、
        # 数值秩 24/32 的字典。这里留下可审计的警告，与构造期质量门口径一致。
        rank, cond = _grid_rank_condition(theta0, L=max(512, 4 * num_bases))
        if rank < num_bases or cond > 1e6:
            warnings.warn(
                f"init_mode='logfreq': damped-oscillator grid is "
                f"rank-deficient (rank={rank}/{num_bases}, cond={cond:.2e}); "
                f"consider init_mode='linear' or 'hybrid' for new training.",
                RuntimeWarning, stacklevel=2)
    return rho0, theta0


def _grid_rank_condition(theta: torch.Tensor, L: int = 512) -> Tuple[int, float]:
    """Ψ 矩阵（sin 核基，L 窗口）的数值秩与条件数（fp64，诊断/守护用）.

    Ψ[τ, m] = sin((τ+1)·θ_m)（ρ^τ 衰减对θ-可分辨性的贡献为正，此处置 1
    以隔离 θ 的贡献——最保守口径）。
    """
    theta_d = theta.to(torch.float64).reshape(1, -1)
    tau = torch.arange(1, L + 1, dtype=torch.float64).reshape(-1, 1)
    psi = torch.sin(tau * theta_d).T                            # (M, L)
    s = torch.linalg.svdvals(psi)
    rank = int((s > s.max() * 1e-10).sum())
    cond = float(s.max() / s.min().clamp_min(1e-300))
    return rank, cond


def _assert_numerical_rank(theta: torch.Tensor, num_bases: int,
                           L: Optional[int] = None,
                           max_cond: float = 1e6) -> None:
    """构造期数值秩守护（R2）：基矩阵条件数超限即拒绝.

    logfreq 的实测病态（M=16 rank 14/16、M=128 rank 63/128，cond ≈1e19）
    与命题 5 的 |sinθ|→0 病态同源——低频基在有限窗口内互相不可分辨，
    训练后 coef 在近零奇异方向上漂移无约束。hybrid 网格必须过此门。

    L 自适应：``max(512, 4·M)``——M 个频率需要 ≥2M 的采样窗（Nyquist），
    固定 L=512 会对 M≥256 误报（那是窗口不够，不是网格病态）。
    """
    L_eff = max(L or 0, 512, 4 * num_bases)
    rank, cond = _grid_rank_condition(theta, L=L_eff)
    if rank < num_bases or cond > max_cond:
        raise ValueError(
            f"theta grid is numerically rank-deficient: rank={rank}/{num_bases}, "
            f"cond={cond:.3e} > {max_cond:.0e} (L={L_eff}). "
            f"Use init_mode='hybrid' (uniform+golden-jitter), which keeps "
            f"full rank with bounded condition number.")


def reinit_oscillator_grid(mixer: nn.Module, decay_max: float) -> None:
    """按新 ``decay_max`` 重置 mixer 的 (rho, theta) 初始网格（R3 分层多尺度）.

    仅支持暴露 ``num_bases`` / ``init_mode`` / ``raw_rho`` / ``raw_theta`` 的
    SGC 家族 mixer（其它 mixer 静默跳过，由调用方 hasattr 守卫）。
    logspace 参数化的 ``raw_rho_log`` 同步换算：rho = exp(-softplus(x))
    ⟺ x = log((1-rho)/rho)。
    """
    if not all(hasattr(mixer, a) for a in
               ("num_bases", "init_mode", "raw_rho", "raw_theta")):
        return
    rho0, theta0 = _init_oscillator_grid(mixer.num_bases, decay_max,
                                         mode=getattr(mixer, "init_mode",
                                                      "logfreq"))
    with torch.no_grad():
        mixer.raw_rho.copy_(_logit(rho0))
        mixer.raw_theta.copy_(_logit(theta0 / math.pi))
        if hasattr(mixer, "raw_rho_log"):
            rho_c = rho0.clamp(1e-8, 1.0 - 2.0 ** -23)
            mixer.raw_rho_log.copy_(torch.log((1.0 - rho_c) / rho_c))
    mixer._theta_floor = theta_clamp_floor(decay_max)


def _logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)
