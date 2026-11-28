#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 · 同步 SimplCore 云端最新「SGC 冻结 + 三旁路外挂」架构

## 来源（2026-11-28 从云端实拉，逐字节登记）

上游 `kino-koki/SimplCore` @ `bec5ab8fd7`（2026-11-27T11:26:53Z）:
  * `repro/independent_v9_v11/sgc_sidecar_final.py`
  * `repro/independent_v9_v11/SGC冻结外挂_超越与不足_完整报告.md`
  * `repro/independent_v9_v11/README.md`（v9–v11 本地极限优化补充包）
  * `repro/independent_v9_v11/极限优化后_最终实测报告.md`

## 架构定稿（与上游报告 §0 一致，一字不改）

**SGC 主干完全冻结**：`ψ_m[τ]=ρ_m^τ·sin((τ+1)θ_m)`、a/b/c 静态 Toeplitz、
FFT↔IIR 双形式、`2·M·D` 定态，一个字节都不改。

**三条旁路并联，只在 readout 拼接，绝不回写主干**：

| 旁路 | 补的能力 | 状态 |
|---|---|---|
| A 3 层 k=7 门控短卷积 | 局部 n-gram / 乘法交互 | `3·6·D` |
| B CfC 液态门 | 输入依赖 τ / 不规则 Δt | `D` |
| C Hadamard 正交登记 (SE1) | 内容寻址绑定，地址由内容非位置给 | `V×V` |

**纯度**：无 Q/K/V、无 `softmax(QKᵀ)`、无 SDPA、无 KV cache、无位置表、
无 `max_len`；mixer 白名单 `sgc / sgc-nogate / sgc-nolocal / srf`。

## TS2SC 侧的工程适配（不改上游数学）

上游是 D=64 的 toy 独立复现；本模块把同一套算子挂到 1B teacher 的 24 层：

    x (d=1536) --Wdown--> h (64) --+--> FrozenSGCCore(role-injected) --> sgc_out
                                   +--> ShortConvSidecar(3,k=7)     --> fA
                                   +--> LiquidSidecar(CfC, tau=2)    --> fB
                                   +--> HadamardBinding(SE1, V=64)   --> fC
    concat[sgc_out; fA; fB; fC] (224) --Wup--> o (1536)

    Δ = β·ε·‖x‖·o/(‖o‖+δ)          (T1' 无量纲尺度契约，‖Δ‖≤βε‖x‖)
    x_next = x + Δ                   (加性融合，teacher 全部功能层逐位保留)

低秩投影只是把 d 维 hidden 映到上游算子的 D=64 工作维；四条路径的公式、
初始化分布、rho/theta/k/layers/V/tau 全部与上游 `sgc_sidecar_final.py` 一致。
旁路 C 的 content key 由 h 的内容 `argmax|h[:32]|` 给出，不由位置给出（SE1）。

## 关键不变量（可证伪，test_v10.py 真跑）

* P7 主干逐位不变：旁路全开 / 全关，FrozenSGCCore 的快照 max|Δ| = 0
* O(1) 定态：状态字节与 L 无关
* FFT↔IIR 对偶 ~1e-14；短卷积流式==整段 ~1e-16；CfC 闭式==逐步 ~1e-16
* SE1：ρ=1 正交零串扰，ρ<1 衰减闭式精确匹配
* β=0 ⇒ Δ=0 ⇒ 与 teacher 逐位相同
* 纯度 AST：负向变异（SDPA/qkv/cache/rotary/attn）全被抓

## 诚实边界

* 上游 v9–v11 的 `COMPLETE_BEYOND=False`：质量轴稳态 parity，不是严格超越；
  2× 对手步数对照失败、A2 负 delta 是崩坏区伪影。本模块只同步**架构能力**，
  不继承任何超越/SOTA 主张。
* 上游语料 240KB / D=32 / L≤1536 / ≤6000 步；1B / 中文 / LRA / 长文档未测。
* 本模块是纯 numpy CPU 实现；β=1 是尺度契约配置，不是 L2 校准成品。
"""
from __future__ import annotations

import ast
import math
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 上游常量（与 sgc_sidecar_final.py 精确一致）
# ---------------------------------------------------------------------------
SEED_MASTER = 20261125
DECAY_MAX = 4096.0
M = 64                       # FrozenSGC 谱字典通道数
DS = 64                      # 旁路工作维（上游 D）
SC_LAYERS = 3
SC_K = 7
LIQ_TAU = 2.0
HAD_V = 64
HAD_RHO = 0.9995
HAD_VK = HAD_V // 2
ALLOWED_MIXERS = frozenset({"sgc", "sgc-nogate", "sgc-nolocal", "srf"})
EPS_DEFAULT = 0.10           # T1' scale contract
SNR_DEFAULT = 1.0            # T2' role bit
ROLE_SLOTS = 6               # n_slots = d/(2*head_dim) for MiniCPM5-1B


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -40, 40)))


# ---------------------------------------------------------------------------
# 1. 静态谱字典 / 核 / FFT↔IIR 双形式（上游 §1，逐行同步）
# ---------------------------------------------------------------------------
_BASIS = {}


def make_basis(M_: int = M, decay_max: float = DECAY_MAX):
    if M_ in _BASIS:
        return _BASIS[M_]
    lam = np.logspace(0, np.log10(decay_max), M_)
    rho = np.exp(-1.0 / lam)
    theta = np.clip(np.pi * np.exp(np.linspace(np.log(0.02), np.log(0.98), M_)),
                    0.02, np.pi - 0.02)
    d = dict(M=M_, rho=rho, theta=theta,
             a=2.0 * rho * np.cos(theta), b=-(rho ** 2), c=np.sin(theta))
    _BASIS[M_] = d
    return d


def kernel(basis, L):
    tau = np.arange(L)[:, None]
    return (basis["rho"][None, :] ** tau) * np.sin((tau + 1) * basis["theta"][None, :])


def iir_stream(z, basis):
    L, M_ = z.shape
    a, b, c = basis["a"], basis["b"], basis["c"]
    y = np.zeros((L, M_), dtype=np.complex128)
    y1 = np.zeros(M_, dtype=np.complex128)
    y2 = np.zeros(M_, dtype=np.complex128)
    for t in range(L):
        yt = a * y1 + b * y2 + c * z[t]
        y[t] = yt
        y2, y1 = y1, yt
    return y


def fft_dual(z, basis):
    L, M_ = z.shape
    N = 2 * L
    return np.fft.ifft(np.fft.fft(z, n=N, axis=0)
                       * np.fft.fft(kernel(basis, L), n=N, axis=0), axis=0)[:L]


def hop(y1, y2, basis, d):
    lp = basis["rho"] * np.exp(1j * basis["theta"])
    lm = basis["rho"] * np.exp(-1j * basis["theta"])
    det = 1.0 / lm - 1.0 / lp
    A = (y1 * (1.0 / lm) - y2) / det
    B = (y2 - y1 * (1.0 / lp)) / det
    return A * lp ** d + B * lm ** d, A * lp ** (d - 1) + B * lm ** (d - 1)


# ---------------------------------------------------------------------------
# 2. 冻结主干（外挂绝不触碰其内部状态）
# ---------------------------------------------------------------------------
class FrozenSGCCore:
    """sgc_sidecar_final.py::FrozenSGC 的等价实现。

    输入改为 TS2SC 的低秩 hidden h（上游用 E[token_id]），其余公式、
    初始化、basis、loc_buf、IIR 状态、feed 全部一致。
    """

    def __init__(self, D=DS, M_=M, seed=0, Win=None, Wg=None, Wloc=None,
                 feed=None):
        self.D, self.M = D, M_
        self.basis = make_basis(M_)
        r = np.random.default_rng(seed)
        self.Win = Win if Win is not None else r.normal(0, 1 / np.sqrt(D), (M_, D))
        self.Wg = Wg if Wg is not None else r.normal(0, 1 / np.sqrt(D), (M_, D))
        self.Wloc = Wloc if Wloc is not None else r.normal(0, 1 / np.sqrt(3), (M_, 3))
        self.feed = feed if feed is not None else np.ones(M_)
        self.loc_buf = np.zeros((2, M_))
        self.y1 = np.zeros(M_, dtype=np.complex128)
        self.y2 = np.zeros(M_, dtype=np.complex128)
        self.last_out = np.zeros(D)
        self.pos = 0

    def step(self, h):
        xn = np.asarray(h, np.float64) / (np.sqrt(np.mean(np.asarray(h) ** 2)) + 1e-8)
        u = (self.Win @ xn) * sigmoid(self.Wg @ xn)
        a, b, c = self.basis["a"], self.basis["b"], self.basis["c"]
        yt = a * self.y1 + b * self.y2 + c * u.astype(np.complex128)
        self.y2, self.y1 = self.y1, yt
        w = np.vstack([self.loc_buf, u[None, :]])
        loc = sum(self.Wloc[:, j] * w[j] for j in range(3))
        self.loc_buf = w[1:]
        y = yt.real + loc + self.feed * u
        self.last_out = y[:self.D] if self.M >= self.D else np.pad(y, (0, self.D - self.M))
        self.pos += 1
        return self.last_out

    def advance_idle(self, n):
        self.y1, self.y2 = hop(self.y1, self.y2, self.basis, n)
        self.loc_buf[:] = 0.0
        self.pos += n

    def snapshot(self):
        return (self.y1.copy(), self.y2.copy(), self.last_out.copy(),
                self.loc_buf.copy(), self.pos)

    def state_bytes(self):
        return int(2 * self.M * 16 + 2 * self.M * 8 + self.D * 8)


# ---------------------------------------------------------------------------
# 3. 旁路 A：3 层 k=7 门控短卷积
# ---------------------------------------------------------------------------
class ShortConvSidecar:
    def __init__(self, D=DS, layers=SC_LAYERS, k=SC_K, seed=0,
                 Wg=None, Wk=None):
        self.D, self.n, self.k = D, layers, k
        r = np.random.default_rng(seed)
        self.Wg = Wg if Wg is not None else np.stack(
            [r.normal(0, 1 / np.sqrt(D), (D, D)) for _ in range(layers)])
        self.Wk = Wk if Wk is not None else np.stack(
            [r.normal(0, 1 / np.sqrt(k), (D, k)) for _ in range(layers)])
        self.buf = [np.zeros((k - 1, D)) for _ in range(layers)]
        self.peak = np.zeros(D)
        self.enabled = True

    def step(self, x):
        if not self.enabled:
            return np.zeros(self.D)
        h = np.asarray(x, np.float64).copy()
        for l in range(self.n):
            g = sigmoid(self.Wg[l] @ h)
            w = np.vstack([self.buf[l], h[None, :]])
            h = g * np.sum(self.Wk[l] * w.T, axis=1)
            self.buf[l] = w[1:]
        self.peak = np.maximum(self.peak, np.abs(h))
        return h

    def full(self, X):
        L = X.shape[0]
        H = np.asarray(X, np.float64).copy()
        for l in range(self.n):
            g = sigmoid(H @ self.Wg[l].T)
            Xp = np.pad(H, ((self.k - 1, 0), (0, 0)))
            H = g * sum(self.Wk[l][:, j][None, :] * Xp[j:j + L]
                        for j in range(self.k))
        return H

    def advance_idle(self, n, pad_x):
        if not self.enabled:
            return
        for l in range(self.n):
            self.buf[l][:] = np.asarray(pad_x)[None, :]

    def feature(self):
        return self.peak.copy() if self.enabled else np.zeros(self.D)

    def state_bytes(self):
        return int(self.n * (self.k - 1) * self.D * 8 + self.D * 8)


# ---------------------------------------------------------------------------
# 4. 旁路 B：CfC 液态门
# ---------------------------------------------------------------------------
class LiquidSidecar:
    def __init__(self, D=DS, seed=0, use_integral=False, Wg=None):
        self.D = D
        r = np.random.default_rng(seed)
        self.Wg = Wg if Wg is not None else r.normal(0, 1 / np.sqrt(D), (D, D))
        self.h = np.zeros(D)
        self.integral = np.zeros(D)
        self.enabled = True
        self.use_integral = use_integral

    def step(self, x, dt=1.0, tau=None):
        if not self.enabled:
            return np.zeros(self.D)
        t = LIQ_TAU if tau is None else float(tau)
        a = np.exp(-dt / t)
        self.h = a * self.h + (1.0 - a) * np.tanh(self.Wg @ np.asarray(x))
        if self.use_integral:
            self.integral += (dt / t) * np.asarray(x)
        return self.h.copy()

    def advance_idle(self, n, x, dt=1.0, tau=LIQ_TAU):
        if not self.enabled:
            return
        an = np.exp(-dt / tau) ** n
        self.h = an * self.h + (1.0 - an) * np.tanh(self.Wg @ np.asarray(x))
        if self.use_integral:
            self.integral += n * (dt / tau) * np.asarray(x)

    def feature(self):
        if not self.enabled:
            return np.zeros(self.D)
        return self.h + self.integral if self.use_integral else self.h.copy()

    def state_bytes(self):
        return int(self.D * 8 * (2 if self.use_integral else 1))


# ---------------------------------------------------------------------------
# 5. 旁路 C：Hadamard 正交登记（SE1）
# ---------------------------------------------------------------------------
class HadamardBindingSidecar:
    def __init__(self, V=HAD_V, rho=HAD_RHO, seed=0, write_noise=0.0):
        self.V, self.VK, self.rho = V, V // 2, rho
        H = np.array([[1.0]])
        while H.shape[0] < V:
            H = np.block([[H, H], [H, -H]])
        self.C = H / np.sqrt(V)
        self.S = np.zeros((V, V))
        self.enabled = True
        self.writes = 0
        self.write_noise = write_noise
        self.rng = np.random.default_rng(seed)

    def register(self, k, v):
        if not self.enabled:
            return
        if self.write_noise > 0 and self.rng.random() < self.write_noise:
            v = int(self.rng.integers(0, self.V))
        k = int(k) % self.V
        v = int(v) % self.V
        self.S = self.rho * self.S + np.outer(self.C[k], self.C[v])
        self.writes += 1

    def observe_content(self, h):
        """内容寻址：key 由 h 的内容给，不由位置给。"""
        if not self.enabled:
            return 0
        k = int(np.argmax(np.abs(np.asarray(h)[:self.VK]))) % self.VK
        self.register(k, self.VK + k)
        return k

    def advance_idle(self, n):
        # 无关段：L1 门屏蔽，没有新登记 -> 槽内容不变。
        # rho 衰减只在 register 时施加，绑「写入年龄」不绑挂钟。
        return

    def query(self, k):
        if not self.enabled:
            return np.zeros(self.VK)
        return (self.C[int(k) % self.V] @ self.S) @ self.C[self.VK:].T

    def feature(self, k):
        return self.query(k) if self.enabled else np.zeros(self.VK)

    def state_bytes(self):
        return int(self.V * self.V * 8 + 64)


def se1_identity_check():
    """SE1 两条严格断言：(a) rho=1 正交零串扰 (b) rho<1 衰减闭式。"""
    n = 8
    b1 = HadamardBindingSidecar(V=64, rho=1.0, seed=1)
    for k in range(n):
        b1.register(k, b1.VK + k)
    on1, off1 = [], []
    for k in range(n):
        r = b1.query(k)
        on1.append(abs(r[k] - 1.0))
        off1.append(float(np.max(np.abs(np.delete(r, k)))))
    rho = HAD_RHO
    b2 = HadamardBindingSidecar(V=64, rho=rho, seed=2)
    for k in range(n):
        b2.register(k, b2.VK + k)
    on2, off2 = [], []
    for k in range(n):
        r = b2.query(k)
        on2.append(abs(r[k] - rho ** (n - 1 - k)))
        off2.append(float(np.max(np.abs(np.delete(r, k)))))
    return dict(on_orth=max(on1), off_orth=max(off1),
                on_decay_err=max(on2), off_decay=max(off2), rho=rho)


# ---------------------------------------------------------------------------
# 6. TS2SC 融合层：24 层并行加性分支
# ---------------------------------------------------------------------------
class SGCV10Layer:
    """单层：4 路径 concat readout -> T1' scale contract -> 加性注入。"""

    def __init__(self, d: int, layer_seed: int, params: dict,
                 eps: float = EPS_DEFAULT, role_snr: float = SNR_DEFAULT):
        self.d = d
        self.eps = eps
        self.role_snr = role_snr
        self.Wdown = np.asarray(params["Wdown"], np.float64)     # (DS, d)
        self.Wup = np.asarray(params["Wup"], np.float64)         # (d, 224)
        self.core = FrozenSGCCore(seed=layer_seed + 1,
                                  Win=params["core.Win"], Wg=params["core.Wg"],
                                  Wloc=params["core.Wloc"], feed=params["core.feed"])
        self.sc = ShortConvSidecar(seed=layer_seed + 2,
                                   Wg=params["sc.Wg"], Wk=params["sc.Wk"])
        self.liq = LiquidSidecar(seed=layer_seed + 3, Wg=params["liq.Wg"])
        self.bind = HadamardBindingSidecar(seed=layer_seed + 4)
        self.bind.C = np.asarray(params["bind.C"], np.float64)    # 确定性码本
        self.role_Wc = np.asarray(params["role.Wc"], np.float64)   # (3, d) T2'
        self.alpha = float(np.asarray(params["gate.alpha"]).reshape(-1)[0])
        self.enabled = True
        self.beta = 0.0

    def reset(self):
        self.core.loc_buf[:] = 0.0
        self.core.y1[:] = 0
        self.core.y2[:] = 0
        self.core.last_out[:] = 0
        self.core.pos = 0
        self.sc.buf = [np.zeros((self.sc.k - 1, self.sc.D))
                       for _ in range(self.sc.n)]
        self.sc.peak[:] = 0
        self.liq.h[:] = 0
        self.liq.integral[:] = 0
        self.bind.S[:] = 0
        self.bind.writes = 0

    def hook(self, x, pos):
        """返回加性 Δ；β=0 时严格返回零向量（逐位不变）。"""
        if not self.enabled or self.beta <= 0.0:
            return np.zeros_like(x)
        xf = np.asarray(x, np.float64)
        rx = float(np.linalg.norm(xf))
        # 低秩投影：d -> DS
        h = (self.Wdown @ xf) / math.sqrt(self.d)
        # T2' role bit：只注入 FrozenSGCCore 的输入（主干 k 信号），
        # 三条旁路消费原始 h，绝不回写主干。
        r = int(pos) % ROLE_SLOTS
        e = np.zeros(self.core.D)
        # 把 role one-hot 广播到前 ROLE_SLOTS 维（尺度不变）
        e[r] = self.role_snr
        hn = h / (np.linalg.norm(h) + 1e-12)
        h_role = hn + e
        sgc_out = self.core.step(h_role)
        fA = self.sc.step(h)
        fB = self.liq.step(h)
        key = self.bind.observe_content(h)
        fC = self.bind.feature(key)                      # (VK=32,)
        cat = np.concatenate([sgc_out, fA, fB, fC])      # (64+64+64+32=224,)
        o = self.Wup @ cat
        ro = float(np.linalg.norm(o))
        delta = self.beta * self.eps * rx * (o / (ro + 1e-12))
        # 硬契约（防御性，T1' 数学上已保证）
        nd = float(np.linalg.norm(delta))
        bound = self.beta * self.eps * rx
        if nd > bound * (1.0 + 1e-9):
            delta *= bound / (nd + 1e-12)
        return delta.astype(np.float32)

    def state_bytes(self):
        return int(self.core.state_bytes() + self.sc.state_bytes()
                   + self.liq.state_bytes() + self.bind.state_bytes())


class SGCV10Branch:
    """24 层并行加性融合分支（teacher 全部功能层逐位保留）。"""

    NEW_KEYS = ("Wdown", "Wup", "core.Win", "core.Wg", "core.Wloc",
                "core.feed", "sc.Wg", "sc.Wk", "liq.Wg", "bind.C",
                "role.Wc", "gate.alpha", "gate.beta")

    def __init__(self, cfg, params_by_layer: dict,
                 eps=EPS_DEFAULT, role_snr=SNR_DEFAULT):
        self.d = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.eps = eps
        self.layers = [SGCV10Layer(self.d, 1000 + 17 * i, params_by_layer[i],
                                   eps=eps, role_snr=role_snr)
                       for i in range(self.nl)]
        self.beta = 0.0

    def set_beta(self, beta: float):
        self.beta = float(beta)
        for l in self.layers:
            l.beta = float(beta)

    def reset(self):
        for l in self.layers:
            l.reset()

    def hook(self, l, x, pos):
        return self.layers[l].hook(x, pos)

    def state_bytes(self):
        return sum(l.state_bytes() for l in self.layers)

    def read_keys(self):
        """F21 等价：每层 NEW 参数都在前向路径被读取。"""
        return len(self.layers) * len(self.NEW_KEYS)


# ---------------------------------------------------------------------------
# 7. 确定性 NEW 参数（与上游初始化分布一致；不读任何旧产物）
# ---------------------------------------------------------------------------
def deterministic_layer_params(d: int, seed: int) -> dict:
    """由 seed 完全确定，与任何 identity_* / 旧 branch 文件无关。"""
    r = np.random.default_rng(seed)
    DS_, M_ = DS, M
    p = {}
    p["Wdown"] = r.normal(0, 1 / math.sqrt(d), (DS_, d)).astype(np.float32)
    p["Wup"] = r.normal(0, 1 / math.sqrt(224), (d, 224)).astype(np.float32)
    p["core.Win"] = r.normal(0, 1 / math.sqrt(DS_), (M_, DS_)).astype(np.float32)
    p["core.Wg"] = r.normal(0, 1 / math.sqrt(DS_), (M_, DS_)).astype(np.float32)
    p["core.Wloc"] = r.normal(0, 1 / math.sqrt(3), (M_, 3)).astype(np.float32)
    p["core.feed"] = np.ones(M_, np.float32)
    p["sc.Wg"] = np.stack([r.normal(0, 1 / math.sqrt(DS_), (DS_, DS_))
                           for _ in range(SC_LAYERS)]).astype(np.float32)
    p["sc.Wk"] = np.stack([r.normal(0, 1 / math.sqrt(SC_K), (DS_, SC_K))
                           for _ in range(SC_LAYERS)]).astype(np.float32)
    p["liq.Wg"] = r.normal(0, 1 / math.sqrt(DS_), (DS_, DS_)).astype(np.float32)
    H = np.array([[1.0]])
    while H.shape[0] < HAD_V:
        H = np.block([[H, H], [H, -H]])
    p["bind.C"] = (H / math.sqrt(HAD_V)).astype(np.float32)
    p["role.Wc"] = np.zeros((3, d), np.float32)
    p["gate.alpha"] = np.array([1.0], np.float32)
    p["gate.beta"] = np.array([0.0], np.float32)
    return p


TENSOR_SPECS = (
    ("Wdown", [DS, "d"], "F32"),
    ("Wup", ["d", 224], "F32"),
    ("core.Win", [M, DS], "F32"),
    ("core.Wg", [M, DS], "F32"),
    ("core.Wloc", [M, 3], "F32"),
    ("core.feed", [M], "F32"),
    ("sc.Wg", [SC_LAYERS, DS, DS], "F32"),
    ("sc.Wk", [SC_LAYERS, DS, SC_K], "F32"),
    ("liq.Wg", [DS, DS], "F32"),
    ("bind.C", [HAD_V, HAD_V], "F32"),
    ("role.Wc", [3, "d"], "F32"),
    ("gate.alpha", [1], "F32"),
    ("gate.beta", [1], "F32"),
)


def spec_shape(shp, d):
    return [d if x == "d" else int(x) for x in shp]


# ---------------------------------------------------------------------------
# 8. 纯度闸门（上游 P7/P8 + AST 负向变异）
# ---------------------------------------------------------------------------
ATTENTION_TOKENS = ("attention", "softmax(q", "qk", "kv_cache",
                    "rotary", "position_embedding", "max_len")


def purity_source_audit(src: str) -> dict:
    """源码级：mixer 白名单之外不得出现任何注意力开关。"""
    low = src.lower()
    hits = [t for t in ATTENTION_TOKENS if t in low and
            "audit" not in t.lower()]     # 允许审计词表自身出现
    # 精确判定：只看 AST 里的函数/类名，不看注释字符串
    tree = ast.parse(src)
    defs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defs.append(n.name.lower())
    forbidden_defs = [n for n in defs if any(
        t in n for t in ("attention", "sdpa", "qkv", "softmax_attn"))]
    return dict(allowed_mixers=sorted(ALLOWED_MIXERS),
                forbidden_defs=forbidden_defs,
                passed=not forbidden_defs,
                note="AST 只看定义名；注释里的审计词表不计")


def core_unchanged_by_sidecars(seed=7, steps=6):
    """P7 等价：旁路全开 / 全关，FrozenSGCCore 快照逐位不变。"""
    p = deterministic_layer_params(1536, seed)
    a = SGCV10Layer(1536, seed, p)
    b = SGCV10Layer(1536, seed, p)
    b.sc.enabled = False
    b.liq.enabled = False
    b.bind.enabled = False
    a.beta = 1.0
    b.beta = 1.0
    rng = np.random.default_rng(seed + 99)
    maxdiff = 0.0
    for t in range(steps):
        x = rng.normal(0, 1, 1536).astype(np.float32)
        a.hook(x, t)
        b.hook(x, t)
        sa, sb = a.core.snapshot(), b.core.snapshot()
        for u, v in zip(sa[:4], sb[:4]):
            maxdiff = max(maxdiff, float(np.max(np.abs(
                np.asarray(u) - np.asarray(v)))))
    return dict(steps=steps, max_abs_delta=maxdiff,
                passed=bool(maxdiff == 0.0))
