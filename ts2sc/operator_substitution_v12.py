#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v12 · Transformer → SimplCore **算子级替换**（不是 hybrid，不是壳）

## 与 v10.x / v11 的本质区别
* v10.x hybrid : y = T(x) + 旁挂，T(x) 是 Llama 全注意力 ⇒ 关掉旁挂退回原模型
* v11 纯重建   : 219 个 teacher 张量全部丢弃、随机初始化 ⇒ 能力归零
* **v12 本文件** : 把「token 混合」这一个子算子整个换掉：

      h   = RMSNorm(x, input_ln_COPY)
      m   = RotationalSpectralMixer.step(h)        # 冻结谱核 LTI，O(1) 状态
      x   = x + m                                  # 没有 Q/K/V/o_proj
      h2  = RMSNorm(x, post_ln_COPY)
      ff  = down_COPY @ ( silu(gate_COPY@h2) * (up_COPY@h2) )
      y   = x + ff
      out = lm_head_COPY @ RMSNorm(y, norm_COPY)

  推理链里 **没有 T(x) 残差主干**，96 个 self_attn 张量一个都不加载；
  123 个非注意力张量逐位 COPY。这是 operator substitution，不是 sidecar。

## SimplCore 特性（全部保留且可证）
* 冻结静态核：每模态 A_m = ρ_m·[[cosθ,-sinθ],[sinθ,cosθ]]（常矩阵 ⇒ LTI）
  冲激响应 ψ_m[τ] = ρ_m^τ·sin((τ+1)θ_m) ⇒ Toeplitz / FFT↔IIR 对偶成立
* O(1) 常驻状态：s_m ∈ R²，与 L 无关；advance_idle(n) = A_m^n 闭式
* 无输入依赖门控进转移矩阵（守定理 1/2/7 的静态核前提）
* 解析尺度契约（沿用 T1'）：m = β·‖h‖·û，û 单位方向，β 无量纲
  ⇒ 注入量级受控，RMSNorm 不可能溢出，不需要拟合/不需要 GPU

## 诚实边界
* 能力来自 **123 个真实权重逐位继承**，不是随机初始化，也不是蒸馏出来。
* 未做全量梯度蒸馏；小样本答题对照只是「是否保留」的探针，
  不等于全分布能力不劣（需 ppl / benchmark / 配对 CI）。
* ρ<1 ⇒ ρ^10M 下溢，长程谱响应衰减为 0，O(1) 状态恒定 ≠ 长程记忆。
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

SCHEMA = "TS2SC/operator-substitution-v12"
ARCH = "SIMPLCORE_NATIVE_OPERATOR_SUBSTITUTION_NO_ATTENTION"
BETA = 0.1                 # T1' 无量纲注入比
RHO = 0.9995
THETA_BASE = 0.3
N_MODES = 8

# teacher 219 张量里**禁止进入推理链**的 96 个注意力张量模式
FORBIDDEN = ("self_attn", "q_proj", "k_proj", "v_proj", "o_proj",
             "rotary", "rope", "kv_cache")


class RotationalSpectralMixer:
    """冻结旋转谱核 LTI mixer：每模态 2 维实状态，A_m 常矩阵。

    s_m ← ρ_m R(θ_m) s_m + B_m h
    m   = β·‖h‖ · normalize( Σ_m C_m s_m )
    """

    def __init__(self, d: int, seed: int, n_modes: int = N_MODES,
                 rho: float = RHO, theta_base: float = THETA_BASE,
                 beta: float = BETA):
        self.d, self.m, self.beta = d, n_modes, beta
        rng = np.random.default_rng(seed)
        self.rho = rho * (1.0 - 1e-4 * np.arange(n_modes))
        self.theta = theta_base * (1.0 + rng.random(n_modes))
        self.A = np.stack([self.rho[k] * np.array(
            [[math.cos(self.theta[k]), -math.sin(self.theta[k])],
             [math.sin(self.theta[k]), math.cos(self.theta[k])]],
            dtype=np.float32) for k in range(n_modes)])       # (m,2,2) 冻结
        self.B = rng.normal(0, 1.0 / math.sqrt(d),
                            (n_modes, 2, d)).astype(np.float32)
        self.C = rng.normal(0, 1.0 / math.sqrt(2 * n_modes),
                            (d, n_modes, 2)).astype(np.float32)
        self.s = np.zeros((n_modes, 2), np.float32)             # O(1) 状态
        self.tau = 0

    def step(self, h, fit=None):
        """fit=None  → v12 随机 readout；fit=dict(Rh,Ru,W,beta) → v12.1 解析迁移。

        W 只在 readout：A_m/B_m/C_m 全部冻结，转移矩阵不回写，
        静态核 / Toeplitz / FFT↔IIR 对偶 / O(1) 状态一个不变。
        """
        h = h.astype(np.float32)
        bh = np.einsum("mkd,d->mk", self.B, h)                  # (m,2)
        self.s = np.einsum("mij,mj->mi", self.A, self.s) + bh
        self.tau += 1
        u = np.einsum("dmk,mk->d", self.C, self.s)              # 谱核状态读出
        hn = float(np.linalg.norm(h))
        if fit is not None:
            z = np.concatenate([fit["Rh"] @ h, fit["Ru"] @ u])  # (2r,)
            raw = z @ fit["W"]                                  # (d,)
            n = float(np.linalg.norm(raw))
            if n < 1e-12:
                return np.zeros(self.d, np.float32)
            return (float(fit["beta"]) * hn / n) * raw          # T1' 契约
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            return np.zeros(self.d, np.float32)
        return (self.beta * hn / n) * u                          # T1' 尺度契约

    def advance_idle(self, n):
        """O(1) 闭式：A_m^n = ρ_m^n R(nθ_m)，不循环 n 次。"""
        n = int(n)
        rn = self.rho.astype(np.float64) ** n
        ang = n * self.theta.astype(np.float64)
        ca, sa = np.cos(ang), np.sin(ang)
        Rn = np.stack([np.stack([[rn[k] * ca[k], -rn[k] * sa[k]],
                                 [rn[k] * sa[k], rn[k] * ca[k]]])
                       for k in range(self.m)]).astype(np.float32)
        self.s = np.einsum("mij,mj->mi", Rn, self.s)
        self.tau += n

    def state_bytes(self):
        return int(self.s.nbytes + self.A.nbytes + self.B.nbytes
                   + self.C.nbytes + self.rho.nbytes
                   + self.theta.nbytes + 8)

    def tensors(self, prefix):
        t = {prefix + "A": self.A, prefix + "B": self.B,
             prefix + "C": self.C, prefix + "state": self.s,
             prefix + "rho": self.rho.astype(np.float32),
             prefix + "theta": self.theta.astype(np.float32)}
        return t


class SubstitutedLM:
    """v12 完整模型：123 个 teacher 张量逐位继承 + 24 个谱核 mixer。"""

    def __init__(self, V, cfg, seed: int = 20261129, fit=None):
        self.V, self.cfg = V, cfg
        self.d = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.vocab = cfg["vocab_size"]
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.norm_w = V.f32("model.norm.weight")
        self.mixers = [RotationalSpectralMixer(self.d, seed + 101 * i)
                       for i in range(self.nl)]
        # v12.1：每层一个解析迁移的静态读出投影（只在 readout，不回写核）
        self.fit = fit
        self._fits = None
        if fit is not None:
            W = fit["W"]
            assert W.shape[0] == self.nl, (W.shape, self.nl)
            G = fit.get("gamma")
            self._fits = [dict(Rh=fit["Rh"], Ru=fit["Ru"],
                               W=W[l],
                               beta=(float(G[l]) if G is not None
                                     else float(fit["beta"])))
                          for l in range(self.nl)]
        # 运行期证据：记录所有被加载的 teacher 张量名
        self.loaded = []
        self.loaded.append("model.norm.weight")

    @classmethod
    def from_fit_npz(cls, V, cfg, path, seed: int = 20261129):
        z = np.load(path, allow_pickle=False)
        fit = dict(W=z["W"], Rh=z["Rh"], Ru=z["Ru"],
                   gamma=z["gamma"] if "gamma" in z.files else None,
                   beta=float(z["beta"]))
        return cls(V, cfg, seed=seed, fit=fit)

    @staticmethod
    def _rms(x, w, eps):
        return x * (1.0 / np.sqrt(float(np.mean(x * x)) + eps)) * w

    @staticmethod
    def _silu(a):
        return a / (1.0 + np.exp(-a))

    def _load(self, name):
        self.loaded.append(name)
        return self.V.f32(name)

    def _logits(self, y):
        chunks = []
        for r0 in range(0, self.vocab, 8192):
            r1 = min(r0 + 8192, self.vocab)
            Wb = self.V.rows_f32("lm_head.weight", r0, r1)
            chunks.append(Wb @ y)
        self.loaded.append("lm_head.weight")
        lg = np.concatenate(chunks)
        return (int(np.argmax(lg)) if bool(np.isfinite(lg).all()) else 0), lg

    def step(self, tid, pos, need_logits=True, return_hidden=False,
             with_logits=False):
        x = self.V.rows_f32("model.embed_tokens.weight",
                            int(tid), int(tid) + 1)[0].copy()
        self.loaded.append("model.embed_tokens.weight")
        for l in range(self.nl):
            p = f"model.layers.{l}."
            w_in = self._load(p + "input_layernorm.weight")
            w_post = self._load(p + "post_attention_layernorm.weight")
            w_gate = self._load(p + "mlp.gate_proj.weight")
            w_up = self._load(p + "mlp.up_proj.weight")
            w_down = self._load(p + "mlp.down_proj.weight")
            h = self._rms(x, w_in, self.eps)
            m = self.mixers[l].step(            # ← 替换掉的 self_attn 位置
                h, fit=(self._fits[l] if self._fits is not None else None))
            x = x + m
            h2 = self._rms(x, w_post, self.eps)
            ff = w_down @ (self._silu(w_gate @ h2) * (w_up @ h2))
            x = x + ff
            del w_in, w_post, w_gate, w_up, w_down, h, h2, ff, m
        y = self._rms(x, self.norm_w, self.eps)
        if return_hidden and with_logits:
            return y.copy(), self._logits(y)
        if return_hidden:
            return y.copy()
        if not need_logits:
            return None
        return self._logits(y)

    def generate(self, ids, max_new=12):
        out = []
        fins = []
        for j in range(max_new):
            tid, lg = self.step(out[-1] if out else int(ids[-1]), j,
                                need_logits=True)
            out.append(int(tid))
            fins.append(bool(np.isfinite(lg).all()))
            if tid in (1, 130073):
                break
        return out, all(fins)

    def advance_idle(self, n):
        for m in self.mixers:
            m.advance_idle(n)

    def state_bytes(self):
        return sum(m.state_bytes() for m in self.mixers)

    def tensors(self):
        t = {}
        for i, m in enumerate(self.mixers):
            t.update(m.tensors(f"pure.mixer.{i}."))
        return t

    def tensor_sha256(self):
        out = {}
        for n, a in self.tensors().items():
            a = np.ascontiguousarray(a)
            h = hashlib.sha256()
            h.update(memoryview(a).cast("B"))
            out[n] = dict(shape=list(a.shape), bytes=int(a.nbytes),
                          sha256=h.hexdigest())
        return out

    # ---- 纯度 / 继承账本（运行期证据，不是静态声明）----
    def purity_report(self, teacher_names, teacher_copy_sha=None):
        loaded_attn = sorted({n for n in self.loaded
                              if any(f in n.lower() for f in FORBIDDEN)})
        inherited = sorted({n for n in self.loaded
                            if not any(f in n.lower() for f in FORBIDDEN)})
        # 逐位对账：manifest 的 sha256 是 **BF16 原始字节**。
        # 分块流式哈希，避免 lm_head(401 MiB) / embed(401 MiB) 整块复制 OOM。
        mism = []
        checked = 0
        if teacher_copy_sha:
            ds = self.V.ds
            for n in inherited:
                a, b = self.V.header[n]["data_offsets"]
                total, off, step = b - a, 0, 16 << 20
                h = hashlib.sha256()
                while off < total:
                    h.update(self.V.read(ds + a + off,
                                         min(step, total - off)))
                    off += step
                checked += 1
                if h.hexdigest() != teacher_copy_sha.get(n):
                    mism.append(n)
        return dict(
            architecture=ARCH,
            teacher_tensor_total=len(teacher_names),
            teacher_self_attn_tensors=sum(
                1 for n in teacher_names if "self_attn" in n),
            loaded_self_attn_tensors=loaded_attn,        # 必须为空
            n_loaded_self_attn=len(loaded_attn),
            inherited_tensor_names=inherited,
            n_inherited=len(inherited),
            sha_checked=checked, sha_mismatches=mism,
            hybrid_residual_T=False,                      # 没有 T(x) 主干
            pure=bool(not loaded_attn and not mism and checked > 0),
        )

    def snapshot_state(self):
        return [dict(s=m.s.copy(), tau=m.tau) for m in self.mixers]

    def restore_state(self, snap):
        for m, s in zip(self.mixers, snap):
            m.s[:] = s["s"]
            m.tau = s["tau"]
