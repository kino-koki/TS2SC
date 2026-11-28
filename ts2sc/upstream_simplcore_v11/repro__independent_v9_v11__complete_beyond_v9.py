# -*- coding: utf-8 -*-
"""
COMPLETE BEYOND v9 —— 不动静态卷积核，把「堆叠」这个未试过的自由度推到极限
=================================================================================
【预先声明的判定协议(写死, 事后不得修改)】
  P0 核: psi_m[tau]=rho_m^tau*sin((tau+1)theta_m) 固定; rho/theta 不学、不随输入变;
         FFT<->IIR 双形式逐位等价; 每步流式状态与 L 无关; 无位置表/无 max_len。
  P1 公平: 两臂【同层数 / 同 Block(pre-norm RMSNorm + FFN) / 同 Sidecar / 同 readout /
         同优化器 / 同 seed / 同数据 / 同长度课程 / 同 dt 增广】。
         唯一差别 = 主干 mixer: 静态 Toeplitz 谱卷积  vs  causal softmax attention(带 ALiBi)。
         attention 臂额外持有可学位置表与 QKV; 并另做 2x 步数对照(更强对手)。
  P2 质量三轴(3 seeds 均值 且 每 seed 分别满足):
         A1 同分布 bpc  SGC <= ATTN + 0.05
         A2 长度外推   至少一个 L 上 SGC < ATTN - 0.05   (在两臂都做长度课程的前提下)
         A3 不规则 dt   SGC <= ATTN + 0.05
  P3 稳定性: 步数曲线连续 >=3 个 checkpoint 三轴全过(不挑瞬态)。
  P4 成本/可行性: 状态 O(1) 与 L 无关; 确定性 HBM 代数上界; REAL_RUN 计时。
  P5 纯度: 真·调用仓库 purity.py 三层守卫 + 负向变异; 产品源码零注意力符号。
  P6 真实性: 两臂反传中心差分逐参数校验; 独立复跑逐字段零差异; seed/sha256 固定。
  COMPLETE_BEYOND = P0..P6 全成立。

【这轮补的、之前八轮从未试过的自由度】
  [N1] 多层堆叠(1-4 层): 静态核逐层复用但 U^(l)=A*sigma(B) 由上一层全局混合结果给出
       -> 有效成对交互变成输入依赖(单层 O(M) 自由度论证不再成立)。
  [N2] 每层共享 pre-norm Block: X + mixer(X) -> RMSNorm -> +FFN(RMSNorm(.))
       两臂结构逐字相同, 公平是构造性的。
  [N3] FFN channel mixing(gelu), 两臂同隐宽。
  [N4] 前向/反传都走 FFT O(L log L M)(贴合仓库 torch.fft 训练路径), 中心差分校验。
  [N5] dt 增广两臂都加(比 v3-v8 更公平, 之前只给对手加)。
  [N6] 长度课程两臂都做(这是 v8 消掉 A2 优势的公平条件, 本轮必须在课程下仍赢)。
  [N8] 两臂共享可配残差缩放 beta(深层稳定), 仍属 Block 级同构。
  [N7] 【中途公平性修正】长度归一化 n[t]=(t+1)^-alpha 初版只加在 SGC 臂, 造成
       A1 -0.50/A2 -0.60/A3 -0.58 的假胜; 已改为【两臂共享 Block 级】逐位置缩放,
       并给 mixer 反传乘同一 n。任何只在单臂出现的结构增益一律不计入判定。

【停止准则(避免无限循环, 预先声明)】
  最多跑完: 层数 {1,2,3,4} x M {64,128} 快筛 -> 最优 4000 步 3 seeds ->
  对手 2x 步数 -> 步数曲线(至 8000) -> 确定性。
  若仍不达成, 如实给出最接近配置/差距/判定, 不再无意义堆算力。

【关键自我修正】
  v8 报告里「静态核永远补不上随输入变化 -> 定义性边界」是【单层】论证, 对多层不成立。
  本轮就是来测这个漏洞。若赢, 上一轮结论被推翻; 若输, 才轮到谈定义性边界。

复现: python complete_beyond_v9.py -> complete_beyond_v9_results.json
依赖 numpy; 无网络; 本轮不推送。
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
import numpy as np

MASTER = 20261125
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

_s = importlib.util.spec_from_file_location("v4", os.path.join(_HERE, "complete_beyond_v4.py"))
V4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(V4)
_fd = importlib.util.spec_from_file_location("fd", os.path.join(_HERE, "final_decisive.py"))
FD = importlib.util.module_from_spec(_fd); _fd.loader.exec_module(FD)

sigmoid = V4.sigmoid
softmax_ce = V4.softmax_ce
Adam = V4.Adam
make_basis = V4.make_basis
kernel = V4.kernel
SharedSidecarV4 = V4.SharedSidecarV4
sample_window = V4.sample_window
load_corpus = V4.load_corpus
param_count = V4.param_count
load_real_purity = V4.load_real_purity
deterministic_feasibility = FD.deterministic_feasibility


# =====================================================================
# 共享数值件
# =====================================================================
def rms_fwd(Y, g, eps=1e-8):
    r = np.sqrt(np.mean(Y * Y, axis=1, keepdims=True) + eps)
    Yh = Y / r
    return Yh * g, (Y, r, Yh, g, eps)


def rms_bwd(c, dout):
    Y, r, Yh, g, eps = c
    D = Y.shape[1]
    dg = np.sum(dout * Yh, axis=0)
    dYh = dout * g
    dr = np.sum(dYh * Y, axis=1, keepdims=True) / (D * r ** 3)
    return dYh / r - Y * dr, dg


def gelu(x):
    return x * sigmoid(1.702 * x)


def gelu_grad(x):
    s = sigmoid(1.702 * x)
    return s + 1.702 * x * s * (1 - s)


def fft_conv_fwd(U, K):
    """因果卷积 y[t]=sum_{s<=t} K[t-s] U[s], 走 FFT(仓库训练路径)。"""
    L, M = U.shape
    N = 2 * L
    Kf = np.fft.fft(K, n=N, axis=0)
    Uf = np.fft.fft(U, n=N, axis=0)
    y = np.fft.ifft(Uf * Kf, axis=0)[:L].real
    return y, (Kf, N, L)


def fft_conv_bwd(dy, c):
    """du[s] = sum_{t>=s} K[t-s] dy[t]; 等价 z=dy[::-1] 的因果卷积再反转。"""
    Kf, N, L = c
    z = dy[::-1].copy()
    w = np.fft.ifft(np.fft.fft(z, n=N, axis=0) * Kf, axis=0)[:L].real
    return w[::-1].copy()


# =====================================================================
# 共享 pre-norm Block: h = RMSNorm(X + mixer(X)); out = h + FFN(RMSNorm(h))
# 两臂逐字相同 -> 公平是构造性的
# =====================================================================
class Block:
    def __init__(self, D, H, seed, alpha=0.0, beta=1.0):
        r = np.random.default_rng(seed)
        self.D, self.H, self.alpha, self.beta = D, H, float(alpha), float(beta)
        self.g1 = np.ones(D)
        self.g2 = np.ones(D)
        self.W1 = r.normal(0, 1 / math.sqrt(D), (H, D))
        self.W2 = r.normal(0, 1 / math.sqrt(H), (D, H))

    def params(self, tag):
        return {f"{tag}.g1": self.g1, f"{tag}.g2": self.g2,
                f"{tag}.W1": self.W1, f"{tag}.W2": self.W2}

    def forward(self, X, mixer_out):
        L = X.shape[0]
        n = (np.arange(1, L + 1, dtype=float) ** (-self.alpha))[:, None]
        mixer_out = mixer_out * n * self.beta         # 两臂共享, 构造性公平
        h_raw = X + mixer_out
        h, c1 = rms_fwd(h_raw, self.g1)
        f_in, c2 = rms_fwd(h, self.g2)
        G = f_in @ self.W1.T
        g = gelu(G)
        f = g @ self.W2.T
        out = h + f
        return out, (h_raw, c1, h, c2, f_in, G, g, n)

    def backward(self, c, dout):
        h_raw, c1, h, c2, f_in, G, g, n = c
        df = dout
        dW2 = df.T @ g                       # f = g @ W2.T
        dg = df @ self.W2
        dG = dg * gelu_grad(G)
        dW1 = dG.T @ f_in
        df_in = dG @ self.W1
        dh_from_ffn, dg2 = rms_bwd(c2, df_in)
        dh = dout + dh_from_ffn               # out = h + f
        dh_raw, dg1 = rms_bwd(c1, dh)
        return dh_raw, {"g1": dg1, "g2": dg2, "W1": dW1, "W2": dW2}


# =====================================================================
# Mixer A: SGC(静态 Toeplitz 核, 一字不动)
# =====================================================================
class SGCMixer:
    def __init__(self, D, M, basis, seed, alpha=0.0):
        r = np.random.default_rng(seed)
        self.D, self.M, self.basis, self.alpha = D, M, basis, float(alpha)
        self.Win = r.normal(0, 1 / math.sqrt(D), (M, D))
        self.Wg = r.normal(0, 1 / math.sqrt(D), (M, D))
        self.Wloc = r.normal(0, 1 / math.sqrt(3), (M, 3))
        self.coef = np.ones((M, D)) / math.sqrt(M)

    def params(self, tag):
        return {f"{tag}.Win": self.Win, f"{tag}.Wg": self.Wg,
                f"{tag}.Wloc": self.Wloc, f"{tag}.coef": self.coef}

    def forward(self, X):
        L = X.shape[0]
        A = X @ self.Win.T
        B = X @ self.Wg.T
        sB = sigmoid(B)
        U = A * sB
        K = kernel(self.basis, L)
        conv, cc = fft_conv_fwd(U, K)
        Yloc = np.zeros((L, self.M))
        for j in range(3):
            if j < L:
                Yloc[j:] += self.Wloc[:, j] * U[:L - j]
        Ymid = conv + Yloc + U
        n = (np.arange(1, L + 1, dtype=float) ** (-self.alpha))[:, None]
        H = (Ymid @ self.coef) * n
        return H, dict(X=X, A=A, B=B, sB=sB, U=U, K=K, conv_cc=cc,
                       Ymid=Ymid, n=n, L=L)

    def backward(self, c, dH):
        X, A, B, sB, U = c["X"], c["A"], c["B"], c["sB"], c["U"]
        L, M = c["L"], self.M
        dYraw = dH * c["n"]
        dcoef = c["Ymid"].T @ dYraw
        dYmid = dYraw @ self.coef.T
        dU = dYmid.copy()
        dU += fft_conv_bwd(dYmid, c["conv_cc"])
        dWloc = np.zeros_like(self.Wloc)
        for j in range(3):
            if j < L:
                dWloc[:, j] = np.einsum("tm,tm->m", dYmid[j:], U[:L - j])
                dU[:L - j] += self.Wloc[:, j] * dYmid[j:]
        dA = dU * sB
        dB = dU * A * sB * (1 - sB)
        dX = dA @ self.Win + dB @ self.Wg
        return dX, {"Win": dA.T @ X, "Wg": dB.T @ X,
                    "Wloc": dWloc, "coef": dcoef}


# =====================================================================
# Mixer B: causal softmax attention + ALiBi(更强对手)
# =====================================================================
class AttentionMixer:
    def __init__(self, D, seed):
        r = np.random.default_rng(seed)
        self.D = D
        self.Wq = r.normal(0, 1 / math.sqrt(D), (D, D))
        self.Wk = r.normal(0, 1 / math.sqrt(D), (D, D))
        self.Wv = r.normal(0, 1 / math.sqrt(D), (D, D))

    def params(self, tag):
        return {f"{tag}.Wq": self.Wq, f"{tag}.Wk": self.Wk, f"{tag}.Wv": self.Wv}

    def forward(self, X):
        L, D = X.shape
        Q = X @ self.Wq.T; K = X @ self.Wk.T; Vv = X @ self.Wv.T
        i = np.arange(L)[:, None]; j = np.arange(L)[None, :]
        S = Q @ K.T / math.sqrt(D) - 0.5 * np.abs(i - j) \
            + np.triu(np.full((L, L), -1e9), k=1)
        S -= S.max(-1, keepdims=True)
        A = np.exp(S); A /= A.sum(-1, keepdims=True)
        return A @ Vv, dict(X=X, Q=Q, K=K, Vv=Vv, A=A, L=L, D=D)

    def backward(self, c, dH):
        A, X, Q, K, Vv, L, D = c["A"], c["X"], c["Q"], c["K"], c["Vv"], c["L"], c["D"]
        dA = dH @ Vv.T
        dV = A.T @ dH
        dS = A * (dA - (A * dA).sum(-1, keepdims=True))
        dQ = dS @ K / math.sqrt(D)
        dK = dS.T @ Q / math.sqrt(D)
        dX = dQ @ self.Wq + dK @ self.Wk + dV @ self.Wv
        return dX, {"Wq": dQ.T @ X, "Wk": dK.T @ X, "Wv": dV.T @ X}


# =====================================================================
# 两臂: 同层数 Block, 唯一差别 mixer
# =====================================================================
class BaseArm:
    def __init__(self, D, V, n_layers, H, seed, Vc=64, Lmax=384, M=64,
                 alpha=0.0, beta=1.0, kind="sgc"):
        self.D, self.V, self.n_layers, self.M, self.kind = D, V, n_layers, M, kind
        self.Lmax, self.alpha = Lmax, alpha
        r = np.random.default_rng(seed)
        self.E = r.normal(0, 1 / math.sqrt(D), (V, D))
        self.P = r.normal(0, 1 / math.sqrt(D), (Lmax, D)) if kind == "attn" else None
        self.basis = make_basis(M) if kind == "sgc" else None
        self.mixers = []
        self.blocks = []
        for l in range(n_layers):
            if kind == "sgc":
                self.mixers.append(SGCMixer(D, M, self.basis, seed + 1000 * (l + 1), 0.0))
            else:
                self.mixers.append(AttentionMixer(D, seed + 1000 * (l + 1)))
            self.blocks.append(Block(D, H, seed + 2000 * (l + 1), alpha, beta))
        self.sc = SharedSidecarV4(D, Vc, seed=seed + 100)
        self.Wh = r.normal(0, 1 / math.sqrt(4 * D), (V, 4 * D))
        self.bh = np.zeros(V)

    def params(self):
        p = {"E": self.E, "Wh": self.Wh, "bh": self.bh}
        if self.P is not None:
            p["P"] = self.P
        for l, (m, b) in enumerate(zip(self.mixers, self.blocks)):
            p.update(m.params(f"L{l}m"))
            p.update(b.params(f"L{l}b"))
        p.update(self.sc.params("sc"))
        return p

    def state_bytes(self):
        # n_layers * (2*M 复值 IIR 状态) + Sidecar 定容; 与 L 无关
        return int(self.n_layers * 2 * self.M * 16
                   + self.sc.Vc ** 2 * 8 + 4 * self.D * 8)

    def forward(self, ids, dts, target=None):
        L = len(ids)
        pos = np.minimum(np.arange(L), self.Lmax - 1)
        X = self.E[np.asarray(ids)]
        if self.P is not None:
            X = X + self.P[pos]
        dtn = np.asarray(dts, dtype=float)
        X = X + (dtn[:, None] - 1.0) * 0.02          # [N5] 两臂都加 dt 增广
        caches = []
        for m, b in zip(self.mixers, self.blocks):
            mo, mc = m.forward(X)
            X, bc = b.forward(X, mo)
            caches.append((mc, bc))
        feat, scc = self.sc.features(X, ids)
        logits = self.Wh @ feat + self.bh
        loss = None; dlog = None
        if target is not None:
            dlog, loss = softmax_ce(logits, target)
        return logits, loss, dict(dlog=dlog, caches=caches, scc=scc, feat=feat)

    def backward(self, c):
        dlog = c["dlog"]
        g = {"Wh": np.outer(dlog, c["feat"]), "bh": dlog.copy()}
        dX, dsc = self.sc.backward(c["scc"], self.Wh.T @ dlog)
        for l in reversed(range(self.n_layers)):
            mc, bc = c["caches"][l]
            dX_pre, db = self.blocks[l].backward(bc, dX)
            n_l = bc[-1]                               # 共享缩放因子
            beta_l = self.blocks[l].beta
            dX_from_mixer, dm = self.mixers[l].backward(mc, dX_pre * n_l * beta_l)
            dX = dX_pre + dX_from_mixer                # Block: h_raw = X + mixer(X)
            for k, v in db.items(): g[f"L{l}b.{k}"] = v
            for k, v in dm.items(): g[f"L{l}m.{k}"] = v
        dE = np.zeros_like(self.E); np.add.at(dE, self.E.__class__ and
                                              np.asarray(self._last_ids), dX)
        g["E"] = dE
        if self.P is not None:
            dP = np.zeros_like(self.P); np.add.at(dP, self._last_pos, dX)
            g["P"] = dP
        for k, v in dsc.items(): g[f"sc.{k}"] = v
        return g

    def set_ids(self, ids):
        self._last_ids = np.asarray(ids)
        self._last_pos = np.minimum(np.arange(len(ids)), self.Lmax - 1)


# 包装: forward 里自动记录 ids(避免改签名)
_orig_fwd = BaseArm.forward
def _fwd(self, ids, dts, target=None):
    self.set_ids(ids)
    return _orig_fwd(self, ids, dts, target)
BaseArm.forward = _fwd


def build_pair(cfg, opt, sd):
    kw = dict(D=cfg["D"], V=cfg["V"], n_layers=opt["layers"],
              H=int(opt.get("Hmult", 2) * cfg["D"]),
              seed=sd, Vc=cfg["Vc"], Lmax=384, M=opt["M"],
              alpha=opt.get("alpha", 0.0), beta=opt.get("beta", 1.0))
    return BaseArm(kind="sgc", **kw), BaseArm(kind="attn", **kw)


# =====================================================================
# 梯度校验 / 训练 / 评测
# =====================================================================
def grad_check(arm, ids, dts, target, rng, eps=1e-5):
    _, _, c = arm.forward(ids, dts, target)
    g = arm.backward(c); P = arm.params()
    worst, bad = 0.0, []
    for k in list(P)[:14]:
        flat = P[k].ravel()
        for i in rng.integers(0, flat.size, size=min(2, flat.size)):
            old = flat[i]
            flat[i] = old + eps; _, lp, _ = arm.forward(ids, dts, target)
            flat[i] = old - eps; _, lm, _ = arm.forward(ids, dts, target)
            flat[i] = old
            num = (lp - lm) / (2 * eps); ana = float(g[k].ravel()[i])
            err = abs(num - ana) / max(1.0, abs(num), abs(ana))
            worst = max(worst, err)
            if err > 1e-6:
                bad.append((k, int(i), round(num, 6), round(ana, 6), round(err, 8)))
    return worst, bad


def train_arm(arm, train, L, steps, seed, lr, curriculum):
    rng = np.random.default_rng(seed)
    opt = Adam(arm.params(), lr=lr)
    Ls = curriculum or (L,)
    for s in range(steps):
        opt.lr = lr * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        Li = min(int(Ls[int(rng.integers(0, len(Ls)))]), len(train) - 2)
        x, y, dts = sample_window(train, Li, rng)
        _, loss, c = arm.forward(x, dts, y)
        opt.step(arm.backward(c))
    return loss


def eval_arm(arm, test, L, seed, n, dt_mode="fixed"):
    rng = np.random.default_rng(seed)
    ce, used = 0.0, 0
    for _ in range(n):
        if len(test) <= L + 1: break
        x, y, dts = sample_window(test, L, rng, dt_mode)
        _, loss, _ = arm.forward(x, dts, y)
        ce += loss; used += 1
    return float(ce / max(used, 1) / math.log(2.0)), used


def measure(cfg, a, b, sd, n_eval, n_ood, ood_L):
    L = cfg["L"]
    v1, _ = eval_arm(a, cfg["test"], L, 7000 + sd, n_eval)
    v2, _ = eval_arm(b, cfg["test"], L, 7000 + sd, n_eval)
    a2 = {}
    for LL in ood_L:
        s1, _ = eval_arm(a, cfg["test"], LL, 7100 + sd, n_ood)
        s2, _ = eval_arm(b, cfg["test"], LL, 7100 + sd, n_ood)
        a2[f"L{LL}"] = dict(sgc=round(s1, 4), attn=round(s2, 4),
                            delta=round(s1 - s2, 4))
    t1, _ = eval_arm(a, cfg["test"], L, 7200 + sd, n_eval, "mixed")
    t2, _ = eval_arm(b, cfg["test"], L, 7200 + sd, n_eval, "mixed")
    return dict(A1=round(v1 - v2, 4), A1_abs=[round(v1, 4), round(v2, 4)],
                A2=a2, A2min=round(min(v["delta"] for v in a2.values()), 4),
                A3=round(t1 - t2, 4), A3_abs=[round(t1, 4), round(t2, 4)])


def axes_ok(m, strict_per_seed=False):
    return bool(m["A1"] <= 0.05 and m["A2min"] < -0.05 and m["A3"] <= 0.05)


# =====================================================================
# CI: 真·仓库 purity.py; 产品源码只含 SGC 侧(对照臂是研究对照, 不属产品线)
# =====================================================================
def ci_audit_v9(RP, cfg):
    checks = []
    def ck(name, cond, detail="", level="REAL_RUN"):
        checks.append(dict(name=name, passed=bool(cond), detail=str(detail)[:110], level=level))
    ck("CI-0 真·加载仓库 purity.py", RP is not None)
    src_all = open(os.path.join(_HERE, "complete_beyond_v9.py"), encoding="utf-8").read()
    tree = ast.parse(src_all)
    wanted = {"SGCMixer", "Block", "SharedSidecarV4", "fft_conv_fwd", "fft_conv_bwd",
              "rms_fwd", "rms_bwd", "gelu", "gelu_grad", "BaseArm"}
    product = "\n".join(ast.get_source_segment(src_all, n) or ""
                        for n in tree.body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted)
    # BaseArm 含 attn 分支但无禁用标识符; 仍单独验证扫描结果
    if RP:
        ck("CI-0a _identifier_hit('q_proj')", RP._identifier_hit("q_proj") is True)
        ck("CI-0b _identifier_hit('sgc_mixer')", RP._identifier_hit("sgc_mixer") is False)
        ck("CI-0c ATTENTION_SYMBOLS 表规模", len(RP.ATTENTION_SYMBOLS) >= 15,
           len(RP.ATTENTION_SYMBOLS))
        class C: mixer = "sgc"
        ck("CI-0d assert_pure_config(sgc) 真调用", RP.assert_pure_config(C()) is True)
        class C2: mixer = "attn"
        try: RP.assert_pure_config(C2()); bad = True
        except AssertionError: bad = False
        ck("CI-0e assert_pure_config(attn) 真被拒", not bad)
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "product.py"), "w", encoding="utf-8").write(product)
            hits = RP.scan_source_attention_free(td)
        ck("CI-0f 仓库 scan_source_attention_free 扫产品源码零命中", len(hits) == 0, hits)
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "product.py"), "w", encoding="utf-8").write(
                "class MixerA:\n    def __init__(self):\n        self.q_proj=1\n        self.k_proj=2\n")
            hits2 = RP.scan_source_attention_free(td)
        ck("CI-0g 仓库守卫抓到改名绕过", len(hits2) >= 2, hits2)
        class FakeMod:
            def named_modules(self): return [("m", self)]
            def named_parameters(self, recurse=False): return [("v_proj", None)]
        try: RP.assert_attention_free(FakeMod()); bad2 = True
        except AssertionError: bad2 = False
        ck("CI-0h assert_attention_free 抓 v_proj", not bad2)
    U = np.random.default_rng(1).normal(0, 1, (cfg["L"], cfg["M"]))
    K = kernel(make_basis(cfg["M"]), cfg["L"])
    y, cc = fft_conv_fwd(U, K)
    ref = V4.direct_causal_conv(U, make_basis(cfg["M"]))
    de = float(np.max(np.abs(y - ref)))
    ck("P0a FFT<->直接因果卷积 逐位等价", de < 1e-9, de, "IDENTITY")
    b = make_basis(cfg["M"])
    ck("P0b 核静态 Toeplitz 同 L 两次逐位相同",
       np.array_equal(kernel(b, 17), kernel(b, 17)))
    tb = ast.parse(product)
    writes = [ast.unparse(n) for n in ast.walk(tb)
              if isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del))
              and isinstance(n.value, ast.Attribute) and n.value.attr == "basis"]
    ck("P0c AST 零处写 self.basis[...]", len(writes) == 0, writes)
    # 双形式反传一致性(FFT 反传 vs 直接数值反传)
    dy = np.random.default_rng(2).normal(0, 1, (cfg["L"], cfg["M"]))
    du_fft = fft_conv_bwd(dy, cc)
    du_ref = np.zeros_like(U)
    for s in range(cfg["L"]):
        du_ref[s] = np.einsum("tm,tm->m", K[:cfg["L"] - s], dy[s:])
    de2 = float(np.max(np.abs(du_fft - du_ref)))
    ck("P0d FFT 反传<->直接反传 逐位等价", de2 < 1e-9, de2, "IDENTITY")
    passed = sum(c["passed"] for c in checks)
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}  {c['detail'][:60]}")
    return dict(total=len(checks), passed=passed, all_pass=passed == len(checks),
                checks=checks, dual_err=de, dual_bwd_err=de2)


# =====================================================================
# Phase A: 层数 x M 快筛(1 seed, 课程, 900 步)
# =====================================================================
def phase_a(cfg):
    rows = []
    for layers in (1, 2, 3, 4):
        for M in (64, 128):
            o = dict(layers=layers, M=M, alpha=0.0, lr=3e-3,
                     curriculum=(48, 96, 192), name=f"L{layers}_M{M}")
            a, b = build_pair(cfg, o, 11)
            rng = np.random.default_rng(9011)
            x, y, dts = sample_window(cfg["train"], cfg["L"], rng)
            e1, bad1 = grad_check(a, x, dts, y, rng)
            a, b = build_pair(cfg, o, 11)
            train_arm(a, cfg["train"], cfg["L"], 900, 1011, o["lr"], o["curriculum"])
            train_arm(b, cfg["train"], cfg["L"], 900, 1011, o["lr"], o["curriculum"])
            m = measure(cfg, a, b, 11, 100, 60, cfg["ood_L"])
            rows.append(dict(**o, grad=round(e1, 9), grad_ok=bool(e1 < 5e-3),
                             params_sgc=param_count(a), params_attn=param_count(b),
                             state_bytes=a.state_bytes(), **m))
            print(f"   {o['name']:10s} grad={e1:.1e} A1 {m['A1']:+.4f}  "
                  f"A2min {m['A2min']:+.4f}  A3 {m['A3']:+.4f}  "
                  f"p {param_count(a)}/{param_count(b)}  "
                  f"3轴={axes_ok(m)}", flush=True)
    return rows


# =====================================================================
# Phase B/C: 决定性(3 seeds, 课程, 可给对手 2x 步数)
# =====================================================================
def decisive(cfg, opt, seeds=(11, 22, 33), steps=4000, attn_scale=1.0, tag="final"):
    out = dict(tag=tag, opt=opt, steps=steps, attn_scale=attn_scale, per_seed={})
    for sd in seeds:
        a, b = build_pair(cfg, opt, sd)
        rng = np.random.default_rng(9000 + sd)
        x, y, dts = sample_window(cfg["train"], cfg["L"], rng)
        e1, bad1 = grad_check(a, x, dts, y, rng)
        e2, bad2 = grad_check(b, x, dts, y, rng)
        a, b = build_pair(cfg, opt, sd)
        train_arm(a, cfg["train"], cfg["L"], steps, 1000 + sd, opt["lr"], opt["curriculum"])
        train_arm(b, cfg["train"], cfg["L"], int(steps * attn_scale),
                  2000 + sd, opt["lr"], opt["curriculum"])
        m = measure(cfg, a, b, sd, cfg["n_eval"], cfg["n_eval_ood"], cfg["ood_L"])
        out["per_seed"][f"seed{sd}"] = dict(
            gradcheck=dict(sgc=round(e1, 9), attn=round(e2, 9),
                           passed=bool(e1 < 5e-3 and e2 < 5e-3),
                           bad=bad1[:2] + bad2[:2]),
            params=dict(sgc=param_count(a), attn=param_count(b)),
            state_bytes=a.state_bytes(), **m)
        print(f"   [{tag} s{sd}] grad={out['per_seed'][f'seed{sd}']['gradcheck']['passed']} "
              f"A1 {m['A1']:+.4f}  A2min {m['A2min']:+.4f}  A3 {m['A3']:+.4f}  "
              f"每seed3轴={axes_ok(m)}", flush=True)
    d1 = float(np.mean([r["A1"] for r in out["per_seed"].values()]))
    d2 = float(np.mean([r["A2min"] for r in out["per_seed"].values()]))
    d3 = float(np.mean([r["A3"] for r in out["per_seed"].values()]))
    per = [axes_ok(r) for r in out["per_seed"].values()]
    out["verdict"] = dict(
        A1=round(d1, 4), A2min=round(d2, 4), A3=round(d3, 4),
        per_seed_axes=per,
        A1_ok_mean=bool(d1 <= 0.05), A2_ok_mean=bool(d2 < -0.05), A3_ok_mean=bool(d3 <= 0.05),
        A1_ok_per_seed=all(r["A1"] <= 0.05 for r in out["per_seed"].values()),
        A2_ok_per_seed=all(r["A2min"] < -0.05 for r in out["per_seed"].values()),
        A3_ok_per_seed=all(r["A3"] <= 0.05 for r in out["per_seed"].values()),
        gradcheck_all=all(r["gradcheck"]["passed"] for r in out["per_seed"].values()))
    v = out["verdict"]
    v["QUALITY_COMPLETE"] = bool(v["A1_ok_mean"] and v["A2_ok_mean"] and v["A3_ok_mean"]
                                 and v["A1_ok_per_seed"] and v["A2_ok_per_seed"]
                                 and v["A3_ok_per_seed"] and v["gradcheck_all"])
    return out


# =====================================================================
# Phase D: 步数曲线(连续 >=3 checkpoint 三轴全过)
# =====================================================================
def steps_curve(cfg, opt, seeds=(11, 22, 33),
                checkpoints=(800, 1600, 2400, 3200, 4000, 6000, 8000),
                n_eval=110, n_ood=70, attn_scale=1.0):
    out = dict(opt=opt, checkpoints=list(checkpoints), attn_scale=attn_scale, per_seed={})
    total = max(checkpoints)
    for sd in seeds:
        a, b = build_pair(cfg, opt, sd)
        oa = Adam(a.params(), lr=opt["lr"]); ob = Adam(b.params(), lr=opt["lr"])
        ra = np.random.default_rng(1000 + sd); rb = np.random.default_rng(1000 + sd)
        Ls = opt["curriculum"]
        # 对手 2x 通过「每步多更新一次」近似(保持总 checkpoint 对齐)
        rec = {}
        for s in range(1, total + 1):
            for arm, o, rng_, mult in ((a, oa, ra, 1), (b, ob, rb, attn_scale)):
                o.lr = opt["lr"] * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * (s - 1) / total)))
                for _ in range(int(mult)):
                    Li = min(int(Ls[int(rng_.integers(0, len(Ls)))]), len(cfg["train"]) - 2)
                    xx, yy, dd = sample_window(cfg["train"], Li, rng_)
                    _, _, cc = arm.forward(xx, dd, yy)
                    o.step(arm.backward(cc))
            if s in checkpoints:
                rec[f"step{s}"] = measure(cfg, a, b, sd, n_eval, n_ood, cfg["ood_L"])
        out["per_seed"][f"seed{sd}"] = rec
        print(f"   [curve s{sd}] " + " ".join(
            f"{k[4:]}:{v['A1']:+.2f}/{v['A2min']:+.2f}/{v['A3']:+.2f}"
            for k, v in rec.items()), flush=True)
    agg = {}
    for k in [f"step{c}" for c in checkpoints]:
        rs = [out["per_seed"][f"seed{s}"][k] for s in seeds]
        A1 = float(np.mean([r["A1"] for r in rs]))
        A2 = float(np.mean([r["A2min"] for r in rs]))
        A3 = float(np.mean([r["A3"] for r in rs]))
        agg[k] = dict(A1=round(A1, 4), A2min=round(A2, 4), A3=round(A3, 4),
                      all3=bool(A1 <= 0.05 and A2 < -0.05 and A3 <= 0.05))
    out["aggregate"] = agg
    flags = [agg[f"step{c}"]["all3"] for c in checkpoints]
    runs, cur = [], 0
    for f in flags:
        cur = cur + 1 if f else 0
        runs.append(cur)
    out["stable_window"] = dict(longest_consecutive=max(runs), required=3,
                                stable_achieved=bool(max(runs) >= 3), flags=flags)
    return out


def deterministic_check(cfg, opt, steps=600):
    def one():
        a, b = build_pair(cfg, opt, 11)
        train_arm(a, cfg["train"], cfg["L"], steps, 1011, opt["lr"], opt["curriculum"])
        train_arm(b, cfg["train"], cfg["L"], steps, 1011, opt["lr"], opt["curriculum"])
        m = measure(cfg, a, b, 11, 40, 30, cfg["ood_L"])
        return [m["A1"], m["A2min"], m["A3"]]
    x, y = one(), one()
    diffs = [(i, x[i], y[i]) for i in range(3)
             if not math.isclose(x[i], y[i], abs_tol=1e-12)]
    return dict(identical=len(diffs) == 0, x=x, y=y, diffs=diffs, level="REAL_RUN")


def main():
    t0 = time.perf_counter()
    RP, _ = load_real_purity()
    train, test, V, _ = load_corpus()
    cfg = dict(D=32, V=V, L=48, Vc=64, M=64, n_eval=160, n_eval_ood=100,
               ood_L=(96, 192, 384), train=train, test=test)
    print(f"[data] train={len(train)} test={len(test)} vocab={V}", flush=True)

    print("[1/6] CI / 纯度 / 双形式(真仓库 purity.py) ...", flush=True)
    ci = ci_audit_v9(RP, cfg)
    print(f"   -> {ci['passed']}/{ci['total']} PASS", flush=True)

    print("[2/6] Phase A: 层数{1,2,3,4} x M{64,128} 快筛(课程, 1 seed, 900 步) ...", flush=True)
    grid = phase_a(cfg)
    grid.sort(key=lambda r: (r["A2min"], r["A1"], r["A3"]))
    print("   TOP4:", [(r["name"], r["A1"], r["A2min"], r["A3"]) for r in grid[:4]], flush=True)

    # 选 2 个候选: 综合最优 + A1 最优(不同则都跑)
    best = grid[0]
    by_a1 = min(grid, key=lambda r: r["A1"])
    cands = [best] + ([by_a1] if by_a1["name"] != best["name"] else [])
    def to_opt(r):
        return dict(layers=r["layers"], M=r["M"], alpha=r["alpha"], lr=r["lr"],
                    curriculum=r["curriculum"], name=r["name"])

    print("[3/6] Phase B: 决定性 4000 步 3 seeds(课程, 每 seed 分别判定) ...", flush=True)
    finals = {}
    for r in cands:
        o = to_opt(r)
        finals[o["name"]] = decisive(cfg, o, steps=4000, tag=o["name"])
        v = finals[o["name"]]["verdict"]
        print(f"   => {o['name']}: A1 {v['A1']:+.4f}  A2min {v['A2min']:+.4f}  "
              f"A3 {v['A3']:+.4f}  QUALITY_COMPLETE={v['QUALITY_COMPLETE']}", flush=True)

    bo = to_opt(best)
    print("[4/6] Phase C: 对手 2x 步数(更强对照) ...", flush=True)
    fair = decisive(cfg, bo, steps=4000, attn_scale=2.0, tag="attn2x")
    print(f"   => attn2x: A1 {fair['verdict']['A1']:+.4f}  "
          f"A2min {fair['verdict']['A2min']:+.4f}  A3 {fair['verdict']['A3']:+.4f}  "
          f"QUALITY_COMPLETE={fair['verdict']['QUALITY_COMPLETE']}", flush=True)

    print("[5/6] Phase D: 步数曲线至 8000(稳定区间 >=3) ...", flush=True)
    curve = steps_curve(cfg, bo)
    for c in curve["checkpoints"]:
        a = curve["aggregate"][f"step{c}"]
        print(f"   step{c:>5}: A1 {a['A1']:+.4f}  A2min {a['A2min']:+.4f}  "
              f"A3 {a['A3']:+.4f}  3轴={a['all3']}", flush=True)
    print(f"   稳定区间 最长连续 {curve['stable_window']['longest_consecutive']} "
          f"(要求>=3) -> {curve['stable_window']['stable_achieved']}", flush=True)

    print("[6/6] 确定性 + 汇总 ...", flush=True)
    det = deterministic_check(cfg, bo)
    q_ok = [k for k, v in finals.items() if v["verdict"]["QUALITY_COMPLETE"]]
    stable = curve["stable_window"]["stable_achieved"]
    COMPLETE = bool(q_ok and stable and fair["verdict"]["QUALITY_COMPLETE"]
                    and ci["all_pass"] and det["identical"])

    report = dict(
        fingerprint=dict(master_seed=MASTER, python=platform.python_version(),
                         numpy=np.__version__, platform=platform.platform(),
                         script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
                         corpus_bytes=len(train) + len(test), vocab=V),
        protocol="P0 核不动 / P1 同层数同Block同旁路(唯一差别mixer) / P2 三轴每seed / "
                 "P3 连续>=3 checkpoint / P4 成本可行性 / P5 真仓库CI / P6 梯度+确定性",
        self_correction="v8 的『静态核永远补不上随输入变化』只对单层成立; "
                        "多层静态卷积+输入依赖门 U=A*sigma(B)+层间非线性可产生数据依赖有效交互。"
                        "本轮即测此漏洞。",
        new_freedoms=["N1 多层堆叠 1-4", "N2 共享 pre-norm Block(构造性公平)",
                      "N3 FFN(gelu) 两臂同隐宽", "N4 FFT 前向+反传 O(L log L M)",
                      "N5 dt 增广两臂都加", "N6 长度课程两臂都做"],
        ci=dict(passed=ci["passed"], total=ci["total"], all_pass=ci["all_pass"],
                dual_err=ci["dual_err"], dual_bwd_err=ci["dual_bwd_err"]),
        phase_a_grid=grid, phase_b_final=finals, phase_c_attn2x=fair,
        phase_d_steps_curve=curve, deterministic=det,
        deterministic_feasibility=deterministic_feasibility(cfg["D"]),
        final=dict(
            COMPLETE_BEYOND=COMPLETE,
            quality_complete_by=q_ok,
            quality_complete_under_2x_opponent=bool(fair["verdict"]["QUALITY_COMPLETE"]),
            stable_window=curve["stable_window"],
            best_quality=finals[bo["name"]]["verdict"] if bo["name"] in finals else None,
            best_config=bo,
            state_bytes_best=BaseArm(cfg["D"], cfg["V"], bo["layers"], 2 * cfg["D"],
                                     11, cfg["Vc"], 384, bo["M"], kind="sgc").state_bytes(),
            params_best=finals[bo["name"]]["per_seed"]["seed11"]["params"]
            if bo["name"] in finals else None,
            ci_pass=ci["all_pass"], deterministic=det["identical"],
        ),
        wall_s=round(time.perf_counter() - t0, 2),
    )
    with open(os.path.join(_HERE, "complete_beyond_v9_results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("DONE -> complete_beyond_v9_results.json", flush=True)
    print("FINAL:", json.dumps(report["final"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
