#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v9.3 · 真·梯度微调（CE + Adam），跨过「解码期编排」这条线

## 为什么前 7 版都不算「真正微调模型」

v9.0–v9.2c 全部在**推理时**挂 `logit_bias_fn`：按位置 k 切策略、按语言给
前缀 token 集合、relative 硬抬到 top1 之上、recent-3 抑制、基座身份负惩罚……
这些本质仍是「解码期编排」。用户从第一轮就点破过：
> 不是只做提示词层面的最表层，而是真正去自己做一个新方案，来在 CPU 上微调模型

v9.2c 机判 passed=true，但语义重判只有 1/4 有效：
`青奈科技是开发MiniCPM系列模型的公司。我是一个MiniCPM` ——
关键词命中，语义完全颠倒。**关键词 ≠ 身份**。

## v9.3 做了什么（本质区别）

只训一个**输出层权重残差** `Δ ∈ R^{|C|×d}`，写进 `identity_head_delta.npz`：

    logits_c(h) = teacher_logits_c(h) + g(h) · (Δ_c · ĥ)      for c ∈ C
    logits_v(h) = teacher_logits_v(h)                          for v ∉ C

    g(h) = 1[ σ(κ(ĥ·u − b)) ≥ 0.5 ]     ← u,b,κ 是 v9.0 学到的门控，冻结

训练：标准 **next-token cross-entropy + Adam**，不是闭式 ridge：

    L = mean_i CE( softmax(M_i + g_i·onehot_C(Δ ĥ_i)), t_i ) + λ‖Δ‖²
    M_i = teacher logits = W_lm_head · h_i      （全词表 130560，预计算一次）
    ∂L/∂Δ_c = mean_i g_i·(P_i,c − Y_i,c)·ĥ_i + 2λΔ_c

候选集 C **数据驱动**：8 条 expert 答案的前 6 个 token（26 个）
∪  teacher 在这 144 个 hidden 上的真实 top-5 竞争 token（主要是 `<think>`、
`MiniCPM` 首 piece 等）—— 不手工列词表。

推理期**零编排**：没有位置切换、没有 prefix/anchor 集合、没有 margin 硬抬、
没有 recent 抑制、没有负惩罚。只有「一个学到的矩阵乘 + 一个学到的门控」，
对所有位置、所有语言完全一样。非身份 g=0 ⇒ Δ 严格 0 ⇒ 逐位零回归。

## OOM（用户硬要求）

* **训练阶段完全不 preload 2.01 GiB teacher**：只用已落盘的 hidden 缓存 +
  按需读 `lm_head.weight`（峰值 ~1.0 GiB），比之前每一轮都低；
* 训练结束 `del W, M` + `gc.collect()` + 清分片缓存，**才**进入验证阶段
  preload 全量；两阶段严格分离；
* 单进程串行 + /proc 精确并行检测 + MemAvailable 守卫。

## 诚实边界

* 仍是**只改 lm_head 的 |C| 行**，不是梯度反传穿过 24 层 transformer；
  纯 numpy CPU 上后者不可行。门控 u,b,κ 冻结自 v9.0，本轮不重训。
* 训练目标是让 expert token 在 teacher-forced / DAgger hidden 上占优，
  exposure bias 只能被 DAgger 缓解、不能数学消除。
* 不声称完整准确复述 FACTS 全句、能力更强、ppl 改善。
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import struct
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_minicpm import (VirtualWeights, MiniCPMTokenizer, TeacherLM,  # noqa: E402
                         shard_shas, BASE)
from scale_theory import decode_gate  # noqa: E402
from identity_alignment import sig  # noqa: E402
from identity_full_sft import (SAMPLES, PROBES, KW, chatml, norm, guard,  # noqa: E402
                               HIDDEN, STOP_IDS)

OUT = "/data/workspace/output"
ALIGN = os.path.join(OUT, "identity_alignment.npz")
DELTA = os.path.join(OUT, "identity_head_delta.npz")
VERIFY = os.path.join(OUT, "identity_gradient_verify.json")
REAUDIT = os.path.join(OUT, "identity_semantic_reaudit.json")

K = 6
EPOCHS = 200
LR = 0.015
L2 = 2e-3
MAX_NEW = 16
COMPETITOR_TOPK = 5

BASE_ID = ["MiniCPM", "面壁", "ModelBest", "OpenBMB", "清华", "北大",
           "Qwen", "ChatGPT", "OpenAI", "通义", "混元", "文心",
           "学术研究机构", "研究机构", "大型语言模型"]
OWN = ["青奈", "青小团", "青小団", "Qingnai", "Qingxiaotuan",
       "Tuan-Models", "Tuan", "六安"]


def semantic_judge(text):
    import re
    leak = [w for w in BASE_ID if w in text]
    own = [w for w in OWN if w in text]
    inverted = bool(re.search(
        r"(我|模型).{0,12}(是|为).{0,8}(MiniCPM|Qwen|面壁|GPT)", text)) or \
        bool(re.search(
            r"青奈.{0,10}(开发|研发).{0,8}(MiniCPM|面壁|ModelBest)", text))
    return dict(leak=leak, own=own, inverted=inverted,
                valid=bool(own) and not leak and not inverted)


# ---------------------------------------------------------------------------
# 阶段 1：训练（不 preload teacher）
# ---------------------------------------------------------------------------
def build_pairs(tok):
    pairs = []
    srcs = [("teacher_forced", HIDDEN),
            ("dagger_r1", os.path.join(OUT, "identity_dagger_r1.json")),
            ("dagger_r2", os.path.join(OUT, "identity_dagger_r2.json"))]
    for name, p in srcs:
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for key, row in sorted(d.items()):
            hs, tg = row["hidden"], row["target_tokens"]
            if len(hs) < K or len(tg) < K:
                continue
            for k in range(K):
                pairs.append((np.asarray(hs[k], np.float64),
                              int(tg[k]), name))
    print(f"[data] {len(pairs)} identity pairs from {len(srcs)} sources",
          flush=True)
    # 非身份 hidden（只用于零回归验证，训练 g 硬置 0 ⇒ 零梯度）
    cache = json.load(open(os.path.join(OUT, "identity_cache.json"),
                           encoding="utf-8"))
    neg = [np.asarray(cache[k], np.float64)
           for k in cache if k.startswith("non|")]
    print(f"[data] {len(neg)} non-identity hidden for zero-regression check",
          flush=True)
    return pairs, neg


def train(head_rows, pairs, neg, u, b, kappa, d, tok):
    # float32：M=(N,V) 从 143 MiB 降到 72 MiB，softmax 中间也省一半
    H = np.stack([p[0] for p in pairs]).astype(np.float32)
    T = np.asarray([p[1] for p in pairs], np.int64)
    g_nat = np.asarray([1.0 if sig(kappa * (float(np.asarray(h, np.float64)
                                                    @ u) - b)) >= 0.5
                        else 0.0 for h in H])
    # 训练时身份样本的监督门硬置 1：这些 (h,t) 全部来自 expert 答案，
    # 本来就该被学进去。DAgger 的 off-policy hidden 有 49/144 天然 g=0
    # （生成偏离身份方向），若按自然门会把它们整批丢掉 —— 那正是要补的数据。
    # 推理时才用冻结门控判 0/1，负责非身份逐位零回归。
    g = np.ones(len(H), np.float64)
    print(f"[gate] natural g=1 on {int(g_nat.sum())}/{len(H)} identity pairs; "
          "training gate forced to 1 for all (inference still hard-gated)",
          flush=True)
    neg_g = np.asarray([sig(kappa * (float(h @ u) - b)) for h in neg])
    print(f"[gate] non-identity g max = {neg_g.max():.4f} "
          f"(hard-gated to 0 at inference)", flush=True)

    # 全词表 teacher logits：一次 BLAS gemm，之后固定
    t0 = time.time()
    M = (H @ head_rows.T).astype(np.float32)            # (N, V)
    print(f"[train] M = {M.shape} ({M.nbytes/2**20:.1f} MiB) "
          f"computed in {time.time()-t0:.1f}s", flush=True)

    # 数据驱动的候选集 C = expert tokens ∪ teacher 真实 top-k 竞争者
    expert = sorted(set(int(t) for t in T))
    comp = set()
    for i in range(len(M)):
        comp.update(int(x) for x in np.argpartition(M[i], -COMPETITOR_TOPK)
                    [-COMPETITOR_TOPK:])
    C = sorted(set(expert) | comp)
    ci = {t: j for j, t in enumerate(C)}
    nC = len(C)
    print(f"[train] |C| = {nC} (expert {len(expert)} + competitor "
          f"{len(comp) - len(set(expert) & comp)}) -> "
          f"{[tok.decode([t], skip_special=True) for t in C[:14]]} …",
          flush=True)
    del comp, expert
    Cidx = np.asarray(C, np.int64)

    # v9.3b 关键修复：不能用全 130560 词表 softmax。
    # 只有 |C| 行可训，其余 130258 个固定 teacher logits 的 exp 之和会把
    # P(target) 稀释到 1e-4 → CE≈8.8、梯度几乎全被常数项吃掉（v9.3a 实测
    # loss 11.7→8.82 但 full-vocab acc 只有 0.646 且随 epoch 振荡）。
    # 改成【受限对比 softmax】：logits = [ M_C + Δ项 , m_i ]
    #   m_i = max_{v ∉ C} M_i[v]，即 teacher 当前最强的「未受控」token，
    #   作为固定常数项参与归一化但不参与梯度。
    # 优化目标因此精确变成「target 压过 teacher 最强竞争者」，
    # 这正是推理时 argmax 需要的条件。
    Cmask = np.ones(M.shape[1], bool)
    Cmask[Cidx] = False
    m_const = M[:, Cmask].max(1).astype(np.float64)      # (N,)
    print(f"[train] restricted contrastive softmax over |C|+1 = {nC+1} "
          f"logits; m_const range [{m_const.min():.2f}, "
          f"{m_const.max():.2f}]", flush=True)
    del Cmask

    # Adam 状态
    Delta = np.zeros((nC, d), np.float64)
    m = np.zeros_like(Delta); v = np.zeros_like(Delta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    Y = np.zeros((len(H), len(C)), np.float64)
    for i, t in enumerate(T):
        Y[i, ci[int(t)]] = 1.0

    tcol = np.asarray([ci[int(t)] for t in T], np.int64)
    best_acc, best_Delta, best_ep = -1.0, None, 0
    hist = []
    for ep in range(1, EPOCHS + 1):
        add = (H @ Delta.T) * g[:, None]                # (N, nC)
        base = M[:, Cidx].astype(np.float64)            # (N, nC)
        base += add
        L = np.concatenate([base, m_const[:, None]], axis=1)   # (N, nC+1)
        L -= L.max(1, keepdims=True)
        np.exp(L, out=L)
        L /= L.sum(1, keepdims=True)
        Pc = L[:, :nC]                                  # (N, nC)
        ce = -np.log(np.clip(Pc[np.arange(len(H)), tcol], 1e-12, 1.0))
        loss = float(ce.mean()) + L2 * float((Delta ** 2).sum())
        G = (Pc - Y) * g[:, None] / len(H)             # (N, nC)
        grad = G.T @ H.astype(np.float64) + 2 * L2 * Delta
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        mh = m / (1 - beta1 ** ep); vh = v / (1 - beta2 ** ep)
        Delta -= LR * mh / (np.sqrt(vh) + eps)
        acc = float(np.mean(base.argmax(1) == tcol))
        if acc > best_acc:                              # 防后期振荡
            best_acc, best_ep = acc, ep
            best_Delta = Delta.copy()
        if ep in (1, 10, 40, 80, 120, 160, EPOCHS):
            hist.append(dict(epoch=ep, loss=round(loss, 5),
                             train_acc=round(acc, 4)))
            print(f"  ep {ep:3d}/{EPOCHS}  loss={loss:.5f}  "
                  f"contrastive acc={acc:.4f}", flush=True)
        del add, base, L, Pc, G, grad
    Delta = best_Delta
    print(f"[train] best snapshot @ ep {best_ep} "
          f"contrastive acc={best_acc:.4f}", flush=True)
    # 最终 teacher 全词表 argmax 命中（真实指标，不只是 C 内）
    add = (H @ Delta.T) * g[:, None]
    L = M.copy(); L[:, Cidx] += add
    full_acc = float(np.mean([int(L[i].argmax()) == int(T[i])
                              for i in range(len(H))]))
    print(f"[train] FINAL full-vocab argmax acc = {full_acc:.4f}",
          flush=True)
    # 释放大矩阵，验证阶段才 preload teacher
    del M, L, add, H, Y, m, v, mh, vh, best_Delta, m_const
    gc.collect()
    return Delta.astype(np.float32), C, dict(
        epochs=EPOCHS, lr=LR, l2=L2, n_pairs=len(pairs),
        best_epoch=best_ep, contrastive_train_acc=round(best_acc, 4),
        loss_objective="restricted contrastive CE over |C|+1 logits "
                       "(target vs strongest uncontrolled teacher token)",
        n_candidates=nC, n_params=int(nC * d),
        param_pct_of_1B=round(100.0 * nC * d / 1e9, 5),
        full_vocab_train_acc=round(full_acc, 4), history=hist,
        identity_natural_gate_on=int(g_nat.sum()),
        training_gate_forced_on=True,
        non_identity_gate_max=round(float(neg_g.max()), 4))


# ---------------------------------------------------------------------------
# 阶段 2：验证（零编排，只有学到的线性残差 + 冻结门控）
# ---------------------------------------------------------------------------
def make_weight_hook(Delta, C, u, b, kappa, trace):
    ci = {int(t): j for j, t in enumerate(C)}

    def hook(h):
        hraw = np.asarray(h, np.float64)
        hh = norm(h)
        gv = sig(kappa * (float(np.asarray(hh, np.float64) @ u) - b))
        g = 1.0 if gv >= 0.5 else 0.0
        trace.append(dict(g=round(gv, 5), hard_gate=g,
                          hnorm=round(float(np.linalg.norm(hraw)), 3)))
        if g == 0.0:
            return None                     # 严格逐位零回归
        # v9.3c 尺度对齐（v9.3b 失败根因）：
        # 训练用归一化 hidden ⇒ M = ĥ·Wᵀ、add = ĥ·Δᵀ，都在 O(1) 量级；
        # 推理时 teacher logits 却是 W·h_raw = ‖h‖·(ĥ·Wᵀ)，量级 ×‖h‖。
        # 若仍把 Δ·ĥ（O(1)）加进 W·h_raw（O(‖h‖)），推力被淹没 →
        # 4/4 身份 probe 退回 <think>。直接用 Δ·h_raw = ‖h‖·(Δ·ĥ)，
        # 两边同乘 ‖h‖，argmax 关系与训练完全一致。
        z = Delta.astype(np.float64) @ hraw
        return {t: float(z[j]) for t, j in ci.items()}

    def arm():
        trace.clear()
    return hook, arm


def generate(lm, tok, hook, arm, prompt, V_size, max_new=MAX_NEW):
    arm()
    ids, _ = tok.encode(prompt, add_bos=False)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    Lp = len(ids); r = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                r = lm.step(int(tid), pos, kvs,
                            need_logits=(pos == Lp - 1),
                            fusion_hook=None, logit_bias_fn=hook)
            out = [int(r[0])]
            fin = [bool(getattr(lm, "last_logits_finite", True))]
            for j in range(1, max_new):
                r = lm.step(int(out[-1]), Lp + j - 1, kvs,
                            need_logits=True, fusion_hook=None,
                            logit_bias_fn=hook)
                out.append(int(r[0]))
                fin.append(bool(getattr(lm, "last_logits_finite", True)))
                if r[0] in STOP_IDS:
                    break
    return dict(text=tok.decode(out, skip_special=True), ids=out,
                gate=decode_gate(out, V_size, all(fin)),
                logits_finite=all(fin), n_new=len(out))


def main():
    t0 = time.time()
    guard(2.6)
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    d, V_size = cfg["hidden_size"], cfg["vocab_size"]
    A0 = np.load(ALIGN, allow_pickle=True)
    u = A0["u"].astype(np.float64)
    b, kappa = float(A0["b"]), float(A0["kappa"])

    part0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    V = VirtualWeights(part0, shard_shas(), hdr, 8 + n)
    tok = MiniCPMTokenizer(os.path.join(BASE, "tokenizer.json"))

    # ---- 阶段 1：只读 lm_head，不 preload teacher ----
    print("[stage1] reading lm_head rows only (NO 2.01 GiB preload)",
          flush=True)
    W = V.rows_f32("lm_head.weight", 0, V_size).astype(np.float32)
    n_sh = V.n_fetched
    V.parts.clear()                 # 立刻释放：否则 W 与 parts 各占 765 MiB
    gc.collect()
    print(f"[stage1] W={W.shape} {W.nbytes/2**20:.1f} MiB | "
          f"shards_touched={n_sh} | parts cache cleared", flush=True)
    pairs, neg = build_pairs(tok)
    Delta, C, train_info = train(W, pairs, neg, u, b, kappa, d, tok)
    del W
    V.parts.clear()
    gc.collect()
    print(f"[stage1] released W+M; peak-safe for stage2", flush=True)

    np.savez(DELTA, delta=Delta,
             candidate_tokens=np.asarray(C, np.int32),
             u=u.astype(np.float32), b=np.float32(b),
             kappa=np.float32(kappa), K=np.int32(K),
             epochs=np.int32(EPOCHS), lr=np.float32(LR), l2=np.float32(L2),
             mode=np.asarray("learned_lm_head_row_residual + frozen learned "
                             "hard gate; NO decode-time orchestration"),
             method=np.asarray("next-token CE + Adam on full-vocab teacher "
                               "logits with gated sparse lm_head residual"),
             created=np.asarray(time.strftime("%Y-%m-%d %H:%M:%S")))
    sha = hashlib.sha256(open(DELTA, "rb").read()).hexdigest()
    print(f"[delta] {DELTA} {os.path.getsize(DELTA)/2**10:.1f} KiB "
          f"sha256={sha[:16]}…", flush=True)

    # ---- 阶段 2：preload teacher，零编排验证 ----
    print("\n[stage2] preload teacher for end-to-end verification",
          flush=True)
    nf, nb = V.preload()
    print(f"[pull] {nf} shards / {nb/2**30:.2f} GiB", flush=True)
    lm = TeacherLM(V, cfg)
    trace = []
    hook, arm = make_weight_hook(Delta, C, u, b, kappa, trace)

    rows = []
    for lang, q, tag in PROBES:
        g = generate(lm, tok, hook, arm, chatml(q), V_size)
        ids = g["ids"]
        run = best = 1
        for x, y in zip(ids, ids[1:]):
            run = run + 1 if x == y else 1
            best = max(best, run)
        bg = {}
        for x, y in zip(ids, ids[1:]):
            bg[(x, y)] = bg.get((x, y), 0) + 1
        div = len(set(ids)) / max(len(ids), 1)
        hits = [k for k in KW if k in g["text"]]
        s = semantic_judge(g["text"])
        g0 = trace[0]["g"] if trace else 0.0
        rows.append(dict(lang=lang, tag=tag, q=q, text=g["text"],
                         identity_keywords=hits, n_keywords=len(hits),
                         max_token_run=best, token_diversity=round(div, 4),
                         max_bigram_repeat=max(bg.values()) if bg else 0,
                         semantic=s, semantically_valid=s["valid"],
                         first_g=round(g0, 4), gate_open=bool(g0 >= 0.5),
                         logits_finite=g["logits_finite"],
                         decode_gate=g["gate"]["passed"],
                         think_prefix=g["text"].lstrip().startswith("<think>"),
                         verbatim_training_answer=any(
                             g["text"].strip() == a for _, _, a in SAMPLES),
                         n_new=g["n_new"]))
        print(f"  [{lang:3s} {tag:12s}] g0={g0:.3f} div={div:.3f} "
              f"kw={hits} sem={s['valid']} -> {g['text'][:72]!r}", flush=True)

    ident = [r for r in rows if r["tag"] == "identity"]
    non = [r for r in rows if r["tag"] == "non_identity"]
    kw_hits = sum(1 for r in ident if r["n_keywords"] >= 1)
    sem_hits = sum(1 for r in ident if r["semantically_valid"])
    v = dict(
        n_probes=len(rows),
        identity_keyword_hits=kw_hits,
        identity_keyword_hit_rate=round(kw_hits / max(len(ident), 1), 4),
        semantically_valid=sem_hits,
        semantic_hit_rate=round(sem_hits / max(len(ident), 1), 4),
        unique_identity_texts=len({r["text"] for r in ident}),
        not_a_fixed_template=bool(len({r["text"] for r in ident}) >= 3),
        no_verbatim_training_answer=all(
            not r["verbatim_training_answer"] for r in rows),
        min_token_diversity=round(min(r["token_diversity"]
                                      for r in ident), 4) if ident else 0,
        max_bigram_repeat=max(r["max_bigram_repeat"]
                              for r in ident) if ident else 0,
        max_token_run=max(r["max_token_run"] for r in ident) if ident else 0,
        no_degenerate_repeat=bool(
            min(r["token_diversity"] for r in ident) >= 0.55
            and max(r["max_bigram_repeat"] for r in ident) <= 2
            and max(r["max_token_run"] for r in ident) < 5) if ident else False,
        base_identity_leaks=sum(len(r["semantic"]["leak"]) for r in ident),
        semantic_inversions=sum(1 for r in ident
                                if r["semantic"]["inverted"]),
        identity_think_prefixes=sum(1 for r in ident if r["think_prefix"]),
        all_identity_gates_open=all(r["gate_open"] for r in ident),
        all_non_identity_gates_closed=all(not r["gate_open"] for r in non),
        non_identity_leaks=sum(r["n_keywords"] for r in non),
        all_finite=all(r["logits_finite"] for r in rows),
        all_decode_gate=all(r["decode_gate"] for r in rows),
    )
    v["passed"] = bool(kw_hits >= 3 and sem_hits >= 3
                       and v["not_a_fixed_template"]
                       and v["no_verbatim_training_answer"]
                       and v["no_degenerate_repeat"]
                       and v["base_identity_leaks"] == 0
                       and v["semantic_inversions"] == 0
                       and v["identity_think_prefixes"] == 0
                       and v["all_identity_gates_open"]
                       and v["all_non_identity_gates_closed"]
                       and v["non_identity_leaks"] == 0
                       and v["all_finite"] and v["all_decode_gate"])

    rep = dict(
        title="身份认同 · 真梯度输出层微调（v9.3c，尺度对齐 + 推理期零编排）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        question="能否跨过 v9.1 天花板，并且不再是解码期编排？",
        why_this_is_real_finetuning=
            "标准 next-token CE + Adam 训练 `Δ ∈ R^{|C|×d}`，"
            "写进 identity_head_delta.npz；推理期只有一个学到的线性残差 + "
            "冻结的学到的硬门控，没有位置切换 / 锚集合 / margin 硬抬 / "
            "recent 抑制 / 负惩罚。非身份 g=0 ⇒ 严格逐位零回归。",
        previous_versions_audit=REAUDIT,
        v93a_failure_recorded=dict(
            what="v9.3a 用全 130560 词表 CE，但只有 |C|=302 行可训",
            observed="loss 11.74→8.82，full-vocab acc 只到 0.646 且随 epoch 振荡",
            diagnosis="其余 130258 个固定 teacher logits 的 exp 之和把 P(target) "
                      "稀释到 ~1e-4，梯度几乎全被常数项吃掉",
            fix="v9.3b 改受限对比 softmax：logits=[M_C+Δ项, m_const]，"
                "m_const=max_{v∉C} M_v 为固定常数项，目标精确变成"
                "「target 压过 teacher 最强未受控竞争者」+ best-snapshot"),
        v93b_failure_recorded=dict(
            what="v9.3b 训练收敛（contrastive acc 0.944 / full-vocab 0.931），"
                 "但 4/4 身份 probe 退回 <think>、关键词命中 0",
            diagnosis="训练用归一化 hidden（M=ĥ·Wᵀ、add=ĥ·Δᵀ 均 O(1)），"
                      "推理 teacher logits 却是 W·h_raw=‖h‖·(ĥ·Wᵀ)；"
                      "把 O(1) 的 Δ·ĥ 加进 O(‖h‖) 的 raw logits ⇒ 推力被淹没",
            fix="v9.3c 改用 Δ·h_raw = ‖h‖·(Δ·ĥ)，两边同乘 ‖h‖，"
                "argmax 关系与训练完全一致；门控仍用 ĥ 方向"),
        train=train_info,
        delta=dict(path=DELTA, sha256=sha,
                   bytes=os.path.getsize(DELTA),
                   shape=list(Delta.shape)),
        verification=v, probe_rows=rows,
        verdict=("真梯度权重微调后，多语言身份 probe 在零编排下自然说出身份、"
                 "无基座身份泄漏、无语义颠倒、文本多样度达标、非身份逐位零回归"
                 " —— 跨过天花板且不再是提示词层。"
                 if v["passed"] else "仍未通过全部判据，见 verification"),
        honest_boundaries=[
            "只改 lm_head 的 |C| 行，不是梯度反传穿过 24 层 transformer；"
            "纯 numpy CPU 上后者不可行",
            "门控 u,b,κ 冻结自 v9.0，本轮不重训",
            "训练目标是 expert token 占优；exposure bias 被 DAgger 缓解但未数学消除",
            "只引导语义方向，不保证逐字复述 FACTS 全句",
            "无 ppl / KL / 配对 CI / 能力结论；β_l 仍待 L2 校准",
            "'上下文理论无限' 是品牌陈述非实测",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(VERIFY, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n[verify]", json.dumps(v, ensure_ascii=False), flush=True)
    print("[done]", VERIFY, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
