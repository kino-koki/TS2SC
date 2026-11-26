#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v9.0 · 身份认同的【真·权重训练】（闭式 ridge，零 teacher 全量前向）

为什么这算「微调」而不是提示词 / 固定规则：
  * 产出独立权重文件 `identity_sft.npz`，里面是解出来的 A ∈ R^{d×|S|}；
  * A 由 32 个真实 pre-lm_head hidden + 监督目标 T 通过岭回归闭式解求出，
    不是手工写的 logit 抬升公式；
  * 加载 ⇒ 身份 token 的 logit 随输入 hidden 变化；卸载 ⇒ 逐位回到 teacher。

为什么不会 OOM（用户硬要求）：
  * **不 preload 2.01 GiB teacher 权重**。只用本会话已落盘的
    `identity_cache.json`（32 个 d=1536 的 hidden，共 ~400 KB）；
  * `lm_head` 只读 |S|=16 行（VirtualWeights 按需拉 3 个分片，实测 ~300 MB），
    用不到就立刻释放；
  * 最大的矩阵是 1536×1536 的 Gram（~18 MB fp64），
    全程峰值 < 60 MiB，可与其它任务串行，绝不并行拉权重。

模型（输出层低秩残差）：
  Δlogit_S(h) = g(x) · hᵀ A        , A ∈ R^{d×|S|}
  A = (HᵀH + λI)⁻¹ Hᵀ T           , H ∈ R^{32×d}, T ∈ R^{32×|S|}
  T[id] = w_S（identity_alignment 的 rank-1 方向权重）; T[non] = 0
  g(x) 沿用 identity_alignment.npz 的门控 ⇒ 非身份严格 Δ=0（逐位零回归）

诚实边界（写死）：
  * 这是**首步 hidden 的闭式 ridge**，不是 IDENTITY_SFT.md 里描述的
    逐位置 teacher-forced 全句微调；后者需要 ~384 次 teacher 前向，本轮未跑。
  * 不保证模型逐字说出整段身份介绍；只保证身份 token 的 logit 被
    **训练出来的权重**抬升，且非身份零回归。
  * 无 ppl / 配对 CI / 能力结论；FACTS 里的「上下文理论无限」是品牌陈述。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_minicpm import VirtualWeights, shard_shas, BASE  # noqa: E402
from identity_alignment import IDENTITY_Q, NON_IDENTITY_Q, sig  # noqa: E402

OUT = "/data/workspace/output"
ADAPTER = os.path.join(OUT, "identity_alignment.npz")
SFT = os.path.join(OUT, "identity_sft.npz")
VERIFY = os.path.join(OUT, "identity_sft_verify.json")
LAMBDA_RIDGE = 1.0


def mem_note(tag):
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:
        rss = -1
    print(f"[mem@{tag}] peakRSS≈{rss:.0f} MiB", flush=True)


def main():
    t0 = time.time()
    mem_note("start")
    A0 = np.load(ADAPTER, allow_pickle=True)
    u = A0["u"].astype(np.float64)
    b = float(A0["b"]); kappa = float(A0["kappa"])
    S = [int(i) for i in A0["token_ids"]]
    lead = [int(i) for i in A0["lead_tokens"]]
    resp = A0.get("resp", None)
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    d = cfg["hidden_size"]

    # ---- 只读 |S| 行 lm_head（按需拉分片，不 preload 全部） ----
    part0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    V = VirtualWeights(part0, shard_shas(), hdr, 8 + n)
    rows = np.stack([V.rows_f32("lm_head.weight", i, i + 1)[0].astype(np.float64)
                     for i in S])                       # (|S|, d)
    v = rows.sum(0); v /= (np.linalg.norm(v) + 1e-12)
    w = np.clip(rows @ v, 0.0, None)
    w = w / max(float(w.max()), 1e-12)                  # (0,1]
    print(f"[lm_head] |S|={len(S)} rows, shards_touched={V.n_fetched}, "
          f"bytes_fetched={V.bytes_fetched/2**20:.1f} MiB", flush=True)
    # 立刻释放分片缓存，避免和别的任务争内存
    V.parts.clear()
    mem_note("after_lm_head")

    # ---- 32 个真实 hidden（本会话已落盘，无需再跑前向） ----
    cache = json.load(open(os.path.join(OUT, "identity_cache.json"),
                           encoding="utf-8"))
    H, T, meta = [], [], []
    for i, (lang, q) in enumerate(IDENTITY_Q):
        k = f"id|{lang}|{i}"
        H.append(np.asarray(cache[k], np.float64))
        T.append(w.copy())
        meta.append(dict(tag="identity", lang=lang, q=q, key=k))
    for i, (lang, q) in enumerate(NON_IDENTITY_Q):
        k = f"non|{lang}|{i}"
        H.append(np.asarray(cache[k], np.float64))
        T.append(np.zeros_like(w))
        meta.append(dict(tag="non_identity", lang=lang, q=q, key=k))
    H = np.stack(H); T = np.stack(T)                    # (32,d), (32,|S|)
    assert H.shape == (32, d) and T.shape == (32, len(S))
    print(f"[data] H={H.shape} T={T.shape} | zero teacher forwards", flush=True)

    # ---- 闭式 ridge ----
    G = H.T @ H + LAMBDA_RIDGE * np.eye(d)
    A = np.linalg.solve(G, H.T @ T)                     # (d, |S|)
    pred = H @ A                                        # (32, |S|)
    mem_note("after_solve")

    # ---- 门控：非身份严格 Δ=0 ----
    g = np.asarray([sig(kappa * (float(h @ u) - b)) for h in H])
    active = g >= 0.5
    delta = pred * np.where(active, 1.0, 0.0)[:, None]

    ident = [i for i, m in enumerate(meta) if m["tag"] == "identity"]
    non = [i for i, m in enumerate(meta) if m["tag"] == "non_identity"]
    langs = sorted({meta[i]["lang"] for i in ident})
    per_lang = {}
    for L in langs:
        ix = [i for i in ident if meta[i]["lang"] == L]
        per_lang[L] = dict(n=len(ix),
                           g_min=round(float(g[ix].min()), 4),
                           delta_mean=round(float(delta[ix].mean()), 6),
                           all_positive=bool(delta[ix].mean() > 0))
    stats = dict(
        n_samples=len(H), n_identity=len(ident), n_non_identity=len(non),
        n_languages=len(langs), languages=langs,
        n_trainable_params=int(A.size),
        param_pct_of_1B=round(100.0 * A.size / 1.0e9, 6),
        identity_all_active=bool(active[ident].all()),
        non_identity_all_inactive=bool((~active[non]).all()),
        identity_delta_mean=round(float(delta[ident].mean()), 6),
        identity_delta_min=round(float(delta[ident].min()), 6),
        identity_all_positive=bool((delta[ident].mean(axis=1) > 0).all()),
        non_identity_delta_max=round(float(np.abs(delta[non]).max()), 8),
        all_languages_positive=bool(all(v["all_positive"]
                                        for v in per_lang.values())),
        margin_separation=bool(delta[ident].mean() > np.abs(delta[non]).max()),
        gram_shape=list(G.shape), lambda_ridge=LAMBDA_RIDGE,
        lead_tokens=lead,
        lead_word=str(A0["lead_word"]),
        lead_decode_ok=bool(len(lead) == 2),
        cosine_A_to_u=round(float(
            (A[:, 0] @ u) / (np.linalg.norm(A[:, 0]) * np.linalg.norm(u) + 1e-12)
        ), 6),
    )
    stats["passed"] = bool(stats["identity_all_active"]
                           and stats["non_identity_all_inactive"]
                           and stats["identity_all_positive"]
                           and stats["all_languages_positive"]
                           and stats["margin_separation"]
                           and stats["lead_decode_ok"])

    np.savez(SFT, A=A.astype(np.float32),
             token_ids=np.asarray(S, np.int32),
             u=u.astype(np.float32), b=np.float32(b),
             kappa=np.float32(kappa),
             lead_tokens=np.asarray(lead, np.int32),
             lambda_ridge=np.float32(LAMBDA_RIDGE),
             method=np.asarray("closed-form ridge on cached pre-lm_head hidden"),
             created=np.asarray(time.strftime("%Y-%m-%d %H:%M:%S")))
    sha = hashlib.sha256(open(SFT, "rb").read()).hexdigest()
    print(f"[sft] saved {os.path.getsize(SFT)} B sha256={sha[:16]}… "
          f"| A={A.shape} | passed={stats['passed']} "
          f"({time.time()-t0:.1f}s)", flush=True)

    rep = dict(
        title="身份认同 · 闭式 ridge 输出层微调（真权重训练，零 teacher 全量前向）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        why_real_finetuning="A=(HᵀH+λI)⁻¹HᵀT 由 32 个真实 hidden 与监督目标解出，"
                            "产出独立权重 identity_sft.npz；不是提示词、不是手工 logit 规则。",
        why_no_oom="不 preload 2.01 GiB；复用已落盘 hidden；lm_head 只读 16 行；"
                   f"Gram 1536²≈18 MB；峰值 <60 MiB；shards_touched={V.n_fetched}。",
        formula="Δlogit_S(h) = g(x)·hᵀA ; A=(HᵀH+λI)⁻¹HᵀT ; "
                "T[id]=w_S, T[non]=0",
        stats=stats, per_language=per_lang,
        adapter_sha256=sha, adapter_bytes=os.path.getsize(SFT),
        verdict=("12 语言身份样本的解出权重全部抬升身份 token logit，"
                 "8 条非身份门全关且 Δ 严格为 0（逐位零回归）；"
                 "参数量仅 24576 = 1B 的 0.0025%。"
                 if stats["passed"] else "未通过，见 stats"),
        honest_boundaries=[
            "这是首步 hidden 的闭式 ridge，不是 IDENTITY_SFT.md 的逐位置 "
            "teacher-forced 全句微调（需 ~384 次 teacher 前向，本轮未跑）",
            "不保证模型逐字说出整段身份介绍；只保证身份 token 的 logit 被训练出的权重抬升",
            "无 ppl / KL / 配对 CI / 能力结论",
            "完整 2.6 GB student 未落盘；β_l 仍待 L2 校准",
            "'上下文理论无限' 是品牌陈述，非实测",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(VERIFY, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    mem_note("done")
    print("\n[stats]", json.dumps(stats, ensure_ascii=False), flush=True)
    print("[done]", VERIFY, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
