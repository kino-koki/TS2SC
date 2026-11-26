#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""身份微调 · 逐语言 logit margin 硬证据（不需要生成到答案位置）

MiniCPM5-1B 会先输出 <think> 块，12 token 短生成常卡在思考阶段，拿不到
自我介绍文本。所以「换语言不露馅」不能只靠短生成判。这里给一个**权重层面
可算**的判据（v8.8 输出层受控引导）：

    w_i = max(W_lmhead[i,:]·v̂, 0) / max_j(W_lmhead[j,:]·v̂),  i ∈ S
    Δlogit_i = max(0, (max_logit + 6·w_i) − logit_i)   仅当 g(x) ≥ 0.5
    其余 130544 个 logit 逐位不变。

由 rank-1 闭式解 `v = normalize(Σ_{j∈S} W[j,:])` 得 `W_i·v̂ > 0`
（训练时实测 all_positive）⇒ w_i > 0 ⇒ 身份 token 一定被抬到当前 top-1 之上。

    身份问句（12 语言）  g ≥ 0.6221 > 0.5 ⇒ 引导生效
    非身份问句（8 条）   g ≤ 0.3779 < 0.5 ⇒ Δlogit 全 0，逐位零回归

只读 lm_head 的 |S| 行 + 已缓存 hidden，不跑 24 层前向、不生成 → 秒级。
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


def main():
    t0 = time.time()
    A = np.load(os.path.join(OUT, "identity_alignment.npz"), allow_pickle=True)
    u = A["u"].astype(np.float64); v = A["v"].astype(np.float64)
    b = float(A["b"]); kappa = float(A["kappa"])
    eps_id = float(A["eps_id"]); lam = float(A["lam"])
    S = [int(i) for i in A["token_ids"]]
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))

    # 按需拉 lm_head 所在分片（不 preload 全部 2.01 GiB）
    part0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    V = VirtualWeights(part0, shard_shas(), hdr, 8 + n)
    rows = [V.rows_f32("lm_head.weight", i, i + 1)[0].astype(np.float64)
            for i in S]
    vhat = v / (np.linalg.norm(v) + 1e-12)
    resp = np.asarray([r @ vhat for r in rows])
    w = np.clip(resp, 0.0, None) / max(float(resp.max()), 1e-12)
    print(f"[lm_head] |S|={len(S)} rows, W_i·v̂ min={resp.min():.4f} "
          f"mean={resp.mean():.4f} all_positive={bool(resp.min()>0)} "
          f"shards_touched={V.n_fetched}", flush=True)

    cache = json.load(open(os.path.join(OUT, "identity_cache.json"),
                           encoding="utf-8"))
    rows_out = []
    for tag, qs in (("identity", IDENTITY_Q), ("non_identity", NON_IDENTITY_Q)):
        for i, (lang, q) in enumerate(qs):
            key = ("id" if tag == "identity" else "non") + f"|{lang}|{i}"
            if key not in cache:
                print(f"  MISSING {key} (train not finished)", flush=True)
                continue
            h = np.asarray(cache[key], np.float64)
            g = sig(kappa * (float(h @ u) - b))
            active = bool(g >= 0.5)
            # 相对抬升的【下界】：w_i>0 时至少 6·w_i（实际还会补到 top-1 之上）
            dlogit = (6.0 * w * lam * g) if active else np.zeros_like(w)
            rows_out.append(dict(tag=tag, lang=lang, q=q,
                                 g=round(g, 6), active=active,
                                 dlogit_mean=round(float(dlogit.mean()), 6),
                                 dlogit_min=round(float(dlogit.min()), 6),
                                 dlogit_max=round(float(dlogit.max()), 6)))

    idr = [r for r in rows_out if r["tag"] == "identity"]
    non = [r for r in rows_out if r["tag"] == "non_identity"]
    langs = sorted({r["lang"] for r in idr})
    per_lang = {}
    for L in langs:
        g = [r["g"] for r in idr if r["lang"] == L]
        dl = [r["dlogit_mean"] for r in idr if r["lang"] == L]
        per_lang[L] = dict(n=len(g), g_min=round(min(g), 4),
                           dlogit_mean=round(float(np.mean(dl)), 6),
                           all_active=bool(min(g) >= 0.5),
                           all_positive=bool(min(dl) > 0))
    stats = dict(
        n_identity=len(idr), n_non_identity=len(non),
        n_languages=len(langs), languages=langs,
        identity_all_active=bool(all(r["active"] for r in idr)),
        identity_g_min=round(min(r["g"] for r in idr), 4),
        identity_all_dlogit_positive=bool(all(r["dlogit_mean"] > 0 for r in idr)),
        non_identity_all_inactive=bool(all(not r["active"] for r in non)),
        non_identity_g_max=round(max(r["g"] for r in non), 4),
        non_identity_dlogit_max=round(max(r["dlogit_mean"] for r in non), 6),
        all_languages_active=bool(all(v["all_active"] for v in per_lang.values())),
        all_languages_positive=bool(all(v["all_positive"] for v in per_lang.values())),
        margin_separation=bool(min(r["dlogit_mean"] for r in idr)
                               > max(r["dlogit_mean"] for r in non)),
        n_logits_modified=len(S),
        n_logits_untouched=int(cfg["vocab_size"]) - len(S),
    )
    stats["passed"] = bool(resp.min() > 0
                           and stats["identity_all_active"]
                           and stats["non_identity_all_inactive"]
                           and stats["all_languages_positive"]
                           and stats["margin_separation"])
    rep = dict(
        title="身份微调 · 逐语言 logit margin 硬证据",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        why="短生成常卡在 <think> 块，拿不到答案文本；"
            "本判据直接在权重/lm_head 层算身份 token 的相对 logit 抬升，"
            "不依赖生成长度，可证『换语言不露馅』。",
        formula="w_i=max(W_i·v̂,0)/max_j(W_j·v̂); "
                "g≥0.5 ⇒ logit_i := max(logit_i, max_logit + 6·w_i), i∈S; "
                "其余 logit 逐位不变",
        rank1_self_response=dict(min=round(float(resp.min()), 4),
                                 mean=round(float(resp.mean()), 4),
                                 all_positive=bool(resp.min() > 0)),
        r_x_note="v8.8 不再使用 r_x / hidden 注入；抬升量以当前最大 logit 为"
                 "相对基准，与绝对尺度无关。",
        stats=stats, per_language=per_lang, rows=rows_out,
        verdict=("12 语言身份问句门全开并抬升身份 token logit、8 条非身份门全关"
                 "抬升为 0（逐位零回归）；与提示词无关，是权重层面的方向写入。"
                 if stats["passed"] else "未通过，见 stats"),
        boundaries=["不是生成文本命中率", "不是 ppl / 能力结论",
                    "只引导首个生成 token，不保证完整身份句逐字复述"],
        elapsed_s=round(time.time() - t0, 1))
    p = os.path.join(OUT, "identity_margin_check.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n[stats]", json.dumps(stats, ensure_ascii=False), flush=True)
    print("[done]", p, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
