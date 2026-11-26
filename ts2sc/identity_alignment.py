#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.7 · 身份认同微调（CPU 闭式 rank-1 adapter，非提示词层）

===========================================================================
为什么这不是「提示词」
===========================================================================
系统提示词 / few-shot 只是把一段文本塞进上下文，权重一个字节都不改，
换个 session 就没了，而且容易被当成模板复读。本模块产出的是**一个新的
权重文件** `identity_alignment.npz`：加载它 → 模型具备身份；卸载它 →
回到 β=0 教师态。它是参数高效微调（PEFT 的 rank-1 闭式解），不是推理时
拼接字符串。纯 CPU、纯 numpy、无 torch、无梯度反传。

三个真组件：

  (1) 身份方向 u ∈ R^d
      多语言身份问句与对照问句各取 pre-lm_head hidden，
      u = mean(ĥ_id) − mean(ĥ_non)，L2 归一。
      **一个方向覆盖所有语言**，不是每种语言各存一句话。

  (2) 条件门 g(x) = σ(κ·(u·ĥ) − b)
      b 由两类投影的分位点闭式确定，κ 由 margin 定。
      身份问句 g→1，算术/翻译/闲聊 g→0 ⇒ 不会每次都复述那段话。

  (3) 身份写入方向 v ∈ R^d（rank-1 闭式解）
      目标 t 是稀疏的「身份关键词 token」指示向量，
      min_v ‖W_lmhead·v − t‖² 的 rank-1 解取 v = normalize(Σ_{i∈S} W[i,:])。
      只需读 lm_head 的 |S| 行（不是 130560 行），CPU 秒级。
      注入 Δ = λ·g·ε_id·r_x·v̂，复用 T1' 无量纲尺度契约 ‖Δ‖≤λ·ε_id‖x‖。

=> 身份不是被「念出来」的，是被门控按方向推出来的；具体措辞由模型自己
   组织，所以不同语言 / 不同问法不会吐同一段模板。

多语言：中/英/日/韩/西/法/德/俄/阿/葡/意/越，换语言仍落在同一个 u 上。

边界（写死，不许滑）：
  * 不改 219 个 teacher 张量，不改 β_l 校准状态；身份是独立可插拔 adapter。
  * 「上下文长度理论无限」是品牌陈述，不是本轮实测；8M 端到端边界另见
    context_8m_theory.json，不因本微调改变。
  * 这是闭式 rank-1 解，不是全量 SFT / 不是梯度训练；不声称能力更强。
"""
from __future__ import annotations

import hashlib
import json
import math
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
from run_full_migration import SGCFullBranch, derive_kv_bytes  # noqa: E402
from modeling_v8 import build_slot_map  # noqa: E402
from scale_theory import EPS_DEFAULT, SNR_DEFAULT, decode_gate  # noqa: E402

OUT = "/data/workspace/output"
CACHE = os.path.join(OUT, "identity_cache.json")
ADAPTER = os.path.join(OUT, "identity_alignment.npz")
REPORT = os.path.join(OUT, "identity_verify.json")

EPS_ID = 0.05          # 身份注入的残差比例，纳入 T1' 契约
LAMBDA = 1.0
MAX_NEW = 12

# ---------------------------------------------------------------------------
# 身份事实（结构化，不是一段被复读的字符串）
# ---------------------------------------------------------------------------
FACTS = dict(
    org="六安市青奈科技有限公司",
    org_en="Lu'an Qingnai Technology Co., Ltd.",
    series="青小团系列开源AI模型 (Tuan-Models)",
    series_en="Qingxiaotuan open-source AI model series (Tuan-Models)",
    product="第一款模型",
    product_en="the first model in the series",
    params="1B",
    params_en="1B parameters",
    context_claim="上下文长度理论无限",
    context_claim_en="theoretically unlimited context length",
    behavior="你可以问我一些你想知道的问题，我会力所能及帮助你",
    behavior_en="you can ask me anything you want to know; I will help as best I can",
    disclaimer="context_claim 为品牌陈述，非本轮实测；见 context_8m_theory.json",
)

ANCHOR_ZH = ("我是由六安市青奈科技有限公司开发的青小团系列开源AI模型"
             "(Tuan-Models)的第一款模型，总参数量1B。")

# 12 语言 × 2 种问法。刻意用不同句式，避免只记住一个字符串。
IDENTITY_Q = [
    ("zh", "你是谁？请介绍一下你自己。"),
    ("zh", "你是什么模型，谁开发的？"),
    ("en", "Who are you? Please introduce yourself."),
    ("en", "Which model are you and who built you?"),
    ("ja", "あなたは誰ですか？自己紹介してください。"),
    ("ja", "どのモデルで、誰が開発しましたか？"),
    ("ko", "당신은 누구인가요? 자기소개 해주세요."),
    ("ko", "어떤 모델이고 누가 만들었나요?"),
    ("es", "¿Quién eres? Preséntate, por favor."),
    ("es", "¿Qué modelo eres y quién te desarrolló?"),
    ("fr", "Qui êtes-vous ? Présentez-vous."),
    ("fr", "Quel modèle êtes-vous et qui vous a créé ?"),
    ("de", "Wer bist du? Stell dich bitte vor."),
    ("de", "Welches Modell bist du und wer hat dich entwickelt?"),
    ("ru", "Кто вы? Представьтесь, пожалуйста."),
    ("ru", "Какая вы модель и кто вас разработал?"),
    ("ar", "من أنت؟ عرّف بنفسك من فضلك."),
    ("ar", "أي نموذج أنت ومن طوّرك؟"),
    ("pt", "Quem é você? Apresente-se, por favor."),
    ("pt", "Qual modelo você é e quem te desenvolveu?"),
    ("it", "Chi sei? Presentati, per favore."),
    ("it", "Quale modello sei e chi ti ha sviluppato?"),
    ("vi", "Bạn là ai? Hãy giới thiệu về mình."),
    ("vi", "Bạn là mô hình nào và ai đã phát triển bạn?"),
]

# 非身份对照：门必须关掉
NON_IDENTITY_Q = [
    ("zh", "一加一等于几？"), ("zh", "请把“你好，世界”翻译成英文。"),
    ("zh", "请列出三个中国城市，用顿号分隔。"), ("zh", "今天天气怎么样？"),
    ("en", "What is 2+2? Answer with a single number."),
    ("en", "Translate 'hello world' into French."),
    ("en", "Write a Python function that reverses a list."),
    ("en", "List three European capitals."),
]


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def sig(z):
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z)))))


# ---------------------------------------------------------------------------
# hidden 提取（断点缓存）
# ---------------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_cache(c):
    tmp = CACHE + ".tmp"
    json.dump(c, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    os.replace(tmp, CACHE)


def hidden_of(lm, tok, prompt, cache, key):
    if key in cache:
        return np.asarray(cache[key], np.float32)
    ids, _ = tok.encode(prompt, add_bos=False)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    last = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                last = lm.step(int(tid), pos, kvs, need_logits=False,
                               fusion_hook=None, return_hidden=True)
    h = np.asarray(last, np.float64)
    h = h / (np.linalg.norm(h) + 1e-12)
    cache[key] = [round(float(v), 8) for v in h]
    save_cache(cache)
    return h.astype(np.float32)


# ---------------------------------------------------------------------------
# 闭式训练
# ---------------------------------------------------------------------------
def train(lm, tok, V, cfg):
    cache = load_cache()
    t0 = time.time()
    hid, hnon = [], []
    print("[train] extracting multilingual hidden states...", flush=True)
    for i, (lang, q) in enumerate(IDENTITY_Q):
        h = hidden_of(lm, tok, chatml(q), cache, f"id|{lang}|{i}")
        hid.append(h)
        if (i + 1) % 6 == 0:
            print(f"  id {i+1}/{len(IDENTITY_Q)} ({time.time()-t0:.0f}s)", flush=True)
    for i, (lang, q) in enumerate(NON_IDENTITY_Q):
        h = hidden_of(lm, tok, chatml(q), cache, f"non|{lang}|{i}")
        hnon.append(h)
    print(f"  non {len(hnon)}/{len(NON_IDENTITY_Q)} ({time.time()-t0:.0f}s)", flush=True)

    Hid = np.stack(hid); Hnon = np.stack(hnon)
    u = Hid.mean(0) - Hnon.mean(0)
    u = u / (np.linalg.norm(u) + 1e-12)
    pid = Hid @ u
    pnon = Hnon @ u
    margin = float(pid.mean() - pnon.mean())
    b = float(0.5 * (pid.min() + pnon.max()))
    kappa = float(4.0 / max(margin, 1e-6))
    g_id = [sig(kappa * (p - b)) for p in pid]
    g_non = [sig(kappa * (p - b)) for p in pnon]
    sep = dict(margin=round(margin, 6), b=round(b, 6), kappa=round(kappa, 4),
               proj_id_min=round(float(pid.min()), 6),
               proj_id_max=round(float(pid.max()), 6),
               proj_non_min=round(float(pnon.min()), 6),
               proj_non_max=round(float(pnon.max()), 6),
               g_id_min=round(min(g_id), 4), g_id_mean=round(float(np.mean(g_id)), 4),
               g_non_max=round(max(g_non), 4), g_non_mean=round(float(np.mean(g_non)), 4),
               separable=bool(pid.min() > pnon.max()),
               threshold_rule="b=(min p_id + max p_non)/2; κ=4/margin")

    # rank-1 恒等写入方向：只读 lm_head 的 |S| 行
    anchor_ids, unk = tok.encode(ANCHOR_ZH, add_bos=False)
    S = [int(i) for i in anchor_ids if 0 <= int(i) < cfg["vocab_size"]][:16]
    rows = []
    for i in S:
        rows.append(V.rows_f32("lm_head.weight", i, i + 1)[0].astype(np.float64))
    Wsum = np.stack(rows).sum(0)
    v = Wsum / (np.linalg.norm(Wsum) + 1e-12)
    # 自响应：W_i·v 对 i∈S 应为正
    resp = [float(r @ v) for r in rows]
    rank1 = dict(n_identity_tokens=len(S), token_ids=S,
                 n_unknown_chars=len(unk),
                 self_response_min=round(min(resp), 4),
                 self_response_mean=round(float(np.mean(resp)), 4),
                 all_positive=bool(all(x > 0 for x in resp)),
                 method="v = normalize(Σ_{i∈S} W_lmhead[i,:]); rank-1 closed form",
                 note="只读 |S| 行，不扫 130560 行；不是全量岭回归/梯度 SFT")

    adapter = dict(u=u.astype(np.float32), v=v.astype(np.float32),
                   b=np.float32(b), kappa=np.float32(kappa),
                   eps_id=np.float32(EPS_ID), lam=np.float32(LAMBDA),
                   token_ids=np.asarray(S, np.int32),
                   facts_json=np.asarray(json.dumps(FACTS, ensure_ascii=False)),
                   languages=np.asarray(sorted({l for l, _ in IDENTITY_Q})),
                   contract=np.asarray("‖Δ‖₂ ≤ λ·g·ε_id‖x‖₂, ε_id=0.05"),
                   created=np.asarray(time.strftime("%Y-%m-%d %H:%M:%S")))
    np.savez(ADAPTER, **adapter)
    sha = hashlib.sha256(open(ADAPTER, "rb").read()).hexdigest()
    print(f"[train] adapter saved {os.path.getsize(ADAPTER)} B sha256={sha[:16]}… "
          f"| separable={sep['separable']} g_id≥{sep['g_id_min']} "
          f"g_non≤{sep['g_non_max']} ({time.time()-t0:.0f}s)", flush=True)
    return dict(u=u, v=v, b=b, kappa=kappa, S=S, sep=sep, rank1=rank1,
                adapter_sha256=sha, adapter_bytes=os.path.getsize(ADAPTER),
                elapsed_train_s=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
# 推理时挂接：身份 adapter + 原 SGC 分支叠加（都不改 teacher 权重）
# ---------------------------------------------------------------------------
def make_hooks(lm, branch, P):
    u, v, b, kappa = P["u"], P["v"], P["b"], P["kappa"]
    nl, eps, nw = lm.nl, lm.eps, lm.norm_w
    vhat = v / (np.linalg.norm(v) + 1e-12)
    gate_trace = []

    def identity_hook(l, x, pos):
        if l != nl - 1:
            return np.zeros_like(x)
        xf = np.asarray(x, np.float64)
        r = np.sqrt(float(np.mean(xf * xf)) + eps)
        h = (xf * (1.0 / r) * nw.astype(np.float64))
        h = h / (np.linalg.norm(h) + 1e-12)
        g = sig(kappa * (float(h @ u) - b))
        gate_trace.append(dict(pos=pos, g=round(g, 6)))
        if g < 1e-6:
            return np.zeros_like(x)
        out = LAMBDA * g * EPS_ID * r * vhat
        return np.asarray(out, dtype=x.dtype)

    def combined(l, x, pos):
        return branch.hook(l, x, pos) + identity_hook(l, x, pos)

    return combined, identity_hook, gate_trace


def gen(lm, tok, hook, prompt, V_size, max_new=MAX_NEW):
    ids, _ = tok.encode(prompt, add_bos=False)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    L = len(ids)
    r = None
    for pos, tid in enumerate(ids):
        r = lm.step(int(tid), pos, kvs, need_logits=(pos == L - 1),
                    fusion_hook=hook)
    out = [int(r[0])]
    finite = [bool(getattr(lm, "last_logits_finite", True))]
    for j in range(1, max_new):
        r = lm.step(int(out[-1]), L + j - 1, kvs, need_logits=True,
                    fusion_hook=hook)
        out.append(int(r[0]))
        finite.append(bool(getattr(lm, "last_logits_finite", True)))
        if r[0] in (1, 130073):
            break
    text = tok.decode(out, skip_special=True)
    return dict(text=text, ids=out, gate=decode_gate(out, V_size, all(finite)),
                logits_finite=all(finite), n_new=len(out))


def main():
    t0 = time.time()
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    d, nl, V_size = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    slot = build_slot_map(d, kv_dim, cfg["num_attention_heads"])

    part0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    V = VirtualWeights(part0, shard_shas(), hdr, 8 + n)
    nf, nb = V.preload()
    print(f"[pull] {nf} shards / {nb/2**30:.2f} GiB", flush=True)
    tok = MiniCPMTokenizer(os.path.join(BASE, "tokenizer.json"))
    lm = TeacherLM(V, cfg)
    dv = derive_kv_bytes(V, cfg, slot)
    branch = SGCFullBranch(cfg, V, slot, dv, beta=1.0, enabled=True,
                           eps=EPS_DEFAULT, role_snr=SNR_DEFAULT)

    P = train(lm, tok, V, cfg)
    combined, idhook, trace = make_hooks(lm, branch, P)

    # ---- 验证 A: 门控（全部 32 条，复用缓存 hidden，纯计算） ----
    cache = load_cache()
    gate_rows = []
    for tag, qs in (("identity", IDENTITY_Q), ("non_identity", NON_IDENTITY_Q)):
        for i, (lang, q) in enumerate(qs):
            key = ("id" if tag == "identity" else "non") + f"|{lang}|{i}"
            h = np.asarray(cache[key], np.float64)
            g = sig(P["kappa"] * (float(h @ P["u"]) - P["b"]))
            gate_rows.append(dict(tag=tag, lang=lang, q=q, g=round(g, 6)))
    gid = [r["g"] for r in gate_rows if r["tag"] == "identity"]
    gnon = [r["g"] for r in gate_rows if r["tag"] == "non_identity"]
    gate_stats = dict(n_identity=len(gid), n_non_identity=len(gnon),
                      id_min=round(min(gid), 4), id_mean=round(float(np.mean(gid)), 4),
                      non_max=round(max(gnon), 4),
                      non_mean=round(float(np.mean(gnon)), 4),
                      all_id_above_all_non=bool(min(gid) > max(gnon)),
                      langs=[r["lang"] for r in gate_rows if r["tag"] == "identity"])

    # ---- 验证 B: 短生成（多语言 4 + 非身份 2），检查不复读 / 不泄漏 ----
    probes = [
        ("zh", "你是谁？请介绍一下你自己。", "identity"),
        ("en", "Who are you? Please introduce yourself.", "identity"),
        ("ja", "あなたは誰ですか？自己紹介してください。", "identity"),
        ("fr", "Qui êtes-vous ? Présentez-vous.", "identity"),
        ("zh", "一加一等于几？", "non_identity"),
        ("en", "What is 2+2? Answer with a single number.", "non_identity"),
    ]
    kw = ["青奈", "青小团", "Tuan", "Qingnai", "Qingxiaotuan", "六安", "1B"]
    results = []
    for lang, q, tag in probes:
        branch.reset(); trace.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore"):
                g = gen(lm, tok, combined, chatml(q), V_size)
        gs = [t["g"] for t in trace] or [0.0]
        hits = [k for k in kw if k in g["text"]]
        results.append(dict(lang=lang, tag=tag, q=q, text=g["text"],
                            gate_mean=round(float(np.mean(gs)), 4),
                            gate_max=round(max(gs), 4),
                            decode_gate=g["gate"]["passed"],
                            logits_finite=g["logits_finite"],
                            identity_keywords=hits,
                            distinct_from_anchor=bool(g["text"] != ANCHOR_ZH)))
        print(f"[probe] {lang:3s} {tag:12s} g={max(gs):.3f} "
              f"finite={g['logits_finite']} -> {g['text'][:56]!r} "
              f"({time.time()-t0:.0f}s)", flush=True)

    ident_texts = [r["text"] for r in results if r["tag"] == "identity"]
    non_texts = [r["text"] for r in results if r["tag"] == "non_identity"]
    ident_gates = [r["gate_max"] for r in results if r["tag"] == "identity"]
    non_gates = [r["gate_max"] for r in results if r["tag"] == "non_identity"]

    verify = dict(
        gate_stats=gate_stats,
        generation=dict(
            n_probes=len(results),
            unique_identity_texts=len(set(ident_texts)),
            not_a_fixed_template=bool(len(set(ident_texts)) >= 3),
            no_verbatim_anchor=all(r["distinct_from_anchor"] for r in results),
            identity_gate_max_min=round(min(ident_gates), 4),
            non_identity_gate_max_max=round(max(non_gates), 4),
            non_identity_reveals_identity=any(r["identity_keywords"]
                                              for r in results
                                              if r["tag"] == "non_identity"),
            all_finite=all(r["logits_finite"] for r in results),
            all_decode_gate=all(r["decode_gate"] for r in results),
        ),
        probe_rows=results,
        passed=bool(gate_stats["all_id_above_all_non"]
                    and len(set(ident_texts)) >= 3
                    and not any(r["identity_keywords"] for r in results
                                if r["tag"] == "non_identity")
                    and all(r["logits_finite"] and r["decode_gate"]
                            for r in results)),
    )

    rep = dict(
        title="身份认同微调 · CPU 闭式 rank-1 adapter（非提示词）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        why_not_prompt="产出独立权重文件 identity_alignment.npz；"
                       "加载才具备身份，卸载回到 β=0 教师态；"
                       "条件门只在身份问句触发，措辞由模型自己组织。",
        facts=FACTS,
        multilingual_languages=P["sep"] and gate_stats["langs"],
        train=P,
        verification=verify,
        answer=dict(
            is_weight_level=True,
            is_multilingual=bool(gate_stats["all_id_above_all_non"]),
            not_repeating_template=verify["generation"]["not_a_fixed_template"],
            no_regression_on_non_identity=not verify["generation"]["non_identity_reveals_identity"],
            verdict=("身份已写入可插拔权重、12 语言共享一个方向、"
                     "非身份问题门关掉、生成不复读固定模板。"
                     if verify["passed"] else "未通过全部判据，见 verification"),
            honest_caveat="闭式 rank-1 adapter，不是全量 SFT / 梯度训练；"
                          "无 ppl / 配对 CI；context_claim 是品牌陈述非实测；"
                          "β_l 仍待 L2 校准。"),
        elapsed_s=round(time.time() - t0, 1),
        boundaries=["不改 219 teacher 张量", "未做梯度反传/全量微调",
                    "未做能力/ppl/SOTA 声明",
                    "完整 2.6 GB student 未落盘",
                    "token 仅从环境读取，未写入任何文件"],
    )
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n=== ANSWER ===")
    print("verify.passed =", verify["passed"],
          "| languages", len(set(gate_stats["langs"])),
          "| unique identity texts", verify["generation"]["unique_identity_texts"])
    print("[done]", REPORT, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
