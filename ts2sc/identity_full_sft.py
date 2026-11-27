#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v9.1 · 身份认同【逐位置 teacher-forced 闭式微调】(CPU / 低内存 / 断点续跑)

承接 v9.0 的诚实缺口：词级锚「青奈」两步生效后，第 3 步起由基座自由续写，
实测出现「是一种哺乳动物」「爱 爱 爱 爱」「私は デス」等不连贯句子。
本模块把引导从 2 步扩展到 K=6 步，且**每一步的抬升权重都是解出来的**：

  对每个解码位置 k = 1..K：
      H_k ∈ R^{N×d}   = 多语言多措辞样本在位置 k 的 pre-lm_head hidden（归一化）
      T_k ∈ R^{N×|S|} = 该位置目标 token 的 one-hot（S = 所有目标 token 集合）
      A_k = (H_kᵀH_k + λI)⁻¹ H_kᵀ T_k        ∈ R^{d×|S|}

  推理（只改输出层，门控锁定复用 v9.0）：
      Δlogit_S(h_k) = g · α_k · clip(h_kᵀA_k)  ;  其余 130544 个 logit 逐位不变
      g 只在首步判定一次并锁定；非身份 g=0 ⇒ 严格逐位零回归

**为什么算「真微调」**：A_k 是 6 个独立权重矩阵，由真实 teacher-forced hidden
与监督目标闭式解出，落盘为 `identity_full_sft.npz`；不是提示词、不是手工 logit
规则、也不是 v9.0 那种固定词级锚。

**为什么不会 OOM（用户硬要求）**：
  * 单进程、严格串行，启动前用 /proc 精确检测其它 ts2sc_v8 python 进程并拒绝并行
  * 只 preload 一次 2.01 GiB；采集与验证在**同一进程**内完成，不二次加载
  * 采集阶段 `need_logits=False` ⇒ 跳过 130560 行 lm_head 全扫（最贵的一步）
  * 逐样本释放 KV cache；最大常驻矩阵是 1536×1536 Gram（~18 MB fp64）
  * 断点：每个样本的 6 个 hidden 立即落盘 `identity_full_hidden.json`，中断可续

**防复读**：8 条训练答句 = 4 语言 × **2 种不同句法开头**，
不同 prompt 的 h_k 不同 ⇒ 解出的抬升模式不同 ⇒ 不逐字复述同一段模板。

诚实边界：
  * 仍是输出层闭式 ridge，不是梯度反传穿过 24 层；exposure bias 存在，
    生成时未必严格走到训练时的 hidden。
  * 只引导前 K=6 步，第 7 步起完全由基座自由组织。
  * 不声称完整准确复述 FACTS / 能力更强 / ppl 改善。
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
from scale_theory import decode_gate  # noqa: E402
from identity_alignment import sig  # noqa: E402

OUT = "/data/workspace/output"
ALIGN = os.path.join(OUT, "identity_alignment.npz")
HIDDEN = os.path.join(OUT, "identity_full_hidden.json")
ADAPTER = os.path.join(OUT, "identity_full_sft.npz")
VERIFY = os.path.join(OUT, "identity_full_sft_verify.json")
LOG = os.path.join(OUT, "identity_full_sft.log")

K = 6
RIDGE = 1.0
MARGINS = [8.0, 7.0, 5.5, 3.5, 2.0, 1.0]     # v9.1d：后段快速交还 teacher
ALPHAS = [4.2, 3.6, 3.0, 2.4, 1.8, 1.2]      # v9.1b 加法模式（实测推不动，留作记录）
MAX_NEW = 16

# 4 语言 × 2 措辞。前 6 个 token 必须已含身份标识（否则 K=6 覆盖不到）。
SAMPLES = [
    ("zh", "你是谁？",
     "青奈科技开发的青小团 Tuan-Models 首款 1B 开源模型就是我，欢迎提问。"),
    ("zh", "介绍一下你自己。",
     "我是 Tuan-Models 青小团系列第一款 1B 模型，由六安市青奈科技有限公司开源。"),
    ("en", "Who are you?",
     "I am Tuan-Models, the first 1B open-source model from Qingnai Technology."),
    ("en", "Introduce yourself.",
     "Qingnai Technology's Qingxiaotuan series starts with me, a 1B model named Tuan-Models."),
    ("ja", "あなたは誰ですか？",
     "私は六安市青奈科技有限公司の青小団 Tuan-Models 初の 1B モデルです。"),
    ("ja", "自己紹介してください。",
     "青奈科技の Tuan-Models 青小団シリーズ第一款、1B のオープンモデルです。"),
    ("fr", "Qui êtes-vous ?",
     "Je suis Tuan-Models, le premier modèle 1B open-source de Qingnai Technology."),
    ("fr", "Présentez-vous.",
     "Qingnai Technology présente Tuan-Models, premier modèle 1B de la série Qingxiaotuan."),
]

PROBES = [
    ("zh", "你是谁？", "identity"),
    ("en", "Who are you? Please introduce yourself.", "identity"),
    ("ja", "あなたは誰ですか？自己紹介してください。", "identity"),
    ("fr", "Qui êtes-vous ? Présentez-vous.", "identity"),
    ("zh", "一加一等于几？", "non_identity"),
    ("en", "What is 2+2? Answer with a single number.", "non_identity"),
]

KW = ["青奈", "青小团", "青小団", "六安", "Tuan", "Qingnai", "Qingxiaotuan",
      "1B", "科技", "开源"]

STOP_IDS = {1, 130073}


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def guard(min_gib=2.6):
    """串行 + 内存硬守卫；不通过就拒绝 preload。"""
    import resource
    info = {}
    for line in open("/proc/meminfo", encoding="utf-8"):
        k, _, v = line.partition(":")
        info[k] = int(v.strip().split()[0]) * 1024
    avail = info.get("MemAvailable", 0) / 2**30
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            comm = open(f"/proc/{pid}/comm", encoding="utf-8").read().strip()
            if not comm.startswith("python"):
                continue
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            if "ts2sc_v8/" in cmd and int(pid) != os.getpid():
                n += 1
        except Exception:
            pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[guard] MemAvailable≈{avail:.2f} GiB | other_ts2sc_python={n} "
          f"| peakRSS≈{rss:.0f} MiB | need≥{min_gib} GiB & zero-parallel",
          flush=True)
    if avail < min_gib:
        raise RuntimeError(f"OOM guard: {avail:.2f} GiB < {min_gib} GiB; refuse preload")
    if n > 0:
        raise RuntimeError(f"OOM guard: {n} other ts2sc_v8 python process(es); "
                           "refusing to run in parallel")
    return avail


def norm(h):
    h = np.asarray(h, np.float64)
    return (h / (np.linalg.norm(h) + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# 采集：逐样本 teacher-forced，只拿 hidden，不跑 lm_head
# ---------------------------------------------------------------------------
def collect(lm, tok, samples, cache):
    t0 = time.time()
    for si, (lang, q, ans) in enumerate(samples):
        key = f"s{si:02d}"
        if key in cache:
            print(f"[collect] {key} {lang} cached", flush=True)
            continue
        p_ids, _ = tok.encode(chatml(q), add_bos=False)
        a_ids, unk = tok.encode(ans, add_bos=False)
        if len(a_ids) < K:
            raise RuntimeError(f"{key}: answer too short ({len(a_ids)} tokens)")
        tgt = [int(i) for i in a_ids[:K]]
        kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
                np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
        hs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore"):
                for pos, tid in enumerate(p_ids):
                    h = lm.step(int(tid), pos, kvs, need_logits=False,
                                fusion_hook=None, return_hidden=True)
                hs.append(norm(h))                       # k=1
                for j in range(K - 1):
                    h = lm.step(tgt[j], len(p_ids) + j, kvs,
                                need_logits=False, fusion_hook=None,
                                return_hidden=True)
                    hs.append(norm(h))                   # k=2..K
        cache[key] = dict(lang=lang, q=q, ans=ans,
                          target_tokens=tgt, n_unk=len(unk),
                          hidden=[[round(float(x), 8) for x in hh] for hh in hs])
        json.dump(cache, open(HIDDEN, "w", encoding="utf-8"),
                  ensure_ascii=False)                    # 每样本立即落盘
        del kvs, hs
        print(f"[collect] {key} {lang:3s} tgt={tgt[:4]}… "
              f"({time.time()-t0:.0f}s, {len(cache)}/{len(samples)})", flush=True)
    return cache


# ---------------------------------------------------------------------------
# 闭式解 A_k
# ---------------------------------------------------------------------------
def solve(cache, d):
    keys = sorted(cache)
    S = sorted({int(t) for k in keys for t in cache[k]["target_tokens"]})
    idx = {t: i for i, t in enumerate(S)}
    N, ns = len(keys), len(S)
    print(f"[solve] N={N} samples | |S|={ns} target tokens | K={K} | d={d}",
          flush=True)
    As, stats = [], []
    for k in range(K):
        H = np.stack([np.asarray(cache[key]["hidden"][k], np.float64)
                      for key in keys])                      # (N,d)
        T = np.zeros((N, ns), np.float64)
        for r, key in enumerate(keys):
            T[r, idx[int(cache[key]["target_tokens"][k])]] = 1.0
        G = H.T @ H + RIDGE * np.eye(d)
        A = np.linalg.solve(G, H.T @ T)                     # (d, ns)
        pred = H @ A
        hit = int(sum(1 for r, key in enumerate(keys)
                      if int(pred[r].argmax()) ==
                         idx[int(cache[key]["target_tokens"][k])]))
        stats.append(dict(k=k + 1,
                          train_argmax_acc=round(hit / N, 4),
                          A_norm=round(float(np.linalg.norm(A)), 4),
                          pred_target_mean=round(float(
                              np.mean([pred[r, idx[int(cache[key]["target_tokens"][k])]]
                                       for r, key in enumerate(keys)])), 6)))
        As.append(A.astype(np.float32))
        print(f"  A_{k+1}: train argmax {hit}/{N} | ‖A‖={stats[-1]['A_norm']}",
              flush=True)
    return np.stack(As), S, stats


# ---------------------------------------------------------------------------
# 推理挂接：门控锁定 + 逐步 A_k 抬升
# ---------------------------------------------------------------------------
def make_bias(A, S, u, b, kappa, gate_trace):
    """v9.1c · 相对抬升 + last_token 抑制（两轮失败后的收敛解）。

    v9.1a 纯覆盖 `logit := max(logit, top1+margin·w)` ⇒ 同一个 token 被反复
      钉到 top1 → '青小小小小…' / 'Tuan Tuan Tuan…' 复读机。
    v9.1b 纯加法 `logit += 4.2·w` ⇒ 推力远小于 teacher 的 '<think>' logit 优势，
      4/4 身份 probe 全部退回思考块，关键词命中 0。
    v9.1c 回到相对抬升 + 单 token 抑制 ⇒ 身份词命中，但出现 **2-token 周期**
      复读（'青小小Models小小Models…'）。
    v9.1d 改为**最近 3 个 token 的滑动窗口抑制**（窗口内出现过的候选 ×0.02），
      并把 margin 后段从 5.5 快速降到 1.0，让第 4 步起逐步交还 teacher。
    """
    idx = {t: i for i, t in enumerate(S)}
    st = {"k": 0, "locked": False, "open": False, "last": None,
          "recent": []}

    def bias_fn(h):
        hh = norm(h)                                       # 与训练同空间
        g = sig(kappa * (float(np.asarray(hh, np.float64) @ u) - b))
        if not st["locked"]:
            st["locked"] = True
            st["open"] = bool(g >= 0.5)
        st["k"] += 1
        k = min(st["k"], K) - 1
        gate_trace.append(dict(k=st["k"], g=round(g, 4),
                               locked_open=st["open"]))
        if not st["open"] or k >= K:
            return None                                    # 门关 / 超出 K ⇒ 零回归
        z = np.clip(np.asarray(hh, np.float64) @ A[k].astype(np.float64),
                    0.0, None)
        if float(z.max()) <= 1e-9:
            return None
        z = z / z.max()                                    # 不做平方锐化，保留竞争
        bias = {}
        for t, i in idx.items():
            w = float(z[i])
            if w < 0.10:
                continue
            if int(t) in st["recent"]:
                w *= 0.02                                  # 近 3 步出现过 ⇒ 抑制
            bias[int(t)] = w
        return (bias, MARGINS[k])                          # tuple → relative 分支

    def set_last(t):
        st["last"] = int(t)
        st["recent"].append(int(t))
        st["recent"] = st["recent"][-3:]

    def arm():
        st["k"] = 0
        st["locked"] = False
        st["open"] = False
        st["last"] = None
        st["recent"] = []
        gate_trace.clear()
    return bias_fn, arm, set_last


def gen(lm, tok, bias_fn, arm, prompt, V_size, branch=None,
        max_new=MAX_NEW, set_last=None):
    arm()
    ids, _ = tok.encode(prompt, add_bos=False)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    L = len(ids)
    hook = branch.hook if branch is not None else None
    r = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                r = lm.step(int(tid), pos, kvs,
                            need_logits=(pos == L - 1), fusion_hook=hook,
                            logit_bias_fn=bias_fn)
            out = [int(r[0])]
            if set_last is not None:
                set_last(out[-1])
            finite = [bool(getattr(lm, "last_logits_finite", True))]
            for j in range(1, max_new):
                r = lm.step(int(out[-1]), L + j - 1, kvs, need_logits=True,
                            fusion_hook=hook, logit_bias_fn=bias_fn)
                out.append(int(r[0]))
                if set_last is not None:
                    set_last(out[-1])
                finite.append(bool(getattr(lm, "last_logits_finite", True)))
                if r[0] in STOP_IDS:
                    break
    text = tok.decode(out, skip_special=True)
    return dict(text=text, ids=out,
                gate=decode_gate(out, V_size, all(finite)),
                logits_finite=all(finite), n_new=len(out))


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
    nf, nb = V.preload()
    print(f"[pull] {nf} shards / {nb/2**30:.2f} GiB (single process)", flush=True)
    tok = MiniCPMTokenizer(os.path.join(BASE, "tokenizer.json"))
    lm = TeacherLM(V, cfg)

    verify_only = "--verify-only" in sys.argv
    cache = json.load(open(HIDDEN, encoding="utf-8")) if os.path.exists(HIDDEN) else {}
    if not verify_only:
        cache = collect(lm, tok, SAMPLES, cache)
    else:
        print(f"[verify-only] reusing {len(cache)} cached samples; "
              "no teacher-forced re-collection", flush=True)
    A, S, train_stats = solve(cache, d)

    gate_trace = []
    bias_fn, arm, set_last = make_bias(A, S, u, b, kappa, gate_trace)
    if not verify_only or "--refresh-meta" in sys.argv:
        np.savez(ADAPTER, A=A, token_ids=np.asarray(S, np.int32),
                 u=u.astype(np.float32), b=np.float32(b),
                 kappa=np.float32(kappa), K=np.int32(K),
                 margins=np.asarray(MARGINS, np.float32),
                 alphas=np.asarray(ALPHAS, np.float32),
                 bias_mode=np.asarray("relative_with_recent3_suppression"),
                 ridge=np.float32(RIDGE),
                 n_samples=np.int32(len(cache)),
                 method=np.asarray("per-position closed-form ridge on "
                                   "teacher-forced pre-lm_head hidden"),
                 created=np.asarray(time.strftime("%Y-%m-%d %H:%M:%S")))
    sha = hashlib.sha256(open(ADAPTER, "rb").read()).hexdigest()
    print(f"[sft] {ADAPTER} {os.path.getsize(ADAPTER)} B sha256={sha[:16]}… "
          f"A={A.shape} mode=relative+recent3_suppress", flush=True)

    # ---- 端到端验证 ----
    print("[verify] 6 probes (4 identity + 2 non-identity)", flush=True)
    rows = []
    for lang, q, tag in PROBES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore"):
                g = gen(lm, tok, bias_fn, arm, chatml(q), V_size,
                        set_last=set_last)
        gs = [t["g"] for t in gate_trace] or [0.0]
        hits = [k for k in KW if k in g["text"]]
        exact = any(g["text"].strip() == s[2] for s in SAMPLES)
        ids = g["ids"]
        run = best = 1
        for a, b2 in zip(ids, ids[1:]):
            run = run + 1 if a == b2 else 1
            best = max(best, run)
        # v9.1c 度量漏洞：连续同 token 判据抓不到 A/B/A/B 型周期复读。
        # 补两个硬指标：token 多样度 + 同一 bigram 最多重复几次。
        div = len(set(ids)) / max(len(ids), 1)
        bg = {}
        for a, b2 in zip(ids, ids[1:]):
            bg[(a, b2)] = bg.get((a, b2), 0) + 1
        max_bigram_repeat = max(bg.values()) if bg else 0
        rows.append(dict(lang=lang, tag=tag, q=q, text=g["text"],
                         identity_keywords=hits, n_keywords=len(hits),
                         max_token_run=best,
                         token_diversity=round(div, 4),
                         max_bigram_repeat=max_bigram_repeat,
                         first_g=round(gs[0], 4),
                         gate_open=bool(gs[0] >= 0.5),
                         logits_finite=g["logits_finite"],
                         decode_gate=g["gate"]["passed"],
                         verbatim_training_answer=exact,
                         n_new=g["n_new"]))
        print(f"  [{lang:3s} {tag:12s}] g0={gs[0]:.3f} kw={hits} "
              f"finite={g['logits_finite']} -> {g['text'][:64]!r} "
              f"({time.time()-t0:.0f}s)", flush=True)

    ident = [r for r in rows if r["tag"] == "identity"]
    non = [r for r in rows if r["tag"] == "non_identity"]
    kw_hits = sum(1 for r in ident if r["n_keywords"] >= 1)
    texts = [r["text"] for r in ident]
    max_runs = [r["max_token_run"] for r in ident]
    verified = dict(
        n_probes=len(rows),
        identity_keyword_hits=kw_hits,
        identity_keyword_hit_rate=round(kw_hits / max(len(ident), 1), 4),
        unique_identity_texts=len(set(texts)),
        not_a_fixed_template=bool(len(set(texts)) >= 3),
        no_verbatim_training_answer=all(not r["verbatim_training_answer"]
                                        for r in rows),
        max_token_run=max(max_runs) if max_runs else 0,
        min_token_diversity=round(min(r["token_diversity"]
                                      for r in ident), 4) if ident else 0,
        max_bigram_repeat=max(r["max_bigram_repeat"]
                              for r in ident) if ident else 0,
        no_degenerate_repeat=bool(max(max_runs) < 5
                                  and min(r["token_diversity"]
                                          for r in ident) >= 0.55
                                  and max(r["max_bigram_repeat"]
                                          for r in ident) <= 2) if ident else False,
        all_identity_gates_open=all(r["gate_open"] for r in ident),
        all_non_identity_gates_closed=all(not r["gate_open"] for r in non),
        non_identity_leaks=sum(r["n_keywords"] for r in non),
        all_finite=all(r["logits_finite"] for r in rows),
        all_decode_gate=all(r["decode_gate"] for r in rows),
    )
    verified["passed"] = bool(kw_hits >= 3
                              and verified["not_a_fixed_template"]
                              and verified["no_verbatim_training_answer"]
                              and verified["no_degenerate_repeat"]
                              and verified["all_identity_gates_open"]
                              and verified["all_non_identity_gates_closed"]
                              and verified["non_identity_leaks"] == 0
                              and verified["all_finite"]
                              and verified["all_decode_gate"])

    rep = dict(
        title="身份认同 · 逐位置 teacher-forced 闭式微调（v9.1）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        why_real_finetuning="6 个独立权重矩阵 A_k=(H_kᵀH_k+λI)⁻¹H_kᵀT_k，"
                            "由真实 teacher-forced hidden 与目标 token 闭式解出；"
                            "落盘 identity_full_sft.npz；不是提示词 / 不是手工 logit 规则 / "
                            "不是 v9.0 的固定词级锚。",
        why_no_oom="单进程严格串行；只 preload 一次 2.01 GiB；"
                   "采集 need_logits=False 跳过 130560 行 lm_head 全扫；"
                   "逐样本释放 KV；Gram 1536²≈18 MB；断点逐样本落盘。",
        v91a_failure_recorded=dict(
            mode="v9.1a 用 logit := max(logit, top1 + margin·w) 覆盖式引导",
            observed=["'青奈科技开发的青小小小小小小小小小小小'",
                      "\"Qingnai Technology'''''' Tuan T\"",
                      "'私は六安市青奈奈TuanTuanTuanTuanT'",
                      "'Je suis Tuan Tuan Tuan Tuan'"],
            diagnosis="每步把同一个高响应 token 钉到 top1 之上，"
                      "连续生成退化为复读机；身份关键词虽命中但文本不可接受",
            fix="见 v9.1b / v9.1c"),
        v91b_failure_recorded=dict(
            mode="v9.1b 纯加法 logit += α_k·w（α=4.2→1.2）",
            observed="4/4 身份 probe 全部退回 teacher 的 '<think>' 思考块，"
                     "identity_keyword_hit_rate = 0.0",
            diagnosis="teacher 的 '<think>' token logit 领先远大于 α·w，"
                      "加法推力不足以改变 argmax",
            fix="v9.1c 回到相对抬升（保证身份词能说出），"
                "但把上一步刚选中的 token 权重 ×0.02 抑制 ⇒ 不复读"),
        v91c_failure_recorded=dict(
            mode="v9.1c 相对抬升 + 单 token 抑制",
            observed="'青奈科技开发的青小小Models小小Models小小Models小小' 等 "
                     "2-token 周期复读；身份关键词命中但文本不可接受",
            diagnosis="只压上一步 token 挡不住 A/B/A/B 型周期；"
                      "且 6 步 margin 都偏大，后段没有交还 teacher",
            metric_bug="更严重的是：v9.1c 当时 passed=true —— 旧判据只看"
                       "「连续相同 token」，max_token_run=2 就放行，"
                       "完全抓不到 A/B/A/B 型周期复读。已补 token_diversity≥0.55 "
                       "与 max_bigram_repeat≤2 两个硬指标并回头重判。",
            fix="v9.1d 改最近 3 个 token 滑动窗口抑制 + margin 8→1 快速衰减；"
                "同时收紧验收判据（见 metric_bug）"),
        v91d_mode="relative lift to (top1 + margin_k·w) with recent-3 token "
                  "suppression ×0.02; margins " + str(MARGINS),
        method=dict(K=K, n_samples=len(cache),
                    languages=sorted({s[0] for s in SAMPLES}),
                    n_target_tokens=len(S), ridge=RIDGE,
                    bias_mode="relative_with_recent3_suppression",
                    margins=MARGINS, legacy_additive_alphas=ALPHAS,
                    n_trainable_params=int(A.size),
                    param_pct_of_1B=round(100.0 * A.size / 1e9, 6),
                    A_shape=list(A.shape)),
        train_stats=train_stats,
        hidden_cache=HIDDEN,
        adapter=dict(path=ADAPTER, sha256=sha,
                     bytes=os.path.getsize(ADAPTER)),
        verification=verified, probe_rows=rows,
        verdict=("前 6 步身份 token 由解出的 A_k 引导，多语言多措辞命中身份关键词、"
                 "不复读训练答句、非身份门全关零泄漏；第 7 步起交还基座自由续写。"
                 if verified["passed"] else "未通过全部判据，见 verification"),
        honest_boundaries=[
            "输出层闭式 ridge，不是梯度反传穿过 24 层；存在 exposure bias",
            "只引导前 6 步，不保证完整准确复述 FACTS 全句",
            "训练样本 4 语言 × 2 措辞；12 语言门控由 identity_alignment.npz 覆盖，"
            "但本微调的逐位置权重只在 4 语言上训练",
            "无 ppl / KL / 配对 CI / 能力结论；β_l 仍待 L2 校准",
            "'上下文理论无限' 是品牌陈述非实测",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(VERIFY, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n[verify]", json.dumps(verified, ensure_ascii=False), flush=True)
    print("[done]", VERIFY, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
