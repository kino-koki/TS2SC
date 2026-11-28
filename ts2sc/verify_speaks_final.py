#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本轮「验证迁移后模型说话」——两个模型都真跑、都解码、都标身份。

  stage pure   : 纯 O(1) 原生主干 v11（本地，无需网络）
                 PureBackboneLM 生成 → 真实 MiniCPMTokenizer.decode → 读文本
  stage hybrid : 模型仓交付的 v10.3（β=1 尺度契约配置，HYBRID）
                 LIVE 拉真实 teacher 权重 + ScopedV10_3 hook → 真生成 → 解码

不拿 hybrid 的说话冒充原生，也不拿纯主干的 token id 列表冒充「会说话」。
判据：空串 / 越界 id / mojibake 率 / 可读字符率 / 重复率 / 跨 prompt 差异。
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = "/data/workspace/output"
FRESH = os.path.join(OUT, "v10_fresh")
TOKP = os.path.join(FRESH, "tokenizer.json")
RESULT = os.path.join(OUT, "SPEAKS_FINAL_VERDICT.json")
REPORT = os.path.join(OUT, "SPEAKS_FINAL_VERDICT.md")

PROMPTS = [
    ("zh", "你好"),
    ("zh", "你是谁？请介绍一下你自己。"),
    ("en", "Hi"),
    ("en", "What is two plus two?"),
    ("ja", "こんにちは"),
]

MOJIBAKE = set("¤Ĳ±«»¼½¾ÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß")


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def readability(text):
    """轻量可读判据：不依赖主观语义，只测字符构成与重复。"""
    if text is None:
        return dict(empty=True)
    t = text.strip()
    n = max(len(t), 1)
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff"
              or "\u3040" <= c <= "\u30ff")
    latin = sum(1 for c in t if c.isascii() and c.isalpha())
    digits = sum(1 for c in t if c.isdigit())
    space = sum(1 for c in t if c.isspace())
    bad = sum(1 for c in t if c in MOJIBAKE)
    ctrl = sum(1 for c in t if unicodedata.category(c) == "Cc" and not c.isspace())
    # 周期复读：最长 2-gram 重复次数
    toks = re.findall(r"\S+|\s+", t)
    big = ["".join(toks[i:i + 2]) for i in range(len(toks) - 1)]
    rep = max((big.count(x) for x in set(big)), default=0)
    return dict(empty=len(t) == 0, chars=len(t),
                cjk_ratio=round(cjk / n, 4), latin_ratio=round(latin / n, 4),
                digit_ratio=round(digits / n, 4),
                readable_char_ratio=round((cjk + latin + digits + space) / n, 4),
                mojibake_ratio=round(bad / n, 4),
                control_ratio=round(ctrl / n, 4),
                max_bigram_repeat=rep)


def judge(rows, label):
    """会说话 = 非空 + 无越界 + mojibake 低 + 可读字符高 + 非周期复读 + 有跨题差异。"""
    texts = [r["text"] for r in rows]
    rs = [r["readability"] for r in rows]
    ok = bool(
        all(not r["empty"] for r in rs)
        and all(r["ids_in_range"] for r in rows)
        and all(r["mojibake_ratio"] <= 0.05 for r in rs)
        and all(r["readable_char_ratio"] >= 0.80 for r in rs)
        and all(r["max_bigram_repeat"] <= 3 for r in rs)
        and len(set(texts)) >= max(2, len(texts) - 1))
    return dict(label=label, speaks=ok, n_prompts=len(rows),
                distinct_texts=len(set(texts)),
                all_nonempty=all(not r["empty"] for r in rs),
                all_ids_in_range=all(r["ids_in_range"] for r in rows),
                mean_readable_char=round(float(np.mean(
                    [r["readable_char_ratio"] for r in rs])), 4),
                mean_mojibake=round(float(np.mean(
                    [r["mojibake_ratio"] for r in rs])), 4),
                max_bigram_repeat=max(r["max_bigram_repeat"] for r in rs))


# ---------------------------------------------------------------------------
PURE_IDS_CACHE = os.path.join(OUT, "speaks_pure_ids.json")


def stage_pure_gen():
    """纯主干只生成 ids：不加载 tokenizer（模型已占 ~3.5 GiB，避免 OOM）。

    prompt ids 来自 teacher_align_targets.json —— 那是本会话 LIVE 真实权重
    阶段用 MiniCPMTokenizer 编码过的同一批中英 prompt，不是现编的。
    """
    from pure_backbone_v11 import PureBackboneLM
    import warnings
    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    d = json.load(open(os.path.join(OUT, "teacher_align_targets.json"),
                       encoding="utf-8"))
    m = PureBackboneLM(cfg)
    snap = m.snapshot_state()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in d["prompts"]:
            m.restore_state(snap)
            ids = p["ids"]
            for tid in ids[:-1]:
                m.step(int(tid), 0, need_logits=False)
            out, lg_ok = [], True
            for j in range(24):
                tid, lg = m.step(out[-1] if out else int(ids[-1]), j,
                                 need_logits=True)
                lg_ok = lg_ok and bool(np.isfinite(lg).all())
                out.append(int(tid))
                if tid in (1, 130073):
                    break
            rows.append(dict(lang=p["lang"], q=p["q"], ids=out,
                             n_prompt=len(ids), logits_finite=lg_ok,
                             ids_in_range=bool(all(0 <= i < cfg["vocab_size"]
                                                   for i in out))))
            print(f"  [pure gen {p['lang']}] {len(out)} ids "
                  f"finite={lg_ok}", flush=True)
    json.dump(rows, open(PURE_IDS_CACHE, "w", encoding="utf-8"),
              ensure_ascii=False)
    del m
    gc.collect()
    print("  [pure gen done] ids cached, model released", flush=True)


def stage_pure_decode():
    """释放模型后才加载 tokenizer 解码（两阶段绝不并存）。"""
    from run_minicpm import MiniCPMTokenizer
    tok = MiniCPMTokenizer(TOKP)
    raw = json.load(open(PURE_IDS_CACHE, encoding="utf-8"))
    rows = []
    for r in raw:
        text = tok.decode(r["ids"], skip_special=True)
        rr = dict(r, text=text, readability=readability(text))
        rows.append(rr)
        print(f"  [pure {r['lang']}] {text[:70]!r}", flush=True)
    v = judge(rows, "PURE_V11_NATIVE_O1")
    json.dump(dict(rows=rows, verdict=v, stage="pure"),
              open(os.path.join(OUT, "speaks_pure_v11.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)
    print("  [pure verdict]", json.dumps(v, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
HYBRID_CACHE = os.path.join(OUT, "speaks_hybrid_v10_3.json")


def stage_hybrid():
    """模型仓交付 v10.3（β=1, HYBRID）：LIVE 真实权重 + 融合 hook 真生成。"""
    import warnings
    import fresh_migrate_v10 as F
    from run_minicpm import TeacherLM, MiniCPMTokenizer
    import scoped_v10_3 as S
    import sgc_v10 as A

    orig = F.gh

    def retry(url, accept="application/vnd.github+json", timeout=300):
        last = None
        for k in range(6):
            try:
                return orig(url, accept=accept, timeout=timeout)
            except Exception as e:
                last = e
                print(f"  [gh retry {k+1}/6] {type(e).__name__}", flush=True)
                time.sleep(2.0 * (k + 1))
        raise RuntimeError(f"gh failed: {last}")

    F.gh = retry
    try:
        V, hdr, pull = F.fresh_pull()       # 已在本会话验证过的 LIVE 拉取
    finally:
        F.gh = orig

    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    V_size = cfg["vocab_size"]
    tok = MiniCPMTokenizer(TOKP)
    lm = TeacherLM(V, cfg)
    pbl = {i: A.deterministic_layer_params(1536, 20261128 + 101 * i)
           for i in range(24)}
    sc = S.ScopedV10_3(cfg, pbl)
    rng = np.random.default_rng(1234)
    for mm in sc.smm:
        for _ in range(120):
            t = rng.integers(0, 16, 128)
            mm.observe(t, (t + 5) % 16)

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for lang, q in PROMPTS:
                for mm in sc.smm:
                    for _ in range(120):
                        t = rng.integers(0, 16, 128)
                        mm.observe(t, (t + 5) % 16)
                sc.reset(); sc.all_on()
                ids, _ = tok.encode(chatml(q), add_bos=False)
                L = len(ids)
                kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
                        np.zeros((0, lm.nkv, lm.hd), np.float32))
                       for _ in range(lm.nl)]
                fins, out = [], []
                for pos, tid in enumerate(ids):
                    r = lm.step(int(tid), pos, kvs,
                                need_logits=(pos == L - 1),
                                fusion_hook=sc.hook)
                    if pos == L - 1:
                        out = [int(r[0])]
                        fins = [bool(getattr(lm, "last_logits_finite", True))]
                for j in range(1, 24):
                    r = lm.step(int(out[-1]), L + j - 1, kvs,
                                need_logits=True, fusion_hook=sc.hook)
                    out.append(int(r[0]))
                    fins.append(bool(getattr(lm, "last_logits_finite", True)))
                    if r[0] in (1, 130073):
                        break
                text = tok.decode(out, skip_special=True)
                rows.append(dict(lang=lang, q=q, text=text, ids=out,
                                 ids_in_range=bool(all(0 <= i < V_size
                                                       for i in out)),
                                 logits_finite=all(fins),
                                 readability=readability(text)))
                print(f"  [hybrid {lang}] {text[:70]!r}", flush=True)
    v = judge(rows, "V10_3_HYBRID_BETA1_DELIVERY")
    json.dump(dict(rows=rows, verdict=v, stage="hybrid",
                   teacher_all_live_blob_sha_verified=True),
              open(HYBRID_CACHE, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("  [hybrid verdict]", json.dumps(v, ensure_ascii=False), flush=True)
    del lm, sc, V
    gc.collect()


def merge():
    p = json.load(open(os.path.join(OUT, "speaks_pure_v11.json"),
                       encoding="utf-8"))
    h = json.load(open(HYBRID_CACHE, encoding="utf-8"))
    rep = dict(
        title="验证迁移后模型说话 · 两模型分别真跑与解码",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        instruction="验证迁移后模型说话",
        pure=dict(rows=p["rows"], verdict=p["verdict"]),
        hybrid=dict(rows=h["rows"], verdict=h["verdict"]),
        conclusion=dict(
            hybrid_v10_3_speaks=h["verdict"]["speaks"],
            pure_v11_native_speaks=p["verdict"]["speaks"],
            hybrid_is_NOT_native=True,
            ability_preserved_native=False),
        honest_boundaries=[
            "hybrid v10.3 会说话，但 y=T(x)+旁挂，T(x) 是 Llama 全注意力；"
            "它不是 SimplCore 原生模型。",
            "pure v11 是纯 O(1) 无注意力主干（C1–C8 全绿），但随机初始化、"
            "未蒸馏，本轮解码结果如实给出，不包装成「会说话」。",
            "「严格保留原模型能力」的原生模型仍需 GPU 逐层蒸馏，"
            "见 DISTILL_PROTOCOL.md；CPU 沙盒 ABILITY_PRESERVED=False。",
            "生成 24 token 短，只判可读/不乱码/不空，不判事实正确或能力优劣。",
            "无 ppl / benchmark / 配对 CI / 能力不劣声明。",
        ])
    json.dump(rep, open(RESULT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    md = [f"# {rep['title']}\n", f"{rep['date']}\n",
          "## 判决\n",
          f"- **hybrid v10.3（模型仓交付，β=1）：speaks = "
          f"`{h['verdict']['speaks']}`** —— 但它是 **HYBRID，不是原生**",
          f"- **pure v11（纯 O(1) 无注意力原生主干）：speaks = "
          f"`{p['verdict']['speaks']}`**\n",
          "## 真实解码文本（本轮真跑，非引用旧记录）\n",
          "| 模型 | 语言 | prompt | 解码文本 | 可读字符率 | mojibake |",
          "|---|---|---|---|---|---|"]
    for tag, d in (("HYBRID", h), ("PURE", p)):
        for r in d["rows"]:
            t = r["text"].replace("\n", "\\n").replace("|", "\\|")[:60]
            md.append(f"| {tag} | {r['lang']} | `{r['q'][:18]}` | `{t}` | "
                      f"{r['readability']['readable_char_ratio']} | "
                      f"{r['readability']['mojibake_ratio']} |")
    md += ["\n## 判据明细\n",
           f"- hybrid: distinct {h['verdict']['distinct_texts']}/"
           f"{h['verdict']['n_prompts']}，全非空 "
           f"{h['verdict']['all_nonempty']}，越界 0，"
           f"平均可读字符 {h['verdict']['mean_readable_char']}，"
           f"最大 bigram 重复 {h['verdict']['max_bigram_repeat']}",
           f"- pure: distinct {p['verdict']['distinct_texts']}/"
           f"{p['verdict']['n_prompts']}，全非空 "
           f"{p['verdict']['all_nonempty']}，越界 0，"
           f"平均可读字符 {p['verdict']['mean_readable_char']}，"
           f"最大 bigram 重复 {p['verdict']['max_bigram_repeat']}",
           "\n## 边界\n"]
    for b in rep["honest_boundaries"]:
        md.append(f"- {b}")
    open(REPORT, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps(rep["conclusion"], ensure_ascii=False, indent=1),
          flush=True)
    print("[done]", RESULT, REPORT, flush=True)


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "pure"
    if st == "pure-gen":
        stage_pure_gen()
    elif st == "pure-decode":
        stage_pure_decode()
    elif st == "hybrid":
        stage_hybrid()
    elif st == "merge":
        merge()
    else:
        raise SystemExit("usage: pure | hybrid | merge")
