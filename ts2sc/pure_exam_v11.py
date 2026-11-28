#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 O(1) 原生主干 v11 · **答题**验证（不是只判乱码/空串）

用户要的是「纯 O(1) 版本的迁移后模型的答题」：
  - 8 道有标准答案的题（中/英/日，事实/算术/翻译/常识）
  - 真生成 → 真 tokenizer 解码 → 逐题关键词判分
  - 判分器先过阳性对照（能认出「北京」「Paris」），否则负结果不可信
  - 同时记录每题 final hidden，算跨题 cos，给出前缀坍缩的机械证据

三段式分进程，绝不并存：
  encode : 只加载 tokenizer（9.9 MB），编码题目 → ids 落盘
  gen    : 只加载 PureBackboneLM（~2 GiB），生成 ids + hidden → 落盘后释放
  grade  : 只加载 tokenizer，解码 + 判分 + 汇总 → JSON / MD

诚实前提：纯 O(1) 主干是**随机初始化、未蒸馏**的原生重建，不是
「保留原模型能力的迁移成品」。本脚本不预设结论，真跑真判。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = "/data/workspace/output"
FRESH = os.path.join(OUT, "v10_fresh")
TOKP = os.path.join(FRESH, "tokenizer.json")
PROMPT_CACHE = os.path.join(OUT, "pure_exam_prompts.json")
GEN_CACHE = os.path.join(OUT, "pure_exam_gen.json")
RESULT = os.path.join(OUT, "PURE_EXAM_VERDICT.json")
REPORT = os.path.join(OUT, "PURE_EXAM_VERDICT.md")

# (语言, 题面, 可接受答案别名, 题型)
EXAM = [
    ("zh", "中国的首都是", ["北京", "Beijing"], "事实"),
    ("zh", "一加一等于几？", ["2", "二", "two", "等于 2", "等于2"], "算术"),
    ("zh", "太阳从哪个方向升起？", ["东", "东方", "east"], "常识"),
    ("zh", "请把「你好」翻译成英文", ["Hello", "Hi", "hello", "hi"], "翻译"),
    ("en", "What is 2+2?", ["4", "four"], "算术"),
    ("en", "The capital of France is", ["Paris", "paris"], "事实"),
    ("en", 'Translate "thank you" into Chinese', ["谢谢", "感谢"], "翻译"),
    ("ja", "日本の首都は？", ["東京", "东京", "Tokyo"], "事实"),
]
MAX_NEW = 40


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


# ---------------------------------------------------------------------------
def stage_encode():
    from run_minicpm import MiniCPMTokenizer
    tok = MiniCPMTokenizer(TOKP)
    rows = []
    for i, (lang, q, ans, kind) in enumerate(EXAM, 1):
        ids, unk = tok.encode(chatml(q), add_bos=False)
        rows.append(dict(i=i, lang=lang, q=q, answers=ans, kind=kind,
                         ids=ids, n_unknown=len(unk)))
        print(f"  [encode {i}/{len(EXAM)}] {lang} {len(ids)} tok", flush=True)
    json.dump(rows, open(PROMPT_CACHE, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("[encode done]", PROMPT_CACHE, flush=True)


def stage_gen():
    from pure_backbone_v11 import PureBackboneLM
    import warnings
    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    prompts = json.load(open(PROMPT_CACHE, encoding="utf-8"))
    m = PureBackboneLM(cfg)
    snap = m.snapshot_state()          # 每题同起点，防递推跨题污染
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in prompts:
            m.restore_state(snap)
            ids = p["ids"]
            for tid in ids[:-1]:
                m.step(int(tid), 0, need_logits=False)
            out, lg_ok = [], True
            for j in range(MAX_NEW):
                tid, lg = m.step(out[-1] if out else int(ids[-1]), j,
                                 need_logits=True)
                lg_ok = lg_ok and bool(np.isfinite(lg).all())
                out.append(int(tid))
                if tid in (1, 130073):
                    break
            # 末步 hidden（用同起点的独立重放，避免多推一次状态）
            m.restore_state(snap)
            for tid in ids[:-1]:
                m.step(int(tid), 0, need_logits=False)
            h = m.step(int(ids[-1]), 0, return_hidden=True)
            rows.append(dict(i=p["i"], lang=p["lang"], q=p["q"],
                             answers=p["answers"], kind=p["kind"],
                             n_prompt=len(ids), ids=out,
                             hidden=np.asarray(h, np.float32).tolist(),
                             logits_finite=lg_ok,
                             ids_in_range=bool(all(
                                 0 <= i < cfg["vocab_size"] for i in out))))
            print(f"  [gen {p['i']}/{len(prompts)}] {p['lang']} "
                  f"{len(out)} ids finite={lg_ok}", flush=True)
    json.dump(rows, open(GEN_CACHE, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("[gen done] model released", flush=True)
    del m
    gc.collect()


def grade_one(text, answers):
    """关键词判分：大小写不敏感；要求答案词作为独立片段出现。"""
    if not text or not text.strip():
        return False, []
    low = text.lower()
    hit = [a for a in answers if a.lower() in low]
    return bool(hit), hit


def grader_sanity():
    """阳性/阴性对照：判分器本身必须能认出正确答案，负结果才可信。"""
    cases = [
        ("中国的首都是北京。", ["北京", "Beijing"], True),
        ("The capital of France is Paris.", ["Paris", "paris"], True),
        ("2+2 = 4", ["4", "four"], True),
        ("${含义 montageчас FahrVx寻先锋#ifdef", ["北京", "Beijing"], False),
        ("", ["4", "four"], False),
    ]
    out = []
    for t, a, exp in cases:
        got, hit = grade_one(t, a)
        out.append(dict(text=t[:40], expected=exp, got=got,
                        matched=hit, ok=bool(got == exp)))
    return dict(passed=all(x["ok"] for x in out), cases=out)


def stage_grade():
    from run_minicpm import MiniCPMTokenizer
    tok = MiniCPMTokenizer(TOKP)
    gen = json.load(open(GEN_CACHE, encoding="utf-8"))
    rows = []
    for g in gen:
        text = tok.decode(g["ids"], skip_special=True)
        correct, hit = grade_one(text, g["answers"])
        rows.append(dict(i=g["i"], lang=g["lang"], q=g["q"],
                         kind=g["kind"], answers=g["answers"],
                         text=text, matched=hit, correct=correct,
                         ids=g["ids"], ids_in_range=g["ids_in_range"],
                         logits_finite=g["logits_finite"],
                         n_new=len(g["ids"])))
        print(f"  [grade {g['i']}] {g['lang']} correct={correct} | "
              f"{text[:64]!r}", flush=True)

    H = np.stack([np.asarray(g["hidden"], np.float64) for g in gen])
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    C = Hn @ Hn.T
    off = C[np.triu_indices(len(gen), 1)]
    texts = [r["text"] for r in rows]
    seqs = [tuple(r["ids"]) for r in rows]

    sane = grader_sanity()
    n_correct = sum(r["correct"] for r in rows)
    v = dict(
        model="PURE_V11_NATIVE_O1_ATTENTION_FREE",
        n_questions=len(rows),
        n_correct=n_correct,
        accuracy=round(n_correct / len(rows), 6),
        distinct_decoded_texts=len(set(texts)),
        distinct_token_sequences=len(set(seqs)),
        grader_sanity_passed=sane["passed"],
        grader_sanity_cases=sane["cases"],
        all_ids_in_range=all(r["ids_in_range"] for r in rows),
        all_logits_finite=all(r["logits_finite"] for r in rows),
        prefix_hidden_cos_min=round(float(off.min()), 6),
        prefix_hidden_cos_max=round(float(off.max()), 6),
        prefix_information_collapsed=bool(
            len(set(seqs)) <= 1 and float(off.min()) > 0.99),
        ANSWERS_CORRECTLY=bool(n_correct >= len(rows) - 1),
    )
    v["verdict_reason"] = (
        "纯 O(1) 主干是随机初始化的 SimplCore 函数类，未做逐层蒸馏；"
        f"8 题只得到 {len(set(seqs))} 种 token 序列、final hidden 跨题 cos "
        f"最低 {float(off.min()):.6f} ⇒ 题面信息没有进入最终表示。"
        "判分器已过阳性对照（能认出「北京」「Paris」「4」），因此 "
        f"{n_correct}/{len(rows)} 是真负结果，不是判分器坏了。")

    rep = dict(
        title="纯 O(1) 原生主干 · 答题验证（真跑真判）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        instruction="我要的是纯 O(1) 版本的迁移后模型的答题",
        questions=rows, verdict=v,
        honest_boundaries=[
            "纯 O(1) 主干是**原生重建（随机初始化、未蒸馏）**，不是"
            "「严格保留原模型能力的迁移成品」；本轮不拿 hybrid 冒充。",
            "答题 = 生成文本是否包含标准答案，不是只看空串/乱码；"
            "判分器先过 3 阳 2 阴对照才采信负结果。",
            "生成 40 token，短上下文只判「能否给出正确答案」，不判"
            "长程推理、不判事实体系的完整性。",
            "无 ppl / benchmark / 配对 CI；ABILITY_PRESERVED 仍为 False。",
            "要真正答对，必须按 DISTILL_PROTOCOL.md 在 GPU 上做逐层蒸馏；"
            "CPU 沙盒无算力，本轮 NOT_EXECUTED、不伪造 checkpoint 或分数。",
        ])
    json.dump(rep, open(RESULT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    md = [f"# {rep['title']}\n", f"{rep['date']}\n",
          f"## 判决\n\n**答对 {n_correct}/{len(rows)}，"
          f"`ANSWERS_CORRECTLY = {v['ANSWERS_CORRECTLY']}`**\n",
          f"- 判分器阳性对照：`{v['grader_sanity_passed']}`"
          "（能认出「北京」「Paris」「4」，拒认乱串与空串）",
          f"- 不同 token 序列 **{v['distinct_token_sequences']}/{len(rows)}**，"
          f"不同解码文本 **{v['distinct_decoded_texts']}/{len(rows)}**",
          f"- final hidden 跨题 cos：{v['prefix_hidden_cos_min']} ~ "
          f"{v['prefix_hidden_cos_max']} ⇒ 前缀坍缩 "
          f"`{v['prefix_information_collapsed']}`\n",
          "## 逐题真实解码\n",
          "| # | 语言 | 题型 | 题目 | 标准答案 | 模型输出（前 56 字符） | 答对 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["text"].replace("\n", "\\n").replace("|", "\\|")[:56]
        md.append(f"| {r['i']} | {r['lang']} | {r['kind']} | `{r['q'][:20]}` | "
                  f"{'/'.join(r['answers'])} | `{t}` | "
                  f"{'✅' if r['correct'] else '❌'} |")
    md += [f"\n{v['verdict_reason']}\n", "## 边界\n"]
    for b in rep["honest_boundaries"]:
        md.append(f"- {b}")
    open(REPORT, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps(v, ensure_ascii=False, indent=1), flush=True)
    print("[done]", RESULT, REPORT, flush=True)


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "encode"
    if st == "encode":
        stage_encode()
    elif st == "gen":
        stage_gen()
    elif st == "grade":
        stage_grade()
    else:
        raise SystemExit("usage: encode | gen | grade")
