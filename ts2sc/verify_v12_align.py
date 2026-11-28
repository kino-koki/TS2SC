#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v12 最终验证：Transformer → SimplCore 算子级替换（非 hybrid）+ 保留原能力

主判据换成**函数级对齐**，不再用「12 token 生成里有没有答案词」：
MiniCPM5 是思考型基座，短生成全落在 <think> 块里，两边都是 0 分，
判分器没有判别力。改用：
  * teacher / v12 同 prompt 的 pre-head hidden cos（≥0.90）
  * 首个 next-token argmax 一致率（≥5/6）与 top-5 重叠
  * 跨题 hidden cos（验证前缀信息**没有**坍缩）
  * 真生成 6 token，只作「会说话/不乱码」辅助证据（不作能力门槛）
  * 运行期纯度：0 个 self_attn 张量被加载；123 个继承张量逐位 sha256 对账
  * O(1) 常驻 + advance_idle(10M) 闭式实跑

单进程只持有一份 2.01 GiB LIVE 拉取的 teacher 分片，绝不并存两个模型。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = "/data/workspace/output"
FRESH = os.path.join(OUT, "v10_fresh")
TOKP = os.path.join(FRESH, "tokenizer.json")
RESULT = os.path.join(OUT, "V12_NATIVE_MIGRATION_VERDICT.json")
REPORT = os.path.join(OUT, "V12_NATIVE_MIGRATION_VERDICT.md")

from pure_exam_v11 import EXAM, chatml, grade_one, grader_sanity  # noqa: E402
from operator_substitution_v12 import SubstitutedLM, ARCH  # noqa: E402
from run_minicpm import TeacherLM, MiniCPMTokenizer  # noqa: E402
import fresh_migrate_v10 as F  # noqa: E402

PROMPTS = EXAM[:6]          # 6 题：中/英/日，事实/算术/翻译/常识
GEN_NEW = 6


def log(*a):
    print(*a, flush=True)


def readable(text):
    if not text or not text.strip():
        return False
    MOJI = set("¤Ĳ±«»¼½¾ÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß")
    n = len(text.strip())
    bad = sum(1 for c in text if c in MOJI)
    good = sum(1 for c in text if c.isalnum() or c.isspace()
               or "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff"
               or c in "，。！？：；、（）「」.,!?;:'\"-")
    return bool(n > 0 and bad / n <= 0.05 and good / n >= 0.80)


def main():
    t0 = time.time()
    # ---------- A LIVE 拉取 ----------
    log("=== A LIVE pull ===")
    orig = F.gh

    def retry(url, accept="application/vnd.github+json", timeout=300):
        last = None
        for k in range(6):
            try:
                return orig(url, accept=accept, timeout=timeout)
            except Exception as e:
                last = e
                log(f"  [retry {k+1}/6] {type(e).__name__}")
                time.sleep(2.0 * (k + 1))
        raise RuntimeError(f"gh failed: {last}")

    F.gh = retry
    try:
        V, hdr, pull = F.fresh_pull()
    finally:
        F.gh = orig
    log(f"  {pull['n_shards']} shards / "
        f"{pull['total_bytes']/2**30:.2f} GiB | "
        f"all_blob_sha={pull['all_blob_sha_verified']}")

    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    tok = MiniCPMTokenizer(TOKP)
    prompts = []
    for i, (lang, q, ans, kind) in enumerate(PROMPTS, 1):
        ids, unk = tok.encode(chatml(q), add_bos=False)
        prompts.append(dict(i=i, lang=lang, q=q, answers=ans,
                            kind=kind, ids=ids))
    log(f"  {len(prompts)} prompts encoded")

    # ---------- B teacher：单次前向拿 hidden + 首 token 分布 ----------
    log("=== B teacher (real Llama attention) ===")
    lm = TeacherLM(V, cfg)
    trows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for p in prompts:
                L = len(p["ids"])
                kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
                        np.zeros((0, lm.nkv, lm.hd), np.float32))
                       for _ in range(lm.nl)]
                h = None
                for pos, tid in enumerate(p["ids"]):
                    r = lm.step(int(tid), pos, kvs, need_logits=False,
                                fusion_hook=None,
                                return_hidden=(pos == L - 1))
                    if pos == L - 1:
                        h = r
                lg = lm.logits(h)                     # (argmax, top_ids, best_v)
                top5 = [int(x) for x in lg[1][:5]]
                # 6 token 短生成（只作可读性证据，from 同一起点重放 cache）
                kvs2 = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
                         np.zeros((0, lm.nkv, lm.hd), np.float32))
                        for _ in range(lm.nl)]
                out, fin = [], []
                for pos, tid in enumerate(p["ids"]):
                    rr = lm.step(int(tid), pos, kvs2,
                                 need_logits=(pos == L - 1),
                                 fusion_hook=None)
                    if pos == L - 1:
                        out = [int(rr[0])]
                        fin = [bool(getattr(lm, "last_logits_finite", True))]
                for j in range(1, GEN_NEW):
                    rr = lm.step(int(out[-1]), L + j - 1, kvs2,
                                 need_logits=True, fusion_hook=None)
                    out.append(int(rr[0]))
                    fin.append(bool(getattr(lm, "last_logits_finite", True)))
                    if rr[0] in (1, 130073):
                        break
                text = tok.decode(out, skip_special=True)
                trows.append(dict(hidden=np.asarray(h, np.float32).tolist(),
                                  argmax=int(lg[0]), top5=top5,
                                  text=text, finite=all(fin)))
                log(f"  [T {p['i']}/{len(prompts)}] argmax={lg[0]} | "
                    f"{text[:40]!r}")
    del lm
    gc.collect()

    # ---------- C v12：算子替换 ----------
    log("=== C v12 (operator substitution) ===")
    m = SubstitutedLM(V, cfg)
    snap0 = m.snapshot_state()
    vrows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for p in prompts:
                m.restore_state(snap0)
                for tid in p["ids"][:-1]:
                    m.step(int(tid), 0, need_logits=False)
                h, (a0, lg0) = m.step(int(p["ids"][-1]), 0,
                                      return_hidden=True, with_logits=True)
                top5 = [int(x) for x in np.argpartition(lg0, -5)[-5:]]
                m.restore_state(snap0)
                for tid in p["ids"][:-1]:
                    m.step(int(tid), 0, need_logits=False)
                out, fin = m.generate(p["ids"], max_new=GEN_NEW)
                text = tok.decode(out, skip_special=True)
                vrows.append(dict(hidden=np.asarray(h, np.float32).tolist(),
                                  argmax=int(a0), top5=top5,
                                  text=text, finite=fin,
                                  ids_in_range=bool(all(
                                      0 <= i < cfg["vocab_size"]
                                      for i in out))))
                log(f"  [V {p['i']}/{len(prompts)}] argmax={a0} | "
                    f"{text[:40]!r}")

    # ---------- D 函数级对齐 ----------
    T = np.stack([np.asarray(r["hidden"], np.float64) for r in trows])
    P = np.stack([np.asarray(r["hidden"], np.float64) for r in vrows])
    cos_tp = [float(np.dot(T[i], P[i]) /
                    (np.linalg.norm(T[i]) * np.linalg.norm(P[i]) + 1e-12))
              for i in range(len(prompts))]
    def offdiag_cos(A):
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        return (An @ An.T)[np.triu_indices(len(A), 1)]
    oT, oP = offdiag_cos(T), offdiag_cos(P)
    arg_agree = sum(int(trows[i]["argmax"] == vrows[i]["argmax"])
                    for i in range(len(prompts)))
    top5_ov = [len(set(trows[i]["top5"]) & set(vrows[i]["top5"]))
               for i in range(len(prompts))]
    sane = grader_sanity()
    tc = [grade_one(trows[i]["text"], prompts[i]["answers"])[0]
          for i in range(len(prompts))]
    vc = [grade_one(vrows[i]["text"], prompts[i]["answers"])[0]
          for i in range(len(prompts))]

    # ---------- E 纯度 / 继承账本 ----------
    log("=== E purity ledger ===")
    idx = json.load(open(os.path.join(FRESH,
        "model.safetensors.index.json"), encoding="utf-8"))
    tnames = sorted(idx["weight_map"].keys())
    man = json.load(open(os.path.join(OUT,
        "student_scoped_manifest_v10_3.json"), encoding="utf-8"))
    csha = {e["name"]: e["sha256"] for e in man["entries"]
            if e["op"] == "COPY"}
    pur = m.purity_report(tnames, csha)
    log("  " + json.dumps({k: pur[k] for k in
        ("n_loaded_self_attn", "n_inherited", "sha_checked",
         "sha_mismatches", "pure")}, ensure_ascii=False))

    # ---------- F O(1) / 10M ----------
    log("=== F O(1) / 10M ===")
    m.restore_state(snap0)
    b1 = m.state_bytes()
    tt = time.time()
    m.advance_idle(10_000_000)
    dt = time.time() - tt
    b2 = m.state_bytes()
    post, post_fin = m.generate(prompts[0]["ids"], max_new=4)
    o1 = dict(bytes_before=b1, bytes_after_10m=b2, constant=bool(b1 == b2),
              advanced=10_000_000, elapsed_s=round(dt, 6),
              post_10m_finite=post_fin, o1_closed_form=bool(dt < 1.0),
              rho10m=float(0.9995 ** 10_000_000),
              note="ρ^L 下溢 ⇒ 长程谱响应归零；O(1) 状态恒定 ≠ 长程记忆")

    v = dict(
        architecture=ARCH,
        teacher_all_live_blob_sha_verified=bool(pull["all_blob_sha_verified"]),
        n_prompts=len(prompts),
        hidden_cos=[round(x, 6) for x in cos_tp],
        hidden_cos_mean=round(float(np.mean(cos_tp)), 6),
        hidden_cos_min=round(float(np.min(cos_tp)), 6),
        first_argmax_agree=f"{arg_agree}/{len(prompts)}",
        top5_overlap=top5_ov,
        top5_overlap_mean=round(float(np.mean(top5_ov)), 4),
        teacher_cross_prompt_cos_min=round(float(oT.min()), 6),
        v12_cross_prompt_cos_min=round(float(oP.min()), 6),
        v12_prefix_information_collapsed=bool(float(oP.min()) > 0.99),
        grader_sanity_passed=sane["passed"],
        short_gen_teacher_correct=f"{sum(tc)}/{len(prompts)}",
        short_gen_v12_correct=f"{sum(vc)}/{len(prompts)}",
        short_gen_discriminative=bool(max(sum(tc), sum(vc)) > 0),
        v12_all_readable=all(readable(r["text"]) for r in vrows),
        v12_all_finite=all(r["finite"] and r["ids_in_range"]
                           for r in vrows),
        purity=pur, o1_10m=o1, hybrid_residual_T=False,
    )
    # 硬判决：写死门槛，不事后放宽
    v["THREE_CONSTRAINTS_MET"] = bool(
        pur["pure"]
        and not v["hybrid_residual_T"]
        and o1["constant"] and o1["o1_closed_form"]
        and float(np.mean(cos_tp)) >= 0.90
        and float(np.min(cos_tp)) >= 0.80
        and arg_agree >= 5
        and not v["v12_prefix_information_collapsed"]
        and v["v12_all_finite"] and sane["passed"])

    rows = [dict(i=p["i"], lang=p["lang"], q=p["q"], kind=p["kind"],
                 answers=p["answers"],
                 teacher_text=trows[i]["text"], v12_text=vrows[i]["text"],
                 teacher_argmax=trows[i]["argmax"],
                 v12_argmax=vrows[i]["argmax"],
                 argmax_agree=bool(trows[i]["argmax"] == vrows[i]["argmax"]),
                 top5_overlap=top5_ov[i], hidden_cos=round(cos_tp[i], 6))
            for i, p in enumerate(prompts)]

    rep = dict(
        title="Transformer → SimplCore 算子级替换 v12 · 真迁移验证",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        instruction="真正使其与 Transformer 真正迁移 SimplCore 并且保留原能力、"
                    "SimplCore 特性，不走 hybrid",
        what_changed=(
            "96 个 self_attn 张量从推理链**摘除**，替换为冻结旋转谱核 LTI "
            "mixer（A_m=ρ_m R(θ_m) 常矩阵，冲激响应 ρ^τ sin((τ+1)θ)，"
            "O(1) 状态、Toeplitz/FFT↔IIR 对偶）；"
            "123 个非注意力 teacher 张量（embed / RMSNorm / 24 层 FFN / "
            "lm_head）逐位继承。没有 y=T(x)+旁挂，也不是随机初始化重建。"),
        methodology_note=(
            "主判据是**函数级对齐**（hidden cos / 首 token argmax / top-5 / "
            "跨题 cos），不是短生成答案词：MiniCPM5 是思考型基座，6–12 token "
            "全落在 <think> 块里，teacher 与 v12 都会得 0，判分器没有判别力。"
            "短生成只作「会说话、不乱码」辅助证据，如实登记两边得分。"),
        rows=rows, verdict=v,
        honest_boundaries=[
            "6 题小样本是**函数级对齐探针**，不是全分布能力证明；"
            "无 ppl / benchmark / 配对 Wilson CI，不写「原生级 / 全分布能力不劣」。",
            "mixer 用解析 T1' 尺度契约（β=0.1）、冻结 ρ/θ，未做全量梯度蒸馏；"
            "DISTILL_PROTOCOL.md 仍是达到全分布保留的路径。",
            "ρ^10M 下溢如实登记；O(1) 常驻状态 ≠ 长程谱响应保留。",
            "judge 的答题分数只登记不进门槛；判分器先过 3 阳 2 阴对照。",
            "本轮不推模型仓、不覆盖 hybrid 交付；只在源码仓登记新架构。",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(RESULT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    md = [f"# {rep['title']}\n", f"{rep['date']} · {rep['elapsed_s']} s\n",
          f"## 做了什么\n\n{rep['what_changed']}\n",
          f"## 为什么换判据\n\n{rep['methodology_note']}\n",
          "## 判决\n",
          f"- **THREE_CONSTRAINTS_MET = `{v['THREE_CONSTRAINTS_MET']}`**",
          f"- hidden cos 均值 **{v['hidden_cos_mean']}**"
          f"（最低 {v['hidden_cos_min']}，门槛 ≥0.90 / ≥0.80）",
          f"- 首 token argmax 一致 **{v['first_argmax_agree']}**"
          f"（门槛 ≥5/6），top-5 平均重叠 **{v['top5_overlap_mean']}**",
          f"- 跨题 cos 最低：teacher {v['teacher_cross_prompt_cos_min']} / "
          f"v12 {v['v12_cross_prompt_cos_min']} ⇒ 前缀坍缩 "
          f"`{v['v12_prefix_information_collapsed']}`",
          f"- 纯度：运行期加载 self_attn **{pur['n_loaded_self_attn']}** 个；"
          f"继承 {pur['n_inherited']} 个逐位 sha256 对账，"
          f"mismatch={len(pur['sha_mismatches'])}",
          f"- O(1)：10M idle {o1['elapsed_s']} s，状态 "
          f"{o1['bytes_before']} → {o1['bytes_after_10m']} B；"
          f"hybrid 残差 T(x) = `{v['hybrid_residual_T']}`",
          f"- 短生成答题（仅登记）：teacher "
          f"{v['short_gen_teacher_correct']} / v12 "
          f"{v['short_gen_v12_correct']}，是否有判别力 "
          f"`{v['short_gen_discriminative']}`\n",
          "## 逐题（真实解码）\n",
          "| # | 题 | T argmax | V argmax | 一致 | hidden cos | top5∩ | "
          "teacher 文本 | v12 文本 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def esc(s):
            return s.replace("\n", "\\n").replace("|", "\\|")[:30]
        md.append(f"| {r['i']} | `{r['q'][:14]}` | {r['teacher_argmax']} | "
                  f"{r['v12_argmax']} | {'✅' if r['argmax_agree'] else '❌'} | "
                  f"{r['hidden_cos']} | {r['top5_overlap']} | "
                  f"`{esc(r['teacher_text'])}` | `{esc(r['v12_text'])}` |")
    md += ["\n## 边界\n"]
    for b in rep["honest_boundaries"]:
        md.append(f"- {b}")
    open(REPORT, "w", encoding="utf-8").write("\n".join(md) + "\n")

    log("\n" + json.dumps({k: v[k] for k in
        ("THREE_CONSTRAINTS_MET", "hidden_cos_mean", "hidden_cos_min",
         "first_argmax_agree", "top5_overlap_mean",
         "v12_prefix_information_collapsed", "short_gen_teacher_correct",
         "short_gen_v12_correct")}, ensure_ascii=False, indent=1))
    log(f"[done] {RESULT} {REPORT} {rep['elapsed_s']}s")
    del m, V
    gc.collect()


if __name__ == "__main__":
    main()
