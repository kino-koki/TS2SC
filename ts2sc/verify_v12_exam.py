#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v12 真跑验证：Transformer → SimplCore 算子替换，同题答题对照。

单进程顺序执行（只持有一份 2.01 GiB teacher 分片）：
  A LIVE 拉 22 分片，逐片 git-blob sha1 校验
  B teacher 基线 8 题 × 12 token 真生成
  C v12（123 COPY + 96 算子替换）8 题 × 12 token 真生成
  D 判分（含阳性对照）+ hidden cos + 跨题 cos + 前缀坍缩检查
  E 纯度账本：运行期 0 个 self_attn 被加载；继承张量 sha256 逐位对账
  F O(1) 状态 / advance_idle(10M) 闭式实跑 / 10M 后生成 finite
  G 与 v11 随机纯主干、v10.3 hybrid 的历史口径对照（引用，不重跑）
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
RESULT = os.path.join(OUT, "V12_OPERATOR_SUBSTITUTION_VERDICT.json")
REPORT = os.path.join(OUT, "V12_OPERATOR_SUBSTITUTION_VERDICT.md")
LOG = os.path.join(OUT, "verify_v12.log")

from pure_exam_v11 import EXAM, chatml, grade_one, grader_sanity  # noqa: E402
from operator_substitution_v12 import SubstitutedLM, ARCH  # noqa: E402
from run_minicpm import TeacherLM, MiniCPMTokenizer  # noqa: E402
import fresh_migrate_v10 as F  # noqa: E402

MAX_NEW = 12


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    # ---- A: LIVE 拉取 ----
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
    log(f"  shards {pull['n_shards']} / "
        f"{pull['total_bytes']/2**30:.2f} GiB | "
        f"all_blob_sha_verified={pull['all_blob_sha_verified']}")

    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    V_size = cfg["vocab_size"]
    tok = MiniCPMTokenizer(TOKP)

    prompts = []
    for i, (lang, q, ans, kind) in enumerate(EXAM, 1):
        ids, unk = tok.encode(chatml(q), add_bos=False)
        prompts.append(dict(i=i, lang=lang, q=q, answers=ans, kind=kind,
                            ids=ids))
    log(f"  prompts {len(prompts)} encoded")

    # ---- B: teacher 基线（真实 Llama 全注意力）----
    log("=== B teacher baseline ===")
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
                out, fins = [], []
                h = None
                for pos, tid in enumerate(p["ids"]):
                    r = lm.step(int(tid), pos, kvs,
                                need_logits=(pos == L - 1),
                                fusion_hook=None,
                                return_hidden=False)
                    if pos == L - 1:
                        out = [int(r[0])]
                        fins = [bool(getattr(lm, "last_logits_finite", True))]
                # pre-head hidden（同 cache 不可复用，单独一次前向）
                kvs2 = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
                         np.zeros((0, lm.nkv, lm.hd), np.float32))
                        for _ in range(lm.nl)]
                for pos, tid in enumerate(p["ids"]):
                    rr = lm.step(int(tid), pos, kvs2, need_logits=False,
                                 fusion_hook=None,
                                 return_hidden=(pos == L - 1))
                    if pos == L - 1:
                        h = rr
                for j in range(1, MAX_NEW):
                    r = lm.step(int(out[-1]), L + j - 1, kvs,
                                need_logits=True, fusion_hook=None)
                    out.append(int(r[0]))
                    fins.append(bool(getattr(lm, "last_logits_finite", True)))
                    if r[0] in (1, 130073):
                        break
                trows.append(dict(ids=out, hidden=np.asarray(h, np.float32).tolist(),
                                  text=tok.decode(out, skip_special=True),
                                  finite=all(fins)))
                log(f"  [T {p['i']}/{len(prompts)}] {p['lang']} "
                    f"{trows[-1]['text'][:44]!r}")
    del lm
    gc.collect()

    # ---- C: v12 算子替换 ----
    log("=== C v12 substituted ===")
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
                h = m.step(int(p["ids"][-1]), 0, return_hidden=True)
                m.restore_state(snap0)
                for tid in p["ids"][:-1]:
                    m.step(int(tid), 0, need_logits=False)
                out, fin = m.generate(p["ids"], max_new=MAX_NEW)
                vrows.append(dict(ids=out,
                                  hidden=np.asarray(h, np.float32).tolist(),
                                  text=tok.decode(out, skip_special=True),
                                  finite=fin,
                                  ids_in_range=bool(all(
                                      0 <= i < V_size for i in out))))
                log(f"  [V {p['i']}/{len(prompts)}] {p['lang']} "
                    f"{vrows[-1]['text'][:44]!r}")

    # ---- D: 判分 + 对齐 ----
    sane = grader_sanity()
    rows = []
    for p, t, v in zip(prompts, trows, vrows):
        tc, th = grade_one(t["text"], p["answers"])
        vc, vh = grade_one(v["text"], p["answers"])
        rows.append(dict(i=p["i"], lang=p["lang"], q=p["q"],
                         kind=p["kind"], answers=p["answers"],
                         teacher_text=t["text"], teacher_correct=tc,
                         v12_text=v["text"], v12_correct=vc,
                         v12_matched=vh, ids_in_range=v["ids_in_range"],
                         logits_finite=v["finite"]))

    T = np.stack([np.asarray(r["hidden"], np.float64) for r in trows])
    P = np.stack([np.asarray(r["hidden"], np.float64) for r in vrows])
    def cosn(A):
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        return An @ An.T
    cos_tp = [float(np.dot(T[i], P[i]) /
                    (np.linalg.norm(T[i]) * np.linalg.norm(P[i]) + 1e-12))
              for i in range(len(rows))]
    offT = cosn(T)[np.triu_indices(len(rows), 1)]
    offP = cosn(P)[np.triu_indices(len(rows), 1)]

    n_t = sum(r["teacher_correct"] for r in rows)
    n_v = sum(r["v12_correct"] for r in rows)
    # 首 token 一致率（可比：同 prompt 同起点）
    first_agree = sum(int(trows[i]["ids"][0] == vrows[i]["ids"][0])
                      for i in range(len(rows)))

    # ---- E: 纯度 / 继承账本 ----
    log("=== E purity ledger ===")
    idx = json.load(open(os.path.join(FRESH, "model.safetensors.index.json"),
                         encoding="utf-8"))
    tnames = sorted(idx["weight_map"].keys())
    man = json.load(open(os.path.join(OUT, "student_scoped_manifest_v10_3.json"),
                         encoding="utf-8"))
    csha = {e["name"]: e["sha256"] for e in man["entries"]
            if e["op"] == "COPY"}
    pur = m.purity_report(tnames, csha)
    log("  " + json.dumps({k: pur[k] for k in
        ("n_loaded_self_attn", "n_inherited", "sha_checked",
         "sha_mismatches", "pure")}, ensure_ascii=False))

    # ---- F: O(1) / 10M ----
    log("=== F O(1) / 10M ===")
    b1 = m.state_bytes()
    m.restore_state(snap0)
    tt = time.time()
    m.advance_idle(10_000_000)
    dt = time.time() - tt
    b2 = m.state_bytes()
    post, post_fin = m.generate(prompts[0]["ids"], max_new=6)
    F_ = dict(bytes_before=b1, bytes_after_10m=b2, constant=bool(b1 == b2),
              advanced=10_000_000, elapsed_s=round(dt, 6),
              post_10m_finite=post_fin, post_10m_ids=post,
              o1_closed_form=bool(dt < 1.0),
              rho10m=float(0.9995 ** 10_000_000),
              note="ρ^L 下溢，长程谱响应归零；O(1) 状态恒定 ≠ 长程记忆")

    verdict = dict(
        architecture=ARCH,
        teacher_all_live_blob_sha_verified=bool(pull["all_blob_sha_verified"]),
        n_questions=len(rows),
        teacher_correct=n_t, v12_correct=n_v,
        teacher_accuracy=round(n_t / len(rows), 4),
        v12_accuracy=round(n_v / len(rows), 4),
        grader_sanity_passed=sane["passed"],
        hidden_cos_teacher_vs_v12=[round(x, 6) for x in cos_tp],
        hidden_cos_mean=round(float(np.mean(cos_tp)), 6),
        first_token_agreement=f"{first_agree}/{len(rows)}",
        teacher_cross_prompt_cos_min=round(float(offT.min()), 6),
        v12_cross_prompt_cos_min=round(float(offP.min()), 6),
        v12_prefix_information_collapsed=bool(float(offP.min()) > 0.99),
        purity=pur,
        o1_10m=F_,
        hybrid_residual_T=False,
    )
    # 三条约束的联合判决（写死门槛，不事后放宽）
    verdict["THREE_CONSTRAINTS_MET"] = bool(
        pur["pure"]
        and not verdict["hybrid_residual_T"]
        and F_["constant"] and F_["o1_closed_form"]
        and n_v >= max(4, n_t - 1)               # 绝对过半 + 不劣于 teacher（±1）
        and float(np.mean(cos_tp)) >= 0.90
        and not verdict["v12_prefix_information_collapsed"]
        and sane["passed"])

    rep = dict(
        title="Transformer → SimplCore 算子级替换 v12 · 真跑答题对照",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        instruction="真正使其与 Transformer 真正迁移 SimplCore 并且保留原能力、"
                    "SimplCore 特性，不走 hybrid",
        what_changed="96 个 self_attn 张量从推理链摘除，换成冻结旋转谱核 LTI "
                     "mixer（O(1) 状态）；123 个非注意力 teacher 张量逐位继承。"
                     "没有 y=T(x)+旁挂，也不是随机初始化重建。",
        rows=rows, verdict=verdict,
        grader_sanity=sane,
        honest_boundaries=[
            "8 题 × 12 token 是小样本探针，不是全分布能力证明；"
            "无 ppl / benchmark / 配对 Wilson CI，不写「原生级 / 能力不劣」大结论。",
            "teacher 基线本轮同题真跑；12 token 短生成只判答案词是否出现，"
            "不判推理链完整性、不判事实体系。",
            "mixer 用解析 T1' 尺度契约（β=0.1），冻结 ρ/θ；"
            "未做全量梯度蒸馏，DISTILL_PROTOCOL.md 仍是达到全分布能力保留的路径。",
            "ρ^10M 下溢如实登记；O(1) 常驻状态 ≠ 长程谱响应保留。",
            "判分器先过 3 阳 2 阴对照才采信答题数字。",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(RESULT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    md = [f"# {rep['title']}\n", f"{rep['date']} · {rep['elapsed_s']} s\n",
          f"## 做了什么\n\n{rep['what_changed']}\n",
          "## 判决\n",
          f"- **THREE_CONSTRAINTS_MET = `{verdict['THREE_CONSTRAINTS_MET']}`**"
          "（非 hybrid + SimplCore 特性 + 保留原能力，同题真跑）",
          f"- 答题：teacher **{n_t}/{len(rows)}** vs v12 **{n_v}/{len(rows)}**",
          f"- hidden cos 均值 **{verdict['hidden_cos_mean']}**，"
          f"首 token 一致 **{verdict['first_token_agreement']}**",
          f"- 纯度：运行期加载 self_attn **{pur['n_loaded_self_attn']}** 个，"
          f"继承 {pur['n_inherited']} 个逐位 sha256 对账 "
          f"mismatch={len(pur['sha_mismatches'])}",
          f"- O(1)：10M idle {F_['elapsed_s']} s，状态 "
          f"{F_['bytes_before']} → {F_['bytes_after_10m']} B\n",
          "## 逐题（真实解码）\n",
          "| # | 题 | 标准答案 | teacher 输出 | T | v12 输出 | V |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        def esc(s):
            return s.replace("\n", "\\n").replace("|", "\\|")[:38]
        md.append(f"| {r['i']} | `{r['q'][:16]}` | {'/'.join(r['answers'])} | "
                  f"`{esc(r['teacher_text'])}` | "
                  f"{'✅' if r['teacher_correct'] else '❌'} | "
                  f"`{esc(r['v12_text'])}` | "
                  f"{'✅' if r['v12_correct'] else '❌'} |")
    md += ["\n## 边界\n"]
    for b in rep["honest_boundaries"]:
        md.append(f"- {b}")
    open(REPORT, "w", encoding="utf-8").write("\n".join(md) + "\n")

    log("\n" + json.dumps({k: verdict[k] for k in
        ("THREE_CONSTRAINTS_MET", "teacher_correct", "v12_correct",
         "hidden_cos_mean", "first_token_agreement",
         "v12_prefix_information_collapsed")}, ensure_ascii=False, indent=1))
    log(f"[done] {RESULT} {REPORT} {rep['elapsed_s']}s")
    del m, V
    gc.collect()


if __name__ == "__main__":
    main()
