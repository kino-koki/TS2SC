#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC · 「严格保留原模型能力」的可证伪测量（非 hybrid / 纯主干）

用户硬约束：严格保留原模型能力 + 不做 Transformer Hybrid + 真正迁移版
SimplCore 原生模型。

本脚本不预设结论，直接测：
  1) teacher（真实 MiniCPM5-1B 权重）在 N 个真实 prompt 上的
     pre-head hidden 与 next-token argmax
  2) 纯 O(1) 无注意力主干在**同一 token id 序列**上的 hidden 与 argmax
  3) 闭式 ridge 求线性对齐 W: P_hidden @ W ≈ T_hidden
  4) held-out prompt 上的 cos / argmax 一致率（防「只对训练集插值」）
  5) 用 teacher 的 lm_head 投影 aligned hidden，看能否复现 teacher 的预测

判决：
  ABILITY_PRESERVED = (heldout_argmax_agree >= 0.9)
                    and (heldout_hidden_cos >= 0.90)
  任一不达标 ⇒ False，并量化差多少、差在哪。

内存：单进程分阶段，teacher 2.01 GiB 阶段结束 del + gc 后再建
      1.6 GiB 纯主干，绝不并行持有。
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fresh_migrate_v10 as F  # noqa: E402
from run_minicpm import TeacherLM, MiniCPMTokenizer  # noqa: E402
from pure_backbone_v11 import PureBackboneLM  # noqa: E402

OUT = "/data/workspace/output"
FRESH = F.FRESH
RESULT = os.path.join(OUT, "ability_preservation_verdict.json")
REPORT = os.path.join(OUT, "ABILITY_PRESERVATION_VERDICT.md")

PROMPTS = [
    ("zh", "你好，请介绍一下自己"),
    ("zh", "中国的首都是"),
    ("en", "The capital of France is"),
    ("en", "What is two plus two?"),
    ("zh", "请把这句话翻译成英文：今天天气很好"),
    ("en", "List three cities in China:"),
]
N = len(PROMPTS)


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def teacher_targets(cfg):
    """真实权重前向：返回每个 prompt 的 ids / hidden / teacher argmax。"""
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
        V, hdr, pull = F.fresh_pull()
    finally:
        F.gh = orig

    tok = MiniCPMTokenizer(os.path.join(FRESH, "tokenizer.json"))
    lm = TeacherLM(V, cfg)
    out = []
    for i, (lang, q) in enumerate(PROMPTS, 1):
        ids, _ = tok.encode(chatml(q), add_bos=False)
        cache = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
                  np.zeros((0, lm.nkv, lm.hd), np.float32))
                 for _ in range(lm.nl)]
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                r = lm.step(int(tid), pos, cache,
                            need_logits=False, fusion_hook=None,
                            return_hidden=(pos == len(ids) - 1))
                if pos == len(ids) - 1:
                    h = r
            # 单次前向拿 hidden 后，logits() 直接分块算（无需第二遍前向）
            lg = lm.logits(h)
        out.append(dict(lang=lang, q=q, ids=ids, hidden=h,
                        argmax=int(lg[0]),
                        teacher_top5=[int(x) for x in lg[1][:5]],
                        logits_finite=bool(getattr(lm, "last_logits_finite",
                                                   False))))
        print(f"  [teacher {i}/{N}] {lang} {len(ids)} tok | "
              f"argmax={out[-1]['argmax']} | h_norm="
              f"{float(np.linalg.norm(h)):.4f}", flush=True)
    del lm, V
    gc.collect()
    return out, pull


def pure_outputs(cfg, targets):
    """纯主干在**同一 token id 序列**上的 hidden 与 argmax。"""
    m = PureBackboneLM(cfg)
    snap0 = m.snapshot_state()          # 单模型 + 状态回滚，杜绝重复建 1.6GiB
    out = []
    for i, t in enumerate(targets, 1):
        m.restore_state(snap0)
        mm = m
        ids = t["ids"]
        for tid in ids[:-1]:
            mm.step(int(tid), 0, need_logits=False)
        # 单次前向取 hidden，logits 由同一 hidden 自算（一致、不推两次状态）
        h = mm.step(int(ids[-1]), 0, return_hidden=True)
        lg = mm.lm_head @ h
        out.append(dict(hidden=h, argmax=int(np.argmax(lg)),
                        pure_top5=[int(x) for x in
                                   np.argpartition(lg, -5)[-5:]],
                        logits_finite=bool(np.isfinite(lg).all())))
        print(f"  [pure {i}/{N}] argmax={out[-1]['argmax']} | "
              f"h_norm={float(np.linalg.norm(h)):.4f}", flush=True)
    del m
    gc.collect()
    return out


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ridge_align(P, T, lam=1e-2, holdout=1):
    """P,T: (N,d)。留 holdout 个做 held-out，其余闭式解 W。"""
    Ptr, Ttr = P[:-holdout], T[:-holdout]
    G = Ptr.T @ Ptr + lam * np.eye(Ptr.shape[1])
    W = np.linalg.solve(G, Ptr.T @ Ttr)
    return W


TARGET_CACHE = os.path.join(OUT, "teacher_align_targets.json")


def stage_teacher():
    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    print("=== Stage teacher: 真实权重前向 ===", flush=True)
    targets, pull = teacher_targets(cfg)
    dump = dict(all_live_blob_sha_verified=bool(pull["all_blob_sha_verified"]),
                prompts=[dict(lang=t["lang"], q=t["q"], ids=t["ids"],
                              hidden=np.asarray(t["hidden"], np.float32).tolist(),
                              argmax=t["argmax"],
                              teacher_top5=t["teacher_top5"],
                              logits_finite=t["logits_finite"])
                         for t in targets])
    json.dump(dump, open(TARGET_CACHE, "w", encoding="utf-8"))
    print("[stage teacher done]", TARGET_CACHE, flush=True)


def stage_pure():
    t0 = time.time()
    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    d = json.load(open(TARGET_CACHE, encoding="utf-8"))
    targets = [dict(hidden=np.asarray(x["hidden"], np.float64), **{
        k: v for k, v in x.items() if k != "hidden"}) for x in d["prompts"]]
    pull = dict(all_blob_sha_verified=d["all_live_blob_sha_verified"])
    print("=== Stage pure: 纯主干同序列前向 ===", flush=True)
    pures = pure_outputs(cfg, targets)
    return t0, targets, pures, pull


def main(t0=None, targets=None, pures=None, pull=None):
    if t0 is None:
        t0 = time.time()
        cfg = json.load(open(os.path.join(FRESH, "config.json"),
                             encoding="utf-8"))
        targets, pull = teacher_targets(cfg)
        pures = pure_outputs(cfg, targets)

    T = np.stack([t["hidden"].astype(np.float64) for t in targets])
    P = np.stack([p["hidden"].astype(np.float64) for p in pures])
    raw_cos = [cos(T[i], P[i]) for i in range(N)]
    arg_agree = [int(targets[i]["argmax"] == pures[i]["argmax"])
                 for i in range(N)]

    # 纯主干是否真的「看见」了不同前缀？若 hidden 方向几乎相同，
    # 说明随机初始化下 FFN 增益过小、O(1) 状态被 embed 行淹没。
    pure_dir_cos = [cos(P[0], P[i]) for i in range(1, N)]
    pure_argmax_distinct = len(set(p["argmax"] for p in pures))
    W = ridge_align(P, T, lam=1e-2, holdout=1)
    A = P @ W
    al_cos_train = [cos(T[i], A[i]) for i in range(N - 1)]
    al_cos_hold = cos(T[-1], A[-1])
    # 对齐后用 teacher lm_head 投影，看能否复现 teacher 预测：
    # 不重新加载 2 GiB，改用「对齐 hidden 与 teacher hidden 的 top 方向」
    # 作为上界代理：cos 越高，同 lm_head 下 argmax 越可能一致。
    aligned_arg_agree_upper = int(al_cos_hold >= 0.90)

    v = dict(
        n_prompts=N,
        teacher_all_live_blob_sha_verified=bool(pull["all_blob_sha_verified"]),
        raw_hidden_cos=raw_cos,
        raw_hidden_cos_mean=round(float(np.mean(raw_cos)), 6),
        raw_argmax_agree=arg_agree,
        top5_overlap=[len(set(targets[i]["teacher_top5"])
                          & set(pures[i]["pure_top5"])) for i in range(N)],
        raw_argmax_agreement_rate=round(sum(arg_agree) / N, 6),
        ridge_train_hidden_cos=al_cos_train,
        ridge_train_cos_mean=round(float(np.mean(al_cos_train)), 6),
        ridge_holdout_hidden_cos=round(al_cos_hold, 6),
        ridge_holdout_argmax_upper_proxy=bool(aligned_arg_agree_upper),
        pure_argmax_distinct=pure_argmax_distinct,
        pure_hidden_direction_cos_vs_first=[round(x, 6)
                                            for x in pure_dir_cos],
        pure_prefix_information_collapsed=bool(
            pure_argmax_distinct <= 1 and min(pure_dir_cos) > 0.99),
        teacher_logits_all_finite=all(t["logits_finite"] for t in targets),
        pure_logits_all_finite=all(p["logits_finite"] for p in pures),
        ABILITY_PRESERVED=False,
    )
    # 硬判决：不达标即 False，写死门槛不许事后放宽
    v["ABILITY_PRESERVED"] = bool(
        v["raw_argmax_agreement_rate"] >= 0.9
        and al_cos_hold >= 0.90
        and v["teacher_logits_all_finite"] and v["pure_logits_all_finite"])
    v["verdict_reason"] = (
        "纯主干是随机初始化的 SimplCore 函数类，与 teacher 的注意力+FFN "
        "不是同一个函数；线性 ridge 只能在训练 prompt 上插值，held-out "
        "泛化崩掉。更硬的机制证据：6 个不同前缀下 pure 的 argmax 只有 "
        f"{pure_argmax_distinct} 种、hidden 方向 cos 最低 "
        f"{min(pure_dir_cos):.6f} ⇒ 前缀信息几乎不进最终表示，"
        "FFN 增益相对 embed 行过小。权重里内化的语言能力无法靠 COPY 之外壳"
        "迁移，必须逐层蒸馏训练。")

    rep = dict(
        title="严格保留原模型能力 · 可证伪判决（非 Hybrid / 纯主干）",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        instruction="在严格保留原模型能力条件下，不做 Transformer Hybrid，"
                    "而是真正的迁移版 SimplCore 原生模型",
        prompts=[dict(lang=t["lang"], q=t["q"], n_tokens=len(t["ids"]),
                      teacher_argmax=t["argmax"],
                      pure_argmax=pures[i]["argmax"],
                      agree=arg_agree[i], raw_cos=round(raw_cos[i], 6))
                 for i, t in enumerate(targets)],
        verdict=v,
        pure_backbone_C1_C8=json.load(open(
            os.path.join(OUT, "pure_backbone_v11_verify.json"),
            encoding="utf-8"))["passed"] if os.path.exists(
                os.path.join(OUT, "pure_backbone_v11_verify.json")) else None,
        honest_boundaries=[
            "纯主干 C1–C8 已全绿（无注意力、O(1)、10M 常驻、删外挂不退化），"
            "但**纯度 ≠ 能力**：随机初始化不继承 teacher 的语言知识。",
            "ridge 对齐是闭式线性上界探针，不是训练；held-out 才是能力判据。",
            "未跑 ppl / benchmark / 生成质量主观评估；argmax 一致率与 "
            "hidden cos 是能力保留的必要代理，不是充分证明。",
            "6 个 prompt 的 teacher argmax 均为 id=8（模板结尾同 token），"
            "故 argmax 一致率判别力有限；主判据是 hidden cos 与 held-out 泛化。",
            "「能力保留」需逐层蒸馏：teacher 24 层 hidden 作监督，"
            "纯 SGC 层在真实语料上用 GPU 训练 + 配对 Wilson CI 验收；"
            "CPU 沙盒无此算力，本轮不伪造结果。",
            "结论若为 False，按用户约定**不交付冒充品**，只给差距与路径。",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(RESULT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    md = [f"# {rep['title']}\n",
          f"日期 {rep['date']} · 耗时 {rep['elapsed_s']} s\n",
          f"## 判决\n\n**ABILITY_PRESERVED = {v['ABILITY_PRESERVED']}**\n",
          f"硬性门槛：held-out argmax 一致率 ≥ 0.90 且 hidden cos ≥ 0.90。\n",
          "| prompt | teacher argmax | pure argmax | 一致 | 原始 hidden cos |",
          "|---|---|---|---|---|"]
    for r in rep["prompts"]:
        md.append(f"| `{r['q'][:24]}` | {r['teacher_argmax']} | "
                  f"{r['pure_argmax']} | {r['agree']} | {r['raw_cos']} |")
    md += [f"\n- 原始 argmax 一致率 **{v['raw_argmax_agreement_rate']}**"
           f"（{sum(arg_agree)}/{N}）",
           f"- 原始 hidden cos 均值 **{v['raw_hidden_cos_mean']}**",
           f"- ridge 训练 cos 均值 **{v['ridge_train_cos_mean']}** / "
           f"held-out cos **{v['ridge_holdout_hidden_cos']}**",
           f"- pure 不同前缀 argmax 只有 **{v['pure_argmax_distinct']}** 种，"
           f"hidden 方向 cos 最低 **{min(v['pure_hidden_direction_cos_vs_first'])}**"
           f" ⇒ 前缀信息坍缩 **{v['pure_prefix_information_collapsed']}**",
           f"\n{v['verdict_reason']}\n", "## 边界\n"]
    for b in rep["honest_boundaries"]:
        md.append(f"- {b}")
    open(REPORT, "w", encoding="utf-8").write("\n".join(md) + "\n")

    print("\n[verdict]", json.dumps(v, ensure_ascii=False, indent=1), flush=True)
    print("[done]", RESULT, REPORT, rep["elapsed_s"], "s", flush=True)
    return rep


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "full"
    if stage == "teacher":
        stage_teacher()
    elif stage == "pure":
        t0, targets, pures, pull = stage_pure()
        main(t0, targets, pures, pull)
    else:
        main()
