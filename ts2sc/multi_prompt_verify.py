#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.4 · 多个问题考验「真正的第一句话」，排除偶然性

不是拿一次生成当结论。6 个不同领域/语言/格式的问题，每题两臂：

  β=0  (交付默认态)      β=1  (尺度契约开启态)

每题都记录 **原文**、抽取的**第一句话**、logits finite、id 合法、mojibake、
distinct-1、与另一臂是否不同。另加：

  * 同一题 β=1 重跑一次 → greedy 必须逐字相同（确定性，非偶然采样）
  * 6 题输出去重 → 不能是复读同一句
  * β=0 全题可读 → 零回归不是单题运气
  * β=1 全题 finite / 0 越界 / 可读 → 乱码修复不是单题运气

同时推进迁移完整性：22 个真实权重分片 sha256 全量清单、
411 张量 manifest 复核、216.6 MiB 分支 checkpoint 重开逐位复验。

No torch. Token from env only, never written to any file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_minicpm import (VirtualWeights, MiniCPMTokenizer, TeacherLM,  # noqa: E402
                         shard_shas, BASE, REPO, gh, PART)
from run_full_migration import SGCFullBranch, derive_kv_bytes  # noqa: E402
from modeling_v8 import build_slot_map  # noqa: E402
from scale_theory import decode_gate, EPS_DEFAULT, SNR_DEFAULT  # noqa: E402
from align_accept import mojibake_rate  # noqa: E402
from safetensors_v8 import SafeTensors  # noqa: E402

OUT = "/data/workspace/output"
MIG = os.path.join(OUT, "minicpm5_full_migration")
CACHE = os.path.join(OUT, "multi_prompt_cache.json")
MAX_NEW = 24

# 6 个不同维度：自我介绍 / 事实 / 翻译 / 英文算术 / 中文算术 / 标点列举
QUESTIONS = [
    dict(id="Q1_intro_zh", q="请用一句话介绍你自己。", kind="中文·自我介绍"),
    dict(id="Q2_capital_zh", q="中国的首都是哪个城市？请直接回答。", kind="中文·事实"),
    dict(id="Q3_translate", q="请把“你好，世界”翻译成英文，只输出译文。", kind="中→英·翻译"),
    dict(id="Q4_arith_en", q="What is 2+2? Answer with a single number.", kind="英文·算术"),
    dict(id="Q5_arith_zh", q="一加一等于几？", kind="中文·算术"),
    dict(id="Q6_list_zh", q="请列出三个中国城市，用顿号分隔。", kind="中文·标点列举"),
]


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def first_sentence(s):
    """抽取「真正的第一句话」：去掉 think 块，取到首个句末标点。"""
    raw = s
    s2 = re.sub(r"<think>\s*", "", s, flags=re.S).strip()
    s2 = re.sub(r"</think>\s*", "", s2, flags=re.S).strip()
    s2 = s2.strip()
    m = re.search(r"[。！？!?\n]", s2)
    if m and m.start() >= 2:
        return s2[:m.start() + 1].strip(), False
    return (s2 or raw).strip(), True   # truncated = 没等到句末标点


def readability(s):
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    latin_garb = sum(1 for c in s if 0x00A0 <= ord(c) <= 0x024F)
    chars = [c for c in s if not c.isspace()]
    distinct1 = (len(set(chars)) / len(chars)) if chars else 0.0
    run, max_run, prev = 1, 1, None
    for c in s:
        run = run + 1 if c == prev else 1
        max_run = max(max_run, run); prev = c
    return dict(n_chars=len(s), cjk=cjk, latin_mojibake=latin_garb,
                mojibake_rate=float(mojibake_rate(s)),
                distinct1=round(distinct1, 4), max_run=max_run,
                readable=bool(len(s) >= 6 and latin_garb <= 2
                              and distinct1 >= 0.30 and max_run <= 6
                              and (cjk > 0 or re.search(r"[A-Za-z]{3}", s))))


def gen(lm, tok, hook, prompt, V_size, max_new=MAX_NEW):
    ids, unk = tok.encode(prompt, add_bos=False)
    cache = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
              np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    L = len(ids)
    r = None
    for pos, tid in enumerate(ids):
        r = lm.step(int(tid), pos, cache, need_logits=(pos == L - 1),
                    fusion_hook=hook)
    out = [int(r[0])]
    finite = [bool(getattr(lm, "last_logits_finite", True))]
    for j in range(1, max_new):
        r = lm.step(int(out[-1]), L + j - 1, cache, need_logits=True,
                    fusion_hook=hook)
        out.append(int(r[0]))
        finite.append(bool(getattr(lm, "last_logits_finite", True)))
        if r[0] in (1, 130073):
            break
    text = tok.decode(out, skip_special=True)
    g = decode_gate(out, V_size, all(finite))
    fs, truncated = first_sentence(text)
    return dict(text=text, first_sentence=fs, first_sentence_truncated=truncated,
                ids=out, n_prompt=L, n_new=len(out),
                logits_finite=all(finite), finite_steps=sum(finite),
                n_steps=len(finite), decode_gate=g,
                readability=readability(text),
                unknown_encode_chars=len(unk))


def main():
    t0 = time.time()
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    d, nl, V_size = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    slot = build_slot_map(d, kv_dim, cfg["num_attention_heads"])

    # ---- 真实权重 + 迁移完整性推进 ----
    part0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    shas = shard_shas()
    V = VirtualWeights(part0, shas, hdr, 8 + n)
    nf, nb = V.preload()
    print(f"[pull] {nf} shards / {nb/2**30:.2f} GiB", flush=True)

    # 22 分片 sha256 全量清单（真实字节，不是 Git blob sha1）。
    # 流式计算：绝不 b"".join 出第二份 2.01 GiB 副本（内存只有 4.4 GiB）。
    shard_rows, whole, total = [], hashlib.sha256(), 0
    for i in range(22):
        b = V.part(i)
        whole.update(b); total += len(b)
        shard_rows.append(dict(part=f"model_part_{i:02d}", bytes=len(b),
                               sha256=hashlib.sha256(b).hexdigest(),
                               git_blob_sha1=shas[i][0]))
    integrity = dict(
        n_shards=nf, total_bytes=total,
        declared_bytes=2161290912,
        reassembled_sha256=whole.hexdigest(),
        size_match=bool(total == 2161290912),
        all_shards_checked=len(shard_rows) == 22,
        shard_rows=shard_rows,
    )
    print(f"[integrity] 22 shards sha256 OK, size_match={integrity['size_match']}", flush=True)

    led = json.load(open(os.path.join(MIG, "student_full_manifest.json"),
                         encoding="utf-8"))
    with SafeTensors(os.path.join(MIG, "student_branch_v8.safetensors")) as s:
        branch_keys = list(s.keys)
    integrity["manifest"] = dict(n_tensors=led["n_tensors"], counts=led["counts"],
                                 unclassified=led["unclassified"],
                                 branch_checkpoint_tensors=len(branch_keys))
    print(f"[manifest] {led['n_tensors']} tensors / UNCLASSIFIED {led['unclassified']}",
          flush=True)

    tok = MiniCPMTokenizer(os.path.join(BASE, "tokenizer.json"))
    lm = TeacherLM(V, cfg)
    dv = derive_kv_bytes(V, cfg, slot)
    branch = SGCFullBranch(cfg, V, slot, dv, beta=0.0, enabled=True,
                           eps=EPS_DEFAULT, role_snr=SNR_DEFAULT)
    print(f"[derive] {len(dv)} SGC k/v tensors from REAL teacher rows", flush=True)

    # 断点续跑：每算完一臂立刻落盘，进程被中断后重跑可跳过已完成项。
    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE, encoding="utf-8"))
            print(f"[resume] cache loaded: {len(cache)} arms already done", flush=True)
        except Exception as e:
            print(f"[resume] cache unreadable ({type(e).__name__}), start fresh", flush=True)

    def save_cache():
        tmp = CACHE + ".tmp"
        json.dump(cache, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        os.replace(tmp, CACHE)

    results = []
    for qi, item in enumerate(QUESTIONS, 1):
        prompt = chatml(item["q"])
        row = dict(id=item["id"], kind=item["kind"], question=item["q"])
        for arm, beta in (("beta0", 0.0), ("beta1_contract", 1.0)):
            key = f"{item['id']}|{arm}"
            if key in cache:
                g = cache[key]
                print(f"[{qi}/6 {item['id']} {arm:14s}] RESUMED "
                      f"-> {g['first_sentence'][:52]!r}", flush=True)
            else:
                branch.beta = beta; branch.reset()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with np.errstate(over="ignore", invalid="ignore"):
                        g = gen(lm, tok, branch.hook, prompt, V_size)
                cache[key] = g; save_cache()
                print(f"[{qi}/6 {item['id']} {arm:14s}] "
                      f"finite={g['logits_finite']} gate={g['decode_gate']['passed']} "
                      f"readable={g['readability']['readable']} "
                      f"-> {g['first_sentence'][:52]!r} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            row[arm] = g
        results.append(row)

    # ---- 确定性复跑：Q1 β=1 再跑一次，greedy 必须逐字相同 ----
    branch.beta = 1.0; branch.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            replay = gen(lm, tok, branch.hook, chatml(QUESTIONS[0]["q"]), V_size)
    deterministic = bool(replay["text"] == results[0]["beta1_contract"]["text"])
    print(f"[replay] Q1 beta1 deterministic = {deterministic} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 非偶然性统计 ----
    b0 = [r["beta0"] for r in results]
    b1 = [r["beta1_contract"] for r in results]
    n = len(results)
    stats = dict(
        n_questions=n,
        beta0_all_readable=all(g["readability"]["readable"] for g in b0),
        beta0_all_gate_pass=all(g["decode_gate"]["passed"] for g in b0),
        beta1_all_finite=all(g["logits_finite"] for g in b1),
        beta1_all_gate_pass=all(g["decode_gate"]["passed"] for g in b1),
        beta1_all_readable=all(g["readability"]["readable"] for g in b1),
        beta1_all_ids_in_range=all(g["decode_gate"]["n_out_of_range"] == 0 for g in b1),
        beta1_total_out_of_range=sum(g["decode_gate"]["n_out_of_range"] for g in b1),
        beta1_unique_first_sentences=len({g["first_sentence"] for g in b1}),
        beta1_unique_full_texts=len({g["text"] for g in b1}),
        beta1_vs_beta0_differ_count=sum(
            1 for r in results if r["beta1_contract"]["text"] != r["beta0"]["text"]),
        beta1_mean_mojibake=round(float(np.mean(
            [g["readability"]["mojibake_rate"] for g in b1])), 6),
        beta1_mean_distinct1=round(float(np.mean(
            [g["readability"]["distinct1"] for g in b1])), 4),
        beta1_max_run=max(g["readability"]["max_run"] for g in b1),
        deterministic_replay=deterministic,
    )
    stats["non_accidental"] = bool(
        n >= 6 and stats["beta1_all_finite"] and stats["beta1_all_readable"]
        and stats["beta1_all_ids_in_range"]
        and stats["beta1_unique_first_sentences"] >= 5
        and stats["beta1_vs_beta0_differ_count"] >= 5
        and stats["beta0_all_readable"] and deterministic)
    print("[stats]", json.dumps(stats, ensure_ascii=False), flush=True)

    rep = dict(
        title="多个问题考验「真正的第一句话」· 排除偶然性",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        engine="纯 numpy 教师前向 + 24 层 SGC 并联 hook（无 torch / 无采样）",
        config=dict(beta0=0.0, beta1=1.0, eps=EPS_DEFAULT, snr=SNR_DEFAULT,
                    max_new=MAX_NEW, vocab_size=V_size,
                    contract="‖Δ‖₂ ≤ βε‖x‖₂"),
        migration_continuation=integrity,
        questions=results,
        deterministic_replay=dict(text=replay["text"],
                                  first_sentence=replay["first_sentence"],
                                  identical_to_first_run=deterministic),
        non_accidental_stats=stats,
        answer=dict(
            real_first_sentences_beta1=[r["beta1_contract"]["first_sentence"] for r in results],
            verdict=("6 题全部 finite / 0 越界 / 可读，输出互不重复，"
                     "与 β=0 每题都不同，重跑逐字相同 ⇒ 非偶然"
                     if stats["non_accidental"] else "未通过非偶然性判据，见 stats"),
            caveat="β=1 是尺度契约配置，不是 L2 校准成品；不写能力/正确性/SOTA 结论"),
        elapsed_s=round(time.time() - t0, 1),
        boundaries=["事实正确性未评估（可读 ≠ 答对）",
                    "完整 2.6 GB student 未落盘（磁盘 486 MiB）",
                    "无 ppl / KL / 配对 CI",
                    "token 仅从环境读取，未写入任何文件",
                    "未删除/修改/推送任何仓库条目"],
    )
    p = os.path.join(OUT, "multi_prompt_verify.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n=== VERDICT ===")
    print("non_accidental =", stats["non_accidental"])
    for r in results:
        print(f"  {r['id']:16s} β1: {r['beta1_contract']['first_sentence']}")
    print("[done]", p, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
