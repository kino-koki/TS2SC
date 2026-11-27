#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v9.4 · 真正跨过天花板的路径：LoRA SFT（⚠️ 本环境未执行）

## 为什么必须走到这一层

v9.0 → v9.3c 共 12 轮，全部只在 **pre-lm_head hidden → 输出 logits** 这一层做文章
（闭式 ridge / DAgger 重解 / 真 CE+Adam 梯度 / 尺度对齐 / 解码期编排）。
`IDENTITY_FINAL_VERDICT.md` 用实验钉死了两难：

  A. 要区分生成位置 ⇒ 必须解码期按 k 切策略 ⇒ 不是纯权重；
  B. 要纯权重零编排 ⇒ 统一 Δ 在所有 k 反复抬同一批 token ⇒ exposure bias 复读。

根因：「青奈科技 / Tuan-Models / 六安市」的语义关系写在 24 层
attention/FFN 里，不在 lm_head。只改 302 行输出向量，最多改变
「在特定 hidden 上吐哪个 token」，不能让模型把新身份组织成连贯多步语句。

⇒ 梯度必须穿过 transformer 层。

## 这个脚本是什么

标准 PyTorch + PEFT LoRA SFT 骨架，把本轮已产出的资产直接接上：

  * 基座：MiniCPM5-1B 融合态（219 COPY + 24 层 SGC 分支，由 replay_full.py 组装）
  * 可训：q_proj / v_proj 的 LoRA r=16 α=32（约 0.1% 参数）
  * 数据：12 语言 × 多措辞身份问答（IDENTITY_SFT.md）
          + identity_full_hidden.json（8 样本 × 6 位置，teacher-forced）
          + identity_dagger_r1/r2.json（off-policy 轨迹，2 轮）
          + 通用能力回放语料（防灾难性遗忘，比例 ≥30%）
  * 目标：teacher-forced 全句 CE，**带 scheduled sampling**
          （训练时按 p 把前 k-1 个 ground-truth token 换成模型自己生成的前缀）
  * 验收：FINAL_VERDICT 的六道闸门 + 配对 Wilson CI + ppl 不回归

## 状态（如实）

**未执行、未验证**：本沙盒无 torch / transformers / peft / GPU。
文件里没有任何「训练结果」「分数」「生成样本」。
在 4 卡（用户此前规划）上是标准 SFT，不需要新研究，只需要算力时间。
不要在 README / 模型卡里写成本轮成果。

## 用已有资产复跑的命令（目标机）

    pip install torch transformers peft datasets accelerate bitsandbytes
    python ts2sc/lora_identity_sft.py \
        --base ./student_full \
        --hidden-cache ./identity_full_hidden.json \
        --dagger ./identity_dagger_r1.json ./identity_dagger_r2.json \
        --facts ./IDENTITY_SFT.md \
        --replay ./replay_corpus.jsonl \
        --out ./tuan_0.1_lora_adapter \
        --scheduled-sampling 0.3 --epochs 3 --lr 1e-4
    python ts2sc/verify_student_speaks.py --adapter ./tuan_0.1_lora_adapter
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

SCHEMA = {
    "status": "UNEXECUTED_SKELETON",
    "reason": "no torch / transformers / peft / CUDA in the CPU-only sandbox",
    "what_it_would_train": "LoRA on q_proj,v_proj across 24 transformer layers "
                           "(NOT just lm_head rows)",
    "why_it_crosses_the_ceiling":
        "gradients reach the layers that encode multi-step identity semantics; "
        "scheduled sampling makes h_k on the generation path in-distribution",
    "inputs": [
        "student_full/ (from replay_full.py: 219 COPY + 24-layer SGC branch)",
        "identity_full_hidden.json (8 samples x 6 positions, teacher-forced)",
        "identity_dagger_r1.json / r2.json (off-policy trajectories)",
        "IDENTITY_SFT.md (12-language multi-phrasing identity QA)",
        "replay_corpus.jsonl (>=30% general-capability replay)",
    ],
    "hyperparams": dict(r=16, alpha=32, dropout=0.05,
                        target_modules=["q_proj", "v_proj"],
                        epochs=3, lr=1e-4, scheduled_sampling_p=0.3,
                        max_seq_len=512, bf16=True, grad_ckpt=True),
    "acceptance_gates": [
        "identity_keyword_hits >= 3/4 on multilingual identity probes",
        "token_diversity >= 0.55",
        "max_bigram_repeat <= 2",
        "max_token_run < 5",
        "base_identity_leaks == 0 and semantic_inversions == 0",
        "think_prefix == 0",
        "non-identity probes: zero leakage + paired Wilson CI vs teacher",
        "general-capability ppl non-regression on replay set",
        "beta_l still pending L2 calibration (separate gate)",
    ],
    "explicitly_not_claimed": [
        "no training run, no checkpoint, no score, no generated sample",
        "no ppl / SOTA / native-level / context-8M claim",
        "does not change the 219 bit-identical teacher tensors",
        "does not touch the mixer transition matrix",
    ],
}


def load_assets(args):
    """只做存在性与结构校验；不 import torch，不读权重。"""
    out = {}
    for path in [args.hidden_cache] + list(args.dagger):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = json.load(open(path, encoding="utf-8"))
        out[os.path.basename(path)] = dict(n_keys=len(d),
                                           sample_keys=list(d)[:2])
    if not os.path.exists(args.base):
        raise FileNotFoundError(
            f"{args.base} missing — run replay_full.py on the target machine first")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./student_full")
    ap.add_argument("--hidden-cache",
                    default="/data/workspace/output/identity_full_hidden.json")
    ap.add_argument("--dagger", nargs="*", default=[
        "/data/workspace/output/identity_dagger_r1.json",
        "/data/workspace/output/identity_dagger_r2.json"])
    ap.add_argument("--facts", default="IDENTITY_SFT.md")
    ap.add_argument("--replay", default="replay_corpus.jsonl")
    ap.add_argument("--out", default="./tuan_0.1_lora_adapter")
    ap.add_argument("--scheduled-sampling", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()
    random.seed(args.seed)

    print(json.dumps(SCHEMA, indent=2, ensure_ascii=False))
    try:
        assets = load_assets(args)
        print("\n[assets-present]", json.dumps(assets, ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"\n[assets] not all present on this machine: {e}")

    # torch / peft 延迟导入：本机没有就明确停在这里，绝不伪造训练循环输出
    try:
        import torch                                   # noqa: F401
        import transformers                            # noqa: F401
        import peft                                    # noqa: F401
    except ImportError as e:
        print(f"\n[STOP] {e}: this skeleton is NOT executed in the CPU-only "
              "sandbox. Run it on the 4-GPU box with the command in the "
              "module docstring. No checkpoint or metric is produced here.",
              file=sys.stderr)
        return 3

    # ---- 以下为 TARGET MACHINE ONLY 的占位，不在本环境执行 ----
    # model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=bfloat16)
    # cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], ...)
    # model = get_peft_model(model, cfg)
    # dataset = build_mixed_dataset(facts + hidden_cache + dagger + replay)
    # trainer with scheduled_sampling(args.scheduled_sampling)
    # model.save_pretrained(args.out)
    # run verify_student_speaks.py with the six gates
    raise NotImplementedError(
        "LoRA training body is intentionally absent: it needs CUDA + PEFT. "
        "See IDENTITY_FINAL_VERDICT.md §『真正跨过去需要什么』.")


if __name__ == "__main__":
    raise SystemExit(main())
