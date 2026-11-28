#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 · L2 逐层 β_l 校准协议骨架（PLAN-ONLY / NOT EXECUTED）

=============================================================================
状态：本文件**没有被执行过**。不产出 checkpoint、不产出分数、不产出任何数字。
      它只把「β=1 是尺度契约配置，不是校准成品」这句话变成一份可核对的
      实验协议，供维护者在真实 GPU / 目标机上执行。
=============================================================================

## 为什么需要它

v10 交付默认 `beta=1.0` 证明的是：门开着不溢出、不吐乱码、架构生效。
`SGCV10Layer.hook` 里 β 是无量纲比例，满足 T1' 契约 `‖Δ‖≤βε‖x‖`。
但 24 层各自该取多少 β_l，从未拟合。全局焊死 β=1 有三宗罪：

1. 不同层 teacher residual 尺度不同，统一 β 不可能是逐层最优；
2. 某些层分支可能有害，最优解本就是 `β_l = 0` —— 该层特性是
   「登记可用、实际未启用」，必须写进账本，不许偷换成「全层继承生效」；
3. 没有 teacher 同条件对照，就没有「不回归」的可信区间。

对应台账 `TS2SC_V10_FALSIFIABLE_LEDGER.md` 的 **P-β / P-quality**。

## 协议（七步，缺一不可）

  S1  冻结 teacher：219 COPY 张量逐位锁定，任何一步不得改动 teacher 权重。
  S2  同条件采集：同一批 prompt、同一 seed、同一数据顺序，
      分别跑 `teacher_only` 与 `v10_branch(β_l=grid)` 的 forward，
      记录每层 hidden（pre/post hook）与最终 logits。
  S3  逐层目标：`hidden_cos(l)` 与 `logit_KL`，**分层**而不是只看最终输出。
  S4  网格：每层 `β_l ∈ {0, 0.05, 0.1, ..., 1.0}`（21 档），
      24 层 → 504 次受控 forward；坐标下降两轮，禁止联合暴力搜索。
  S5  通用能力回放 ≥30%：校准集里混入通用语料，防止 β 只拟合身份/短问句
      （这正是上游 v11「2× 对手步数对照」与「崩坏区伪影」教训的落地）。
  S6  锁定：产出 `beta_l.json`（24 个值，允许含 0），
      每个值附 `hidden_cos / KL / 回放占比 / seed / 数据 sha256`。
      β_l=0 的层必须显式登记为 `REGISTERED_NOT_ACTIVE`。
  S7  L3 配对验收：可读率 / mojibake 率 / 配对 Wilson 95% CI；
      下界 ≤ 0 即**不通过**，回退 `beta_l ≡ 0`（逐位零回归臂），如实留档。

## 六道闸门（任一不过即中止，不产出半成品）

  G1  teacher 219 COPY sha256 与 `student_full_manifest_v10.json` 全一致
  G2  branch 312 NEW 重开逐字节复验通过
  G3  β_l=0 时与 teacher **逐 token 相同**（端到端，不只是单元级 Δ=0）
  G4  β_l>0 时全 finite、解码守卫通过、零越界 id
  G5  回放占比 ≥0.30，且回放集上无显著退化（配对 CI）
  G6  L3 配对 CI 下界 > 0，否则回退零回归臂并登记负结果

## 明确不做 / 不声称

  * 不跑 ppl 领先、不写 SOTA、不写「原生级 / 能力更强」
  * 不把 parity 当超越；上游 `COMPLETE_BEYOND=False` 随附
  * 不在纯 CPU 上做输出层微调（已知几乎只有污染、没有收益）
  * 不伪造任何未测数字；本骨架 `NOT_EXECUTED`
  * 不碰 8M 端到端（hybrid 保留 teacher 全注意力，不可行结论不变）

## 用法（目标机）

    python calibrate_beta_v10.py --plan     # 只打印协议 + 闸门，零计算（默认）
    python calibrate_beta_v10.py --execute # 需 torch + GPU + teacher 权重；
                                          # 本沙盒不满足，直接 REFUSE

"""
from __future__ import annotations

import json
import os
import sys

PROTOCOL = dict(
    schema="TS2SC/v10-beta-calibration-protocol",
    status="NOT_EXECUTED / PLAN-ONLY / NO_CHECKPOINT / NO_NUMBERS",
    targets=["P-beta", "P-quality"],
    ledger="TS2SC_V10_FALSIFIABLE_LEDGER.md",
    steps=["S1 freeze teacher COPY",
           "S2 paired same-condition forward",
           "S3 per-layer hidden_cos + logit_KL",
           "S4 beta_l grid {0,0.05,...,1.0}, 2-round coordinate descent",
           "S5 >=30% general-capability replay",
           "S6 lock beta_l.json with per-layer provenance",
           "S7 L3 paired Wilson CI"],
    gates=["G1 219 COPY sha256", "G2 312 NEW byte recheck",
           "G3 beta=0 tokenwise identical", "G4 finite/decode-gate",
           "G5 replay>=0.30", "G6 paired CI lower bound > 0"],
    fallback="beta_l == 0 for all layers (bit-identical teacher arm), "
             "negative result registered, never hidden",
    forbidden_claims=["SOTA", "strictly better than Transformer/Mamba2",
                      "native-level", "capability improved",
                      "8M end-to-end verified", "all SimplCore features reproduced"],
    upstream_boundary="SimplCore bec5ab8fd7 v9-v11: COMPLETE_BEYOND=False; "
                      "2x opponent-step control failed; A2 negative deltas are "
                      "collapse-region artifacts",
    requirements=dict(torch=True, gpu=True, teacher_weights=True,
                      paired_eval_set=True, l3_wilson_ci=True),
    note="This file is a protocol, not a result. Running --execute in this "
         "sandbox is refused by design.")


def plan():
    print(json.dumps(PROTOCOL, indent=2, ensure_ascii=False))
    print("\n[plan-only] 0 forwards run, 0 numbers produced, 0 checkpoints "
          "written. Execute only on the maintainer's GPU target machine.")


def execute():
    # 显式拒绝：本沙盒无 torch / GPU，且不伪造结果。
    print("REFUSE: this sandbox has no torch/GPU/teacher checkpoint and the "
          "protocol forbids fabricating calibration numbers.", file=sys.stderr)
    print("Run on the target machine with real teacher weights; see S1–S7 "
          "and G1–G6 above. This negative gate is itself registered in the "
          "falsifiable ledger.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    if "--execute" in sys.argv:
        raise SystemExit(execute())
    plan()
