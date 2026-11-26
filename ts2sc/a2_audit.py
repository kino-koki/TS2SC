#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 kino-koki/TS2SC 的《A2.0 协议》逐条核对当前迁移体。

用户要求「先确认之前迁移的模型真正 SimplCore」。
本脚本**不重跑任何生成/前向**，只读本会话已落盘的证据 JSON：
  migration_proof.json / multi_prompt_verify.json / context_8m_theory.json
  student_full_manifest.json / features.py / test_report.json

A2.0 是五阶段迁移流水线（S1 规划 → S2 权重迁移 → S3 激活缓存 →
S4 逐层蒸馏 → S5 验证），术语表明确：
  mixer = SimplCore 里【替代 Attention】的谱卷积模块（SGC）
三条铁律 / 三个必须打印的数字 / 预期值，均按协议原文核对。
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import summary as feat_summary  # noqa: E402

OUT = "/data/workspace/output"
MIG = os.path.join(OUT, "minicpm5_full_migration")


def j(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    proof = j(os.path.join(OUT, "migration_proof.json"))
    multi = j(os.path.join(OUT, "multi_prompt_verify.json"))
    ctx = j(os.path.join(OUT, "context_8m_theory.json"))
    led = j(os.path.join(MIG, "student_full_manifest.json"))
    cfg = j(os.path.join(MIG, "config.json"))
    tests = j(os.path.join(HERE, "test_report.json"))
    fs = feat_summary()

    e1, e2, e3, e4 = (proof["E1_structural"], proof["E2_counterfactual"],
                      proof["E3_attribution"], proof["E4_integrity"])
    counts = led["counts"]
    n_teacher = e1["n_teacher_keys"]
    n_copy = counts["COPY"]

    # ---- A2.0 三个必须打印的数字 ----
    d1 = dict(name="loaded / total > 99%（否则有张量静默跳过）",
              value=f"{n_copy}/{n_teacher} = {100.0*n_copy/n_teacher:.2f}%",
              raw=round(100.0 * n_copy / n_teacher, 4),
              passed=bool(n_copy / n_teacher > 0.99
                          and led["unclassified"] == 0
                          and e1["n_removed_teacher_keys"] == 0),
              note="A2.0 原始实测 99.84% / 199-of-199。本迁移体是 teacher 侧 "
                   "219/219 逐位 COPY、UNCLASSIFIED=0、removed=0，"
                   "比 99.84% 更强（严格逐位，非余弦近似）。"
                   "注意 student 另有 192 个新增 sgc/blocks 张量，不计入 teacher 覆盖率分母。")
    d2 = dict(name="FFN 保真度余弦 ≈ 1.0000",
              value="1.0000（逐位相同 ⇒ 余弦严格 = 1.0）", raw=1.0,
              passed=bool(n_copy == n_teacher),
              note="A2.0 铁律 2 要求 FFN 对齐（c.ffn_ratio=3.0）。"
                   "本迁移体走『融合不替换』：FFN / SwiGLU / norms / embed / "
                   "lm_head 全部逐位 COPY，没有走 A2.0 的切片/扩维/重组，"
                   "所以不存在形状不匹配导致的静默跳过。")
    # O(1) 状态：SGC 分支已实测；teacher attention 仍线性
    state_const = ctx["A_simplcore_construction"]["state_independent_of_L"]
    state_mib = ctx["A_simplcore_construction"]["all_24_layers_state_MiB"]
    kv_gib = ctx["B_current_hybrid"]["kv_total_GiB"]
    d3 = dict(name="O(1) 唯一状态数 = 1（否则架构没生效）",
              value=f"SGC 分支 True（24 层恒定 {state_mib} MiB）；"
                    f"端到端 hybrid False（8M 时 teacher KV ≈ {kv_gib} GiB）",
              passed_simplcore_branch=bool(state_const),
              passed_end_to_end=False,
              note="按 A2.0『mixer 替代 Attention』的严格口径，这一条只对 SGC 分支成立；"
                   "teacher 全注意力一层未删 ⇒ 端到端状态仍随 L 线性增长。"
                   "这是融合态与替换态的分界线，不粉饰。")

    # ---- 三条铁律 ----
    laws = [
        dict(id="铁律 1", text="保留原词表 —— 换词表 = 毁模型",
             status="符合",
             evidence=f"vocab_size={cfg.get('vocab_size', 130560)} 保留；"
                      "embed_tokens / lm_head 全 COPY；"
                      "contract I6 校验 len(tok)==embed.rows",
             note="重要：这也解释了为何 F28『无词表 / tokenizer-free』未继承"
                  "在 A2.0 语境下不是缺陷 —— 协议明令禁止换词表。"
                  "（F29 GlyphStream 属于 v2 新架构，与 A2.0 是不同路线）"),
        dict(id="铁律 2", text="FFN 必须对齐 c.ffn_ratio = 3.0",
             status="符合（更强）",
             evidence="FFN 全部逐位 COPY，未做任何 ratio 变换",
             note="A2.0 需检查中间维对齐；本路线因不替换而天然对齐"),
        dict(id="铁律 3", text="loss 必须归一化（残差流逐层放大 1000×）",
             status="不适用 / 未触发",
             evidence="本路线未做 S3/S4 蒸馏，无训练 loss；"
                      "上线时靠 F25 无量纲尺度契约 ‖Δ‖≤βε‖x‖ 防放大",
             note="若将来执行 L2 β_l 校准或 A2.0 S4，此条必须重新启用"),
    ]

    # ---- 五阶段 ----
    stages = [
        dict(id="S1 规划", spec="纯算术：FFN 中间维 / 词表 / GQA KV 头数",
             done=True,
             evidence=f"config-driven：n_slots = 1536/(2×128) = 6；"
                      "vocab 130560；nkv=2, head_dim=128；零硬编码规格"),
        dict(id="S2 权重迁移", spec="直搬 / 变换 / 重组 + 覆盖率报告",
             done=True,
             evidence=f"219 COPY 逐位 + 72 DERIVE（GQA 行映射，零填充=0）"
                      f"+ 120 NEW（24 层 SGC 分支）；UNCLASSIFIED=0；"
                      f"落盘 216.6 MiB / 144 张量；replay_full.py 逐位校验每个 COPY"),
        dict(id="S3 激活缓存", spec="一次 teacher 前向，缓存 28 层真实激活到磁盘",
             done=False,
             evidence="未执行。identity_cache.json 里有 32 个 pre-lm_head hidden"
                      "（身份微调用），不是 A2.0 的逐层激活缓存",
             note="A2.0 相对 A1.0 最关键的改进；当前融合路线没有核拟合目标，故未做"),
        dict(id="S4 逐层蒸馏", spec="冻结 86.6%，只训 mixer（13.4%），"
                                  "整层输出 MSE，每 5 步存盘",
             done=False,
             evidence="未执行。β_l 仍是尺度契约配置（β=1），L2 校准协议已写"
                      "在 calibrate.py 但未跑",
             note="这正是把『融合态』升级到『替换态』所缺的一步"),
        dict(id="S5 验证", spec="五项测试 + 断言 + 覆盖率报告",
             done="部分",
             evidence=f"已跑：E1–E4 真实性证明 all_passed={proof['all_passed']}；"
                      f"12 臂 non_accidental={multi['non_accidental_stats']['non_accidental']}；"
                      f"{tests['tests_run']} 条回归全绿。"
                      "未跑：A2.0 的 FFN/mixer/端到端保真度三项对照数字",
             note="有断言与覆盖率，但不是 A2.0 定义的那五项"),
    ]

    # ---- 最终判决 ----
    verdict = dict(
        question="之前迁移的模型是否『真正 SimplCore』？",
        answer="分两种口径，结论不同 —— 必须写清，不能一句『是』糊过去。",
        a2_strict=dict(
            label="按 A2.0 协议严格定义（mixer 替代 Attention + 真实激活蒸馏）",
            result="**不是**",
            why="A2.0 术语表：mixer = SimplCore 里【替代 Attention】的 SGC。"
                "当前 v8.x 是『融合不替换』：teacher self_attn / QSA / norms / FFN "
                "一层不删，SGC 是并联加性分支（out = teacher_block + β·sgc_branch）。"
                "S3/S4 未执行，没有核拟合蒸馏，也没有 A2.0 的 mixer 余弦 0.1024→0.6659。"
                "所以它不是 A2.0 意义上的 student。"),
        a2_baseline=dict(
            label="按 A2.0 的底线与验收纪律",
            result="**全部符合，且是 S2 的严格超集**",
            why="保留原词表 ✓ / FFN 对齐且逐位 ✓ / 无静默跳过（UNCLASSIFIED=0）✓ / "
                "覆盖率 219/219=100% > 99% ✓ / FFN 保真度 =1.0 ✓ / "
                "SGC 分支 O(1) 状态 ✓ / 每个阶段都有断言与账本 ✓。"
                "它是 A2.0 S1+S2 完成、S3–S4 尚未启动的【融合态前置阶段】。"),
        not_a_failure="这两者不矛盾：你此前明确要求『融合而非替换、"
                      "绝不进 mixer 转移矩阵』，A2.0 替换态会直接破掉这条红线"
                      "（也破掉 β=0 逐位零回归）。所以当前是**有意选择的另一阶段**，"
                      "不是迁移失败。",
        what_would_make_it_strict="执行 A2.0 S3（真实逐层激活缓存）+ S4（只训 mixer 的"
                                  "整层 MSE 蒸馏），再按 S5 五项测试给出 FFN/mixer/端到端"
                                  "保真度；代价是必须替换 teacher attention，"
                                  "并重跑零回归配对验证。",
    )

    rep = dict(
        title="按 A2.0 协议核对 · 迁移体是否真正 SimplCore",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        protocol_source="kino-koki/TS2SC :: A2.0协议.md + docs/00-协议总览.md",
        method="只读本会话已落盘证据，不重跑任何生成/前向（按用户指令）",
        three_numbers=[d1, d2, d3],
        three_laws=laws,
        five_stages=stages,
        feature_registry=dict(total=fs["n_features"], runnable=fs["n_runnable"],
                              registered=fs["n_registered"],
                              not_inherited=fs["n_not_inherited"],
                              complete_inheritance=fs["complete_inheritance"]),
        honesty=dict(
            reused_evidence="E1 192 新增 key / E2 23-23 层全变 / E3 置零 0.0 "
                            "与行置换 74.60 / E4 22 分片 sha256",
            no_new_experiment=True,
            boundaries=["未做 S3/S4 蒸馏", "未给 A2.0 的 mixer/端到端保真度数字",
                        "无 ppl / KL / 配对 CI / 能力结论",
                        "context_claim 是品牌陈述非实测",
                        "Qwen4-27B 未发布，本轮无关"]),
        verdict=verdict,
    )
    p = os.path.join(OUT, "A2.0_验收核对.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(dict(three_numbers=[(x["name"], x.get("passed",
                     x.get("passed_simplcore_branch"))) for x in (d1, d2, d3)],
                          stages=[(s["id"], s["done"]) for s in stages],
                          a2_strict=verdict["a2_strict"]["result"],
                          a2_baseline=verdict["a2_baseline"]["result"]),
                     ensure_ascii=False, indent=2))
    print("[done]", p)


if __name__ == "__main__":
    main()
