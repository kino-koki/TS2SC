#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.3 · SimplCore 全特性注册表

"继承 SimplCore 所有特性" 必须先变成一份可核对的清单，否则就是口号。
每条特性都带：权威出处（本地真实文件）、V7 生产代码里的真实状态、
v8.3 的落点、以及验证类型。

验证类型（非常重要，防止把理论前提包装成 LM 结论）：
  RUNNABLE  -- 本轮在真实 MiniCPM5-1B 权重上用纯 numpy 真跑出断言结果
  REGISTERED-- 论文理论前提 / 工程契约，结构化继承并声明 scope；
               明确 NOT 宣称在真实 LM 上被复现

出处文件均为沙盒内已存在的真实文件，不引用任何未发布资源。
"""
from __future__ import annotations

# (id, 名称, 出处, V7真实状态, v8.3落点, 类型, 断言)
FEATURES = [
    ("F01", "静态 LTI Toeplitz 核 K(d)=ρ^d, 0<ρ<1",
     "SimplCore_完整稿.md §4.1 / Theorem 7; README_V7_PROD.md §1",
     "V7 step(): S = rho*S + upd, rho=0.90 常量",
     "SGCFullBranch.state_step, rho 来自 config 且恒定",
     "RUNNABLE", "K(d)=ρ^d 下三角 Toeplitz；ρ 与 t、与输入无关"),
    ("F02", "FFT–IIR 对偶（Toeplitz 稠密 ≡ irfft(x̂K̂) ≡ 递推）",
     "SimplCore_完整稿.md §4.1; final_arch_verify.json C1",
     "V7 只实现 IIR 递推，未实现对偶断言",
     "verify_fft_iir_duality()",
     "RUNNABLE", "三种形式相对误差 < 1e-8"),
    ("F03", "严格因果",
     "TS2SC_VL47_PREFLIGHT.md §6; MIGRATION_TIMING_ACCEPTANCE.md §2.3",
     "V7 递推天然因果，但未做扰动断言",
     "verify_causality()",
     "RUNNABLE", "扰动未来 token，过去步输出 bit 不变"),
    ("F04", "order-2 分槽 outer-product 状态 S=ρS+k⊗v",
     "SimplCore_完整稿.md §3.2; README_V7_PROD.md §1",
     "V7 step(): upd = k.unsqueeze(-1)*v.unsqueeze(-2)",
     "SGCFullBranch.state_step, n_slots 由真实 head 布局推导",
     "RUNNABLE", "递推 S 与显式 Σρ^(t-τ)k_τ⊗v_τ 数值一致"),
    ("F05", "O(1) 状态，与序列长 L 无关",
     "README_V7_PROD.md §1; final_arch_verify.json C4/C5",
     "V7 init_state 固定 (B,R,ds,ds)",
     "verify_state_independent_of_L()",
     "RUNNABLE", "L=1/4/16/64 状态字节恒定"),
    ("F06", "无 N×N score、无随 L 增长的 KV cache",
     "README_V7_PROD.md §1; SimplCore_完整稿.md §2.3",
     "V7 SGC 分支无 softmax / 无 KV cache",
     "SGCFullBranch（结构性：状态形状不含 L 维）",
     "RUNNABLE", "状态张量 shape 中不出现 L"),
    ("F07", "有界因果深度可分离卷积 + 逐点门 (Wc K=3, Wg)",
     "README_V7_PROD.md §1; modeling_simplcore_v7.py L112-117",
     "V7 已实现 Wc/Wg，但零初始化",
     "SGCFullBranch.local_path, hist ring",
     "RUNNABLE", "K=3 有界；hist 长度恒 K-1；逐点门作用于 local"),
    ("F08", "输入侧 role bit (k mod R)",
     "SimplCore_完整稿.md §2.1/§3.3/P3; README_V7_PROD.md §1 声明",
     "**真实缺口**：README 声明，modeling_simplcore_v7.py step() 未显式实现",
     "SGCFullBranch.role_bit() + verify_role_bit_degeneracy()",
     "RUNNABLE", "注入 R 维 role one-hot；B=0 时 per-fibre 读出退化到 chance"),
    ("F09", "输入侧信息预算 B ≥ log₂|A(v)|",
     "SimplCore_完整稿.md §3.3 Theorem 3 / R1",
     "V7 无对应实现",
     "feature registry 结构化继承 + counting-bound 登记",
     "REGISTERED", "理论计数下界；NOT 宣称在 MiniCPM5 上被复现"),
    ("F10", "校准写入门 wgate",
     "README_V7_PROD.md §1; modeling_simplcore_v7.py L99/L125",
     "V7 sigmoid(0)=0.5 零初始化，未校准",
     "SGCFullBranch.write_gate；标注 MUST calibrate at L2",
     "RUNNABLE", "写入门参与前向且梯度可达；初始值登记为未校准"),
    ("F11", "可学混合门 β（显式门控，非无门控自演化）",
     "SimplCore_完整稿.md §1 'self-evolution asserted without a gate'; v8.1 I2",
     "V7 alpha=-2 硬编码，mixer 结构性失效（R5）",
     "beta 初值 0 + enabled 显式开关",
     "RUNNABLE", "β=0 ⇒ 与 teacher 逐位相同；β>0 ⇒ 分支贡献非零"),
    ("F12", "全展开 forward == O(1) 状态 step（stream equivalence）",
     "README_V7_PROD.md §1; final_arch_verify.json C4/C5",
     "V7 check_stream_equivalence 已实现但未在真权重上跑",
     "verify_stream_equivalence()",
     "RUNNABLE", "逐 token step 与一次性递推输出相对误差 < 1e-8"),
    ("F13", "teacher SwiGLU / RMSNorm / o_proj / embed / lm_head 全保留",
     "README_V7_PROD.md §1; v8.1 融合等式",
     "V7 删掉 self_attn（替换），导致语义错乱",
     "219 张量全 COPY，SGC 仅并联",
     "RUNNABLE", "COPY 数 == teacher 张量数；UNCLASSIFIED == 0"),
    ("F14", "GQA 行映射禁零填充",
     "v8.1 R6 / modeling_v8.build_slot_map",
     "V7 Wk[:kv_dim]=k, 其余补零 → 一半 slot 恒零",
     "build_slot_map + DERIVE GQA_ROW_REMAP",
     "RUNNABLE", "zero_padded_rows == 0，每行映射到真实 teacher 行"),
    ("F15", "QK-norm 折叠（不 SKIP）",
     "v8.1 R8; migrate_qwen3.py L134-139",
     "V7 直接 SKIP q_norm/k_norm",
     "fold_qk_norm DERIVE",
     "REGISTERED", "MiniCPM5-1B 无 q_norm/k_norm 张量；折叠路径保留并声明 0 dropped"),
    ("F16", "tie_word_embeddings 继承 + lm_head 显式",
     "v8.1 R1 / I5",
     "V7 强制 tie=True 且丢弃 lm_head.weight",
     "lm_head_plan(tie=false) → COPY_STANDALONE",
     "RUNNABLE", "lm_head.weight 在 219 张量内且被 COPY"),
    ("F17", "H6 层接口 / A4 末层读出 / exactly-H 层",
     "SimplCore_完整稿.md §2.3 / §3.2; claim_registry H1-H7",
     "V7 无对应实现",
     "接口登记；明确是 conditional interface property",
     "REGISTERED", "条件性接口性质，NOT 架构无关定律，NOT 在 LM 上复现"),
    ("F18", "层间接口可学且非唯一（Theorem 5, 2V_K² 维仿射空间）",
     "SimplCore_完整稿.md §1 R4; 上轮修稿记录",
     "V7 无对应实现",
     "T_h 登记为可学；不焊死解析 value→key 映射",
     "REGISTERED", "理论结果；工程侧保留可学接口，不宣称复现维数"),
    ("F19", "门控巩固 monotone / 必要 / T·w·b bit 率界 / 配对 no-regression gate",
     "SimplCore_完整稿.md §4.2 Theorems 8-10; claim_registry SE1-SE5",
     "V7 无对应实现",
     "align_accept L3 配对 Wilson CI；Twb 作为预算非训练率",
     "REGISTERED", "有限巩固过程理论；本轮未跑巩固实验"),
    ("F20", "撤回登记 W1–W15 边界",
     "claim_registry.md; PAPER_TS2SC_ADDENDUM.md W14",
     "V7 失败登记在文档里但未做成门禁",
     "R1–R10 闭环表 + L0–L4 硬门禁",
     "REGISTERED", "边界声明；不把撤回内容写成结果"),
    ("F21", "每个可训参数在前向图拿到非零梯度",
     "train_simplcore.py 第一步断言; v8.1 R5",
     "V7 有断言脚本但零初始化使贡献≈0",
     "SGCFullBranch：Wc/Wg/wgate/alpha/beta 全部参与 hook",
     "RUNNABLE", "每个 NEW 张量在融合前向中被读取；结构连通性"),
    ("F22", "零硬编码规格（config-driven）",
     "v8.1 R7 / spec.py",
     "V7 硬编码 bos/eos/max_pos/R=2/self_attn.*",
     "TeacherSpec 从真实 config 推导 n_slots=6",
     "RUNNABLE", "n_slots = 1536/(2×128) = 6，无常量"),
    ("F23", "tokenizer↔embedding 绑定、UTF-8 流式、id 空间",
     "v8.1 R2/R3 / contract.py",
     "V7 best-effort + 子集 BPE",
     "static_contract + offline_tokenizer_check（真实文件已 PASS）",
     "RUNNABLE", "5/5 往返；max_id < vocab_size；chat template 存在"),
    ("F24", "L0–L4 分级验收 / mojibake 率 / 配对 Wilson CI",
     "align_accept.py; v8.1 §五",
     "V7 gen_quality 只测非退化（乱码会 PASS）",
     "GATES + mojibake_rate + wilson_ci",
     "REGISTERED", "门禁定义完整；L2/L3/L4 需 T+0 teacher 同条件对照"),
    # ---- v8.4: 乱码理论修复（本轮新增） ----
    ("F25", "无量纲尺度契约 ‖Δ‖₂ ≤ βε‖x‖₂",
     "scale_theory.py T1'; run_full_migration.py hook()",
     "v8.3 用 probe_gain 归一化 + clip 掩盖数值溢出（workaround）",
     "dimensionless_gate(x,o,β,ε=0.10)，clip 移除",
     "RUNNABLE", "1e-5~1e5 量级的 o 都满足比值 ≤0.1；β=0 逐位零"),
    ("F26", "尺度不变 role bit（跨尺度可分辨）",
     "scale_theory.py T2'; SimplCore_完整稿.md §3.3 Theorem 3",
     "v8.3 naive ρ=‖e‖/‖k‖ 随输入尺度漂移（真实 r_k=0.324），"
     "名义 B=log₂R 时稳时不稳",
     "k̂=k/(r_k+δ); k'=k̂+snr·e (snr=0.50)",
     "RUNNABLE", "假设间距离 CV=0 跨 1e4× 尺度；naive 相对能量比衰减 10000×；B=0 距离 0"),
    ("F27", "finite-and-in-range 解码守卫",
     "scale_theory.py T3'; run_minicpm.py logits()",
     "V7 best_i 初始 -1 + NaN 比较静默失败 + gen_quality 漏检",
     "合法初始 id + last_logits_finite + decode_gate",
     "RUNNABLE", "拒绝 -1/NaN/空串；接受合法 id；根因可复现"),
    # ---- v8.6: 源码核实发现的【未继承】项（不是遗漏编号，是状态修正） ----
    ("F28", "无词表 / tokenizer-free 字节级主路径（ByteLevelCodec + EntropyPatcher + BytePatchBridge）",
     "simplcore_src/README.md L287 docs/BYTELEVEL.md; L302 §7 标 ✅『无词表模式（字符/字节/BPE 可插拔）』; "
     "L507 tokenization.py/tokenizer.py『无词表编解码器』; train/README_REMOTE_CURRENT.md L282/L328/L472",
     "V7 生产链路实际走 Qwen3 BPE 子集（bpe.py QwenBPESubset）；产品线实现未推送到本地",
     "**未实现**：当前迁移体 219 张量全 COPY，含 embed_tokens[130560,1536] / "
     "lm_head[130560,1536]，并用真实 tokenizer.json；contract.py I6 反向绑定词表",
     "NOT_INHERITED", "需要 TS2SC 双轨桥接（tokenizer 轨防乱码 + byte 轨无词表入口）；"
                      "本地缺 docs/BYTELEVEL.md 与 simplcore/core/model/tokenization.py，接口待推送对齐"),
    ("F29", "GlyphStream 连续符号接口（v2：删 nn.Embedding / lm_head / 分词器）",
     "simplcore_src/train/corpus.txt L816 起『无词表连续符号接口 (GlyphStream) + TriMemory』; "
     "L859 G1; L877 第三部分; L1045 P1『4–6 周』计划; L1969 提及 v2 重构方案文档",
     "文档明确标 v2.0-draft、P1 为计划；本地无 native_mm.py / GlyphEncoder / "
     "ContinuousTextHead 任何实现，MANIFEST.json 无相关条目",
     "**未实现**：这是 v2 新架构（连续向量输入 + 原型库解码 + 教师蒸馏），"
     "不是迁移能做的；与当前『融合不替换』红线冲突",
     "NOT_INHERITED", "OOV=0 / 删除 embed+lm_head / 困惑度差 ≤2% 均为方案验收项，"
                      "今天一条都没跑；不能算已继承"),
]

RUNNABLE = [f for f in FEATURES if f[5] == "RUNNABLE"]
REGISTERED = [f for f in FEATURES if f[5] == "REGISTERED"]
NOT_INHERITED = [f for f in FEATURES if f[5] == "NOT_INHERITED"]


def summary():
    return dict(
        n_features=len(FEATURES),
        n_runnable=len(RUNNABLE),
        n_registered=len(REGISTERED),
        n_not_inherited=len(NOT_INHERITED),
        runnable_ids=[f[0] for f in RUNNABLE],
        registered_ids=[f[0] for f in REGISTERED],
        not_inherited_ids=[f[0] for f in NOT_INHERITED],
        complete_inheritance=False,
        real_gap_found=["F08 role bit（README 声明，V7 代码缺失）",
                        "F10 wgate 未校准", "F11 alpha=-2 硬编码",
                        "F14 GQA 零填充", "F15 QK-norm SKIP", "F16 lm_head 丢弃",
                        "v8.4 F25/F26/F27：乱码的两个理论根因 + 解码守卫",
                        "v8.6 F28/F29：无词表设计在源码文档中存在，但实现未推送、"
                        "当前迁移体未继承（方向相反）"],
        honesty=("【口径已修正】此前『27 项全特性继承』不成立："
                 "F28 无词表/tokenizer-free 与 F29 GlyphStream 是 README §7 标 ✅ 的"
                 "产品线特性与 v2 方案，但本地 167 个文件里没有对应实现模块，"
                 "当前 MiniCPM5-1B 迁移体反而绑定 130560 词表并全 COPY embed/lm_head。"
                 "REGISTERED 项是理论前提/契约；只有 RUNNABLE 项真跑过数值断言；"
                 "NOT_INHERITED 项明确未继承。三者都不构成 LM parity / ppl / SOTA 声明。"),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(dict(features=[dict(id=f[0], name=f[1], source=f[2],
                                         v7=f[3], v83=f[4], kind=f[5], assertion=f[6])
                                    for f in FEATURES],
                          summary=summary()), indent=2, ensure_ascii=False))
