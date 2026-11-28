# Transformer → SimplCore 算子级替换 v12 · 真迁移验证

2026-11-28 21:32:40 · 1839.4 s

## 做了什么

96 个 self_attn 张量从推理链**摘除**，替换为冻结旋转谱核 LTI mixer（A_m=ρ_m R(θ_m) 常矩阵，冲激响应 ρ^τ sin((τ+1)θ)，O(1) 状态、Toeplitz/FFT↔IIR 对偶）；123 个非注意力 teacher 张量（embed / RMSNorm / 24 层 FFN / lm_head）逐位继承。没有 y=T(x)+旁挂，也不是随机初始化重建。

## 为什么换判据

主判据是**函数级对齐**（hidden cos / 首 token argmax / top-5 / 跨题 cos），不是短生成答案词：MiniCPM5 是思考型基座，6–12 token 全落在 <think> 块里，teacher 与 v12 都会得 0，判分器没有判别力。短生成只作「会说话、不乱码」辅助证据，如实登记两边得分。

## 判决

- **THREE_CONSTRAINTS_MET = `False`**
- hidden cos 均值 **0.039122**（最低 -0.041143，门槛 ≥0.90 / ≥0.80）
- 首 token argmax 一致 **0/6**（门槛 ≥5/6），top-5 平均重叠 **0.0**
- 跨题 cos 最低：teacher 0.208791 / v12 0.125428 ⇒ 前缀坍缩 `False`
- 纯度：运行期加载 self_attn **0** 个；继承 123 个逐位 sha256 对账，mismatch=0
- O(1)：10M idle 0.001682 s，状态 4731072 → 4731072 B；hybrid 残差 T(x) = `False`
- 短生成答题（仅登记）：teacher 1/6 / v12 0/6，是否有判别力 `True`

## 逐题（真实解码）

| # | 题 | T argmax | V argmax | 一致 | hidden cos | top5∩ | teacher 文本 | v12 文本 |
|---|---|---|---|---|---|---|---|---|
| 1 | `中国的首都是` | 8 | 730 | ❌ | 0.063028 | 0 | `<think>\nThinking Process:\n\n` | `ely#<在<` |
| 2 | `一加一等于几？` | 8 | 47739 | ❌ | 0.106595 | 0 | `<think>\n我们被问到` | `essespace` |
| 3 | `太阳从哪个方向升起？` | 8 | 18749 | ❌ | 0.102119 | 0 | `<think>\n嗯，用户问的是` | ` mature referring.s###` |
| 4 | `请把「你好」翻译成英文` | 8 | 64891 | ❌ | 0.000764 | 0 | `<think>\n嗯，用户需要将` | `隐约挑 Wrong###` |
| 5 | `What is 2+2?` | 8 | 4828 | ❌ | -0.041143 | 0 | `<think>\nWe are asked:` | `限< /<` |
| 6 | `The capital of` | 8 | 8273 | ❌ | 0.003372 | 0 | `<think>\nThinking Process:\n\n` | `酸,弹ppaond low` |

## 边界

- 6 题小样本是**函数级对齐探针**，不是全分布能力证明；无 ppl / benchmark / 配对 Wilson CI，不写「原生级 / 全分布能力不劣」。
- mixer 用解析 T1' 尺度契约（β=0.1）、冻结 ρ/θ，未做全量梯度蒸馏；DISTILL_PROTOCOL.md 仍是达到全分布保留的路径。
- ρ^10M 下溢如实登记；O(1) 常驻状态 ≠ 长程谱响应保留。
- judge 的答题分数只登记不进门槛；判分器先过 3 阳 2 阴对照。
- 本轮不推模型仓、不覆盖 hybrid 交付；只在源码仓登记新架构。
