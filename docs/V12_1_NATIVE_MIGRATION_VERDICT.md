# v12.1 解析算子迁移 · 拟合 + 端到端验证（同进程真跑）

2026-11-28 22:13:58 · 2293.1 s

## 做了什么

96 个 self_attn 张量从推理链摘除；123 个非注意力 teacher 张量逐位继承；attention 子层的 I/O 用闭式 ridge **解析迁移**成静态读出投影 W_l（只在 readout，冻结谱核 A/B/C 不回写）。无 QKV / SDPA / KV cache / softmax-over-position / T(x) 残差主干。

## 判决

- **THREE_CONSTRAINTS_MET = `False`**
- **OPERATOR_MIGRATION_EFFECTIVE = `False`**
- hidden cos 均值 **0.115028**（最低 0.005243，held-out 0.076355）
- 首 token argmax 一致 **1/8**，top-5 平均重叠 **0.25**
- v12 随机 readout 基线：cos 0.039122 / argmax 0/6 / top5 0.0
- 纯度：加载 self_attn **0** 个，继承 123 个逐位对账 mismatch=0；hybrid 残差 T(x) = `False`
- O(1)：10M idle 0.001215 s，状态 4731072 → 4731072 B

## 逐题

| # | 角色 | 题 | T argmax | V argmax | 一致 | hidden cos | top5∩ | teacher 文本 | v12.1 文本 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fit | `中国的首都是` | 8 | 220 | ❌ | 0.102338 | 0 | `<think>\nThinking Process:\n` | `\n1hem作高 Alps` |
| 2 | fit | `一加一等于几？` | 8 | 41 | ❌ | 0.06166 | 0 | `<think>\n我们被问到` | `4 friendly1impl下目录` |
| 3 | fit | `太阳从哪个方向升起？` | 8 | 4246 | ❌ | 0.005243 | 1 | `<think>\n嗯，用户问的是` | `的一浅# manualursruit` |
| 4 | fit | `请把「你好」翻译成英文` | 8 | 8 | ✅ | 0.344801 | 1 | `<think>\n嗯，用户需要将` | `<think>ECBsworthcf次**` |
| 5 | fit | `What is 2+2?` | 8 | 34 | ❌ | 0.144734 | 0 | `<think>\nWe are asked:` | `-豆a and has œ` |
| 6 | fit | `The capital of` | 8 | 107206 | ❌ | 0.126919 | 0 | `<think>\nThinking Process:\n` | `calculation2界iper across __` |
| 7 | fit | `Translate "tha` | 8 | 4838 | ❌ | 0.058176 | 0 | `<think>\nHmm, the user` | ` actually本本iontr` |
| 8 | heldout | `日本の首都は？` | 8 | 220 | ❌ | 0.076355 | 0 | `<think>\n用户询问日本的` | `\nscrOf usual在.n` |

## 边界

- 拟合是 teacher-forced（用 teacher 的 h_l），本报告给出的是端到端自回归实测指标，不拿训练 cos 冒充保留能力。
- 8 题小样本探针；无 ppl / benchmark / 配对 Wilson CI，不写「全分布能力不劣 / 原生级」。
- ρ^10M 下溢如实登记；O(1) 常驻 ≠ 长程谱响应保留。
- 若 THREE_CONSTRAINTS_MET=False 但 OPERATOR_MIGRATION_EFFECTIVE=True，表示解析迁移确实大幅缩小差距，但仍需 GPU 蒸馏补齐全分布保留。
- 本轮不推模型仓、不覆盖 hybrid 交付。
