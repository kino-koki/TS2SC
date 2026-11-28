# 严格保留原模型能力 · 可证伪判决（非 Hybrid / 纯主干）

日期 2026-11-28 20:10:54 · 耗时 29.4 s

## 判决

**ABILITY_PRESERVED = False**

硬性门槛：held-out argmax 一致率 ≥ 0.90 且 hidden cos ≥ 0.90。

| prompt | teacher argmax | pure argmax | 一致 | 原始 hidden cos |
|---|---|---|---|---|
| `你好，请介绍一下自己` | 8 | 18738 | 0 | -0.033376 |
| `中国的首都是` | 8 | 18738 | 0 | 0.008991 |
| `The capital of France is` | 8 | 18738 | 0 | -0.002086 |
| `What is two plus two?` | 8 | 18738 | 0 | -0.011529 |
| `请把这句话翻译成英文：今天天气很好` | 8 | 18738 | 0 | -0.028568 |
| `List three cities in Chi` | 8 | 18738 | 0 | -0.016231 |

- 原始 argmax 一致率 **0.0**（0/6）
- 原始 hidden cos 均值 **-0.0138**
- ridge 训练 cos 均值 **0.775929** / held-out cos **0.761171**
- pure 不同前缀 argmax 只有 **1** 种，hidden 方向 cos 最低 **0.999999** ⇒ 前缀信息坍缩 **True**

纯主干是随机初始化的 SimplCore 函数类，与 teacher 的注意力+FFN 不是同一个函数；线性 ridge 只能在训练 prompt 上插值，held-out 泛化崩掉。更硬的机制证据：6 个不同前缀下 pure 的 argmax 只有 1 种、hidden 方向 cos 最低 0.999999 ⇒ 前缀信息几乎不进最终表示，FFN 增益相对 embed 行过小。权重里内化的语言能力无法靠 COPY 之外壳迁移，必须逐层蒸馏训练。

## 边界

- 纯主干 C1–C8 已全绿（无注意力、O(1)、10M 常驻、删外挂不退化），但**纯度 ≠ 能力**：随机初始化不继承 teacher 的语言知识。
- ridge 对齐是闭式线性上界探针，不是训练；held-out 才是能力判据。
- 未跑 ppl / benchmark / 生成质量主观评估；argmax 一致率与 hidden cos 是能力保留的必要代理，不是充分证明。
- 6 个 prompt 的 teacher argmax 均为 id=8（模板结尾同 token），故 argmax 一致率判别力有限；主判据是 hidden cos 与 held-out 泛化。
- 「能力保留」需逐层蒸馏：teacher 24 层 hidden 作监督，纯 SGC 层在真实语料上用 GPU 训练 + 配对 Wilson CI 验收；CPU 沙盒无此算力，本轮不伪造结果。
- 结论若为 False，按用户约定**不交付冒充品**，只给差距与路径。
