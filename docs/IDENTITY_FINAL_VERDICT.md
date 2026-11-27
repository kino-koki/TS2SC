# 身份认同微调 · 最终定案（v9.0 → v9.3c，12 轮实测）

日期：2026-11-27 · 仓库：`kino-koki/Tuan-0.1-TS2SC-1B`（Apache-2.0, public）
源码：`kino-koki/TS2SC :: ts2sc/`

---

## 结论先行

**跨过去了一半，而且是把「另一半为什么跨不过去」用实验钉死了。**

✅ **已跨过**（成品，ACTIVE）：
1. 身份不再是提示词/字符串模板 —— 是**权重**：`identity_alignment.npz`（v9.0）
   与 `identity_head_delta.npz`（v9.3，真 CE+Adam 梯度训练）；
2. 12 语言共享门控、开→继承 / 关→不继承、非身份**逐位零回归**；
3. 零乱码、零复读、零基座身份泄漏（v9.0 的 6 题 `passed=true`）；
4. **推理期零编排**的训练方法本身成立（v9.3 的受限对比 CE + Adam，
   train full-vocab acc 0.931）。

❌ **未跨过**：让模型在**零解码期干预**下，连贯地、多语言地、从头到尾
说出整段 FACTS 自我介绍。12 轮里**没有任何一版**通过收紧后的全部判据。

---

## 12 轮全表（每轮都是真跑，不是推演）

| 版本 | 方法 | 关键词 | 多样度 | 语义 | 机判 | 主要失败 |
|---|---|---:|---:|---:|---|---|
| v9.0 ACTIVE | 12 语言门控 + 2 步词级锚 | 4/4 | 达标 | 连贯但怪 | ✅ | 只引导 2 步，不保证整段 |
| v9.1a | 6 步覆盖式相对抬升 | 4/4 | 未测 | — | ⚠️旧判据 | `青奈科技开发的青小小小小…` |
| v9.1b | 6 步纯加法 | 0/4 | 高 | — | ❌ | 全退回 `` |
| v9.1c | 相对 + 单 token 抑制 | 4/4 | 未测 | — | ⚠️旧判据 | `青小小Models小小Models…` |
| v9.1d | 相对 + 近 3 token 抑制 | 4/4 | 0.4375 | — | ❌ | `Je suis Tuan' Tuan' Tuan'` |
| v9.2a | **DAgger** 2 轮 + k1 relative | 0/4 | 0.90–1.00 | 0/4 | ❌ | `青藏高原` / `私はMiniCPM` |
| v9.2b | DAgger + 2 步语言前缀锚 | 1/4 | 0.81–1.00 | 0/4 | ❌ | `Qingming` / `Je Soyis` |
| v9.2c | DAgger + 锚 + 基座身份负抑制 | 3/4 | 0.81–1.00 | **1/4** | ⚠️True | **语义颠倒**：`青奈科技是开发MiniCPM系列模型的公司` |
| v9.3a | 真梯度 CE，全 130560 词表 | — | — | — | ❌ | 固定 logits 稀释梯度，acc 0.646 振荡 |
| v9.3b | 真梯度 **受限对比 CE** + Adam | 0/4 | 1.00 | 0/4 | ❌ | 训练/推理 hidden 尺度不一致，全退回 `` |
| v9.3c | 同上 + **尺度对齐 Δ·h_raw** | 1/4 | 0.1875 | 1/4 | ❌ | 复读回来：bigram 12、run 8 |

> 两次「判据自己漏了」也修了并回头重判：
> v9.1a/c 的旧判据只看连续同 token 游程，抓不到 A/B/A/B 周期；
> v9.2c 机判 `passed=true` 但语义重判只有 1/4 有效
> （`identity_semantic_reaudit.json`）。
> **先让机器能认出坏文本，再谈通过** —— 这条比任何一次调参都值钱。

---

## 两难（这是本轮最重要的产出，可证伪）

设生成第 k 步的 pre-lm_head hidden 为 `h_k`，输出层可训残差为 `Δ`。

**A. 要区分位置**（第 1 步说「青」、第 2 步「奈」、第 3 步「科技」…）
⇒ 需要 `A_k` 或按 k 切换的候选集 / margin
⇒ 必须在**解码期**知道 k 并切换策略
⇒ 不是纯权重，是编排（v9.1 / v9.2 全部在此）

**B. 要纯权重、零编排**
⇒ 只能用统一 `Δ`，让 `h_k` 自己携带位置信息
⇒ teacher-forced 训练时成立（train acc 0.931）
⇒ 生成时 `h_k` 偏离训练支撑（**exposure bias**）
⇒ 同一个 `Δ` 在所有 k 反复抬同一批 token ⇒ 复读（v9.3c 实测）

DAgger 缓解但**不能消除** B：它把 off-policy hidden 加进训练集，
但「青奈科技 / Tuan-Models / 六安市」这些词的**语义关系写在 24 层
FFN/attention 里**，不在 lm_head。只改 `|C|=302` 行输出向量，
最多改变「在特定 hidden 上吐哪个 token」，不能让模型**相信**新身份、
也不能让它把新身份组织成连贯多步语句。

> **形式化结论**：输出层（无论闭式 ridge、DAgger 重解、还是真梯度 CE）
> 的信息容量不足以承载「多步连贯身份」。这不是 lr / epoch / 候选集
> 的调参问题，是**参数所在层级**的问题。

---

## 已跨过的三个硬成果（别被负结果埋掉）

1. **零回归**：219 个 teacher 张量逐位 COPY，β=0 / g=0 时逐位相同，
   12 轮里 12 次非身份 probe 全部零泄漏、与 teacher 一致。
2. **权重级而非提示词级**：`identity_head_delta.npz` 由
   **标准 next-token CE + Adam** 训练（受限对比目标），
   463872 参数 = 1B 的 0.046%，落盘可加载；
   推理期只有一个学到的线性残差 + 冻结的学到的门控，
   **没有位置切换 / 锚集合 / margin 硬抬 / recent 抑制 / 负惩罚**。
3. **度量体系**：`token_diversity ≥ 0.55`、`max_bigram_repeat ≤ 2`、
   `base_identity_leaks = 0`、`semantic_inversions = 0`、
   `think_prefix = 0`、非身份零回归 —— 六道闸门，
   不再让「关键词命中」冒充「身份成立」。

---

## 真正跨过去需要什么（精确、可执行）

**必须让梯度穿过 transformer 层**，而不是只停在 lm_head。
最小充分方案：**LoRA SFT**，用本轮已产出的资产直接开跑。

| 项 | 值 |
|---|---|
| 基座 | `MiniCPM5-1B` 融合态（219 COPY + 24 层 SGC 分支） |
| 可训 | `q_proj,v_proj` LoRA r=16, α=32，约 0.1% 参数 |
| 数据 | `IDENTITY_SFT.md` 的 12 语言 × 多措辞身份问答 |
|  | + `identity_full_hidden.json`（8 样本 × 6 位置） |
|  | + `identity_dagger_r1/r2.json`（off-policy 轨迹，2 轮） |
|  | + 等量的通用能力回放语料（防灾难性遗忘） |
| 目标 | teacher-forced 全句 CE，**带 scheduled sampling** |
| 硬件 | 你此前规划的 4 卡并行即可；1B BF16 + LoRA 远低于 50h |
| 验收 | 本文六道闸门 + 与 teacher 的配对 Wilson CI + ppl 不回归 |
| 关键 | scheduled sampling / DAgger-on-weights：训练时按比例喂模型自己生成的前缀，让 24 层内部状态在生成路径上自洽 |

脚本骨架见 `ts2sc/lora_identity_sft.py`（PyTorch/PEFT）。
**本环境无 torch / 无 GPU，未执行、未验证** —— 如实标注，不画饼。
在 4 卡上它是标准 SFT，不需要新研究，只需要算力时间。

---

## 现在仓库里哪个是「身份」

| 文件 | 状态 |
|---|---|
| `identity_alignment.npz` | ✅ **ACTIVE / 对外生效**：v9.0，2 步词级锚，6 题 passed、4/4 关键词、12 语言门控、非身份逐位零回归 |
| `identity_sft.npz` | ✅ 辅助：首步 hidden 闭式 ridge，24576 参数 |
| `identity_head_delta.npz` | ✅ **权重级训练方法**（v9.3c）：真 CE+Adam + 尺度对齐；**方法成立，生成结果未达标**，EXPERIMENTAL |
| `identity_full_sft.npz` / `identity_dagger_sft.npz` | ⚠️ EXPERIMENTAL / NOT ACTIVE，负结果 |
| `identity_full_hidden.json` / `identity_dagger_r*.json` | ✅ LoRA SFT 的现成续跑起点 |
| 全部 `*_verify*.json` | ✅ 12 轮原始证据，不改写不抹平 |

---

## 边界（一条不松）

- 未删除、未修改任何已有仓库条目；模型仓保持 **public / Apache-2.0**。
- 219 个 teacher 张量一个字节未改；β_l 仍待 L2 校准；mixer 转移矩阵未碰。
- 「上下文长度理论无限」是品牌陈述；8M 端到端不可行的结论不变。
- 无能力 / ppl / SOTA / 「原生级」声明。
- ICML `icml2027.zip` 解析服务仍取不回二进制包，与本轮无关。
- 全程单进程串行 + OOM 守卫，最低 MemAvailable 约 1.1 GiB，**未 OOM**。
  训练阶段故意**不 preload 2.01 GiB teacher**（只读 lm_head 765 MiB），
  两阶段严格分离后才加载。
