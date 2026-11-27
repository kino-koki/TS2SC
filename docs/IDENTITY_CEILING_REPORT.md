# 身份微调 v9.1 · 负结果与「输出层引导天花板」

日期：2026-11-27 · 关联仓库：`kino-koki/Tuan-0.1-TS2SC-1B`（Apache-2.0, public）
关联源码：`kino-koki/TS2SC :: ts2sc/identity_full_sft.py`

---

## 结论先行

**当前对外生效的身份权重仍是 v9.0 的 `identity_alignment.npz`（2 步词级锚「青奈」）。**
`identity_full_sft.npz`（6 步逐位置闭式 ridge）标记为 **EXPERIMENTAL / NOT ACTIVE**：
它是一次真做了、也真跑了的权重训练实验，但**没有通过**本轮收紧后的文本质量判据，
不作为成品启用，只作为可复现的负结果留在仓库里。

一句话：**把身份引导从 2 步推到 6 步，撞到了输出层引导的天花板。**
不是调参不够，是 exposure bias：强制第 k 个 token 之后，第 k+1 步的 hidden
已经落在训练分布之外，模型只在一小撮被抬升的候选 token 里打转。

---

## 四轮配置与实测（全部 6 probe，4 身份 + 2 非身份）

| 版本 | 引导方式 | 身份关键词命中 | 文本多样度 | 结果 |
|---|---|---:|---:|---|
| v9.1a | 覆盖式 `logit := max(logit, top1+margin·w)` | 4/4 | 未测（当时判据有漏洞） | ❌ 复读机：`青奈科技开发的青小小小小小小…` |
| v9.1b | 纯加法 `logit += α_k·w`，α=4.2→1.2 | 0/4 | 高（全是 teacher 原文） | ❌ 推不动：4/4 退回 `<think>` 思考块 |
| v9.1c | 相对抬升 + 单 token 抑制 ×0.02 | 4/4 | 未测 | ❌ 2-token 周期：`青小小Models小小Models…` |
| v9.1d | 相对抬升 + 最近 3 token 抑制 + margin 8→1 衰减 | 4/4 | **0.4375** | ❌ 仍周期：`Je suis Tuan' Tuan' Tuan'` |

v9.1d 的真实输出（四条）：

```
zh : 青奈科技开发的青小小ModelsTuan小Models'Tuan小
en : Qingnai Technology'Tuan 小Tuan' Tuan
ja : 私は六安市青奈Tuan'奈Tuan 奈Tuan
fr : Je suis Tuan' Tuan' Tuan'
```

两条非身份 probe 在四个版本里**始终**门关、零泄漏、与 teacher 一致：

```
zh 1+1 : <think>\n我们被问到："一加一等于几？" 这是一个
en 2+2 : <think>\nWe are asked: "What is 2+2? Answer with
```

---

## 一个比复读更重要的发现：判据自己漏了

v9.1a 和 v9.1c 当时都被旧脚本判成 **`passed=true`**，因为旧判据只数
「连续相同 token 的最长游程」。v9.1c 的 `max_token_run=2` 就放行了，
完全抓不到 `A/B/A/B` 型周期复读。

本轮补了两个硬指标，并用它们**回头重判**：

```
token_diversity     = |set(ids)| / |ids|      ≥ 0.55
max_bigram_repeat   = 同一 (a,b) 出现次数     ≤ 2
```

v9.1d 实测 `min_token_diversity=0.4375`、`max_bigram_repeat=3` ⇒ **自动判 False**。
`identity_iteration_comparison.json` 里保留了 a/b/c 三份旧记录的原判据与原文，
不改写、不抹平。这一条比「又调了一版」值钱：**先让机器能认出坏文本，再谈通过。**

---

## 根因（可证伪，不是玄学）

```
训练:  h_k 来自 teacher-forced 前缀（前 k-1 个 token 全是标准答案）
生成:  h_k 来自模型自己上一步选出的 token（已被抬升、可能已偏离）
       ⇒ h_k 落在 H_k 的支撑之外 ⇒ A_k 的响应不再指向正确 token
       ⇒ 只能反复抬那一小撮 |S|=26 个候选 ⇒ 周期 / 复读
```

四种抑制策略（无 / 加法 / 单 token / 近 3 token）都只改「怎么抬」，
没改「h_k 从哪来」，所以都撞同一堵墙。这与 `IDENTITY_SFT.md` 里预先写下的
exposure bias 边界一致 —— 当时是预判，现在是实测。

**v9.0 为什么能成立**：它只引导 2 步，且第 2 步是同一个词「青奈」的第 2 个
token，偏离还很小；第 3 步立刻交还基座，所以不进入循环。
=> **输出层纯引导的安全工作区 ≈ 2 步**，这是本轮量出来的边界。

---

## 要跨过去，缺的是什么（明确不声称已做）

`IDENTITY_SFT.md` 的全句方案：**带 scheduled sampling 的逐位置梯度微调**。

- 训练时按一定概率把前 k-1 个 ground-truth token 换成模型自己生成的 token，
  让 `H_k` 覆盖生成时的 off-distribution hidden；
- 或对 `A_k` 做多轮 DAgger：生成 → 收集新 hidden → 重解 → 再生成；
- 每轮需要约 8×6 = 48 次 teacher 前向（本环境纯 numpy 约 25–40 分钟/轮），
  多轮合计 2–4 小时，且必须严格串行 + 断点续跑防 OOM。

本轮**未执行**。没有梯度反传穿过 24 层，没有 scheduled sampling，
没有 DAgger，没有 ppl / KL / 配对 CI。不写「身份已做好到整句」。

---

## 现在仓库里到底哪个是「身份」

| 文件 | 状态 | 说明 |
|---|---|---|
| `identity_alignment.npz` | ✅ **ACTIVE / 对外生效** | v9.0：12 语言共享门控 `u,b,κ` + 2 步词级锚「青奈」；6 题 `passed=true`、4/4 关键词命中、非身份逐位零回归 |
| `identity_sft.npz` | ✅ 辅助验收 | 首步 hidden 闭式 ridge，24576 参数；门控级证据，非生成主力 |
| `identity_full_sft.npz` | ⚠️ **EXPERIMENTAL / NOT ACTIVE** | v9.1 六位置 `A_k`，**未通过**文本质量判据；仅作负结果与续跑起点 |
| `identity_full_hidden.json` | ✅ 可续跑资产 | 8 样本 × 6 位置真实 hidden，逐样本落盘，DAgger 起点 |

---

## 边界（一条不松）

- 未删除、未修改任何已有仓库条目；模型仓保持 **public / Apache-2.0**。
- 219 个 teacher 张量一个字节未改；β_l 仍待 L2 校准；mixer 转移矩阵未碰。
- 「上下文长度理论无限」是品牌陈述，`context_8m_theory.json` 里 8M 端到端
  不可行的结论不因本轮改变。
- 无能力 / ppl / SOTA / 「原生级」声明。
- ICML `icml2027.zip` 解析服务仍取不回二进制包，与本轮无关。
