# 身份认同微调 · CPU 闭式 rank-1 adapter

不是 system prompt，不是 few-shot，不是每轮拼接模板。产出的是独立权重文件
`identity_alignment.npz`，属于 TS2SC 微调系统的核心资产。

## 三个真组件

1. **身份方向 `u`**：12 语言身份问句与对照问句各取 pre-lm_head hidden，
   `u = mean(ĥ_id) − mean(ĥ_non)`。一个方向覆盖所有语言，换语言不露馅。
2. **条件门 `g(x)=σ(κ(u·ĥ)−b)`**：`b=(min p_id + max p_non)/2`，`κ=4/margin`。
   身份问句 g→1，算术/翻译/闲聊 g→0 ⇒ 不会每次复述那段话。
3. **输出层方向 `v`**：`v = normalize(Σ_{i∈S} W_lmhead[i,:])`（rank-1 闭式解，
   只读 |S|=16 行，不扫 130560 行）。`w_i = max(W_i·v̂,0)/max_j(W_j·v̂)`。
   门开时把 `S` 中 token 的 logit 抬到 `max_logit + 6·w_i`，
   **其余 130544 个 logit 逐位不变**（v8.7 曾把 Δ 注入 final RMSNorm 之前，
   被归一化搅乱语义，已弃用并记录）。
4. **词级锚**：从 FACTS 关键词「青奈」的 tokenizer 编码取前 2 个 id，
   只引导前 2 个生成 token；第 3 步起完全自由续写 ⇒ 不复读整段模板。

## 验收

- 12 语言身份问句门全开（`g ≥ 0.6221`），8 条非身份门全关（`g ≤ 0.3779`，抬升 0）
- 逐语言 logit margin 硬证据见 `identity_margin_check.json`
- 多语言短生成互不重复（不是固定模板）、v9.0 实测 4/4 身份 probe 关键词命中「青奈」
- 非身份问句不泄漏身份关键词、门关时逐位零回归
- logits 全 finite、0 越界 id、decode gate 全过
- **语义边界**：词级锚之后由基座自由续写，实测四句里出现「是一种哺乳动物」
  「爱 爱 爱 爱」「私は デス」等不连贯续写；不声称已能完整准确复述 FACTS

## 边界

不改 219 个 teacher 张量；不改 β_l 校准状态；**是闭式输出层 adapter，
不是梯度 SFT / 全量微调**；只引导前 2 token，不保证整段身份介绍逐字复述。
要稳定说出完整身份需 `IDENTITY_SFT.md` 的 teacher-forced rank-r 微调（下一轮，未跑）。
「上下文长度理论无限」是品牌陈述非实测；无能力/ppl/SOTA 声明。
