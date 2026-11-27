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

## v9.1 · 逐位置闭式微调（⚠️ EXPERIMENTAL / NOT ACTIVE）

`identity_full_sft.npz` 把引导扩到 **K=6**，每位置独立解
`A_k=(H_kᵀH_k+λI)⁻¹H_kᵀT_k`，`H_k` 来自 8 条样本（4 语言 × 2 措辞）的
逐位置 teacher-forced hidden。真跑了、也真解出了 6 个矩阵，
但**四轮配置全部未通过文本质量判据**（`passed=false`）：
覆盖式→复读机；纯加法→退回 `<think>`；单/近3 token 抑制→周期复读。

根因是 **exposure bias**：生成时 h_k 已落在训练 H_k 的支撑外，
模型只在一小撮被抬升的候选里打转。由此量出
**输出层纯引导安全工作区 ≈ 2 步**（即 v9.0 词级锚成立的原因）。

**判据修复**（比调参更值钱）：旧脚本只看连续同 token 游程，
把 `A/B/A/B` 周期判成 passed=true；现补
`token_diversity ≥ 0.55` 与 `max_bigram_repeat ≤ 2`，并回头重判 a/b/c。

低内存：单进程串行、`need_logits=False` 跳过 lm_head 全扫、
逐样本释放 KV、Gram 1536²≈18 MB、hidden 断点落盘可续跑。
`identity_full_hidden.json` 是 scheduled-sampling / DAgger 的续跑起点。

## v9.2 / v9.3 · 最终定案

- `identity_dagger_sft.py`：DAgger 2 轮 off-policy 增广重解 `A_k`；
  周期复读解决（diversity 0.90–1.00），但 v9.2c 机判 passed 时
  **语义重判仅 1/4 有效** —— `青奈科技是开发MiniCPM系列模型的公司`，
  关键词命中、语义颠倒。见 `identity_semantic_reaudit.json`。
- `identity_gradient_sft.py`：**真 next-token CE + Adam + best-snapshot**
  训出 `identity_head_delta.npz`（302×1536 = 1B 的 0.046%），
  推理期零编排；train full-vocab acc 0.931，生成端 `passed=false`
  （尺度不一致 → 全退回 `<think>`；尺度对齐后 → bigram 12 复读）。
- `lora_identity_sft.py`：跨过天花板的正确层级（LoRA 穿 24 层 +
  scheduled sampling + 30% 通用能力回放），**未执行**。

12 轮全表、两难证明、六道闸门见 `IDENTITY_FINAL_VERDICT.md`。

## 边界

**当前 ACTIVE 仍是 `identity_alignment.npz`（v9.0，2 步）**；
其余 `identity_*sft.npz` / `identity_head_delta.npz` 均为 EXPERIMENTAL，
不要加载当成品。不改 219 个 teacher 张量；不改 β_l 校准状态；
v9.0–v9.3c 全部只在 pre-lm_head → logits 一层，**不是梯度反传穿 24 层**；
`IDENTITY_SFT.md` 的全句方案与 LoRA SFT 仍未执行。
「上下文长度理论无限」是品牌陈述非实测；无能力/ppl/SOTA 声明。
