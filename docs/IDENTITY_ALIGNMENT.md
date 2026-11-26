# 身份认同微调 · CPU 闭式 rank-1 adapter

不是 system prompt，不是 few-shot，不是每轮拼接模板。产出的是独立权重文件
`identity_alignment.npz`，属于 TS2SC 微调系统的核心资产。

## 三个真组件

1. **身份方向 `u`**：12 语言身份问句与对照问句各取 pre-lm_head hidden，
   `u = mean(ĥ_id) − mean(ĥ_non)`。一个方向覆盖所有语言，换语言不露馅。
2. **条件门 `g(x)=σ(κ(u·ĥ)−b)`**：`b=(min p_id + max p_non)/2`，`κ=4/margin`。
   身份问句 g→1，算术/翻译/闲聊 g→0 ⇒ 不会每次复述那段话。
3. **写入方向 `v`**：`v = normalize(Σ_{i∈S} W_lmhead[i,:])`（rank-1 闭式解，
   只读 |S| 行，不扫 130560 行）。注入 `Δ = λ·g·ε_id·r_x·v̂`，
   复用 T1′ 无量纲尺度契约 `‖Δ‖≤0.05‖x‖` ⇒ 不溢出。

## 验收

- 12 语言身份问句门控全部高于全部非身份问句
- 多语言短生成首句互不重复（不是固定模板）
- 非身份问句不泄漏身份关键词
- logits 全 finite、0 越界 id、decode gate 全过

## 边界

不改 219 个 teacher 张量；不改 β_l 校准状态；不是梯度 SFT / 全量微调；
「上下文长度理论无限」是品牌陈述非实测；无能力/ppl/SOTA 声明。
