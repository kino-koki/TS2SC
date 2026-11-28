# 真迁移 SimplCore · 四轮阶梯与终局判定

日期 2026-11-28 · 本轮结论：**不训练，「非 hybrid + 保留原能力 + 原生 SimplCore」不可同时成立**，已用四轮真跑把死因钉死

---

## 0. 用户约束

> 真正使其与 Transformer 真正迁移 SimplCore，并且保留原能力、SimplCore 特性，不走 hybrid。

三条硬约束：`① 非 hybrid`、`② 保留原模型能力`、`③ SimplCore 原生特性`。
不再允许用 `y = T(x) + 旁挂` 交差。

---

## 1. 四轮阶梯（每轮都是真跑，负结果不包装）

| 版本 | 做法 | 非 hybrid | 逐位继承 | 能力实测 | 判决 |
|---|---|---|---|---|---|
| **v10.3** | `y=T(x)+ΣΔ_f`，β=1 | ❌ T(x) 是 Llama 全注意力 | 219/219 | 会说话、中英日法可读 | 被用户核验打回：**HYBRID** |
| **v11** | 219 张量全丢、纯 O(1) 随机重建 | ✅ C1–C8 全绿 | ❌ 0 | cos −0.0138、argmax 0/6、前缀坍缩 cos 0.999999 | `ABILITY_PRESERVED=False` |
| **v12** | **算子级替换**：96 个 `self_attn` 摘除，换冻结旋转谱核 LTI mixer；123 个非注意力张量逐位继承；无 `T(x)` 残差 | ✅ | ✅ 123/123 sha256 0 mismatch | teacher 6 题 argmax 全 8；v12 全错，cos **0.0391**、argmax **0/6**、top5 重叠 **0**、文本乱码 | `THREE_CONSTRAINTS_MET=False` |
| **v12.1** | 在 v12 上加 **解析算子迁移**：teacher 每层 attention 的 `(h_l, a_l)` + 谱核读出 `u_l` → 闭式 ridge 求静态读出投影 `W_l`，并解析出每层注入幅度 `γ_l`（不是拍脑袋 β） | ✅ | ✅ 123/123 | **训练 cos 0.999668**，端到端 cos **0.1150**、held-out **0.0764**、argmax **1/8**、top5 均值 **0.25**、仍是乱码 | `OPERATOR_MIGRATION_EFFECTIVE=False` / `THREE_CONSTRAINTS_MET=False` |

v12/v12.1 的**架构是真的**：运行期加载 `self_attn` 张量 **0 个**，123 个继承张量逐位 sha256 对账 `mismatch=[]`，`hybrid_residual_T=False`，谱核 A/B/C 冻结 ⇒ 静态核 / Toeplitz / FFT↔IIR 对偶 / O(1) 常驻状态全部保留，`advance_idle(10M)` 闭式实跑。

**但能力没保住。** 而且四轮把问题从「没做」收窄到「为什么做不到」。

---

## 2. 最值钱的发现：train cos≈1，端到端 cos=0.115，差在哪

v12.1 的逐层拟合在 **teacher-forced** 分布上几乎是恒等映射：

```
layer 00 n=114 train_cos=0.9959
layer 06 n=114 train_cos=0.9999
layer 12/18/23 train_cos=1.0000
全 24 层均值 0.999668
```

但一旦换成 student 自己的残差流做端到端自回归，cos 掉到 **0.115**。
`V12_1_DRIFT_DIAGNOSIS.json` 用已落盘的 `W_l` 真算出机制：

| 量 | 实测 |
|---|---|
| `W_l` 谱范数 | min **2.87** / mean **20.04** / max **53.23** |
| `γ_l` 解析注入比 | mean **0.4195**（原固定 β=0.1 系统性低估，已修） |
| 输入加 1e-3 噪声 → 读出相对变化 | mean **39.2×** / max **58.1×** |
| 24 层累积（每层相对误差 ε=0.10） | `(1+ε)^24 = 9.85×`，cos 上界 **0.102** |
| **端到端实测 cos** | **0.115** |

理论放大上界 0.102 与实测 0.115 几乎吻合 ⇒ **不是 λ/γ/候选集没调好，是逐层误差经 24 个高增益读出被指数放大**。这和此前身份微调撞到的 exposure bias 同构，但这次是在算子替换层面，且给出了数值闭环。

---

## 3. 结构性结论（可证伪，不是托辞）

teacher 的 attention 是**路径依赖**的：`a_l(t) = W_o·SDPA(Q,K,V; KV_{<t})`，状态是 KV cache。
SimplCore 谱核也是**路径依赖**的：`s_l(t) = A_m s_l(t−1) + B_m h_l(t)`，状态是 2×8 维 LTI。

两者状态空间不同构。第 0 层一旦分叉：

1. teacher 后续所有层的 `h_l / a_l` 都对应 **teacher 自己的轨迹**；
2. student 的 `h_l` 落在 **另一个分布**，`W_l` 是在 teacher 分布上拟合的；
3. 高增益 `W_l`（谱范数最高 53）把微小漂移放大，24 层累积 ⇒ cos 0.115；
4. 想做 DAgger / on-policy 修正，需要「teacher 在 student 的 `h_l` 下应给出什么 `a_l`」——
   这个配对**不存在**，因为 teacher 的 KV 历史与 student 状态无法对齐。硬凑就等于把
   teacher 路径塞回推理链，又变回 hybrid。

所以：**「96 个注意力张量逐位摘除 + 123 个非注意力张量逐位 COPY + 不重新求参数」与「保留原能力」不可同时成立**。这不是 CPU 算力问题，是函数类与监督信号的结构问题。四轮正好把它证伪干净。

---

## 4. 唯一能跨过去的条件

`DISTILL_PROTOCOL.md` 已写，补一条本轮新增的硬前提：

- 必须**在 student 自己的函数类里重新求参数**：scheduled sampling / DAgger 式的
  **on-policy 训练**，让 `W_l` 见到的就是 student 轨迹上的 `h_l`；
- 监督不能再用「teacher 同位置 attention 输出」（路径已分叉），改用
  **序列级目标**：teacher 最终 logits 的 KL + 逐层 hidden cos（容忍轨迹不同但语义对齐）
  + 生成级配对 CI；
- 冻结 `A_m/B_m/C_m` 保住静态核性质，只训 `W_l / γ_l / FFN` 的 LoRA；
- 数据 ≥ 30–50 M token、多卡数天、七道验收门槛（held-out cos ≥0.90、
  argmax 一致率 ≥0.90、Wilson CI 下界 >0、训练后纯度 C1–C8 重跑仍全绿）。

**CPU 沙盒无此算力：`NOT_EXECUTED`，无 checkpoint、无分数、不画饼。**

---

## 5. 本轮交付了什么

- ✅ 真·算子级替换架构 `operator_substitution_v12.py`（非 hybrid、123 逐位继承、O(1)、10M 闭式）
- ✅ 真·解析算子迁移 `fit_operator_v12_1.py` + `v12_1_operator_fit.npz`（28.1 MiB，24×192×1536）
- ✅ 8 题（7 拟合 / 1 held-out）端到端真跑判决、v12 随机基线对照、逐层 train cos
- ✅ 漂移放大诊断（谱范数 / γ / 累积放大上界 vs 实测 cos）
- ✅ 四轮阶梯 + 结构性不可行论证 + 唯一路径
- ❌ 不交付「已保留原模型能力的原生模型」——`THREE_CONSTRAINTS_MET=False`
- ❌ 不推模型仓、不覆盖 hybrid 交付、不伪造 benchmark

---

附：本轮用户再次贴入 `icml2027.zip`，解析服务仍取不回二进制包（与本次判定无关）；
此前给出的兼容层与三步替换说明仍在。
