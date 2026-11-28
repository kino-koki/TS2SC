# 原生 SimplCore · 逐层蒸馏协议（唯一可行路径）

目标：同时满足 **非 hybrid** + **严格保留原模型能力** + **原生 SimplCore**。
前提：承认这三条在不训练时不可同时成立（见 `NATIVE_SIMPLCORE_STATUS.md` §3），
因此本文件是**训练协议**，不是迁移脚本。

---

## 0. 状态

| 项 | 状态 |
|---|---|
| 纯 O(1) 无注意力主干 C1–C8 | ✅ CPU 沙盒真跑全绿 |
| 能力保留 | ❌ CPU 实测 `ABILITY_PRESERVED=False` |
| 本蒸馏训练 | ⏸ **NOT EXECUTED**：需 GPU + 真实语料，本沙盒无算力 |
| 伪造结果 | 禁止；无 checkpoint / 无分数 / 无生成样本 |

---

## 1. 监督信号

teacher = MiniCPM5-1B（真实权重，24 层 Llama）；student = `PureBackboneLM`。

对每个 batch、每层 `l ∈ [0,23]`：

```
h_l^T  = teacher 第 l 层输出（pre-lm_head，RMSNorm 后）
h_l^P  = pure 第 l 层输出
L_l    = 1 - cos(h_l^P, h_l^T)  +  λ_mse · || h_l^P/||h_l^P|| - h_l^T/||h_l^T|| ||²
```

末端：

```
L_logit = KL( softmax(lg_T / τ) || softmax(lg_P / τ) )      τ=2.0
L_arg   = 1 - 1[ argmax(lg_P) == argmax(lg_T) ]   （只作监控，不可微）
```

总损失 `L = Σ_l w_l·L_l + α·L_logit`，`w_l` 先均匀、后按梯度范数均衡。

**关键**：必须先让 `L_0..L_23` 收敛，再放开 `L_logit`。直接只训 logit KL
会重演 V7「权重搬对了、接口接错了」——输出头拟合了，中间表示没对齐。

---

## 2. 数据

- 真实语料 ≥ 30–50 M token（FineWeb-Edu / 自建中文语料混合），
  **不是** 6 个 prompt；本轮 6 prompt 只够证明「线性对齐不够」。
- teacher 与 student 用**同一 tokenizer / 同一 token id 序列**，
  避免词表错位（V7 乱码根因之一）。
- 序列长度阶梯：512 → 2K → 8K → 32K，每档独立验收。
- 保留 5% held-out，全程不进梯度。

---

## 3. 训练配置（建议起点，不是已验证超参）

| 项 | 值 |
|---|---|
| 优化器 | AdamW，β=(0.9,0.95)，wd=0.1 |
| LR | 3e-4 cosine，warmup 2% |
| 精度 | BF16 混合，teacher 冻结 |
| 并行 | 4–8 卡 ZeRO-2 / FSDP；teacher 只前向不存梯度 |
| 冻结项 | 谱核 `ρ, θ` 保持冻结（保住 Toeplitz/FFT-IIR 性质）；只训投影 `P/Q`、γ 门、谱卷积、FFN、embed/head |
| batch | 按显存放到最大，grad accum 到 ~0.5 M token/step |

> 若放开 `ρ, θ` 一起训，会破坏 C2「冻结静态核」性质，纯度判据需重定义。
> 默认不放开，先在冻结核下看能力上限；不够再单独做一轮并记录性质损失。

---

## 4. 验收硬门槛（写死，不许事后放宽）

| 判据 | 门槛 |
|---|---|
| held-out 逐层 hidden cos | ≥ 0.90（24 层全部） |
| held-out 生成 argmax 一致率 | ≥ 0.90 |
| 配对 Wilson 95% CI 下界 | > 0（相对 teacher 不劣） |
| 下游 benchmark | 不劣于 teacher（配对重跑 ≥10 次） |
| 乱码率 / mojibake | 0 |
| 纯度 C1–C8 | 训练后重新全跑，必须仍全绿 |
| 开关法则 | 删任一外挂仍可观测、可回滚、不退化回注意力 |
| 10M 常驻状态 | 仍 O(1)，且如实登记 ρ^L 长程衰减 |

任一项不过 ⇒ 判 `ABILITY_PRESERVED=False`，不发布「原生级 / 能力不劣」。

---

## 5. 与现有资产的关系

- `pure_backbone_v11.py`：student 定义，C1–C8 已绿，可直接接训练循环。
- `verify_pure_v11.py`：训练后**必须重跑**，确认纯度没被训练破坏。
- `align_pure_ability.py`：本轮 6-prompt 线性对齐探针；训练后换成
  全量 held-out 版本重跑，替换掉 `ABILITY_PRESERVED` 判决。
- `TS2SC_V10_FALSIFIABLE_LEDGER.md`：新增 `P-native` 命题，
  证伪条件 = 任一验收门槛不达标。
- 云端 `MiniCPM5-1B-TS2SC` 仓是 **hybrid 交付**，本轮成果**不混进该仓**
  冒充模型权重；只作为 TS2SC 源码仓的新架构分支与可证伪台账。

---

## 6. 一句话

纯主干已经造出来且验过纯度；**能力还没保住**。把上面这套蒸馏在 GPU 上跑完、
七道门槛全过，才配叫「严格保留原模型能力的原生 SimplCore 模型」。在那之前，
本轮结论就是 `False`，不包装。
