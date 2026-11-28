# TS2SC v10.3 · 范围收窄迁移 + 10M 上下文 + 验证

日期：2026-11-28 18:13:27 · `passed = True` · 耗时 580.2 s

> 指令：**「迁移并且验证模型」** → **「就继承 SGC 以及其外挂模块
> SMM记忆与自进化模块就行了，上下文开到10M」**

## 1. 范围（只继承这三组）

| 组 | 内容 | 来源 |
|---|---|---|
| **G1 SGC_CORE** | 冻结静态核 `ρ^τ·sin((τ+1)θ)`、FFT↔IIR、O(1) 定态、严格因果、输入侧 role bit、低秩 readout | `sgc_v10.FrozenSGCCore` |
| **G2 SIDECARS** | A 3层 k=7 门控短卷积 / B CfC 液态门 / C Hadamard SE1 内容寻址，P7 不回写主干 | 上游 `sgc_sidecar_final.py` v9–v11 |
| **G3 SMM** | `S_t=ρ_mem·S_{t-1}+Σ C[k]ᵀC[VK+v]`、`add_t=C[bucket(tok)]S_tW_c`、零阶巩固 `observe()` | **上游真实源码 `evolution_mixer.py` 的 `FusedEvolutionMixer`/`CalibratedFusedMixer`** |

**SMM 不是编造的**：源码 docstring 明写
`STEP 2 -- SMM x SGC fused state, written by a zero-order gate`，
STEP 3 `CalibratedFusedMixer` 带 `rho_mem` 分离时间尺度 / per-key
cooldown / dominance gate；`observe()` 是 CT7/SE1-SE5 零阶巩固，
要求 τ 次重复证据、读回不命中才写、capacity 淘汰，无梯度穿过检索。
本轮把它移植为推理期逐 token 的 64 维低秩分支（上游是 `(B,L,d)`
训练期 mixer），**形状不同但语义与门禁逐条对齐**。

历史 `capability_registry.json` 的 9 CONTRACT / 2 DEFERRED 等条目按
「不删除任何条目」原样留在仓里，本次交付**排除**在继承声明之外。

## 2. 开关法则（沿用你的硬要求）

```
y = T(x) + 𝟙[master] · ( Δ_SGC+SIDECAR(x) + s_smm · Δ_SMM(x) )
全关        ⇒ y ≡ T(x) 逐位（结构性 early-return）
SMM 关      ⇒ 与 v10.2 全开**逐位相同** ⇒ 之前 4 语言 10 臂证据仍有效
全开        ⇒ 三组都继承
```

**SMM 是显式门禁，不是自发自进化**：`S` 初值全零，未调用 `observe()`
时贡献严格为零。本轮巩固用的是结构化强共现演示数据
（`tgt=(tok+5)%16`），**不是训练结果、不是模型自己学会了记忆**。

## 3. 迁移与验证

### 3.1 真迁移（LIVE 重拉，非复用旧产物）

| 判据 | 结果 |
|---|---|
| 22 分片 LIVE git blob sha1 | ✅ `True` |
| 账本 | `579` 张量 `{'COPY': 219, 'DERIVE': 0, 'NEW': 360}` / UNCLASSIFIED `0` |
| **219/219 COPY 全量逐位** | ✅ mismatch `0`，2.01 GiB |
| 312 v10.2 NEW 重开复验 | ✅ mismatch `0` |
| branch 重建 == 已推送 | ✅ `True`（确定性可复现） |
| 新增 SMM checkpoint | `48` 张量 / 772.2 KiB |
| phase1 passed | **True** |

### 3.2 端到端生成（真实权重 + 24 层 hook）

| 语言 / 臂 | gate | 首句 |
|---|---|---|
| `zh` / `T` | `True` | `<think>

</think>

你好！有什么我可以` |
| `zh` / `OFF` | `True` | `<think>

</think>

你好！有什么我可以` |
| `zh` / `ON_ALL` | `True` | `<think>
嗯，用户发送的是一条简单的` |
| `en` / `ON_ALL` | `True` | `<think>
The user's message is just` |

- **全关 ≡ 纯 teacher 逐 token 相同**：`True`
- **全开与全关不同（三组真进前向）**：`True`
- 全 finite `True` / 全解码守卫 `True`
  / 身份标记泄漏 **0**
- SMM 显式巩固：每层 commits `16`、
  rejects `1278`、slots `16`、
  ‖S‖ `3.7672`、24/24 层 `True`

> 8 token 短生成只证 finite / 不乱码 / 零回归 / 架构生效；
> **不是**质量、事实正确性或能力结论。

## 4. 上下文开到 10M

`max_position_embeddings: 131072 → **10,485,760**`
（10 Mi tokens；十进制 10,000,000 同登记）。

| 侧 | 结论 |
|---|---|
| **分支侧** | `advance_idle(10M)` **O(1) 闭式实跑**（不是循环 10M 次）；状态 `3,675,648 B` 跳跃前后恒定 `True`；闭式 == 逐步 gate=0 `True`；SMM 绑定 10M 后仍可读 |
| **teacher 侧** | 全注意力 KV **240.0 GiB**、13.74 量级对数、`80.0×` RoPE 外推 ⇒ **`INFEASIBLE_LEDGER`**（HBM 账本判定，非实跑失效） |

> **配置开到 10M + 分支可承载，不等于端到端 10M 已验证。**
> hybrid 保留 teacher 全注意力，240 GiB KV 不可行；80× 位置外推未验证。
> 这是账本结论，不包装成「10M 已跑通」。

## 5. 本轮自查抓到并修掉的两个真问题

**① SMM 差点又成空挂。** 第一次冒烟全开 `‖Δ‖` 与 `smm_off` 完全相同——
`S` 初值全零、随机散共现被 τ/dominance/cooldown 门禁全拒（commits=0）。
这与 v10.1 抓到的「72/312 空挂」同构。改用结构化强共现后
commits=16、rejects≈1278、slots=16、读回 `key_value=5` 与
`tgt=(tok+5)%16` 对齐 ⇒ 记忆**真的写入且可读**，不是空挂。

**② 两项相加突破 T1' 总契约。** SGC 与 SMM 各自归一化到 `βε‖x‖`，
直接相加可到 `2βε‖x‖`。现对**总和**做统一上界投影：实测全开
ratio=1.0；`smm_off` 时总和就是 d_sgc、本就在界内、零缩放 ⇒
与 v10.2 全开逐位相同，证据链不破。

## 6. 边界（不删减）

- SMM 来源为上游真实源码 evolution_mixer.py 的 FusedEvolutionMixer / CalibratedFusedMixer（SMM x SGC 融合状态），移植为推理期逐 token 的 64 维低秩分支；非编造。
- SMM 的 observe() 是**显式门禁**：S 初值全零，未调用 observe 时贡献严格为零（与 v10.2 全开逐位相同）。本轮巩固用的是结构化强共现演示数据，不是训练结果、不是模型自发记忆。
- 自查抓到并修复：SGC 与 SMM 各自归一化到 βε‖x‖，直接相加会突破 T1' 总契约；现对总和统一投影，smm_off 时零缩放保持逐位相同。
- 10M 是**配置值 + 分支侧 O(1) 闭式实跑**；teacher 侧全注意力KV 240 GiB、80× RoPE 外推未验证 ⇒ 端到端 10M 仍 INFEASIBLE_LEDGER，不是实跑失效。
- 8 token 短生成只证 finite / 不乱码 / 全关零回归 / 三组继承生效 / 零身份标记；不是质量、事实正确性或能力结论。
- β=1 仍是尺度契约配置，不是 L2 校准成品；P-β 仍 NOT_TESTED。
- 无 ppl / KL / 配对 CI / 吞吐 / SOTA / 原生级 / 定理在 LM 复现声明。
- 本次交付范围排除 9 CONTRACT + 2 DEFERRED 等历史条目，但按用户「不删除任何条目」原样保留在仓里，只是不纳入继承声明。
- 端到端生成是「真实权重 + 24 层融合 hook」运行时挂接，不是对已组装 2.6 GB 静态大文件的直接加载（磁盘不足）。

权威注册表：`scoped_capability_registry.json`；配置：`context_10m_config.json` / `inference_default_v10_3.json`；原始证据：`v10_3_scoped_verify.json`、`student_scoped_manifest_v10_3.json`。
