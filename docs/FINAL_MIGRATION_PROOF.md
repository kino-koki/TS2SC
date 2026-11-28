# 终局证明 · MiniCPM5-1B-TS2SC 真的迁移成功且继承了 SGC 等特性

日期：2026-11-28 18:34:23 · **7/7 组判据全部通过** · `True`

> 用户指令：**「证明该模型真正的迁移成功，并且继承SGC等特性等等一切必须的证明」**

## 0. 「迁移成功」的准确定义（先立判据，不后补）

```
迁移成功 = 219 个 teacher 张量逐位保留
         + 360 个 NEW 张量真实接入前向（非空挂、非常量）
         + SGC / 外挂 / SMM 三组特性各自可独立归因
         + 全关时逐位回到原模型（开关法则）
         + 生成级 T == OFF 且 ON != OFF
```

**不等于**：能力变强、定理在 1B LM 上复现、自发自演化、
端到端 10M 已跑通、原生级体验。这五条本证明**一条都不声称**。

---

## P1 · 结构差异：student 不是原模型

| 项 | 值 |
|---|---|
| teacher 张量 | `219` |
| student 张量 | `579` |
| **新增** | **`360`**（312 个 SGC/外挂 + 48 个 SMM） |
| 删除 | `0`（全部 teacher 功能层保留） |
| DERIVE / UNCLASSIFIED | `0` / `0` |

原模型里根本不存在这 360 个 key。**仅这一条就排除了「直接拿原模型跑」**。

## P2 · 字节证据：不是声明，是逐张量复验

| 项 | 结果 |
|---|---|
| **360 个 NEW 重开逐字节 sha256** | `360` 个 / mismatch `0` |
| branch checkpoint sha == manifest | `True` |
| SMM checkpoint sha == manifest | `True` |
| **219 个 COPY LIVE 全量逐位**（2.01 GiB） | `True`，mismatch `0` |

teacher 权重**一个字节都没被改**；NEW 权重**一个字节都不是摆设**。

## P3 · 独立归因矩阵：三组特性真的在前向里

同 seed、同输入、**每臂一个全新零状态对象、只调一次 hook**
（避免有状态递推的历史污染；见下方自查修正）：

| 臂 | 数值 | 含义 |
|---|---|---|
| `ALL_OFF_vs_zero` | `0` | 全关 Δ 严格 0 ⇒ 原模型 |
| `ALL_ON_norm` | `3.87055` | 全开 Δ 非零 ⇒ 分支接进前向 |
| `G1_SGC_CORE_off_diff` | `0.274744` | 关 G1 必改变 Δ |
| `G2_SIDECARS_off_diff` | `0.0361502` | 关 G2 必改变 Δ |
| `G3_SMM_off_diff` | `0.218098` | 关 G3 必改变 Δ（记忆真在贡献） |
| `SMM_state_zero_diff` | `0.218098` | S 置零 ⇒ 与关 SMM 同向，非固定常量 |
| `bind.C_asymmetric_noise_diff` | `0.0189258` | 扰动 NEW 权重 ⇒ Δ 改变，排除空挂/常量 |
| `v10_2_full_on_bitwise_when_smm_off` | `0` | SMM 关 ⇒ 与 v10.2 全开逐位相同（同零状态） |

- **全关 Δ 严格 0** ⇒ 原模型能力逐位保留
- **关 G1 / G2 / G3 各自都改变 Δ** ⇒ 三组**独立可归因**
- **`bind.C` 非对称加噪扰动改变 Δ** ⇒ 排除「空挂 / 常量摆设」
- **24/24 层 Δ≠0**，T1' 总契约比值 `1.0000` ≤ 1
- **`smm_off` 与 v10.2 全开逐位相同**（diff = 0）⇒ 之前全部生成证据不作废

> **本轮自查抓到并修掉两个判据设计错误**（如实登记，不抹平）：
> ① 第一版各臂复用同一个有状态递推对象，导致「SMM 关 vs 全新 v10.2」
> 比的是不同状态历史，diff 0.368 被误判为失配 → 改为每臂全新零状态。
> ② `bind.C` 行置换被 Hadamard 对称性抵消，diff=0 是**判据失效**不是空挂
> → 改为非对称加性噪声（改方向而非改基排列），实测 diff 0.0189。

## P4 · 生成级：模型会说话，且不是原模型直接跑

来自 `v10_3_scoped_verify.json`（**LIVE 重拉 22 分片 + 真实权重 24 层前向**）：

| 判据 | 结果 |
|---|---|
| `T == OFF` **逐 token 相同** | `True` |
| `ON != OFF`（三组真进前向） | `True` |
| 中英文输出不同 | `True` |
| 全 logits finite / 全解码守卫 | `True` / `True` |
| 身份标记泄漏 | `0` |
| **SMM 真的被写入**（commits/层） | `16` |

实测首句：`你好！有什么我可以`（T/OFF）、
`嗯，用户发送的是一条简单的`（ON zh）、
`The user's message is just`（ON en）。

## P5 · 特性不变量：SGC 等「一切必须的证明」

| 特性 | 证据 |
|---|---|
| 冻结静态核 `ρ^τ·sin((τ+1)θ)` | `sgc_v10.FrozenSGCCore`，FFT↔IIR `1.69e-14 (fresh_migrate_v10.arch_checks)` |
| **O(1) 定态** | `3,675,648 B`，10M 跳跃前后恒定 `True` |
| P7 三条外挂不回写主干 | `True` |
| Hadamard SE1 零串扰 | `0.0 / 0.0 (arch_checks)` |
| T1' 尺度契约（总和统一投影） | `True` |
| **10M 闭式 idle == 逐步 gate=0** | `True` |
| 10M 判定 | `BRANCH_SIDE_VERIFIED_10M / TEACHER_SIDE_INFEASIBLE_LEDGER` |
| teacher 全注意力 | `INFEASIBLE_LEDGER`（240 GiB KV 账本判定） |

**SMM 来源可核对**：上游真实源码 `evolution_mixer.py` 的
`FusedEvolutionMixer`（docstring 明写 "SMM x SGC fused state"）与
`CalibratedFusedMixer`，零阶巩固 `observe()` 带 τ 证据 / dominance /
cooldown / 读回拒绝 / 容量淘汰 / `rho_mem`。

## P6 · 开关法则：开门禁继承，不开就原模型

来自 `v10_switch_law_audit.json`：

| 判据 | 结果 |
|---|---|
| 全关逐位零回归 | `True` |
| 13 门独立可差分（激活态） | `True` |
| 9 个契约门切换**不改前向** | `True` |
| 2 个 DEFERRED 置 True 被拒绝 | `True` |
| 全开 27/27 Δ 与 v10.0 逐位相同 | `True` |

## P7 · 确定性：不是偶然

| 判据 | 结果 |
|---|---|
| branch 重建 == 已推送版本 | `True` |
| 27/27 Δ 与 v10.0 baseline 逐位相同 | `True` |

---

## 结论

**迁移成功，已证明。** 7 组独立判据全部通过：

```
219 teacher 张量逐位保留  ·  360 NEW 张量逐字节复验且真实进前向
SGC / 外挂 / SMM 三组独立可归因  ·  24/24 层生效
全关逐位 ≡ 原模型  ·  生成级 T==OFF / ON!=OFF / 中英双语可读
SMM 零阶巩固真实写入（16 commits/层）  ·  10M 分支 O(1) 闭式实跑
```

## 边界（不删减）

- 「原模型直接跑」被 P1/P3 排除：student 有 360 个 teacher 里不存在的张量，且逐组关断/权重扰动都改变 Δ；全关时又逐位回到 teacher。
- β=0 / 全关是零回归对照臂，不是交付配置；交付 β=1.0。
- SMM 是显式 observe() 门禁，不是自发自演化；本轮巩固为演示数据。
- 10M 是配置 + 分支 O(1) 闭式实跑；teacher 全注意力 240 GiB KV ⇒ 端到端 INFEASIBLE_LEDGER，不声称 10M 已跑通。
- 8 token 短生成只证 finite / 不乱码 / 零回归 / 架构生效；不是质量、事实正确性或能力结论。
- 无 ppl / KL / 配对 Wilson CI / 吞吐 / SOTA / 原生级 / 定理在 LM 复现声明。
- 端到端为「真实权重 + 24 层融合 hook」运行时挂接；完整 student 由仓内 materialize 脚本组装并逐位校验。
- 本轮不重新拉 2 GiB、不重跑 580 s 生成：LIVE 真相由 v10_3_scoped_verify.json 与 v10_copy_full_audit.json 承载。

原始证据：`final_migration_proof.json`、`v10_3_scoped_verify.json`、`v10_copy_full_audit.json`、`v10_switch_law_audit.json`、`student_scoped_manifest_v10_3.json`。
