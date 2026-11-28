# TS2SC v10 · 可证伪台账（防 overclaim 闸门）

日期：2026-11-28 · 上游：`kino-koki/SimplCore` @ `bec5ab8fd7`

> **这不是结论表。** 它登记的是「哪些命题还没被证、哪些已被上游证伪、
> 哪些话永远不许写」。每一条都给证伪条件与所需实验，供后续 GPU 工作逐条销账。
> 目的只有一个：v10 同步了上游**架构能力**，但绝不顺手继承上游**没有拿到的主张**。

---

## 0. 上游 v11 已经替我们证伪的两条（必须随 v10 一起读）

上游 `repro/independent_v9_v11/README.md` 头部写明：
`MECHANICAL_COMPLETE = False` / `HONEST_COMPLETE = False` / `COMPLETE_BEYOND = False`。
质量轴只到**稳态 parity**（A1 +0.0029、A3 +0.0012 bpc @6000 步，3 seeds），
两条独立否决：

| # | 否决理由 | 原始数字 |
|---|---|---|
| V1 | **2× 对手步数对照失败** | s11 A1 +0.1156、s22 +0.1835 超阈值 → `fair_all=False`；机械判定本身也是 False |
| V2 | **A2 的负 delta 全是崩坏区伪影** | 所有「赢」落在 L768/L1536，两臂 bpc 相对 L48 基线劣化 +1.2~+7.7，「双方都崩、S 崩得稍慢」；加绝对质量约束后 `effA2min` +0.0068~+0.1494，稳定区间归 0 |

另有**三次自我推翻**（v1 A3 假赢、`final_decisive` 的「纯结构上限」、v6 的
「M=512 parity」）与**两次抓到的假胜**（α=0.75 只加 SGC 臂 → 两臂共享 Block 后归零；
v11 只看 delta 的 A2 判据 → 加「赢所在 L 上 S 的 bpc 劣化 ≤ +1.0」收紧）全部留档。

**结论**：上游在纯度内自由度已穷举、按 v11 头部停止准则收束。
要全轴赢必须放弃静态 Toeplitz / O(1) 定态 / 纯度三者之一 ——
任取一条即不再属于该仓库定义的架构。这是**定义性边界，不是工程未及**。

> TS2SC v10 把这套外挂挂到 1B teacher 上，**不改变这个边界**。
> 下面每一条命题都带着它。

---

## 1. 命题登记（9 条）

| ID | 命题 | 当前状态 | 证伪条件 | 所需实验 |
|---|---|---|---|---|
| **P-β** | 24 个逐层 `β_l ∈ [0,1]` 经 L2 校准后，存在 ≥1 层最优 `β_l = 0` | `NOT_TESTED` | 若某层最优 β_l=0，则该层特性只是「登记可用、实际未启用」 | teacher 同条件 forward + 逐层 hidden cos / logit KL + 网格 |
| **P-2×** | v10 在等算力 2× 对手步数下仍保持 parity | `NOT_TESTED`（上游已 `FALSIFIED`） | 复现 V1 即证伪；不得沿用上游 parity 数字 | 1B 级等算力双跑 + 配对 CI |
| **P-A2** | v10 在 OOD 长 L 上的负 delta 不是崩坏区伪影 | `NOT_TESTED`（上游已 `FALSIFIED`） | 加绝对质量约束后稳定区间仍归 0 即证伪 | L384/768/1536 + 绝对 bpc 约束 |
| **P-8M** | 8M token 端到端可行 | `PARTIAL_FALSIFIED` | 当前 hybrid 保留 teacher 全注意力：192 GiB KV + 10^16.1 对 + 64× RoPE 外推 ⇒ 不可行 | 只有 W=4096 滑窗可降到 96 MiB，但**撞「融合不替换」红线** |
| **P-O1** | 24 层 SGC 分支状态与 L 无关 | `VERIFIED`（v10） | L=1/8/64 恒 46,656 B/层；状态出现 L 维即证伪 | 已 `test_v10.py::test_11` |
| **P-P7** | 三条旁路不回写冻结主干 | `VERIFIED`（v10） | 旁路全开/全关主干快照 `max\|Δ\|>0` 即证伪 | 已 `test_v10.py::test_07` |
| **P-β0** | β=0 与纯 teacher 端到端逐 token 相同 | `VERIFIED`（本轮深审） | 任一 token 不同即证伪 | 4 语言中 zh/en 双跑 T vs C0 |
| **P-quality** | v10 的 6-token 生成可读 ⇒ 质量/能力不降 | `NOT_TESTED` | 短生成全落在 `<think>` 或答错即不能升级为质量主张 | ppl / 下游集 / 配对 Wilson CI（**纯 CPU 不做**） |
| **P-identity** | 身份层退役后模型「原生级 / 能力更强」 | `NOT_CLAIMABLE` | 任何此类句子都需 GPU LoRA SFT + 配对 CI；CPU 输出层微调已知几乎只有污染 | 用户目标机自跑 |
| **P-read** | 312 个 NEW 张量全部进前向 / 有归属，零空挂 | `VERIFIED`（v10.1） | 任一 key 扰动后 Δ 不变且构造期也未读取即证伪 | 已 `test_v10.py::test_16/17`，见 `V10_1_READ_CONTRACT_REPORT.md` |
| **P-switch** | 31/31 能力都有可切换 slot，且全关逐位 ≡ 原模型 | `VERIFIED`（v10.2） | 全关 Δ 非零、或某门无法独立差分、或契约门改变前向、或 DEFERRED 可置 True 即证伪 | 已 `test_v10.py::test_19/20/21`，见 `SWITCH_LAW.md` |
| **P-scope** | 交付只继承 SGC_CORE + SIDECARS + SMM 三组，且 SMM 真的被门禁写入（非空挂） | `VERIFIED`（v10.3） | `observe()` 后 commits=0 / 读回不命中 / 全开与 smm_off 无差异即证伪 | 已 `test_v10_3.py::test_04/05/06/07`，见 `V10_3_SCOPED_DELIVERY.md` |
| **P-10M** | 配置开到 10M 且分支状态 O(1) 恒定；端到端仍不可行 | `PARTIAL`（v10.3） | 分支状态随 L 增长、或闭式 ≠ 逐步 gate=0 即证伪分支侧；teacher 侧 240 GiB KV 账本判定不可行 | 已 `test_v10_3.py::test_09/10`，见 §1e |
| **P-proof** | 7 组独立判据证明迁移真实且三组特性可归因，排除「原模型直接跑」 | `VERIFIED`（终局） | 全关 Δ 非零、360 NEW 任一 sha 不符、任一组关断不改变 Δ、生成级 T≠OFF 即证伪 | 已 `final_migration_proof.py`，见 `FINAL_MIGRATION_PROOF.md` |
| **P-fullweights** | 仓库自包含完整权重：2.01 GiB teacher COPY 字节在仓，可逐位组装 579 张量 | `VERIFIED`（终局） | 子片 sha256 不符或 materialize 任一 entry 校验失败即证伪 | `FULL_WEIGHTS_PUSH_REPORT.json` / `materialize_full_student_v10_3.py` / 组装逻辑 5 张量小规模真跑 `materialize_logic_test.json` |

---

## 1b. 本轮深审的一处未满分观察（不洗白）

`v10_multilingual_verify.json` 实测 `distinct_C1_texts = 3/4`：
法语短问候 `Bonjour` 与英语 `Hi` 的 C1 都收敛到 `\nThe user's message`。

- 成因：极短问候下**基座**倾向用英文 `<think>` 开头；β=1 分支没有强行改写
  teacher 的语言选择行为 —— 这与「零身份/零语言偏置注入」一致，是正面旁证。
- 它**不是**硬门槛，未进 `passed` 判据；硬门槛是 219/219 全量逐位 +
  T==C0 端到端零回归 + C1 架构生效 4/4 + 全 finite + 零泄漏。
- **不重跑换 prompt 取巧、不改写、不抹平**；原始 checkpoint 保留在
  `evidence/gen_deep_checkpoint.json`。
- 因此也永远不会写「四语言生成文本必然互不相同」。

## 1c. 一处自身声明错误，已纠正（不抹平）

v10.0 的 `read_keys()` 返回 `24×13=312` 并被当作「全部进前向」的证据；
扰动审计实测 **10/13 ACTIVE、72/312 空挂**（`role.Wc` / `gate.alpha` /
`gate.beta`），与 V7「空挂」同构。

v10.1 把三者以**中性初值**接入（0 / 1.0 / 0.0），
`27/27` 个 Δ 快照逐位相同 ⇒ 权重字节、manifest、分片、全部既有生成证据
**一个没动、全部仍有效**。现在 12 ACTIVE + 1 PERSISTED_INIT、
`NOT_CONTRIBUTING=0`，由 `test_16/17/18` 钉死。

**这不等于「3 个参数已在贡献」**：中性初值只是消除空挂并变成 L2 可校准
槽位。`gate.beta` 现在才真正成为 `beta_l.json` 的写回落点；`P-β` 仍
`NOT_TESTED`。详见 `V10_1_READ_CONTRACT_REPORT.md`。

## 1d. 「完全继承所有能力」的准确口径（v10.2）

用户以**开关法则**重新定义：`y = T(x) + 𝟙[master]·Σ_f s_f·Δ_f(x)`，
全关逐位 ≡ 原模型。按此定义已达成，**31/31 都有 slot**：

- 20 FORWARD（13 个可差分前向门）、9 CONTRACT（开关只管登记/校验，
  phase F 实证切换前后 Δ 逐位相同）、2 DEFERRED（`NOT_IN_SOURCE`，
  置 True 被拒绝）。
- 全关 12 例 Δ 严格零 + `set_beta(1.0)` 不复活；单关默认 12/13 可观测
  （`role.Wc` 中性初值），注入探针后 13/13；全开 27/27 逐位相同。
- 注册表 `capability_registry.json` 为**唯一权威**（历史 24/27/29/30 口径作废）。

**这条定义不覆盖**：31 项都在前向贡献数值、定理在 LM 上复现、
能力变强、无词表设计已继承。四者都不成立，见 §2 红线。

## 1e. v10.3 范围收窄 + 10M（用户指令）

> 「就继承 SGC 以及其外挂模块 SMM记忆与自进化模块就行了，上下文开到10M」

**本次交付只继承三组**：G1 SGC_CORE、G2 三条 SIDECARS、G3 SMM。
SMM 来自上游真实源码 `evolution_mixer.py` 的 `FusedEvolutionMixer` /
`CalibratedFusedMixer`（零阶巩固：τ 证据 / dominance / cooldown /
读回拒绝 / capacity 淘汰 / `rho_mem`）。历史 31 项注册表、9 CONTRACT、
2 DEFERRED **原样保留不删除**，只是排除在本次继承声明之外。

**两处自查修正**（不抹平）：
1. SMM 初值全零 + 散共现被门禁全拒 ⇒ 全开与 `smm_off` 的 ‖Δ‖ 相同，
   差点又是「空挂」。改结构化强共现后 16 commits / 1278 rejects /
   读回 `key_value=5` 命中 ⇒ 记忆真写入可读。
2. SGC 与 SMM 各自归一化到 `βε‖x‖`，相加可到 2× ⇒ 对**总和**统一
   投影；`smm_off` 零缩放保持与 v10.2 逐位相同。

**10M 分层**：`max_position_embeddings = 10,485,760`（配置值）；
分支侧 `advance_idle(10M)` 是 **O(1) 闭式实跑**，状态恒定
3,675,648 B、闭式 == 逐步 gate=0；teacher 全注意力 **240 GiB KV**
+ **80× RoPE 外推** ⇒ `INFEASIBLE_LEDGER`（账本判定，非实跑失效）。
**配置开到 10M ≠ 端到端 10M 已验证**。

## 2. 永远不许写的句子（红线）

- ❌ 「v10 超越 / SOTA / 严格优于 Transformer / Mamba2」
- ❌ 「质量轴整个超越」——上游两条独立否决已封死
- ❌ 「8M / 无限上下文**端到端已验证**」——只能写「分支 O(1) 状态性质」
- ❌ 「继承 SimplCore **所有**能力」——7 项 REGISTERED 是理论前提/工程契约，不是 LM 上复现
- ❌ 「β=1 是校准成品 / 原生级 / 能力更强」
- ❌ 「无知识污染 = 能力不变」——口径只到权重逐位 + 生成零身份标记
- ❌ 「等于上游 PyTorch 源码执行」——TS2SC 是纯 numpy CPU 等价实现
- ❌ 「四语言生成文本必然互不相同」——实测 3/4，见 §1b
- ❌ 「312 个 NEW 张量全部在前向贡献」——其中 3 个是中性初值的校准槽位，见 §1c
- ❌ 「31 项能力全部在前向贡献数值」——9 项是契约门、2 项 DEFERRED，见 §1d
- ❌ 「无词表设计已继承」——源码中不存在、未推送，`set_contract` 显式拒绝
- ❌ 「完全继承 = 能力更强 / 定理在 LM 上复现」——定义只到 slot + 全关零回归
- ❌ 「10M 上下文端到端已验证 / 已跑通」——只有配置 + 分支侧 O(1) 实跑，teacher 侧账本不可行
- ❌ 「SMM 自发自进化 / 模型自己学会记忆」——`observe()` 是显式门禁，未调用时贡献严格为零
- ❌ 「v10.3 继承了 31 项全部能力」——本次范围只到 3 组，其余排除在继承声明外

**允许写**：`β=0 逐位零回归`、`219/219 COPY 全量逐位一致`、
`FFT↔IIR 1.69e−14`、`O(1) 定态`、`P7 主干不回写`、
`架构同步 bec5ab8fd7`、`COMPLETE_BEYOND=False 随附`。

---

## 3. v10 已销账的证据（与台账对照）

| 证据 | 文件 | 状态 |
|---|---|---|
| 22 分片 LIVE git blob sha1 | `upstream_pull_report.json` | ✅ 22/22 |
| 219/219 COPY 全量逐位 | `v10_copy_full_audit.json` | ✅ 本轮 |
| 312 NEW 重开逐字节复验 | `replay_full_v10.py` / manifest | ✅ 312/312 |
| 531 张量 UNCLASSIFIED=0 | `student_full_manifest_v10.json` | ✅ |
| 4 语言多臂生成 T/C0/C1 | `v10_multilingual_verify.json` | ✅ 本轮 |
| 15 条 v10 架构回归 / 39 条 v8 回归 | `test_v10.py` / `test_v8.py` | ✅ 54/54 |
| 上游 13 文件 LIVE sha256 快照 | TS2SC 仓 `ts2sc/upstream_simplcore_v11/` | ✅ |

---

## 4. 下一步的唯一正当顺序

1. **目标机** `replay_full_v10.py --execute` 组装完整 student（沙盒磁盘 417 MiB 装不下 2.6 GB）。
2. **L2 校准**：`calibrate_beta_v10.py` 协议（本轮只落骨架，`NOT_EXECUTED`，
   无 checkpoint、无分数、不产出任何数字）→ 销 P-β。
3. **L3 配对评估**：可读率 / mojibake 率 / 配对 Wilson CI → 才碰得到 P-quality。
4. **等算力 2× 对照** → 销 P-2×（预期复现 V1，如实留档，不抹平）。
5. GPU LoRA SFT 由维护者自跑；本仓不做身份微调、不声称。

> 台账的用处不是让 v10 看起来更强，是让**每一步升级都有明确的、可被推翻的条件**。
> 上游 v11 最值钱的产出恰恰是那两行 `False` —— 把它抄进 TS2SC，比抄任何数字都重要。
