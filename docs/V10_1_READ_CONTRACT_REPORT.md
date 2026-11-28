# TS2SC v10.1 · 读取契约修正：72/312 空挂张量的发现与修复

日期：2026-11-28 · 性质：**自查发现的自身缺陷**，不是新增能力

---

## 1. 为什么回头查这一条

V7 历史最致命的根因之一是：「NEW 参数全零 + SGC 卷积注意力只是空挂在
`state_dict` 里，根本没进前向」——权重搬对了、接口接错了。

v10 的 `SGCV10Branch.read_keys()` 直接返回 `24 × 13 = 312`，被当作
「312 个 NEW 参数都在前向路径」的证据。那是**声明**，不是验证。
本轮用独立扰动判据复查：对每个 NEW key 单独扰动 → 重建同 seed 单层 →
同输入 / 同 β / 同 pos 比较 Δ。参数没进前向，扰动就不会改变 Δ。

## 2. 查出来的真问题（v10.0）

| key | 形状 | v10.0 扰动后 Δ 变化 | 判定 |
|---|---|---|---|
| `role.Wc` | `[3, 1536]` | **0** | 空挂 |
| `gate.alpha` | `[1]` | **0** | 空挂 |
| `gate.beta` | `[1]` | **0** | 空挂（且构造期未读） |

其余 10 个 key 全部 ACTIVE。即 **10/13 生效、72/312 空挂**（24 层 × 3）。
`read_keys()` 谎报 312/312。这与 V7「空挂」是同一类缺陷，只是规模小得多，
且**不影响**已验证的 10 条路径与全部生成结果。

源码层根因：`SGCV10Layer.hook` 里
- `e[r] = role_snr` 用 one-hot 直接构造，`self.role_Wc` 从未被读取；
- `self.alpha` 在 `__init__` 赋值后从未出现在 `hook`；
- `self.beta = 0.0` 硬编码，`params["gate.beta"]` 从未被读取。

## 3. 修法：让 13/13 都有归属，且**数值逐位不变**

关键约束：不能让上一轮的全部生成证据链作废。所以三个参数都以
**中性初值**接入，修正前后 Δ 完全一致：

| 参数 | 接入方式 | 初值 | 语义（L2 校准槽位） |
|---|---|---|---|
| `role.Wc` | `role_gain = Wc[r%3] @ x / √d`，加到 role 向量 `e` | 全零 | 可校准的输入侧 role bit 增益 |
| `gate.alpha` | `gain = alpha · beta` 进入 T1' 契约 | `1.0` | 可校准的逐层增益 |
| `gate.beta` | `__init__` 读 `params["gate.beta"]` 作为该层 β 初值 | `0.0` | **L2 校准产出的 `beta_l.json` 写回槽位** |

`gate.beta` 用独立的「构造期持久化」判据复核（不调用 `set_beta`）：
扰动到 `0.77` 后 `lay.beta == 0.77` ⇒ `PERSISTED_INIT`。
交付时 `set_beta(1.0)` 仍覆盖，行为不变。

`read_keys()` 的文档改为明确「声明数、不是读取证据」，新增
`verified_read_keys()` 返回三态归类：`forward_active(10)` /
`forward_neutral_init(2)` / `persisted_init(1)`。

## 4. 验证（`passed = true`）

```
v10.1 读取契约审计   ACTIVE 12/13 + PERSISTED_INIT 1/13
                     NOT_CONTRIBUTING 0 → all_keys_accounted_for = True
v10.1 vs v10.0       27/27 Δ 快照逐位相同（3 层 × 3 量级 × 3 pos）
                     mismatched 0 → 修正数值中性
架构不变量           FFT↔IIR 1.69e−14 / SE1 0.0 / P7 max|Δ|=0
                     短卷积 8.33e−17 / 纯度 PASS / β=0 严格零
                     β=1 契约比 0.1000000 / O(1) 46,656 B 全部一字未变
回归                 test_v10 18/18 + test_v8 39/39 = 57/57
```

**因为 27/27 逐位相同，上一轮 4 语言 10 臂生成、219/219 全量 COPY 审计、
β=0 端到端零回归全部保持有效，无需重跑、无需改动权重字节。**
312 个 NEW 张量、`student_branch_v10.safetensors`、manifest、2 个权重分片
**字节一个没动**；本轮只改 `sgc_v10.py` 与测试/审计/文档。

## 5. 诚实边界（不借修正夸大）

- **不声称这 3 个参数「已在贡献」**：`role.Wc=0`、`gate.alpha=1` 是中性
  初值，接入只为消除空挂、并把它们变成 L2 可校准槽位。`verified_read_keys()`
  明确区分 `forward_active` 与 `forward_neutral_init`。
- **不声称「所有特性已继承」**：这只是 312 个 NEW 张量的读取契约，
  与 7 项 REGISTERED 理论前提、上游 `COMPLETE_BEYOND=False` 无关。
- 本轮**没有**跑 L2 校准、没有 GPU forward、没有新生成样本、没有 ppl / CI；
  `calibrate_beta_v10.py` 仍是 `NOT_EXECUTED`。
- `gate.beta` 现在才真正成为 `beta_l.json` 的落点；在此之前，
  `P-β` 仍是 `NOT_TESTED`（见 `TS2SC_V10_FALSIFIABLE_LEDGER.md`）。
- 未删除任何云端条目；历史 `weights/` 与 `identity_*` 原位保留。

## 6. 这一轮真正的价值

不是「又加了一个能力」，而是**把一句我自己写错的声明换成了可复验的判据**。
`read_keys()` 谎报 312/312 与 V7 空挂是同构问题；现在有
`v10_read_contract_audit.py` 的扰动 + 持久化双判据，
以及 `test_v10.py::test_16/17/18` 三条回归钉死，以后再改架构不会静默退化。
