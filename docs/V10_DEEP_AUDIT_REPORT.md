# TS2SC v10 深审 · 抽查→全量 / 2题→4语言 / 单元级→端到端

日期：2026-11-28 09:08:08 · `passed = True` · 耗时 1124.7 s

## 这一轮「继续」做了什么

| 上一轮 | 本轮 |
|---|---|
| 8 个 COPY 抽查逐位 | **219/219 全量**逐位 sha256 对照 |
| 2 题 × 2 臂 × 4 token | **4 语言 × (T/C0/C1) × 6 token**，共 10 臂 |
| β=0 零回归仅单元级 Δ=0 | **端到端 T 与 C0 逐 token 相同** |

## 结论

| 判据 | 结果 |
|---|---|
| 22/22 分片 LIVE git blob sha1 | ✅ `True` |
| **219/219 COPY 全量逐位** | ✅ `True`，mismatch `0`，2.01 GiB |
| T == C0 逐 token（zh/en 双跑） | ✅ `True` |
| C1 架构生效（4/4 语言） | ✅ `True` |
| C1 文本互不重复（**观察量，未满分**） | `3/4` |
| 全 finite / 全解码守卫 | ✅ / ✅ |
| 身份标记泄漏 | **0** |
| **passed** | **True** |

## 全量 COPY 审计（非抽查）

`v10_copy_full_audit.json` 逐条记录 219 个 teacher 张量的
`name / bytes / matches`。sha256 全部从**本次 LIVE blob-sha 校验过的字节**
流式重算；任一不符即整体 FAIL。审计总字节
`2.01 GiB`，耗时 1.6 s，
`all_bit_identical=True`。

这把上一轮「干净」的口径从抽样证据升级为全量证据，但**仍不是**
ppl / 下游 benchmark / 知识集配对 CI。

## 多语言多臂生成（节选）

| 语言 | T（纯 teacher） | C0（β=0） | C1（β=1 交付） | T==C0 |
|---|---|---|---|---|
| `zh` | `<think>

</think>

你好！` | `<think>

</think>

你好！` | `<think>
嗯，用户发送的` | `True` |
| `en` | `<think>
Okay, the user` | `<think>
Okay, the user` | `<think>
The user's message` | `True` |
| `ja` | `—（未跑 T 臂）` | `<think>
ふあ、こんにちは` | `<think>
ふふふ、` | `n/a` |
| `fr` | `—（未跑 T 臂）` | `<think>
Hmm, l'util` | `<think>
The user's message` | `n/a` |

> T 臂只在 zh/en 两语言跑以控制纯 numpy CPU 时间；ja/fr 只有 C0/C1。
> 6 token 短生成只证 finite / 不乱码 / 架构生效 / 零回归 / 零身份标记，
> **不是**质量、能力或事实正确性结论。

### 一处没满分的观察（不包装、不重跑洗白）

`distinct_C1_texts = 3/4`：
法语短问候 `Bonjour` 与英语 `Hi` 的 C1 都收敛到
`\nThe user's message`。成因是**基座在极短问候下倾向用英文
`<think>` 开头**，不是身份/语言偏置注入 —— 反而佐证 β=1 分支
没有强行改写 teacher 的语言选择行为。

- 它没有进 `passed` 判据（硬门槛是架构生效 4/4 + 零回归 + 零泄漏 + 全量逐位）；
- 它是**观察量**，不是「四语言必然不同」的质量主张；
- 若把 prompt 换成更长的法语句子很可能变成 4/4，但那属于调 prompt 取巧，
  本轮**不重跑、不改写、不抹平**，原始 checkpoint 保留在
  `evidence/gen_deep_checkpoint.json`。

## 同批落地的两份防 overclaim 资产

- `TS2SC_V10_FALSIFIABLE_LEDGER.md`：把上游 v11 的两条独立否决
  （2× 对手步数对照失败 / A2 崩坏区伪影）与三次自我推翻映射为
  TS2SC 侧 9 条可证伪命题 + 7 条永远不许写的句子。
- `calibrate_beta_v10.py`：L2 逐层 β_l 校准协议骨架，
  **NOT_EXECUTED / PLAN-ONLY / 无 checkpoint / 无分数**，
  `--execute` 在无 GPU 的沙盒显式 REFUSE。

## 边界（不删减）

- 全量 COPY 逐位 = 219 个 teacher 张量与本次 LIVE blob-sha 校验字节一致；不是 ppl / 下游 benchmark / 知识集配对 CI
- 6 token 短生成只证 finite / 不乱码 / 架构生效 / 零回归 / 零身份标记；不是质量、能力、事实正确性结论
- T 臂只在 zh/en 两语言跑以控制纯 numpy CPU 时间；其余 2 语言只有 C0/C1
- β=1 仍是尺度契约配置（T1'），不是 L2 校准成品；24 个 β_l 待校准
- 上游 v9–v11 COMPLETE_BEYOND=False 不变；本轮不继承超越/SOTA 主张
- 纯 numpy CPU 等价实现，未跑上游 PyTorch pytest；无 GPU/bf16/多卡验证
- Qwen4-27B 仍未发布；无原生级 / 上下文理论无限实测声明

原始证据：`v10_copy_full_audit.json` / `v10_multilingual_verify.json` / `upstream_pull_report.json`。
回归：`test_v8.py` 39/39 + `test_v10.py` 15/15。
