# TS2SC 系统路线与不变量 · v8.1

日期：2026-11-25 · 定位：**持续迭代的工程系统**，不是一次性交付

---

## 一、「一直做」怎么定义

不是等发布日再动手，也不是无限改稿。每一轮固定走同一个闭环：

```
① 源码审计   对真实 TS2SC/V7 文件做字面检查，找根因（不靠回忆）
② 契约化     每个根因变成一条机器可判的断言
③ 实现       补上对应的模块 / 修法，纯函数部分必须沙盒可跑
④ 门禁       分级 L0–L4，下游阶段拒绝在前置未通过时执行
⑤ 回归       全套 unittest + 负对照，任何 FAIL 阻止打包
⑥ 登记       根因 → 修法 → 测试的闭环表；做不到就如实留在待办
⑦ 交付       打包 + 报告，不含权重字节、不含未实测数字
```

本轮走完 ①–⑦，版本 **v8.0（骨架）→ v8.1（系统）**。

---

## 二、v8.1 新增了什么

| 模块 | v8.0 | v8.1 |
|---|---|---|
| `modeling_v8.py` | 无 | **融合层实现**：保留 teacher 全部功能层，SGC 并联注入 |
| `migrate.py` | 只有账本分类 | config-driven 执行器：`plan / copy_assert / simulate / execute(T+0)` |
| `ts2sc.py` | 无 | **CLI 硬闸门状态机**：7 个 stage，缺前置直接拒跑 |
| `test_v8.py` | 无 | 26 条回归 + 负对照 + 报告落盘 |
| `preflight.py` | A/B 两部分 | 新增 **part_C 融合不变量**（12 检查） |

---

## 三、核心设计：融合，不是替换

这是你反复强调的那条，我把它写成可证伪的形式了。

**V7 做错了什么**：`migrate_qwen3.py` 把 teacher 的 `self_attn` 整个删掉，
学生里一个注意力层都不留。历史记录那句最准——「知识仍在（FFN 逐位迁移
cos 1.0000），但循环与语义错乱」。权重搬对了，接口接错了。

**v8 的做法**：

```
out = teacher_block(x) + enabled · β · sgc_branch(x)
```

- teacher 的 attention / QSA / MoE / norms / FFN / embed / lm_head **一层不删**；
- SGC 卷积注意力是**并联的加性分支**，走可学 blend 门 β；
- `enabled=False` → 输出与 teacher **逐位相同**；
- `enabled=True` 但 β 初值 0 → 仍逐位相同，只有经 L2 校准/适配后才贡献。

这正好把你之前在 1.7B 上观察到的三件事变成了机器判据：

| 观察 | v8 里的形式 |
|---|---|
| 原本模型能力完整保留 | **I1**：关闭时逐位等价，不需要配对 CI 就能证 |
| 手动开启才继承 | **I2**：β=0 初始化 + 显式 enabled 开关 |
| 手动关闭的特性没继承 | **I1** 的反向：关 = 分支完全不进前向 |

而且它**明确不是无门控自演化**——β 必须经 L2 校准，不许停在 0，也不许像
V7 那样硬编码 `alpha=-2`。

**一处必须说清的张力**：工程侧 v8 是 **hybrid 服务架构**（保留 teacher
attention）；ICML 稿子刻画的是**静态核、无 softmax、无 KV cache** 的构造。
两者不是同一个架构，**互不作为对方的证据**。这条我写进了
`equivalence_invariant()` 和 preflight part_C，不藏。

---

## 四、不变量（每轮回归都必须全过）

| ID | 不变量 | 对应根因 |
|---|---|---|
| I1 | `enabled=False` ⇒ 与 teacher 逐位相同 | 原能力不回归 |
| I2 | `enabled=True, β=0` ⇒ 仍逐位相同；β 必须校准 | R5 |
| I3 | GQA slot **零填充为 0**，每行映射到真实 teacher 行 | R6 |
| I4 | QK-norm 折叠进 Wq/Wk，`dropped_columns=0` | R8 |
| I5 | `tie_word_embeddings` 继承；untied 时显式 COPY `lm_head.weight` | R1 |
| I6 | tokenizer 六文件 REAL、非 LFS 指针、`len(tok)==embed.rows` | R2/R3 |
| I7 | `copy_bit_identical` 只是 L0 必要条件，L2 必须有 teacher forward | R9 |
| I8 | L3 是 **no-regression**，不是 improvement；不写 SOTA | 口径 |

---

## 五、R1–R10 闭环表

| 根因 | 闭环位置 | 回归用例 |
|---|---|---|
| R1 输出头被丢 | `modeling_v8.lm_head_plan` / `provenance` / `spec.student_config` | `test_full_coverage_and_lm_head` |
| R2 tokenizer best-effort | `contract.static_contract` 六文件 REQUIRED | `test_missing_file_rejected` |
| R3 子集 BPE 错位 | vocab/embedding 行数 + 5 条往返 + UTF-8 流式 | `test_vocab_embedding_mismatch_rejected` |
| R4 非退化即通过 | `align_accept.mojibake_rate` + Wilson CI | `test_old_gate_passes_mojibake_v8_rejects` |
| R5 mixer 结构性失效 | β=0 初始等价 + 强制校准 | `test_disabled_is_bit_identical` |
| R6 半个 slot 恒零 | `build_slot_map` 禁零填充 | `test_slot_map_no_zero_padding` |
| R7 硬编码 Qwen3 形状 | `TeacherSpec` 缺字段即 blocker | `test_missing_required_is_blocker` |
| R8 QK-norm 被 SKIP | `fold_qk_norm` DERIVE | `test_qknorm_folded` |
| R9 只验逐位复制 | `migrate.copy_assert` 降级为 L0；L2 需 teacher forward | `test_copy_assert_catches_dtype_and_shape` |
| R10 历史失败未闭环 | 本表 + L0–L4 硬门禁 | 全套 26 条 |

---

## 六、CLI 状态机（发布日不会现场改硬编码）

```
python ts2sc.py --stage preflight                 # 审计 + 自测 + 融合不变量
python ts2sc.py --stage plan --teacher DIR --index keys.json
python ts2sc.py --stage bind --teacher DIR --out STUDENT
python ts2sc.py --stage migrate --t0             # 仅发布日机器
python ts2sc.py --stage align  --t0
python ts2sc.py --stage accept --t0
python ts2sc.py --stage status
```

前置未过 → `GATE_BLOCKED`（exit 2）；无真实运行时 → `T0_REQUIRED`（exit 3）；
断言失败 → `FAIL`（exit 4）。状态写入 `ts2sc_state.json`，可审计、可回溯。

**本轮端到端实跑**（合成 fixture，非 Qwen4 数据）：

```
preflight PASS | plan PASS | bind PASS
migrate  -> T0_REQUIRED (exit 3)
accept   -> GATE_BLOCKED missing=['migrate','align'] (exit 2)
```

顺手抓到并修了一个真 bug：preflight 的自测目录原本会擦掉 CLI 状态目录，
现在 fixture 隔离到 `_selftest/fixture`，状态独立到 `_run`。

---

## 七、回归结果（真实跑过）

```
unittest: 26 tests / 26 PASS / 0 failures / 0 errors
preflight A: 10/10 历史根因字面命中
preflight B: 契约 PASS，三个负对照全部被拦，provenance PASS，
             乱码文本被拒（旧 gen_quality 会判 PASS）
preflight C: 12 项融合不变量，0 failures
py_compile: 全部通过
```

---

## 八、下一轮队列（不夸大，按顺序来）

1. **T+0 执行器落地**：`migrate.execute()` 目前是显式 stub，需要真实
   safetensors index + torch；Qwen4 若 QSA/MoE，`adapt_teacher_output` 和
   per-expert 映射要按真实 block 签名写，不猜。
2. **L2 对齐脚本**：逐层 hidden cos / logit KL / top-1 一致率，
   需要 teacher 同条件 forward。
3. **β 校准协议**：把「手动开启」变成可复现的校准步，而不是人工调参。
4. **L3 配对评估集**：N≥200 prompts，中文可读率 / mojibake / EOS /
   chat template 身份 / 配对 Wilson CI。
5. **vLLM / HF 服务路径验收**：L4「原生级」的最后一关，零自定义 decode。
6. **撤回登记续写**：延续 W1–W15 口径，把 R1–R10 的修法登记成 W16+。

---

## 九、边界（每轮都重写一遍，防止自己滑出去）

- 截至 2026-11-25，Qwen4-27B 未发布：无 config、无权重、无 tokenizer、
  无许可证、无日期。本轮**未访问任何 Qwen4 数据**。
- 无 torch / teacher forward / 蒸馏 / ppl / 生成样本 / 吞吐 / 能力数字。
- `migrate.execute()` 与 L2/L3/L4 **未执行**，是 T+0 的活，不是今天的成果。
- 工程 v8（hybrid）≠ 论文构造（no-softmax），互不引用为证据。
- 论文维持「工程验证不进投稿包」的裁定，本系统不改稿、不加定理。
