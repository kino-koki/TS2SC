# TS2SC v9.0 · Apache-2.0 开源发布 + 身份认同微调（门控锁定修复）

> **v9.0 本轮**：`Tuan-0.1-TS2SC-1B` 以 **Apache License 2.0** 公开；
> `identity_alignment.py` 修好 v8.9 的门控 bug（只在首步判定并锁定），
> 词级锚「青奈」两步完整生效；新增 `identity_sft.py`（闭式 ridge 真权重训练，
> 零 teacher 全量前向、峰值 <60 MiB，满足「不要 OOM」）；
> `mem_guard()` 拒绝并行 preload，所有重任务严格串行。
> 口径澄清见 `../output/APACHE_2.0_RELEASE.md`：用户说的「A2.0」是 Apache-2.0
> 许可证，不是仓里那份 `A2.0协议.md` 架构流水线；后者核对报告保留、不删。

---

# TS2SC v8.6 · 口径修正：不是「完全继承」

> ⚠️ **v8.6 更正**：源码核实发现 **「无词表 / tokenizer-free」设计确实存在**
> （`simplcore_src/README.md` L287/L302/L507、L348；`train/corpus.txt` GlyphStream 整节），
> 但**对应实现模块未推送到本地**，当前 MiniCPM5-1B 迁移体也**没有继承它**
> ——反而绑定 130560 词表、全 COPY `embed_tokens` / `lm_head`。
> 所以此前「27 项全特性继承」的说法**不成立**。详见
> `../output/无词表设计核实与继承缺口报告.md` 与 `features.py` 的 F28/F29（NOT_INHERITED）。

**定位**：MiniCPM5-1B 上已完成 24 层融合迁移；**29 项特性登记 = 20 RUNNABLE +
7 REGISTERED + 2 NOT_INHERITED**；乱码的两个理论根因已可证修复，β>0 开启态实测会说话。
同时保留 Qwen4-27B 发布日 T+0 的就绪层。

```bash
cd ts2sc_v8
python test_v8.py          # 38 条回归，写 test_report.json
python scale_theory.py     # 乱码理论 6 条自测
python preflight.py        # A 源码审计 / B 契约自测 / C 融合不变量
python features.py         # 打印 27 项特性注册表
python ts2sc.py --stage status
```

## 文件

| 文件 | 职责 |
|---|---|
| `spec.py` | TeacherSpec：零硬编码，缺字段即 T0 blocker；slot 由真实 head 布局推导 |
| `contract.py` | tokenizer↔embedding 绑定；支持 Qwen3 六文件 / MiniCPM5 tokenizer.json-only 两种真实形态 |
| `provenance.py` | 逐张量 COPY/DERIVE/NEW 账本；lm_head 显式化、禁零填充、QK-norm 折叠 |
| `modeling_v8.py` | 融合层定义：保留 teacher 全部功能层，SGC 并联 β 门控 |
| `features.py` | **27 项 SimplCore 特性注册表**：出处 / V7 真实状态 / v8.4 落点 / 断言类型 |
| `scale_theory.py` | **乱码理论**：T1 无量纲尺度契约 / T2 尺度不变 role bit / T3 解码守卫 + 自测 |
| `run_minicpm.py` | 真实拉取 22 分片（4 次重试）、纯 numpy 教师前向、合法初始 id + finite 标志 |
| `run_full_migration.py` | **全 24 层融合迁移 + 19 项 RUNNABLE 特性真跑断言** |
| `verify_student_speaks.py` | 四臂端到端说话验证：T / β=0 / β=1 契约 / v8.3 负对照 |
| `migrate_minicpm_full.py` | 219 张量融合账本 + 目标机全量 replay |
| `credential_guard.py` | 凭据卫生扫描（⚠️ 不叫 secrets.py，会遮蔽标准库） |
| `align_accept.py` | L0–L4 门禁；配对 Wilson CI、`mojibake_rate` |
| `ts2sc.py` | 7 stage 硬闸门 CLI |
| `preflight.py` / `test_v8.py` | 审计 + 35 条回归 |

## 核心等式（融合而非替换）

```
out_l = teacher_block_l(x) + Δ_l(x, o_l)                       l = 0..23
Δ = β · ε · r_x · o/(r_o+δ) ,   ‖Δ‖₂ ≤ βε‖x‖₂  (ε=0.10)
```

teacher 的 attention / FFN / norms / embed / lm_head **一层不删**（219 张量全 COPY）。
`β=0` ⇒ 与 teacher 逐位相同；`β>0` 时注入被**可证地约束在 10% 残差比例内**，
RMSNorm 不会溢出 ⇒ logits finite ⇒ 不再产生 `-1` 哨兵 / 空串。
β 是 [0,1] 无量纲比值，不是 v8.3 的硬编码系数，也不是 clip workaround。

⚠️ 工程 v8 是 hybrid；ICML 稿子刻画的是无 softmax 的静态核构造。
**两者互不作为对方的证据。**

## 29 项特性（含 2 项明确未继承）

`20 RUNNABLE`（静态核 / FFT–IIR 对偶 / 因果 / outer-product / O(1) 状态 /
role bit / β 门 / stream 等价 / 零填充 / lm_head / **F25 尺度契约 / F26 尺度不变
role / F27 解码守卫** …）合并为 **19 条断言**在真实权重上真跑；
`7 REGISTERED`（输入侧预算 / H6-A4 / 门控巩固 / 撤回登记 / L0–L4 …）为理论前提
或工程契约，结构化继承并声明 scope，**不宣称在 LM 上复现**；
`2 NOT_INHERITED`：**F28 无词表/tokenizer-free 字节级主路径**（README §7 标 ✅，
但 `docs/BYTELEVEL.md` / `tokenization.py` 未推送）、**F29 GlyphStream v2 连续符号接口**
（corpus.txt 标 v2.0-draft、P1 4–6 周计划，本地无实现）。
**当前迁移体用的是有词表路径**：`tokenizer.json` 130560 词表 + embed/lm_head 全 COPY。
`contract.py` 的 I6「tokenizer REAL、`len(tok)==embed.rows`」在**修乱码**上正确，
但它定义的是「有词表兼容」，与「无词表兼容」是相反方向。

V7 真实缺口已补：**role bit** 只在 README 声明、代码里没有 → 注入 24 层；
`alpha=-2` 硬编码 → `beta=0` + 强制 L2 校准；
**乱码**：NaN → `best_i=-1` → 空串，已由尺度契约 + 合法初始 id + 解码守卫闭环。

## 迁移产物

`../output/minicpm5_full_migration/`：

- `student_branch_v8.safetensors` — 216.6 MiB / 144 张量 / 24 层，可加载
- `student_full_manifest.json` — 411 张量完整定义（219 COPY + 72 DERIVE + 120 NEW）
- `config.json` / `replay_full.py` / `full_migration_report.json`

目标机 `cat model_part_* > teacher.safetensors` 后跑 `replay_full.py` 出完整
2.6 GB student，并逐位校验每个 COPY。

## 三条铁律

1. **继承而非强制**：`tie_word_embeddings` 从 teacher 继承；untied 时显式 COPY `lm_head.weight`。
2. **契约而非 best-effort**：tokenizer 必需文件缺一即拒；133 B LFS 指针即拒。
3. **函数等价而非逐位复制**：`copy_bit_identical` 只是 L0 必要条件；L2 必须有 teacher forward。

## 边界

- Qwen4-27B 未发布（2026-11-25）；本轮未访问其任何数据。
- 完整 2.6 GB student 未落盘（磁盘限制）；分支 checkpoint 含真实派生权重，勿公开上传。
- **β=1 尺度契约态已实测会说话**（0 越界 id / logits 全 finite），但它是契约配置，
  **不是 L2 校准后的成品**；无 ppl / KL / 配对 CI / 生成质量 / 能力数字；论文不改稿。
- token 只在进程内存使用，交付物 0 hits；**未删除/修改/推送任何仓库条目**。
