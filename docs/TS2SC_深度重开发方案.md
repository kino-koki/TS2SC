# TS2SC 深度重开发方案 · Qwen4-27B 迁移就绪

日期：2026-11-25 · 状态：**T-minus preflight，不是迁移结果**

---

## 结论先行

方向是对的，而且比自训练更省：**不去预训练，只做权重级迁移 + 函数级对齐**，费用相对从零训练几乎可忽略。「发布后立即开始完整迁移」也是正确的节奏——因为现在是唯一的窗口期，能做的不是迁移本身，而是把 TS2SC 重做成 **config-driven + 契约驱动 + 逐层函数对齐**，让发布日 T+0 只需要填一份真实 config、拉权重、跑门禁，而不是现场改硬编码。

但有一句必须先说清：**Qwen4-27B 今天还没发布**。它在 2026-09-22 云栖大会上作为开放权重档被官宣（同台 Max / Flash / Plus），至今无模型卡、无权重、无 config、无 tokenizer、无许可证、无发布日期；Qwen Code 的 `qwen4*` → 1M token 回退值只是代码线索，不是规格 [citation:11][citation:7]。已发布的是 **Qwen4-Max**（2026-11-16，2.4T 总参 MoE、每 token 激活 71B、QSA 稀疏注意力、原生视觉、百万上下文），阿里云同一份官方稿写明「Qwen4 系列中规模较小的开放权重模型亦将于短期内发布」 [citation:3][citation:4][citation:8]。

**所以：现在不碰任何 Qwen4 权重，也不猜任何架构参数。** Max 的 QSA/MoE/1M 只能作为预案线索，不能写进 spec。

---

## 一、源码已拉齐，缺口如实说

本地已有两份镜像，我逐文件核过：

| 位置 | 内容 |
|---|---|
| `simplcore_src/` | V7 生产链路（`migrate_qwen3.py` / `modeling_simplcore_v7.py` / `train_simplcore.py` / `verify_arch.py` / `contract_check.py` / `bpe.py` / `numpy_lm.py`）、README、验收文档、5 个 TS2SC/VL47 报告、`MANIFEST.json` 哈希清单、`final_arch_verify.json` |
| `src_probe/simplcore_git/` | 带 `.git` 的仓库镜像，含 `.archive/` 历史文档、legacy tests、`PAPER_TS2SC_ADDENDUM.md`、SMM/SGC 融合理论文档 |

**不在本地的**：公开 TS2SC 仓的 `scripts/migrate_real.py` / `real_act_distill.py`（`TS2SC_QWEN3_VERIFY.md` B.0 记录过它们的硬编码 Qwen3-0.6B 配置与 `SRC=/data/workspace/mig/model.safetensors`），以及 Qwen3 完整 4.06 GB 权重。历史环境里 huggingface / PyPI / GitHub Release asset 全部 403，本轮不重试也不伪造。交付的源码基线包只含代码、文档与 JSON，不含权重字节。

---

## 二、「只会吐乱码」不是玄学：源码级十大根因

我对本地 **真实 V7 文件**做了字面审计，**10/10 全部命中**（`preflight_report.json` part A）。这不是复述历史报告，是逐行在源码里找到的：

| # | 根因 | 源码位置 / 证据 |
|---|---|---|
| R1 | **输出头被丢掉**：只 COPY `embed_tokens.weight`，没写 `lm_head.weight`，却把 `tie_word_embeddings` 强制成 `True`；Qwen3-1.7B 实为 untied | `migrate_qwen3.py` L93-95, L161 |
| R2 | tokenizer 复制是 **best-effort**：`if os.path.exists(...)` 静默跳过，缺文件也照发 | `migrate_qwen3.py` L171-175 |
| R3 | 自造 **2048/1024 词表子集 BPE** 配 151936 行 embedding；`decode` 的 `except` 直接返回 byte-to-unicode 字符串 | `bpe.py` `QwenBPESubset` |
| R4 | 验收只测「非退化」：`n/max_run/rep3/distinct1`。拉丁扩展字符乱码 distinct 高、rep3 低，**必然 PASS** | `verify_arch.py::gen_quality` |
| R5 | NEW 参数全零 + `alpha=-2` → `sigmoid≈0.119`、`wgate=0.5`，mixer 贡献约 `0.06×ys`，注意力能力被结构性压掉 | `migrate_qwen3.py` L124-132 |
| R6 | GQA 的 k/v 从 `(d/2,d)` **零填充**到 `(d,d)`，`R=2` 切两 slot → **slot 1 的 K/V 恒零**，半个状态是死的 | `migrate_qwen3.py` L114-121 |
| R7 | `bos=151643 / eos=151645 / max_pos=40960 / R=2 / self_attn.*` 全硬编码，Qwen4 若 QSA/MoE 必失效 | `migrate_qwen3.py` 全文 |
| R8 | `q_norm/k_norm` 直接 `SKIP`，无功能对应物 | `migrate_qwen3.py` L134-139 |
| R9 | 断言只有 `keys_differ / no_self_attn / copy_bit_identical`，**没有 teacher forward、没有 KL、没有 ppl** | `migrate_qwen3.py` L186-199 |
| R10 | 失败早被登记却没闭环：W14 记「RoPE 绝对位置 / mask 广播 / embedding-lm_head 基址」四处缺陷；VL47 记 `rep3 0.8125`、`otionalotional… 0.9286`、「中国的首都是 → 城市，有，北京，北京，北京…」 | `PAPER_TS2SC_ADDENDUM.md` W14、`TS2SC_VL47_PREFLIGHT.md` §5 |

最刺眼的是 R10 那句原始记录：**「知识仍在（FFN 逐位迁移 cos 1.0000），但循环与语义错乱」**。这说明从前的迁移不是「没学到」，而是 **权重搬对了、接口接错了**——正好对应通用工程里「词表与 embedding 行数/id 映射不对齐」和「UTF-8 多字节被逐 token 截断」两类乱码机制 [citation:24][citation:21]。

这也解释了为什么「融合」路线才是对的：R5/R6 不修，SGC 卷积注意力只是**空挂**在 state_dict 里，根本没融进前向。R1/R3 不修，融合得再好，输出头也是乱码。

---

## 三、v8 重开发架构：五层，三条铁律

```
L0 spec.py        TeacherSpec：零硬编码，缺字段即 T0 blocker
L1 contract.py    tokenizer↔embedding 绑定契约（乱码总闸）
L2 provenance.py  逐张量 COPY/DERIVE/NEW 账本 + shape/dtype/lm_head
L3 align_accept   逐层 hidden cos / logit KL / top-1 一致率
L4 acceptance     配对 teacher 对照 + Wilson CI + 中文可读率 + mojibake 率
```

**三条铁律**

1. **继承而非强制**：`tie_word_embeddings` 从 teacher 继承；untied 时显式 `COPY lm_head.weight`。slot 数由 `hidden_size / (num_key_value_heads × head_dim)` 推导，绝不写 `R=2`；检测到 MoE 直接要求 per-expert 映射声明后才放行。
2. **契约而非 best-effort**：tokenizer 六文件逐个 REQUIRED；133 B LFS 指针即拒发；`len(tokenizer) == embed.rows`；五条往返（中/英/混排/emoji/多字节边界）+ UTF-8 流式缓冲 + chat template + 全部 special id 在界内。任一 FAIL 就**不落 student checkpoint**。
3. **函数等价而非逐位复制**：`copy_bit_identical` 降级为 L0 的必要条件。L2 必须有同条件 teacher forward：逐层 hidden cos、logit KL、top-1 一致率、全展开 == O(1) 状态流式、且 attention 路径贡献经校准非零（`alpha` 不许停在 −2）。

配套修法：QK-norm 从 `SKIP` 改成 `DERIVE FOLD_INTO_WqWk`；GQA 用真实 head 展开代替零填充；RoPE/mask 缺陷在 L0/L1 做成硬门禁，不再留在撤回登记里。

**验收范式也换掉**：`gen_quality` 的 `non_degenerate` 被 `mojibake_rate`（U+00A0–U+024F 泄漏检测）+ 配对 Wilson 95% CI 取代。自测里我特意喂了 `¤Ĳ城市 «闪店 ±展 北京北京` 这种历史签名文本——**旧的 `non_degenerate` 会判 PASS，v8 判 FAIL**（实测 `avg_mojibake_rate=0.25`，门禁拒绝）。

---

## 四、T+0 执行序列（发布日当天）

```
1. spec.from_dir(released)            缺字段 → 停，不猜
2. static_contract(teacher, student)  任一 FAIL → 停，不发权重
3. classify(real_safetensors_index)   UNCLASSIFIED > 0 → 停
4. runtime_contract(real tokenizer)   往返 / UTF-8 / chat template → 停
5. 迁移落盘 + COPY 逐位 + DERIVE 变换登记
6. L2 对齐：teacher forward 同条件 → hidden cos / KL / top-1
7. L3 配对：N≥200 prompts，同 seed / 采样 / tokenizer
8. L4 才允许讨论「原生级体验」
```

顺序不可跳。**第 4 步之前产出的任何"迁移完成"都不算数**——这正是 V7 的 `Stage0 PASS + Verify PASS ⇏ 模型会说话`（`MIGRATION_TIMING_ACCEPTANCE.md` §2.5）被制度化后的样子。

---

## 五、「原生级体验」必须可证伪

不说「感觉一样」，说这五条：

- **L1**：标准 tokenizer 文件、标准 HF/vLLM 加载路径，**服务端零自定义 BPE / 零 id 重映射**；
- **L2**：逐层函数对齐达标，logit top-1 一致率过声明阈值；
- **L3**：配对生成中文可读率 ≥ 阈值、`mojibake_rate ≤ 0.01`、EOS 正常终止、chat template 身份一致、no-regression 的 95% CI 下界 ≥ 0；
- **L4**：上下文窗口、special tokens、停止行为全部来自**真实 config**，不是推测；标准推理框架直接跑通；
- **红线**：L3 是 **no-regression，不是 improvement**；不写 SOTA / 不写「比 teacher 强」。

---

## 六、费用与硬件的实话

「省去极大费用」成立，但成立的是**相对从零预训练**，不是「零成本」：

- 不预训练 → 省掉的是 27B 级从零训练的数千 GPU 小时；
- 但 **L2/L3 必须有 teacher 同条件 forward 对照**：27B BF16 权重约 54 GB，加激活/KV 需要单卡 80GB 或 4 卡分片/量化推理，这部分显存省不掉；
- 轻量对齐（只训 SGC/local 路径）仍需要少量算力，但不是全参训练；
- 真正的省钱来自 **不做无谓重跑**：T−∞ 把契约和门禁做完，发布日一次通过，比发布后反复「改硬编码→吐乱码→再改」便宜得多。

另外提醒一次：你附的 `icml2027.zip` 解析服务仍取不回二进制包（本轮同样失败）。这不影响本方案；上轮已给过兼容层与三步替换说明，需要时再补即可。

---

## 七、与论文边界

`PAPER_TS2SC_ADDENDUM.md` 已经裁定：工程验证**不进投稿包**，稿件定位是 boundary-characterization，混入权重迁移账本会被读成拼盘。本轮维持：

- v8 的 L0–L4 是**工程交付**，不是定理、不是 LM 结论；
- 不写「Qwen4-27B 迁移后保住能力」——今天没有权重、没有 config、没有一次 forward；
- 不写任何 ppl / 生成样本 / logit-KL 数值 / 吞吐 / 发布日期预测；
- 后续若真跑通 L3，最多在 Limitations 增补一句带 scope 的结构性观察。

---

## 八、交付与本轮真实自检

| 交付 | 内容 |
|---|---|
| `TS2SC_深度重开发方案.md` | 本文 |
| `TS2SC_源码基线.zip` | 本地拉齐的全部 TS2SC / SimplCore 源码、文档、JSON（无权重字节） |
| `TS2SC_v8_骨架.zip` | `spec / contract / provenance / align_accept / preflight` + README + 自检报告 |

**本轮真实跑过的**（无 torch、无 GPU、无 Qwen4 数据）：

```
[A] 真实 V7 源码字面审计：10/10 历史根因命中
[B] 合成 fixture 自测：
    static_contract = PASS
    三个负对照（LFS 指针 / 缺文件 / 词表行数不符）= 全部被拦
    provenance：UNCLASSIFIED=0、lm_head 已登记、QK-norm 已折叠
    乱码门禁：旧 non_degenerate 判 PASS 的文本，v8 判 FAIL
```

`preflight.py` 纯标准库，任何机器可直接复跑：

```bash
cd ts2sc_v8 && python preflight.py
```

**没有做的事，如实列着**：未访问 Qwen4-27B 任何权重/config/tokenizer；未跑 teacher forward；未做蒸馏或对齐训练；未产出任何能力数字。这些全部是 T+0 的活，不是今天能补的。
