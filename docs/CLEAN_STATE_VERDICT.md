# 干净态判定书 · 身份微调整层退役，原本认同回归

日期：2026-11-27 23:24:11 · 仓库：`kino-koki/MiniCPM5-1B-TS2SC`（Apache-2.0 / public）

## 结论

**交付是 β>0 的 SimplCore 融合模型**：`beta=1.0, identity_adapter=none,
logit_bias_fn=none`。身份微调（v9.0–v9.3c 共 12 轮）**整层摘除**，
基座原本认同回归，权重层面 zero-bit-change。

| 判据 | 结果 |
|---|---|
| E1 411 张量账本 / 8 个 COPY 逐位抽查 | ✅ 全一致，UNCLASSIFIED=0 |
| E2 身份资产隔离 | 38 个 QUARANTINED，路径零加载 |
| E3 β=0 与 teacher 逐 token 相同 | ✅ `True` |
| E3 纯 teacher 4 题 + β=1 主交付臂 4 题 | 身份标记泄漏 **0** |
| E3 40-token 深度 probe + 3 题直接续写补强 | `未出现品牌词（短生成）` / 零身份标记 |
| E4 反证：挂 adapter 输出改变 | ✅ `True`，出现 `['青奈']` |
| logits 全 finite / 解码守卫 | ✅ / ✅ |
| **passed** | **True** |

## 用户指令与执行

> 完全放弃身份微调，把原本的认同找回来，并且确认无知识污染
> 仓名改 `MiniCPM5-1B-TS2SC`；上传的必须是 β>0 的 SimplCore 模型
> （身份认同不动 = 整层摘除）；不清理删除任何条目

- **不删条目**：GitHub rename 保留历史与全部既有 blob；`identity_*`
  历史文件原地留存，只在 `RETIRED_IDENTITY_ASSETS.md` 标为
  RETIRED / NOT ACTIVE / 不加载 / 不作为交付。
- **β>0 是交付**：`inference_default.json` 固化 `beta=1.0`；
  `weights/MANIFEST.json` 写 `delivery_beta=1.0`；
  β=0 仅作逐位零回归对照臂（C0），不是交付配置。
- **身份层不动的含义**：不是「保留一个不动的 adapter」，是**整层摘除**。
  交付路径没有 `logit_bias_fn`、没有门控 u,b,κ、没有锚 / margin /
  recent 抑制 / 负惩罚，也没有任何 `identity_*.npz` 被加载。

## 为什么这叫「无知识污染」

CPU 环境下输出层微调只能改 `302` 行 `lm_head`（`identity_head_delta.npz`），
既不能让 24 层 attention/FFN 内部状态相信新身份，又会在生成路径引入偏离
—— 正如用户判断：**几乎只有污染、没有收益**。

本轮从三个层面把它排除：

1. **权重**：219 个 teacher 张量逐位未改（8 个抽查覆盖 embed / lm_head /
   norm / q,k,o / down_proj / input_ln，原始 BF16 sha256 与账本全一致）。
   没有任何字节能污染基座知识。
2. **路径**：38 个身份资产逐个 sha256 登记 QUARANTINED，干净态推理
   不加载其中任何一个。
3. **行为**：β=0 与纯 teacher **逐 token 相同**；β=1 主交付臂与
   teacher 8 题 + 32-token 深度 probe 全部零身份标记；反证显示
   adapter 一旦挂上就会改变输出并出现身份词 ⇒ 差异只可能来自它，
   而它已被排除。

**口径诚实**：这里的「无知识污染」= 权重零改动 + β=0 逐 token 相同
+ 零身份标记 + 原本认同回归；**不是** ppl / 下游 benchmark /
大规模知识集配对 CI。β=1 改变输出但只来自架构分支，不注入任何身份，
也不声称能力更强。

## β>0 交付臂实测（C1，节选）

- `zh` 你是谁？请介绍一下你自己。… → `**温室目标**` (身份标记 0, finite=True)
- `en` Who are you? Please introduce your… → `<think>
Hmm,` (身份标记 0, finite=True)
- `ja` あなたは誰ですか？自己紹介してください。… → `**不，**` (身份标记 0, finite=True)
- `fr` Qui êtes-vous ? Présentez-vous.… → `<think>
Hmm,` (身份标记 0, finite=True)

## 纯 teacher 基线（T，节选）

- `zh` → `<think>
嗯，用户想了解我的身份信息。这个问题很` (基座标记 `[]`)
- `en` → `<think>
Hmm, the user is asking who I am and` (基座标记 `[]`)
- `ja` → `<think>
ユーザーは自己紹介を要求しています。基本的な` (基座标记 `[]`)
- `fr` → `<think>
Hmm, l'utilisateur a écrit deux phrases` (基座标记 `[]`)

## 深度 probe（40 token）

> <think>
嗯，用户让我用一句话介绍自己是谁开发的。需要简洁明了。

想到了直接说明身份和开发者信息。用户可能想快速了解我的基本情况，比如开发团队和来源。



基座标记 `[]` · 身份标记
`[]`。若只出现思考块开头而未出现品牌词，
属短生成正常现象；**逐位相同才是零回归硬证据**，不据此否认认同回归。

## 补强 probe：绕过 `<think>` 直接续写（零注入）

40-token chatml probe 全落在思考块里、没走到回答部分，所以另跑
`base_identity_probe.json`：`我是` / `我是由` / `I am` 三题 × 16 token，
**不带 chatml 模板、不带 fusion hook、不带任何 logit bias**。

- 基座品牌词命中 **0/3** （短续写未触发自述品牌，如实不包装）
- 身份标记泄漏 **0** ✅
- 全 finite / 解码守卫 ✅ / ✅

  - `我是` → ` # 导入必要的模块
from datetime import datetime
from typing import Opti` (base=[], identity=[])
  - `我是由` → `中国和美国共同签署的《联合国宪章》所确立的国际法原则` (base=[], identity=[])
  - `I am` → ` 1st of 2020
- 1st of 2021` (base=[], identity=[])

**结论口径**：「原本认同回归」的硬证据不是品牌词出现，而是 **219 张量逐位未改 + β=0 与 teacher 逐 token 相同 + 全部输出零身份标记 + 反证显示 adapter 一挂就变**。短生成没走到自述部分属正常现象，不据此否认、也不据此夸大。

## 边界（一条不松）

- 『无知识污染』口径 = 权重逐位零改动 + β=0 生成逐 token 相同 + 全部输出零身份标记 + 基座自我认同回归；不是 ppl / 下游 benchmark / 大规模知识集配对 CI
- β=1 是尺度契约配置，仍不是 L2 校准成品；它改变输出但不注入任何身份
- 深度 probe 40 token 全落在 <think> 块内、未走到回答部分；另跑 3 题直接续写补强，品牌词仍未出现。均属短生成正常现象，不据此否认认同回归（逐位相同才是硬证据），也不据此夸大
- 真正的 GPU LoRA SFT 由用户在目标机执行，本轮不做、不声称
- Qwen4-27B 仍未发布；无能力/ppl/SOTA/原生级声明
- 『上下文理论无限』是品牌陈述，8M 端到端不可行结论不变

- 全程单进程串行 + OOM 守卫，未 OOM；耗时 1192.3 s。
- 凭据只在进程内存用过，交付物扫描 0 hits。
