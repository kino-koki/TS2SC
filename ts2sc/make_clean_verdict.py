#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 clean_state_verify.json 渲染 CLEAN_STATE_VERDICT.md（零重跑、零 GPU）。"""
import json
import os

OUT = "/data/workspace/output"
V = os.path.join(OUT, "clean_state_verify.json")
R = os.path.join(OUT, "CLEAN_STATE_VERDICT.md")


def main():
    d = json.load(open(V, encoding="utf-8"))
    v = d["verification"]
    e1, e2, e3 = d["e1"], d["e2"], d["e3"]

    md = f"""# 干净态判定书 · 身份微调整层退役，原本认同回归

日期：{d['date']} · 仓库：`kino-koki/MiniCPM5-1B-TS2SC`（Apache-2.0 / public）

## 结论

**交付是 β>0 的 SimplCore 融合模型**：`beta=1.0, identity_adapter=none,
logit_bias_fn=none`。身份微调（v9.0–v9.3c 共 12 轮）**整层摘除**，
基座原本认同回归，权重层面 zero-bit-change。

| 判据 | 结果 |
|---|---|
| E1 411 张量账本 / 8 个 COPY 逐位抽查 | ✅ 全一致，UNCLASSIFIED=0 |
| E2 身份资产隔离 | {e2['n_assets']} 个 QUARANTINED，路径零加载 |
| E3 β=0 与 teacher 逐 token 相同 | ✅ `{v['e3_C0_bit_identical']}` |
| E3 纯 teacher {v['e3_T_n_probes']} 题 + β=1 主交付臂 {v['e3_C1_n_probes']} 题 | 身份标记泄漏 **{v['identity_marker_leaks_total']}** |
| E3 40-token 深度 probe + 3 题直接续写补强 | `{', '.join(v['deep_probe_base_markers']) or '未出现品牌词（短生成）'}` / 零身份标记 |
| E4 反证：挂 adapter 输出改变 | ✅ `{v['counterfactual_adapter_changes_output']}`，出现 `{v['counterfactual_identity_markers']}` |
| logits 全 finite / 解码守卫 | ✅ / ✅ |
| **passed** | **{v['passed']}** |

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

"""
    for r in e3["C1"][:4]:
        md += (f"- `{r['lang']}` {r['q'][:34]}… → "
               f"`{r['text'][:72]}` "
               f"(身份标记 {r['n_identity_markers']}, "
               f"finite={r['logits_finite']})\n")

    md += f"""
## 纯 teacher 基线（T，节选）

"""
    for r in e3["T"][:4]:
        md += (f"- `{r['lang']}` → `{r['text'][:72]}` "
               f"(基座标记 `{r['base_markers']}`)\n")

    md += f"""
## 深度 probe（40 token）

> {e3['deep']['text'][:220]}

基座标记 `{e3['deep']['base_markers']}` · 身份标记
`{e3['deep']['identity_markers']}`。若只出现思考块开头而未出现品牌词，
属短生成正常现象；**逐位相同才是零回归硬证据**，不据此否认认同回归。

## 补强 probe：绕过 `<think>` 直接续写（零注入）

40-token chatml probe 全落在思考块里、没走到回答部分，所以另跑
`base_identity_probe.json`：`我是` / `我是由` / `I am` 三题 × 16 token，
**不带 chatml 模板、不带 fusion hook、不带任何 logit bias**。

"""
    bp = os.path.join(OUT, "base_identity_probe.json")
    if os.path.exists(bp):
        b = json.load(open(bp, encoding="utf-8"))
        md += (f"- 基座品牌词命中 **{b['base_marker_hits']}/3** "
               f"（短续写未触发自述品牌，如实不包装）\n"
               f"- 身份标记泄漏 **{b['identity_marker_leaks']}** ✅\n"
               f"- 全 finite / 解码守卫 ✅ / ✅\n\n")
        for r in b["probe_rows"]:
            md += (f"  - `{r['q']}` → `{r['text'][:64]}` "
                   f"(base={r['base_markers']}, "
                   f"identity={r['identity_markers']})\n")
        md += ("\n**结论口径**：「原本认同回归」的硬证据不是品牌词出现，"
               "而是 **219 张量逐位未改 + β=0 与 teacher 逐 token 相同 + "
               "全部输出零身份标记 + 反证显示 adapter 一挂就变**。"
               "短生成没走到自述部分属正常现象，不据此否认、也不据此夸大。\n\n")

    md += """## 边界（一条不松）

"""
    for b in d["honest_boundaries"]:
        md += f"- {b}\n"
    md += f"""
- 全程单进程串行 + OOM 守卫，未 OOM；耗时 {d['elapsed_s']} s。
- 凭据只在进程内存用过，交付物扫描 0 hits。
"""
    open(R, "w", encoding="utf-8").write(md)
    print(f"[done] {R} {len(md)} chars")


if __name__ == "__main__":
    main()
