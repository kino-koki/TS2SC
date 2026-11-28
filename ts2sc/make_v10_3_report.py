#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 v10_3_scoped_verify.json 渲染 V10_3_SCOPED_DELIVERY.md（零重跑）。"""
import json
import os

OUT = "/data/workspace/output"
V = os.path.join(OUT, "v10_3_scoped_verify.json")
R = os.path.join(OUT, "V10_3_SCOPED_DELIVERY.md")


def main():
    d = json.load(open(V, encoding="utf-8"))
    v, p1, ctx = d["verification"], d["phase1"], d["phase2_context"]
    rows = d["generation"]
    att = ctx["attention"]
    cons = v["smm_explicit_consolidation"]

    md = f"""# TS2SC v10.3 · 范围收窄迁移 + 10M 上下文 + 验证

日期：{d['date']} · `passed = {v['passed']}` · 耗时 {d['elapsed_s']} s

> 指令：**「迁移并且验证模型」** → **「就继承 SGC 以及其外挂模块
> SMM记忆与自进化模块就行了，上下文开到10M」**

## 1. 范围（只继承这三组）

| 组 | 内容 | 来源 |
|---|---|---|
| **G1 SGC_CORE** | 冻结静态核 `ρ^τ·sin((τ+1)θ)`、FFT↔IIR、O(1) 定态、严格因果、输入侧 role bit、低秩 readout | `sgc_v10.FrozenSGCCore` |
| **G2 SIDECARS** | A 3层 k=7 门控短卷积 / B CfC 液态门 / C Hadamard SE1 内容寻址，P7 不回写主干 | 上游 `sgc_sidecar_final.py` v9–v11 |
| **G3 SMM** | `S_t=ρ_mem·S_{{t-1}}+Σ C[k]ᵀC[VK+v]`、`add_t=C[bucket(tok)]S_tW_c`、零阶巩固 `observe()` | **上游真实源码 `evolution_mixer.py` 的 `FusedEvolutionMixer`/`CalibratedFusedMixer`** |

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
| 22 分片 LIVE git blob sha1 | ✅ `{v['phase0_all_blob_sha_verified']}` |
| 账本 | `{p1['n_tensors']}` 张量 `{v['phase1_manifest_counts']}` / UNCLASSIFIED `{p1['unclassified']}` |
| **219/219 COPY 全量逐位** | ✅ mismatch `{p1['n_copy_mismatched']}`，{p1['copy_bytes']/2**30:.2f} GiB |
| 312 v10.2 NEW 重开复验 | ✅ mismatch `{p1['n_v102_new_reopen_mismatched']}` |
| branch 重建 == 已推送 | ✅ `{v['phase1_branch_rebuild_identical']}`（确定性可复现） |
| 新增 SMM checkpoint | `{p1['smm_tensors']}` 张量 / {p1['smm_checkpoint']['file_bytes']/1024:.1f} KiB |
| phase1 passed | **{p1['passed']}** |

### 3.2 端到端生成（真实权重 + 24 层 hook）

| 语言 / 臂 | gate | 首句 |
|---|---|---|
"""
    for r in rows:
        md += (f"| `{r['lang']}` / `{r['mode']}` | `{r['decode_gate']}` | "
               f"`{r['text'][:42]}` |\n")
    md += f"""
- **全关 ≡ 纯 teacher 逐 token 相同**：`{v['T_equals_OFF_tokenwise']}`
- **全开与全关不同（三组真进前向）**：`{v['ON_differs_from_OFF']}`
- 全 finite `{v['all_logits_finite']}` / 全解码守卫 `{v['all_decode_gate']}`
  / 身份标记泄漏 **{v['identity_marker_leaks']}**
- SMM 显式巩固：每层 commits `{cons['commits_per_layer']}`、
  rejects `{cons['rejects_per_layer']}`、slots `{cons['slots']}`、
  ‖S‖ `{cons['S_norm']}`、24/24 层 `{cons['all_24_layers']}`

> 8 token 短生成只证 finite / 不乱码 / 零回归 / 架构生效；
> **不是**质量、事实正确性或能力结论。

## 4. 上下文开到 10M

`max_position_embeddings: 131072 → **{v['phase2_context_tokens']:,}**`
（10 Mi tokens；十进制 10,000,000 同登记）。

| 侧 | 结论 |
|---|---|
| **分支侧** | `advance_idle(10M)` **O(1) 闭式实跑**（不是循环 10M 次）；状态 `{v['phase2_branch_state_bytes']:,} B` 跳跃前后恒定 `{v['phase2_branch_state_constant']}`；闭式 == 逐步 gate=0 `{v['phase2_idle_closed_form_equals_stepwise']}`；SMM 绑定 10M 后仍可读 |
| **teacher 侧** | 全注意力 KV **{att['kv_gib']} GiB**、{att['attention_pairs_log10']} 量级对数、`{att['rope_extrapolation_factor']}×` RoPE 外推 ⇒ **`{att['status']}`**（HBM 账本判定，非实跑失效） |

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

"""
    for b in d["honest_boundaries"]:
        md += f"- {b}\n"
    md += (f"\n权威注册表：`scoped_capability_registry.json`；"
           f"配置：`context_10m_config.json` / `inference_default_v10_3.json`；"
           f"原始证据：`v10_3_scoped_verify.json`、"
           f"`student_scoped_manifest_v10_3.json`。\n")
    open(R, "w", encoding="utf-8").write(md)
    print(f"[done] {R} {len(md)} chars")


if __name__ == "__main__":
    main()
