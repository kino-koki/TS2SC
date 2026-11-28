# RETIRED · 身份微调资产清单

按用户指令「完全放弃身份微调」，以下资产**整层退役**：

  - 不加载：交付推理路径 `beta=1.0, identity_adapter=none, logit_bias_fn=none`
  - 不删除：用户明确要求「不要清理删除任何条目，我自己会动」
  - 不作为交付：新仓 `MiniCPM5-1B-TS2SC` 的交付物清单不含任何一项

12 轮实验的完整记录、判据修复、两难证明保留在
`IDENTITY_FINAL_VERDICT.md` / `IDENTITY_CEILING_REPORT.md` /
`identity_all_rounds_summary.json` / `identity_semantic_reaudit.json`，
不改写不抹平，供后续 GPU LoRA SFT 参考。

原因：CPU 环境下输出层微调只能改 302 行 lm_head，既不能让 24 层内部
状态相信新身份，又会在生成路径引入偏离 —— 几乎只有污染、没有收益。
