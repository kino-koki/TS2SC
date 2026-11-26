# 口径澄清 · Apache-2.0 开源发布（不是「A2.0 协议」）

日期：2026-11-26 · 适用仓库：`kino-koki/Tuan-0.1-TS2SC-1B` / `kino-koki/TS2SC`

## 一次必须纠正的误读

用户说「使用 A2.0 协议公开仓库」，指的是 **Apache License 2.0 开源许可证**。
上一轮被读成了 `kino-koki/TS2SC` 仓里那份 `A2.0协议.md`（Architecture 转换 v2，
S1–S5 权重迁移流水线），并据此生成了 `A2.0_验收核对报告.md`。

**两份文档都存在，但说的完全是两件事：**

| | `A2.0协议.md`（架构） | **Apache-2.0（本轮）** |
|---|---|---|
| 性质 | 内部迁移流水线速查卡 | OSI 批准的开源许可证 |
| 管什么 | 权重怎么搬 / 蒸馏 / 验证 | 别人能不能用、能不能改、要不要署名 |
| 关键词 | S1 规划 / S3 激活缓存 / FFN 余弦 | LICENSE / NOTICE / patent grant / 保留版权声明 |
| 与本轮关系 | 仍有效的架构核对，**不是**本轮依据 | **本轮发布依据** |

那份架构核对报告**保留、不删除**（按用户「不要清理删除任何条目」），
但它不构成开源许可。

## 本轮实际做的

1. `Tuan-0.1-TS2SC-1B` 从 `private=True` 改为 **public**；
2. 新增 `LICENSE` —— 与 `kino-koki/TS2SC` 仓同字节的 **Apache License 2.0 官方全文**
   （11357 B，含 `TERMS AND CONDITIONS` 与附录）；
3. 新增 `NOTICE` —— 版权归属六安市青奈科技有限公司、TS2SC 归属、
   219 个 teacher COPY 张量不在此分发（由 `replay_full.py` 从上游重建）、
   上游模型许可需另阅、品牌词与「上下文理论无限」免责；
4. `README.md` 顶部写明 Apache-2.0 与本仓公开状态；
5. **身份做好之后**才补齐身份资产并公开：`identity_alignment.npz`（v9.0 门控锁定）、
   `identity_sft.npz`（闭式 ridge 训练权重）、三份验证 JSON、`IDENTITY_SFT.md` 协议；
6. TS2SC 源码仓同步（base_tree，**零删除**）。

## Apache-2.0 对被发布内容的硬要求（已满足）

- 随附许可证全文 ✅
- 保留原有版权 / 归属声明 ✅（NOTICE）
- 修改过的文件带显著变更说明 ✅（README 迭代记录 + 各 JSON 的 `version` / `date`）
- 不冒充上游 ✅（219 个 COPY 张量不重分发，manifest 带每个 sha256）
- 专利授权随许可证自动授予 ✅（Apache-2.0 §3）
- **上游模型自身许可另阅** —— 本仓只发 SGC 分支与迁移账本，
  `MiniCPM5-1B` 的权重许可不在本仓 license 覆盖范围内，已写进 NOTICE。

## 边界

- ICML 的 `icml2027.zip` 解析服务仍取回失败，与本次开源发布无关。
- 未删除、未修改任何已有仓库条目；只做新增、覆盖本轮自己产出的路径、
  以及把新建仓库的可见性从 private 改 public。
- 「上下文长度理论无限」仍是品牌陈述，非实测；8M 端到端边界见
  `evidence/context_8m_theory.json`。
- 身份微调是输出层闭式解，不是全量梯度 SFT；无能力 / ppl / SOTA 声明。
