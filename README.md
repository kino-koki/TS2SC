# TS2SC — Transformer → SimplCore 迁移技术栈

> **A2.0 协议**：把任意标准 Llama 架构（Transformer）预训练模型，迁移到 SimplCore
> 谱卷积（SGC）架构，**保留预训练知识**并**继承 O(1) 流式状态 / 次二次复杂度 / 长度外推**。

```
┌─────────────────────────────────────────────────────────────────┐
│  输入：任意 LlamaForCausalLM 预训练权重（Qwen3 / MiniCPM5 / LLaMA …）│
│  输出：SimplCore 权重，O(1) 状态，知识继承                          │
└─────────────────────────────────────────────────────────────────┘
```

**当前实测基线（Qwen3-0.6B → SimplCore 0.6646B）**

## 最新：SimplUnity v2 融合完成

**MiniCPM5-1B × Qwen3-0.6B → SimplCore，1,438,585,652 参数（1.439 B）**

| 指标 | 数值 | 说明 |
|---|---:|---|
| **逐位真实权重** | **1,175,062,016（81.8%）** | 263 张量 `torch.equal` 全等，缺失 0 不一致 0 |
| O(1) 流式状态 | 每 batch 内跨 L 唯一状态数 = 1 | ✅ 与序列长度无关 |
| @4M token | **20.41 MiB** vs MiniCPM5 KV **91.6 GiB** | |
| 流式/并行对偶 | max\|Δ\| = **1.01e-06** | ✅ < 4e-5，定理 3 首次在真权重上通过 |
| 权重下载 | [Release v1.0-unity](https://github.com/kino-koki/TS2SC/releases/tag/v1.0-unity) | |

详见 [05b](docs/05b-SimplUnity-v2实测.md)。

---

## Qwen3-0.6B 单模型迁移基线

| 指标 | 数值 | 说明 |
|---|---:|---|
| 迁移参数 | 664,607,516 | |
| **逐位保留原权重** | **663,544,832（99.84%）** | FFN / embedding / norm 原样搬 |
| FFN 保真度 | **余弦 1.0000** | 知识主体完整继承 |
| 裸迁移 mixer 保真度 | 余弦 0.1024 | 唯一失真点 |
| 蒸馏后 mixer 保真度 | **余弦 0.6659**（195 步，仍在上升） | 起始 0.0757 |
| O(1) 流式状态 | **274,432 B 恒定** | 跨 batch × 跨长度 |
| 对照 Qwen3 KV @1M token | 106.8 GiB vs **0.262 MiB** | |

---

## 快速开始

```bash
git clone https://github.com/kino-koki/TS2SC.git && cd TS2SC
pip install torch safetensors

# ① 规划（纯计算，不加载权重，秒出）
python3 scripts/simplunity.py --plan

# ② 迁移（需要本地 teacher 权重）
python3 scripts/migrate_real.py --teacher-path ./Qwen3-0.6B

# ③ 缓存真实激活（一次前向）
python3 scripts/real_act_distill.py cache

# ④ 逐层蒸馏 mixer（可续跑）
python3 scripts/real_act_distill.py train      # RA_LAYER=0..L-1

# ⑤ 端到端验证
python3 scripts/real_act_distill.py eval
python3 scripts/test_capability.py
```

---

## 文档导航（A2.0 协议）

| # | 文档 | 内容 |
|---:|---|---|
| 00 | [协议总览](docs/00-协议总览.md) | A2.0 五阶段流水线、术语、设计哲学 |
| 01 | [架构映射](docs/01-架构映射.md) | 张量级对应表、三种通道（直搬/变换/重组） |
| 02 | [核拟合](docs/02-核拟合.md) | Attention→SGC 闭式拟合、基函数、残差 |
| 03 | [蒸馏恢复](docs/03-蒸馏恢复.md) | **为什么必须蒸馏**、真实激活缓存、可续跑训练 |
| 04 | [能力验证](docs/04-能力验证.md) | 五项测试、判读标准 |
| 05 | [SimplUnity 融合](docs/05-SimplUnity融合.md) | 多模型合体、参数量提升 |
| 05b | **[SimplUnity v2 实测](docs/05b-SimplUnity-v2实测.md)** | **MiniCPM5-1B × Qwen3-0.6B → 1.439B，81.8% 逐位真权重** |
| 06 | [踩坑记录](docs/06-踩坑记录.md) | 静默 bug 全集（不报错但毁结果） |
| 07 | [API 参考](docs/07-API参考.md) | 全部脚本参数与返回值 |
| 08 | [实测数据](docs/08-实测数据.md) | 所有实验原始数字 |

---

## 核心结论（先看这个）

### ① 知识继承是真实的，但有边界

```
FFN（66% 参数）  → 余弦 1.0000   ✅ 完整继承
mixer（13% 参数）→ 余弦 0.1024   ❌ 结构性失真
```

**机理**：注意力是**输入相关**（content-based）算子，SGC 是**输入无关**的线性卷积核。
核拟合只能复现**平均位置分布**，复现不了逐 token 的内容路由。
**这是数学上的结构性损失，不是 bug。**

### ② 蒸馏能救，且很省

```
layer 0：起始 0.0757 → 195 步 0.6659（曲线仍在上升）
只训 mixer（13.4%），86.6% 参数完全冻结不参与反向
```

### ③ SimplCore 特性完整继承

O(1) 状态、长度外推、次二次复杂度 —— **与知识继承互不干扰，全部实测通过**。

### ④ 裸迁移 ≠ 可用模型

裸迁移端到端余弦约 0.057。它是一个**已装好 86.6% 真知识、只差 13.4% 待训**的
初始化，不是成品。**必须走蒸馏阶段。**

---

## 目录结构

```
TS2SC/
├── README.md               本文档
├── docs/                   A2.0 协议文档（00-08）
├── scripts/
│   ├── migrate_real.py     迁移主程序（Qwen3 → SimplCore）
│   ├── real_act_distill.py 真实激活缓存 + 逐层蒸馏 + 端到端验证
│   ├── test_capability.py  五项能力验证
│   ├── diag_gap.py         失真来源定位（mixer vs FFN）
│   ├── simplunity.py       SimplUnity 多模型融合
│   ├── fast_distill.py     轻量可续跑蒸馏（高斯输入版）
│   ├── distill_mixer.py    单层蒸馏（对照用）
│   └── fix_coef.py         coef 形状 bug 修复验证
├── A2.0协议.md             协议速查卡（单页）
└── CHANGELOG.md
```

---

## 支持哪些 teacher

| 模型 | 架构 | 状态 |
|---|---|---|
| Qwen3-0.6B / 1.7B | LlamaForCausalLM | ✅ 已实测 |
| MiniCPM5-1B | LlamaForCausalLM | ⏳ 融合脚本就绪，权重待接入 |
| MiniCPM-V（视觉塔） | SigLIP-So400M | ⏳ 几何已验证（1024→256→64） |
| TinyLlama / LLaMA / Mistral | LlamaForCausalLM | ✅ 命名兼容，无需改代码 |

**判定方法**：`config.json` 里 `architectures == ["LlamaForCausalLM"]` 且权重键名是
`model.layers.{i}.self_attn.*` / `mlp.*` 的，均可直接迁移。

---

## License

Apache-2.0。

## 引用

```
青小团 / 六安市青奈科技有限责任公司
TS2SC: Transformer → SimplCore Migration Stack, A2.0 Protocol
https://github.com/kino-koki/TS2SC
```
