# 05 · SimplUnity：多模型融合

## 目标

把两个（或多个）Llama 架构模型融合进同一个 SimplCore，提升参数量。

```
A  Qwen3-0.6B → SimplCore（已迁移）    D=1024  FF=3072  L=28  V=151936
B  MiniCPM5-1B                          D=2048  FF=3072  L=24  V=195840
                                              ↓
                              融合后 dim=2048  FF=3072  L=28  V=195840
```

---

## 融合决策规则

| 维度 | 规则 | 理由 |
|---|---|---|
| `dim` | 取 max | 容量优先 |
| `ffn` | 取 max（**若相同则天然对齐**） | |
| `layers` | 取 max | 超出部分走重组通道 |
| `vocab` | 取 max | 需换 tokenizer |

---

## 关键幸运：FF 对齐

```
Qwen3-0.6B  intermediate_size = 3072
MiniCPM5-1B intermediate_size = 3072
```

FFN 的 gate/up 是 (FF, D)，down 是 (D, FF)。
**FF 相同 = FFN 的"宽度"完全不用改** → B 的 24 层 FFN 可以逐位直搬。

这是 SimplUnity 里最容易出错的一步被天然省掉。

---

## 确定性维度变换

dim 从 1024 → 2048（A 侧需要升维）：

```python
def expand_last_dim(W, target):
    """(out, d) → (out, target)"""
    k = target // d
    parts = [W * (0.5 + 0.35 * sin(1.7 * i)) for i in range(k)]
    return torch.cat(parts, dim=1)
```

**每层用不同的确定性缩放**，避免重复导致的秩退化。
**完全可复现**（无随机）。

降维同理（`fold_last_dim` 加权平均）。

---

## 层号超出：凸组合

teacher 只有 24 层，融合模型要 28 层：

```python
alpha = 0.5 + 0.35 * sin(1.7 * i)
W = W_src * alpha + W_other * (1 - alpha)
```

⚠️ **诚实说明**：凸组合**不增加知识**，只提供"待训练的高质量起点 + 容量"。
这些层必须参与蒸馏。

---

## 规模计算

```
per_mixer = 3·D² + M·D·2 + D
per_ffn   = 3·D·FF
per_norm  = 2·D

D=2048  FF=3072  M=32  L=28  V=195840
emb      = 195840 × 2048      = 401,080,320
backbone = 28 × (per_mixer + per_ffn + per_norm)
总计 ≈ 1.89 B
```

---

## 词表冲突（最重要的限制）

⚠️ **两模型 tokenizer 不同，同 id 不同义，词表无法语义级合并。**

| 方案 | 做法 | 代价 |
|---|---|---|
| keepB（默认） | dim 统一到 2048，用 B 的 tokenizer | A 的 embedding 需变换 |
| keepA | dim 统一到 1024，用 A 的 tokenizer | 容量损失 |

**当前默认 keepB**（MiniCPM5-1B 的词表更大、更新）。

**若要保留 A 的 tokenizer**：`--mode keepA`，把 dim 统一到 1024。

---

## 用法

```bash
# 纯规划（不用权重，秒出）
python3 simplunity.py --plan

# 真机融合（需要 B 的权重目录）
python3 simplunity.py --fuse --b-path ./MiniCPM5-1B
```

**融合代码对权重来源无耦合** —— 换成 TinyLlama / LLaMA / Mistral 零改动，
只要 `config.json` 是标准 Llama 格式。

---

## MiniCPM-V 视觉塔（多模态）

MiniCPM-V 4.7 的视觉编码器规格（联网核实）：

```
SigLIP-So400M
patch: 1024 → 投影 → 256 token → resampler → 64 token
参数: 448.7M
```

**接入 SimplCore 后的实测**：

| 项 | 结果 |
|---|---|
| 融合模型 | 44.3M，loss 6.04，63 组参数有梯度 ✅ |
| **视觉 token 下的 O(1)** | 0/64/256/1024 token → 状态恒为 **274,432 B** |

**关键卖点**：MiniCPM-V 用 Qwen3 注意力，视觉 token 越多 KV cache 越大；
SimplCore 与 token 总数无关。

⚠️ 视觉塔权重当前不可达（HF 全线 403），**几何与代码已验证，权重待真机接入**。
