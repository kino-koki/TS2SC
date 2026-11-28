# TS2SC v10 · 严重污染排查 + SimplCore 云端最新 SGC 外挂同步

日期：2026-11-28 · 上游：`kino-koki/SimplCore` @ `bec5ab8fd7`（2026-11-27T11:26:53Z）
基座：`kino-koki/MiniCPM5-1B` @ `f0019e05f3`（2026-11-01T12:08:57Z）

---

## 1. 「严重污染」是什么，怎么根除的

用户指出上一版存在严重污染。逐项排查后的**真实结论**：

| 嫌疑 | 排查结果 | 处理 |
|---|---|---|
| 权重被身份微调改坏 | 上一版 E1 抽查 8 个 COPY 逐位一致、β=0 逐 token 相同 ⇒ **权重字节没被改** | 仍全部重拉，不复用 |
| 仓库里留着 33 个 `identity_*` 历史文件 | 用户明确「不要清理删除任何条目」⇒ 只能留着，标 RETIRED | 保留，但新交付物**一条都不读** |
| **真正的问题：架构滞后** | 云端 SimplCore 已于 11-27 推 v9–v11「SGC 冻结 + 三旁路外挂」补充包；TS2SC 仍停在 v8 的单层静态核分支（216.6 MiB / 144 张量 / 72 DERIVE） | **本轮同步为 v10** |
| 旧分支的 72 个 DERIVE 依赖 teacher k/v 行映射 | v10 上游冻结 SGC 外挂不消费 teacher k/v，保留 DERIVE 反而是遗留耦合 | v10 账本去掉 DERIVE：`219 COPY + 312 NEW` |

**根除方式（不是打补丁，是隔离重建）**：

- 全新目录 `v10_fresh/`，**零复用**：不 open 任何 `identity_*` / 旧 `student_branch_v8.safetensors` / 旧 manifest / 旧缓存。
- 22 个分片**全部重新从上游下载**，每片验证 `git blob sha1 == LIVE contents API sha`（不是本地文件哈希，是上游实时对象哈希）。22/22 OK，2.01 GiB。
- `config.json` / `tokenizer.json`（9.89 MB 真字节，非 133 B LFS 指针）/ index / chat_template 同批重拉并落 sha256。
- 219 个 COPY 的 sha256 **从本次新拉的字节流式重算**，不复用旧值；8 个抽查逐位一致。
- 312 个 NEW 参数由 `sgc_v10.deterministic_layer_params(seed=20261128+101*l)` 确定性生成，与任何历史产物无关。
- 生成验证路径**没有任何** `logit_bias_fn` / 门控 / 锚 / margin / 抑制 / 负惩罚；`identity_marker_leaks = 0`。

> 诚实口径：这里的「无污染」= 逐片上游 blob sha 校验 + COPY 逐位 + NEW 零复用 +
> 零身份标记；**不是** ppl / 下游 benchmark / 知识集配对 CI。CPU 上的输出层
> 微调确实如用户判断几乎只有污染没有收益，真正的 GPU LoRA SFT 由维护者自跑。

---

## 2. 云端「极大 SGC 外挂更新」到底是什么

上游最新 commit `bec5ab8fd7` 是 **append-only 补充包**
`repro/independent_v9_v11/`，不是新 mixer 替换。核心文件：

| 文件 | 内容 |
|---|---|
| `sgc_sidecar_final.py` | **SGC 冻结主干 + 三旁路外挂**定稿实现 |
| `SGC冻结外挂_超越与不足_完整报告.md` | 37/37 PASS、P7 主干逐位不变、P8 零回写、长度/成本轴 |
| `README.md` | v9–v11 本地极限优化，结论 `COMPLETE_BEYOND=False` |
| `本地极限优化_最终报告.md` / `完全超越_最终边界报告.md` | 停止准则与定义性边界 |
| `simplcore/core/arch/sgc.py` | 主仓 `SpectralGlobalConv` / `fft_causal_conv` / `depthwise_causal_conv`（PyTorch 数值规格） |

### 架构定稿（与上游 §0 一字不改）

**SGC 主干完全冻结**：`ψ_m[τ]=ρ_m^τ·sin((τ+1)θ_m)`、a/b/c 静态 Toeplitz、
FFT↔IIR 双形式、`2·M·D` 定态，一个字节都不改。

**三条旁路并联，只在 readout 拼接，绝不回写主干**：

| 旁路 | 补的能力 | 状态 |
|---|---|---|
| A 3 层 k=7 门控短卷积 | 局部 n-gram / 乘法交互 | `3·6·D` |
| B CfC 液态门（输入依赖 τ） | 不规则 Δt | `D` |
| C Hadamard 正交登记 SE1 | 内容寻址绑定，地址由内容非位置给 | `V×V` |

**纯度闸门**：无 Q/K/V、无 `softmax(QKᵀ)`、无 SDPA、无 KV cache、无位置表、
无 `max_len`；mixer 白名单 `sgc / sgc-nogate / sgc-nolocal / srf`。

### v9 推翻了 v8 的一个论断（重要）

v8 曾写「静态核成对交互 O(M) 且与输入无关，堆 M 永远补不上随输入变化 →
定义性边界」。**该论证只对单层成立**：多层堆叠时第 l 层输入
`U^(l)=A(X^(l))·σ(B(X^(l)))` 由上一层全局混合给出，叠加层间 RMSNorm + FFN
非线性后有效交互变成输入依赖，而核一字未动。所以「极大外挂」= 多层 + 三旁路，
不是把静态核改成输入依赖。

---

## 3. TS2SC 侧怎么同步（`ts2sc/sgc_v10.py`）

上游是 D=64 的 toy 独立复现；TS2SC 要把同一套算子挂到 1B teacher 的 24 层：

```
x (d=1536) --Wdown--> h (64) --+--> FrozenSGCCore(role-injected) --> sgc_out (64)
                               +--> ShortConvSidecar(3,k=7)      --> fA (64)
                               +--> LiquidSidecar(CfC, tau=2)    --> fB (64)
                               +--> HadamardBinding(SE1, V=64)   --> fC (32)
concat[sgc_out; fA; fB; fC] (224) --Wup--> o (1536)

Δ = β·ε·‖x‖·o/(‖o‖+δ)      (T1' 无量纲尺度契约，‖Δ‖≤βε‖x‖)
x_next = x + Δ               (加性融合，teacher 全部功能层逐位保留)
```

低秩投影只是把 d 维 hidden 映到上游算子的 D=64 工作维；四条路径的公式、
初始化分布、`rho/theta/k/layers/V/tau` 全部与上游一致。
旁路 C 的 content key 由 `argmax|h[:32]|` 给出，不由位置给出（SE1）。
`β=0 ⇒ Δ=0 ⇒ 与 teacher 逐位相同`；`β=1` 为交付默认。

**逐文件同步映射**：

| 上游 | TS2SC v10 | 一致性 |
|---|---|---|
| `make_basis / kernel / iir_stream / fft_dual / hop` | 同名同实现 | 逐行 |
| `FrozenSGC` | `FrozenSGCCore`（输入改 hidden h） | 公式/初始化/basis 全同 |
| `ShortConvSidecar` | 同名 | `layers=3, k=7` 全同 |
| `LiquidSidecar` | 同名 | `tau=2.0` 全同 |
| `HadamardBindingSidecar` | 同名 + `observe_content` | `V=64, rho=0.9995` 全同 |
| `SGCSidecarBundle.readout` | `SGCV10Layer.hook` concat | 224 维拼接 |
| `purity_audit` / AST 守卫 | `purity_source_audit` | 白名单/负向变异 |

**v10 实测不变量**（`v10_fresh_verify.json`）：

```
FFT↔IIR 对偶          1.69e−14
SE1 ρ=1 正交零串扰     on=0.0 / off=0.0
SE1 ρ<1 衰减闭式误差   3.33e−16
短卷积流式 vs 整段     8.33e−17
P7 旁路全开关主干不变   max|Δ| = 0.0
纯度 AST               PASS，forbidden_defs=[]
β=0 Δ 严格零向量       True
β=1 尺度契约比值       0.1000000 (= ε)
O(1) 状态              L=1/8/64 恒 46,656 B/层
生成 β=0 / β=1         全 finite / 全解码守卫 / 架构生效 2/2
身份标记泄漏            0
```

**账本**：`531 张量 = 219 COPY + 0 DERIVE + 312 NEW`，UNCLASSIFIED=0，
无遗留耦合。分支 checkpoint 43.7 MiB（v8 是 216.6 MiB，因为不再存 48 个
Wk/Wv 派生矩阵）。

---

## 4. 必须随结论一起读的限定（不删减）

1. 上游 v9–v11 结论是**否定性**的：`MECHANICAL_COMPLETE=False` /
   `HONEST_COMPLETE=False` / `COMPLETE_BEYOND=False`；质量轴只到**稳态 parity**
   （A1 +0.0029、A3 +0.0012 bpc @6000 步），2× 对手步数对照失败；
   A2 的负 delta 全是 L768/L1536 崩坏区伪影。
   **本轮只同步架构能力，不继承任何超越 / SOTA 主张。**
2. 上游语料 240 KB / 62 字符 / D=32 / L≤1536 / ≤6000 步；
   **1B / 中文 / LRA / 长文档 / 多 needle / 层级运算完全未测**。
3. 上游对照臂非完整 Transformer；EFFECTIVE 判据是 v11 事后新增、非预注册；
   `+1.0 bpc` 阈值是工程选择。
4. TS2SC 侧是**纯 numpy CPU 等价实现**，不是上游 PyTorch 源码执行；
   没有跑上游 pytest 套件，没有 GPU / bf16 / 多卡验证。
5. `β=1` 是尺度契约配置（T1'），**不是 L2 校准成品**；24 个 β_l 仍待校准。
6. 生成 probe 只有 4 token（纯 numpy 每 token 重解码 24 层 BF16→f32 极贵），
   只证明 finite / 不乱码 / 架构生效 / 零身份标记，**不是质量结论**。
7. `identity_*` 历史文件按用户指令**未删除**，仍留在云端仓；
   新交付路径一条都不加载。
8. 无 ppl / KL / 配对 CI / 能力 / 原生级声明；Qwen4-27B 仍未发布。
