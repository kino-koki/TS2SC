# 独立复现补充包 · 本地极限优化 v9–v11（2026-11-27）

> **定位**：外部第三方独立复现与边界核验，**不是**仓库作者的工作，**不是** SOTA 主张。
> **结论是否定性的**：`MECHANICAL_COMPLETE = False` / `HONEST_COMPLETE = False` /
> `COMPLETE_BEYOND = False`。质量轴从「落后」推进到**稳态 parity**，但仍非严格超越。
> 本包同时包含 v1–v8 的完整基底材料（`sgc_sidecar_final.py` … `limit_test_v8.py`）。

## 一句话结论

在**完全不动静态卷积核**（`ψ_m[τ]=ρ_m^τ·sin((τ+1)θ_m)` 固定、FFT↔IIR 双形式、
O(1) 定态、无位置表 / 无 max_len / 无 KV cache）与仓库纯度闸门的前提下：

- **成立**：成本 / 可行性 / 纯度 / 确定性四轴完全超越；质量轴达到**稳态 parity**
  （A1 +0.0029、A3 +0.0012 bpc @6000 步，3 seeds 连续 3 个 checkpoint 三轴机械全过）。
- **不成立**：质量轴「整个超越」不可达。两条独立否决理由：
  1. **2× 对手步数对照失败**（s11 A1 +0.1156、s22 +0.1835 超阈值 → `fair_all=False`），
     机械判定本身也是 False；
  2. **A2 的负 delta 全是崩坏区伪影**：所有赢都落在 L768/L1536，两臂 bpc 相对 L48
     基线劣化 +1.2~+7.7，属「双方都崩、S 崩得稍慢」。加绝对质量约束收紧后
     有效外推区 `effA2min` 为 +0.0068~+0.1494，S 没赢，稳定区间归 0。

## 推翻了上一轮的一个论断

v8 报告写过「静态核成对交互 O(M) 且与输入无关，堆 M 永远补不上随输入变化 →
定义性边界」。**该论证只对单层成立**。多层堆叠时第 l 层的输入
`U^(l)=A(X^(l))·σ(B(X^(l)))` 由上一层全局混合结果给出，叠加层间 RMSNorm + FFN
非线性后有效交互变成输入依赖，而核一字未动。v9 即为此而做。

## 目录与复现

```bash
cd repro/independent_v9_v11
python push_gate_check_v11.py          # 轻量门禁复核（推荐先跑）
python complete_beyond_v9.py           # v9 主脚本（Phase A 快筛）
python v9b_validate.py                 # α=0.75 假胜证伪 / 反向对照
python v9c_grid.py                     # 纯净线层数×lr×β×FFN 网格
python v9d_final.py                    # v9 决定性：5 seeds × 6000 步 + 2x 对手 + CI
python v10_probe.py                    # 每层独立固定谱字典 / EMA
python v11_stageA.py 11                # v11 曲线，逐 seed 增量落盘
python v11_stageA.py 22
python v11_stageA.py 33
python v11_stageBC.py                  # 聚合 / 崩坏区收紧判定 / 2x / CI / 确定性
```

依赖仅 `numpy`。全部脚本自包含（自动 chdir + 同目录相对依赖），可从任意 cwd 调用；
已实测从 `/tmp` 外部调用门禁脚本 PASS。seed / 环境 / sha256 写入各 JSON 的 `fingerprint`；
v11 分阶段结果 `v11_A_seed*.json` 与 `v11_partial.json` 均保留。

| 文件 | 说明 |
|---|---|
| `本地极限优化_最终报告.md` | **最终报告**（含全部超越、不足、停止准则） |
| `v11_alibi_results.json` | v11 最终决定性原始结果（双判定） |
| `v9d_final_results.json` / `v10_probe_results.json` | 前两轮决定性结果 |
| `v11_A_seed*.json` | 3 seeds × 5 checkpoints，含绝对 bpc |
| `push_gate_v11.json` / `push_gate_check_v11.py` | 推送前门禁与复核 |
| `repo_corpus.txt` / `repo_simplcore_core_arch_purity.py` | 语料与守卫快照 |
| 其余 `*.py` / `*_results.json` / `*.log` | v1–v8 基底与 v9–v11 全部迭代留档 |

## 证据分级

`REAL_RUN` 本机端到端训练/评测/计时/独立复跑 ｜ `IDENTITY` 解析与闭式等价
｜ `LEDGER_ALGEBRAIC` 复杂度/HBM 确定性上界 ｜ `NOT_MEASURED` 明确未测。

实跑核验：FFT↔直接因果卷积 7.1e-15、FFT 反传↔直接反传 1.1e-14、
两臂反传中心差分 5.9e-10 / 9.4e-11、真调仓库 `purity.py` **13/13 PASS**
（含负向变异全被抓）、独立复跑逐字段零差异、常驻状态 **46,080 B** 与 L 无关。

## 试过但无效的自由度（全部留档）

层数 {1,2,3,4,6,8,10} · M {64,128} · lr {2e-3,3e-3} · FFN 隐宽 {2D,4D} ·
残差缩放 β {1.0,0.5} · **每层独立固定谱字典（banded）反而更差**
（A1 +0.0108 / A2min +0.0261 / A3 +0.0023） · **EMA 逐位相同、无效** ·
两臂共享归一化 / RMSNorm / residual / FFN / pre-norm Block / Sidecar / readout /
优化器 / seed / 数据 / 长度课程 / dt 增广 · 更强对手：纯 ALiBi **无位置表**、
2× 训练步数、OOD 推到 L384/768/1536。

## 两次抓到的假胜（未抹除）

1. **α=0.75 长度归一化初版只加 SGC 臂** → A1 −0.50 / A2 −0.60 / A3 −0.58 的漂亮假胜。
   改成两臂共享 Block 级后归零；反向对照证实「谁拿偏置谁赢」
   （`sgc_only` 全轴大胜 / `attn_only` 全轴大败）。
2. **v11 只看 delta 的 A2 判据漏洞** → 加「赢所在 L 上 S 的 bpc 劣化 ≤ +1.0」收紧。

## 必须随结论一起读的限定（不得删减）

1. **等价 numpy 实现，不是仓库源码执行**：CI 真调用 `purity.py` 的函数，但**没跑仓库 pytest 套件**。
2. **语料极小**：仓库 `corpus.txt` 240 KB / 62 字符 / D=32 / L≤1536 / ≤6000 步；
   PPL 与通用 LM 质量结论不适用。
3. **对照臂非完整 Transformer**：单层 attention（两版：可学位置表+ALiBi / 纯 ALiBi），
   无多头 / FFN 堆叠 / 真实 tokenizer。
4. **parity 有代价**：参数 100,094 vs 92,798（1.08×）、6 层 Block + FFN。
5. **EFFECTIVE 判据是 v11 事后新增、非预注册**；`+1.0 bpc` 阈值是工程选择，非理论常数。
6. **OOD L1536 处于双方崩坏区**，其 delta 只作诊断，不作质量主张。
7. **1B+ / 中文 / LRA / 长文档 / 多 needle / 层级运算完全未测**；
   精确逐字指向仍是 O(1) 定态的信息论上限（主仓 README 自承）。
8. **fp16/bf16、长核误差累积、HIP 16-op 契约、`--train-chunk`、多卡 DDP 未测**。
9. **无第三方独立复现、未经同行评审**；仓库 0 star / 0 fork。
10. 会话中断曾打断两次运行，已改为分阶段增量落盘；所有中间文件保留。
11. 三次自我推翻（v1 A3 假赢、`final_decisive` 的「纯结构上限」、v6 的「M=512 parity」）
    与一项弃用判据（wall-clock）均留档。

## 要全轴赢必须放弃什么

| 放弃 | 得到 | 代价 |
|---|---|---|
| 静态 Toeplitz 核 | 输入依赖选择性 | 破坏 FFT↔IIR 与 O(L log L)，变 SRF/Mamba |
| O(1) 定态 | 精确召回 / 任意 needle | 效率主张消失，回到 KV / 外挂档案 |
| 纯度 | 直接拿最强质量 | CI 直接拒，不再是仓库定义的架构 |

任取一条即不再属于本仓库定义的架构。这是定义性边界，不是工程未及。
按 v11 头部预先声明的停止准则，纯度内自由度已穷举，**本轮收束**。
