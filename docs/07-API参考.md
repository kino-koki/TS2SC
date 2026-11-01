# 07 · API 参考

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RA_SEQ` | 64 | 序列长度（cache 阶段生效，改了要重新 cache） |
| `RA_N` | 8 | 训练样本数 |
| `RA_NVA` | 4 | 验证样本数 |
| `RA_STEPS` | 300 | 每层蒸馏步数 |
| `RA_CHUNK` | 20 | 每轮跑多少步就退出（配合 shell 循环） |
| `RA_LR` | 3e-3 | 学习率 |
| `RA_LAYER` | 0 | 当前蒸馏第几层 |
| `OMP_NUM_THREADS` | 2 | CPU 线程 |

⚠️ `RA_SEQ` 改了必须重新跑 `cache`，否则激活 shape 对不上。

---

## `real_act_distill.py`

### `cache`

一次 teacher 前向，缓存全部 28 层真实激活到 `MIGREAL/act/`。

```
输入：teacher 权重（mig/model.safetensors）
输出：act/tr_l{i}.pt, act/va_l{i}.pt  每个含 {"X":..., "Y":...}
耗时：约 4 分钟（2 核 CPU，8+4 样本 × 28 层）
```

### `train`

逐层蒸馏 mixer，可续跑。

```bash
RA_LAYER=0 python3 real_act_distill.py train
```

检查点：`MIGREAL/ck/mixer_l{i}.pt`，含
`{"mixer": state_dict, "opt": optimizer_state, "step": int, "best": float}`

### `eval`

全层串联，输出逐层余弦 + 端到端余弦。

---

## `migrate_real.py`

迁移主程序。

```bash
python3 migrate_real.py --teacher-path ./Qwen3-0.6B [--plan-only]
```

输出：`MIGREAL/model-{0..4}.safetensors` + `config.json` + `result.json`

`result.json` 关键字段：
```json
{
  "plan":    {"total":..., "keep":..., "keep_ratio":..., "emb":..., "ffn":..., "mixer":...},
  "migrate": {"total":664607516, "copied":663544832, "copied_ratio":0.9984, "resid":0.0773}
}
```

---

## `test_capability.py`

五项能力验证，结果写入 `result.json` 的 `cap_*` 字段。

| 字段 | 内容 |
|---|---|
| `cap_layer` | 逐层保真度（rows + mean/min/max） |
| `cap_cumulative` | 累积漂移曲线 |
| `cap_o1` | O(1) 状态（detail + unique + bytes） |
| `cap_longctx` | 长上下文外推 |
| `cap_dual` | 流式↔并行对偶 |

---

## `diag_gap.py`

失真来源定位。**这是排查问题第一个该跑的脚本。**

输出三个对照：
```
① teacher注意力输出 vs 迁移mixer输出        → mixer 保真度
② teacher整层 vs [teacher注意力→迁移FFN]    → FFN 保真度
③ teacher整层 vs 迁移整层                   → 端到端
```

**判读**：
- ② 应该 ≈ 1.0000（若不是，说明 FFN 没载入 → 检查 ffn_ratio）
- ① < 0.5 是正常的（结构性损失），靠蒸馏补

---

## `simplunity.py`

```bash
python3 simplunity.py --plan                    # 融合规划
python3 simplunity.py --fuse --b-path DIR       # 真机融合
python3 simplunity.py --plan --b-path DIR       # 用真实 config 规划
```

---

## `fast_distill.py` / `distill_mixer.py` / `fix_coef.py`

| 脚本 | 用途 | 状态 |
|---|---|---|
| `fast_distill.py` | 高斯输入版可续跑蒸馏 | A1.0，**已被 real_act_distill 取代** |
| `distill_mixer.py` | 单层蒸馏对照实验 | 保留作对照 |
| `fix_coef.py` | coef 形状 bug 修复验证 | 结论：修了但没提升 |

---

## 返回值约定

所有脚本：
- 日志同时写 stdout 和 `MIGREAL/{name}log.txt`
- 结果写 `MIGREAL/result.json`（增量更新，不覆盖其他 key）
- **不抛异常**：失败时打印 `❌` 并继续
