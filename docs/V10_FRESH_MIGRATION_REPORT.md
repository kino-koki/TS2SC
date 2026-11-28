# TS2SC v10 · 上游干净权重全量重迁移报告

日期：2026-11-28 08:32:06 · `passed = True` · 耗时 660.3 s

## 指令

> 严重污染问题 → 重新拉 1B 干净权重重新做一次迁移；
> SimplCore 云端已更新，TS2SC 同步极大 SGC 外挂。

执行：**全新隔离目录、零复用旧产物、22 分片逐片 LIVE git blob sha1 校验**，
架构从 v8 单层静态核升级为 v10「冻结主干 + 三旁路外挂」。

## 结果

| 判据 | 结果 |
|---|---|
| 22/22 分片 LIVE blob sha1 | ✅ 2.01 GiB |
| 账本 `{'COPY': 219, 'DERIVE': 0, 'NEW': 312}` / UNCLASSIFIED | ✅ 0 |
| 8 个 COPY 逐位抽查 | ✅ |
| 312 NEW 重开逐字节复验 | ✅ 312/312 |
| FFT↔IIR 对偶 | 1.69e-14 |
| SE1 ρ=1 零串扰 | on 0.0 / off 0.0 |
| SE1 衰减闭式误差 | 3.33e-16 |
| 短卷积流式==整段 | 8.33e-17 |
| P7 旁路全开关主干不变 | ✅ |
| 纯度 AST | ✅ |
| β=0 Δ 严格零 | ✅ |
| β=1 契约比 / O(1) 状态 | 0.1 / ✅ |
| 生成全 finite / 解码守卫 / 架构生效 | ✅ / ✅ / 2/2 |
| 身份标记泄漏 | **0** |
| **passed** | **True** |

## 生成实测

- `zh` β=0 → `<think>
嗯，用户`
  β=1 → `<think>
用户要求` (active=True, markers=[])
- `en` β=0 → `<think>
Hmm,`
  β=1 → `<think>
The user` (active=True, markers=[])

## 边界（不删减）

- 『干净』口径 = 22 分片逐片 LIVE git blob sha1 校验重拉 + COPY sha256 从新字节计算 + NEW 参数零复用旧产物；不是 ppl / 下游 benchmark 知识集配对 CI
- β=1 是尺度契约配置，不是 L2 校准成品；β_l 仍待校准
- 上游 v9–v11 结论为否定性（COMPLETE_BEYOND=False），无 SOTA/超越声明
- 上游语料 240KB / D=32 / L≤1536；1B / 中文 / LRA / 长文档未测
- 无 torch / GPU；纯 numpy CPU；生成 probe 仅 4 token
- Qwen4-27B 仍未发布；『上下文理论无限』是品牌陈述非实测
- 未删除任何云端条目；identity_* 历史文件保留但 RETIRED / 不加载

架构同步映射与污染排查详见 `SGC_V10_SYNC.md`；上游快照 13 文件带 LIVE sha256 见 TS2SC 仓 `ts2sc/upstream_simplcore_v11/MANIFEST.json`。
