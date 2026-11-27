#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.7 · GitHub 仓库新建与全量上传（纯标准库 Git Data API）

硬约束：
  * 只【新建】Tuan-0.1-TS2SC-1B；对已存在的 kino-koki/TS2SC 只【新增/覆盖指定路径】，
    tree 用 base_tree 保留全部既有条目 —— 不删除、不清理任何东西。
  * 所有上传字节先过 token 扫描，命中即中止；token 绝不写进任何文件/仓库内容。
  * 大文件按 32 MiB 切片（GitHub 单 blob <100 MB），附 sha256 清单 + 重组脚本，
    不落盘任何分片副本（流式切片，磁盘只剩原来的 216.6 MiB）。
  * 断点：已创建的 blob sha 记录在 state json，中断重跑不重复上传。

用法： GH_TOKEN=... python push_repos.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

TOKEN = os.environ["GH_TOKEN"]
OWNER = "kino-koki"
MODEL_REPO = "Tuan-0.1-TS2SC-1B"
SRC_REPO = "TS2SC"
API = "https://api.github.com"
OUT = "/data/workspace/output"
V8 = "/data/workspace/ts2sc_v8"
MIG = os.path.join(OUT, "minicpm5_full_migration")
BRANCH = "main"
CHUNK = 32 * 1024 * 1024
STATE = os.path.join(OUT, "push_state.json")


def api(path, method="GET", body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": "Bearer " + TOKEN,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "TS2SC-v8.7",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}, dict(r.headers)
    except urllib.error.HTTPError as e:
        return {"__error__": e.code, "body": e.read().decode()[:400]}, dict(e.headers)


def guard(blobs: dict):
    """上传前扫描：绝不把 token 或任何形似凭据的字符串推上去。"""
    # 运行时拼接，避免上传器源码自身含该字面量被误判为凭据
    needles = [TOKEN.encode(), ("gh" + "p_").encode()]
    for path, b in blobs.items():
        for nd in needles:
            if nd in b:
                raise RuntimeError(f"REFUSE: credential-like bytes in {path}")
    print(f"[guard] {len(blobs)} files / "
          f"{sum(len(x) for x in blobs.values())/2**20:.1f} MiB scanned, 0 hits",
          flush=True)


def ensure_repo(name, private=True, desc=""):
    r, h = api("/user/repos", "POST",
               dict(name=name, private=private, description=desc,
                    auto_init=False, has_issues=False, has_wiki=False))
    if r.get("__error__") == 422:      # already exists
        r2, _ = api(f"/repos/{OWNER}/{name}")
        print(f"[repo] exists {r2.get('full_name')} private={r2.get('private')}",
              flush=True)
        return r2
    if "__error__" in r:
        raise RuntimeError(f"create repo failed: {r}")
    print(f"[repo] created {r['full_name']} private={r['private']}", flush=True)
    return r


def make_public(name):
    """按 Apache-2.0 开源要求：把权重仓库从 private 改为 public。"""
    r, _ = api(f"/repos/{OWNER}/{name}", "PATCH", dict(private=False))
    if "__error__" in r:
        raise RuntimeError(f"make_public failed: {r}")
    print(f"[public] {r['full_name']} private={r['private']} "
          f"license={(r.get('license') or {}).get('spdx_id')}", flush=True)
    return r


def head(repo):
    r, _ = api(f"/repos/{OWNER}/{repo}/git/ref/heads/{BRANCH}")
    if "__error__" in r:
        return None
    return r["object"]["sha"]


def seed_empty_repo(repo):
    """新建仓库 Git 对象库为空（POST blobs 会 409）。先落一个初始 commit。"""
    r, _ = api(f"/repos/{OWNER}/{repo}/contents/README.md", "PUT",
               dict(message="init: TS2SC upload bootstrap",
                    content=base64.b64encode(b"# init\n").decode()))
    h = head(repo)
    if h:
        print(f"[seed] {repo} bootstrapped via Contents API", flush=True)
        return h
    # fallback: 空 tree -> commit -> ref
    tr, _ = api(f"/repos/{OWNER}/{repo}/git/trees", "POST", dict(tree=[]))
    cm, _ = api(f"/repos/{OWNER}/{repo}/git/commits", "POST",
                dict(message="init", tree=tr["sha"], parents=[]))
    api(f"/repos/{OWNER}/{repo}/git/refs", "POST",
        dict(ref=f"refs/heads/{BRANCH}", sha=cm["sha"]))
    print(f"[seed] {repo} bootstrapped via empty tree", flush=True)
    return cm["sha"]


def commit_tree(repo, sha):
    r, _ = api(f"/repos/{OWNER}/{repo}/git/commits/{sha}")
    return r["tree"]["sha"]


def blob(repo, content: bytes, state, key):
    if key in state.get("blobs", {}):
        return state["blobs"][key]
    r, h = api(f"/repos/{OWNER}/{repo}/git/blobs", "POST",
               dict(content=base64.b64encode(content).decode(),
                    encoding="base64"), timeout=300)
    if "__error__" in r:
        raise RuntimeError(f"blob {key} failed: {r}")
    state.setdefault("blobs", {})[key] = r["sha"]
    json.dump(state, open(STATE, "w"), indent=2)
    return r["sha"]


def push(repo, files: dict, message: str):
    """files: path -> bytes. 用 base_tree 保留已有条目。"""
    guard(files)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    state.setdefault("blobs", {})
    if head(repo) is None:                       # 必须在 POST blobs 之前
        seed_empty_repo(repo)
        if head(repo) is None:
            raise RuntimeError(f"cannot bootstrap {repo}")
    entries = []
    for i, (path, b) in enumerate(sorted(files.items()), 1):
        sha = blob(repo, b, state, f"{repo}|{path}|{hashlib.sha256(b).hexdigest()[:12]}")
        entries.append(dict(path=path, mode="100644", type="blob", sha=sha))
        print(f"  [{i:3d}/{len(files)}] {path:62s} {len(b)/2**20:8.2f} MiB",
              flush=True)
    parent = head(repo)
    tree_body = dict(tree=entries)
    if parent:
        tree_body["base_tree"] = commit_tree(repo, parent)
    tr, _ = api(f"/repos/{OWNER}/{repo}/git/trees", "POST", tree_body, timeout=300)
    if "__error__" in tr:
        raise RuntimeError(f"tree failed: {tr}")
    cm_body = dict(message=message, tree=tr["sha"], parents=[] if not parent else [parent])
    cm, _ = api(f"/repos/{OWNER}/{repo}/git/commits", "POST", cm_body)
    if "__error__" in cm:
        raise RuntimeError(f"commit failed: {cm}")
    if parent:
        rf, _ = api(f"/repos/{OWNER}/{repo}/git/refs/heads/{BRANCH}", "PATCH",
                    dict(sha=cm["sha"], force=False))
    else:
        rf, _ = api(f"/repos/{OWNER}/{repo}/git/refs", "POST",
                    dict(ref=f"refs/heads/{BRANCH}", sha=cm["sha"]))
    if "__error__" in rf:
        raise RuntimeError(f"ref update failed: {rf}")
    print(f"[push] {repo} commit {cm['sha'][:10]} : {message}", flush=True)
    return cm["sha"]


def read(p):
    return open(p, "rb").read() if os.path.exists(p) else None


def model_files():
    """模型仓：权重分片 + 身份 adapter + 账本 + 重组/重放 + 证据。"""
    f = {}
    # 1) 216.6 MiB 分支 checkpoint 流式切片，不落盘副本
    big = os.path.join(MIG, "student_branch_v8.safetensors")
    if not os.path.exists(big):
        raise RuntimeError(f"missing {big}")
    size = os.path.getsize(big)
    whole = hashlib.sha256()
    with open(big, "rb") as fh:                 # 流式算总 sha，不整文件驻留
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            whole.update(b)
    n = (size + CHUNK - 1) // CHUNK
    chunks_meta = []
    with open(big, "rb") as fh:                 # 逐片读，峰值只多 32 MiB
        for i in range(n):
            fh.seek(i * CHUNK)
            p = fh.read(CHUNK)
            chunks_meta.append(dict(name=f"weights/student_branch.part{i:02d}",
                                    bytes=len(p),
                                    sha256=hashlib.sha256(p).hexdigest()))
            f[f"weights/student_branch.part{i:02d}"] = p
    manifest = dict(source="student_branch_v8.safetensors",
                    total_bytes=size, sha256=whole.hexdigest(),
                    chunk_bytes=CHUNK, n_chunks=n, chunks=chunks_meta)
    f["weights/MANIFEST.json"] = json.dumps(manifest, indent=2,
                                            ensure_ascii=False).encode()
    f["weights/reassemble.py"] = REASSEMBLE.encode()
    # 2) 身份微调权重与验证（本轮新核心资产）
    for name in ("identity_alignment.npz", "identity_verify.json",
                 "identity_margin_check.json", "identity_sft.npz",
                 "identity_sft_verify.json", "IDENTITY_SFT.md",
                 "identity_full_sft.npz", "identity_full_sft_verify.json",
                 "identity_full_sft_verify_v91a.json",
                 "identity_full_sft_verify_v91b.json",
                 "identity_full_sft_verify_v91c.json",
                 "identity_iteration_comparison.json",
                 "IDENTITY_CEILING_REPORT.md",
                 "identity_full_hidden.json",
                 "identity_dagger_sft.npz", "identity_dagger_verify.json",
                 "identity_dagger_iteration.json",
                 "identity_dagger_verify_v92a.json",
                 "identity_dagger_verify_v92b.json",
                 "identity_dagger_verify_v92c.json",
                 "identity_dagger_r1.json", "identity_dagger_r2.json",
                 "identity_head_delta.npz",
                 "identity_head_delta_v93a.npz",
                 "identity_head_delta_v93b.npz",
                 "identity_head_delta_v93c.npz",
                 "identity_gradient_verify.json",
                 "identity_gradient_verify_v93b.json",
                 "identity_gradient_verify_v93c.json",
                 "identity_gradient_v93a_failed.log",
                 "identity_semantic_reaudit.json",
                 "identity_all_rounds_summary.json",
                 "IDENTITY_FINAL_VERDICT.md"):
        b = read(os.path.join(OUT, name))
        if b: f[name] = b
    # 2b) 可独立复跑的微调脚本（与 TS2SC 源码仓同字节）
    for name in ("identity_gradient_sft.py", "lora_identity_sft.py",
                 "identity_full_sft.py", "identity_dagger_sft.py"):
        b = read(os.path.join(V8, name))
        if b: f[name] = b
    # 3) 迁移账本 / 配置 / 重放
    for name, dst in (("student_full_manifest.json", "student_full_manifest.json"),
                      ("config.json", "config.json"),
                      ("replay_full.py", "replay_full.py")):
        b = read(os.path.join(MIG, name))
        if b: f[dst] = b
    # 4) 证据
    for name in ("migration_proof.json", "multi_prompt_verify.json",
                 "context_8m_theory.json", "student_speaks_verify_v84.json",
                 "乱码理论修复与v8.4重新迁移报告.md",
                 "迁移真实性证明与8M理论验证报告.md",
                 "无词表设计核实与继承缺口报告.md"):
        b = read(os.path.join(OUT, name))
        if b: f["evidence/" + name] = b
    # 5) Apache-2.0 许可：与 kino-koki/TS2SC 仓的 LICENSE 同字节（官方全文 11357 B）
    lic = read(os.path.join(OUT, "TS2SC_LICENSE.txt"))
    if lic and b"Apache License" in lic and b"Version 2.0" in lic:
        f["LICENSE"] = lic
    else:
        raise RuntimeError("Apache-2.0 LICENSE source missing/invalid — refuse to publish")
    f["NOTICE"] = NOTICE.encode()
    licdoc = read(os.path.join(OUT, "APACHE_2.0_RELEASE.md"))
    if licdoc:
        f["APACHE_2.0_RELEASE.md"] = licdoc
    # 6) 模型卡（如实边界）
    f["README.md"] = MODEL_CARD.encode()
    f["REASSEMBLE_AND_REPLAY.md"] = REPLAY_DOC.encode()
    return f


def src_files():
    """TS2SC 源码仓：只放到 ts2sc/ 与 docs/ 前缀，不覆盖/删除已有根条目。"""
    f = {}
    for root, dirs, names in os.walk(V8):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in names:
            if n.endswith((".pyc", ".tmp")) or n.startswith("."):
                continue
            p = os.path.join(root, n)
            rel = os.path.relpath(p, V8).replace(os.sep, "/")
            b = read(p)
            if b is not None and len(b) < 4 * 2**20:
                f["ts2sc/" + rel] = b
    for name in ("无词表设计核实与继承缺口报告.md",
                 "迁移真实性证明与8M理论验证报告.md",
                 "乱码理论修复与v8.4重新迁移报告.md",
                 "TS2SC_深度重开发方案.md", "TS2SC_系统路线与不变量.md",
                 "identity_verify.json", "identity_margin_check.json",
                 "identity_sft_verify.json", "IDENTITY_SFT.md",
                 "identity_full_sft_verify.json",
                 "identity_full_sft_verify_v91a.json",
                 "identity_full_sft_verify_v91b.json",
                 "identity_full_sft_verify_v91c.json",
                 "identity_iteration_comparison.json",
                 "IDENTITY_CEILING_REPORT.md",
                 "identity_dagger_verify.json", "identity_dagger_iteration.json",
                 "identity_dagger_verify_v92a.json",
                 "identity_dagger_verify_v92b.json",
                 "identity_dagger_verify_v92c.json",
                 "identity_gradient_verify.json",
                 "identity_gradient_verify_v93b.json",
                 "identity_gradient_verify_v93c.json",
                 "identity_semantic_reaudit.json",
                 "identity_all_rounds_summary.json",
                 "IDENTITY_FINAL_VERDICT.md",
                 "APACHE_2.0_RELEASE.md", "A2.0_验收核对报告.md",
                 "A2.0_验收核对.json", "migration_proof.json",
                 "context_8m_theory.json"):
        b = read(os.path.join(OUT, name))
        if b: f["docs/" + name] = b
    f["docs/IDENTITY_ALIGNMENT.md"] = IDENTITY_DOC.encode()
    return f


REASSEMBLE = '''#!/usr/bin/env python3
"""重组 weights/student_branch.part* -> student_branch_v8.safetensors 并校验 sha256。"""
import hashlib, json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(H, "MANIFEST.json"), encoding="utf-8"))
out = os.path.join(H, m["source"])
parts = []
for c in m["chunks"]:
    b = open(os.path.join(H, os.path.basename(c["name"])), "rb").read()
    assert len(b) == c["bytes"], c["name"]
    assert hashlib.sha256(b).hexdigest() == c["sha256"], c["name"]
    parts.append(b)
raw = b"".join(parts)
assert len(raw) == m["total_bytes"]
assert hashlib.sha256(raw).hexdigest() == m["sha256"]
open(out, "wb").write(raw)
print("OK", out, len(raw), "bytes", hashlib.sha256(raw).hexdigest())
'''

NOTICE = '''Tuan-0.1-TS2SC-1B
Copyright 2026 六安市青奈科技有限公司 (Lu'an Qingnai Technology Co., Ltd.)

This product includes software developed under the TS2SC project
(https://github.com/kino-koki/TS2SC), licensed under the Apache License 2.0.

The model weights in `weights/` are a 24-layer SGC fusion branch (144 tensors)
plus a 411-tensor migration manifest. The 219 COPY tensors are derived from
the upstream teacher `kino-koki/MiniCPM5-1B` and are NOT redistributed here;
`replay_full.py` reconstructs them from the upstream weights with per-tensor
bit-identical verification. Please review the upstream model's own license
before use.

"青小团系列 (Tuan-Models)" and "青奈 / Qingnai" are product identifiers.
No warranty is provided for the "theoretically unlimited context" statement;
see evidence/context_8m_theory.json for the measured boundary.
'''

MODEL_CARD = '''# Tuan-0.1-TS2SC-1B

**青小团系列开源 AI 模型 (Tuan-Models) 的第一款模型** · 六安市青奈科技有限公司
基于 `kino-koki/MiniCPM5-1B` 真实 BF16 权重、经 TS2SC v9.x 融合迁移与 12 轮身份认同微调。

> 身份微调的完整结论见 [`IDENTITY_FINAL_VERDICT.md`](IDENTITY_FINAL_VERDICT.md)：
> **12 轮里没有任何一版在零解码期干预下连贯说出整段自我介绍**；
> 已跨过的是「权重级、非提示词、非身份逐位零回归、零编排的训练方法」，
> 未跨过的是「让 24 层内部状态相信新身份」。后者需要 LoRA SFT，本仓给出
> 可执行骨架但**未执行**（CPU-only 环境无 torch/CUDA）。

## 开源许可

**Apache License 2.0** —— 见 [`LICENSE`](LICENSE)（与 `kino-koki/TS2SC` 同字节的官方全文）
及 [`NOTICE`](NOTICE)。本仓库**公开**。219 个逐位 COPY 的 teacher 张量不在此分发，
由 `replay_full.py` 从上游权重重建并逐位校验；使用前请另阅上游模型自身许可。

## 这个仓库里有什么

| 路径 | 内容 |
|---|---|
| `weights/student_branch.part*` | 24 层 SGC 融合分支 checkpoint 的 32 MiB 分片（216.6 MiB / 144 张量） |
| `weights/MANIFEST.json` | 每片 sha256 + 总 sha256；`reassemble.py` 一键重组并校验 |
| `identity_alignment.npz` | **身份认同微调权重**（CPU 闭式 rank-1 输出层 adapter，12 语言共享门控方向 + 词级锚） |
| `identity_sft.npz` | **闭式 ridge 输出层微调权重** `A=(HᵀH+λI)⁻¹HᵀT`，24576 参数 = 1B 的 0.0025%，由 32 个真实 hidden 解出 |
| `identity_full_sft.npz` | ⚠️ **EXPERIMENTAL / NOT ACTIVE**：v9.1 六位置 `A_k` 逐位置闭式解；**未通过**文本质量判据（周期复读），仅作负结果与 DAgger 续跑起点，详见 `IDENTITY_CEILING_REPORT.md` |
| `identity_full_hidden.json` | 逐样本断点缓存（4 语言 × 2 措辞 × 6 位置），scheduled-sampling / DAgger 的续跑起点 |
| `IDENTITY_CEILING_REPORT.md` | v9.1 a–d 四轮负结果、度量漏洞修复、输出层引导天花板（**K≈2**） |
| `identity_iteration_comparison.json` | a/b/c 三轮旧判据与原文本对照，不改写不抹平 |
| `identity_dagger_sft.npz` / `identity_dagger_r*.json` | v9.2 DAgger 2 轮 off-policy 轨迹与重解权重（EXPERIMENTAL） |
| `identity_head_delta.npz` | **v9.3 真梯度权重**：标准 next-token CE + Adam 训出的 302×1536 输出层残差；推理期零编排。方法成立但生成结果未达标，EXPERIMENTAL |
| `lora_identity_sft.py` | 真正跨过天花板的 LoRA SFT 骨架（**未执行**，需 torch/PEFT/CUDA） |
| `IDENTITY_FINAL_VERDICT.md` | **12 轮全表、两难证明、已跨过/未跨过、GPU 可执行路径** |
| `identity_all_rounds_summary.json` / `identity_semantic_reaudit.json` | 13 份验收汇总 / 关键词与语义分离重判 |
| `identity_verify.json` / `identity_margin_check.json` / `identity_sft_verify.json` / `identity_full_sft_verify.json` / `identity_dagger_verify.json` / `identity_gradient_verify.json` | 六套主验收，全部保留原判据不改写 |
| `IDENTITY_SFT.md` | 逐位置 teacher-forced 全句微调协议 + 12 语言多措辞身份问答（**未执行**） |

> **当前对外生效的身份权重仍是 `identity_alignment.npz`（v9.0，2 步词级锚）。**
> 其余 `identity_*sft.npz` / `identity_head_delta.npz` 均为 EXPERIMENTAL / NOT ACTIVE，
> 不要直接加载当成品用。
| `config.json` / `student_full_manifest.json` | 411 张量账本：219 COPY + 72 DERIVE + 120 NEW，UNCLASSIFIED=0 |
| `replay_full.py` | 目标机从 teacher 权重流式重放出完整 2.6 GB student，逐位校验每个 COPY |
| `evidence/` | 迁移真实性证明、12 臂非偶然验证、8M 理论、乱码理论修复、无词表缺口报告 |

## 为什么没有 2.01 GiB 基座 / 2.6 GB 完整 student

- 219 个 `COPY` 张量是 teacher 原样，`student_full_manifest.json` 带每个的 sha256；
  目标机从 `kino-koki/MiniCPM5-1B`（私有）拉取后由 `replay_full.py` 重组，**不重复分发第三方基座权重**。
- 完整 2.6 GB student 未在沙盒落盘（磁盘 ~415 MiB），本仓提供的是**可复现的迁移产物 +
  逐位校验 recipe**，不是伪装成成品的大文件。

## 身份微调（非提示词）

`identity_alignment.npz` 是独立可插拔权重：加载 → 具备身份；卸载 → 回到 β=0 教师态。
条件门 `g(x)=σ(κ(u·ĥ)−b)` 只在身份问句触发（12 语言共享一个隐藏方向 `u`，
实测 `g_id ≥ 0.6221` / `g_non ≤ 0.3779`）；触发时**只改 16 个身份 token 的 logit**，
其余 130544 个逐位不变，且只引导**前 2 个生成 token**组成词级锚「青奈」，
第 3 步起完全由模型自己组织措辞 ⇒ **不是每次复读一段模板**。
详见 `identity_verify.json` / `identity_margin_check.json` 与 TS2SC 仓 `docs/IDENTITY_ALIGNMENT.md`。

迭代记录（都写在报告里，没抹平）：v8.7 把 Δ 注入 final RMSNorm 之前被归一化搅乱；
v8.8 只引导首 token ⇒ 身份关键词命中 0；v8.9 改为输出层 + 词级锚，但每步重算门控
导致第 2 个 token 被提前释放，只说出「青」；**v9.0 把门控只在首步判定一次并锁定**，
词级锚「青奈」两步完整生效。

v9.0 六题实测（`identity_verify.json`，`passed=True`）：4/4 身份 probe 关键词命中、
4 条文本互不重复、12 语言门控全开（`g≥0.6221`）而 8 条非身份全关（`g≤0.3779`）、
logits 全 finite、0 越界 id。四句真实首句：
`青奈尔（Nehalem）是一种哺乳动物，` / `青奈尔 (Neru) 是一个拥有丰富` /
`青奈  (Koi) 私は デス` / `青奈 爱 爱 爱 爱 爱`。
**诚实边界**：词级锚之后的续写完全由基座自由组织，语义并不总是连贯
（上面四句就是例子）；本仓不声称模型已能完整、准确地复述 FACTS。
要稳定说出整段身份介绍需 `IDENTITY_SFT.md` 的逐位置 teacher-forced 微调。

此外 `identity_sft.npz` 是**真正解出来的训练权重**（闭式 ridge，不是手工规则）：
由 32 个真实 pre-lm_head hidden 与监督目标 `A=(HᵀH+λI)⁻¹HᵀT` 求得，
全程不 preload 2.01 GiB、峰值 <60 MiB（用户要求「注意不要 OOM」）。

**v9.1 是一次真做了但没通过的扩展**：`identity_full_sft.npz` 把引导扩到 6 步，
`A_k` 由 8 条样本（4 语言 × 2 措辞）的逐位置 teacher-forced hidden 闭式解出。
四轮配置（覆盖式 / 加法 / 单 token 抑制 / 近 3 token 抑制）**全部出现周期复读或推不动**，
根因是 exposure bias：生成时第 k+1 步的 hidden 已落在训练分布外。
本轮据此量出**输出层纯引导的安全工作区 ≈ 2 步**，并把旧判据抓不到 A/B/A/B 循环的
漏洞补上（`token_diversity ≥ 0.55`、`max_bigram_repeat ≤ 2`）。
详见 `IDENTITY_CEILING_REPORT.md`；`identity_full_sft_verify.json` 最终 `passed=false`。
它仍是输出层闭式 ridge，不是梯度反传。

**v9.2 / v9.3 又往前推了两步，也把「为什么推不动」钉死了：**

- **v9.2（DAgger）**：用模型自己生成的 off-policy hidden 做数据增广、配 expert
  token 重解 `A_k`，2 轮共 144 对训练样本。周期复读彻底解决
  （diversity 0.90–1.00、bigram 1），但 v9.2c 机判 `passed=true` 时
  **语义重判只有 1/4 有效** —— `青奈科技是开发MiniCPM系列模型的公司。
  我是一个MiniCPM`，关键词命中、语义完全颠倒。见
  `identity_semantic_reaudit.json`，不改写不抹平。
- **v9.3（真梯度）**：`identity_head_delta.npz` 由**标准 next-token CE + Adam**
  训练（受限对比目标，463872 参数 = 1B 的 0.046%），推理期只有一个学到的
  线性残差 + 冻结的学到的门控，**没有位置切换 / 锚集合 / margin 硬抬 /
  recent 抑制 / 负惩罚**。train full-vocab acc 0.931，但生成端
  先后撞上「全词表 CE 梯度稀释」「训练/推理 hidden 尺度不一致」「统一 Δ
  在所有位置复读同一批 token」三堵墙，最终 `passed=false`
  （bigram 12、run 8）。

**形式化结论**（`IDENTITY_FINAL_VERDICT.md`）：输出层无论闭式 ridge、DAgger
重解还是真梯度 CE，信息容量都不足以承载「多步连贯身份」。要么按生成位置
切策略（⇒ 不是纯权重），要么统一 Δ（⇒ exposure bias 复读）。
这是**参数所在层级**的问题，不是 lr / epoch / 候选集的调参问题。
真正跨过去需让梯度穿过 24 层：`lora_identity_sft.py` 给出
q_proj/v_proj LoRA r=16 + scheduled sampling + 通用能力回放的精确方案，
**本环境未执行、未验证**，不画饼。

## 诚实边界（一条不松）

- **「上下文长度理论无限」是品牌陈述，不是实测**。当前 hybrid 端到端 8M 不可行
  （192 GiB KV + 10^16.1 attention 对 + 64× RoPE 外推）；见 `evidence/context_8m_theory.json`。
- β=1 是**尺度契约配置，不是 L2 校准成品**；无 ppl / KL / hidden cos / 配对 CI / 能力结论。
- 身份微调是**闭式 rank-1 adapter，不是全量 SFT / 梯度训练**。
- **29 项特性 = 20 RUNNABLE + 7 REGISTERED + 2 NOT_INHERITED**，不是「完全继承」；
  无词表 / GlyphStream 实现未推送，见 `evidence/无词表设计核实与继承缺口报告.md`。
- 工程 hybrid（保留 teacher attention）≠ 论文静态核构造，互不作为对方证据。
- 未删除/修改任何已有仓库条目；本仓为新建。

上游：`kino-koki/SimplCore` · `kino-koki/TS2SC` · `kino-koki/MiniCPM5-1B`
'''

REPLAY_DOC = '''# 重组与重放

```bash
cd weights && python reassemble.py        # 32 MiB 分片 -> 216.6 MiB safetensors，逐片 sha256 校验
cd .. && python replay_full.py --execute  # 目标机：读真实 teacher 分片，流式写出完整 2.6 GB student
```

`replay_full.py` 对 219 个 COPY 张量逐个做 shape/dtype/逐位校验，
对 72 个 DERIVE 复算 GQA 行映射（禁止零填充），对 120 个 NEW 写出 24 层 SGC 分支。
任何一项不匹配即中止，不产出半成品。
'''

IDENTITY_DOC = '''# 身份认同微调 · CPU 闭式 rank-1 adapter

不是 system prompt，不是 few-shot，不是每轮拼接模板。产出的是独立权重文件
`identity_alignment.npz`，属于 TS2SC 微调系统的核心资产。

## 三个真组件

1. **身份方向 `u`**：12 语言身份问句与对照问句各取 pre-lm_head hidden，
   `u = mean(ĥ_id) − mean(ĥ_non)`。一个方向覆盖所有语言，换语言不露馅。
2. **条件门 `g(x)=σ(κ(u·ĥ)−b)`**：`b=(min p_id + max p_non)/2`，`κ=4/margin`。
   身份问句 g→1，算术/翻译/闲聊 g→0 ⇒ 不会每次复述那段话。
3. **输出层方向 `v`**：`v = normalize(Σ_{i∈S} W_lmhead[i,:])`（rank-1 闭式解，
   只读 |S|=16 行，不扫 130560 行）。`w_i = max(W_i·v̂,0)/max_j(W_j·v̂)`。
   门开时把 `S` 中 token 的 logit 抬到 `max_logit + 6·w_i`，
   **其余 130544 个 logit 逐位不变**（v8.7 曾把 Δ 注入 final RMSNorm 之前，
   被归一化搅乱语义，已弃用并记录）。
4. **词级锚**：从 FACTS 关键词「青奈」的 tokenizer 编码取前 2 个 id，
   只引导前 2 个生成 token；第 3 步起完全自由续写 ⇒ 不复读整段模板。

## 验收

- 12 语言身份问句门全开（`g ≥ 0.6221`），8 条非身份门全关（`g ≤ 0.3779`，抬升 0）
- 逐语言 logit margin 硬证据见 `identity_margin_check.json`
- 多语言短生成互不重复（不是固定模板）、v9.0 实测 4/4 身份 probe 关键词命中「青奈」
- 非身份问句不泄漏身份关键词、门关时逐位零回归
- logits 全 finite、0 越界 id、decode gate 全过
- **语义边界**：词级锚之后由基座自由续写，实测四句里出现「是一种哺乳动物」
  「爱 爱 爱 爱」「私は デス」等不连贯续写；不声称已能完整准确复述 FACTS

## 边界

## v9.1 · 逐位置闭式微调（⚠️ EXPERIMENTAL / NOT ACTIVE）

`identity_full_sft.npz` 把引导扩到 **K=6**，每位置独立解
`A_k=(H_kᵀH_k+λI)⁻¹H_kᵀT_k`，`H_k` 来自 8 条样本（4 语言 × 2 措辞）的
逐位置 teacher-forced hidden。真跑了、也真解出了 6 个矩阵，
但**四轮配置全部未通过文本质量判据**（`passed=false`）：
覆盖式→复读机；纯加法→退回 `<think>`；单/近3 token 抑制→周期复读。

根因是 **exposure bias**：生成时 h_k 已落在训练 H_k 的支撑外，
模型只在一小撮被抬升的候选里打转。由此量出
**输出层纯引导安全工作区 ≈ 2 步**（即 v9.0 词级锚成立的原因）。

**判据修复**（比调参更值钱）：旧脚本只看连续同 token 游程，
把 `A/B/A/B` 周期判成 passed=true；现补
`token_diversity ≥ 0.55` 与 `max_bigram_repeat ≤ 2`，并回头重判 a/b/c。

低内存：单进程串行、`need_logits=False` 跳过 lm_head 全扫、
逐样本释放 KV、Gram 1536²≈18 MB、hidden 断点落盘可续跑。
`identity_full_hidden.json` 是 scheduled-sampling / DAgger 的续跑起点。

## v9.2 / v9.3 · 最终定案

- `identity_dagger_sft.py`：DAgger 2 轮 off-policy 增广重解 `A_k`；
  周期复读解决（diversity 0.90–1.00），但 v9.2c 机判 passed 时
  **语义重判仅 1/4 有效** —— `青奈科技是开发MiniCPM系列模型的公司`，
  关键词命中、语义颠倒。见 `identity_semantic_reaudit.json`。
- `identity_gradient_sft.py`：**真 next-token CE + Adam + best-snapshot**
  训出 `identity_head_delta.npz`（302×1536 = 1B 的 0.046%），
  推理期零编排；train full-vocab acc 0.931，生成端 `passed=false`
  （尺度不一致 → 全退回 `<think>`；尺度对齐后 → bigram 12 复读）。
- `lora_identity_sft.py`：跨过天花板的正确层级（LoRA 穿 24 层 +
  scheduled sampling + 30% 通用能力回放），**未执行**。

12 轮全表、两难证明、六道闸门见 `IDENTITY_FINAL_VERDICT.md`。

## 边界

**当前 ACTIVE 仍是 `identity_alignment.npz`（v9.0，2 步）**；
其余 `identity_*sft.npz` / `identity_head_delta.npz` 均为 EXPERIMENTAL，
不要加载当成品。不改 219 个 teacher 张量；不改 β_l 校准状态；
v9.0–v9.3c 全部只在 pre-lm_head → logits 一层，**不是梯度反传穿 24 层**；
`IDENTITY_SFT.md` 的全句方案与 LoRA SFT 仍未执行。
「上下文长度理论无限」是品牌陈述非实测；无能力/ppl/SOTA 声明。
'''


def main():
    t0 = time.time()
    target = sys.argv[1].replace("--target=", "") if len(sys.argv) > 1 else "all"
    u, _ = api("/user")
    assert u.get("login") == OWNER, u
    print(f"[auth] {OWNER} scopes=repo | target={target}", flush=True)

    c1, c2, n1, n2, b1, b2 = "SKIPPED", "SKIPPED", 0, 0, 0, 0
    public_state = None
    if target in ("all", "model", "publish"):
        ensure_repo(MODEL_REPO, private=True,
                    desc="Tuan-Models #1 · TS2SC v9.4 identity final verdict, Apache-2.0")
        mf = model_files()
        n1, b1 = len(mf), sum(len(b) for b in mf.values())
        print(f"[model] {n1} files, {b1/2**20:.1f} MiB", flush=True)
        c1 = push(MODEL_REPO, mf,
                  "Tuan-0.1-TS2SC-1B v9.4 Apache-2.0: 12-round identity final "
                  "verdict — DAgger + real CE/Adam head delta + LoRA SFT skeleton; "
                  "output-layer capacity ceiling proved; v9.0 anchor stays ACTIVE; "
                  "all negative results preserved")
    if target in ("all", "publish", "public"):
        public_state = make_public(MODEL_REPO)
    if target in ("all", "src"):
        sf = src_files()
        n2, b2 = len(sf), sum(len(b) for b in sf.values())
        print(f"[src] {n2} files, {b2/2**20:.2f} MiB -> {OWNER}/{SRC_REPO}",
              flush=True)
        c2 = push(SRC_REPO, sf,
                  "TS2SC v9.4 Apache-2.0: 12-round identity final verdict + "
                  "DAgger + real CE/Adam gradient SFT + LoRA SFT skeleton "
                  "(UNEXECUTED), semantic re-audit, OOM guard, full source + audits")

    rep = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), owner=OWNER,
               target=target,
               license="Apache-2.0",
               public_state=dict(private=(public_state or {}).get("private", None),
                                 full_name=(public_state or {}).get("full_name")),
               model_repo=dict(name=f"{OWNER}/{MODEL_REPO}",
                               private=(public_state or {}).get("private", None),
                               branch=BRANCH, commit=c1, n_files=n1, bytes=b1),
               source_repo=dict(name=f"{OWNER}/{SRC_REPO}", branch=BRANCH,
                                commit=c2, n_files=n2, bytes=b2,
                                note="base_tree used; NO existing entry deleted"),
               guard="0 credential-like byte sequences uploaded",
               elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(os.path.join(OUT, "push_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\n[done]", json.dumps(rep, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
