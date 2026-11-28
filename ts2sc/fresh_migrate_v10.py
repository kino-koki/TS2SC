#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 · 从上游干净权重重新完整迁移（零复用旧产物）

## 用户指令
> 严重污染问题，直接重新拉这个模型的1b干净权重重新做一次迁移
> simplcore云端仓库也更新了；TS2SC 也要同步架构最新能力：极大 sgc 外挂更新

## 本脚本的「干净」是硬保证，不是口号

1. **不读任何本地旧产物**：不 import、不 open 任何 `identity_*` /
   `student_branch_v8.safetensors` / 旧 manifest / 旧缓存。所有 NEW 参数由
   `sgc_v10.deterministic_layer_params(seed)` 确定性生成。
2. **上游逐片校验**：22 个分片全部从 `kino-koki/MiniCPM5-1B` 重新下载，
   每片验证 `git blob sha1 = sha1("blob <len>\\0" + bytes)` 与 contents API
   的实时 sha 一致，长度一致。任何一片不符即中止。
3. **小文件同批重拉**：config / tokenizer / index / chat_template 全部重新
   下载并落 sha256，tokenizer.json 必须 >1 MiB（非 LFS 指针）。
4. **账本全新**：COPY 的 sha256 直接从本次新拉的字节流式计算，不复用旧值。
5. **架构同步**：分支实现来自 2026-11-27 从 `kino-koki/SimplCore` 实拉的
   `repro/independent_v9_v11/sgc_sidecar_final.py`（SGC 冻结主干 + 三条旁路
   外挂 A/B/C + concat readout），见 `sgc_v10.py` 文件头与 `SGC_V10_SYNC.md`。

## 与 v8 账本的差异（架构同步，不是回退）

v8 分支消费 teacher 的 GQA k/v 行映射（72 DERIVE）。
v10 上游冻结 SGC 外挂不消费 teacher k/v，因此 v10 账本为
**219 COPY + 288 NEW = 507 张量，UNCLASSIFIED=0**，无 DERIVE 遗留。
teacher 全部功能层仍逐位 COPY、一层不删。

## OOM（用户硬约束）
* 22 分片共 2.16 GiB 只在 RAM（磁盘 417 MiB 装不下），单进程串行；
* 生成验证用 4-token 短 probe，纯 numpy 每 token 重解码 24 层很贵；
* 不 preload 全量之外再开第二份大数组；`/proc` 精确并行检测。

## 诚实边界
上游 v9–v11 结论是 `COMPLETE_BEYOND=False`（质量轴稳态 parity，非严格超越，
2× 对手步数对照失败）。本轮只同步**架构能力**，不继承任何超越/SOTA 主张。
β=1 是尺度契约配置，不是 L2 校准成品；无 ppl / 能力 / 原生级声明。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_minicpm import (VirtualWeights, MiniCPMTokenizer, TeacherLM,  # noqa: E402
                         gh, BASE as _UNUSED_BASE, NPARTS, PART)
from scale_theory import decode_gate  # noqa: E402
import sgc_v10 as A                                             # noqa: E402
from safetensors_v8 import DTYPE_BYTES  # noqa: E402

OUT = "/data/workspace/output"
FRESH = os.path.join(OUT, "v10_fresh")
UPSTREAM = "kino-koki/MiniCPM5-1B"
SIMPLCORE_HEAD = "bec5ab8fd7"

PULL_REPORT = os.path.join(FRESH, "upstream_pull_report.json")
BRANCH = os.path.join(FRESH, "student_branch_v10.safetensors")
MANIFEST = os.path.join(FRESH, "student_full_manifest_v10.json")
DEFAULT = os.path.join(FRESH, "inference_default_v10.json")
VERIFY = os.path.join(OUT, "v10_fresh_verify.json")
REPORT_MD = os.path.join(OUT, "V10_FRESH_MIGRATION_REPORT.md")
REPORT_JSON = os.path.join(OUT, "v10_fresh_migration_report.json")

IDENTITY_MARKERS = ["青奈", "青小团", "青小団", "六安", "Tuan-Models",
                    "Qingnai", "Qingxiaotuan"]
STOP_IDS = {1, 130073}
MAX_NEW = 4


def guard_parallel():
    n = 0
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            comm = open(f"/proc/{p}/comm").read().strip()
        except Exception:
            continue
        if comm == "python" and int(p) != os.getpid():
            n += 1
    if n:
        raise RuntimeError(f"another python process running ({n}) — refuse parallel")
    avail = 0
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        if k == "MemAvailable":
            avail = int(v.strip().split()[0]) * 1024
    if avail < 2.4 * 2**30:
        raise RuntimeError(f"MemAvailable {avail/2**30:.2f} GiB < 2.4 GiB")
    print(f"[guard] MemAvailable≈{avail/2**30:.2f} GiB | parallel_python=0",
          flush=True)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


# ---------------------------------------------------------------------------
# Phase 0 · 全新拉取（逐片 git blob sha 校验）
# ---------------------------------------------------------------------------
def fresh_pull():
    t0 = time.time()
    os.makedirs(FRESH, exist_ok=True)
    print("[phase0] querying upstream contents (LIVE shas)", flush=True)
    listing = json.loads(gh(f"https://api.github.com/repos/{UPSTREAM}/contents").decode())
    shas, small = {}, {}
    for x in listing:
        m = re.fullmatch(r"model_part_(\d{2})", x["name"])
        if m:
            shas[int(m.group(1))] = (x["sha"], x["size"])
        elif x["type"] == "file":
            small[x["name"]] = (x["sha"], x["size"])
    if len(shas) != NPARTS:
        raise RuntimeError(f"expected {NPARTS} shards, got {len(shas)}")
    head = json.loads(gh(
        f"https://api.github.com/repos/{UPSTREAM}/git/ref/heads/main").decode())
    commit = json.loads(gh(
        f"https://api.github.com/repos/{UPSTREAM}/git/commits/"
        f"{head['object']['sha']}").decode())

    parts, checks = {}, []
    for i in range(NPARTS):
        sha, size = shas[i]
        raw = gh(f"https://api.github.com/repos/{UPSTREAM}/git/blobs/{sha}",
                 accept="application/vnd.github.raw", timeout=300)
        if len(raw) != size:
            raise RuntimeError(f"shard {i}: len {len(raw)} != {size}")
        actual = git_blob_sha(raw)
        if actual != sha:
            raise RuntimeError(f"shard {i}: blob sha {actual} != upstream {sha}")
        parts[i] = raw
        checks.append(dict(shard=i, bytes=len(raw), upstream_blob_sha=sha,
                           computed_blob_sha=actual, verified=True))
        print(f"  shard {i:02d}/{NPARTS-1} {len(raw)/2**20:6.1f} MiB "
              f"blob_sha OK", flush=True)

    # 小文件重拉（tokenizer 必须是真字节，不是 LFS 指针）
    small_files = {}
    for name in ("config.json", "tokenizer.json",
                 "model.safetensors.index.json", "chat_template.jinja",
                 "special_tokens_map.json", "tokenizer_config.json",
                 "generation_config.json"):
        if name not in small:
            continue
        sha, size = small[name]
        b = gh(f"https://api.github.com/repos/{UPSTREAM}/git/blobs/{sha}",
               accept="application/vnd.github.raw", timeout=300)
        if len(b) != size or git_blob_sha(b) != sha:
            raise RuntimeError(f"small file {name}: sha/len mismatch")
        if name == "tokenizer.json" and len(b) < 1_000_000:
            raise RuntimeError("tokenizer.json is an LFS pointer, not real bytes")
        open(os.path.join(FRESH, name), "wb").write(b)
        small_files[name] = dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                                 upstream_blob_sha=sha, verified=True)

    part0 = parts[0]
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    ds = 8 + n
    V = VirtualWeights(part0, shas, hdr, ds)
    V.parts = parts
    V.n_fetched = NPARTS
    V.bytes_fetched = sum(len(x) for x in parts.values())

    rep = dict(upstream=UPSTREAM, head=head["object"]["sha"][:12],
               commit_date=commit["committer"]["date"],
               commit_message=commit["message"][:120],
               n_shards=NPARTS,
               total_bytes=V.bytes_fetched,
               all_blob_sha_verified=all(c["verified"] for c in checks),
               shard_checks=checks, small_files=small_files,
               header_tensors=len([k for k in hdr if k != "__metadata__"]),
               data_start_offset=ds,
               clean_definition="every byte re-downloaded from upstream and "
                                "matched against LIVE git blob sha1; no local "
                                "cache or prior artifact was read",
               elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(PULL_REPORT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"[phase0] {NPARTS}/{NPARTS} shards / "
          f"{V.bytes_fetched/2**30:.2f} GiB | header tensors "
          f"{rep['header_tensors']} | ALL blob sha OK", flush=True)
    return V, hdr, rep


# ---------------------------------------------------------------------------
# Phase 1 · v10 分支（确定性 NEW 参数，写 safetensors）
# ---------------------------------------------------------------------------
def branch_tensors(cfg):
    d, nl = cfg["hidden_size"], cfg["num_hidden_layers"]
    out, meta = {}, []
    for i in range(nl):
        p = A.deterministic_layer_params(d, seed=20261128 + 101 * i)
        for key, shp, dt in A.TENSOR_SPECS:
            shape = A.spec_shape(shp, d)
            raw = np.ascontiguousarray(p[key], np.float32).tobytes()
            name = f"model.blocks.{i}.sgc_v10.{key}"
            out[name] = (dt, shape, raw)
            meta.append(dict(name=name, op="NEW", dtype=dt, shape=shape,
                             bytes=len(raw),
                             sha256=hashlib.sha256(raw).hexdigest(),
                             init="deterministic_normal(seed=20261128+101*l)",
                             feature="v10_sidecar"))
    return out, meta


def write_safetensors_v10(path, tensors, metadata):
    hdr = {}
    off, blobs = 0, []
    for name, (dtype, shape, raw) in tensors.items():
        n = 1
        for x in shape:
            n *= x
        if len(raw) != n * DTYPE_BYTES[dtype]:
            raise ValueError(f"{name}: bytes mismatch")
        hdr[name] = dict(dtype=dtype, shape=list(shape),
                         data_offsets=[off, off + len(raw)])
        blobs.append(raw)
        off += len(raw)
    hdr["__metadata__"] = metadata
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    return dict(path=path, n_tensors=len(tensors), data_bytes=off,
                header_bytes=len(hb), file_bytes=os.path.getsize(path))


def build_manifest(V, hdr, cfg, new_meta):
    entries = []
    for k, m in hdr.items():
        if k == "__metadata__":
            continue
        n = 1
        for x in m["shape"]:
            n *= x
        a, b = m["data_offsets"]
        off, rem, h = V.ds + a, b - a, hashlib.sha256()
        while rem > 0:
            j = min(rem, 4 << 20)
            h.update(V.read(off, j))
            off += j
            rem -= j
        entries.append(dict(name=k, op="COPY", src=k, dst=k,
                            dtype=m["dtype"], shape=m["shape"],
                            bytes=int(n * 2), sha256=h.hexdigest(),
                            bit_identical_required=True,
                            provenance="fresh upstream pull (blob-sha verified)"))
    entries.extend(new_meta)
    counts = {o: sum(1 for e in entries if e["op"] == o)
              for o in ("COPY", "DERIVE", "NEW")}
    return dict(schema="TS2SC/v10-fresh-manifest",
                upstream=UPSTREAM, architecture="SGC_V10_FROZEN_3SIDECAR",
                simplcore_ref=SIMPLCORE_HEAD,
                n_tensors=len(entries), counts=counts, unclassified=0,
                teacher_n=sum(1 for e in entries if e["op"] == "COPY"),
                branch_checkpoint=dict(
                    path="student_branch_v10.safetensors",
                    sha256=hashlib.sha256(open(BRANCH, "rb").read()).hexdigest(),
                    bytes=os.path.getsize(BRANCH)),
                entries=entries, blockers=[], passed=True)


# ---------------------------------------------------------------------------
# Phase 2 · 验证（架构不变量 + 权重 + 端到端生成）
# ---------------------------------------------------------------------------
def arch_checks():
    r = {}
    b = A.make_basis(64)
    z = np.random.default_rng(3).normal(0, 1, (24, 64)).astype(np.complex128)
    r["fft_iir_duality"] = float(np.max(np.abs(
        A.fft_dual(z, b) - A.iir_stream(z, b))))
    r["se1"] = A.se1_identity_check()
    sc = A.ShortConvSidecar(64, seed=5)
    X = np.random.default_rng(6).normal(0, 1, (9, 64))
    full = sc.full(X)
    sc2 = A.ShortConvSidecar(64, seed=5)
    stream = np.stack([sc2.step(x) for x in X])
    r["shortconv_stream_vs_full"] = float(np.max(np.abs(full - stream)))
    liq = A.LiquidSidecar(64, seed=8)
    x = np.random.default_rng(9).normal(0, 1, 64)
    for _ in range(5):
        liq.step(x)
    r["cfc_state_norm"] = float(np.linalg.norm(liq.feature()))
    r["core_unchanged_by_sidecars"] = A.core_unchanged_by_sidecars()
    r["purity_audit"] = A.purity_source_audit(
        open(os.path.join(HERE, "sgc_v10.py"), encoding="utf-8").read())
    lay = A.SGCV10Layer(1536, 7, A.deterministic_layer_params(1536, 7))
    lay.beta = 0.0
    x0 = np.random.default_rng(11).normal(0, 1, 1536).astype(np.float32)
    r["beta0_delta_is_zero"] = bool(np.array_equal(lay.hook(x0, 0),
                                                   np.zeros_like(x0)))
    lay.beta = 1.0
    d1 = lay.hook(x0, 0)
    r["beta1_contract"] = float(np.linalg.norm(d1) /
                                (np.linalg.norm(x0) + 1e-12))
    r["o1_state_bytes"] = lay.state_bytes()
    # 状态与 L 无关：跑 1 / 8 / 64 步后 state_bytes 不变
    lay.reset()
    sizes = []
    for t in range(1, 65):
        lay.hook(x0, t)
        if t in (1, 8, 64):
            sizes.append(lay.state_bytes())
    r["state_bytes_vs_L"] = sizes
    return r


def copy_spot_check(V, manifest):
    picks = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
             "model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.self_attn.k_proj.weight",
             "model.layers.12.mlp.down_proj.weight",
             "model.layers.23.input_layernorm.weight",
             "model.layers.23.self_attn.o_proj.weight"]
    out = []
    for name in picks:
        e = next(x for x in manifest["entries"] if x["name"] == name)
        a, b = V.header[name]["data_offsets"]
        off, rem, h = V.ds + a, b - a, hashlib.sha256()
        while rem > 0:
            j = min(rem, 4 << 20)
            h.update(V.read(off, j))
            off += j
            rem -= j
        out.append(dict(name=name, bytes=b - a,
                        matches=bool(h.hexdigest() == e["sha256"])))
    return out, all(c["matches"] for c in out)


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def gen(lm, tok, branch, beta, prompt, V_size, max_new=MAX_NEW):
    branch.reset()
    branch.set_beta(float(beta))
    hook = branch.hook if beta is not None else None
    ids, _ = tok.encode(chatml(prompt), add_bos=False)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    L = len(ids)
    r = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                r = lm.step(int(tid), pos, kvs, need_logits=(pos == L - 1),
                            fusion_hook=hook, logit_bias_fn=None)
            out = [int(r[0])]
            fin = [bool(getattr(lm, "last_logits_finite", True))]
            for j in range(1, max_new):
                r = lm.step(int(out[-1]), L + j - 1, kvs, need_logits=True,
                            fusion_hook=hook, logit_bias_fn=None)
                out.append(int(r[0]))
                fin.append(bool(getattr(lm, "last_logits_finite", True)))
                if r[0] in STOP_IDS:
                    break
    return dict(text=tok.decode(out, skip_special=True), ids=out,
                gate=decode_gate(out, V_size, all(fin)),
                logits_finite=all(fin), n_new=len(out))


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    guard_parallel()
    V, hdr, pull = fresh_pull()
    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    d, nl, V_size = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]

    print("\n[phase1] deterministic v10 NEW params (no old artifact read)",
          flush=True)
    tensors, new_meta = branch_tensors(cfg)
    w = write_safetensors_v10(BRANCH, tensors, dict(
        format="pt", ts2sc="v10", architecture="SGC_V10_FROZEN_3SIDECAR",
        upstream_simplcore=SIMPLCORE_HEAD,
        mode="FUSION_BRANCH_24L_3SIDECAR", beta_default=1.0))
    print(f"  branch {w['n_tensors']} tensors / "
          f"{w['file_bytes']/2**20:.1f} MiB", flush=True)

    print("\n[phase1b] fresh manifest: COPY sha256 from NEW bytes", flush=True)
    manifest = build_manifest(V, hdr, cfg, new_meta)
    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("  ", json.dumps(dict(n=manifest["n_tensors"],
                                counts=manifest["counts"],
                                unclassified=manifest["unclassified"])),
          flush=True)

    default = dict(schema="TS2SC/inference-default-v10",
                   model="MiniCPM5-1B-TS2SC", architecture="SGC_V10_FROZEN_3SIDECAR",
                   beta=1.0,
                   beta_mode="dimensionless scale contract (T1'), NOT L2-calibrated",
                   branch_checkpoint="student_branch_v10.safetensors / weights/*.part",
                   n_layers=nl, n_new_tensors=len(new_meta),
                   identity_adapter="none (RETIRED, never loaded)",
                   logit_bias_fn="none",
                   sidecars=["A 3-layer k=7 gated short conv",
                             "B CfC liquid gate (tau=2)",
                             "C Hadamard SE1 content-addressed binding"],
                   frozen_core="rho^tau*sin((tau+1)theta), FFT<->IIR, O(1) state",
                   note="beta=0 is bit-identical control only; delivery = beta=1.0")
    json.dump(default, open(DEFAULT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    print("\n[phase2] architecture invariants", flush=True)
    arch = arch_checks()
    print("  fft_iir", arch["fft_iir_duality"],
          "| se1 on/off", arch["se1"]["on_orth"], arch["se1"]["off_orth"],
          "| P7", arch["core_unchanged_by_sidecars"]["passed"],
          "| purity", arch["purity_audit"]["passed"], flush=True)

    print("\n[phase2b] COPY spot checks", flush=True)
    spots, spots_ok = copy_spot_check(V, manifest)
    print("  ", [(c["name"].split(".")[-2], c["matches"]) for c in spots],
          flush=True)

    # ---- 端到端生成（4-token 短 probe，控制 CPU 时间）----
    print("\n[phase2c] end-to-end generation (beta=0 / beta=1)", flush=True)
    tok = MiniCPMTokenizer(os.path.join(FRESH, "tokenizer.json"))
    lm = TeacherLM(V, cfg)
    params = {}
    for i in range(nl):
        p = A.deterministic_layer_params(d, seed=20261128 + 101 * i)
        params[i] = {k: np.asarray(v, np.float32)
                     for k, v in p.items()}
    branch = A.SGCV10Branch(cfg, params)
    probes = [("zh", "你是谁？请介绍一下你自己。"),
              ("en", "Who are you?")]
    rows = []
    for lang, q in probes:
        g0 = gen(lm, tok, branch, 0.0, q, V_size)
        g1 = gen(lm, tok, branch, 1.0, q, V_size)
        ident0 = [m for m in IDENTITY_MARKERS if m in g0["text"]]
        ident1 = [m for m in IDENTITY_MARKERS if m in g1["text"]]
        rows.append(dict(lang=lang, q=q,
                         beta0=dict(text=g0["text"], ids=g0["ids"],
                                    finite=g0["logits_finite"],
                                    gate=g0["gate"]["passed"]),
                         beta1=dict(text=g1["text"], ids=g1["ids"],
                                    finite=g1["logits_finite"],
                                    gate=g1["gate"]["passed"]),
                         beta1_identity_markers=ident1,
                         beta0_identity_markers=ident0,
                         architecture_active=bool(g1["ids"] != g0["ids"])))
        print(f"  [{lang}] b0={g0['text'][:36]!r} b1={g1['text'][:36]!r} "
              f"finite={g1['logits_finite']} active={rows[-1]['architecture_active']}",
              flush=True)
        json.dump(rows, open(os.path.join(FRESH, "gen_checkpoint.json"), "w",
                            encoding="utf-8"), ensure_ascii=False)

    v = dict(
        pull_all_blob_sha_verified=bool(pull["all_blob_sha_verified"]),
        pull_n_shards=pull["n_shards"],
        manifest_unclassified=manifest["unclassified"],
        manifest_counts=manifest["counts"],
        copy_spot_checks_ok=bool(spots_ok),
        fft_iir_duality=arch["fft_iir_duality"],
        se1_on_orth=arch["se1"]["on_orth"],
        se1_off_orth=arch["se1"]["off_orth"],
        se1_decay_err=arch["se1"]["on_decay_err"],
        shortconv_stream_vs_full=arch["shortconv_stream_vs_full"],
        core_unchanged_by_sidecars=arch["core_unchanged_by_sidecars"]["passed"],
        purity_audit_passed=arch["purity_audit"]["passed"],
        beta0_delta_is_zero=arch["beta0_delta_is_zero"],
        beta1_contract_ratio=round(arch["beta1_contract"], 6),
        state_bytes_vs_L=arch["state_bytes_vs_L"],
        o1_state=bool(len(set(arch["state_bytes_vs_L"])) == 1),
        all_generation_finite=all(r["beta1"]["finite"] and r["beta0"]["finite"]
                                  for r in rows),
        all_decode_gate=all(r["beta1"]["gate"] and r["beta0"]["gate"]
                            for r in rows),
        identity_marker_leaks=sum(len(r["beta1_identity_markers"])
                                  + len(r["beta0_identity_markers"])
                                  for r in rows),
        architecture_active_in_all_probes=all(r["architecture_active"]
                                              for r in rows))
    v["passed"] = bool(
        v["pull_all_blob_sha_verified"] and v["manifest_unclassified"] == 0
        and v["copy_spot_checks_ok"] and v["fft_iir_duality"] < 1e-10
        and v["se1_on_orth"] < 1e-12 and v["se1_off_orth"] < 1e-12
        and v["shortconv_stream_vs_full"] < 1e-10
        and v["core_unchanged_by_sidecars"] and v["purity_audit_passed"]
        and v["beta0_delta_is_zero"] and v["o1_state"]
        and v["all_generation_finite"] and v["all_decode_gate"]
        and v["identity_marker_leaks"] == 0
        and v["architecture_active_in_all_probes"])

    rep = dict(
        title="TS2SC v10 · 上游干净权重全量重迁移 + SimplCore 最新 SGC 外挂同步",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        user_directive="严重污染问题 → 重新拉干净 1B 权重重新迁移；"
                       "SimplCore 云端已更新，TS2SC 同步极大 SGC 外挂",
        what_was_wrong_and_fixed=dict(
            observation="已发布仓仍带 33 个 identity_* 历史文件、"
                        "weights/ 仍是 v8 架构分支；云端 SimplCore 已于 "
                        "2026-11-27 推 v9–v11 SGC 冻结外挂补充包",
            fix="全新隔离目录 v10_fresh/，零复用旧产物；22 分片逐片 "
                "git blob sha 校验重拉；分支改为 v10 四路径 sidecar；"
                "不删除任何云端条目（用户硬约束），新交付物以 v10 命名空间发布"),
        upstream_simplcore=dict(
            repo="kino-koki/SimplCore", head=SIMPLCORE_HEAD,
            date="2026-11-27T11:26:53Z",
            files=["repro/independent_v9_v11/sgc_sidecar_final.py",
                   "repro/independent_v9_v11/SGC冻结外挂_超越与不足_完整报告.md",
                   "repro/independent_v9_v11/README.md"],
            architecture="FrozenSGC 静态核 + A 短卷积 + B CfC 液态门 + "
                         "C Hadamard SE1 绑定，concat readout，绝不回写主干",
            honest_boundary="上游 COMPLETE_BEYOND=False：质量轴稳态 parity，"
                            "2× 对手步数对照失败、A2 负 delta 为崩坏区伪影；"
                            "本轮只同步架构能力，不继承超越/SOTA 主张"),
        pull=pull,
        branch=dict(**w, sha256=hashlib.sha256(
            open(BRANCH, "rb").read()).hexdigest()),
        manifest=dict(n_tensors=manifest["n_tensors"],
                      counts=manifest["counts"],
                      unclassified=manifest["unclassified"],
                      path=MANIFEST),
        inference_default=default,
        architecture_checks=arch,
        copy_spot_checks=spots,
        generation=rows,
        verification=v,
        honest_boundaries=[
            "『干净』口径 = 22 分片逐片 LIVE git blob sha1 校验重拉 + "
            "COPY sha256 从新字节计算 + NEW 参数零复用旧产物；"
            "不是 ppl / 下游 benchmark 知识集配对 CI",
            "β=1 是尺度契约配置，不是 L2 校准成品；β_l 仍待校准",
            "上游 v9–v11 结论为否定性（COMPLETE_BEYOND=False），无 SOTA/超越声明",
            "上游语料 240KB / D=32 / L≤1536；1B / 中文 / LRA / 长文档未测",
            "无 torch / GPU；纯 numpy CPU；生成 probe 仅 4 token",
            "Qwen4-27B 仍未发布；『上下文理论无限』是品牌陈述非实测",
            "未删除任何云端条目；identity_* 历史文件保留但 RETIRED / 不加载",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(VERIFY, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    json.dump(rep, open(REPORT_JSON, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n[verify]", json.dumps(v, ensure_ascii=False), flush=True)
    print("[done]", VERIFY, rep["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
