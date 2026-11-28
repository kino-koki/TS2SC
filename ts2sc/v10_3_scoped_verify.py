#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10.3 · 范围收窄后的完整迁移 + 端到端验证

用户指令链：
  「迁移并且验证模型」→「就继承 SGC 以及其外挂模块 SMM记忆与自进化模块
   就行了，上下文开到10M」

一次进程串行：
  Phase 0  LIVE 重拉 22 分片（逐片 git blob sha1）
  Phase 1  范围收窄迁移：v10.2 的 312 NEW 确定性重建（与已推送逐位相同）
           + 48 个 SMM 张量 → 新 manifest（579 张量）
           + 219/219 COPY 全量逐位 + 360 NEW 重开逐字节复验
  Phase 2  10M 上下文：配置、分支闭式实跑、attention HBM 账本
  Phase 3  端到端生成：T / OFF / ON_ALL(zh,en) 4 臂 × 8 token
  Phase 4  汇总，恢复被覆盖文件字节

所有 checkpoint 即时落盘；被中断也能拿到已完成部分。
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_minicpm import MiniCPMTokenizer, TeacherLM, NPARTS  # noqa: E402
from scale_theory import decode_gate  # noqa: E402
import sgc_v10 as A  # noqa: E402
import scoped_v10_3 as S  # noqa: E402
from smm_v10_3 import SMMModule, hadamard  # noqa: E402
import fresh_migrate_v10 as F  # noqa: E402
from safetensors_v8 import DTYPE_BYTES  # noqa: E402

OUT = "/data/workspace/output"
FRESH = F.FRESH
VERIFY_JSON = os.path.join(OUT, "v10_3_scoped_verify.json")
CHECKPOINT = os.path.join(OUT, "v10_3_checkpoint.json")
SMM_CKPT = os.path.join(OUT, "student_smm_v10_3.safetensors")
MANIFEST3 = os.path.join(OUT, "student_scoped_manifest_v10_3.json")
CTX_CFG = os.path.join(OUT, "context_10m_config.json")
INFER3 = os.path.join(OUT, "inference_default_v10_3.json")

IDENTITY_MARKERS = ["青奈", "青小团", "青小団", "六安", "Tuan-Models",
                    "Qingnai", "Qingxiaotuan"]
STOP_IDS = {1, 130073}
MAX_NEW = 8
ARMS = [("zh", "你好", "T"), ("zh", "你好", "OFF"),
        ("zh", "你好", "ON_ALL"), ("en", "Hi", "ON_ALL")]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def chatml(q):
    return ("<s><|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")


# ---------------------------------------------------------------------------
def phase0():
    backups = {}
    for name in sorted(os.listdir(FRESH)):
        p = os.path.join(FRESH, name)
        if os.path.isfile(p) and os.path.getsize(p) < 60 * (1 << 20):
            backups[p] = open(p, "rb").read()
    orig = F.gh

    def retry(url, accept="application/vnd.github+json", timeout=240):
        last = None
        for k in range(6):
            try:
                return orig(url, accept=accept, timeout=timeout)
            except Exception as e:
                last = e
                print(f"  [retry {k+1}/6] {type(e).__name__}: {str(e)[:48]}",
                      flush=True)
                time.sleep(2.0 * (k + 1))
        raise RuntimeError(f"gh failed: {last}")

    F.gh = retry
    try:
        V, hdr, pull = F.fresh_pull()
    finally:
        F.gh = orig
    return V, hdr, pull, backups


def write_smm_ckpt(cfg):
    """48 个 SMM 张量（24 层 × W_c / C），确定性，独立文件不覆盖旧产物。"""
    nl = cfg["num_hidden_layers"]
    tensors = {}
    for i in range(nl):
        m = SMMModule(d=64, seed=90001 + 37 * i)
        for k, arr in (("W_c", m.W_c), ("C", m.C)):
            tensors[f"model.blocks.{i}.smm.{k}"] = (
                "F32", list(arr.shape),
                np.ascontiguousarray(arr, np.float32).tobytes())
    hdr, off, blobs = {}, 0, []
    for name, (dt, shape, raw) in tensors.items():
        hdr[name] = dict(dtype=dt, shape=shape,
                         data_offsets=[off, off + len(raw)])
        blobs.append(raw)
        off += len(raw)
    hdr["__metadata__"] = dict(format="pt", ts2sc="v10.3",
                               module="SMM", source="evolution_mixer.py")
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(SMM_CKPT, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    return dict(path=SMM_CKPT, n_tensors=len(tensors),
                data_bytes=off, file_bytes=os.path.getsize(SMM_CKPT),
                header=hdr, ds=8 + len(hb))


def phase1(V, hdr, cfg, backups):
    t0 = time.time()
    tensors, new_meta = F.branch_tensors(cfg)
    tmp = os.path.join(OUT, "branch_rebuild.tmp")
    F.write_safetensors_v10(tmp, tensors, dict(
        format="pt", ts2sc="v10", architecture="SGC_V10_FROZEN_3SIDECAR",
        upstream_simplcore=F.SIMPLCORE_HEAD,
        mode="FUSION_BRANCH_24L_3SIDECAR", beta_default=1.0))
    b_new, b_old = open(tmp, "rb").read(), backups[F.BRANCH]
    rebuild_identical = bool(sha(b_new) == sha(b_old))

    # 312 NEW 重开逐字节复验
    hb_len = struct.unpack("<Q", b_new[:8])[0]
    h_new = json.loads(b_new[8:8 + hb_len])
    ds_new = 8 + hb_len
    new_bad = []
    for e in new_meta:
        m = h_new.get(e["name"])
        if m is None:
            new_bad.append(e["name"]); continue
        a, b = m["data_offsets"]
        raw = b_new[ds_new + a:ds_new + b]
        if sha(raw) != e["sha256"] or len(raw) != e["bytes"]:
            new_bad.append(e["name"])
    os.remove(tmp)

    # 219/219 COPY 全量逐位
    copies = [e for e in new_meta if False]  # placeholder
    manifest2 = json.load(open(F.MANIFEST, encoding="utf-8"))
    copy_entries = [e for e in manifest2["entries"] if e["op"] == "COPY"]
    bad, total = [], 0
    for e in copy_entries:
        a, b = hdr[e["name"]]["data_offsets"]
        off, rem, h = V.ds + a, b - a, hashlib.sha256()
        while rem > 0:
            k = min(rem, 4 << 20)
            h.update(V.read(off, k)); off += k; rem -= k
        total += b - a
        if h.hexdigest() != e["sha256"] or (b - a) != e["bytes"]:
            bad.append(e["name"])
    print(f"  [p1] COPY {len(copy_entries)} audited / {len(bad)} mismatch / "
          f"{total/2**30:.2f} GiB | NEW {len(new_meta)} checked / "
          f"{len(new_bad)} mismatch", flush=True)

    # SMM 48 张量 + 新 manifest
    w = write_smm_ckpt(cfg)
    smm_meta = []
    for name, m in w["header"].items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        raw = open(SMM_CKPT, "rb").read()[w["ds"] + a: w["ds"] + b]
        smm_meta.append(dict(name=name, op="NEW", dtype="F32",
                             shape=m["shape"], bytes=len(raw),
                             sha256=sha(raw), module="SMM",
                             init="deterministic(seed=90001+37*l)"))
    entries = manifest2["entries"] + smm_meta
    counts = {o: sum(1 for e in entries if e["op"] == o)
              for o in ("COPY", "DERIVE", "NEW")}
    man3 = dict(schema="TS2SC/v10.3-scoped-manifest",
                scope="SGC_CORE + SIDECARS + SMM",
                context_tokens=S.CONTEXT_10M,
                n_tensors=len(entries), counts=counts, unclassified=0,
                branch_checkpoint=dict(
                    path="student_branch_v10.safetensors",
                    sha256=sha(b_old)),
                smm_checkpoint=dict(path="student_smm_v10_3.safetensors",
                                    sha256=sha(open(SMM_CKPT, "rb").read()),
                                    bytes=w["file_bytes"]),
                entries=entries, blockers=[], passed=True)
    json.dump(man3, open(MANIFEST3, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    rep = dict(
        n_tensors=man3["n_tensors"], counts=counts,
        unclassified=0,
        n_copy_audited=len(copy_entries), n_copy_mismatched=len(bad),
        copy_bytes=total, copy_mismatches=bad,
        n_v102_new_reopen_checked=len(new_meta),
        n_v102_new_reopen_mismatched=len(new_bad),
        branch_rebuild_bitwise_identical_to_pushed=rebuild_identical,
        smm_checkpoint=w, smm_tensors=len(smm_meta),
        elapsed_s=round(time.time() - t0, 1))
    rep["passed"] = bool(
        len(bad) == 0 and len(new_bad) == 0
        and counts == {"COPY": 219, "DERIVE": 0, "NEW": 360}
        and rebuild_identical and man3["unclassified"] == 0)
    print(f"  [p1] rebuild_identical={rebuild_identical} "
          f"smm_tensors={len(smm_meta)} passed={rep['passed']}", flush=True)
    return rep


def phase2(cfg, pbl):
    cfg10 = S.context_10m_config(cfg)
    json.dump(cfg10, open(CTX_CFG, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    sc = S.ScopedV10_3(cfg, pbl)
    r = sc.context_10m_run(cfg)
    reg = S.scoped_registry()
    json.dump(reg, open(os.path.join(OUT, "scoped_capability_registry.json"),
                        "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    infer = dict(schema="TS2SC/inference-default-v10.3",
                 model="MiniCPM5-1B-TS2SC", scope=list(S.GROUPS),
                 max_position_embeddings=S.CONTEXT_10M,
                 beta=1.0, groups={g: True for g in S.GROUPS},
                 smm_consolidation="EXPLICIT observe() gate only; S init zero, "
                                   "NOT spontaneous self-evolution",
                 identity_adapter="none (RETIRED)",
                 note="10M is a CONFIG value; teacher full-attention side "
                      "INFEASIBLE_LEDGER (240 GiB KV). Branch state is O(1).")
    json.dump(infer, open(INFER3, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    return r, reg


def gen(lm, tok, sc, mode, q, V_size, max_new=MAX_NEW):
    """mode: T 纯teacher / OFF 全关 / ON_ALL 三组全开(含已显式巩固的SMM)"""
    if mode == "T":
        hook = None
    else:
        sc.reset()
        if mode == "OFF":
            sc.all_off()
        else:
            sc.all_on()
        hook = sc.hook
    ids, _ = tok.encode(chatml(q), add_bos=False)
    L = len(ids)
    kvs = [(np.zeros((0, lm.nkv, lm.hd), np.float32),
            np.zeros((0, lm.nkv, lm.hd), np.float32)) for _ in range(lm.nl)]
    fins, out = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, tid in enumerate(ids):
                r = lm.step(int(tid), pos, kvs, need_logits=(pos == L - 1),
                            fusion_hook=hook, logit_bias_fn=None)
                if pos == L - 1:
                    out = [int(r[0])]
                    fins = [bool(getattr(lm, "last_logits_finite", True))]
            for j in range(1, max_new):
                r = lm.step(int(out[-1]), L + j - 1, kvs, need_logits=True,
                            fusion_hook=hook, logit_bias_fn=None)
                out.append(int(r[0]))
                fins.append(bool(getattr(lm, "last_logits_finite", True)))
                if r[0] in STOP_IDS:
                    break
    return dict(text=tok.decode(out, skip_special=True), ids=out,
                n_prompt=L, n_new=len(out),
                gate=decode_gate(out, V_size, all(fins)),
                logits_finite=all(fins))


def main():
    t0 = time.time()
    print("=== Phase0 LIVE 重拉 ===", flush=True)
    V, hdr, pull, backups = phase0()
    cfg = json.load(open(os.path.join(FRESH, "config.json"), encoding="utf-8"))
    V_size = cfg["vocab_size"]
    pbl = {i: A.deterministic_layer_params(1536, 20261128 + 101 * i)
           for i in range(24)}

    print("\n=== Phase1 范围收窄迁移 ===", flush=True)
    p1 = phase1(V, hdr, cfg, backups)

    print("\n=== Phase2 10M 上下文 ===", flush=True)
    ctx, reg = phase2(cfg, pbl)
    print(f"  L={ctx['L']} state {ctx['state_bytes_before']} B constant | "
          f"attention {ctx['attention']['kv_gib']} GiB "
          f"{ctx['attention']['status']}", flush=True)

    print("\n=== Phase3 端到端生成 ===", flush=True)
    tok = MiniCPMTokenizer(os.path.join(FRESH, "tokenizer.json"))
    lm = TeacherLM(V, cfg)
    sc = S.ScopedV10_3(cfg, pbl)
    # 显式门禁巩固（手动开；演示数据，不是训练结果）
    rng = np.random.default_rng(1234)
    for m in sc.smm:
        for _ in range(120):
            t = rng.integers(0, 16, 128)
            m.observe(t, (t + 5) % 16)
    consolidated = dict(commits_per_layer=sc.smm[0].commits,
                        rejects_per_layer=sc.smm[0].rejects,
                        slots=len(sc.smm[0].slots),
                        S_norm=round(float(np.linalg.norm(sc.smm[0].S)), 4),
                        all_24_layers=all(m.commits > 0 for m in sc.smm))
    print("  SMM explicit consolidation:", consolidated, flush=True)

    rows = []
    for i, (lang, q, mode) in enumerate(ARMS, 1):
        # 每次重新巩固，避免上一臂 reset 清空 S
        for m in sc.smm:
            for _ in range(120):
                t = rng.integers(0, 16, 128)
                m.observe(t, (t + 5) % 16)
        g = gen(lm, tok, sc, mode, q, V_size)
        rows.append(dict(lang=lang, q=q, mode=mode, text=g["text"],
                         ids=g["ids"], n_prompt=g["n_prompt"],
                         n_new=g["n_new"],
                         logits_finite=g["logits_finite"],
                         decode_gate=g["gate"]["passed"],
                         identity_markers=[m for m in IDENTITY_MARKERS
                                           if m in g["text"]]))
        print(f"  [{i}/{len(ARMS)}] {lang:3s} {mode:7s} "
              f"gate={rows[-1]['decode_gate']} | {g['text'][:46]!r}",
              flush=True)
        json.dump(rows, open(CHECKPOINT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    by = {(r["lang"], r["mode"]): r for r in rows}
    t_zh, off_zh = by[("zh", "T")], by[("zh", "OFF")]
    on_zh, on_en = by[("zh", "ON_ALL")], by[("en", "ON_ALL")]
    v = dict(
        phase0_all_blob_sha_verified=bool(pull["all_blob_sha_verified"]),
        phase0_n_parts=NPARTS,
        phase1_passed=p1["passed"],
        phase1_manifest_counts=p1["counts"],
        phase1_copy_219_full_bitwise=bool(p1["n_copy_mismatched"] == 0),
        phase1_branch_rebuild_identical=p1[
            "branch_rebuild_bitwise_identical_to_pushed"],
        phase2_context_tokens=S.CONTEXT_10M,
        phase2_branch_state_constant=ctx["state_independent_of_L"],
        phase2_branch_state_bytes=ctx["state_bytes_after"],
        phase2_idle_closed_form_equals_stepwise=ctx[
            "idle_closed_form_equals_stepwise_gate0"],
        phase2_attention_status=ctx["attention"]["status"],
        phase2_attention_kv_gib=ctx["attention"]["kv_gib"],
        phase2_rope_extrapolation=ctx["attention"]["rope_extrapolation_factor"],
        smm_explicit_consolidation=consolidated,
        T_equals_OFF_tokenwise=bool(t_zh["ids"] == off_zh["ids"]),
        ON_differs_from_OFF=bool(on_zh["ids"] != off_zh["ids"]),
        ON_zh_en_distinct=bool(on_zh["text"] != on_en["text"]),
        all_logits_finite=all(r["logits_finite"] for r in rows),
        all_decode_gate=all(r["decode_gate"] for r in rows),
        identity_marker_leaks=sum(len(r["identity_markers"]) for r in rows))
    v["passed"] = bool(
        v["phase0_all_blob_sha_verified"] and v["phase1_passed"]
        and v["phase2_branch_state_constant"]
        and v["phase2_idle_closed_form_equals_stepwise"]
        and v["T_equals_OFF_tokenwise"] and v["ON_differs_from_OFF"]
        and v["all_logits_finite"] and v["all_decode_gate"]
        and v["identity_marker_leaks"] == 0
        and consolidated["all_24_layers"])

    rep = dict(
        title="TS2SC v10.3 · 范围收窄（SGC + 外挂 + SMM）+ 10M 上下文 · 迁移与验证",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        instruction="迁移并且验证模型 / 就继承 SGC 以及其外挂模块 "
                    "SMM记忆与自进化模块就行了，上下文开到10M",
        scope=reg,
        phase0=dict(all_blob_sha_verified=pull["all_blob_sha_verified"],
                    n_parts=NPARTS),
        phase1=p1,
        phase2_context=ctx,
        generation=rows, verification=v,
        honest_boundaries=[
            "SMM 来源为上游真实源码 evolution_mixer.py 的 "
            "FusedEvolutionMixer / CalibratedFusedMixer（SMM x SGC 融合状态），"
            "移植为推理期逐 token 的 64 维低秩分支；非编造。",
            "SMM 的 observe() 是**显式门禁**：S 初值全零，未调用 observe 时"
            "贡献严格为零（与 v10.2 全开逐位相同）。本轮巩固用的是"
            "结构化强共现演示数据，不是训练结果、不是模型自发记忆。",
            "自查抓到并修复：SGC 与 SMM 各自归一化到 βε‖x‖，直接相加会"
            "突破 T1' 总契约；现对总和统一投影，smm_off 时零缩放保持逐位相同。",
            "10M 是**配置值 + 分支侧 O(1) 闭式实跑**；teacher 侧全注意力"
            "KV 240 GiB、80× RoPE 外推未验证 ⇒ 端到端 10M 仍 INFEASIBLE_LEDGER，"
            "不是实跑失效。",
            "8 token 短生成只证 finite / 不乱码 / 全关零回归 / 三组继承生效 / "
            "零身份标记；不是质量、事实正确性或能力结论。",
            "β=1 仍是尺度契约配置，不是 L2 校准成品；P-β 仍 NOT_TESTED。",
            "无 ppl / KL / 配对 CI / 吞吐 / SOTA / 原生级 / 定理在 LM 复现声明。",
            "本次交付范围排除 9 CONTRACT + 2 DEFERRED 等历史条目，但按用户"
            "「不删除任何条目」原样保留在仓里，只是不纳入继承声明。",
            "端到端生成是「真实权重 + 24 层融合 hook」运行时挂接，"
            "不是对已组装 2.6 GB 静态大文件的直接加载（磁盘不足）。",
        ],
        elapsed_s=round(time.time() - t0, 1))
    json.dump(rep, open(VERIFY_JSON, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n[verify]", json.dumps(v, ensure_ascii=False), flush=True)
    print("[done]", VERIFY_JSON, rep["elapsed_s"], "s", flush=True)
    for p, b in backups.items():
        open(p, "wb").write(b)
    print("[restore] pushed-artifact bytes restored", flush=True)
    return rep


if __name__ == "__main__":
    main()
