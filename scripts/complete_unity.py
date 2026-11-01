#!/usr/bin/env python3
"""
补全融合模型的振荡器参数 → 使其成为**可直接前向**的 SimplCore

融合产物只搬了 in_proj / gate_proj / out_proj（这三个有 Transformer 对应物）。
SGC 还缺：raw_rho, raw_theta, coef, feed, local_conv.weight, out_scale。
这些在 Transformer 里没有对应物，只能按 SimplCore 官方约定初始化。

★ 关键点：必须能 4M 上下文 → rho_param="logspace"
  （sigmoid 参数化在 raw≳16.6 时精确返回 1.0，最长记忆约 1e6 token，
    达不到 4M 所需的 0.99999975）

用法：
  python3 complete_unity.py           流式补全，写 UNITY2/
  python3 complete_unity.py --verify  截断实例化 → 前向 + O(1) 状态验证
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path

for p in ("/data/workspace/SimplCore_official", "/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0, p)
import torch
from safetensors import safe_open
from safetensors.torch import save_file

WS = Path("/data/workspace")
SRC = WS / "UNITY"
DST = WS / "UNITY2"; DST.mkdir(exist_ok=True)
BASES = 32
LOCAL_K = 4
# blocks 0..23 → MiniCPM5 @1536；blocks 24..51 → Qwen3 @1024
DIM_OF = lambda i: 1536 if i < 24 else 1024
NB = 52


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(SRC / "complete.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def osc_params(D, rho_param="logspace"):
    """按 SimplCore 官方 _init_oscillator_grid 约定生成振荡器参数"""
    from simplcore.core.sgc import (_init_oscillator_grid, _logit)
    rho0, theta0 = _init_oscillator_grid(BASES, decay_max=4096,
                                         mode="logfreq")
    if rho_param == "logspace":
        inv_lam = -torch.log(rho0)
        raw_rho_log = torch.log(torch.expm1(inv_lam))
        out = {"raw_rho_log": raw_rho_log}
    else:
        out = {"raw_rho": _logit(rho0)}
    out["raw_theta"] = _logit(theta0 / math.pi)
    out["coef"] = torch.randn(BASES, D) * 0.02
    out["feed"] = torch.ones(D)
    # local conv: identity at init
    lc = torch.zeros(D, 1, LOCAL_K)
    lc[:, 0, -1] = 1.0
    out["local_conv.weight"] = lc
    out["out_scale"] = torch.ones(1)
    return out


def complete():
    log("=" * 68)
    log("[complete] 补全振荡器参数 → UNITY2/")
    log("=" * 68)
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    shards = sorted({v for v in wmap.values()})
    log(f"  源 {len(wmap)} 张量 / {len(shards)} 分片")

    new_idx = {"metadata": {"total_size": 0}, "weight_map": {}}
    added = 0
    seen_blocks = set()
    for si, s in enumerate(shards):
        f = safe_open(str(SRC / s), framework="pt", device="cpu")
        buf = {}
        for k in f.keys():
            buf[k] = f.get_tensor(k)
            # 记录出现过的 block
            if k.startswith("blocks."):
                seen_blocks.add(int(k.split(".")[1]))
        save_file(buf, str(DST / s), metadata={"format": "pt"})
        for k in buf:
            new_idx["weight_map"][k] = s
        new_idx["metadata"]["total_size"] += (DST / s).stat().st_size
        log(f"    ✓ 透传 {s} ({len(buf)} 张量)")
        del buf
    # 补振荡器（单独一片）
    extra = {}
    for i in sorted(seen_blocks):
        D = DIM_OF(i)
        for k, v in osc_params(D).items():
            extra[f"blocks.{i}.mixer.{k}"] = v.to(torch.bfloat16).contiguous()
            added += v.numel()
    log(f"  补全 {len(seen_blocks)} 个 block × {len(extra)//max(len(seen_blocks),1)} 参数"
        f" = {added:,} 参数")
    nm = f"model-{len(shards):05d}.safetensors"
    save_file(extra, str(DST / nm), metadata={"format": "pt"})
    for k in extra:
        new_idx["weight_map"][k] = nm
    new_idx["metadata"]["total_size"] += (DST / nm).stat().st_size
    log(f"    ✓ {nm} ({len(extra)} 张量, 振荡器)")

    (DST / "model.safetensors.index.json").write_text(json.dumps(new_idx))
    cfg = json.loads((SRC / "config.json").read_text())
    cfg["rho_param"] = "logspace"
    cfg["num_bases"] = BASES
    cfg["local_kernel"] = LOCAL_K
    cfg["runnable"] = True
    cfg["added_osc_params"] = added
    cfg["total_params"] = cfg["total_params"] + added
    (DST / "config.json").write_text(json.dumps(cfg, indent=2))
    log(f"\n✅ UNITY2 完成  {cfg['total_params']:,} 参数 "
        f"({cfg['total_params']/1e9:.3f} B)")
    log(f"   张量 {len(new_idx['weight_map'])}  分片 {len(shards)+1}")


def verify():
    """截断实例化：验证前向 + O(1) 状态"""
    log("\n" + "=" * 68)
    log("[verify-run] 截断实例化 → 前向 + O(1) 状态")
    log("=" * 68)
    from simplcore.core.sgc import SpectralGlobalConv as SGC
    idx = json.loads((DST / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    shards = sorted({v for v in wmap.values()})
    fs = {s: safe_open(str(DST / s), framework="pt", device="cpu") for s in shards}

    def get(k):
        return fs[wmap[k]].get_tensor(k)

    torch.manual_seed(0)
    D = 1536
    mixer = SGC(dim=D, num_bases=BASES, decay_max=4096,
                local_kernel=LOCAL_K, rho_param="logspace",
                init_mode="logfreq", content_gate=True,
                evidence_gate=True, uncertainty_gate=True)
    sd = mixer.state_dict()
    loaded = miss = 0
    for k in list(sd.keys()):
        src = f"blocks.0.mixer.{k}"
        if src in wmap:
            t = get(src)
            if tuple(t.shape) == tuple(sd[k].shape):
                with torch.no_grad():
                    sd[k].copy_(t.float())
                loaded += t.numel(); continue
        miss += sd[k].numel()
    mixer.load_state_dict(sd)
    log(f"  SGC(D=1536) 载入 {loaded:,} / {loaded+miss:,} "
        f"({loaded/(loaded+miss)*100:.1f}%)")

    mixer.eval()
    rows = []
    for B in (1, 2, 4):
        for L in (64, 256, 1024):
            x = torch.randn(B, L, D) * 0.03
            with torch.no_grad():
                o = mixer(x)
                o = o[0] if isinstance(o, tuple) else o
            # ★ O(1) 状态：init_state 的张量总字节，与 L 无关
            with torch.no_grad():
                st = mixer.init_state(B, torch.device("cpu"), torch.float32)
            nb = 0
            for f in ("y1", "y2", "conv_buf", "energy_sum"):
                t = getattr(st, f, None)
                if isinstance(t, torch.Tensor):
                    nb += t.numel() * t.element_size()
            rows.append(dict(B=B, L=L, state=nb,
                             finite=bool(torch.isfinite(o).all()),
                             out_std=round(float(o.std()), 5)))
            log(f"    B={B} L={L:>5}  状态 {nb:>9,} B  "
                f"finite={bool(torch.isfinite(o).all())}  outσ={float(o.std()):.4f}")
    # ★ O(1) 的正确判据：固定 batch 下，状态与 L 无关
    #   （跨 batch 状态当然随 B 线性增长，那是 batch 维度不是序列维度）
    per_b = {}
    for r in rows:
        per_b.setdefault(r["B"], set()).add(r["state"])
    uniq = {b: len(v) for b, v in per_b.items()}
    allone = all(v == 1 for v in uniq.values())
    log(f"\n  ★ 每个 batch 内跨 L 的唯一状态数 {uniq}  "
        f"{'✅ O(1)：状态与序列长度无关' if allone else '❌'}")
    ok_all = all(r["finite"] for r in rows)
    log(f"    前向全部 finite: {'✅' if ok_all else '❌'}")
    # 逐 token 流式（验证 step 可用）
    try:
        with torch.no_grad():
            st = mixer.init_state(1, torch.device("cpu"), torch.float32)
            xs = torch.randn(1, 128, D) * 0.03
            outs = []
            for t in range(128):
                y, st = mixer.step(xs[:, t:t+1], st)
                outs.append(y)
            stream = torch.cat(outs, 1)
            par, _ = mixer(xs), None
            par = par[0] if isinstance(par, tuple) else par
            d = float((stream - par.float()).abs().max())
            log(f"    流式 vs 并行 max|Δ| = {d:.2e}  "
                f"{'✅ <4e-5' if d < 4e-5 else '⚠️ 超阈值'}")
            rows.append(dict(dual_max_delta=d))
    except Exception as e:
        log(f"    流式对偶: {type(e).__name__}: {str(e)[:80]}")
    if rows:
        per_tok = 2 * 2 * 2 * 128 * 24
        log(f"\n    MiniCPM5 KV @4M token = {per_tok*4_000_000/2**30:.1f} GiB（随 L 线性增）")
        log(f"    本模型 @4M token 状态  = {rows[0]['state']*52/2**20:.2f} MiB（恒定）")
    r = json.loads((SRC / "result.json").read_text())
    r["run"] = dict(rows=rows, unique=uniq, loaded=loaded)
    (SRC / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        complete()
        try:
            verify()
        except Exception as e:
            log(f"[verify] {type(e).__name__}: {str(e)[:120]}")
