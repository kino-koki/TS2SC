#!/usr/bin/env python3
"""
性能测试：SimplCore vs Transformer 单层时延 / 复杂度 / O(1) 状态

为什么测单层：
  0.6646B 全模型在 2 核 CPU + 2GB 内存下放不下，
  但**复杂度差异是逐层成立的**，单层即可外推全模型。

测三项：
  ① 时延 vs 长度（2 的幂，3 次取中位数）
  ② 复杂度指数（log-log 回归斜率）
  ③ O(1) 状态 @ 长上下文（含蒸馏后权重）
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

for p in ("/data/workspace/SimplCore_official", "/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0, p)
import torch
import torch.nn.functional as F
torch.set_num_threads(2)

WS = Path("/data/workspace")
OUT = WS / "MIGREAL"; RES = OUT / "result.json"
SRC = WS / "mig" / "model.safetensors"
D, FF, H, KV, HD, BASES = 1024, 3072, 16, 8, 128, 32
SMALL_V = 4096


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "perf.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def save(k, v):
    d = json.loads(RES.read_text()) if RES.exists() else {}
    d[k] = v
    RES.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def rms(x, w, eps=1e-6):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return x.float() * torch.rsqrt(v + eps) * w.float()


def rope(t, pos, theta=1_000_000.0):
    d = t.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = torch.outer(pos.float(), inv)
    cos, sin = torch.cos(ang)[None, None], torch.sin(ang)[None, None]
    x1, x2 = t.float()[..., 0::2], t.float()[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                       -1).flatten(-2).to(t.dtype)


def teacher_block(x, w, pos):
    B, Ln, _ = x.shape
    h = rms(x, w["input_layernorm.weight"])
    q = F.linear(h, w["self_attn.q_proj.weight"].float())
    k = F.linear(h, w["self_attn.k_proj.weight"].float())
    v = F.linear(h, w["self_attn.v_proj.weight"].float())
    q = q.view(B, Ln, H, HD).transpose(1, 2)
    k = k.view(B, Ln, KV, HD).transpose(1, 2)
    v = v.view(B, Ln, KV, HD).transpose(1, 2)
    q = rms(q, w["self_attn.q_norm.weight"])
    k = rms(k, w["self_attn.k_norm.weight"])
    q, k = rope(q, pos), rope(k, pos)
    rep = H // KV
    k = k.unsqueeze(2).expand(B, KV, rep, Ln, HD).reshape(B, H, Ln, HD)
    v = v.unsqueeze(2).expand(B, KV, rep, Ln, HD).reshape(B, H, Ln, HD)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
    m = torch.triu(torch.ones(Ln, Ln, dtype=torch.bool), 1)
    a = torch.softmax(att.masked_fill(m, float("-inf")), -1)
    o = (a @ v).transpose(1, 2).reshape(B, Ln, H * HD)
    x2 = x.float() + F.linear(o, w["self_attn.o_proj.weight"].float())
    h2 = rms(x2, w["post_attention_layernorm.weight"])
    g = F.linear(h2, w["mlp.gate_proj.weight"].float())
    u = F.linear(h2, w["mlp.up_proj.weight"].float())
    return x2 + F.linear(F.silu(g) * u, w["mlp.down_proj.weight"].float())


def build_unit():
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c = get_preset("tuan-30m")
    c.dim = D; c.num_layers = 1; c.num_bases = BASES; c.num_heads = 8
    c.vocab_size = SMALL_V; c.seq_len = 4096; c.ffn_ratio = 3.0
    return SimplCoreModel(c)


def load_block(unit, mf, mmap, li):
    sd = unit.state_dict(); n = 0
    for k in list(sd.keys()):
        if not k.startswith("blocks.0."):
            continue
        src = k.replace("blocks.0.", f"blocks.{li}.", 1)
        if src in mmap:
            t = mf[mmap[src]].get_tensor(src)
            if tuple(t.shape) == tuple(sd[k].shape):
                with torch.no_grad():
                    sd[k].copy_(t)
                n += t.numel()
    # 装入蒸馏后的 mixer（若有）
    ck = OUT / "ck" / f"mixer_l{li}.pt"
    if ck.exists():
        try:
            st = torch.load(ck, map_location="cpu")
            msd = unit.blocks[0].mixer.state_dict()
            for k, v in st["mixer"].items():
                if k in msd and tuple(msd[k].shape) == tuple(v.shape):
                    msd[k].copy_(v)
            unit.blocks[0].mixer.load_state_dict(msd)
            log(f"    ✓ 已装入蒸馏后 mixer（layer {li}）")
        except Exception as e:
            log(f"    ⚠️ 蒸馏权重载入失败 {type(e).__name__}")
    return n


def bench(fn, x, reps=3):
    """中位数时延（ms）"""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(x)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def slope(Ls, Ts):
    """log-log 回归斜率"""
    lx = [math.log(l) for l in Ls]
    ly = [math.log(t) for t in Ts]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


def main():
    log("=" * 66)
    log("[perf] SimplCore vs Transformer —— 时延 / 复杂度 / O(1)")
    log("=" * 66)
    from safetensors import safe_open
    sf = safe_open(str(SRC), framework="pt", device="cpu")
    shards = sorted(OUT.glob("model-*.safetensors"))
    mf = [safe_open(str(p), framework="pt", device="cpu") for p in shards]
    mmap = {k: i for i, f in enumerate(mf) for k in f.keys()}

    LI = int(os.getenv("PERF_LAYER", "0"))
    p = f"model.layers.{LI}."
    w = {k.split(".")[-2] + "." + k.split(".")[-1] if False else k[len(p):]:
         sf.get_tensor(k) for k in [p + s for s in (
             "input_layernorm.weight", "post_attention_layernorm.weight",
             "self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.o_proj.weight",
             "self_attn.q_norm.weight", "self_attn.k_norm.weight",
             "mlp.gate_proj.weight", "mlp.up_proj.weight",
             "mlp.down_proj.weight")]}
    log(f"  teacher 权重已载入（layer {LI}）")

    u = build_unit()
    load_block(u, mf, mmap, LI)
    u.eval()

    emb = sf.get_tensor("model.embed_tokens.weight")
    sd_ = float(emb.float().std()); del emb
    g = torch.Generator().manual_seed(11)

    Ls = [64, 128, 256, 512, 1024]
    log(f"\n  ① 时延 vs 长度（中位数，3 次）")
    log(f"\n     {'L':>6} {'SimplCore':>12} {'Transformer':>13} {'加速比':>9}")
    log("     " + "-" * 44)
    rows = []
    for Ln in Ls:
        x = torch.randn(1, Ln, D, generator=g) * sd_
        pos = torch.arange(Ln)
        ts = bench(lambda z: u.blocks[0](z), x)
        tt = bench(lambda z: teacher_block(z, w, pos), x)
        rows.append(dict(L=Ln, sc=round(ts, 2), tf=round(tt, 2),
                         speedup=round(tt / ts, 2)))
        log(f"     {Ln:>6} {ts:>10.1f}ms {tt:>11.1f}ms "
            f"{tt/ts:>8.2f}×")
    log("     " + "-" * 44)

    s_sc = slope([r["L"] for r in rows], [r["sc"] for r in rows])
    s_tf = slope([r["L"] for r in rows], [r["tf"] for r in rows])
    log(f"\n  ② 复杂度指数（log-log 斜率）")
    log(f"     SimplCore     {s_sc:.3f}   理论 O(L log L)")
    log(f"     Transformer   {s_tf:.3f}   理论 O(L²)")
    lo, hi = rows[0], rows[-1]
    log(f"     L 涨 {hi['L']//lo['L']}× → "
        f"Transformer 耗时涨 {hi['tf']/lo['tf']:.1f}×，"
        f"SimplCore 只涨 {hi['sc']/lo['sc']:.1f}×")

    log(f"\n  ③ O(1) 状态 @ 长上下文")
    st_rows = []
    for Ln in (512, 2048, 8192):
        try:
            with torch.no_grad():
                u(torch.randint(0, SMALL_V, (1, Ln)))
            b = int(u.stream_bytes(1))
            st_rows.append(dict(L=Ln, state=b))
            log(f"     L={Ln:>5} → {b:,} B")
        except Exception as e:
            log(f"     L={Ln:>5} ✗ {type(e).__name__}: {str(e)[:40]}")
            st_rows.append(dict(L=Ln, state=-1))
    uniq = len({r["state"] for r in st_rows if r["state"] > 0})
    log(f"     唯一状态数 {uniq} {'✅ O(1)' if uniq == 1 else '❌'}")
    kv = 2 * 2 * KV * HD * 28
    if st_rows and st_rows[-1]["state"] > 0:
        log(f"     ★ 全模型 28 层：本模型 {st_rows[-1]['state']*28:,} B 恒定")
        log(f"       Qwen3 KV @8192 = {kv*8192/2**30:.2f} GiB（随 L 线性增长）")

    save("perf", dict(layer=LI, rows=rows,
                      slope_sc=round(s_sc, 3), slope_tf=round(s_tf, 3),
                      o1=st_rows, o1_unique=uniq))
    log("\n[perf] 完成")


if __name__ == "__main__":
    main()
