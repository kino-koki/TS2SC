#!/usr/bin/env python3
"""
快速蒸馏 mixer —— 可续跑（沙箱重置后从检查点继续）

设计要点（针对 2 核 CPU + 频繁重置）：
  * 小配置：SEQ=32, 训练样本 6，每步约 0.5s
  * 每 25 步写一次检查点（mixer 权重 + optimizer + 步数）
  * 启动时自动 resume
  * 只训 mixer，FFN/emb/norm 冻结（86.6% 参数零开销）
  * 外层用 shell 循环反复拉起，累计跑到目标步数

用法：
  DM_LAYER=0 DM_TOTAL=1500 bash run_fast_distill.sh
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
OUT = WS / "MIGREAL"
RES = OUT / "result.json"
SRC = WS / "mig" / "model.safetensors"
CKDIR = OUT / "ck"; CKDIR.mkdir(exist_ok=True)

D, FF, L, H, KV, HD, V, BASES = 1024, 3072, 28, 16, 8, 128, 151936, 32
SMALL_V = 4096
SEQ = int(os.getenv("DM_SEQ", "32"))
NTR = int(os.getenv("DM_N", "6"))
TOTAL = int(os.getenv("DM_TOTAL", "1500"))
CHUNK = int(os.getenv("DM_CHUNK", "25"))     # 每轮跑多少步就存档
LR = float(os.getenv("DM_LR", "3e-3"))
LI = int(os.getenv("DM_LAYER", "0"))
CK = CKDIR / f"mixer_l{LI}.pt"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "fastdist.log", "a", encoding="utf-8") as f:
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


def teacher_block(x, w, i, pos):
    p = f"model.layers.{i}."
    B, Ln, _ = x.shape
    h = rms(x, w[p + "input_layernorm.weight"])
    q = F.linear(h, w[p + "self_attn.q_proj.weight"].float())
    k = F.linear(h, w[p + "self_attn.k_proj.weight"].float())
    v = F.linear(h, w[p + "self_attn.v_proj.weight"].float())
    q = q.view(B, Ln, H, HD).transpose(1, 2)
    k = k.view(B, Ln, KV, HD).transpose(1, 2)
    v = v.view(B, Ln, KV, HD).transpose(1, 2)
    q = rms(q, w[p + "self_attn.q_norm.weight"])
    k = rms(k, w[p + "self_attn.k_norm.weight"])
    q, k = rope(q, pos), rope(k, pos)
    rep = H // KV
    k = k.unsqueeze(2).expand(B, KV, rep, Ln, HD).reshape(B, H, Ln, HD)
    v = v.unsqueeze(2).expand(B, KV, rep, Ln, HD).reshape(B, H, Ln, HD)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
    m = torch.triu(torch.ones(Ln, Ln, dtype=torch.bool, device=x.device), 1)
    a = torch.softmax(att.masked_fill(m, float("-inf")), -1)
    o = (a @ v).transpose(1, 2).reshape(B, Ln, H * HD)
    x2 = x.float() + F.linear(o, w[p + "self_attn.o_proj.weight"].float())
    h2 = rms(x2, w[p + "post_attention_layernorm.weight"])
    gg = F.linear(h2, w[p + "mlp.gate_proj.weight"].float())
    uu = F.linear(h2, w[p + "mlp.up_proj.weight"].float())
    return x2 + F.linear(F.silu(gg) * uu, w[p + "mlp.down_proj.weight"].float())


def build():
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c = get_preset("tuan-30m")
    c.dim = D; c.num_layers = 1; c.num_bases = BASES; c.num_heads = 8
    c.vocab_size = SMALL_V; c.seq_len = 256; c.ffn_ratio = 3.0
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
    return n


def cos(a, b):
    return float(F.cosine_similarity(a.reshape(-1).float(),
                                     b.reshape(-1).float(), dim=0))


def main():
    log("=" * 66)
    log(f"[fast-distill] layer {LI}  目标 {TOTAL} 步  lr={LR}")
    log("=" * 66)
    from safetensors import safe_open
    sf = safe_open(str(SRC), framework="pt", device="cpu")
    shards = sorted(OUT.glob("model-*.safetensors"))
    mf = [safe_open(str(p), framework="pt", device="cpu") for p in shards]
    mmap = {k: i for i, f in enumerate(mf) for k in f.keys()}

    p = f"model.layers.{LI}."
    w = {k: sf.get_tensor(k) for k in [p + s for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight")]}
    pos = torch.arange(SEQ)
    emb = sf.get_tensor("model.embed_tokens.weight")
    sd_ = float(emb.float().std()); del emb

    g = torch.Generator().manual_seed(5)
    Xtr = [torch.randn(1, SEQ, D, generator=g) * sd_ for _ in range(NTR)]
    Xva = [torch.randn(1, SEQ, D, generator=g) * sd_ for _ in range(4)]
    with torch.no_grad():
        Ytr = [teacher_block(x, w, LI, pos) for x in Xtr]
        Yva = [teacher_block(x, w, LI, pos) for x in Xva]
    del w

    u = build()
    load_block(u, mf, mmap, LI)
    # 冻结 FFN + norm
    for nm, pp_ in u.named_parameters():
        if nm.startswith("blocks.0.ffn") or nm.endswith("norm.weight"):
            pp_.requires_grad_(False)
    if hasattr(u.blocks[0].mixer, "norm"):
        u.blocks[0].mixer.norm.weight.requires_grad_(False)
    tr = [pp_ for pp_ in u.parameters() if pp_.requires_grad]
    ntr = sum(pp_.numel() for pp_ in tr)

    opt = torch.optim.AdamW(tr, lr=LR)
    step0 = 0
    best = -1.0
    if CK.exists():
        try:
            st = torch.load(CK, map_location="cpu")
            msd = u.blocks[0].mixer.state_dict()
            for k, v in st["mixer"].items():
                if k in msd and tuple(msd[k].shape) == tuple(v.shape):
                    msd[k].copy_(v)
            u.blocks[0].mixer.load_state_dict(msd)
            opt.load_state_dict(st["opt"])
            step0 = st["step"]; best = st.get("best", -1.0)
            log(f"  [resume] 从 {step0} 步继续（历史最佳 {best:.4f}）")
        except Exception as e:
            log(f"  [resume] 失败 {type(e).__name__}，从头开始")

    if step0 == 0:
        with torch.no_grad():
            c0 = sum(cos(Yva[i], u.blocks[0](Xva[i]))
                     for i in range(len(Xva))) / len(Xva)
        log(f"  起始端到端余弦 {c0:.4f}   可训练 {ntr/1e6:.2f}M")
        save(f"fd_l{LI}_start", round(c0, 4))
    else:
        c0 = best

    n_chunk = min(CHUNK, TOTAL - step0)
    if n_chunk <= 0:
        log(f"  已完成 {step0}/{TOTAL} 步，无需继续")
        return

    sc = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL, eta_min=LR * 0.05)
    for _ in range(step0):
        sc.step()          # 对齐 scheduler

    t0 = time.time()
    u.train()
    for s in range(step0 + 1, step0 + n_chunk + 1):
        tot_l = 0.0
        for xi, yi in zip(Xtr, Ytr):
            o = u.blocks[0](xi)
            o = o[0] if isinstance(o, tuple) else o
            l = F.mse_loss(o.float(), yi.float())
            opt.zero_grad(set_to_none=True); l.backward()
            torch.nn.utils.clip_grad_norm_(tr, 1.0)
            opt.step(); sc.step()
            tot_l += float(l)
        if s % 5 == 0 or s == step0 + n_chunk:
            u.eval()
            with torch.no_grad():
                cv = sum(cos(Yva[i], u.blocks[0](Xva[i]))
                         for i in range(len(Xva))) / len(Xva)
            best = max(best, cv)
            log(f"  step {s:>5}/{TOTAL}  loss {tot_l/NTR:.6f}  "
                f"val余弦 {cv:.4f}  最佳 {best:.4f}  [{time.time()-t0:.0f}s]")
            u.train()
        # 每 5 步存一次，重置损失最多 5 步
        if s % 5 == 0 or s == step0 + n_chunk:
            torch.save({"mixer": u.blocks[0].mixer.state_dict(),
                        "opt": opt.state_dict(), "step": s, "best": best}, CK)

    save(f"fd_l{LI}", dict(step=step0 + n_chunk, total=TOTAL,
                           best=round(best, 4), trainable=ntr))
    log(f"  本轮结束 step={step0+n_chunk}  最佳余弦 {best:.4f}")


if __name__ == "__main__":
    main()
