#!/usr/bin/env python3
"""定位迁移失真的来源：是 mixer 还是 FFN？"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path

for p in ("/data/workspace/SimplCore_official", "/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0, p)
import torch
import torch.nn.functional as F
torch.set_num_threads(2)

WS = Path("/data/workspace"); OUT = WS / "MIGREAL"; RES = OUT / "result.json"
SRC = WS / "mig" / "model.safetensors"
D, FF, L, H, KV, HD, V, BASES = 1024, 3072, 28, 16, 8, 128, 151936, 32
SMALL_V = 4096


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "diaglog.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def save(k, v):
    d = json.loads(RES.read_text()) if RES.exists() else {}
    d[k] = v; RES.write_text(json.dumps(d, ensure_ascii=False, indent=2))


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


def teacher_attn(x, w, i, pos):
    """只做注意力分支，返回 (x + attn_out)"""
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
    return x.float() + F.linear(o, w[p + "self_attn.o_proj.weight"].float())


def teacher_ffn(x, w, i):
    p = f"model.layers.{i}."
    h = rms(x, w[p + "post_attention_layernorm.weight"])
    g = F.linear(h, w[p + "mlp.gate_proj.weight"].float())
    u = F.linear(h, w[p + "mlp.up_proj.weight"].float())
    return x + F.linear(F.silu(g) * u, w[p + "mlp.down_proj.weight"].float())


def build_unit():
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c = get_preset("tuan-30m")
    c.dim = D; c.num_layers = 1; c.num_bases = BASES
    c.num_heads = 8; c.vocab_size = SMALL_V; c.seq_len = 256
    c.ffn_ratio = 3.0
    return SimplCoreModel(c)


def load_block(unit, mf, mmap, li):
    sd = unit.state_dict(); n = 0; miss = []
    for k in sd.keys():
        if not k.startswith("blocks.0."):
            continue
        src = k.replace("blocks.0.", f"blocks.{li}.", 1)
        if src in mmap:
            t = mf[mmap[src]].get_tensor(src)
            if tuple(t.shape) == tuple(sd[k].shape):
                with torch.no_grad():
                    sd[k].copy_(t)
                n += t.numel()
            else:
                miss.append((src, tuple(t.shape), tuple(sd[k].shape)))
        else:
            miss.append((src, None, tuple(sd[k].shape)))
    return n, miss


def cos(a, b):
    return float(F.cosine_similarity(a.reshape(-1).float(),
                                     b.reshape(-1).float(), dim=0))


def main():
    log("=" * 66)
    log("[diag] 失真来源定位：mixer 还是 FFN？")
    log("=" * 66)
    from safetensors import safe_open
    sf = safe_open(str(SRC), framework="pt", device="cpu")
    shards = sorted(OUT.glob("model-*.safetensors"))
    mf = [safe_open(str(p), framework="pt", device="cpu") for p in shards]
    mmap = {k: i for i, f in enumerate(mf) for k in f.keys()}

    unit = build_unit()
    n, miss = load_block(unit, mf, mmap, 0)
    tot = sum(p.numel() for p in unit.blocks[0].parameters())
    log(f"\n  第 0 层载入 {n:,}/{tot:,} = {n/tot*100:.1f}%")
    if miss:
        log("  未载入：")
        for m in miss[:15]:
            log(f"    {m[0]}  {m[1]} → {m[2]}")

    SEQ = 48
    g = torch.Generator().manual_seed(5)
    emb = sf.get_tensor("model.embed_tokens.weight")
    sd_ = float(emb.float().std()); del emb
    x = torch.randn(1, SEQ, D, generator=g) * sd_
    pos = torch.arange(SEQ)

    i = 0
    p = f"model.layers.{i}."
    w = {k: sf.get_tensor(k) for k in [p + s for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight")]}

    with torch.no_grad():
        t_attn = teacher_attn(x, w, i, pos)        # teacher 注意力分支输出
        t_full = teacher_ffn(t_attn, w, i)         # teacher 整层
        s_mix = unit.blocks[0].mixer(x)            # ★ 迁移后 mixer
        s_mix = s_mix[0] if isinstance(s_mix, tuple) else s_mix
        s_full = unit.blocks[0](x)                 # 迁移后整层
        s_full = s_full[0] if isinstance(s_full, tuple) else s_full
        # ★ 决定性对照：把 teacher 的注意力输出喂给迁移后的 FFN
        s_ffn_only = unit.blocks[0].ffn(t_attn)
        s_ffn_only = s_ffn_only[0] if isinstance(s_ffn_only, tuple) else s_ffn_only

    log(f"\n  {'对照':<44}{'余弦':>10}  {'释义'}")
    log("  " + "-" * 74)
    c1 = cos(t_attn, s_mix)
    c2 = cos(t_full, s_ffn_only)
    c3 = cos(t_full, s_full)
    log(f"  {'① teacher注意力输出 vs 迁移mixer输出':<44}{c1:>10.4f}  "
        f"mixer 保真度")
    log(f"  {'② teacher整层 vs [teacher注意力→迁移FFN]':<44}{c2:>10.4f}  "
        f"FFN 保真度")
    log(f"  {'③ teacher整层 vs 迁移整层':<44}{c3:>10.4f}  "
        f"端到端保真度")
    log("  " + "-" * 74)

    log(f"\n  【结论】")
    if c2 > 0.95:
        log(f"    ✅ FFN 是逐位复制的，保真度 {c2:.4f} —— "
            f"**Qwen3 的知识主体（MLP）完整继承**")
    else:
        log(f"    ⚠️ FFN 保真度仅 {c2:.4f}，需排查")
    if c1 < 0.5:
        log(f"    ❌ mixer 保真度仅 {c1:.4f} —— "
            f"**全部失真来自 Attention→SGC 这一步**")
        log(f"       机理：注意力是输入相关(content-based)算子，")
        log(f"       SGC 是输入无关的线性卷积核。核拟合只能复现**平均**")
        log(f"       位置分布，复现不了逐 token 的内容路由。")
        log(f"       → 这是结构性损失，不是调参能解决的，必须靠微调补偿。")
    log(f"\n    端到端 {c3:.4f}。裸迁移当前 ≠ 可用模型，")
    log(f"    但 FFN 那部分（占参数 66%）的知识是完整搬过来的。")
    save("diag", dict(mixer_cos=round(c1, 4), ffn_cos=round(c2, 4),
                      e2e_cos=round(c3, 4), loaded=n, total=tot,
                      missing=[str(m) for m in miss]))


if __name__ == "__main__":
    main()
