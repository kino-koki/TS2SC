#!/usr/bin/env python3
"""
全层真实激活蒸馏 —— 完整继承能力的关键修正

╔══════════════════════════════════════════════════════════════╗
║ 之前为什么慢且不对：用**随机高斯**当输入                      ║
║   真实激活是低秩、各维相关的；高斯不是                        ║
║   → 学到的 mixer 在真实数据上可能完全不 work                  ║
║                                                              ║
║ 这次修正：一次 teacher 前向，缓存 **28 层真实激活**           ║
║   X[i] = 第 i 层真实输入      Y[i] = 第 i 层真实输出          ║
║   再逐层蒸馏。输入分布 100% 真实。                            ║
╚══════════════════════════════════════════════════════════════╝

输入怎么造才真实：
   用**真实 token 序列**过 teacher 的 embedding，逐层前向。
   没有 Qwen tokenizer → 用 teacher embedding 里随机抽的行
   （每行都是一个真实 token 的 embedding，分布完全真实）

流程：
   stage cache     一次前向，缓存 28 层真实激活（落盘）
   stage train     逐层蒸馏，可续跑
   stage eval      全层串联，端到端余弦
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
ACT = OUT / "act"; ACT.mkdir(exist_ok=True)
CK = OUT / "ck"; CK.mkdir(exist_ok=True)

D, FF, L, H, KV, HD, V, BASES = 1024, 3072, 28, 16, 8, 128, 151936, 32
SMALL_V = 4096
SEQ = int(os.getenv("RA_SEQ", "64"))
NTR = int(os.getenv("RA_N", "8"))
NVA = int(os.getenv("RA_NVA", "4"))
STEPS = int(os.getenv("RA_STEPS", "300"))
LR = float(os.getenv("RA_LR", "3e-3"))
CHUNK = int(os.getenv("RA_CHUNK", "20"))


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "realact.log", "a", encoding="utf-8") as f:
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


# ═══════════ stage cache ═══════════
def stage_cache():
    log("=" * 66)
    log("[cache] 一次 teacher 前向，缓存 28 层真实激活")
    log("=" * 66)
    from safetensors import safe_open
    sf = safe_open(str(SRC), framework="pt", device="cpu")
    emb = sf.get_tensor("model.embed_tokens.weight")
    log(f"  embedding {tuple(emb.shape)}")
    g = torch.Generator().manual_seed(7)
    # ★ 真实 token embedding：从词表里随机抽行，每行是真实 token 的向量
    def make_ids(n):
        return torch.randint(0, emb.shape[0], (1, SEQ), generator=g)
    pos = torch.arange(SEQ)

    def run(n, tag):
        tr_X = [[] for _ in range(L)]
        tr_Y = [[] for _ in range(L)]
        for s in range(n):
            ids = make_ids(1)
            x = emb[ids].float()                      # 真实 embedding
            for i in range(L):
                w = {k: sf.get_tensor(k) for k in
                     [f"model.layers.{i}." + su for su in (
                         "input_layernorm.weight",
                         "post_attention_layernorm.weight",
                         "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                         "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                         "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                         "mlp.gate_proj.weight", "mlp.up_proj.weight",
                         "mlp.down_proj.weight")]}
                tr_X[i].append(x.clone())
                with torch.no_grad():
                    y = teacher_block(x, w, i, pos)
                tr_Y[i].append(y.clone())
                x = y
                del w
            if (s + 1) % 4 == 0:
                log(f"    {tag} 样本 {s+1}/{n} [{time.time()-t0:.0f}s]")
        for i in range(L):
            torch.save({"X": torch.stack(tr_X[i]),
                        "Y": torch.stack(tr_Y[i])}, ACT / f"{tag}_l{i}.pt")
        log(f"  ✅ {tag} {n} 样本 × {L} 层已缓存")

    t0 = time.time()
    run(NTR, "tr")
    run(NVA, "va")
    # 记录激活尺度
    x0 = torch.load(ACT / "tr_l0.pt")["X"]
    log(f"\n  真实激活尺度: μ={float(x0.mean()):.4f} σ={float(x0.std()):.4f} "
        f"max={float(x0.abs().max()):.3f}")
    xl = torch.load(ACT / f"tr_l{L-1}.pt")["X"]
    log(f"  最后层输入  : μ={float(xl.mean()):.4f} σ={float(xl.std()):.4f} "
        f"max={float(xl.abs().max()):.3f}")
    save("cache", dict(seq=SEQ, ntr=NTR, nva=NVA,
                       mu0=round(float(x0.mean()), 4),
                       sd0=round(float(x0.std()), 4),
                       sdL=round(float(xl.std()), 4)))


# ═══════════ stage train ═══════════
def build_unit():
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


def stage_train():
    li = int(os.getenv("RA_LAYER", "0"))
    ck = CK / f"mixer_l{li}.pt"
    log("=" * 66)
    log(f"[train] layer {li}  真实激活蒸馏  目标 {STEPS} 步")
    log("=" * 66)
    tr = torch.load(ACT / f"tr_l{li}.pt")
    va = torch.load(ACT / f"va_l{li}.pt")
    Xtr, Ytr = tr["X"], tr["Y"]
    Xva, Yva = va["X"], va["Y"]
    # ★ 关键：残差流随层数急剧放大（layer0 σ≈0.03 → layer27 σ≈33）
    #   不归一化的话深层 MSE 量级巨大，梯度爆掉
    YS = float(Ytr.std())
    log(f"  训练 {tuple(Xtr.shape)}  验证 {tuple(Xva.shape)}")
    log(f"  layer{li} 目标尺度 σ={YS:.3f} "
        f"μ={float(Ytr.mean()):.3f} max={float(Ytr.abs().max()):.1f}")
    log(f"  → 用 σ 归一化 loss，避免深层数值爆炸")

    from safetensors import safe_open
    shards = sorted(OUT.glob("model-*.safetensors"))
    mf = [safe_open(str(p), framework="pt", device="cpu") for p in shards]
    mmap = {k: i for i, f in enumerate(mf) for k in f.keys()}

    u = build_unit()
    load_block(u, mf, mmap, li)
    for nm, pp in u.named_parameters():
        if nm.startswith("blocks.0.ffn") or nm.endswith("norm.weight"):
            pp.requires_grad_(False)
    if hasattr(u.blocks[0].mixer, "norm"):
        u.blocks[0].mixer.norm.weight.requires_grad_(False)
    params = [pp for pp in u.parameters() if pp.requires_grad]
    ntr = sum(pp.numel() for pp in params)
    opt = torch.optim.AdamW(params, lr=LR)

    step0, best = 0, -1.0
    if ck.exists():
        try:
            st = torch.load(ck, map_location="cpu")
            msd = u.blocks[0].mixer.state_dict()
            for k, v in st["mixer"].items():
                if k in msd and tuple(msd[k].shape) == tuple(v.shape):
                    msd[k].copy_(v)
            u.blocks[0].mixer.load_state_dict(msd)
            opt.load_state_dict(st["opt"])
            step0, best = st["step"], st.get("best", -1.0)
            log(f"  [resume] {step0} 步，最佳 {best:.4f}")
        except Exception as e:
            log(f"  [resume] {type(e).__name__} 从头")

    if step0 == 0:
        u.eval()
        with torch.no_grad():
            c0 = sum(cos(Yva[i], u.blocks[0](Xva[i]))
                     for i in range(Xva.shape[0])) / Xva.shape[0]
        log(f"  起始余弦 {c0:.4f}   可训练 {ntr/1e6:.2f}M")
        save(f"ra_l{li}_start", round(c0, 4))
    else:
        c0 = best

    n_chunk = min(CHUNK, STEPS - step0)
    if n_chunk <= 0:
        log(f"  layer {li} 已完成 {step0}/{STEPS}")
        return
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=STEPS, eta_min=LR * 0.05)
    for _ in range(step0):
        sc.step()
    t0 = time.time(); u.train()
    for s in range(step0 + 1, step0 + n_chunk + 1):
        tot = 0.0
        for i in range(Xtr.shape[0]):
            o = u.blocks[0](Xtr[i])
            o = o[0] if isinstance(o, tuple) else o
            l = F.mse_loss(o.float() / YS, Ytr[i].float() / YS)
            opt.zero_grad(set_to_none=True); l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sc.step(); tot += float(l)
        if s % 5 == 0 or s == step0 + n_chunk:
            u.eval()
            with torch.no_grad():
                cv = sum(cos(Yva[i], u.blocks[0](Xva[i]))
                         for i in range(Xva.shape[0])) / Xva.shape[0]
            best = max(best, cv)
            log(f"  L{li} step {s:>4}/{STEPS}  loss {tot/Xtr.shape[0]:.6f} "
                f"cos {cv:.4f} 最佳 {best:.4f} [{time.time()-t0:.0f}s]")
            u.train()
        if s % 5 == 0 or s == step0 + n_chunk:
            torch.save({"mixer": u.blocks[0].mixer.state_dict(),
                        "opt": opt.state_dict(), "step": s, "best": best}, ck)
    save(f"ra_l{li}", dict(step=step0 + n_chunk, best=round(best, 4)))
    log(f"  layer {li} 本轮止于 {step0+n_chunk}  最佳 {best:.4f}")


# ═══════════ stage eval：全层串联 ═══════════
def stage_eval():
    log("=" * 66)
    log("[eval] 全层串联：迁移模型 vs teacher 端到端")
    log("=" * 66)
    from safetensors import safe_open
    shards = sorted(OUT.glob("model-*.safetensors"))
    mf = [safe_open(str(p), framework="pt", device="cpu") for p in shards]
    mmap = {k: i for i, f in enumerate(mf) for k in f.keys()}

    va = torch.load(ACT / "va_l0.pt")
    X0 = va["X"]
    Ytrue = torch.load(ACT / f"va_l{L-1}.pt")["Y"]
    log(f"  验证样本 {tuple(X0.shape)}")

    # teacher 侧已有 Ytrue
    # student：28 个单层单元串联
    xs = X0.clone()
    t0 = time.time()
    per = []
    for i in range(L):
        u = build_unit()
        load_block(u, mf, mmap, i)
        ck = CK / f"mixer_l{i}.pt"
        if ck.exists():
            try:
                st = torch.load(ck, map_location="cpu")
                msd = u.blocks[0].mixer.state_dict()
                for k, v in st["mixer"].items():
                    if k in msd and tuple(msd[k].shape) == tuple(v.shape):
                        msd[k].copy_(v)
                u.blocks[0].mixer.load_state_dict(msd)
            except Exception:
                pass
        Yt = torch.load(ACT / f"va_l{i}.pt")["Y"]
        with torch.no_grad():
            xs = u.blocks[0](xs)
            xs = xs[0] if isinstance(xs, tuple) else xs
        c = sum(cos(Yt[j], xs[j]) for j in range(xs.shape[0])) / xs.shape[0]
        per.append(round(c, 4))
        if i % 7 == 0 or i == L - 1:
            log(f"    layer {i:>3}/{L}  逐层余弦 {c:.4f}  [{time.time()-t0:.0f}s]")
        del u
    e2e = sum(cos(Ytrue[j], xs[j]) for j in range(xs.shape[0])) / xs.shape[0]
    log("  " + "-" * 40)
    log(f"  ★ 端到端余弦 {e2e:.4f}")
    log(f"    逐层均值 {sum(per)/len(per):.4f}  最低 {min(per):.4f}")
    save("ra_eval", dict(e2e=round(e2e, 4), per_layer=per,
                         mean=round(sum(per) / len(per), 4),
                         min=round(min(per), 4)))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"cache": stage_cache, "train": stage_train, "eval": stage_eval}[a]()
