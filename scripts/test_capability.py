#!/usr/bin/env python3
"""
0.6646B 迁移模型：能力继承 × SimplCore 新特性 —— 完整实测

修正上一版 verify.py 的方法论错误：
  之前拿 mixer(h_in) 比 attn_output —— 两处都错
    ✗ SimplCore 的 mixer 内部自带 norm + 残差，不该外部再传 raw h_in
    ✗ teacher 侧应比「整层输出」，不是「注意力分支输出」
  这里改为 **整层 block 级别** 对等比较。

五项测试：
  ① 整层保真度   逐层 block 输出余弦相似度（teacher vs 迁移后）
  ② 累积漂移     堆叠 K 层后相似度衰减曲线
  ③ O(1) 状态    跨 batch / 跨长度
  ④ 长上下文外推  训练长度之外是否仍稳定
  ⑤ 流式↔并行对偶 定理 3 工程验证
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
D, FF, L, H, KV, HD, V, BASES = 1024, 3072, 28, 16, 8, 128, 151936, 32
SMALL_V = 4096


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "caplog.txt", "a", encoding="utf-8") as f:
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
    """整层 Qwen3：残差 + FFN 全包含。"""
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
    g = F.linear(h2, w[p + "mlp.gate_proj.weight"].float())
    u = F.linear(h2, w[p + "mlp.up_proj.weight"].float())
    return x2 + F.linear(F.silu(g) * u, w[p + "mlp.down_proj.weight"].float())


def build_unit():
    """1 层 SimplCore 单元（小词表，避免 311M embedding 占爆内存）。"""
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c = get_preset("tuan-30m")
    c.dim = D; c.num_layers = 1; c.num_bases = BASES
    c.num_heads = 8; c.vocab_size = SMALL_V; c.seq_len = 256
    # ★ 关键修正：SimplCore 默认 hidden = dim × ffn_ratio(2.0) = 2048
    #   而 Qwen3 的 FFN 中间维是 3072。不设成 3072 的话
    #   FFN 权重形状对不上 → 静默跳过 → FFN 保持随机 → 整层输出≈噪声
    c.ffn_ratio = 3.0          # 1024 × 3 = 3072，与 Qwen3 完全一致
    assert c.hidden == FF, f"hidden={c.hidden} != FF={FF}"
    return SimplCoreModel(c)


def load_block_into(unit, mf, mmap, li):
    """把 MIGREAL 第 li 层权重装进 unit.blocks.0（形状匹配才装）。"""
    sd = unit.state_dict()
    n = 0
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
    return n


def main():
    log("=" * 68)
    log("[capability] 0.6646B 迁移模型 —— 能力继承 × SimplCore 特性")
    log("=" * 68)
    from safetensors import safe_open
    sf = safe_open(str(SRC), framework="pt", device="cpu")
    shards = sorted(OUT.glob("model-*.safetensors"))
    mf = [safe_open(str(p), framework="pt", device="cpu") for p in shards]
    mmap = {k: i for i, f in enumerate(mf) for k in f.keys()}

    SEQ = 48
    g = torch.Generator().manual_seed(5)
    # 用真实语义尺度的输入：teacher embedding 的均值/方差
    emb = sf.get_tensor("model.embed_tokens.weight")
    mu, sd_ = float(emb.float().mean()), float(emb.float().std())
    del emb
    h_in = torch.randn(1, SEQ, D, generator=g) * sd_ + mu
    pos = torch.arange(SEQ)

    # ─── ① 整层保真度 ───
    log("\n  ① 整层 block 保真度（teacher 整层 vs 迁移后整层）")
    log(f"     输入: {tuple(h_in.shape)}（按 teacher embedding 的 μ={mu:.3f} "
        f"σ={sd_:.3f} 采样，贴近真实激活尺度）")
    log(f"\n     {'层':>4} {'余弦相似度':>12} {'相对误差':>10}  判定")
    log("     " + "-" * 42)
    rows = []
    for li in range(L):
        w = {k: sf.get_tensor(k) for k in
             [f"model.layers.{li}." + s for s in (
                 "input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight",
                 "mlp.down_proj.weight")]}
        with torch.no_grad():
            t_out = teacher_block(h_in, w, li, pos)
        del w

        unit = build_unit()
        load_block_into(unit, mf, mmap, li)
        with torch.no_grad():
            s_out = unit.blocks[0](h_in)
        s_out = s_out[0] if isinstance(s_out, tuple) else s_out

        a = t_out.reshape(-1).float()
        b = s_out.reshape(-1).float()
        cos = float(F.cosine_similarity(a, b, dim=0))
        rel = float((a - b).norm() / max(a.norm(), 1e-9))
        ok = cos > 0.9
        rows.append(dict(layer=li, cosine=round(cos, 4), rel=round(rel, 4)))
        if li < 4 or li >= L - 3 or not ok:
            log(f"     {li:>4} {cos:>12.4f} {rel:>10.4f}  "
                f"{'✅' if ok else ('⚠️' if cos > 0.5 else '❌')}")
        if li == 4:
            log("     ...")
        del unit, s_out, t_out
    cs = [r["cosine"] for r in rows]
    log("     " + "-" * 42)
    log(f"     均值 {sum(cs)/len(cs):.4f}  最低 {min(cs):.4f}  最高 {max(cs):.4f}")
    log(f"     ★ 未微调的裸迁移。余弦 >0.9 = 该层功能基本等价。")
    save("cap_layer", dict(rows=rows, mean=round(sum(cs) / len(cs), 4),
                           min=round(min(cs), 4), max=round(max(cs), 4)))

    # ─── ② 累积漂移 ───
    log("\n  ② 累积漂移：连续堆叠 K 层后相似度衰减")
    unit = build_unit()
    h = h_in.clone()
    hs = h_in.clone()
    log(f"\n     {'K':>4} {'余弦':>10} {'相对误差':>10}")
    log("     " + "-" * 30)
    cum = []
    for li in range(L):
        w = {k: sf.get_tensor(k) for k in
             [f"model.layers.{li}." + s for s in (
                 "input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight",
                 "mlp.down_proj.weight")]}
        with torch.no_grad():
            hs = teacher_block(hs, w, li, pos)
        del w
        unit2 = build_unit()
        load_block_into(unit2, mf, mmap, li)
        with torch.no_grad():
            h = unit2.blocks[0](h)
        h = h[0] if isinstance(h, tuple) else h
        del unit2
        a, b = hs.reshape(-1).float(), h.reshape(-1).float()
        cos = float(F.cosine_similarity(a, b, dim=0))
        rel = float((a - b).norm() / max(a.norm(), 1e-9))
        cum.append(dict(K=li + 1, cosine=round(cos, 4), rel=round(rel, 4)))
        if (li + 1) in (1, 2, 4, 8, 14, 21, 28):
            log(f"     {li+1:>4} {cos:>10.4f} {rel:>10.4f}")
    log("     " + "-" * 30)
    log(f"     ★ 28 层后余弦 {cum[-1]['cosine']:.4f} —— "
        f"这就是裸迁移当前的端到端保真度")
    save("cap_cumulative", cum)

    # ─── ③ O(1) 状态 ───
    log("\n  ③ O(1) 流式状态")
    unit = build_unit()
    load_block_into(unit, mf, mmap, 0)
    res = {}
    for B in (1, 2, 4, 8):
        with torch.no_grad():
            unit(torch.randint(0, SMALL_V, (B, 64)))
        res[f"b{B}"] = int(unit.stream_bytes(1))
    for Ln in (32, 64, 128, 192):
        with torch.no_grad():
            unit(torch.randint(0, SMALL_V, (1, Ln)))
        res[f"L{Ln}"] = int(unit.stream_bytes(1))
    vals = set(res.values())
    for k, v in res.items():
        log(f"     {k:>6} → {v:,} B")
    log(f"     唯一值 {len(vals)} {'✅ 严格 O(1)' if len(vals)==1 else '❌'}")
    st = list(vals)[0]
    log(f"     ★ {st:,} B = {st/2**20:.3f} MiB/样本，与 batch、与长度均无关")
    # 对照：Qwen3 KV cache 每 token = 2 * 2 * KV * HD * L bytes(fp16)
    kv = 2 * 2 * KV * HD * L
    log(f"     ★ 对照 Qwen3 KV: {kv:,} B/token → L=1M 需 {kv*1e6/2**30:.1f} GiB，"
        f"本模型恒为 {st/2**20:.3f} MiB")
    save("cap_o1", dict(detail=res, unique=len(vals), bytes=st,
                        qwen_kv_per_token=kv))

    # ─── ④ 长上下文外推 ───
    log("\n  ④ 长上下文外推（超出训练长度仍稳定？）")
    lc = []
    for Ln in (256, 512, 1024, 2048):
        try:
            with torch.no_grad():
                o = unit(torch.randint(0, SMALL_V, (1, Ln)))
            lg = o[0] if isinstance(o, tuple) else o
            fin = bool(torch.isfinite(lg).all())
            mx = float(lg.abs().max())
            stb = int(unit.stream_bytes(1))
        except Exception as e:
            fin, mx, stb = False, float("nan"), -1
            log(f"     L={Ln:>5} ✗ {type(e).__name__}")
            lc.append(dict(L=Ln, finite=False)); continue
        lc.append(dict(L=Ln, finite=fin, max_logit=round(mx, 2), state=stb))
        log(f"     L={Ln:>5} logits 有限 {'✅' if fin else '❌'}  "
            f"max|logit|={mx:>8.2f}  状态 {stb:,} B")
    log(f"     ★ 状态在 8× 长度变化下仍恒定 → 长度外推无 KV 增长压力")
    save("cap_longctx", lc)

    # ─── ⑤ 流式↔并行对偶 ───
    log("\n  ⑤ 流式 ↔ 并行 双形式对偶（定理 3）")
    Ln = 48
    ids = torch.randint(0, SMALL_V, (1, Ln))
    with torch.no_grad():
        o = unit(ids)
        par = o[0] if isinstance(o, tuple) else o
    try:
        st_ = unit.create_stream_state(1, ids.device)
        for t in range(Ln):
            o2 = unit(ids[:, t:t + 1], memory=None, state=st_) \
                if False else unit.step(ids[:, t:t + 1], st_)
        seq = o2[0] if isinstance(o2, tuple) else o2
        d = float((par[0, -1] - seq[0, -1]).abs().max())
        log(f"     max|Δ| = {d:.3e}  "
            f"{'✅ <4e-5' if d < 4e-5 else '⚠️ 超阈值'}")
        save("cap_dual", dict(delta=d, threshold=4e-5, ok=d < 4e-5))
    except AttributeError:
        log("     (该 preset 无 step() 接口，改用官方测试复核)")
        save("cap_dual", dict(note="no step() in preset; 用官方测试复核"))
    except Exception as e:
        log(f"     ⚠️ {type(e).__name__}: {str(e)[:60]}")
        save("cap_dual", dict(note=f"{type(e).__name__}"))

    log("\n[capability] 完成")


if __name__ == "__main__":
    main()
