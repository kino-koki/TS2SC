#!/usr/bin/env python3
"""
SimplUnity v2 · 双模型深度融合（dim-aware）

═════════════════════════════════════════════════════════════════
★ 重要更正：之前我说「Qwen3 与 MiniCPM5 的 FFN 中间维都是 3072」
  这是错的。实测：
    Qwen3-0.6B   intermediate_size = 3072
    MiniCPM5-1B  intermediate_size = 4608
  维度不一致 → 无法逐位融合层，必须换设计。
═════════════════════════════════════════════════════════════════

设计：深度串联（depth-union）+ 双桥接

  两个模型的 hidden 维不同（1024 vs 1536），
  无法在同一条残差流上直接合并任何一层。
  但**可以串联**：让两者各自在属于自己的维度上完整工作，
  中间只插入两个待训练的线性桥。

    token ids (MiniCPM5 词表 V=130560)
        ↓
    emb  ← MiniCPM5.model.embed_tokens        [逐位]
        ↓
    ┌─ MiniCPM5 body: 24 层 @ D=1536 FF=4608 ─┐  [FFN 逐位]
    └─────────────────────────────────────────┘
        ↓  bridge_down: Linear(1536 → 1024)     [可训练]
    ┌─ Qwen3 body: 28 层 @ D=1024 FF=3072 ────┐  [FFN 逐位]
    └─────────────────────────────────────────┘
        ↓  bridge_up: Linear(1024 → 1536)       [可训练]
    norm ← MiniCPM5.model.norm                 [逐位]
        ↓
    head ← MiniCPM5.lm_head                    [逐位]

  词表用 MiniCPM5 的（它是 1B 主干，且 emb/head 可逐位保留）。
  **两个模型的 body 权重都是原样的**，只有 2 个桥 + 每层 mixer 需要训练。

用法：
  python3 simplunity2.py --plan     纯计算，秒出
  python3 simplunity2.py --fuse     真正执行，流式写盘
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

for p in ("/data/workspace/SimplCore_official", "/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0, p)
import torch
from safetensors import safe_open
from safetensors.torch import save_file

WS = Path("/data/workspace")
OUT = WS / "UNITY"; OUT.mkdir(exist_ok=True)
RES = OUT / "result.json"
QW = WS / "mig"                      # Qwen3-0.6B
MC = WS / "mcpm5"                    # MiniCPM5-1B

# Qwen3-0.6B
Q1 = dict(D=1024, FF=3072, L=28, H=16, KV=8, HD=128, V=151936,
          eps=1e-6, rope=1_000_000.0)
# MiniCPM5-1B
Q2 = dict(D=1536, FF=4608, L=24, H=16, KV=2, HD=128, V=130560,
          eps=1e-6, rope=5_000_000.0)
BASES = 32


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "unity.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def save(k, v):
    d = json.loads(RES.read_text()) if RES.exists() else {}
    d[k] = v
    RES.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def n_params(cfg):
    """单层 SimplCore 参数量"""
    D, FF, M = cfg["D"], cfg["FF"], BASES
    mixer = 3 * D * D            # in_proj / gate_proj / out_proj
    mixer += 2 * M * D           # raw_rho / raw_theta (per-base, per-channel)
    mixer += M * D               # coef
    mixer += D * D               # feed (approx)
    ffn = 3 * D * FF
    norm = 2 * D
    return mixer, ffn, norm


def plan():
    log("=" * 68)
    log("[SimplUnity v2] 融合规划（depth-union + 双桥）")
    log("=" * 68)
    log(f"\n  A  Qwen3-0.6B    D={Q1['D']}  FF={Q1['FF']}  L={Q1['L']}  "
        f"H={Q1['H']}/{Q1['KV']}  V={Q1['V']}")
    log(f"  B  MiniCPM5-1B   D={Q2['D']}  FF={Q2['FF']}  L={Q2['L']}  "
        f"H={Q2['H']}/{Q2['KV']}  V={Q2['V']}")

    log("\n  ⚠️ 维度不一致，不能做「逐层合并」，只能串联：")
    log(f"     FFN 中间维 {Q1['FF']} vs {Q2['FF']} → 差 {Q2['FF']-Q1['FF']}")
    log(f"     hidden   {Q1['D']} vs {Q2['D']} → 差 {Q2['D']-Q1['D']}")
    log(f"     词表      {Q1['V']} vs {Q2['V']} → 不同 tokenizer，不可语义合并")

    m1, f1, n1 = n_params(Q1)
    m2, f2, n2 = n_params(Q2)
    log(f"\n  单层参数量")
    log(f"     A: mixer {m1/1e6:.2f}M  ffn {f1/1e6:.2f}M  norm {n1/1e3:.1f}K"
        f"  → {(m1+f1+n1)/1e6:.2f}M")
    log(f"     B: mixer {m2/1e6:.2f}M  ffn {f2/1e6:.2f}M  norm {n2/1e3:.1f}K"
        f"  → {(m2+f2+n2)/1e6:.2f}M")

    bodyA = Q1["L"] * (m1 + f1 + n1)
    bodyB = Q2["L"] * (m2 + f2 + n2)
    emb = Q2["V"] * Q2["D"]
    head = Q2["V"] * Q2["D"]
    brg = Q1["D"] * Q2["D"] * 2             # 两个桥
    total = bodyA + bodyB + emb + head + brg + Q2["D"]

    # 逐位保留的部分
    keep = Q1["L"] * f1 + Q2["L"] * f2      # 两个 body 的 FFN 全逐位
    keep += Q1["L"] * n1 + Q2["L"] * n2     # norm
    keep += emb + head + Q2["D"]            # emb / head / final norm

    log(f"\n  {'部件':<28}{'参数':>16}{'占比':>9}")
    log("  " + "-" * 54)
    for nm, v in [("B body: MiniCPM5 24 层", bodyB),
                  ("A body: Qwen3 28 层", bodyA),
                  ("embedding (MiniCPM5)", emb),
                  ("lm_head (MiniCPM5)", head),
                  ("bridge 1536→1024→1536", brg),
                  ("final norm", Q2["D"])]:
        log(f"  {nm:<28}{v:>16,}{v/total*100:>8.1f}%")
    log("  " + "-" * 54)
    log(f"  {'总计':<28}{total:>16,}{100.0:>8.1f}%")
    log(f"\n  ★ 逐位保留（真实预训练权重）{keep:,} = {keep/total*100:.1f}%")
    log(f"    待训练 {total-keep:,} = {(total-keep)/total*100:.1f}%"
        f"  （mixer {Q1['L']*m1+Q2['L']*m2:,} + 桥 {brg:,}）")
    log(f"\n  规模 ≈ {total/1e9:.3f} B   (bf16 {total*2/2**30:.2f} GiB)")
    save("plan", dict(total=int(total), keep=int(keep),
                      keep_ratio=round(keep / total, 4),
                      bodyA=int(bodyA), bodyB=int(bodyB),
                      emb=int(emb), head=int(head), bridge=int(brg),
                      vocab=Q2["V"], layers=Q1["L"] + Q2["L"]))
    return total


# ══════════════ attention → SGC mixer 映射 ══════════════
def derive_mixer(sf, li, cfg, prefix="model.layers"):
    """从 Transformer 层派生 SimplCore mixer 权重"""
    D, M = cfg["D"], BASES
    p = f"{prefix}.{li}."
    out = {}
    # in_proj ← v_proj（可能被 GQA 压缩，需扩维）
    v = sf.get_tensor(p + "self_attn.v_proj.weight")
    out["in_proj"] = _fit(v, D, D, tag="v_proj→in_proj")
    # gate_proj ← k_proj
    k = sf.get_tensor(p + "self_attn.k_proj.weight")
    out["gate_proj"] = _fit(k, D, D, tag="k_proj→gate_proj")
    # out_proj ← o_proj（可能是 H*HD 宽，切片）
    o = sf.get_tensor(p + "self_attn.o_proj.weight")
    out["out_proj"] = _fit(o, D, D, tag="o_proj→out_proj")
    return out


def _fit(t, rows, cols, tag=""):
    """确定性适配到 (rows, cols)：先切片后扩维"""
    t = t.float()
    r, c = t.shape
    # 行
    if r >= rows:
        t = t[:rows]
    else:
        rep = math.ceil(rows / r)
        parts = [t * (0.5 + 0.35 * math.sin(1.7 * i)) for i in range(rep)]
        t = torch.cat(parts, 0)[:rows]
    # 列
    c2 = t.shape[1]
    if c2 >= cols:
        t = t[:, :cols]
    else:
        rep = math.ceil(cols / c2)
        parts = [t * (0.5 + 0.35 * math.sin(1.7 * i)) for i in range(rep)]
        t = torch.cat(parts, 1)[:, :cols]
    return t.contiguous()


def fuse():
    total = plan()
    log("\n" + "=" * 68)
    log("[fuse] 开始融合（流式写盘，避免内存溢出）")
    log("=" * 68)
    t0 = time.time()
    sf_q = safe_open(str(QW / "model.safetensors"), framework="pt", device="cpu")
    sf_m = safe_open(str(MC / "model.safetensors"), framework="pt", device="cpu")
    kq = set(sf_q.keys()); km = set(sf_m.keys())
    log(f"  Qwen3 张量 {len(kq)}   MiniCPM5 张量 {len(km)}")

    # ★ 流式写盘：1.5B 参数 fp32 = 6GB，内存只有 2.4GB
    #   逐层构造 → 攒够一片就落盘，绝不全部驻留内存
    buf = {}
    buf_bytes = 0
    shards_meta = []
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    LIMIT = 150 * 1024 * 1024
    sd = buf
    n_verbatim = 0

    def flush():
        nonlocal buf, buf_bytes
        if not buf:
            return
        nm = f"model-{len(shards_meta):05d}.safetensors"
        save_file(buf, str(OUT / nm), metadata={"format": "pt"})
        sz = (OUT / nm).stat().st_size
        index["metadata"]["total_size"] += sz
        for k in buf:
            index["weight_map"][k] = nm
        shards_meta.append((nm, len(buf), sz))
        log(f"    ✓ 落盘 {nm}  {sz/2**20:.0f} MiB  ({len(buf)} 张量)")
        buf = {}
        buf_bytes = 0

    n_have = [0]

    def put(k, t):
        """★ 立即转 bf16 存盘：fp32 中转会让 1.5B 参数占 6GB 内存（必 OOM）"""
        nonlocal buf_bytes
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        t = t.detach().to(torch.bfloat16).contiguous()
        buf[k] = t
        buf_bytes += t.numel() * 2
        n_have[0] += t.numel()
        if buf_bytes > LIMIT:
            flush()

    # ① embedding / head / final norm（MiniCPM5）
    for src, dst in [("model.embed_tokens.weight", "tok_emb.weight"),
                     ("lm_head.weight", "head.weight"),
                     ("model.norm.weight", "ln_f.weight")]:
        t = sf_m.get_tensor(src)
        put(dst, t)
        n_verbatim += t.numel()
        log(f"    ✓ {src} → {dst}  {tuple(t.shape)}")

    # ② MiniCPM5 body: layers 0..23  →  SimplCore blocks 0..23
    for i in range(Q2["L"]):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        # norm
        put(b + "mixer.norm.weight",
            sf_m.get_tensor(p + "input_layernorm.weight"))
        put(b + "ffn.norm.weight",
            sf_m.get_tensor(p + "post_attention_layernorm.weight"))
        n_verbatim += 2 * Q2["D"]
        # FFN（逐位）
        for s_, d_ in [("mlp.gate_proj.weight", "ffn.gate.weight"),
                       ("mlp.up_proj.weight", "ffn.up.weight"),
                       ("mlp.down_proj.weight", "ffn.down.weight")]:
            t = sf_m.get_tensor(p + s_)
            put(b + d_, t)
            n_verbatim += t.numel()
        # mixer
        for k_, v_ in derive_mixer(sf_m, i, Q2).items():
            put(b + f"mixer.{k_}.weight", v_)
        if i % 8 == 0:
            log(f"    B 层 {i}/{Q2['L']}  [{time.time()-t0:.0f}s]")

    # ③ bridge_down
    brg_d = torch.zeros(Q1["D"], Q2["D"])
    torch.nn.init.xavier_uniform_(brg_d)
    put("bridge_down.weight", brg_d)
    put("bridge_up.weight", torch.nn.init.xavier_uniform_(
        torch.zeros(Q2["D"], Q1["D"])))
    # ⚠️ 不要在这里引用 sd[...]：put 可能已触发 flush，buf 已清空
    log(f"    ✓ bridge {tuple(brg_d.shape)} / ({Q2['D']},{Q1['D']})  xavier")

    # ④ Qwen3 body: layers 0..27 → SimplCore blocks 24..51
    for i in range(Q1["L"]):
        p = f"model.layers.{i}."
        b = f"blocks.{Q2['L'] + i}."
        put(b + "mixer.norm.weight",
            sf_q.get_tensor(p + "input_layernorm.weight"))
        put(b + "ffn.norm.weight",
            sf_q.get_tensor(p + "post_attention_layernorm.weight"))
        n_verbatim += 2 * Q1["D"]
        for s_, d_ in [("mlp.gate_proj.weight", "ffn.gate.weight"),
                       ("mlp.up_proj.weight", "ffn.up.weight"),
                       ("mlp.down_proj.weight", "ffn.down.weight")]:
            t = sf_q.get_tensor(p + s_)
            put(b + d_, t)
            n_verbatim += t.numel()
        for k_, v_ in derive_mixer(sf_q, i, Q1).items():
            put(b + f"mixer.{k_}.weight", v_)
        if i % 8 == 0:
            log(f"    A 层 {i}/{Q1['L']}  [{time.time()-t0:.0f}s]")

    have = n_have[0]
    log(f"\n  已构造 {have:,} 参数（逐位 {n_verbatim:,}）")
    log(f"  张量数 {len(index['weight_map'])}")
    flush()
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index))

    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    log(f"\n✅ 融合完成  {have/1e9:.3f} B   逐位 {n_verbatim/1e9:.3f} B "
        f"({n_verbatim/have*100:.1f}%)")
    log(f"   耗时 {time.time()-t0:.0f}s")
    save("fuse", dict(total=int(have), verbatim=int(n_verbatim),
                      ratio=round(n_verbatim / have, 4),
                      shards=len(shards_meta),
                      tensors=len(index["weight_map"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--fuse", action="store_true")
    a = ap.parse_args()
    if a.fuse:
        fuse()
    else:
        plan()
