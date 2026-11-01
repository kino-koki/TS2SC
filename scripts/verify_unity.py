#!/usr/bin/env python3
"""
逐位验证：融合权重里有多少是**真的** MiniCPM5-1B / Qwen3-0.6B 原权重？

这是整个 SimplUnity 唯一的硬证据 —— 不做这个，前面所有数字都只是"我声称"。

方法：对每一对 (融合张量, 源张量) 做 elementwise 全等比较。
  bf16 直存 → 应该 100% 逐位相同（没有精度损失）
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

for p in ("/data/workspace", "/data/workspace/SimplCore_official"):
    sys.path.insert(0, p)
import torch
from safetensors import safe_open

WS = Path("/data/workspace")
UNI = WS / "UNITY"
RES = UNI / "result.json"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(UNI / "verify.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def save(k, v):
    d = json.loads(RES.read_text()) if RES.exists() else {}
    d[k] = v
    RES.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def main():
    log("=" * 68)
    log("[verify] 逐位验证：融合权重 vs 源模型")
    log("=" * 68)
    idx = json.loads((UNI / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    shards = sorted({v for v in wmap.values()})
    uf = {s: safe_open(str(UNI / s), framework="pt", device="cpu") for s in shards}
    sf_m = safe_open(str(WS / "mcpm5" / "model.safetensors"),
                     framework="pt", device="cpu")
    sf_q = safe_open(str(WS / "mig" / "model.safetensors"),
                     framework="pt", device="cpu")
    log(f"  融合 {len(wmap)} 张量 / {len(shards)} 分片")
    log(f"  MiniCPM5 {len(sf_m.keys())}   Qwen3 {len(sf_q.keys())}")

    # 期望映射：(融合键, 源键, 源句柄)
    pairs = []
    pairs += [("tok_emb.weight", "model.embed_tokens.weight", "M"),
              ("head.weight", "lm_head.weight", "M"),
              ("ln_f.weight", "model.norm.weight", "M")]
    for i in range(24):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        pairs += [(b + "mixer.norm.weight", p + "input_layernorm.weight", "M"),
                  (b + "ffn.norm.weight", p + "post_attention_layernorm.weight", "M"),
                  (b + "ffn.gate.weight", p + "mlp.gate_proj.weight", "M"),
                  (b + "ffn.up.weight", p + "mlp.up_proj.weight", "M"),
                  (b + "ffn.down.weight", p + "mlp.down_proj.weight", "M")]
    for i in range(28):
        p = f"model.layers.{i}."
        b = f"blocks.{24 + i}."
        pairs += [(b + "mixer.norm.weight", p + "input_layernorm.weight", "Q"),
                  (b + "ffn.norm.weight", p + "post_attention_layernorm.weight", "Q"),
                  (b + "ffn.gate.weight", p + "mlp.gate_proj.weight", "Q"),
                  (b + "ffn.up.weight", p + "mlp.up_proj.weight", "Q"),
                  (b + "ffn.down.weight", p + "mlp.down_proj.weight", "Q")]

    H = {"M": sf_m, "Q": sf_q}
    ok = miss = bad = 0
    ok_p = miss_p = 0
    t0 = time.time()
    by_src = {"M": [0, 0], "Q": [0, 0]}       # [张量数, 参数量]
    bad_list = []
    for k, src, tag in pairs:
        if k not in wmap:
            miss += 1
            continue
        a = uf[wmap[k]].get_tensor(k)
        try:
            b = H[tag].get_tensor(src)
        except Exception:
            miss += 1
            continue
        if tuple(a.shape) != tuple(b.shape):
            bad += 1
            bad_list.append((k, tuple(a.shape), tuple(b.shape)))
            continue
        same = bool(torch.equal(a, b.to(a.dtype)))
        if same:
            ok += 1; ok_p += a.numel()
            by_src[tag][0] += 1; by_src[tag][1] += a.numel()
        else:
            bad += 1
            bad_list.append((k, "值不同", ""))

    log(f"\n  {'来源':<26}{'张量':>8}{'参数':>18}")
    log("  " + "-" * 54)
    for tag, nm in (("M", "MiniCPM5-1B (B body 24层)"), ("Q", "Qwen3-0.6B (A body 28层)")):
        n, p = by_src[tag]
        log(f"  {nm:<26}{n:>8}{p:>18,}   ✅ 逐位全等")
    log("  " + "-" * 54)
    log(f"  {'合计':<26}{ok:>8}{ok_p:>18,}")
    log(f"\n  缺失 {miss}   不一致 {bad}")
    for x in bad_list[:6]:
        log(f"    ✗ {x}")

    tot = 1436157440
    log(f"\n  ★ 逐位真实权重 {ok_p:,} / {tot:,} = {ok_p/tot*100:.1f}%")
    log(f"    MiniCPM5 贡献 {by_src['M'][1]/1e6:.0f}M    Qwen3 贡献 {by_src['Q'][1]/1e6:.0f}M")
    log(f"    耗时 {time.time()-t0:.0f}s")

    save("verify", dict(tensors=len(pairs), exact=ok, missing=miss,
                        mismatch=bad, params=ok_p,
                        ratio=round(ok_p / tot, 4),
                        from_minicpm5=by_src["M"][1], from_qwen3=by_src["Q"][1]))


if __name__ == "__main__":
    main()
