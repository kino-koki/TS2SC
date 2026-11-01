#!/usr/bin/env python3
"""
修 coef 形状 bug + 测 content_gate 能否补偿失真

发现：migrate 时存了 coef (32,)，但模型实际需要 (32, D)
      → coef 从未载入 → mixer 的基组合系数是随机的
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
for p in ("/data/workspace/SimplCore_official", "/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0, p)
import torch, torch.nn.functional as F
torch.set_num_threads(2)
WS=Path("/data/workspace"); OUT=WS/"MIGREAL"; RES=OUT/"result.json"
SRC=WS/"mig"/"model.safetensors"
D,FF,L,H,KV,HD,V,BASES=1024,3072,28,16,8,128,151936,32
SMALL_V=4096
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True)
    with open(OUT/"fixlog.txt","a",encoding="utf-8") as f: f.write(s+"\n")
def save(k,v):
    d=json.loads(RES.read_text()) if RES.exists() else {}
    d[k]=v; RES.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def rms(x,w,eps=1e-6):
    v=x.float().pow(2).mean(-1,keepdim=True)
    return x.float()*torch.rsqrt(v+eps)*w.float()
def rope(t,pos,theta=1_000_000.0):
    d=t.shape[-1]
    inv=1.0/(theta**(torch.arange(0,d,2,dtype=torch.float32)/d))
    ang=torch.outer(pos.float(),inv)
    cos,sin=torch.cos(ang)[None,None],torch.sin(ang)[None,None]
    x1,x2=t.float()[...,0::2],t.float()[...,1::2]
    return torch.stack([x1*cos-x2*sin,x1*sin+x2*cos],-1).flatten(-2).to(t.dtype)
def teacher_attn(x,w,i,pos):
    p=f"model.layers.{i}.";B,Ln,_=x.shape
    h=rms(x,w[p+"input_layernorm.weight"])
    q=F.linear(h,w[p+"self_attn.q_proj.weight"].float())
    k=F.linear(h,w[p+"self_attn.k_proj.weight"].float())
    v=F.linear(h,w[p+"self_attn.v_proj.weight"].float())
    q=q.view(B,Ln,H,HD).transpose(1,2);k=k.view(B,Ln,KV,HD).transpose(1,2)
    v=v.view(B,Ln,KV,HD).transpose(1,2)
    q=rms(q,w[p+"self_attn.q_norm.weight"]);k=rms(k,w[p+"self_attn.k_norm.weight"])
    q,k=rope(q,pos),rope(k,pos)
    rep=H//KV
    k=k.unsqueeze(2).expand(B,KV,rep,Ln,HD).reshape(B,H,Ln,HD)
    v=v.unsqueeze(2).expand(B,KV,rep,Ln,HD).reshape(B,H,Ln,HD)
    att=(q@k.transpose(-2,-1))/math.sqrt(HD)
    m=torch.triu(torch.ones(Ln,Ln,dtype=torch.bool,device=x.device),1)
    a=torch.softmax(att.masked_fill(m,float("-inf")),-1)
    o=(a@v).transpose(1,2).reshape(B,Ln,H*HD)
    return x.float()+F.linear(o,w[p+"self_attn.o_proj.weight"].float())
def build(content_gate=False,gate_ab=False):
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c=get_preset("tuan-30m")
    c.dim=D;c.num_layers=1;c.num_bases=BASES;c.num_heads=8
    c.vocab_size=SMALL_V;c.seq_len=256;c.ffn_ratio=3.0
    c.content_gate=content_gate; c.gate_ab=gate_ab
    return SimplCoreModel(c)
def load(unit,mf,mmap,li,fix_coef):
    sd=unit.state_dict();n=0
    for k in list(sd.keys()):
        if not k.startswith("blocks.0."): continue
        src=k.replace("blocks.0.",f"blocks.{li}.",1)
        if src in mmap:
            t=mf[mmap[src]].get_tensor(src)
            want=tuple(sd[k].shape)
            if tuple(t.shape)==want:
                with torch.no_grad(): sd[k].copy_(t); n+=t.numel()
            elif fix_coef and k.endswith("mixer.coef") and t.dim()==1:
                # ★ 修：(32,) → (32, D) 每个基在所有通道共享同一系数
                with torch.no_grad(): sd[k].copy_(t.unsqueeze(1).expand(-1,D).contiguous())
                n+=t.numel()
                log(f"    [fix] mixer.coef {tuple(t.shape)} → {want}（通道共享）")
    return n
def cos(a,b): return float(F.cosine_similarity(a.reshape(-1).float(),b.reshape(-1).float(),dim=0))
def main():
    log("="*66); log("[fix] coef 形状 bug 修复 + content_gate 补偿测试"); log("="*66)
    from safetensors import safe_open
    sf=safe_open(str(SRC),framework="pt",device="cpu")
    shards=sorted(OUT.glob("model-*.safetensors"))
    mf=[safe_open(str(p),framework="pt",device="cpu") for p in shards]
    mmap={k:i for i,f in enumerate(mf) for k in f.keys()}
    SEQ=48; g=torch.Generator().manual_seed(5)
    emb=sf.get_tensor("model.embed_tokens.weight"); sd_=float(emb.float().std()); del emb
    x=torch.randn(1,SEQ,D,generator=g)*sd_; pos=torch.arange(SEQ)
    i=0; p=f"model.layers.{i}."
    w={k:sf.get_tensor(k) for k in [p+s for s in (
        "input_layernorm.weight","post_attention_layernorm.weight",
        "self_attn.q_proj.weight","self_attn.k_proj.weight",
        "self_attn.v_proj.weight","self_attn.o_proj.weight",
        "self_attn.q_norm.weight","self_attn.k_norm.weight",
        "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight")]}
    with torch.no_grad(): t_attn=teacher_attn(x,w,i,pos)
    res={}
    for fix in (False,True):
        for cg in (False,True):
            if fix is False and cg: continue
            tag=f"{'coef已修' if fix else 'coef未修'}{'+content_gate' if cg else ''}"
            try:
                u=build(content_gate=cg)
            except Exception as e:
                log(f"  {tag:<32} ✗ 构造失败 {type(e).__name__}"); continue
            load(u,mf,mmap,i,fix)
            with torch.no_grad():
                m=u.blocks[0].mixer(x)
                m=m[0] if isinstance(m,tuple) else m
            c=cos(t_attn,m)
            res[tag]=round(c,4)
            log(f"  {tag:<32} mixer 余弦 {c:>8.4f}")
            del u
    log("\n  【判读】")
    a=res.get("coef未修",0); b=res.get("coef已修",0); cc=res.get("coef已修+content_gate",0)
    log(f"    coef bug 修复带来的提升: {a:.4f} → {b:.4f}")
    if cc: log(f"    content_gate 额外贡献: {b:.4f} → {cc:.4f}")
    save("fix_coef",res)
if __name__=="__main__": main()
