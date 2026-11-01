#!/usr/bin/env python3
"""
决定性实验：冻结 86.6%（FFN/emb/norm，逐位等价），只训 13.4%（mixer）
用 teacher 整层输出做回归目标 → 看端到端保真度能恢复到多少

这直接回答：裸迁移不可用，微调要多少步才能救回来？
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path
for p in ("/data/workspace/SimplCore_official","/data/workspace",
          "/data/workspace/SimplCore_official/scripts"):
    sys.path.insert(0,p)
import torch, torch.nn.functional as F
torch.set_num_threads(2)
WS=Path("/data/workspace"); OUT=WS/"MIGREAL"; RES=OUT/"result.json"
SRC=WS/"mig"/"model.safetensors"
D,FF,L,H,KV,HD,V,BASES=1024,3072,28,16,8,128,151936,32
SMALL_V=4096
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True)
    with open(OUT/"distlog.txt","a",encoding="utf-8") as f: f.write(s+"\n")
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
def teacher_block(x,w,i,pos):
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
    x2=x.float()+F.linear(o,w[p+"self_attn.o_proj.weight"].float())
    h2=rms(x2,w[p+"post_attention_layernorm.weight"])
    gg=F.linear(h2,w[p+"mlp.gate_proj.weight"].float())
    uu=F.linear(h2,w[p+"mlp.up_proj.weight"].float())
    return x2+F.linear(F.silu(gg)*uu,w[p+"mlp.down_proj.weight"].float())
def build():
    from simplcore.core import get_preset, SimplCoreModel
    torch.manual_seed(1234)
    c=get_preset("tuan-30m")
    c.dim=D;c.num_layers=1;c.num_bases=BASES;c.num_heads=8
    c.vocab_size=SMALL_V;c.seq_len=256;c.ffn_ratio=3.0
    return SimplCoreModel(c)
def load(unit,mf,mmap,li):
    sd=unit.state_dict();n=0
    for k in list(sd.keys()):
        if not k.startswith("blocks.0."): continue
        src=k.replace("blocks.0.",f"blocks.{li}.",1)
        if src in mmap:
            t=mf[mmap[src]].get_tensor(src)
            if tuple(t.shape)==tuple(sd[k].shape):
                with torch.no_grad(): sd[k].copy_(t); n+=t.numel()
    return n
def cos(a,b): return float(F.cosine_similarity(a.reshape(-1).float(),b.reshape(-1).float(),dim=0))
def main():
    steps=int(os.getenv("DM_STEPS","600"))
    log("="*68)
    log("[distill] 冻结 86.6%，只训 mixer —— 保真度恢复曲线")
    log("="*68)
    from safetensors import safe_open
    sf=safe_open(str(SRC),framework="pt",device="cpu")
    shards=sorted(OUT.glob("model-*.safetensors"))
    mf=[safe_open(str(p),framework="pt",device="cpu") for p in shards]
    mmap={k:i for i,f in enumerate(mf) for k in f.keys()}
    LI=int(os.getenv("DM_LAYER","0"))
    SEQ=48
    p=f"model.layers.{LI}."
    w={k:sf.get_tensor(k) for k in [p+s for s in (
        "input_layernorm.weight","post_attention_layernorm.weight",
        "self_attn.q_proj.weight","self_attn.k_proj.weight",
        "self_attn.v_proj.weight","self_attn.o_proj.weight",
        "self_attn.q_norm.weight","self_attn.k_norm.weight",
        "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight")]}
    pos=torch.arange(SEQ)
    emb=sf.get_tensor("model.embed_tokens.weight"); sd_=float(emb.float().std()); del emb
    g=torch.Generator().manual_seed(5)
    # 训练/验证输入：贴近真实激活尺度
    Xtr=[torch.randn(1,SEQ,D,generator=g)*sd_ for _ in range(int(os.getenv("DM_N","24")))]
    Xva=[torch.randn(1,SEQ,D,generator=g)*sd_ for _ in range(6)]
    with torch.no_grad():
        Ytr=[teacher_block(x,w,LI,pos) for x in Xtr]
        Yva=[teacher_block(x,w,LI,pos) for x in Xva]
    del w
    log(f"  layer {LI} | 训练样本 {len(Xtr)} × {SEQ}×{D} | 目标已缓存")
    u=build(); n=load(u,mf,mmap,LI)
    tot=sum(pp.numel() for pp in u.blocks[0].parameters())
    log(f"  权重载入 {n:,}/{tot:,} = {n/tot*100:.1f}%")
    # ★ 冻结 FFN 与 norm，只训 mixer
    for nm,pp_ in u.named_parameters():
        if nm.startswith("blocks.0.ffn") or nm.endswith("norm.weight"):
            pp_.requires_grad_(False)
    if hasattr(u.blocks[0].mixer,"norm"):
        u.blocks[0].mixer.norm.weight.requires_grad_(False)
    tr=[pp_ for pp_ in u.parameters() if pp_.requires_grad]
    ntr=sum(pp_.numel() for pp_ in tr)
    log(f"  可训练 {ntr:,}/{tot:,} = {ntr/tot*100:.1f}%  （冻结 FFN+norm）")
    opt=torch.optim.AdamW(tr,lr=float(os.getenv("DM_LR","1e-3")))
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps)
    with torch.no_grad():
        c0=sum(cos(Yva[i],u.blocks[0](Xva[i])) for i in range(len(Xva)))/len(Xva)
    log(f"\n  起始端到端余弦 {c0:.4f}")
    log(f"\n  {'步':>5} {'train loss':>12} {'val 余弦':>10}")
    log("  "+"-"*32)
    hist=[]; t0=time.time()
    for s in range(1,steps+1):
        u.train(); tot_l=0.0
        for xi,yi in zip(Xtr,Ytr):
            o=u.blocks[0](xi)
            o=o[0] if isinstance(o,tuple) else o
            l=F.mse_loss(o.float(),yi.float())
            opt.zero_grad(set_to_none=True); l.backward()
            torch.nn.utils.clip_grad_norm_(tr,1.0); opt.step(); sc.step()
            tot_l+=float(l)
        if s%100==0 or s==1:
            u.eval()
            with torch.no_grad():
                cv=sum(cos(Yva[i],u.blocks[0](Xva[i])) for i in range(len(Xva)))/len(Xva)
            hist.append(dict(step=s,loss=round(tot_l/len(Xtr),6),cos=round(cv,4)))
            log(f"  {s:>5} {tot_l/len(Xtr):>12.6f} {cv:>10.4f}  [{time.time()-t0:.0f}s]")
    log("  "+"-"*32)
    log(f"  ★ {c0:.4f} → {hist[-1]['cos']:.4f}")
    log(f"    ({steps} 步，仅训 {ntr/1e6:.1f}M 参数)")
    save("distill",dict(layer=LI,start=round(c0,4),end=hist[-1]["cos"],
                        hist=hist,trainable=ntr,total=tot,steps=steps))
if __name__=="__main__": main()
