# -*- coding: utf-8 -*-
"""
v11 —— 最后一轮实质实验: 换掉「可学绝对位置表」这个伪影源
=================================================================================
动机(v9d 实跑发现):
  A2 各外推 L 的 delta 随训练步数【单调收敛到 0】:
     step1500 L96=-0.0167 / L192=-0.0082 / L384=+0.0051
     step6000 L96=+0.0169 / L192=+0.0077 / L384=+0.0007
  => v8 那个 -0.7 bpc 优势在多层 + 长度课程 + 充分训练下彻底消失,
     稳态是 parity, 早期优势是【收敛更快】而非渐近更准。

v10 两个新自由度已实测:
  [N9] 每层独立固定谱字典(多分辨率): 同配置反而更差
       banded-vs-shared gain: A1 +0.0108 / A2min +0.0261 / A3 +0.0023
  [N10] EMA(decay=0.995, 1800步): 0.995^1800≈1e-4, shadow≈在线, 结果逐位相同, 无效
  => 「不动静态核」内、堆层数/换固定字典/权重平均都无法把 A2 拉回 -0.05 稳态。

本轮换一个更根本的问法: A2 的长度外推优势, 会不会只是因为对照臂用了
【可学绝对位置表 Lmax=384】这个有限覆盖组件? 给它换成纯相对位置 ALiBi
(去掉位置表 = 更强对手, 不是削弱), 并把评测推到远超训练课程的长度。

三臂(同层数/同 Block/同 FFN/同 Sidecar/同 seed/同课程/同 dt 增广):
   S   = SGC 6 层, 静态 Toeplitz 谱核(一字不动), 无位置表
   A_L = attention + 可学位置表(384) + ALiBi   (v9/v10 用的旧对照)
   A_R = attention + 纯 ALiBi, 无位置表        (更强、更公平的新对照)

判定协议(与 v9/v10 完全一致, 不放宽、不移动球门):
   A1 同分布 L=48 : S <= A_R + 0.05
   A2 长度外推    : 至少一个 L∈{384,768,1536} 上 S < A_R - 0.05
   A3 不规则 dt   : S <= A_R + 0.05
   3 seeds 每 seed 分别满足; 连续>=3 checkpoint 稳定; 对手 2x 步数仍成立;
   CI/梯度/确定性同 v9。
   COMPLETE_BEYOND = 以上全成立。

停止准则(预先声明): 本轮为最后一轮实质实验。若 A_R 在超长 L 不退化、
A2 仍不可达, 则如实收束为「稳态 parity + 效率轴超越」, 不再堆算力。

本轮不联网、不推送。
复现: python v11_alibi.py -> v11_alibi_results.json
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("v9", os.path.join(_HERE, "complete_beyond_v9.py"))
V9 = importlib.util.module_from_spec(_s); _s.loader.exec_module(V9)
_fd = importlib.util.spec_from_file_location("fd", os.path.join(_HERE, "final_decisive.py"))
FD = importlib.util.module_from_spec(_fd); _fd.loader.exec_module(FD)

sample_window = V9.sample_window
load_corpus = V9.load_corpus
Adam = V9.Adam
param_count = V9.param_count

OPT = dict(layers=6, M=64, alpha=0.0, beta=1.0, Hmult=4, lr=2e-3,
           curriculum=(48, 96, 192), name="L6_M64_H4_lr2e-3")
OOD = (384, 768, 1536)
SEEDS = (11, 22, 33)
CK = (2000, 4000, 6000)


def build3(cfg, sd):
    """三臂; A_R 去掉可学位置表(纯 ALiBi), 其余逐字相同。"""
    S, A_L = V9.build_pair(cfg, OPT, sd)
    _, A_R = V9.build_pair(cfg, OPT, sd)
    A_R.P = None                       # 纯相对位置, 无位置表
    # 除位置表外, A_R 与 A_L 逐参数一致(同 seed 构造), 显式断言
    pL, pR = A_L.params(), A_R.params()
    assert set(pL) - set(pR) == {"P"}, set(pL) - set(pR)
    assert set(pR) - set(pL) == set(), set(pR) - set(pL)
    for k in pR:
        assert np.array_equal(pL[k], pR[k]), k
    return S, A_L, A_R


def train3(S, A_L, A_R, cfg, sd, steps, attn_scale=1.0, checkpoints=()):
    Ls = OPT["curriculum"]
    opts = {n: Adam(a.params(), lr=OPT["lr"]) for n, a in
            (("S", S), ("A_L", A_L), ("A_R", A_R))}
    rngs = {n: np.random.default_rng(1000 + sd) for n in ("S", "A_L", "A_R")}
    rec = {}
    for s in range(1, steps + 1):
        lr = OPT["lr"] * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * (s - 1) / steps)))
        for n, a in (("S", S), ("A_L", A_L), ("A_R", A_R)):
            opts[n].lr = lr
            mult = attn_scale if n != "S" else 1
            for _ in range(int(mult)):
                Li = min(int(Ls[int(rngs[n].integers(0, len(Ls)))]), len(cfg["train"]) - 2)
                x, y, dts = sample_window(cfg["train"], Li, rngs[n])
                _, _, c = a.forward(x, dts, y)
                opts[n].step(a.backward(c))
        if s in checkpoints:
            rec[f"step{s}"] = measure3(cfg, S, A_L, A_R, sd)
    return rec


def _bpc(a, test, L, sd, n, dt_mode="fixed"):
    rng = np.random.default_rng(sd)
    ce, used = 0.0, 0
    for _ in range(n):
        if len(test) <= L + 1:
            break
        x, y, dts = sample_window(test, L, rng, dt_mode)
        _, loss, _ = a.forward(x, dts, y)
        ce += loss; used += 1
    return float(ce / max(used, 1) / math.log(2.0))


def measure3(cfg, S, A_L, A_R, sd, n_eval=120, n_ood=25):
    L = cfg["L"]
    vS = _bpc(S, cfg["test"], L, 7000 + sd, n_eval)
    vL = _bpc(A_L, cfg["test"], L, 7000 + sd, n_eval)
    vR = _bpc(A_R, cfg["test"], L, 7000 + sd, n_eval)
    a2 = {}
    for LL in OOD:
        sS = _bpc(S, cfg["test"], LL, 7100 + sd, n_ood)
        sL = _bpc(A_L, cfg["test"], LL, 7100 + sd, n_ood)
        sR = _bpc(A_R, cfg["test"], LL, 7100 + sd, n_ood)
        a2[f"L{LL}"] = dict(S=round(sS, 4), A_L=round(sL, 4), A_R=round(sR, 4),
                            d_vs_AL=round(sS - sL, 4), d_vs_AR=round(sS - sR, 4))
    tS = _bpc(S, cfg["test"], L, 7200 + sd, n_eval, "mixed")
    tR = _bpc(A_R, cfg["test"], L, 7200 + sd, n_eval, "mixed")
    d2 = min(v["d_vs_AR"] for v in a2.values())
    d2_old = min(v["d_vs_AL"] for v in a2.values())
    return dict(A1=round(vS - vR, 4), A1_abs=[round(vS, 4), round(vR, 4)],
                A1_vs_AL=round(vS - vL, 4),
                A2=a2, A2min=round(d2, 4), A2min_vs_AL=round(d2_old, 4),
                A3=round(tS - tR, 4), A3_abs=[round(tS, 4), round(tR, 4)],
                axes=bool(vS - vR <= 0.05 and d2 < -0.05 and tS - tR <= 0.05))


def main():
    tr, te, V, _ = load_corpus()
    cfg = dict(D=32, V=V, L=48, Vc=64, M=64, train=tr, test=te)
    print(f"[data] train={len(tr)} test={len(te)} vocab={V}", flush=True)
    print("=== A. 3 seeds x 6000 steps, curve, OOD L=(384,768,1536) ===", flush=True)
    per = {}
    for sd in SEEDS:
        S, A_L, A_R = build3(cfg, sd)
        per[f"seed{sd}"] = train3(S, A_L, A_R, cfg, sd, 6000,
                                  checkpoints=CK)
        r = per[f"seed{sd}"]
        print(f"  s{sd}: " + " ".join(
            f"{k[4:]}:{v['A1']:+.2f}/{v['A2min']:+.2f}/{v['A3']:+.2f}={v['axes']}"
            for k, v in r.items()), flush=True)
        last = r["step6000"]["A2"]
        print("        L384/768/1536 (S - A_R): " + " ".join(
            f"{k}={v['d_vs_AR']:+.4f}" for k, v in last.items()), flush=True)

    agg = {}
    for k in [f"step{c}" for c in CK]:
        rs = [per[f"seed{s}"][k] for s in SEEDS]
        A1 = float(np.mean([r["A1"] for r in rs]))
        A2 = float(np.mean([r["A2min"] for r in rs]))
        A2o = float(np.mean([r["A2min_vs_AL"] for r in rs]))
        A3 = float(np.mean([r["A3"] for r in rs]))
        agg[k] = dict(A1=round(A1, 4), A2min=round(A2, 4), A2min_vs_AL=round(A2o, 4),
                      A3=round(A3, 4),
                      all3=bool(A1 <= 0.05 and A2 < -0.05 and A3 <= 0.05),
                      per_seed_axes=[r["axes"] for r in rs])
    print("=== aggregate (vs A_R = 纯 ALiBi 无位置表) ===", flush=True)
    for c in CK:
        a = agg[f"step{c}"]
        print(f"  step{c:>5}: A1 {a['A1']:+.4f}  A2min {a['A2min']:+.4f}  "
              f"(旧对照 {a['A2min_vs_AL']:+.4f})  A3 {a['A3']:+.4f}  "
              f"3轴={a['all3']}  per-seed={sum(a['per_seed_axes'])}/3", flush=True)
    flags = [agg[f"step{c}"]["all3"] for c in CK]
    runs, cur = [], 0
    for f in flags:
        cur = cur + 1 if f else 0; runs.append(cur)
    stable = dict(longest_consecutive=max(runs), required=3,
                  stable_achieved=bool(max(runs) >= 3), flags=flags)

    print("=== B. 2x opponent (A_L/A_R 每步 2 次更新, 3 seeds x 2000 steps) ===", flush=True)
    fair = {}
    for sd in SEEDS:
        S, A_L, A_R = build3(cfg, sd)
        rec = train3(S, A_L, A_R, cfg, sd, 2000, attn_scale=2.0, checkpoints=(2000,))
        fair[f"seed{sd}"] = rec["step2000"]
        r = fair[f"seed{sd}"]
        print(f"  s{sd}: A1 {r['A1']:+.4f} A2min {r['A2min']:+.4f} "
              f"A3 {r['A3']:+.4f} 3轴={r['axes']}", flush=True)
    fair_all = all(v["axes"] for v in fair.values())

    print("=== C. CI / gradcheck / deterministic ===", flush=True)
    S, A_L, A_R = build3(cfg, 11)
    rng = np.random.default_rng(9000)
    x, y, dts = sample_window(tr, 48, rng)
    eS, bS = V9.grad_check(S, x, dts, y, rng)
    eR, bR = V9.grad_check(A_R, x, dts, y, rng)
    RP, _ = V9.load_real_purity()
    ci = V9.ci_audit_v9(RP, dict(cfg, ood_L=OOD))
    # 确定性: 两次短训逐位一致
    def one_short():
        s, al, ar = build3(cfg, 11)
        train3(s, al, ar, cfg, 11, 400, checkpoints=())
        return measure3(cfg, s, al, ar, 11, n_eval=30, n_ood=15)
    d1, d2 = one_short(), one_short()
    det = all(math.isclose(d1[k], d2[k], abs_tol=1e-12)
              for k in ("A1", "A2min", "A3"))
    print(f"  grad S={eS:.2e} A_R={eR:.2e}  CI {ci['passed']}/{ci['total']}  det={det}",
          flush=True)

    COMPLETE = bool(stable["stable_achieved"] and fair_all and ci["all_pass"]
                    and det and eS < 5e-3 and eR < 5e-3
                    and all(agg[f"step{c}"]["all3"] for c in CK[-3:]))
    out = dict(
        opt=OPT, ood_L=list(OOD), seeds=list(SEEDS), checkpoints=list(CK),
        per_seed=per, aggregate=agg, stable=stable, fair=fair, fair_all=fair_all,
        gradcheck=dict(S=eS, A_R=eR, bad=bS[:2] + bR[:2]),
        ci=dict(passed=ci["passed"], total=ci["total"], all_pass=ci["all_pass"],
                dual_err=ci["dual_err"], dual_bwd_err=ci["dual_bwd_err"]),
        deterministic=det,
        params=dict(S=param_count(S), A_L=param_count(A_L), A_R=param_count(A_R)),
        state_bytes=S.state_bytes(),
        deterministic_feasibility=FD.deterministic_feasibility(32),
        v10_summary=dict(N9_banded_vs_shared_gain=dict(A1=0.0108, A2min=0.0261, A3=0.0023),
                         N10_EMA_effect="0.995^1800≈1e-4, 逐位相同, 无效"),
        final=dict(COMPLETE_BEYOND=COMPLETE,
                   quality_stable=stable["stable_achieved"],
                   quality_2x_opponent=fair_all,
                   A1_at_6000=agg["step6000"]["A1"],
                   A2min_at_6000=agg["step6000"]["A2min"],
                   A2min_at_6000_vs_old_baseline=agg["step6000"]["A2min_vs_AL"],
                   A3_at_6000=agg["step6000"]["A3"],
                   ci_pass=ci["all_pass"], deterministic=det,
                   last3_all3=[agg[f"step{c}"]["all3"] for c in CK[-3:]]))
    json.dump(out, open("v11_alibi_results.json", "w"), ensure_ascii=False,
              indent=2, default=str)
    print("DONE -> v11_alibi_results.json", flush=True)
    print("FINAL:", json.dumps(out["final"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
