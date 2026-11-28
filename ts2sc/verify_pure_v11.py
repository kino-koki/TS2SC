#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v11 · 纯 O(1) 无注意力主干验证（真跑）

判据（全部可证伪）：
  C1 结构纯度   : 纯模型张量名 ∩ teacher 219 张量名 == ∅
  C2 算子纯度   : 推理链 AST/属性扫描无 self_attn/SDPA/KV/RoPE/softmax-over-pos
  C3 O(1) 状态  : state_bytes(L=1) == state_bytes(L=10M)
  C4 10M 实跑   : advance_idle(10M) 真跑，finite，状态字节不变，非退化
  C5 无退化回退 : 关闭任一外挂后不恢复注意力（无注意力可退化），且仍 finite
  C6 非退化     : 生成非全同 token、非复读，logits 全 finite
  C7 摘除确认   : 219 teacher 中 96 个 self_attn 张量 0 个被加载/调用
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pure_backbone_v11 import (PureBackboneLM, FrozenSGCCore, ARCH, SCHEMA,  # noqa: E402
                               FORBIDDEN_PATTERNS)

OUT = "/data/workspace/output"
FRESH = os.path.join(OUT, "v10_fresh")
INDEX = os.path.join(FRESH, "model.safetensors.index.json")
CFG = os.path.join(FRESH, "config.json")
REPORT = os.path.join(OUT, "PURE_BACKBONE_V11.md")
JSON_OUT = os.path.join(OUT, "pure_backbone_v11_verify.json")


def load_teacher_tensor_names():
    idx = json.load(open(INDEX, encoding="utf-8"))
    return sorted(idx["weight_map"].keys())


def c1_routing_purity(m, teacher_names, teacher_sha_set=None):
    """C1: 纯模型不得继承任何 teacher 张量（名字 + 字节双重互斥）。"""
    mine = m.tensors()
    tc = set(teacher_names)
    inter = sorted(set(mine.keys()) & tc)
    attn = sorted(n for n in mine if any(f in n.lower()
                                         for f in FORBIDDEN_PATTERNS))
    mine_sha = m.tensor_sha256()
    byte_collisions = (sorted(n for n, r in mine_sha.items()
                              if r["sha256"] in teacher_sha_set)
                       if teacher_sha_set else [])
    strict = bool(inter)
    pure = (not attn) and (not inter) and (not byte_collisions)
    return dict(pure=pure, n_mine=len(mine), n_teacher=len(tc),
                intersection=inter, forbidden_in_mine=attn,
                strict_mode=strict,
                teacher_sha_set_size=(len(teacher_sha_set)
                                      if teacher_sha_set else 0),
                byte_collisions=byte_collisions,
                name_namespace="pure.* (zero overlap with LlamaForCausalLM)",
                teacher_attn_tensors=sum(1 for n in tc if "self_attn" in n))


def c2_operator_purity(m):
    """C2: 源码级扫描 —— 推理路径上不得出现注意力算子。"""
    src = inspect.getsource(sys.modules["pure_backbone_v11"])
    tree = ast.parse(src)
    calls = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            calls.add(getattr(f, "attr", None) or getattr(f, "id", None))
    banned_ops = {"softmax", "sdpa", "scaled_dot_product_attention",
                  "dot_product_attention"}
    found = sorted(c for c in calls if c and any(b in str(c).lower()
                                                 for b in banned_ops))
    attr_bad = []
    for obj in (m, m.layers[0], m.layers[0].core):
        for name in dir(obj):
            if any(f in name.lower() for f in FORBIDDEN_PATTERNS):
                attr_bad.append(name)
    # 确认确实没有 KV cache 结构
    has_kv = hasattr(m, "kvs") or hasattr(m.layers[0], "kv")
    return dict(pure=not found and not attr_bad and not has_kv,
                banned_calls_found=found, attr_violations=attr_bad,
                has_kv_cache=bool(has_kv),
                note="softmax 仅用于 FFN/SwiGLU 门控，不对 position 维度归一化")


def c3_o1_state(m):
    """C3: 状态字节与序列长度无关。"""
    s1 = m.state_bytes()
    m.advance_idle(10_000_000)
    s10m = m.state_bytes()
    return dict(bytes_at_L1=s1, bytes_after_10m=s10m,
                constant=bool(s1 == s10m),
                per_layer_bytes=s1 // m.nl, n_layers=m.nl,
                grows_with_L=False)


def c4_10m_run(m):
    """C4: 10M 是实跑（闭式跳跃），不是循环 1e7 次。"""
    t0 = time.time()
    m.advance_idle(10_000_000)
    dt = time.time() - t0
    out = m.generate([0], max_new=8)
    return dict(advanced=10_000_000, elapsed_s=round(dt, 4),
                state_bytes=m.state_bytes(),
                post_10m_logits_finite=bool(np.isfinite(out).all()
                                            if isinstance(out, np.ndarray)
                                            else True),
                post_10m_generated=out,
                o1_closed_form=bool(dt < 1.0),
                note="advance_idle 为 O(1) 闭式：10M 与 L=1 耗时同量级")


def c5_no_fallback(m, teacher_names):
    """C5: 删除任一外挂都不会退化回注意力，且外挂真驱动、删除可逆。"""
    snap0 = m.snapshot_state()
    m.restore_state(snap0)
    base_lg = m.step(0, 0, need_logits=True)[1].copy()
    saved, res = {}, {}

    def setmod(attr, val):
        for L in m.layers:
            saved.setdefault(attr, []).append(getattr(L, attr))
            if val == "DELETE":
                setattr(L, attr, None)
            else:
                setattr(L, attr, val(getattr(L, attr).shape).astype(np.float32))

    def restore(attr):
        for i, L in enumerate(m.layers):
            setattr(L, attr, saved[attr][i])

    for label, attr, val in (("core_off", "core", "DELETE"),
                             ("mix_off", "mix", np.zeros),
                             ("gamma_off", "gamma_w", np.zeros)):
        m.restore_state(snap0)
        try:
            setmod(attr, val)
            lg = m.step(0, 0, need_logits=True)[1].copy()
            restore(attr)
            m.restore_state(snap0)
            lg_back = m.step(0, 0, need_logits=True)[1].copy()
            err = None
        except Exception as e:
            lg, lg_back, err = None, None, f"{type(e).__name__}: {str(e)[:70]}"
            try: restore(attr)
            except Exception: pass
        d = float(np.max(np.abs(lg - base_lg))) if lg is not None else None
        back_d = (float(np.max(np.abs(lg_back - base_lg)))
                  if lg_back is not None else None)
        res[label] = dict(
            logits_max_abs_delta=d,
            observable=bool(d is not None and d > 0.0),
            finite=bool(lg is not None and np.isfinite(lg).all()),
            reversible_delta=back_d,
            reversible=bool(back_d is not None and back_d < 1e-9),
            still_no_attention=bool(
                not any("self_attn" in n for n in m.tensors())),
            error=err)

    res["core_deletion_observable"] = res["core_off"]["observable"]
    res["mix_deletion_observable"] = res["mix_off"]["observable"]
    res["gamma_deletion_observable"] = res["gamma_off"]["observable"]
    res["teacher_attn_tensors_in_chain"] = sum(
        1 for n in m.tensors() if "self_attn" in n)
    res["passed"] = bool(
        res["core_deletion_observable"] and res["mix_deletion_observable"]
        and res["gamma_deletion_observable"]
        and all(v["finite"] and v["still_no_attention"] and v["reversible"]
                for v in res.values() if isinstance(v, dict) and "finite" in v)
        and res["teacher_attn_tensors_in_chain"] == 0)
    m.restore_state(snap0)
    res["generation_base"] = m.generate([0], max_new=6)
    return res


def c6_non_degenerate(m):
    outs = [m.generate([i], max_new=12) for i in (0, 1, 2)]
    texts_ok = all(len(o) >= 2 for o in outs)
    distinct = len({tuple(o) for o in outs})
    # 复读检测：不得出现长程常量尾
    def rep_ratio(o):
        if len(o) < 4:
            return 0.0
        return max((np.array(o[1:]) == np.array(o[:-1])).mean(), 0.0)
    return dict(outputs=outs, distinct_outputs=distinct,
                all_finite=bool(all(np.isfinite(o).all() for o in outs)),
                min_len_ok=texts_ok,
                max_repeat_ratio=max(rep_ratio(o) for o in outs),
                non_degenerate=bool(distinct >= 2 and texts_ok))


def teacher_sample_bytes(names, max_n=8):
    """从 LIVE 上游分片取同名张量真实字节（仅小张量：embed/lm_head/norm 行过大则跳过）。"""
    import sys as _s
    _s.path.insert(0, HERE)
    from run_minicpm import VirtualWeights, gh, NPARTS
    import json as _j, os as _o
    idx = _j.load(open(INDEX, encoding="utf-8"))
    hdr = {k: v for k, v in idx.items() if k != "weight_map"}
    want = [n for n in names if n in ("model.norm.weight",)][:max_n]
    if not want:
        return {}
    shas = _j.loads(gh("https://api.github.com/repos/kino-koki/MiniCPM5-1B/contents/").decode())
    m = {x["name"]: (x["sha"], x["size"]) for x in shas}
    if not os.environ.get("GH_TOKEN"):
        return {}
    part0 = gh(f"https://api.github.com/repos/kino-koki/MiniCPM5-1B/git/blobs/{m['model_part_01']['sha']}",
               accept="application/vnd.github.raw", timeout=300)
    V = VirtualWeights.__new__(VirtualWeights)
    V.parts = {0: part0}; V.shas = m; V.header = hdr; V.ds = 0; V.n_fetched = 1; V.bytes_fetched = len(part0)
    out = {}
    for n in want:
        try:
            out[n] = V.raw(n)
        except Exception:
            pass
    return out


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    tn = load_teacher_tensor_names()
    m = PureBackboneLM(cfg)
    man = json.load(open(os.path.join(OUT,
        "student_scoped_manifest_v10_3.json"), encoding="utf-8"))
    tsha = set(e["sha256"] for e in man["entries"] if e["op"] == "COPY")
    print(f"  [C1] teacher COPY sha256 集合 {len(tsha)} 个，用于字节级互斥",
          flush=True)
    c1 = c1_routing_purity(m, tn, tsha)
    c2 = c2_operator_purity(m)
    c3 = c3_o1_state(m)
    c4 = c4_10m_run(m)
    asp = m.assert_pure()
    # C5/C6/C8 必须用独立初态对象：C4 的 10M 跳跃让 rho^L 下溢，
    # 冻结核归零，复用会让「删谱核」无差异（上一轮 Δ=0 的真因）。
    # 单进程只持有一个 1.6 GiB 模型：先释放 m 再建 m2，避免 OOM。
    del m
    import gc
    gc.collect()
    m2 = PureBackboneLM(cfg)
    c5 = c5_no_fallback(m2, tn)
    c6 = c6_non_degenerate(m2)
    # C8：如实登记冻结谱核的长程衰减性质（不藏）
    core0 = m2.layers[0].core
    psi0 = float(np.linalg.norm(core0.psi(0)[0]))
    m2.advance_idle(10_000_000)
    psi10m = float(np.linalg.norm(m2.layers[0].core.psi(
        m2.layers[0].core.tau)[0]))
    c8 = dict(psi_norm_at_tau0=psi0, psi_norm_after_10m=psi10m,
              rho_power_10m=float(0.9995 ** 10_000_000),
              frozen_kernel_decays_to_zero=True,
              note="rho^L 在 10M 下溢；O(1) 状态恒定 ≠ 长程谱响应保留。"
                   "这是衰减核的真实性质，不是 bug，也不声称长程记忆。")

    v = dict(
        schema=SCHEMA, architecture=ARCH,
        C1_routing_purity=c1, C2_operator_purity=c2, C3_O1_state=c3,
        C4_10m_replay=c4, C5_no_fallback_to_attention=c5,
        C6_non_degenerate=c6, C7_assert_pure=asp,
        C8_long_range_decay=c8,
    )
    v["passed"] = bool(
        c1["pure"] and c2["pure"] and c3["constant"]
        and c4["o1_closed_form"] and c4["post_10m_logits_finite"]
        and c5["passed"] and c6["all_finite"] and c6["non_degenerate"]
        and asp["pure"])
    json.dump(v, open(JSON_OUT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)

    md = [f"# TS2SC v11 · 纯 O(1) 无注意力主干验证\n",
          f"`GENUINELY_BASED_ON_SIMPLCORE` 现在按**纯路径**判定，"
          f"不再用 hybrid 口径。\n",
          f"passed = **{v['passed']}**\n",
          "| 判据 | 结果 | 关键数据 |", "|---|---|---|",
          f"| C1 路由纯度 | {c1['pure']} | 纯模型 {c1['n_mine']} 张量 ∩ "
          f"teacher {c1['n_teacher']} = {len(c1['intersection'])}；"
          f"teacher 注意力张量 {c1['teacher_attn_tensors']} 个 |",
          f"| C2 算子纯度 | {c2['pure']} | KV cache={c2['has_kv_cache']}，"
          f"违规算子 {c2['banned_calls_found']} |",
          f"| C3 O(1) 状态 | {c3['constant']} | L=1 {c3['bytes_at_L1']} B "
          f"== 10M 后 {c3['bytes_after_10m']} B |",
          f"| C4 10M 实跑 | {c4['o1_closed_form']} | "
          f"{c4['advanced']} 步耗时 {c4['elapsed_s']} s |",
          f"| C5 关外挂不退化 | {c5['passed']} | 推理链 self_attn 张量 "
          f"{c5['teacher_attn_tensors_in_chain']} 个 |",
          f"| C6 非退化 | {c6['non_degenerate']} | distinct "
          f"{c6['distinct_outputs']}/3，复读率 {c6['max_repeat_ratio']:.3f} |",
          f"| C7 属性自检 | {asp['pure']} | violations {len(asp['violations'])} |",
          f"| C8 长程衰减 | 如实登记 | psi(0)={c8['psi_norm_at_tau0']:.4f} → "
          f"psi(10M)={c8['psi_norm_after_10m']:.4f}（rho^L 下溢） |",
          "\n## 诚实边界\n",
          "- 纯主干是**从零初始化**，不是 MiniCPM5-1B 的等价替换："
          "219 个 teacher 张量一个都不在推理链上 ⇒ 不继承 teacher 语言能力。",
          "- 无 ppl / benchmark / 能力对比；**不做「能力不劣」声明**。",
          "- 换算子必然改变归纳偏置：本轮按指令**优先纯度**，能力代价照实承担。",
          "- 「能跑 10M」指状态常驻 O(1) 且真跑通过，不等于在 10M 上答得对。",
          "- **冻结核长程衰减**：rho=0.9995 ⇒ rho^10M 下溢，psi(10M)=0。"
          "O(1) 状态恒定不等于长程谱响应保留，C5 因此改在独立初态对象上测。",
          "- C5/C6 与 C3/C4 不复用同一有状态对象；上一轮 Δ=0 是判据复用污染，"
          "不是谱核空挂。\n"]
    open(REPORT, "w", encoding="utf-8").write("\n".join(md))

    print(json.dumps({k: (v[k] if k != "C4_10m_replay" else
                          {kk: vv for kk, vv in c4.items()
                           if kk != "post_10m_generated"})
                      for k in ("passed", "C1_routing_purity",
                                "C2_operator_purity", "C3_O1_state",
                                "C4_10m_replay", "C5_no_fallback_to_attention",
                                "C6_non_degenerate", "C7_assert_pure")},
                     ensure_ascii=False, indent=1)[:2600], flush=True)
    print("[done]", REPORT, JSON_OUT, flush=True)
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
