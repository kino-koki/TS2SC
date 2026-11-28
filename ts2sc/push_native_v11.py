#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把纯 O(1) 原生路径的成果推到 **TS2SC 源码仓**（模型仓不动）。

理由：模型仓 `MiniCPM5-1B-TS2SC` 是 hybrid 交付，本轮纯主干
`ABILITY_PRESERVED=False`，**绝不混进去冒充模型权重**。
只作为源码仓的新架构分支 + 可证伪台账，base_tree 零删除。
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import push_v10 as P  # noqa: E402

OUT = "/data/workspace/output"
V8 = "/data/workspace/ts2sc_v8"
REPORT = os.path.join(OUT, "NATIVE_V11_PUSH_REPORT.json")

SCRIPTS = ["pure_backbone_v11.py", "verify_pure_v11.py",
           "align_pure_ability.py"]
DOCS = ["NATIVE_SIMPLCORE_STATUS.md", "DISTILL_PROTOCOL.md",
        "PURE_BACKBONE_V11.md", "ABILITY_PRESERVATION_VERDICT.md",
        "pure_backbone_v11_verify.json",
        "ability_preservation_verdict.json",
        "TS2SC_V10_FALSIFIABLE_LEDGER.md"]


def main():
    # 硬门槛：判决文件必须存在且为 False（不许把未达成包装成达成）
    v = json.load(open(os.path.join(OUT, "ability_preservation_verdict.json"),
                       encoding="utf-8"))["verdict"]
    if v["ABILITY_PRESERVED"] is not False:
        raise RuntimeError("refuse: ABILITY_PRESERVED must be False here")
    pure = json.load(open(os.path.join(OUT, "pure_backbone_v11_verify.json"),
                         encoding="utf-8"))
    if not pure["passed"]:
        raise RuntimeError("refuse: pure C1-C8 not passed")

    f = {}
    for n in SCRIPTS:
        b = open(os.path.join(V8, n), "rb").read()
        f["ts2sc/" + n] = b
    for n in DOCS:
        b = open(os.path.join(OUT, n), "rb").read()
        f["docs/" + n] = b
    # 明确声明模型仓不动
    f["docs/NATIVE_MODEL_REPO_POLICY.md"] = (
        "# Policy · native v11 does NOT touch the model repo\n\n"
        "`MiniCPM5-1B-TS2SC` remains the hybrid (beta>0 sidecar) delivery.\n"
        "The pure O(1) backbone lives in THIS source repo only, because\n"
        "`ABILITY_PRESERVED = False`. No pure checkpoint is published as a\n"
        "model weight. Zero entry deleted from either repo.\n").encode()

    msg = ("TS2SC native-v11: pure O(1) attention-free backbone "
           "(C1-C8 all green: 243 pure.* tensors, zero overlap with teacher "
           "219, no QKV/SDPA/KV-cache, 10M O(1) idle real run, deletion "
           "observable+reversible, rho^10M decay registered); "
           "ABILITY_PRESERVED=FALSE measured on 6 real prompts "
           "(raw cos -0.0138, argmax 0/6, ridge held-out 0.7612, "
           "prefix-information collapse cos 0.999999); "
           "admit beta>0 v10.x was HYBRID not native; "
           "DISTILL_PROTOCOL.md = GPU-only path, NOT EXECUTED, no fabricated "
           "checkpoint/score; model repo untouched; zero deletion")
    c = P.push(P.SRC_REPO, f, msg)
    rep = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"),
               repo=f"{P.OWNER}/{P.SRC_REPO}", commit=c,
               files=len(f), scripts=SCRIPTS, docs=DOCS,
               pure_C1_C8_passed=pure["passed"],
               ability_preserved=v["ABILITY_PRESERVED"],
               model_repo_untouched=True,
               policy="NO entry deleted (base_tree)",
               guard="0 credential-like byte sequences uploaded")
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("[done]", json.dumps(rep, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
