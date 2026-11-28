#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 v12 / v12.1 算子级替换成果推 TS2SC 源码仓；模型仓不动。"""
from __future__ import annotations
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import push_v10 as P
OUT = "/data/workspace/output"; V8 = "/data/workspace/ts2sc_v8"

SCRIPTS = ["operator_substitution_v12.py", "fit_operator_v12_1.py",
           "verify_v12_align.py", "verify_v12_exam.py",
           "pure_exam_v11.py", "verify_speaks_final.py"]
DOCS = ["V12_NATIVE_MIGRATION_VERDICT.md", "V12_NATIVE_MIGRATION_VERDICT.json",
        "V12_1_NATIVE_MIGRATION_VERDICT.md", "V12_1_NATIVE_MIGRATION_VERDICT.json",
        "v12_1_fit_report.json", "v12_1_operator_fit.npz",
        "v12_1_teacher_cache.json", "NATIVE_SIMPLCORE_STATUS.md",
        "DISTILL_PROTOCOL.md", "TS2SC_V10_FALSIFIABLE_LEDGER.md"]

def main():
    v = json.load(open(os.path.join(OUT, "V12_1_NATIVE_MIGRATION_VERDICT.json"),
                       encoding="utf-8"))["verdict"]
    f = {}
    for n in SCRIPTS:
        f["ts2sc/" + n] = open(os.path.join(V8, n), "rb").read()
    for n in DOCS:
        p = os.path.join(OUT, n)
        if os.path.exists(p):
            f["docs/" + n] = open(p, "rb").read()
    f["docs/V12_MODEL_REPO_POLICY.md"] = (
        "# Policy · v12/v12.1 native substitution does NOT touch model repo\n\n"
        "`MiniCPM5-1B-TS2SC` remains the hybrid (beta>0 sidecar) delivery.\n"
        "v12/v12.1 live in THIS source repo as architecture + evidence.\n"
        "No native checkpoint published as model weight unless\n"
        "THREE_CONSTRAINTS_MET is True. Zero entry deleted.\n").encode()
    msg = ("TS2SC v12/v12.1: Transformer -> SimplCore OPERATOR SUBSTITUTION "
           "(96 self_attn tensors removed from inference chain, 123 non-attn "
           "teacher tensors bit-identical, no T(x) residual / no hybrid); "
           "v12 random readout FAILED ability (cos 0.039, argmax 0/6); "
           "v12.1 closed-form analytic operator migration (per-layer ridge "
           "readout W_l + analytic injection ratio gamma_l, frozen spectral "
           "A/B/C untouched, Toeplitz/FFT-IIR/O(1) preserved); "
           "THREE_CONSTRAINTS_MET=%s OPERATOR_MIGRATION_EFFECTIVE=%s "
           "cos=%s argmax=%s heldout_cos=%s; "
           "DISTILL_PROTOCOL.md still GPU path; model repo untouched; "
           "zero deletion" % (
               v["THREE_CONSTRAINTS_MET"], v["OPERATOR_MIGRATION_EFFECTIVE"],
               v["hidden_cos_mean"], v["first_argmax_agree"],
               v["heldout_hidden_cos"]))
    c = P.push(P.SRC_REPO, f, msg)
    rep = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"),
               repo=f"{P.OWNER}/{P.SRC_REPO}", commit=c,
               files=len(f), verdict=dict(
                   THREE_CONSTRAINTS_MET=v["THREE_CONSTRAINTS_MET"],
                   OPERATOR_MIGRATION_EFFECTIVE=v["OPERATOR_MIGRATION_EFFECTIVE"],
                   hidden_cos_mean=v["hidden_cos_mean"],
                   first_argmax_agree=v["first_argmax_agree"]),
               model_repo_untouched=True, policy="NO entry deleted")
    json.dump(rep, open(os.path.join(OUT, "V12_1_PUSH_REPORT.json"), "w",
                        encoding="utf-8"), indent=2)
    print("[done]", json.dumps(rep, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
