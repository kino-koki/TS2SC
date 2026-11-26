#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · CLI / stage state machine

One entry point for the whole system. Every downstream stage REFUSES to run
until its prerequisites passed. This turns "发布日再改硬编码" into a gate.

    python ts2sc.py --stage preflight
    python ts2sc.py --stage plan   --teacher DIR --index keys.json
    python ts2sc.py --stage bind   --teacher DIR --out DIR
    python ts2sc.py --stage migrate --dry-run        # -> BLOCKED at T-minus
    python ts2sc.py --stage status

Exit codes: 0 pass / 2 gate blocked / 3 T+0 runtime missing / 4 assertion fail
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STAGES = ("preflight", "plan", "bind", "migrate", "align", "accept", "status")
# Hard prerequisites. A stage cannot be entered unless all listed stages passed.
PREREQ = {
    "preflight": [],
    "plan": ["preflight"],
    "bind": ["preflight", "plan"],
    "migrate": ["preflight", "plan", "bind"],
    "align": ["preflight", "plan", "bind", "migrate"],
    "accept": ["preflight", "plan", "bind", "migrate", "align"],
    "status": [],
}
# Stages that need a real teacher forward on the release-day machine.
REQUIRES_T0 = ("migrate", "align", "accept")


def state_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "ts2sc_state.json")


def load_state(out_dir: str) -> dict:
    p = state_path(out_dir)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return dict(version="v8.1", stages={}, created=time.strftime("%Y-%m-%dT%H:%M:%S"))


def save_state(out_dir: str, st: dict):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(st, open(state_path(out_dir), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


def gate(state: dict, stage: str) -> dict:
    missing = [p for p in PREREQ[stage]
               if not state["stages"].get(p, {}).get("passed")]
    return dict(passed=not missing, missing=missing)


def run(args) -> int:
    st = load_state(args.out)
    stage = args.stage

    if stage == "status":
        print(json.dumps(dict(version=st["version"],
                              stages={k: dict(passed=v.get("passed"),
                                              status=v.get("status"))
                                      for k, v in st["stages"].items()},
                              t0_pending=[s for s in REQUIRES_T0
                                          if not st["stages"].get(s, {}).get("passed")]),
                         indent=2, ensure_ascii=False))
        return 0

    g = gate(st, stage)
    if not g["passed"]:
        rec = dict(passed=False, status="GATE_BLOCKED", missing=g["missing"],
                   note="downstream stage refused; prerequisites have not passed")
        st["stages"][stage] = rec; save_state(args.out, st)
        print(f"[{stage}] GATE_BLOCKED missing={g['missing']}")
        return 2

    if stage in REQUIRES_T0 and not args.t0:
        rec = dict(passed=False, status="T0_REQUIRED",
                   reason="needs real config + tokenizer + safetensors + teacher forward",
                   honest="No checkpoint, no logits, no metrics were produced.")
        st["stages"][stage] = rec; save_state(args.out, st)
        print(f"[{stage}] T0_REQUIRED (use --t0 only on the release-day machine)")
        return 3

    if stage == "preflight":
        import preflight
        preflight.main()
        rep = json.load(open(os.path.join(HERE, "preflight_report.json"), encoding="utf-8"))
        a = rep["part_A_real_source_audit"]; b = rep["part_B_selftest_fixture"]
        c = rep.get("part_C_fusion_invariants", {})
        ok = (a["n_present"] == a["n_findings"] and b["static_contract_pass"]
              and b["negative_controls_all_caught"] and b["provenance"]["passed"]
              and not b["mojibake_gate"]["v8_passed"]
              and c.get("passed", False))
        rec = dict(passed=ok, status="PASS" if ok else "FAIL",
                   root_causes=f"{a['n_present']}/{a['n_findings']}",
                   fusion_invariants=c.get("checks", {}))
    elif stage == "plan":
        from spec import TeacherSpec
        from migrate import plan
        s = TeacherSpec.from_dir(args.teacher)
        try:
            keys = json.load(open(args.index, encoding="utf-8")) if args.index else []
        except (OSError, ValueError) as e:
            rec = dict(passed=False, status="FAIL",
                       failures=["INDEX_UNREADABLE:" + type(e).__name__],
                       note="pass --index with a JSON list of real safetensors keys")
            st["stages"][stage] = rec; save_state(args.out, st)
            print(f"[{stage}] FAIL index_unreadable")
            return 4
        p = plan(s, keys)
        rec = dict(passed=bool(p["passed"]), status="PASS" if p["passed"] else "FAIL",
                   n_copy=p["n_copy"], n_derive=p["n_derive"], n_new=p["n_new"],
                   blockers=p["blockers"])
    elif stage == "bind":
        from spec import TeacherSpec
        from contract import static_contract
        s = TeacherSpec.from_dir(args.teacher)
        r = static_contract(args.teacher, args.out, s)
        rec = dict(passed=r["passed"], status="PASS" if r["passed"] else "FAIL",
                   failures=r["failures"])
    elif stage in REQUIRES_T0:
        # --t0 given: still refuse to fake it in this sandbox.
        rec = dict(passed=False, status="NOT_IMPLEMENTED_AT_T_MINUS",
                   honest="Real T+0 execution is delivered on the release-day machine; "
                          "this sandbox has no torch and no Qwen4 data.")
    else:
        rec = dict(passed=False, status="UNKNOWN")

    st["stages"][stage] = rec; save_state(args.out, st)
    print(f"[{stage}] {rec['status']}")
    return 0 if rec["passed"] else 4


def main():
    ap = argparse.ArgumentParser(description="TS2SC v8 stage state machine")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--teacher", default=os.path.join(HERE, "_selftest", "fixture", "teacher"))
    ap.add_argument("--out", default=os.path.join(HERE, "_run"))
    ap.add_argument("--index", default=None, help="JSON list of real safetensors keys")
    ap.add_argument("--t0", action="store_true",
                    help="declare that this is the release-day machine with real runtime")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan/verify only; never writes a student checkpoint")
    a = ap.parse_args()
    if a.dry_run and a.stage in REQUIRES_T0:
        print(f"[{a.stage}] DRY-RUN: no checkpoint will be written; "
              "T+0 runtime still required for real execution")
    sys.exit(run(a))


if __name__ == "__main__":
    main()
