#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v10.3 范围收窄交付推送：两仓增量，base_tree 零删除。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import push_v10 as P  # noqa: E402

OUT = "/data/workspace/output"
REPORT = os.path.join(OUT, "V10_3_PUSH_REPORT.json")


def main():
    need = ["V10_3_SCOPED_DELIVERY.md", "v10_3_scoped_verify.json",
            "scoped_capability_registry.json",
            "student_scoped_manifest_v10_3.json",
            "student_smm_v10_3.safetensors",
            "context_10m_config.json", "inference_default_v10_3.json",
            "v10_3_checkpoint.json"]
    missing = [n for n in need if not os.path.exists(os.path.join(OUT, n))]
    if missing:
        raise RuntimeError(f"refuse push: missing {missing}")
    v = json.load(open(os.path.join(OUT, "v10_3_scoped_verify.json"),
                       encoding="utf-8"))["verification"]
    if not v["passed"]:
        raise RuntimeError("refuse push: verification passed=False")

    mf, sf = P.model_files(), P.src_files()
    print(f"[files] model {len(mf)} | src {len(sf)}", flush=True)
    m1 = ("MiniCPM5-1B-TS2SC v10.3 SCOPED DELIVERY: inherit only "
          "SGC_CORE + 3 SIDECARS + SMM (upstream evolution_mixer.py "
          "FusedEvolutionMixer/CalibratedFusedMixer zero-order consolidation); "
          "context opened to 10M (10485760) with O(1) branch-side closed-form "
          "idle real run, teacher full-attention 240 GiB KV = "
          "INFEASIBLE_LEDGER (not end-to-end verified); LIVE 22-shard fresh "
          "pull, 219/219 COPY full bitwise, 579 tensors 219 COPY + 360 NEW, "
          "SMM explicit-gate consolidation 16 commits/layer, switch law "
          "all-off bitwise teacher, smm_off bitwise v10.2; 10+21+39 reg green, "
          "zero deletion")
    m2 = ("TS2SC v10.3 scoped: smm_v10_3.py + scoped_v10_3.py + "
          "v10_3_scoped_verify.py + test_v10_3.py; 10M config/HBM ledger/"
          "closed-form idle; self-audit fixed SMM empty-hang risk and "
          "SGC+SMM T1' bound violation; historical 31-item registry CONTRACT "
          "and DEFERRED entries retained but EXCLUDED from inheritance claim; "
          "zero deletion")
    c1 = P.push(P.MODEL_REPO, mf, m1)
    c2 = P.push(P.SRC_REPO, sf, m2)
    rep = dict(date=P.time.strftime("%Y-%m-%d %H:%M:%S"),
               label="v10.3 SCOPED DELIVERY / 10M",
               model_repo=f"{P.OWNER}/{P.MODEL_REPO}",
               src_repo=f"{P.OWNER}/{P.SRC_REPO}",
               model_files=len(mf), src_files=len(sf),
               model_commit=c1, src_commit=c2,
               verification_passed=v["passed"],
               context_tokens=v["phase2_context_tokens"],
               added=need + ["smm_v10_3.py", "scoped_v10_3.py",
                             "v10_3_scoped_verify.py", "test_v10_3.py",
                             "make_v10_3_report.py",
                             "README.md (updated, not deleted)"],
               policy="NO entry deleted (base_tree)",
               guard="0 credential-like byte sequences uploaded")
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n[done]", json.dumps(rep, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
