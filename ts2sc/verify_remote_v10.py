#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v10 推送后终审：仓 / 公开 / 许可 / β>0 / 分片 / 账本 / 零删除 / 凭据 / 判据。"""
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

T = os.environ["GH_TOKEN"]
OWNER, MODEL, SRC = "kino-koki", "MiniCPM5-1B-TS2SC", "TS2SC"


def api(p):
    r = urllib.request.Request("https://api.github.com" + p, headers={
        "Authorization": "Bearer " + T,
        "Accept": "application/vnd.github+json", "User-Agent": "TS2SC-v10"})
    with urllib.request.urlopen(r, timeout=60) as x:
        return json.load(x)


def raw(repo, path):
    u = ("https://api.github.com/repos/kino-koki/" + repo + "/contents/"
         + urllib.parse.quote(path))
    r = urllib.request.Request(u, headers={
        "Authorization": "Bearer " + T,
        "Accept": "application/vnd.github.raw", "User-Agent": "TS2SC-v10"})
    return urllib.request.urlopen(r, timeout=60).read()


def gitsha(b):
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()


def tree(repo):
    h = api(f"/repos/{OWNER}/{repo}/git/ref/heads/main")["object"]["sha"]
    c = api(f"/repos/{OWNER}/{repo}/git/commits/{h}")
    t = api(f"/repos/{OWNER}/{repo}/git/trees/{c['tree']['sha']}?recursive=1")
    return ({e["path"]: e["sha"] for e in t["tree"] if e["type"] == "blob"},
            h, c["message"])


def main():
    r = api(f"/repos/{OWNER}/{MODEL}")
    print("=== 1 仓 / 公开 / 许可 ===")
    print("  full_name =", r["full_name"], "| private =", r["private"],
          "| license =", (r.get("license") or {}).get("spdx_id"))

    sm, mh, mmsg = tree(MODEL)
    ss, sh, smsg = tree(SRC)
    print("\n=== 2 提交 ===")
    print("  model", mh[:10], "|", mmsg[:92])
    print("  src  ", sh[:10], "|", smsg[:92])

    print("\n=== 3 v10 交付文件 ===")
    need = ["LICENSE", "NOTICE", "README.md", "inference_default_v10.json",
            "inference_default.json", "student_full_manifest_v10.json",
            "weights_v10/MANIFEST.json", "weights_v10/reassemble.py",
            "replay_full_v10.py", "upstream_pull_report.json",
            "v10_fresh_verify.json", "V10_FRESH_MIGRATION_REPORT.md",
            "SGC_V10_SYNC.md", "sgc_v10.py", "fresh_migrate_v10.py",
            "test_v10.py", "v10_copy_full_audit.json",
            "v10_multilingual_verify.json", "V10_DEEP_AUDIT_REPORT.md",
            "TS2SC_V10_FALSIFIABLE_LEDGER.md", "v10_deep_audit.py",
            "calibrate_beta_v10.py", "evidence/gen_deep_checkpoint.json",
            "V10_1_READ_CONTRACT_REPORT.md", "v10_read_contract_audit.py",
            "v10_0_delta_baseline.json", "v10_read_contract_audit.json",
            "v10_1_bitwise_equivalence.json",
            "SWITCH_LAW.md", "capability_registry.json",
            "v10_switch_law_audit.json", "v10_switch_law_audit.py",
            "V10_3_SCOPED_DELIVERY.md", "scoped_capability_registry.json",
            "v10_3_scoped_verify.json",
            "student_scoped_manifest_v10_3.json",
            "student_smm_v10_3.safetensors",
            "context_10m_config.json", "inference_default_v10_3.json",
            "smm_v10_3.py", "scoped_v10_3.py", "test_v10_3.py"]
    for p in need:
        print(f"  {'OK ' if p in sm else 'MISS'} {p}")
    shards = [p for p in sm if p.startswith("weights_v10/student_branch_v10.part")]
    print("  v10 shards:", len(shards), sorted(shards))
    print("  old weights/ kept:",
          len([p for p in sm if p.startswith("weights/student_branch.part")]))
    print("  src sgc_v10:", "ts2sc/sgc_v10.py" in ss)
    print("  src upstream snapshot:",
          len([p for p in ss if p.startswith("ts2sc/upstream_simplcore_v11/")]))

    print("\n=== 4 β>0 交付口径 ===")
    idf = json.loads(raw(MODEL, "inference_default_v10.json"))
    root = json.loads(raw(MODEL, "inference_default.json"))
    man = json.loads(raw(MODEL, "weights_v10/MANIFEST.json"))
    print("  beta              =", idf["beta"], "| root =", root["beta"])
    print("  architecture      =", idf["architecture"])
    print("  identity_adapter  =", idf["identity_adapter"])
    print("  delivery_beta     =", man.get("delivery_beta"))
    md = raw(MODEL, "README.md").decode("utf-8", "replace")
    print("  README v10 声明    :", "v10" in md and "β = 1.0" in md)
    print("  README 零复用声明  :", "零复用" in md)
    print("  β>0 交付 =", bool(idf["beta"] > 0 and root["beta"] == 1.0
                              and man.get("delivery_beta") == 1.0
                              and idf["identity_adapter"].startswith("none")))

    print("\n=== 5 账本 / 拉取证据（远端字节） ===")
    lm = json.loads(raw(MODEL, "student_full_manifest_v10.json"))
    print("  n_tensors", lm["n_tensors"], "| counts", lm["counts"],
          "| unclassified", lm["unclassified"])
    pull = json.loads(raw(MODEL, "upstream_pull_report.json"))
    print("  pull shards", pull["n_shards"],
          "| all blob sha verified", pull["all_blob_sha_verified"],
          "| total", round(pull["total_bytes"] / 2**30, 2), "GiB")
    print("  upstream head", pull["head"])
    vv = json.loads(raw(MODEL, "v10_fresh_verify.json"))["verification"]
    print("  verify passed", vv["passed"], "| leaks",
          vv["identity_marker_leaks"], "| copy ok",
          vv["copy_spot_checks_ok"], "| P7",
          vv["core_unchanged_by_sidecars"])

    print("\n=== 5b 深审（继续轮）远端字节核对 ===")
    fa = json.loads(raw(MODEL, "v10_copy_full_audit.json"))
    dv = json.loads(raw(MODEL, "v10_multilingual_verify.json"))
    dvv = dv["verification"]
    print("  full COPY audit", fa["n_copy"], "/", fa["n_copy"],
          "| mismatched", fa["n_mismatched"],
          "| all_identical", fa["all_bit_identical"])
    print("  deep languages", dvv["n_languages"],
          "| arms", dvv["n_arms_total"],
          "| T==C0", dvv["T_equals_C0_tokenwise_all"],
          "| active_all", dvv["architecture_active_all"],
          "| leaks", dvv["identity_marker_leaks"],
          "| distinct C1", dvv["distinct_C1_texts"])
    print("  deep passed", dvv["passed"])
    ledger = raw(MODEL, "TS2SC_V10_FALSIFIABLE_LEDGER.md").decode("utf-8", "replace")
    print("  ledger has COMPLETE_BEYOND=False:",
          "COMPLETE_BEYOND=False" in ledger,
          "| has 2x veto:", "2× 对手步数对照失败" in ledger)
    cal = raw(MODEL, "calibrate_beta_v10.py").decode("utf-8", "replace")
    print("  calibration skeleton NOT_EXECUTED:",
          "NOT_EXECUTED" in cal and "--execute" in cal)

    print("\n=== 5c v10.1 读取契约远端核对 ===")
    rc = json.loads(raw(MODEL, "v10_read_contract_audit.json"))
    eq = json.loads(raw(MODEL, "v10_1_bitwise_equivalence.json"))
    print("  ACTIVE", rc["n_active"], "/ PERSISTED_INIT", rc["n_persisted_init"],
          "/ NOT_CONTRIBUTING", rc["n_not_contributing"],
          "| all_accounted", rc["all_keys_accounted_for"])
    print("  v10.1 vs v10.0 Δ", eq["identical"], "/", eq["n"],
          "| all_identical", eq["all_identical"])
    rcs = raw(MODEL, "sgc_v10.py").decode("utf-8", "replace")
    print("  remote sgc_v10 has beta_at_init:",
          "beta_at_init" in rcs,
          "| has verified_read_keys:", "verified_read_keys" in rcs)
    # 权重字节未变：分片 git blob sha 仍与上轮一致
    print("  weight shards unchanged (git shas):",
          [sm[p][:10] for p in shards])

    print("\n=== 5d v10.2 开关法则远端核对 ===")
    sw = json.loads(raw(MODEL, "v10_switch_law_audit.json"))
    reg = json.loads(raw(MODEL, "capability_registry.json"))
    sv = sw["verification"]
    print("  passed", sv["passed"],
          "| A all_off_zero", sv["A_all_off_bitwise_zero"],
          "| B default", sv["B_single_off_default_observable"], "/13",
          "| C activated 13/13", sv["C_single_off_activated_13_of_13"],
          "| D all_on 27/27", sv["D_all_on_bitwise_identical"])
    print("  E finite/T1", sv["E_random_all_finite"], sv["E_random_within_T1"],
          "| F contract inert", sv["F_contract_flags_never_change_forward"],
          "| G deferred refused", sv["G_deferred_refused"],
          "| H 24x13", sv["H_24x13_broadcast"])
    print("  registry", sv["registry_slots_with_switch"], "/",
          sv["registry_total"], "slots |",
          sv["forward_runnable"], "FORWARD +",
          sv["contract_registered"], "CONTRACT +",
          sv["deferred_not_in_source"], "DEFERRED")
    rcs2 = raw(MODEL, "sgc_v10.py").decode("utf-8", "replace")
    print("  remote has CAP_KEYS:", "CAP_KEYS" in rcs2,
          "| all_off:", "def all_off" in rcs2,
          "| contract_flags:", "contract_flags" in rcs2,
          "| deferred_flags:", "deferred_flags" in rcs2)

    print("\n=== 5e v10.3 范围收窄 + 10M 远端核对 ===")
    v3 = json.loads(raw(MODEL, "v10_3_scoped_verify.json"))
    q3 = v3["verification"]
    reg3 = json.loads(raw(MODEL, "scoped_capability_registry.json"))
    cfg10 = json.loads(raw(MODEL, "context_10m_config.json"))
    man3 = json.loads(raw(MODEL, "student_scoped_manifest_v10_3.json"))
    print("  passed", q3["passed"],
          "| groups", [g["id"] for g in reg3["groups"]],
          "| context", cfg10["max_position_embeddings"])
    print("  manifest", man3["n_tensors"], man3["counts"],
          "| unclassified", man3["unclassified"])
    print("  COPY 219 full bitwise", q3["phase1_copy_219_full_bitwise"],
          "| branch rebuild identical",
          q3["phase1_branch_rebuild_identical"])
    print("  branch 10M state constant", q3["phase2_branch_state_constant"],
          "| bytes", q3["phase2_branch_state_bytes"],
          "| attention", q3["phase2_attention_status"],
          q3["phase2_attention_kv_gib"], "GiB",
          "| RoPE", q3["phase2_rope_extrapolation"], "x")
    print("  SMM commits/layer",
          q3["smm_explicit_consolidation"]["commits_per_layer"],
          "| all 24 layers",
          q3["smm_explicit_consolidation"]["all_24_layers"])
    print("  T==OFF", q3["T_equals_OFF_tokenwise"],
          "| ON!=OFF", q3["ON_differs_from_OFF"],
          "| all finite", q3["all_logits_finite"],
          "| all gate", q3["all_decode_gate"],
          "| leaks", q3["identity_marker_leaks"])
    smm_blob = raw(MODEL, "student_smm_v10_3.safetensors")
    print("  SMM checkpoint sha matches manifest:",
          __import__("hashlib").sha256(smm_blob).hexdigest()
          == man3["smm_checkpoint"]["sha256"])

    print("\n=== 6 字节一致（本地 → 远端） ===")
    sys.path.insert(0, "/data/workspace/ts2sc_v8")
    os.environ.setdefault("GH_TOKEN", T)
    import push_v10 as P
    mfiles = P.model_files()
    sfiles = P.src_files()
    print("  model", sum(1 for p, b in mfiles.items()
                         if p in sm and gitsha(b) == sm[p]), "/", len(mfiles))
    print("  src  ", sum(1 for p, b in sfiles.items()
                         if p in ss and gitsha(b) == ss[p]), "/", len(sfiles))
    print("  v10 shards", sum(1 for p, b in mfiles.items()
                              if p.startswith("weights_v10/student_branch_v10.part")
                              and p in sm and gitsha(b) == sm[p]), "/", len(shards))

    print("\n=== 7 零删除 ===")
    old_sha = "8bc757a25a4b05c54089d8f92e865d2ca18a9c3b"
    ot = api(f"/repos/{OWNER}/{SRC}/git/trees/"
             f"{api(f'/repos/{OWNER}/{SRC}/git/commits/{old_sha}')['tree']['sha']}"
             "?recursive=1")
    so = {e["path"] for e in ot["tree"] if e["type"] == "blob"}
    print("  original TS2SC blobs", len(so), "| still", len(so & set(ss)),
          "| DELETED", sorted(so - set(ss)))
    print("  model identity_* retained:",
          len([p for p in sm if p.startswith("identity_")]))

    print("\n=== 8 凭据零命中 ===")
    from credential_guard import scan_text
    hits = 0
    for p in ["README.md", "NOTICE", "LICENSE", "SGC_V10_SYNC.md",
              "V10_FRESH_MIGRATION_REPORT.md", "inference_default_v10.json",
              "v10_fresh_verify.json"]:
        if p in sm:
            hits += len(scan_text(raw(MODEL, p).decode("utf-8", "replace"), p))
    print("  remote text hits =", hits)

    print("\n=== 9 回归 ===")
    import subprocess
    for name in ("test_v8.py", "test_v10.py"):
        r = subprocess.run([sys.executable, f"/data/workspace/ts2sc_v8/{name}"],
                           capture_output=True, text=True, timeout=120)
        line = [l for l in r.stdout.splitlines() if l.startswith("[report]")]
        print(f"  {name}: rc={r.returncode} {line[-1][:110] if line else ''}")


if __name__ == "__main__":
    main()
