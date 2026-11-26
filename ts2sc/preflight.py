#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · T-minus Preflight

Runs TODAY, on a machine with no torch / no GPU / no released Qwen4-27B.
Two kinds of output, explicitly separated:

  A. REAL SOURCE AUDIT -- reads the actual V7 files in /data/workspace and
     reports which historical 乱码 root causes are literally present.
  B. SELF-TEST FIXTURE -- synthetic 65-token teacher dir used ONLY to prove
     the v8 gates fire. It is NOT Qwen4 data and is not reported as such.

No number below is a model result.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spec import TeacherSpec, QWEN4_27B_PENDING          # noqa: E402
from contract import static_contract                      # noqa: E402
from provenance import classify                           # noqa: E402
from align_accept import evaluate, mojibake_rate, wilson_ci  # noqa: E402

V7 = "/data/workspace/simplcore_src/Simpl-Cloud/SimplCore_架构源码/02_1.7B生产链路_V7"
# Keep the fixture UNDER _selftest/fixture: rmtree() must never wipe the CLI
# state dir (that was a real bug caught by the end-to-end stage run).
SELFTEST = os.path.join(HERE, "_selftest", "fixture")


def src(name):
    p = os.path.join(V7, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def audit_v7():
    mig, mod, bpe, ver, trn = (src("migrate_qwen3.py"), src("modeling_simplcore_v7.py"),
                              src("bpe.py"), src("verify_arch.py"), src("train_simplcore.py"))
    doc_pre = open("/data/workspace/simplcore_src/TS2SC_VL47_PREFLIGHT.md", encoding="utf-8").read()
    doc_add = open("/data/workspace/src_probe/simplcore_git/PAPER_TS2SC_ADDENDUM.md", encoding="utf-8").read()
    findings = []

    def f(id_, title, hit, evidence, fix):
        findings.append(dict(id=id_, title=title, present_in_v7=bool(hit),
                             evidence=evidence, v8_fix=fix))

    f("R1", "lm_head 被丢弃 / tied 被强制",
      'tie_word_embeddings=True' in mig or '"tie_word_embeddings": True' in mig,
      "migrate_qwen3.py 只 put embed_tokens.weight；cfg 强制 tie=True；Qwen3-1.7B 实为 untied",
      "spec.student_config() 继承 tie；provenance 要求 untied 时显式 COPY lm_head.weight")
    f("R2", "tokenizer 复制是 best-effort",
      "if os.path.exists(src_file):" in mig and "shutil.copy(src_file" in mig,
      "migrate_qwen3.py L171-175：缺文件静默跳过，无断言；历史 tokenizer.json 是 133B LFS 指针",
      "contract.static_contract 六文件逐个 REQUIRED + LFS 指针检测 + FAIL 即拒发")
    f("R3", "自造词表子集 BPE 与 embedding 不对齐",
      "vocab_size: int = 2048" in bpe or "self.V = min(vocab_size" in bpe,
      "bpe.py QwenBPESubset 取 2048/1024 前缀 + 自造 <unk>；decode 的 except 直接返回 B2U 字符串",
      "禁用子集 tokenizer；runtime_contract 要求 len(tokenizer)==embed.rows 与 5 条往返")
    f("R4", "验收只测『非退化』，不测语义等价",
      "def gen_quality" in ver and "mojibake" not in ver and "paired" not in ver,
      "verify_arch.py gen_quality 仅 n/max_run/rep3/distinct1；拉丁扩展乱码 distinct 高、rep3 低 → PASS",
      "align_accept L0-L4；mojibake_rate + 配对 teacher 对照 + Wilson CI")
    f("R5", "NEW 参数零初始化使 mixer 近乎失效",
      "torch.tensor(-2.0)" in mig and "torch.zeros(d, d)" in mig,
      "alpha=-2 → sigmoid=0.119；Wc/Wg/wgate=0；mixer 贡献约 0.06×ys，注意力能力被结构性压掉",
      "alpha 必须经 align_accept 校准；L2 要求 attention-path 贡献非零且已标定")
    f("R6", "GQA 零填充导致一半 slot 恒零",
      "zero-padded to (d,d)" in mig and "Wk[:k.shape[0]] = k" in mig,
      "k/v_proj 从 (d/2,d) 补零到 (d,d)，R=2 切两 slot，slot1 的 K/V 恒 0",
      "spec 由 num_key_value_heads*head_dim 推导 slot；provenance 禁止 zero padding")
    f("R7", "Qwen3 形状硬编码，Qwen4 必失效",
      all(s in mig for s in ("151643", "151645", "40960")) or "R = 2" in mig,
      "bos/eos/max_position/R=2 全写死；Qwen4-27B 无 config、可能 QSA/MoE/1M ctx",
      "TeacherSpec 零硬编码，缺字段即 T0 blocker；MOE_DETECTED 需声明 per-expert 映射")
    f("R8", "q_norm/k_norm 被 SKIP 且无功能对应",
      '"self_attn.q_norm.weight"' in mig and "SKIP" in mig,
      "migrate_qwen3.py L134-139 直接 SKIP Qwen3 per-head QK-norm",
      "provenance 改 DERIVE FOLD_INTO_WqWk，并声明变换")
    f("R9", "函数等价从未验证，只验了逐位复制",
      "assertions=dict(" in mig and "copy_bit_identical" in mig
      and "kl_div" not in mig and "teacher_logits" not in mig,
      "migrate 的断言只有 keys_differ/no_self_attn/copy_bit_identical；无 teacher forward、无 KL、无 ppl",
      "align_accept L2：逐层 hidden cos / logit KL / top-1 一致率 / same-condition teacher 对照")
    f("R10", "历史失败已被登记但未闭环",
      "0.8125" in doc_pre and "otionalotional" in doc_pre and "W14" in doc_add,
      "VL47_PREFLIGHT 记 rep3 0.8125 / 0.9286、『中国的首都是→城市，有，北京，北京…』；"
      "PAPER_TS2SC_ADDENDUM W14 记 RoPE/mask/embedding-lm_head 四处缺陷",
      "v8 把 W14 的四处缺陷逐条做成 L0/L1 硬门禁，而不是留在撤回登记里")

    return dict(audit_target=V7, n_findings=len(findings),
                n_present=sum(1 for x in findings if x["present_in_v7"]),
                findings=findings,
                note="REAL SOURCE AUDIT: all hits are literal string matches in the local V7 files")


def selftest():
    """Synthetic 65-token teacher. Proves the gates fire; NOT Qwen4 data."""
    if os.path.exists(SELFTEST):
        shutil.rmtree(SELFTEST)
    tdir, sdir = os.path.join(SELFTEST, "teacher"), os.path.join(SELFTEST, "student")
    os.makedirs(tdir); os.makedirs(sdir)
    cfg = dict(model_type="toy", vocab_size=65, hidden_size=2048,
               num_hidden_layers=2, intermediate_size=6144,
               tie_word_embeddings=False, max_position_embeddings=4096,
               bos_token_id=0, eos_token_id=1, torch_dtype="bfloat16",
               num_attention_heads=16, num_key_value_heads=8, head_dim=128,
               rms_norm_eps=1e-6)
    json.dump(cfg, open(os.path.join(tdir, "config.json"), "w"))
    vocab = {f"tok{i}": i for i in range(65)}
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "generation_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(tdir, "config.json"), os.path.join(tdir, fn))
    json.dump(vocab, open(os.path.join(tdir, "vocab.json"), "w"))
    json.dump(dict(chat_template="{{ bos_token }}"), open(os.path.join(tdir, "tokenizer_config.json"), "w"))
    json.dump(dict(bos_token=dict(id=0), eos_token=dict(id=1)),
              open(os.path.join(tdir, "special_tokens_map.json"), "w"))
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "generation_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(tdir, fn), os.path.join(sdir, fn))

    spec = TeacherSpec.from_dir(tdir)
    spec.blockers = [b for b in spec.blockers if "MOE" not in b]  # toy has no MoE
    good = static_contract(tdir, sdir, spec)

    # Negative controls: each must make the gate FAIL.
    neg = {}
    open(os.path.join(sdir, "tokenizer.json"), "wb").write(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\n")
    neg["lfs_pointer"] = not static_contract(tdir, sdir, spec)["passed"]
    shutil.copy(os.path.join(tdir, "tokenizer.json"), os.path.join(sdir, "tokenizer.json"))
    # OPTIONAL files (vocab.json/merges.txt) may legitimately be absent in
    # tokenizer.json-only releases; delete a REQUIRED file instead.
    os.remove(os.path.join(sdir, "special_tokens_map.json"))
    neg["missing_file"] = not static_contract(tdir, sdir, spec)["passed"]
    shutil.copy(os.path.join(tdir, "special_tokens_map.json"),
                os.path.join(sdir, "special_tokens_map.json"))
    json.dump({f"t{i}": i for i in range(64)}, open(os.path.join(sdir, "vocab.json"), "w"))
    neg["vocab_mismatch"] = not static_contract(tdir, sdir, spec)["passed"]
    json.dump(vocab, open(os.path.join(sdir, "vocab.json"), "w"))

    keys = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for i in range(2):
        p = f"model.layers.{i}."
        keys += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                 p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                 p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight",
                 p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight",
                 p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"]
    led = classify(keys, spec)

    fake = [dict(prompt="中国的首都是", teacher_ok=True, student_ok=True,
                 student_text="¤Ĳ城市 «闪店 ±展 北京北京", eos_hit=True) for _ in range(20)]
    old_style_would_pass = (max(len(set(r["student_text"])) for r in fake) > 6)
    gate = evaluate(fake)

    return dict(
        note="SELF-TEST FIXTURE: synthetic 65-token teacher, NOT Qwen4-27B data",
        spec=dict(n_slots=spec.derived.get("n_slots"),
                  tie_inherited=spec.student_config().get("tie_word_embeddings"),
                  blockers=spec.blockers),
        static_contract_pass=good["passed"],
        static_contract_failures=good["failures"],
        negative_controls=neg,
        negative_controls_all_caught=all(neg.values()),
        provenance=dict(n_teacher=led["n_teacher"], counts=led["counts"],
                        unclassified=len(led["unclassified"]), passed=led["passed"],
                        lm_head_recorded=any(e["src"] == "lm_head.weight" for e in led["entries"]),
                        qknorm_folded=any(e["transform"] == "FOLD_INTO_WqWk" for e in led["entries"])),
        mojibake_gate=dict(old_gen_quality_would_pass=old_style_would_pass,
                           v8_passed=gate["passed"],
                           avg_mojibake_rate=round(gate["avg_mojibake_rate"], 4),
                           ci95=[round(x, 4) for x in wilson_ci(20, 20)]),
    )


def fusion_invariants():
    """part_C: the fusion-layer invariants of v8.1, checked on pure functions."""
    from modeling_v8 import (build_slot_map, fold_qk_norm, blend_output,
                             equivalence_invariant, lm_head_plan)
    d, kv_dim = 2048, 1024                      # Qwen3-shaped fixture, NOT Qwen4
    m = build_slot_map(d, kv_dim, 16)
    inv = equivalence_invariant()
    checks, failures = {}, []

    checks["no_zero_padding"] = (m["zero_padded_rows"] == 0
                                and all(s["all_rows_mapped"] for s in m["slots"]))
    if not checks["no_zero_padding"]:
        failures.append("SLOT_ZERO_PADDING")
    checks["slot_coverage"] = (sum(s["n_rows"] for s in m["slots"]) == d
                               and m["n_slots"] == d // kv_dim)
    if not checks["slot_coverage"]:
        failures.append("SLOT_COVERAGE")

    q = fold_qk_norm(kv_dim, 128, 1.0)
    checks["qknorm_folded_not_skipped"] = (q["transform"] == "FOLD_INTO_WqWk"
                                           and q["dropped_columns"] == 0)
    if not checks["qknorm_folded_not_skipped"]:
        failures.append("QKNORM_DROPPED")

    ty, sy = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    checks["disabled_bit_identical"] = blend_output(ty, sy, 1.0, False) == ty
    checks["enabled_beta0_identical"] = blend_output(ty, sy, 0.0, True) == ty
    checks["enabled_beta_nonzero_differs"] = blend_output(ty, sy, 0.1, True) != ty
    for k in ("disabled_bit_identical", "enabled_beta0_identical",
              "enabled_beta_nonzero_differs"):
        if not checks[k]:
            failures.append(k.upper())

    p_tied = lm_head_plan(True, False)
    p_untied = lm_head_plan(False, True)
    checks["lm_head_tied_plan"] = p_tied["passed"]
    checks["lm_head_untied_plan"] = p_untied["passed"] and p_untied["copy_standalone"]
    if not (p_tied["passed"] and p_untied["passed"]):
        failures.append("LM_HEAD_PLAN")
    checks["beta_init_is_zero"] = (inv["beta_init"] == 0.0)
    checks["calibration_required"] = inv["calibration_required"]
    checks["no_alpha_minus2"] = "alpha=-2 hardcoded" in inv["forbidden"]
    checks["hybrid_not_replacement"] = (
        "removing teacher self_attn / QSA / MoE layers" in inv["forbidden"])

    return dict(passed=not failures, failures=failures, checks=checks,
                note="pure-function fixture; NOT a torch forward, NOT Qwen4 data",
                engineering_paper_boundary=(
                    "v8 fusion keeps teacher attention (hybrid); the ICML manuscript "
                    "characterises a static-kernel no-softmax construction. "
                    "Neither validates the other."))


def main():
    rep = dict(
        title="TS2SC v8.1 · Qwen4-27B T-minus preflight",
        date="2026-11-25",
        part_A_real_source_audit=audit_v7(),
        part_B_selftest_fixture=selftest(),
        part_C_fusion_invariants=fusion_invariants(),
        qwen4_27b_status=QWEN4_27B_PENDING,
        t0_blockers=[
            "real config.json (0 hardcoded fields may be guessed)",
            "real tokenizer.json + 5 companion files, none LFS pointers",
            "safetensors index with full tensor key list (COPY/DERIVE/NEW coverage = 100%)",
            "declared tie_word_embeddings and lm_head key presence",
            "declared attention layout (QSA / MoE / GQA) and per-expert mapping if MoE",
            "teacher forward on the same machine for L2/L3 paired control",
            "serving-path check: HF/vLLM native decode, no custom BPE shim",
        ],
        honest_bound=(
            "No Qwen4-27B weights, config or tokenizer were accessed. Nothing here is a "
            "migration result, a perplexity number, a capability claim or a release-date claim."
        ),
    )
    out = os.path.join(HERE, "preflight_report.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    a, b, c = (rep["part_A_real_source_audit"], rep["part_B_selftest_fixture"],
               rep["part_C_fusion_invariants"])
    print(f"[A] REAL V7 audit: {a['n_present']}/{a['n_findings']} historical root causes literally present")
    for x in a["findings"]:
        print(f"    {x['id']:4s} {'HIT ' if x['present_in_v7'] else 'miss'} {x['title']}")
    print(f"[B] selftest: contract={b['static_contract_pass']} "
          f"neg_controls={b['negative_controls_all_caught']} "
          f"provenance={b['provenance']['passed']} "
          f"mojibake_gate_rejects={not b['mojibake_gate']['v8_passed']}")
    print(f"[C] fusion invariants: passed={c['passed']} "
          f"({len(c['checks'])} checks, {len(c['failures'])} failures)")
    print("    ->", out)


if __name__ == "__main__":
    main()
