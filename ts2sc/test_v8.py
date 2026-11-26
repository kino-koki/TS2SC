#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.1 · Regression suite (pure stdlib, runs in the sandbox)

    python -m unittest -v test_v8.py
    python test_v8.py          # also writes test_report.json

Every test is a real assertion. Nothing here loads Qwen4 data or torch.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spec import TeacherSpec                                    # noqa: E402
from contract import static_contract                              # noqa: E402
from provenance import classify                                   # noqa: E402
from modeling_v8 import (build_slot_map, fold_qk_norm, blend_output,   # noqa: E402
                         equivalence_invariant, lm_head_plan)
from align_accept import evaluate, mojibake_rate, wilson_ci, GATES  # noqa: E402
from migrate import plan, copy_assert, simulate                   # noqa: E402
from ts2sc import gate, PREREQ, REQUIRES_T0                      # noqa: E402

TMP = os.path.join(HERE, "_test")


def make_teacher(vocab=65, tie=False, with_attn=True):
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    t, s = os.path.join(TMP, "teacher"), os.path.join(TMP, "student")
    os.makedirs(t); os.makedirs(s)
    cfg = dict(model_type="toy", vocab_size=vocab, hidden_size=2048,
               num_hidden_layers=2, intermediate_size=6144,
               tie_word_embeddings=tie, max_position_embeddings=4096,
               bos_token_id=0, eos_token_id=1, torch_dtype="bfloat16",
               rms_norm_eps=1e-6)
    if with_attn:
        cfg.update(num_attention_heads=16, num_key_value_heads=8, head_dim=128)
    json.dump(cfg, open(os.path.join(t, "config.json"), "w"))
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "generation_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(t, "config.json"), os.path.join(t, fn))
    json.dump({f"t{i}": i for i in range(vocab)}, open(os.path.join(t, "vocab.json"), "w"))
    json.dump(dict(chat_template="{{ bos_token }}"),
              open(os.path.join(t, "tokenizer_config.json"), "w"))
    json.dump(dict(bos_token=dict(id=0), eos_token=dict(id=1)),
              open(os.path.join(t, "special_tokens_map.json"), "w"))
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "generation_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(t, fn), os.path.join(s, fn))
    return t, s


def teacher_keys(nl=2, tie=False):
    k = ["model.embed_tokens.weight", "model.norm.weight"]
    if not tie:
        k.append("lm_head.weight")
    for i in range(nl):
        p = f"model.layers.{i}."
        k += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
              p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
              p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight",
              p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight",
              p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
              p + "mlp.down_proj.weight"]
    return k


class TestSpec(unittest.TestCase):
    def test_no_hardcoded_shapes(self):
        t, _ = make_teacher()
        s = TeacherSpec.from_dir(t)
        self.assertEqual(s.derived["n_slots"], 2)
        self.assertEqual(s.student_config()["tie_word_embeddings"], False)
        self.assertEqual(s.student_config()["vocab_size"], 65)

    def test_missing_required_is_blocker(self):
        t, _ = make_teacher(with_attn=False)
        cfg = json.load(open(os.path.join(t, "config.json")))
        cfg.pop("num_attention_heads", None); cfg.pop("num_key_value_heads", None)
        cfg.pop("head_dim", None); cfg.pop("eos_token_id")
        json.dump(cfg, open(os.path.join(t, "config.json"), "w"))
        s = TeacherSpec.from_dir(t)
        self.assertIn("eos_token_id", s.missing_required)
        self.assertTrue(any("ATTENTION" in b for b in s.blockers))
        self.assertFalse(s.t0_ready)

    def test_tie_inherited_not_forced(self):
        for tie in (True, False):
            t, _ = make_teacher(tie=tie)
            s = TeacherSpec.from_dir(t)
            self.assertEqual(s.student_config()["tie_word_embeddings"], tie)


class TestContract(unittest.TestCase):
    def test_happy_path(self):
        t, s = make_teacher()
        spec = TeacherSpec.from_dir(t)
        self.assertTrue(static_contract(t, s, spec)["passed"])

    def test_lfs_pointer_rejected(self):
        t, s = make_teacher()
        spec = TeacherSpec.from_dir(t)
        open(os.path.join(s, "tokenizer.json"), "wb").write(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:x\n")
        r = static_contract(t, s, spec)
        self.assertFalse(r["passed"])
        self.assertTrue(any("LFS_POINTER" in f for f in r["failures"]))

    def test_missing_file_rejected(self):
        t, s = make_teacher()
        spec = TeacherSpec.from_dir(t)
        # vocab.json / merges.txt are OPTIONAL in tokenizer.json-only releases;
        # delete a REQUIRED file instead (v8.2 real-release adaptation).
        os.remove(os.path.join(s, "special_tokens_map.json"))
        r = static_contract(t, s, spec)
        self.assertFalse(r["passed"])
        self.assertTrue(any("special_tokens_map" in f for f in r["failures"]))

    def test_tokenizer_json_only_shape_allowed(self):
        """Real MiniCPM5-1B shape: no loose vocab.json / merges.txt."""
        t, s = make_teacher()
        spec = TeacherSpec.from_dir(t)
        os.remove(os.path.join(s, "vocab.json"))
        os.remove(os.path.join(s, "merges.txt"))
        self.assertTrue(static_contract(t, s, spec)["passed"])

    def test_optional_lfs_pointer_still_rejected(self):
        t, s = make_teacher()
        spec = TeacherSpec.from_dir(t)
        open(os.path.join(s, "merges.txt"), "wb").write(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:x\n")
        r = static_contract(t, s, spec)
        self.assertFalse(r["passed"])
        self.assertTrue(any("LFS_POINTER" in f for f in r["failures"]))

    def test_standalone_chat_template_recognised(self):
        t, s = make_teacher()
        spec = TeacherSpec.from_dir(t)
        # remove chat_template from tokenizer_config, add standalone .jinja
        tc = os.path.join(s, "tokenizer_config.json")
        json.dump({}, open(tc, "w"))
        open(os.path.join(s, "chat_template.jinja"), "w", encoding="utf-8").write(
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}" * 40)
        r = static_contract(t, s, spec)
        self.assertTrue(r["checks"]["chat_template_present"])
        self.assertEqual(r["checks"]["chat_template_source"], "chat_template.jinja")

    def test_vocab_embedding_mismatch_rejected(self):
        t, s = make_teacher()
        spec = TeacherSpec.from_dir(t)
        json.dump({f"x{i}": i for i in range(64)}, open(os.path.join(s, "vocab.json"), "w"))
        r = static_contract(t, s, spec)
        self.assertFalse(r["passed"])
        self.assertTrue(any("VOCAB_EMBEDDING_MISMATCH" in f for f in r["failures"]))

    def test_special_id_out_of_range_rejected(self):
        t, s = make_teacher(vocab=2)
        spec = TeacherSpec.from_dir(t)
        json.dump({f"t{i}": i for i in range(2)}, open(os.path.join(s, "vocab.json"), "w"))
        json.dump(dict(bos_token=dict(id=0), eos_token=dict(id=999)),
                  open(os.path.join(s, "special_tokens_map.json"), "w"))
        r = static_contract(t, s, spec)
        self.assertFalse(r["passed"])


class TestProvenance(unittest.TestCase):
    def test_full_coverage_and_lm_head(self):
        t, _ = make_teacher(tie=False)
        spec = TeacherSpec.from_dir(t)
        led = classify(teacher_keys(tie=False), spec)
        self.assertEqual(len(led["unclassified"]), 0)
        self.assertTrue(any(e["src"] == "lm_head.weight" for e in led["entries"]))
        self.assertTrue(any(e["transform"] == "FOLD_INTO_WqWk" for e in led["entries"]))
        self.assertTrue(led["passed"])

    def test_untied_without_lm_head_blocked(self):
        t, _ = make_teacher(tie=False)
        spec = TeacherSpec.from_dir(t)
        led = classify(teacher_keys(tie=True), spec)
        self.assertFalse(led["passed"])
        self.assertTrue(any("UNTIED" in b for b in led["blockers"]))

    def test_unknown_tensor_blocked(self):
        t, _ = make_teacher()
        spec = TeacherSpec.from_dir(t)
        led = classify(teacher_keys() + ["model.mystery.weight"], spec)
        self.assertFalse(led["passed"])
        self.assertEqual(len(led["unclassified"]), 1)


class TestFusion(unittest.TestCase):
    def test_slot_map_no_zero_padding(self):
        m = build_slot_map(2048, 1024, 16)
        self.assertEqual(m["n_slots"], 2)
        self.assertEqual(m["zero_padded_rows"], 0)
        self.assertTrue(all(s["all_rows_mapped"] for s in m["slots"]))
        self.assertEqual(sorted(set(m["row_to_teacher"])), list(range(1024)))

    def test_indivisible_hidden_raises(self):
        with self.assertRaises(ValueError):
            build_slot_map(2049, 1024, 16)

    def test_qknorm_folded(self):
        q = fold_qk_norm(1024, 128, 1.0)
        self.assertEqual(q["transform"], "FOLD_INTO_WqWk")
        self.assertEqual(q["dropped_columns"], 0)
        self.assertEqual(q["folded_columns"], 1024)

    def test_disabled_is_bit_identical(self):
        ty, sy = [1.0, 2.0], [9.0, 9.0]
        self.assertEqual(blend_output(ty, sy, 1.0, False), ty)   # I1
        self.assertEqual(blend_output(ty, sy, 0.0, True), ty)     # I2
        self.assertNotEqual(blend_output(ty, sy, 0.1, True), ty)

    def test_lm_head_plans(self):
        self.assertEqual(lm_head_plan(True, False)["op"], "TIE_TO_EMBED")
        self.assertEqual(lm_head_plan(False, True)["op"], "COPY_STANDALONE")
        self.assertFalse(lm_head_plan(None, True)["passed"])

    def test_invariants_forbid_replacement(self):
        inv = equivalence_invariant()
        self.assertTrue(inv["I1_disabled_identical"])
        self.assertTrue(inv["calibration_required"])
        self.assertIn("removing teacher self_attn / QSA / MoE layers", inv["forbidden"])


class TestAcceptance(unittest.TestCase):
    def test_old_gate_passes_mojibake_v8_rejects(self):
        fake = [dict(prompt="中国的首都是", teacher_ok=True, student_ok=True,
                     student_text="¤Ĳ城市 «闪店 ±展 北京北京", eos_hit=True)
                for _ in range(20)]
        old_would_pass = len(set(fake[0]["student_text"])) > 6 and "北京北京" in fake[0]["student_text"]
        r = evaluate(fake)
        self.assertTrue(old_would_pass)
        self.assertFalse(r["passed"])
        self.assertGreater(r["avg_mojibake_rate"], 0.01)

    def test_clean_paired_control_passes(self):
        clean = [dict(prompt=f"q{i}", teacher_ok=True, student_ok=True,
                      student_text="北京是中国的首都。", eos_hit=True)
                 for i in range(200)]
        r = evaluate(clean)
        self.assertTrue(r["passed"])
        self.assertEqual(r["level"], "L3")
        self.assertEqual(r["paired_rate"], 1.0)

    def test_mojibake_rate(self):
        self.assertEqual(mojibake_rate(""), 1.0)
        self.assertEqual(mojibake_rate("北京"), 0.0)
        self.assertGreater(mojibake_rate("¤Ĳ厅"), 0.0)

    def test_wilson_ci_edges(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))
        lo, hi = wilson_ci(200, 200)
        self.assertGreater(lo, 0.98); self.assertLessEqual(hi, 1.0)

    def test_five_gates_declared(self):
        self.assertEqual(list(GATES.keys()),
                         ["L0_STRUCTURE", "L1_CODEC", "L2_FUNCTION", "L3_BEHAVIOUR", "L4_NATIVE"])


class TestMigrationAndCLI(unittest.TestCase):
    def test_plan_and_simulate(self):
        t, _ = make_teacher()
        spec = TeacherSpec.from_dir(t)
        p = plan(spec, teacher_keys())
        self.assertTrue(p["passed"])
        self.assertEqual(p["slot_map"]["n_slots"], 2)
        self.assertEqual(p["lm_head_plan"]["op"], "COPY_STANDALONE")
        sim = simulate(spec, teacher_keys())
        self.assertTrue(sim["plan_passed"])
        self.assertEqual(sim["zero_padded_rows"], 0)
        self.assertTrue(sim["copy_assert_rejects_bitflip"])

    def test_copy_assert_catches_dtype_and_shape(self):
        base = dict(shape=[4, 4], dtype="BF16", bytes=b"abcd")
        self.assertTrue(copy_assert(base, dict(base))["passed"])
        self.assertFalse(copy_assert(base, dict(base, dtype="F32"))["passed"])
        self.assertFalse(copy_assert(base, dict(base, shape=[4, 5]))["passed"])

    def test_cli_gate_blocks_downstream(self):
        st = dict(stages={})
        self.assertTrue(gate(st, "preflight")["passed"])
        g = gate(st, "migrate")
        self.assertFalse(g["passed"])
        self.assertEqual(g["missing"], ["preflight", "plan", "bind"])
        st["stages"] = {k: dict(passed=True) for k in ("preflight", "plan", "bind")}
        self.assertTrue(gate(st, "migrate")["passed"])
        self.assertFalse(gate(st, "accept")["passed"])

    def test_stage_order_and_t0(self):
        self.assertEqual(PREREQ["accept"][-1], "align")
        self.assertEqual(set(REQUIRES_T0), {"migrate", "align", "accept"})


def collect_cases(suite):
    """Collect BEFORE runner.run(): unittest clears _tests entries afterwards."""
    out = []
    def walk(s):
        for x in s:
            if isinstance(x, unittest.TestSuite):
                walk(x)
            elif x is not None:
                out.append(x.id().replace("test_v8.", ""))
    walk(suite)
    return out


class TestFullFeatures(unittest.TestCase):
    """v8.3: 全特性继承必须可持续回归，不是一次性跑通。"""
    def test_registry_complete(self):
        from features import FEATURES, RUNNABLE, REGISTERED, NOT_INHERITED, summary
        s = summary()
        self.assertEqual(s["n_features"], 29)
        self.assertEqual(s["n_runnable"], 20)
        self.assertEqual(s["n_registered"], 7)
        self.assertEqual(s["n_not_inherited"], 2)
        self.assertFalse(s["complete_inheritance"])   # v8.6 口径修正，钉死
        self.assertEqual(len(FEATURES), 29)
        self.assertIn("F08", [f[0] for f in RUNNABLE])
        self.assertIn("F09", [f[0] for f in REGISTERED])
        self.assertIn("role bit", " ".join(s["real_gap_found"]).lower())
        for fid in ("F25", "F26", "F27"):
            self.assertIn(fid, [f[0] for f in RUNNABLE])
        self.assertEqual([f[0] for f in NOT_INHERITED], ["F28", "F29"])
        self.assertIn("无词表", NOT_INHERITED[0][1])

    def test_fft_iir_duality(self):
        from run_full_migration import verify_fft_iir_duality
        r = verify_fft_iir_duality()
        self.assertTrue(r["passed"])
        self.assertLess(r["dense_vs_iir"], 1e-8)
        self.assertLess(r["dense_vs_fft"], 1e-8)

    def test_causality_and_outer_product(self):
        from run_full_migration import verify_causality, verify_outer_product
        c = verify_causality()
        self.assertTrue(c["passed"])
        self.assertEqual(c["past_bits_changed"], 0)
        o = verify_outer_product(R=6, ds=8)
        self.assertTrue(o["passed"])
        self.assertLess(o["max_abs_diff"], 1e-5)

    def test_O1_state_and_role_bit(self):
        from run_full_migration import (verify_state_independent_of_L,
                                       verify_role_bit_degeneracy)
        s = verify_state_independent_of_L(0.9, 6, 256)
        self.assertTrue(s["passed"])
        self.assertTrue(s["no_L_dimension"])
        rb = verify_role_bit_degeneracy(6)
        self.assertTrue(rb["passed"])
        self.assertEqual(rb["spread_without_role"], 0.0)
        self.assertAlmostEqual(rb["B_bits_with"], 2.585, places=3)

    def test_stdlib_secrets_not_shadowed(self):
        # v8.2 真 bug: 本地 secrets.py 遮蔽标准库，numpy.random 无法 import
        import secrets as std
        self.assertTrue(hasattr(std, "randbits"))
        self.assertNotIn("ts2sc_v8", std.__file__)
        import numpy as np
        self.assertEqual(len(np.random.default_rng(1).standard_normal(3)), 3)

    def test_f11_hook_is_bounded(self):
        """v8.4: prove the CONTRACT, not a clip. 1e20 branch output stays bounded."""
        import numpy as np
        from run_full_migration import SGCFullBranch
        b = SGCFullBranch.__new__(SGCFullBranch)
        b.enabled = True; b.beta = 1.0; b.eps = 0.10
        b.state_step = lambda l, x, pos: (np.full_like(x, 1e20, np.float32), {})
        x = np.ones(16, np.float32)
        add = b.hook(0, x, 0)
        self.assertTrue(np.isfinite(add).all())
        nx = float(np.linalg.norm(x)); nd = float(np.linalg.norm(add))
        self.assertLessEqual(nd, 0.10 * nx + 1e-4)   # ‖Δ‖ ≤ βε‖x‖
        # beta=0 must be exactly zero (I1)
        b.beta = 0.0
        self.assertTrue(np.array_equal(b.hook(0, x, 0), np.zeros_like(x)))

    def test_scale_contract_extreme_magnitudes(self):
        from scale_theory import contract_bound, selftest
        self.assertTrue(selftest()["passed"])
        import numpy as np
        rng = np.random.default_rng(3)
        x = rng.standard_normal(512).astype(np.float32)
        for mag in (1e-5, 1.0, 1e4, 1e5):
            o = (rng.standard_normal(512) * mag).astype(np.float32)
            r = contract_bound(x, o, 1.0, 0.10)
            self.assertTrue(r["passed"])
            self.assertLessEqual(r["ratio"], 0.100001)

    def test_role_snr_at_real_magnitude(self):
        import numpy as np
        from scale_theory import role_resolvability
        rng = np.random.default_rng(5)
        k = (rng.standard_normal((6, 256)) * 100).astype(np.float32)
        r = role_resolvability(k, 6, 0.5)
        self.assertTrue(r["passed"])
        self.assertEqual(r["distance_B0"], 0.0)
        self.assertLess(r["distance_cv"], 1e-3)      # (i) 尺度不变
        self.assertLess(r["ratio_cv"], 1e-3)         # (ii) 能量比恒定
        self.assertGreater(r["naive_ratio_decay"], 1e3)  # naive 不可控

    def test_per_layer_beta(self):
        """v8.5: β_l 逐层向量，标量行为不变，β_0=0 逐位相同，契约仍成立。"""
        import numpy as np
        from run_full_migration import SGCFullBranch
        b = SGCFullBranch.__new__(SGCFullBranch)
        b.enabled = True; b.eps = 0.10
        b.state_step = lambda l, x, pos: (np.ones_like(x, np.float32), {})
        x = np.ones(16, np.float32)
        # scalar: 所有层一致
        b.beta = 1.0
        self.assertTrue(np.array_equal(b.hook(0, x, 0), b.hook(5, x, 0)))
        # per-layer: beta[0]=0 ⇒ 第 0 层逐位零；深层非零且有界
        b.beta = np.linspace(0.0, 1.0, 24)
        d0, d23 = b.hook(0, x, 0), b.hook(23, x, 0)
        self.assertTrue(np.array_equal(d0, np.zeros_like(x)))
        self.assertGreater(float(np.linalg.norm(d23)),
                           float(np.linalg.norm(d0)))
        self.assertLessEqual(float(np.linalg.norm(d23)),
                             0.10 * float(np.linalg.norm(x)) + 1e-4)
        # disabled 仍逐位零
        b.enabled = False
        self.assertTrue(np.array_equal(b.hook(23, x, 0), np.zeros_like(x)))

    def test_decode_gate_and_nan_rootcause(self):
        from scale_theory import decode_gate, nan_to_sentinel_demo
        d = nan_to_sentinel_demo()
        self.assertEqual(d["best_i"], -1)
        self.assertFalse(d["legal"])
        self.assertFalse(decode_gate([-1] * 24, 130560)["passed"])
        self.assertFalse(decode_gate([0, 5], 130560, logits_finite=False)["passed"])
        self.assertTrue(decode_gate([0, 5, 999], 130560)["passed"])


def write_report(result, cases=None):
    cases = cases or []
    rep = dict(
        suite="TS2SC v8.1 regression",
        tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped),
        passed=result.wasSuccessful(),
        cases=cases,
        honest_bound="No torch, no Qwen4-27B weights/config/tokenizer, no teacher forward, "
                     "no perplexity or capability numbers were used.",
    )
    json.dump(rep, open(os.path.join(HERE, "test_report.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    return rep


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase
    all_suite = unittest.TestSuite([suite(c) for c in
                                    (TestSpec, TestContract, TestProvenance,
                                     TestFusion, TestAcceptance, TestMigrationAndCLI,
                                     TestFullFeatures)])
    case_names = collect_cases(all_suite)
    res = unittest.TextTestRunner(verbosity=1).run(all_suite)
    rep = write_report(res, case_names)
    print("\n[report]", json.dumps(rep, ensure_ascii=False))
    sys.exit(0 if res.wasSuccessful() else 1)
