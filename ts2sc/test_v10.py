#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 独立回归：上游 SGC 外挂同步后的全部不变量（纯 numpy，秒级）。"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgc_v10 as A  # noqa: E402


class TestV10Arch(unittest.TestCase):
    def test_01_fft_iir_duality(self):
        b = A.make_basis(64)
        z = np.random.default_rng(3).normal(0, 1, (24, 64)).astype(np.complex128)
        d = np.max(np.abs(A.fft_dual(z, b) - A.iir_stream(z, b)))
        self.assertLess(d, 1e-10)

    def test_02_kernel_matches_recurrence(self):
        b = A.make_basis(64)
        L = 12
        imp = np.zeros((L, 64), np.complex128)
        imp[0] = 1.0
        k = A.kernel(b, L)
        self.assertLess(np.max(np.abs(A.iir_stream(imp, b).real - k)), 1e-12)

    def test_03_se1_orthogonal_zero_crosstalk(self):
        r = A.se1_identity_check()
        self.assertLess(r["on_orth"], 1e-12)
        self.assertLess(r["off_orth"], 1e-12)

    def test_04_se1_decay_closed_form(self):
        r = A.se1_identity_check()
        self.assertLess(r["on_decay_err"], 1e-12)
        self.assertLess(r["off_decay"], 1e-12)

    def test_05_shortconv_stream_equals_full(self):
        sc = A.ShortConvSidecar(64, seed=5)
        X = np.random.default_rng(6).normal(0, 1, (9, 64))
        full = sc.full(X)
        sc2 = A.ShortConvSidecar(64, seed=5)
        stream = np.stack([sc2.step(x) for x in X])
        self.assertLess(np.max(np.abs(full - stream)), 1e-10)

    def test_06_cfc_closed_form_idle_equals_steps(self):
        a = A.LiquidSidecar(64, seed=8)
        b = A.LiquidSidecar(64, seed=8)
        x = np.random.default_rng(9).normal(0, 1, 64)
        for _ in range(7):
            a.step(x)
        b.advance_idle(7, x)
        self.assertLess(np.max(np.abs(a.feature() - b.feature())), 1e-12)

    def test_07_core_unchanged_by_sidecars(self):
        r = A.core_unchanged_by_sidecars()
        self.assertTrue(r["passed"])
        self.assertEqual(r["max_abs_delta"], 0.0)

    def test_08_purity_ast_catches_forbidden_defs(self):
        src = open(os.path.join(os.path.dirname(__file__), "sgc_v10.py"),
                   encoding="utf-8").read()
        self.assertTrue(A.purity_source_audit(src)["passed"])
        bad = src + "\nclass SDPAAttention:\n    pass\n"
        self.assertFalse(A.purity_source_audit(bad)["passed"])

    def test_09_beta0_is_exactly_zero(self):
        lay = A.SGCV10Layer(1536, 7, A.deterministic_layer_params(1536, 7))
        lay.beta = 0.0
        x = np.random.default_rng(11).normal(0, 1, 1536).astype(np.float32)
        self.assertTrue(np.array_equal(lay.hook(x, 0), np.zeros_like(x)))

    def test_10_scale_contract_bound(self):
        lay = A.SGCV10Layer(1536, 7, A.deterministic_layer_params(1536, 7))
        lay.beta = 1.0
        for mag in (1e-5, 1.0, 1e5):
            x = (np.random.default_rng(13).normal(0, 1, 1536)
                 * mag).astype(np.float32)
            d = lay.hook(x, 0)
            self.assertLessEqual(np.linalg.norm(d),
                                 np.linalg.norm(x) * A.EPS_DEFAULT * (1 + 1e-6))
            self.assertTrue(np.all(np.isfinite(d)))

    def test_11_o1_state_independent_of_L(self):
        lay = A.SGCV10Layer(1536, 7, A.deterministic_layer_params(1536, 7))
        lay.beta = 1.0
        x = np.random.default_rng(11).normal(0, 1, 1536).astype(np.float32)
        sizes = []
        for t in range(1, 65):
            lay.hook(x, t)
            if t in (1, 8, 64):
                sizes.append(lay.state_bytes())
        self.assertEqual(len(set(sizes)), 1)
        self.assertEqual(sizes[0], 46656)

    def test_12_branch_all_layers_and_beta_gate(self):
        cfg = dict(hidden_size=1536, num_hidden_layers=24)
        p = {i: A.deterministic_layer_params(1536, 20261128 + 101 * i)
             for i in range(24)}
        br = A.SGCV10Branch(cfg, p)
        self.assertEqual(len(br.layers), 24)
        self.assertEqual(br.read_keys(), 24 * 13)
        br.set_beta(0.0)
        x = np.zeros(1536, np.float32)
        self.assertTrue(np.array_equal(br.hook(0, x, 0), x * 0))
        br.set_beta(1.0)
        self.assertEqual(br.layers[23].beta, 1.0)
        self.assertEqual(br.state_bytes(), 24 * 46656)

    def test_13_deterministic_params_reproducible(self):
        a = A.deterministic_layer_params(1536, 4242)
        b = A.deterministic_layer_params(1536, 4242)
        for k in a:
            self.assertTrue(np.array_equal(a[k], b[k]), k)

    def test_14_no_identity_adapter_symbol_in_v10_module(self):
        src = open(os.path.join(os.path.dirname(__file__), "sgc_v10.py"),
                   encoding="utf-8").read()
        for bad in ("logit_bias", "identity_alignment", "lead_tokens",
                    "anchor_margin", "qingnai", "青奈"):
            self.assertNotIn(bad, src, bad)

    def test_15_frozen_core_basis_snapshot(self):
        b = A.make_basis(64)
        self.assertEqual(b["rho"].shape, (64,))
        self.assertTrue(np.all(b["rho"] > 0))
        self.assertTrue(np.all(b["rho"] < 1))
        self.assertTrue(np.all(np.abs(b["b"] + b["rho"] ** 2) < 1e-15))
        self.assertTrue(np.all(np.isfinite(b["a"])))

    # ---- v10.1 读取契约：钉死「312 个 NEW 张量不得空挂」 ----
    def test_16_every_new_key_affects_forward_or_is_persisted(self):
        """扰动判据：每个 NEW key 必须 ACTIVE 或 PERSISTED_INIT，零空挂。

        v10.0 实测 10/13 ACTIVE、72/312 空挂（role.Wc / gate.alpha /
        gate.beta）；v10.1 修正后 12 ACTIVE + 1 PERSISTED_INIT。
        """
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import v10_read_contract_audit as RC
        p0 = A.deterministic_layer_params(1536, RC.SEED)
        x = np.random.default_rng(4242).normal(0, 1, 1536).astype(np.float32)
        d0 = RC.delta_of(p0, x)
        bad = []
        for key in [k for k, _, _ in A.TENSOR_SPECS]:
            if key == "gate.beta":
                continue                     # 单独用持久化判据
            q = RC.perturb(p0, key, np.random.default_rng(99))
            diff = float(np.max(np.abs(
                RC.delta_of(q, x).astype(np.float64)
                - d0.astype(np.float64))))
            if diff <= 0.0:
                bad.append(key)
        self.assertEqual(bad, [], f"empty-hanging NEW keys: {bad}")
        # gate.beta 必须在构造期被读取
        pb = {k: np.array(v, np.float32, copy=True) for k, v in p0.items()}
        pb["gate.beta"] = np.array([0.77], np.float32)
        lay = A.SGCV10Layer(1536, 1, pb)
        self.assertAlmostEqual(lay.beta, 0.77, places=6)

    def test_17_read_contract_does_not_lie(self):
        """read_keys 是声明；verified_read_keys 必须把所有 key 归类。"""
        cfg = dict(hidden_size=1536, num_hidden_layers=24)
        p = {i: A.deterministic_layer_params(1536, 7) for i in range(24)}
        br = A.SGCV10Branch(cfg, p)
        v = br.verified_read_keys()
        all_keys = (set(v["forward_active"])
                    | set(v["forward_neutral_init"])
                    | set(v["persisted_init"]))
        self.assertEqual(all_keys, set(A.SGCV10Branch.NEW_KEYS))
        self.assertEqual(len(all_keys), 13)
        self.assertEqual(br.read_keys(), 24 * 13)   # 声明数，不是证据

    def test_18_v10_1_fix_is_bitwise_neutral(self):
        """v10.1 读取契约修正不得改变任何 Δ（否则旧生成证据作废）。"""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import v10_read_contract_audit as RC
        base = json.load(open("/data/workspace/output/v10_0_delta_baseline.json",
                             encoding="utf-8"))
        rng = np.random.default_rng(31337)
        bad = []
        for li in (0, 7, 23):
            p = A.deterministic_layer_params(1536, 20261128 + 101 * li)
            for j, mag in enumerate((1e-3, 1.0, 1e3)):
                x = (rng.normal(0, 1, 1536) * mag).astype(np.float32)
                lay = A.SGCV10Layer(1536, 1000 + 17 * li, p)
                lay.beta = RC.BETA
                for pos in (0, 1, 5):
                    lay.reset()
                    sha = RC.fingerprint(
                        np.asarray(lay.hook(x, pos), np.float32))
                    want = next(r["sha256"] for r in base["rows"]
                                if r["layer"] == li and r["case"] == j
                                and r["pos"] == pos)
                    if sha != want:
                        bad.append((li, j, pos))
        self.assertEqual(bad, [])

    # ---- v10.2 开关法则：开门禁继承，不开就原模型能力 ----
    def test_19_all_off_is_structural_bitwise_zero(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import v10_switch_law_audit as SL
        p = SL.fresh_params(False)
        self.assertTrue(SL.phase_a(p)["passed"])
        # 全关后强行把 master 拉到 1.0 也不得复活
        l = SL.make(p, {k: 0.0 for k in A.CAP_KEYS})
        l.beta = l._scaled_scalar("gate.beta", 1.0)
        d = SL.delta(l, np.ones(1536, np.float32))
        self.assertTrue(np.all(d == 0.0))

    def test_20_each_of_13_gates_is_independently_switchable(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import v10_switch_law_audit as SL
        c = SL.phase_b(SL.fresh_params(True), True)
        self.assertEqual(c["n_observable"], 13)
        self.assertTrue(c["all_finite"])
        b = SL.phase_b(SL.fresh_params(False), False)
        # 默认初值下 role.Wc 全零，单关不可观测：如实登记，不假装 13/13
        self.assertEqual([r["gate"] for r in b["rows"]
                          if not r["observable"]], ["role.Wc"])

    def test_21_all_on_bitwise_and_contract_flags_are_inert(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import v10_switch_law_audit as SL
        self.assertTrue(SL.phase_d()["all_identical"])
        self.assertTrue(SL.phase_e(SL.fresh_params(False))["passed"])
        fgh = SL.phase_fgh()
        self.assertTrue(fgh["F"]["passed"])   # 契约门不改前向
        self.assertTrue(fgh["G"]["passed"])   # DEFERRED 拒绝伪造
        self.assertTrue(fgh["H"]["passed"])   # 24 层 × 13 门广播
        reg = json.load(open("/data/workspace/output/capability_registry.json",
                             encoding="utf-8"))
        self.assertEqual(reg["counts"]["total_slots_with_switch"],
                         reg["counts"]["total"])


if __name__ == "__main__":
    r = unittest.main(verbosity=2, exit=False).result
    print("[report]", dict(tests=r.testsRun, failures=len(r.failures),
                           errors=len(r.errors), passed=r.wasSuccessful()))
    raise SystemExit(0 if r.wasSuccessful() else 1)
