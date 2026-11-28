#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 独立回归：上游 SGC 外挂同步后的全部不变量（纯 numpy，秒级）。"""
from __future__ import annotations

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


if __name__ == "__main__":
    r = unittest.main(verbosity=2, exit=False).result
    print("[report]", dict(tests=r.testsRun, failures=len(r.failures),
                           errors=len(r.errors), passed=r.wasSuccessful()))
    raise SystemExit(0 if r.wasSuccessful() else 1)
