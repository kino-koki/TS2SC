#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10.3 回归：范围收窄 / SMM 门禁 / 10M / T1' 总契约。"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgc_v10 as A
import scoped_v10_3 as S
from smm_v10_3 import SMMModule

OUT = "/data/workspace/output"
CFG = json.load(open(os.path.join(OUT, "v10_fresh/config.json"),
                     encoding="utf-8"))
D = 1536
NL = 24


def _pbl():
    return {i: A.deterministic_layer_params(D, 20261128 + 101 * i)
            for i in range(NL)}


def _consolidate(sc, steps=120):
    rng = np.random.default_rng(1234)
    for m in sc.smm:
        for _ in range(steps):
            t = rng.integers(0, 16, 128)
            m.observe(t, (t + 5) % 16)


class TestScope(unittest.TestCase):
    def test_01_registry_has_exactly_three_groups(self):
        r = S.scoped_registry()
        self.assertEqual([g["id"] for g in r["groups"]],
                         ["G1_SGC_CORE", "G2_SIDECARS", "G3_SMM"])
        self.assertEqual(r["context_tokens"], 10 * 1024 * 1024)
        self.assertIn("evolution_mixer.py", r["groups"][2]["source"])
        self.assertIn("EXCLUDED", r["out_of_scope"])

    def test_02_all_off_structural_zero(self):
        sc = S.ScopedV10_3(CFG, _pbl())
        _consolidate(sc)
        sc.all_off()
        x = np.random.default_rng(1).normal(0, 1, D).astype(np.float32)
        for l in (0, 7, 23):
            for pos in (0, 3, 17):
                self.assertTrue(np.all(sc.hook(l, x, pos) == 0.0))

    def test_03_smm_off_is_bitwise_v10_2_full_on(self):
        sc = S.ScopedV10_3(CFG, _pbl())
        _consolidate(sc)
        sc.all_on()
        sc.smm_off()
        v102 = A.SGCV10Branch(CFG, _pbl())
        v102.set_beta(1.0)
        x = np.random.default_rng(2).normal(0, 1, D).astype(np.float32)
        for l in (0, 7, 23):
            self.assertTrue(np.array_equal(
                sc.hook(l, x, 3),
                np.asarray(v102.layers[l].hook(x, 3), np.float32)))


class TestSMMGate(unittest.TestCase):
    def test_04_zero_state_has_zero_delta(self):
        m = SMMModule(d=64, seed=42)
        self.assertEqual(m.commits, 0)
        x = np.ones(D, np.float32)
        self.assertTrue(np.all(m.delta(x, tok=3) == 0.0))

    def test_05_explicit_observe_really_consolidates(self):
        m = SMMModule(d=64, seed=42)
        rng = np.random.default_rng(7)
        for _ in range(120):
            t = rng.integers(0, 16, 128)
            m.observe(t, (t + 5) % 16)
        self.assertGreater(m.commits, 0)
        self.assertGreater(m.rejects, 0)          # 读回拒绝确实工作
        self.assertGreater(len(m.slots), 0)
        self.assertGreater(float(np.linalg.norm(m.S)), 0.0)
        x = np.ones(D, np.float32)
        self.assertGreater(float(np.linalg.norm(m.delta(x, tok=3))), 0.0)
        m.s = 0.0
        self.assertTrue(np.all(m.delta(x, tok=3) == 0.0))
        inv = m.invariants()
        self.assertTrue(inv["state_independent_of_L"])
        self.assertTrue(inv["consolidated_only_by_explicit_observe"])

    def test_06_read_back_matches_cooccurrence_rule(self):
        m = SMMModule(d=64, seed=42)
        rng = np.random.default_rng(7)
        for _ in range(120):
            t = rng.integers(0, 16, 128)
            m.observe(t, (t + 5) % 16)
        kv, margin = m._read(0)
        self.assertEqual(kv, 5)                    # tgt=(tok+5)%16
        self.assertGreater(margin, 0.0)

    def test_07_random_scatter_is_rejected_not_faked(self):
        """散共现被门禁全拒 ⇒ 必须如实登记，不许伪装成已巩固。"""
        m = SMMModule(d=64, seed=42)
        rng = np.random.default_rng(9)
        for _ in range(60):
            m.observe(rng.integers(0, 16, 64), rng.integers(0, 16, 64))
        self.assertEqual(m.commits, 0)
        self.assertEqual(float(np.linalg.norm(m.S)), 0.0)


class TestContractAnd10M(unittest.TestCase):
    def test_08_total_delta_respects_T1_bound(self):
        sc = S.ScopedV10_3(CFG, _pbl())
        _consolidate(sc)
        sc.all_on()
        rng = np.random.default_rng(3)
        for mag in (1e-3, 1.0, 1e3):
            x = (rng.normal(0, 1, D) * mag).astype(np.float32)
            bound = 1.0 * A.EPS_DEFAULT * float(np.linalg.norm(
                np.asarray(x, np.float64)))
            d = sc.hook(7, x, 3)
            self.assertTrue(np.all(np.isfinite(d)))
            self.assertLessEqual(float(np.linalg.norm(d)),
                                 bound * (1.0 + 1e-5))

    def test_09_context_10m_config_and_hbm_ledger(self):
        c = S.context_10m_config(CFG)
        self.assertEqual(c["max_position_embeddings"], 10485760)
        self.assertEqual(c["_ts2sc_context_base"], 131072)
        h = S.hbm_ledger(CFG)
        self.assertEqual(h["L"], 10485760)
        self.assertEqual(h["status"], "INFEASIBLE_LEDGER")
        self.assertAlmostEqual(h["kv_gib"], 240.0, places=1)
        self.assertEqual(h["rope_extrapolation_factor"], 80.0)

    def test_10_10m_idle_state_constant_and_closed_form(self):
        sc = S.ScopedV10_3(CFG, _pbl())
        r = sc.context_10m_run(CFG)
        self.assertEqual(r["L"], 10485760)
        self.assertTrue(r["state_independent_of_L"])
        self.assertEqual(r["logical_pos"], 10485760)
        self.assertTrue(r["idle_closed_form_equals_stepwise_gate0"])
        self.assertTrue(r["smm_observe"]["all_24_layers_consolidated"])
        self.assertEqual(r["attention"]["status"], "INFEASIBLE_LEDGER")
        self.assertIn("BRANCH_SIDE_VERIFIED_10M", r["verdict"])


if __name__ == "__main__":
    r = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromModule(
            sys.modules["__main__"]))
    print("[report]", dict(tests=r.testsRun,
                           failures=len(r.failures), errors=len(r.errors),
                           passed=r.wasSuccessful()))
    raise SystemExit(0 if r.wasSuccessful() else 1)
