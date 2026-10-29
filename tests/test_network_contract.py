#!/usr/bin/env python3
"""The native block's numerical contract, without a capture.

These tests pin the pieces the HLSL and `m_layer_oracle.NaturalBlock` share:
the pair-order permutation, the two RMS trees, the FP8 round trip, the
softmax's bit-level exponent and butterfly, the GELU approximation, and the
natural-order form's equality with the kernel-order form on synthetic weights.

The capture-derived equivalence (NaturalBlock == KernelBlock on all ten real
layers) is recorded in EXP-019; it needs the local capture and is therefore not
a unit test.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'engine'))
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'kernels', 'tools'))

import m_layer_oracle as lo      # noqa: E402
import model_enc3 as me          # noqa: E402
import swin_model as sm          # noqa: E402

F16 = np.float16


class PairOrder(unittest.TestCase):
    def test_the_two_permutations_are_inverse(self):
        c = np.arange(128)
        self.assertTrue(np.array_equal(lo.gp(lo.ginv(c)), c))
        self.assertTrue(np.array_equal(lo.ginv(lo.gp(c)), c))

    def test_the_permutation_stays_inside_a_32_channel_group(self):
        c = np.arange(128)
        self.assertTrue(np.array_equal(lo.gp(c) // 32, c // 32))


class RoundTrip(unittest.TestCase):
    def test_e4m3_round_trip_is_exact_on_every_finite_value(self):
        codes = np.arange(127, dtype=np.uint8)
        values = me.E4M3[codes]
        back = np.array([me.q8(v) for v in values])
        self.assertTrue(np.array_equal(back, values))

    def test_q8_saturates_and_keeps_zero(self):
        self.assertEqual(float(me.q8(1000.0)), 448.0)
        self.assertEqual(float(me.q8(-1000.0)), -448.0)
        self.assertEqual(float(me.q8(0.0)), 0.0)
        self.assertEqual(float(me.q8(2.0 ** -12)), 0.0)


class RmsTrees(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(3)
        self.x = np.asarray(rng.randn(64, 128) * 2.0, F16).astype(np.float64)

    def test_sumsq_natural_matches_the_kernel_tree(self):
        self.assertTrue(np.array_equal(lo.sumsq_natural(self.x), sm.sumsq_tokens(self.x)))

    def test_tree_norm2_natural_matches_the_pair_order_tree(self):
        # the natural row's pair-order form is natural[ginv(n)]
        pair = self.x[:, lo.ginv(np.arange(128))]
        self.assertTrue(np.array_equal(lo.tree_norm2_natural(self.x),
                                       sm.tree_norm2(pair)))


class Softmax(unittest.TestCase):
    def test_exp_trick_is_monotone_in_the_polynomial(self):
        # the polynomial stays in the positive normal range, where the bit
        # pattern order and the value order agree
        poly = np.zeros((1, 16), F16)
        poly[0] = np.linspace(1.0, 1.6, 16)
        w = me.exp_trick(poly).astype(np.float64)
        self.assertTrue(np.all(np.diff(w[0]) > 0))

    def test_the_row_sum_is_identical_in_every_column(self):
        rng = np.random.RandomState(4)
        w = np.asarray(rng.rand(8, 16), F16)
        s = me.row_sum16(w)
        for c in range(16):
            self.assertTrue(np.array_equal(s[:, c], s[:, 0]))

    def test_the_softmax_is_normalised(self):
        rng = np.random.RandomState(5)
        S = np.asarray(rng.randn(16, 16) * 3.0, F16).astype(np.float64)
        P = lo.softmax_rows(S)
        row = P[0].astype(np.float64)
        self.assertAlmostEqual(float(row.sum()), 1.0, places=2)
        self.assertTrue(np.all(row >= 0))


class Gelu(unittest.TestCase):
    def test_the_approximation_is_the_captured_cubic_not_erf(self):
        z = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0], np.float64)
        got = np.array([float(me.gelu_poly(v)) for v in z])
        # the clamped cubic is linear-extrapolated outside [-2,2]
        self.assertAlmostEqual(got[0], -3.0 * (0.5 + -2.0 * (0.412109375 - 0.0810546875 * 2.0)), places=6)
        self.assertAlmostEqual(got[2], 0.0, places=6)
        # and differs from exact erf-GELU where the approximation is known to
        from math import erf, sqrt
        exact = [0.5 * v * (1 + erf(v / sqrt(2))) for v in z]
        self.assertGreater(abs(got[1] - exact[1]), 0.01)


class AccumulationContract(unittest.TestCase):
    def test_one_f16_rounding_per_32_products(self):
        rng = np.random.RandomState(6)
        A = np.asarray(rng.randn(4, 64), F16).astype(np.float64)
        W = np.asarray(rng.randn(64, 8), F16).astype(np.float64)
        got = me.mma_chain(A, W)
        # an implementation that rounds once per 16 products is measurably
        # different (EXP-019 measures the whole-layer effect)
        reference = np.zeros((4, 8))
        for k0 in range(0, 64, 32):
            reference = np.asarray(me.f16(A[:, k0:k0 + 32] @ W[k0:k0 + 32]
                                          + reference), F16).astype(np.float64)
        self.assertTrue(np.array_equal(got, reference))
        per16 = np.zeros((4, 8))
        for k0 in range(0, 64, 16):
            per16 = np.asarray(me.f16(A[:, k0:k0 + 16] @ W[k0:k0 + 16] + per16),
                              F16).astype(np.float64)
        self.assertFalse(np.array_equal(got, per16))


if __name__ == '__main__':
    unittest.main()
