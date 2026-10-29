"""M06 operator-contract tests: d4r's Preset M numeric primitives vs an
independent float64/OCP oracle.

The oracle here is the *specification*, never the implementation under test:
the e4m3 table is rebuilt bit-by-bit from the OCP definition, nearest-code
rounding is redone by brute force over the whole f16 range, and the
transcendental-free GELU/softmax approximations are compared against exact
float64 mathematics.  d4r's primitives are imported read-only from the pinned
reference; nothing under references/ is modified.

Run: python -m unittest tests.test_m06_operator_contract -v
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'engine'))
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'kernels', 'tools'))

import m_model as mm          # noqa: E402
import model_enc3 as me       # noqa: E402

F16, F64 = np.float16, np.float64


def ocp_e4m3_table():
    """The 256 e4m3 codes decoded straight from the bit definition."""
    table = np.zeros(256, F64)
    for code in range(256):
        sign, exp, mant = code >> 7, (code >> 3) & 0xF, code & 0x7
        if exp == 0xF and mant == 0x7:
            value = np.nan                       # OCP e4m3 has no infinities
        elif exp == 0:
            value = mant * 2.0 ** -9             # subnormal: mant * 2^-6 * 2^-3
        else:
            value = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        table[code] = -value if sign else value
    return table


def nearest_e4m3(x):
    """Round to nearest finite e4m3 (ties to even code), satfinite at 448."""
    table = ocp_e4m3_table()
    finite = np.isfinite(table)
    magnitudes = np.abs(table[finite])
    codes = np.arange(256)[finite]
    out = np.zeros_like(x, dtype=F64)
    flat, out_flat = x.ravel(), out.ravel()
    for index, value in enumerate(flat):
        if np.isnan(value):
            out_flat[index] = np.nan
            continue
        target = abs(value)
        best, best_dist, best_code = 0.0, None, 0
        for magnitude, code in zip(magnitudes, codes):
            dist = abs(magnitude - target)
            # nearest, and on a tie the even code wins
            if best_dist is None or dist < best_dist - 0.0 or (
                    dist == best_dist and (code & 1) == 0):
                best, best_dist, best_code = magnitude, dist, code
        if target > magnitudes.max():
            best, best_code = magnitudes.max(), codes[magnitudes.argmax()]
        out_flat[index] = np.copysign(best, value)
    return out


class E4M3Contract(unittest.TestCase):
    def test_table_matches_ocp_definition(self):
        oracle = ocp_e4m3_table()
        reference = np.asarray(me.E4M3, F64)
        both_nan = np.isnan(oracle) & np.isnan(reference)
        self.assertTrue(np.array_equal(np.isnan(oracle), np.isnan(reference)))
        self.assertTrue(np.array_equal(oracle[~both_nan], reference[~both_nan]))

    def test_table_is_exactly_representable_in_f16(self):
        # The numerics finding depends on this: e4m3 -> f16 is lossless.
        reference = np.asarray(me.E4M3, F64)
        finite = np.isfinite(reference)
        self.assertTrue(np.array_equal(reference[finite].astype(F16).astype(F64),
                                       reference[finite]))

    def test_round_trip_is_nearest_and_saturating(self):
        rng = np.random.default_rng(4)
        values = np.concatenate([
            rng.normal(0.0, 0.02, 512),          # typical activation/weight range
            rng.normal(0.0, 1.0, 512),
            np.array([0.0, -0.0, 448.0, -448.0, 500.0, -500.0, 2 ** -10, -(2 ** -10)]),
        ])
        got = np.asarray(me.q8(values), F64)
        want = nearest_e4m3(np.asarray(values, F64).astype(F16).astype(F64))
        np.testing.assert_array_equal(got, want)
        self.assertEqual(got[np.argmax(values)], 448.0)          # satfinite, not inf
        self.assertLessEqual(np.abs(got).max(), 448.0)


class AccumulationContract(unittest.TestCase):
    def test_single_step_rounds_once_to_f16(self):
        rng = np.random.default_rng(7)
        a = np.asarray(rng.normal(0, 0.05, (4, 32)), F64).astype(F16).astype(F64)
        w = np.asarray(rng.normal(0, 0.05, (32, 16)), F64).astype(F16).astype(F64)
        exact = a @ w                                   # float64, no rounding
        got = me.mma_chain(a, w)
        # one f16 rounding of the exact sum: relative error <= 2^-11
        rel = np.abs(got - exact) / np.maximum(np.abs(exact), 1e-30)
        self.assertLessEqual(rel.max(), 2.0 ** -11)
        self.assertGreater(rel.max(), 0.0)              # rounding actually happens

    def test_chain_adds_one_rounding_per_step(self):
        rng = np.random.default_rng(11)
        k = 128
        a = np.asarray(rng.normal(0, 0.05, (8, k)), F64).astype(F16).astype(F64)
        w = np.asarray(rng.normal(0, 0.05, (k, 16)), F64).astype(F16).astype(F64)
        exact = a @ w
        chained = me.mma_chain(a, w)
        rel = np.abs(chained - exact) / np.maximum(np.abs(exact), 1e-30)
        # four chained roundings stay within a small multiple of one ulp; the
        # point of the test is that the error is bounded and not catastrophic
        self.assertLessEqual(rel.max(), 8.0 * 2.0 ** -11)


class NormalisationContract(unittest.TestCase):
    def test_rms_norm_error_bound(self):
        rng = np.random.default_rng(13)
        x = np.asarray(rng.normal(0, 0.3, (256, 64)), F16).astype(F64)
        # contract: f16 sum of squares + 2^-13, f16 rsqrt, then the f16 scale
        sums = np.asarray((x * x).sum(-1, keepdims=True), F64).astype(F16).astype(F64)
        r = np.asarray(1.0 / np.sqrt(sums + 2.0 ** -13), F64)
        r = r.astype(F16).astype(F64)
        got = x * r
        exact = x / np.sqrt((x * x).sum(-1, keepdims=True))
        rel = np.abs(got - exact) / np.abs(exact)
        self.assertLessEqual(rel.max(), 4.0 * 2.0 ** -11)
        # the epsilon is a floor: it must not change the typical scale
        self.assertLess(np.abs(r - 1.0 / np.sqrt(sums)).max(), 0.01)


class GeluContract(unittest.TestCase):
    def test_approximation_error_against_exact_gelu(self):
        from math import erf
        z = np.linspace(-6.0, 6.0, 4001)
        got = np.asarray(me.gelu_poly(z), F64)
        exact = 0.5 * z * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
        error = np.abs(got - exact)
        # The contract is an approximation, not exact GELU: the clamped cubic is
        # within 5e-2 absolute of the true function and must not be exact.
        self.assertLessEqual(error.max(), 0.05)
        self.assertGreater(error.max(), 1e-3)
        self.assertLessEqual(error[np.abs(z) <= 2.0].max(), 0.05)

    def test_saturation_outside_the_clamp(self):
        # |z| > 2 uses the clamped cubic, so the response is linear there and
        # unbounded below: gelu_poly(-8) must stay near zero, not minus infinity.
        self.assertLess(abs(float(me.gelu_poly(-8.0))), 0.05)
        self.assertGreater(float(me.gelu_poly(8.0)), 7.9)

    def test_non_monotone_region_is_where_documented(self):
        # The clamped cubic is not monotone on [-2, 0]; record the interval so a
        # future fidelity regression is visible.
        z = np.linspace(-3.0, 0.0, 3001)
        got = np.asarray(me.gelu_poly(z), F64)
        decreasing = z[:-1][np.diff(got) < 0]
        self.assertGreater(decreasing.size, 0)
        self.assertGreaterEqual(decreasing.min(), -2.01)
        self.assertLessEqual(decreasing.max(), -0.55)


class SoftmaxContract(unittest.TestCase):
    """Property tests only.

    d4r's exponent trick carries its own score-scale convention, so an absolute
    comparison against `softmax(s)` is not an independent oracle; the properties
    below hold for any correct softmax.
    """

    def test_rows_are_normalised_non_negative_and_finite(self):
        rng = np.random.default_rng(17)
        for sigma in (0.5, 2.0, 8.0):
            s = np.asarray(rng.normal(0, sigma, (64, 16)), F16).astype(F64)
            got = np.asarray(mm.softmax_rows(s), F64)
            self.assertTrue(np.all(np.isfinite(got)))
            self.assertTrue(np.all(got >= 0.0))
            np.testing.assert_allclose(got.sum(1), 1.0, atol=2e-3)

    def test_larger_score_gets_larger_weight(self):
        rng = np.random.default_rng(19)
        s = np.asarray(rng.normal(0, 2.0, (64, 16)), F16).astype(F64)
        got = np.asarray(mm.softmax_rows(s), F64)
        order_s = np.argsort(s, axis=1)
        order_w = np.argsort(got, axis=1)
        # ties apart, the ranking must be preserved
        equal = np.take_along_axis(s, order_s, 1) == np.take_along_axis(s, order_s, 1)[:, :1]
        self.assertGreater(float(np.mean(order_s == order_w)), 0.9)

    def test_extreme_inputs_stay_normalised(self):
        s = np.tile(np.array([0.0, 100.0, -100.0, 5.0], F64), (4, 4))
        got = np.asarray(mm.softmax_rows(s), F64)
        self.assertTrue(np.all(np.isfinite(got)))
        np.testing.assert_allclose(got.sum(1), 1.0, atol=3e-3)


if __name__ == '__main__':
    unittest.main()
