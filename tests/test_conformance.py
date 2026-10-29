#!/usr/bin/env python3
"""B4.6 conformance criteria: the classifier, the invariants and the thresholds.

The tests are built from the definitions rather than from the fixture data:
synthetic surfaces whose misclassification count is known, masks whose overlap
is known, and covariance values built from the E4M3 table itself.
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import m_conformance as mc       # noqa: E402


class Format(unittest.TestCase):
    def test_the_e4m3_table(self):
        self.assertEqual(mc.e4m3_value(0x00), 0.0)
        self.assertEqual(mc.e4m3_value(0x38), 1.0)         # e=7, m=0
        self.assertEqual(mc.e4m3_value(0x40), 2.0)         # e=8, m=0
        self.assertEqual(mc.e4m3_value(0x7E), 448.0)       # the largest finite
        self.assertEqual(mc.e4m3_value(0xB8), -1.0)        # sign preserved
        self.assertAlmostEqual(mc.e4m3_value(0x01), 2.0 ** -9)   # subnormal
        self.assertTrue(np.isnan(mc.e4m3_value(0x7F)))     # 0x7F is NaN

    def test_the_activation_table_is_monotonic_and_finite(self):
        values, act = mc.activation_table()
        self.assertEqual(len(values), 254)                 # 256 minus the two NaNs
        # +0 and -0 are both present, so the values are non-decreasing
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertTrue(np.all(np.diff(act) >= 0))
        self.assertTrue(np.all(np.isfinite(act)))
        self.assertEqual(len(np.unique(values)), 253)

    def test_sigmoid_rounds_through_f16(self):
        # the sequence is f16(clamp(v)) -> exp2 -> f16(1/(1+e))
        self.assertEqual(np.float16(mc.sigmoid(0.0)), np.float16(0.5))
        self.assertEqual(np.float16(mc.sigmoid(-20.0)), np.float16(mc.sigmoid(-40.0)))


class Classification(unittest.TestCase):
    def test_identical_surfaces(self):
        a = np.arange(12, dtype=np.uint8).reshape(3, 4)
        result = mc.classify(a, a.copy(), 'codes')
        self.assertEqual(result['identical'], 1.0)
        self.assertEqual(result['differing'], 0)
        self.assertEqual(result['max_code_delta'], 0)

    def test_one_sample_off(self):
        ref = np.zeros((4, 4), np.uint8)
        native = ref.copy()
        native[1, 2] = 3
        result = mc.classify(native, ref, 'codes')
        self.assertEqual(result['differing'], 1)
        self.assertAlmostEqual(result['identical'], 15 / 16)
        self.assertEqual(result['max_code_delta'], 3)

    def test_f16_reports_ulps_and_relative_error(self):
        ref = np.array([1.0, 0.5, 0.25], np.float16)
        native = ref.copy()
        native[0] = np.nextafter(np.float16(1.0), np.float16(2.0))
        result = mc.classify(native, ref, 'f16')
        self.assertEqual(result['differing'], 1)
        self.assertEqual(result['max_ulp_delta'], 1)
        self.assertLess(result['max_relative'], 1e-3)

    def test_disjoint_detects_overlap(self):
        a = np.array([1, 0, 0, 0], bool)
        b = np.array([0, 1, 0, 0], bool)
        self.assertTrue(mc.disjoint({'a': a, 'b': b}))
        self.assertFalse(mc.disjoint({'a': a, 'b': a}))

    def test_the_covariance_code_distance(self):
        _, act = mc.activation_table(8.0)
        aa = (8.0 * act.astype(np.float64)) ** 2
        i = 100
        # the same code twice, then one code apart
        same = (np.array([[aa[i]]], np.float16), np.array([[aa[i]]], np.float16))
        self.assertEqual(mc.covariance_code_distance(*same), 0)
        near = (np.array([[aa[i + 1]]], np.float16), np.array([[aa[i]]], np.float16))
        self.assertEqual(mc.covariance_code_distance(*near), 1)
        far = (np.array([[aa[i + 3]]], np.float16), np.array([[aa[i]]], np.float16))
        self.assertEqual(mc.covariance_code_distance(*far), 3)


class Verdicts(unittest.TestCase):
    def test_an_exact_claim_needs_identity(self):
        self.assertEqual(mc.verdict('s', 'exact', {'identical': 1.0}), [])
        self.assertTrue(mc.verdict('s', 'exact', {'identical': 0.999999}))

    def test_a_surface_floor_and_its_magnitude(self):
        ok = {'identical': 0.9991, 'max_code_delta': 8}
        self.assertEqual(mc.verdict('s', 'dec0-unorm8', ok), [])
        bad_fraction = {'identical': 0.9989, 'max_code_delta': 8}
        self.assertTrue(mc.verdict('s', 'dec0-unorm8', bad_fraction))
        bad_magnitude = {'identical': 0.9991, 'max_code_delta': 9}
        self.assertTrue(mc.verdict('s', 'dec0-unorm8', bad_magnitude))

    def test_the_frame_criterion(self):
        ok = {'identical': 0.9995, 'max_ulp_delta': 7}
        self.assertEqual(mc.verdict('c', 'frame', ok), [])
        self.assertTrue(mc.verdict('c', 'frame',
                                   {'identical': 0.9995, 'max_ulp_delta': 9}))


class GateCompleteness(unittest.TestCase):
    """The gate must never report success for a comparison it did not make."""

    def setUp(self):
        import tempfile
        self.root = tempfile.mkdtemp(prefix='sette-gate-')
        self.fixture = os.path.join(self.root, 'downsample')
        os.makedirs(os.path.join(self.fixture, 'results'))
        self.bytes = 64
        with open(os.path.join(self.fixture, 'fixture.json'), 'w') as handle:
            json.dump({'bindings': {'out': {'output': True, 'bytes': self.bytes}}},
                      handle)
        with open(os.path.join(self.fixture, 'stage.txt'), 'w') as handle:
            handle.write('version 2\n')
        self.ref = bytes(range(self.bytes))
        with open(os.path.join(self.fixture, 'ref-out.bin'), 'wb') as handle:
            handle.write(self.ref)
        with open(os.path.join(self.fixture, 'results', 'out-out.bin'), 'wb') as handle:
            handle.write(self.ref)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def audit(self):
        return mc.audit_stage(self.root, 'downsample')

    def test_a_complete_comparison_passes(self):
        results, problems = self.audit()
        self.assertEqual(problems, [])
        self.assertEqual(results['out']['identical'], 1.0)

    def test_a_missing_result_is_a_failure(self):
        os.remove(os.path.join(self.fixture, 'results', 'out-out.bin'))
        results, problems = self.audit()
        self.assertTrue(any('missing result' in p for p in problems), problems)
        self.assertEqual(results, {})

    def test_a_missing_reference_is_a_failure(self):
        os.remove(os.path.join(self.fixture, 'ref-out.bin'))
        _, problems = self.audit()
        self.assertTrue(any('missing reference' in p for p in problems), problems)

    def test_a_truncated_result_is_a_failure(self):
        with open(os.path.join(self.fixture, 'results', 'out-out.bin'), 'wb') as handle:
            handle.write(self.ref[:32])
        _, problems = self.audit()
        self.assertTrue(any('the reference is' in p for p in problems), problems)

    def test_a_result_shorter_than_its_declaration_is_a_failure(self):
        # both files agree, but the fixture declared more bytes than were compared
        with open(os.path.join(self.fixture, 'ref-out.bin'), 'wb') as handle:
            handle.write(self.ref[:32])
        with open(os.path.join(self.fixture, 'results', 'out-out.bin'), 'wb') as handle:
            handle.write(self.ref[:32])
        _, problems = self.audit()
        self.assertTrue(any('the fixture declares 64 B' in p for p in problems), problems)

    def test_an_undeclared_output_is_a_failure(self):
        with open(os.path.join(self.fixture, 'fixture.json'), 'w') as handle:
            json.dump({'bindings': {}}, handle)
        _, problems = self.audit()
        self.assertTrue(any('does not declare' in p for p in problems), problems)

    def test_a_missing_fixture_json_is_a_failure(self):
        os.remove(os.path.join(self.fixture, 'fixture.json'))
        _, problems = self.audit()
        self.assertTrue(any('no fixture.json' in p for p in problems), problems)


class GateDeterminism(unittest.TestCase):
    """Two runs of one fixture must agree; a difference is a gate failure."""

    def setUp(self):
        import tempfile
        self.root = tempfile.mkdtemp(prefix='sette-det-')
        self.fixture = os.path.join(self.root, 'downsample')
        for name in ('results', 'results-repeat'):
            os.makedirs(os.path.join(self.fixture, name))
        with open(os.path.join(self.fixture, 'fixture.json'), 'w') as handle:
            json.dump({'bindings': {'out': {'output': True, 'bytes': 8}}}, handle)
        self.a = bytes([1, 2, 3, 4, 5, 6, 7, 8])
        self.write('results', self.a)
        self.write('results-repeat', self.a)

    def write(self, folder, data):
        with open(os.path.join(self.fixture, folder, 'out-out.bin'), 'wb') as handle:
            handle.write(data)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_identical_runs_pass(self):
        self.assertEqual(mc.audit_determinism(self.root, 'downsample'), [])

    def test_a_differing_run_fails(self):
        self.write('results-repeat', bytes([1, 2, 3, 4, 5, 6, 7, 9]))
        problems = mc.audit_determinism(self.root, 'downsample')
        self.assertTrue(any('two runs differ' in p for p in problems), problems)

    def test_a_missing_repeat_is_a_failure(self):
        os.remove(os.path.join(self.fixture, 'results-repeat', 'out-out.bin'))
        problems = mc.audit_determinism(self.root, 'downsample')
        self.assertTrue(any('determinism check' in p for p in problems), problems)


if __name__ == '__main__':
    sys.exit(unittest.main())
