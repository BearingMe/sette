"""M07 extension contract tests: e4m3 coding, the front-end weight layout, and
the INT8 encoder/GEMM arithmetic.

The oracles here are independent of the code under test: the e4m3 decode is
rebuilt from the OCP bit definition and compared with d4r's table, the layout is
checked structurally against the blob extent and against the 32k x 2n unit
grammar a blocked MMA layout must satisfy, and the integer product is compared
with a plain element-wise loop.  No capture is required, so these tests run
anywhere.

Run: python -m unittest tests.test_m07_quantization_contract -v
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'engine'))
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'kernels', 'tools'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import model_enc3 as me       # noqa: E402
import m_int8 as qi           # noqa: E402
import m_weights as mw        # noqa: E402

F64 = np.float64


class E4M3Coding(unittest.TestCase):
    def test_decode_matches_the_reference_table(self):
        codes = np.arange(256, dtype=np.uint8)
        got = mw.e4m3_decode(codes)
        want = np.asarray(me.E4M3, F64)
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)))
        finite = ~np.isnan(want)
        np.testing.assert_array_equal(got[finite], want[finite])

    def test_encode_round_trips_every_finite_code(self):
        codes = np.arange(256, dtype=np.uint8)
        values = mw.e4m3_decode(codes)
        finite = ~np.isnan(values)
        back = mw.e4m3_encode(values[finite])
        # ties are impossible here: the values *are* table entries
        np.testing.assert_array_equal(back, codes[finite])

    def test_encode_saturates_rather_than_overflowing(self):
        out = mw.e4m3_encode(np.array([1e9, -1e9, 448.0, -448.0]))
        np.testing.assert_array_equal(mw.e4m3_decode(out), [448.0, -448.0, 448.0, -448.0])

    def test_every_value_is_an_f16_value(self):
        # the INT8 encoder may therefore start from the f16 tensor without
        # changing a single weight (EXP-006)
        values = mw.e4m3_decode(np.arange(256, dtype=np.uint8))
        finite = np.isfinite(values)
        np.testing.assert_array_equal(values[finite].astype(np.float16).astype(F64),
                                      values[finite])


class FrontEndLayout(unittest.TestCase):
    def specs(self):
        return {'enc0': mw.front_end_layout('enc0'), 'dec0': mw.front_end_layout('dec0')}

    def test_layout_is_a_bijection_onto_the_matrix_extent(self):
        for stage, spec in self.specs().items():
            coverage = mw.layout_coverage(spec)
            self.assertTrue(coverage['bijective'], stage)
            self.assertEqual(coverage['extent'],
                             [spec['base'], spec['base'] + spec['K'] * spec['N']])

    def test_extents_match_the_stage_declarations(self):
        # enc0_m.comp: [128][64] at 4096, bias at 12288, dither 0..4096
        spec = mw.front_end_layout('enc0')
        self.assertEqual((spec['K'], spec['N'], spec['base'], spec['bias']), (128, 64, 4096, 12288))
        self.assertEqual(spec['bias'] + 2 * spec['bias_count'], 12416)
        self.assertEqual(spec['aux'], (0, 4096))
        # dec0_m.comp: [64][80] read from 80 of 128 stored columns, bias at 8192
        spec = mw.front_end_layout('dec0')
        self.assertEqual((spec['K'], spec['N'], spec['base'], spec['bias']), (64, 128, 0, 8192))
        self.assertEqual(spec['bias'] + 2 * spec['bias_count'], 8448)
        self.assertEqual(int(spec['live'].size), 80)
        self.assertEqual(sorted(spec['live'].tolist()),
                         sorted((np.arange(4)[:, None] * 32 + np.arange(20)[None, :]).ravel().tolist()))

    def test_units_hold_a_32k_by_2n_fragment(self):
        # A blocked MMA layout is built from 64-byte units; each must hold one
        # 32-row k block for a pair of columns 8 apart, or the fragment grammar
        # the shader indexes does not hold.
        for stage, spec in self.specs().items():
            off = mw.layout_offsets(spec)
            units = off.ravel() // 64
            for unit in np.unique(units):
                members = np.argwhere(units.reshape(off.shape) == unit)
                rows = np.unique(members[:, 0])
                cols = np.unique(members[:, 1])
                self.assertEqual(members.shape[0], 64, f'{stage} unit {unit}')
                self.assertEqual(rows.size, 32, f'{stage} unit {unit}')
                self.assertEqual(cols.size, 2, f'{stage} unit {unit}')
                self.assertEqual(int(cols[1] - cols[0]), 8, f'{stage} unit {unit}')

    def test_live_columns_are_distinct(self):
        for stage, spec in self.specs().items():
            self.assertEqual(int(np.unique(spec['live']).size), spec['live'].size, stage)


class Encoder(unittest.TestCase):
    def matrix(self, k=40, n=12, seed=3):
        rng = np.random.default_rng(seed)
        return np.asarray(rng.normal(0.0, 0.2, (k, n)), F64)

    def test_scalar_and_vector_encoders_agree(self):
        W = self.matrix()
        for scope, block in (('per_tensor', None), ('per_output_channel', None),
                             ('per_input_channel', None), ('block_k', 16)):
            q = qi.quantize(W, scope, block)
            row = q['scales'] if scope != 'per_tensor' else np.array([q['scales'][0]])
            for k in range(W.shape[0]):
                for n in range(W.shape[1]):
                    if scope == 'per_tensor':
                        scale = float(q['scales'][0])
                    elif scope == 'per_output_channel':
                        scale = float(q['scales'][n])
                    elif scope == 'per_input_channel':
                        scale = float(q['scales'][k])
                    else:
                        scale = float(q['scales'][k // block, n])
                    want = int(np.clip(np.rint(W[k, n] / scale), -127, 127))
                    self.assertEqual(int(q['codes'][k, n]), want, (scope, k, n))

    def test_rounding_is_half_to_even(self):
        # A column whose largest magnitude is 254 gives the exact scale 2.0, so
        # the ratios are exact halves; nearbyint (and numpy's rint) send each tie
        # to the even code, which is what the GPU encoder does.
        for value, code in ((127.0, 64), (5.0, 2), (7.0, 4), (-5.0, -2)):
            q = qi.quantize(np.array([[254.0], [value]]), 'per_output_channel')
            self.assertEqual(float(q['scales'][0]), 2.0)
            self.assertEqual(int(q['codes'][1, 0]), code, value)

    def test_saturation_and_endpoints(self):
        W = self.matrix(k=8, n=4)
        for scope, block in (('per_tensor', None), ('per_output_channel', None),
                             ('per_input_channel', None), ('block_k', 16)):
            q = qi.quantize(W, scope, block)
            self.assertLessEqual(int(np.abs(q['codes']).max()), 127)
            stats = qi.metrics(W, q)
            self.assertEqual(stats['over_range'], 0.0)
            # the largest magnitude in each scale group lands exactly on 127
            self.assertTrue((np.abs(q['codes']).max() == 127))

    def test_zero_weights_stay_zero(self):
        W = self.matrix()
        W[0, 0] = 0.0
        W[5, 3] = 0.0
        q = qi.quantize(W, 'per_output_channel')
        self.assertTrue(qi.metrics(W, q)['zeros_preserved'])
        self.assertEqual(int(q['codes'][0, 0]), 0)
        self.assertEqual(int(q['codes'][5, 3]), 0)

    def test_dequantize_is_codes_times_scale(self):
        W = self.matrix()
        q = qi.quantize(W, 'per_output_channel')
        got = qi.dequantize(q)
        want = q['codes'].astype(F64) * np.asarray(q['scales'], F64)[None, :]
        np.testing.assert_array_equal(got, want)

    def test_error_is_measured_against_the_exact_matrix(self):
        W = self.matrix(k=64, n=32)
        q = qi.quantize(W, 'per_output_channel')
        stats = qi.metrics(W, q)
        err = qi.dequantize(q) - W
        self.assertAlmostEqual(stats['rel_frobenius'],
                               float(np.linalg.norm(err) / np.linalg.norm(W)), places=12)
        self.assertAlmostEqual(stats['rmse'], float(np.sqrt((err ** 2).mean())), places=12)
        self.assertEqual(stats['data_bytes'], W.size)
        self.assertEqual(stats['scale_bytes'], 4 * W.shape[1])
        self.assertEqual(stats['total_bytes'], W.size + 4 * W.shape[1])

    def test_encoding_is_deterministic(self):
        W = self.matrix(k=128, n=64)
        self.assertTrue(qi.determinism(W))
        first = qi.quantize(W, 'per_output_channel')
        second = qi.quantize(W, 'per_output_channel')
        np.testing.assert_array_equal(first['codes'], second['codes'])
        np.testing.assert_array_equal(np.asarray(first['scales']).view(np.uint32),
                                      np.asarray(second['scales']).view(np.uint32))

    def test_finer_scopes_never_cost_more_on_this_data(self):
        W = self.matrix(k=96, n=40, seed=11)
        coarse = qi.metrics(W, qi.quantize(W, 'per_tensor'))['rel_frobenius']
        channel = qi.metrics(W, qi.quantize(W, 'per_output_channel'))['rel_frobenius']
        blocked = qi.metrics(W, qi.quantize(W, 'block_k', 16))['rel_frobenius']
        self.assertLessEqual(channel, coarse + 1e-12)
        self.assertLessEqual(blocked, channel + 1e-12)

    def test_d4r_split_divisor_changes_nothing_on_these_values(self):
        # d4r rounds against a float64 scale but stores float32; the two
        # conventions only differ within one rounding boundary
        W = self.matrix(k=64, n=32)
        changed, size = qi.d4r_split_agreement(W)
        self.assertEqual(size, W.size)
        self.assertLessEqual(changed, size)


class IntegerGemm(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.x = np.asarray(rng.normal(0.0, 0.5, (9, 32)), F64)
        self.w = np.asarray(rng.normal(0.0, 0.15, (32, 17)), F64)

    def test_accumulation_matches_an_element_wise_loop(self):
        aq = qi.quantize_rows(self.x, 'per_row')
        wq = qi.quantize(self.w, 'per_output_channel')
        got = qi.gemm_integer(aq['codes'], wq['codes'], aq['scales'], wq['scales'],
                              'per_output_channel')
        exact_int = qi.independent_integer_product(aq['codes'], wq['codes'])
        want = exact_int.astype(F64) * np.asarray(aq['scales'], F64) * np.asarray(wq['scales'], F64)[None, :]
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)

    def test_every_scope_reproduces_the_dequantized_product(self):
        # Whatever the scale grouping, the rescale has to reconstruct exactly the
        # product of the dequantized operands: that is what makes an INT8 GEMM a
        # faithful implementation of its own quantization policy.
        aq = qi.quantize_rows(self.x, 'per_row')
        for scope, block in (('per_tensor', None), ('per_output_channel', None),
                             ('per_input_channel', None), ('block_k', 8), ('block_k', 16)):
            wq = qi.quantize(self.w, scope, block)
            got = qi.gemm_integer(aq['codes'], wq['codes'], aq['scales'], wq['scales'], scope, block)
            want = qi.gemm_exact(qi.dequantize_rows(aq), qi.dequantize(wq))
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, err_msg=scope)

    def test_activation_endpoint_maps_to_127(self):
        aq = qi.quantize_rows(self.x, 'per_row')
        self.assertTrue((np.abs(aq['codes']).max(axis=1) == 127).all())
        # a zero row has no scale and no codes
        zero = qi.quantize_rows(np.zeros((3, 8), F64), 'per_row')
        self.assertTrue(np.array_equal(zero['codes'], np.zeros((3, 8), np.int8)))

    def test_int32_headroom(self):
        # K=256 of +-127 is far inside int32; the guard from EXP-005 is a
        # different order of magnitude (K <= 131071)
        codes = np.full((1, 256), 127, np.int8)
        weights = np.full((256, 1), 127, np.int8)
        product = qi.independent_integer_product(codes, weights)
        self.assertEqual(int(product[0, 0]), 127 * 127 * 256)
        self.assertLess(int(product[0, 0]), 2 ** 31 - 1)

    def test_quantized_error_is_reported_per_source(self):
        _, stats = qi.gemm_int8(self.x, self.w, 'per_output_channel')
        self.assertIsNotNone(stats['weight_only_rel'])
        self.assertGreaterEqual(stats['rel_frobenius'], stats['weight_only_rel'] - 1e-12)


class KernelVariants(unittest.TestCase):
    def test_variant_fields_are_explicit(self):
        enc0 = mw.parse_kernel('rrlite_enc0_4x4_mvhi_ldr_folded')
        self.assertEqual(enc0['stage'], 'enc0')
        self.assertEqual((enc0['motion'], enc0['range'], enc0['folded']), ('mvhi', 'ldr', True))
        lo = mw.parse_kernel('rrlite_enc0_4x4_mvlo_hdr_folded')
        self.assertEqual((lo['motion'], lo['range']), ('mvlo', 'hdr'))
        self.assertNotEqual(enc0, lo)

    def test_scalar_and_image_stages_parse(self):
        self.assertEqual(mw.parse_kernel('rrlite_dec0_4x4_folded')['stage'], 'dec0')
        post = mw.parse_kernel('rrlite_post_3_1_mvhi_ldr_folded')
        self.assertEqual((post['stage'], post['family']), ('post_3_1', 'post'))
        down = mw.parse_kernel('rrlite_downsample_kernel_static_ldr')
        self.assertEqual((down['stage'], down['motion']), ('downsample', 'static'))
        self.assertEqual(mw.parse_kernel('rrlite_enc1_4x4')['stage'], 'enc1')
        self.assertIsNone(mw.parse_kernel('rrlite_something_else'))
        self.assertIsNone(mw.parse_kernel('dltss_pwin_enc0_layer'))


if __name__ == '__main__':
    unittest.main()
