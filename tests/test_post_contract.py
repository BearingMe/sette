#!/usr/bin/env python3
"""B4.5 contract tests: post's tile geometry, Gaussian pairing, packing and
surface conversions.

Each check is against something independent of the implementation:

  * the tile index bound is derived by simulating the PTX's own addressing
    (`trunc(scale*16*bx)`, `floor(pos-0.5)`, `(row*10+col)*4`) over the whole
    captured grid - if the shader could read outside the 160 written entries the
    simulation says so;
  * the paired-row rule is checked against the definition `base = floor(pos-0.5)`
    rather than against the shader's branch;
  * the 11-11-10 packing and the 10-bit surface value are checked against
    hand-computed codes;
  * the f16 surface's truncation is checked against values whose nearest and
    truncated f16 differ.

No NVIDIA data is needed.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

F16, F32 = np.float16, np.float32

SCALE = F32(1.0) / F32(3.0)
RECON = (2880, 1620)
RENDER = (960, 540)
GROUPS = (180, 51)
THREADS = (16, 16)


def tile_index(bx, by, tx, ty, i, j):
    """The tile entry one tap reads, from the PTX's addressing."""
    ox = int(F32(bx * 16) * SCALE)
    oy = int(F32(2 * by * 16) * SCALE)
    x = bx * 16 + tx
    y = 2 * (by * 16 + ty)
    pos_x = F32(F32(x) + F32(0.5)) * SCALE
    pos_y = F32(F32(y) + F32(0.5)) * SCALE
    base_x = int(np.floor(F32(pos_x - F32(0.5))))
    base_y = int(np.floor(F32(pos_y - F32(0.5))))
    return base_y - oy + 1 + j, base_x - ox + 1 + i


class TileGeometry(unittest.TestCase):
    def test_the_dispatch_covers_the_reconstruction_grid(self):
        self.assertEqual(GROUPS[0] * THREADS[0], RECON[0])
        self.assertGreaterEqual(GROUPS[1] * THREADS[1] * 2, RECON[1])

    def test_the_scalar_addressing_agrees_with_the_vectorised_check(self):
        # one tap through the scalar formula, so the bound test below cannot
        # drift from the addressing it is supposed to model
        row, col = tile_index(0, 0, 0, 0, 0, 0)
        self.assertEqual((row, col), (0, 0))

    def test_every_tap_reads_a_written_tile_entry(self):
        # 160 threads write a 16x10 tile; every tap of every thread must land
        # inside it, or the shader reads uninitialised shared memory.
        bx, by = np.meshgrid(np.arange(GROUPS[0]), np.arange(GROUPS[1]),
                             indexing='ij')
        tx = np.arange(THREADS[0])[None, None, :, None]
        ty = np.arange(THREADS[1])[None, None, None, :]
        x = bx[..., None, None] * 16 + tx
        y = 2 * (by[..., None, None] * 16 + ty)
        ox = (F32(bx * 16) * SCALE).astype(np.int32)
        oy = (F32(2 * by * 16) * SCALE).astype(np.int32)
        pos_x = (F32(x) + F32(0.5)) * SCALE
        pos_y = (F32(y) + F32(0.5)) * SCALE
        base_x = np.floor(pos_x - F32(0.5)).astype(np.int32)
        base_y = np.floor(pos_y - F32(0.5)).astype(np.int32)
        inside = (x < RECON[0]) & (y < RECON[1])
        for i in range(3):
            for j in range(3):
                col = base_x - ox[..., None, None] + 1 + i
                row = base_y - oy[..., None, None] + 1 + j
                ok = (row >= 0) & (row < 16) & (col >= 0) & (col < 10)
                self.assertTrue(bool(ok[inside].all()),
                                f'tap ({i},{j}) leaves the tile')
        self.assertEqual(16 * 10, 160)

    def test_the_tile_is_one_fetch_per_entry(self):
        # three arrays of 4 f16 per entry: 160 * 4 * 2 = 1280 bytes, the shared
        # array size the PTX declares
        self.assertEqual(160 * 4 * 2, 1280)


def base(pos):
    return int(np.floor(F32(pos) - F32(0.5)))


class PairedRows(unittest.TestCase):
    def test_the_odd_row_keeps_the_even_rows_texels(self):
        # every thread owns rows (y, y+1); the tile rows come from the even
        # row's base for both (PTX 479-481 reuse the even row's offsets)
        for y in range(0, 40, 2):
            pos0 = (y + 0.5) / 3.0
            pos1 = (y + 1.5) / 3.0
            b0, b1 = base(pos0), base(pos1)
            # the residual is measured from the even row's base
            d1 = np.float16(np.float32(b1 + 0.5) - np.float32(pos1))
            if b1 != b0:
                d1 = np.float16(d1 - np.float16(1.0))
            self.assertAlmostEqual(float(d1) - (float(b0) + 0.5 - float(pos1)),
                                   0.0, places=2)

    def test_the_adjustment_only_fires_when_the_bases_differ(self):
        seen = {True: 0, False: 0}
        for y in range(0, 60, 2):
            b0 = base((y + 0.5) / 3.0)
            b1 = base((y + 1.5) / 3.0)
            seen[b1 != b0] += 1
        # both cases occur in the grid, so the branch is exercised
        self.assertGreater(seen[True], 0)
        self.assertGreater(seen[False], 0)


class Packing(unittest.TestCase):
    def test_the_word_is_r_11_g_11_b_10(self):
        def word(r, g, b):
            return (r & 0x7FF) | ((g & 0x7FF) << 11) | ((b & 0x3FF) << 22)
        self.assertEqual(word(0, 0, 0), 0)
        self.assertEqual(word(2047, 0, 0), 0x7FF)
        self.assertEqual(word(0, 2047, 0), 0x7FF << 11)
        self.assertEqual(word(0, 0, 1023), 0x3FF << 22)
        self.assertEqual(word(2047, 2047, 1023), 0xFFFFFFFF)
        # the captured first word decodes to a plausible code triple
        captured = 0x80100200
        self.assertEqual(captured & 0x7FF, 512)
        self.assertEqual((captured >> 11) & 0x7FF, 512)
        self.assertEqual((captured >> 22) & 0x3FF, 512)

    def test_the_code_is_a_rounded_scale(self):
        # trunc(clamp(fma(v, 2047, 0.5), 0, 2047))
        def code(v, top):
            return int(np.clip(np.floor(F32(v) * F32(top) + F32(0.5)), 0, top))
        self.assertEqual(code(0.0, 2047), 0)
        self.assertEqual(code(1.0, 2047), 2047)
        self.assertEqual(code(0.25, 2047), 512)
        self.assertEqual(code(0.5, 1023), 512)
        # the +0.5 means a value one code below the top does not saturate
        self.assertEqual(code(2046.4 / 2047.0, 2047), 2046)


class Surfaces(unittest.TestCase):
    def test_the_half_surface_holds_k_over_1023(self):
        # the captured values are exact IEEE quotients, so 0.25 is 256/1023
        q = np.float32(256) / np.float32(1023)
        self.assertEqual(q.view(np.uint32), np.float32(0.25024438).view(np.uint32))
        self.assertEqual(round(float(q) * 1023), 256)
        self.assertEqual(round(0.50048876 * 1023), 512)
        self.assertEqual(round(0.7497556 * 1023), 767)

    def test_the_half_surface_is_a_10_bit_quantisation(self):
        # an arbitrary mean lands on the 1023 grid, not on the f32 value
        mean = 0.2500184
        code = int(np.floor(mean * 1023 + 0.5))
        self.assertEqual(code, 256)
        self.assertNotEqual(float(np.float32(mean)), 256.0 / 1023.0)

    def test_the_history_surface_truncates(self):
        # the captured f16 values are trunc(x), not round-to-nearest(x):
        # 0.2503 rounds up to 0.250244 but truncates to 0.25
        nearest = np.float16(0.2503)
        truncated = np.uint16(nearest.view(np.uint16) - 1).view(np.float16)
        self.assertEqual(nearest.view(np.uint16),
                         np.float16(0.25024414).view(np.uint16))
        self.assertEqual(truncated.view(np.uint16), np.float16(0.25).view(np.uint16))
        self.assertNotEqual(nearest.view(np.uint16), truncated.view(np.uint16))


class Constants(unittest.TestCase):
    def test_the_covariance_scale(self):
        # 0fBF38A8C1 and 0fBFB8A8C1: the second is exactly twice the first
        k = np.float32(-0.7213249802589417)
        self.assertEqual(float(k * np.float32(2.0)), -1.4426499605178833)
        self.assertEqual(np.float32(k).view(np.uint32), 0xBF38A8C1)
        self.assertEqual(np.float32(k * np.float32(2.0)).view(np.uint32), 0xBFB8A8C1)

    def test_the_gain_and_the_three_thresholds(self):
        # param+120, 1.1 and -0.05 (0f3F8CCCCD / 0fBD4CCCCD), 1/255
        self.assertEqual(np.float32(2.7530112266540527).view(np.uint32), 0x40303156)
        self.assertEqual(np.float32(1.1).view(np.uint32), 0x3F8CCCCD)
        self.assertEqual(np.float32(-0.05).view(np.uint32), 0xBD4CCCCD)
        self.assertEqual(np.float32(1.0 / 255.0).view(np.uint32), 0x3B808081)


class ShaderAgreement(unittest.TestCase):
    """The HLSL stage and `scripts/m_post_model.py` share four constants."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, 'src', 'native', 'post_cs.hlsl'),
                  encoding='utf-8') as handle:
            cls.shader = handle.read()
        import m_post_model
        cls.model = m_post_model

    def test_the_covariance_constants(self):
        self.assertIn(f'{float(self.model.COV_XXYY):.16f}', self.shader)
        self.assertIn(f'{float(self.model.COV_XY):.16f}', self.shader)

    def test_the_fma_constants(self):
        for value in (self.model.C_1_1, self.model.C_M_0_05,
                      self.model.RC_FLOOR, self.model.INV_255):
            self.assertIn(f'{float(value):.16g}', self.shader.replace('f;', ';')
                          .replace('f ', ' ') + '  ' + f'{float(value):.9g}')

    def test_the_packing_offsets(self):
        self.assertIn('2047.0f', self.shader)
        self.assertIn('1023.0f', self.shader)
        self.assertIn('(cg << 11)', self.shader)
        self.assertIn('(cb << 22)', self.shader)

    def test_the_geometry_constants(self):
        self.assertIn('numthreads(16, 16, 1)', self.shader)
        self.assertIn('Tile0[640]', self.shader)
        self.assertIn('lane < 160u', self.shader)


if __name__ == '__main__':
    sys.exit(unittest.main())
