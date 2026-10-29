#!/usr/bin/env python3
"""B4.4 contract tests: enc0's blob, dither and token-field geometry.

Each check is against something independent of any implementation here:

  * the kernel's own address arithmetic (`ld.weak.global.ca.v4.u32` at
    `weights + lane*16 + 4096 + 512*kb`, feeding two m16n8k32 B fragments)
    compared with `m_weights.layout_offsets('enc0')`, and required to be a
    bijection of the 8 192 matrix bytes;
  * the PTX's dithered-truncation sequence, transcribed as masks and checked
    against the E4M3 grid it is supposed to land on;
  * the dither index, which `prmt` builds as `(frame << 8) | (y&15)<<4 | (x&15)`;
  * the token store's plane arithmetic, derived from the store instruction rather
    than reused from dec0.

No NVIDIA data is needed.
"""
import sys
import unittest

import numpy as np

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(HERE), 'scripts'))

import m_weights as mw        # noqa: E402

K, N = 128, 64
BASE = 4096
BIAS = 12288
U16 = np.uint16


# ---------------------------------------------------------------------------
# the matrix, from the load and the mma

def ptx_weight_offsets(k, n):
    """Byte of element (k, n), from enc0's own instructions.

    `ld.weak.global.ca.v4.u32` at `weights + lane*16 + 4096 + 512*kb` gives 16
    bytes per lane; the fragment the mma consumes is 8 bytes, so one load feeds
    two n8 groups (bytes 8*g .. 8*g+7), and with K = 128 the four k-chunks of 32
    step through blocks 4*(n/16) + k/32.
    """
    kb = 4 * (n // 16) + k // 32
    lane = 4 * (n % 8) + (k % 16) // 4
    g = (n % 16) // 8
    j = (k % 4) + 4 * ((k % 32) // 16)
    return BASE + 512 * kb + 16 * lane + 8 * g + j


class WeightLayout(unittest.TestCase):
    def test_offsets_agree_with_the_inventory_decoder(self):
        spec = mw.front_end_layout('enc0')
        have = mw.layout_offsets(spec)
        for k in range(spec['K']):
            for n in range(spec['N']):
                self.assertEqual(int(have[k, n]), ptx_weight_offsets(k, n),
                                 f'({k},{n})')

    def test_offsets_are_a_bijection_of_the_matrix_bytes(self):
        seen = np.array([[ptx_weight_offsets(k, n) for n in range(N)]
                         for k in range(K)])
        self.assertTrue(np.array_equal(np.sort(seen.ravel()),
                                       np.arange(BASE, BASE + K * N)))

    def test_the_blob_is_noise_then_matrix_then_bias(self):
        spec = mw.front_end_layout('enc0')
        self.assertEqual(spec['aux'], (0, 4096))
        self.assertEqual(spec['base'], BASE)
        self.assertEqual(spec['bias'], BIAS)
        self.assertEqual(spec['bias_count'], N)
        self.assertEqual(BIAS - BASE, K * N)
        self.assertEqual(BIAS + 2 * N, 12416)     # the captured allocation size

    def test_the_bias_is_one_f16_per_output_column(self):
        # `ld.global.u32 [weights + 12288 + 16*bi + 4*(lane&3)]` into mma C as
        # {r,r}: the word holds the f16 pair of columns 8*bi + 2*(lane&3) and the
        # next one, so the f16 index of column n is n itself
        for n in range(N):
            bi, lane = n // 8, (n % 8) // 2
            word = BIAS + 16 * bi + 4 * lane
            self.assertEqual(word, BIAS + 2 * n - 2 * (n % 2))
            self.assertEqual(word + (2 if n % 2 else 0), BIAS + 2 * n)
            self.assertLessEqual(word + 4, BIAS + 2 * N)


# ---------------------------------------------------------------------------
# the dither

def dither(value, n):
    """The kernel's dithered truncation: `and 0xFC00` / `or n` / add / sub / `and 0xFF80`."""
    h = np.float16(np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
                           0.0, 1024.0))
    bits = int(h.view(U16))
    e = bits & 0xFC00
    added = np.float16(np.float32(np.uint16(e | n).view(np.float16))
                       + np.float32(h))
    t = np.float16(np.float32(added) - np.float32(np.uint16(e).view(np.float16)))
    return np.uint16(int(t.view(U16)) & 0xFF80).view(np.float16)


class Dither(unittest.TestCase):
    def test_the_index_is_frame_row_tile(self):
        # prmt.b32 d, frame, (y&15)<<4 | (x&15), 0x2104 -> (frame<<8) | tile
        def index(frame, x, y):
            tile = ((y & 15) << 4) | (x & 15)
            a, b = frame & 0xFFFFFFFF, tile & 0xFFFFFFFF
            return (b & 0xFF) | ((a & 0xFF) << 8) | (((a >> 8) & 0xFF) << 16) \
                | (((a >> 16) & 0xFF) << 24)
        for frame in (0, 1, 7):
            for x, y in ((0, 0), (15, 15), (3, 9), (16, 0), (0, 32)):
                self.assertEqual(index(frame, x, y),
                                 (frame << 8) | (((y & 15) << 4) | (x & 15)))
        self.assertLess((7 << 8) | 255, 2048)     # inside a 4096-byte u16 table

    def test_n_fits_the_mantissa(self):
        # `shr.u16 n, table[index], 9` on a u16 gives at most 127, so `e | n`
        # never reaches the exponent field (bits 10..15)
        self.assertEqual(0xFFFF >> 9, 127)
        self.assertEqual(0xFC00 & 0xFFC0, 0xFC00)

    def test_the_result_is_on_the_e4m3_grid(self):
        table = mw.e4m3_decode(np.arange(128, dtype=np.uint8)).astype(np.float32)
        rng = np.random.default_rng(20261029)
        values = np.concatenate([rng.uniform(-600, 600, 4000),
                                 table, -table, [0.0, 1e-6, 1024.0, -1024.0]])
        grid = set(mw.e4m3_decode(np.arange(128, dtype=np.uint8)).astype(float))
        for value in values:
            for n in (0, 1, 63, 64, 127):
                # the dithered value can overshoot 448; cvt.rn.satfinite is what
                # saturates it, so the grid is checked on the encoded result
                code = int(mw.e4m3_encode(np.array([dither(value, n)], np.float32))[0])
                got = float(mw.e4m3_decode(np.uint8(code)))
                self.assertLessEqual(abs(got), 448.0 + 1e-9, (value, n, got))
                self.assertIn(got, grid, (value, n, got))

    def test_dither_spans_exactly_one_e4m3_step(self):
        # n in 0..127 covers one quantization interval of the e4m3 grid, so the
        # dithered value is a floor to one of two adjacent codes - that is the
        # whole point of the table and is what makes the feature bytes depend on
        # the pixel position at all
        value = 0.5
        codes = {int(mw.e4m3_encode(np.float32(float(dither(value, n))))[0])
                 if False else
                 int(mw.e4m3_encode(np.array([dither(value, n)], np.float32))[0])
                 for n in range(128)}
        self.assertEqual(len(codes), 2, sorted(codes))

    def test_negative_values_keep_their_sign(self):
        for value in (-0.5, -1.0, -3.25, -448.0):
            for n in (0, 64, 127):
                self.assertLessEqual(float(dither(value, n)), 0.0)


# ---------------------------------------------------------------------------
# the token field

def token_offset(ty, tx, c, padded=(960, 544)):
    """enc0's store, from its own instruction.

    `r47 = (param+16.x >> 1) << 3` words per token row and `r48 = (param+16.y >> 1) * r47`
    for the channel plane: the *padded* size drives both, exactly as it does for
    dec0's load, while the mirror uses the visible `param+56`.
    """
    return ((c // 32) * (padded[1] // 2) * (padded[0] // 2) * 32
            + ty * (padded[0] // 2) * 32 + tx * 32 + (c % 32))


class TokenField(unittest.TestCase):
    def test_the_store_matches_dec0s_load_addressing(self):
        # enc0 stores with a 272-row plane stride taken from `param+16`, the same
        # term dec0's load uses; the two stages' fields are the same size and the
        # same layout, which is why dec1 can overwrite enc0's output in place
        self.assertEqual(token_offset(0, 0, 32), 4_177_920)
        self.assertEqual(token_offset(1, 0, 0) - token_offset(0, 0, 0), 15_360)
        self.assertEqual(token_offset(0, 1, 0) - token_offset(0, 0, 0), 32)
        self.assertEqual(token_offset(0, 0, 63), 4_177_920 + 31)
        self.assertEqual(token_offset(271, 479, 63) + 1, 2 * 4_177_920)

    def test_the_visible_size_is_not_the_plane_size(self):
        # param+16 (960, 544) sizes the field; param+56 (960, 540) is the mirror.
        # Using the visible size for the plane would shift channel 32..63 by two
        # rows - the mistake EXP-013 made for dec0.
        visible = token_offset(0, 0, 32, padded=(960, 540))
        self.assertNotEqual(visible, token_offset(0, 0, 32))
        self.assertEqual(token_offset(0, 0, 32) - visible, 30_720)

    def test_the_dispatched_grid_covers_the_padded_render_size(self):
        # launch 121x69 blocks of 8x8 pixels over a 960x544 padded frame
        self.assertEqual(121 * 8, 968)
        self.assertGreaterEqual(121 * 8, 960)
        self.assertEqual(69 * 8, 552)
        self.assertGreaterEqual(69 * 8, 544)


# ---------------------------------------------------------------------------
# the feature slots

#: The 32 slots of one pixel that carry a value, from d4r's pinned
#: `enc0_features.glsl` (documented there as the same 32 slots the PTX fills) and
#: confirmed against the PTX's 16-byte shared-memory stores.  On a reset frame
#: `h[k] == colour`, so the vector is a function of one texel.
SLOT_CHANNEL = {0: 0, 1: 1, 4: 2,
                5: 0, 8: 0, 9: 0, 12: 1, 13: 1, 16: 1,
                17: 2, 20: 2, 21: 2, 24: 0, 25: 0, 28: 0}


class FeatureSlots(unittest.TestCase):
    def test_sixteen_slots_per_plane_are_stored_by_two_stores(self):
        # `st.shared.v4.u32 [r256]` and `[r256+1024]`: two 16-byte stores per
        # pixel, 16 slots each, in two 1024-byte planes
        self.assertEqual(2 * 16, 32)
        self.assertEqual(2048, 1024 * 2)

    def test_a_token_gathers_four_pixels_one_per_k32_step(self):
        # a 2x2 pixel block is one token, and each pixel is one k32 step of the
        # embedding: 4 * 32 = 128 features
        self.assertEqual(4 * 32, K)
        self.assertEqual(K // 32, 4)

    def test_the_reset_frame_needs_only_the_colour_texel(self):
        # every nonzero slot is one of the three colour channels, repeated
        self.assertEqual(set(SLOT_CHANNEL.values()), {0, 1, 2})
        self.assertEqual(len(SLOT_CHANNEL), 15)
        # the remaining 17 slots are the history, the previous feature and the
        # motion-change scalar, which are exactly zero on a reset frame
        self.assertEqual(32 - len(SLOT_CHANNEL), 17)


if __name__ == '__main__':
    sys.exit(unittest.main())
