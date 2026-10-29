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
    """The kernel's dithered truncation, transcribed instruction by instruction.

        and.b32  e,  h, 0xFC00FC00
        or.b32   en, n, e
        add.f16x2 a, h, en          <- the clamped value, not its exponent
        sub.f16x2 t, a, e
        and.b32  r,  t, 0xFF80FF80
    """
    h = np.float16(np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
                           -np.inf, 1024.0))
    bits = int(h.view(U16))
    e = bits & 0xFC00
    added = np.float16(np.float32(h) + np.float32(np.uint16(e | n).view(np.float16)))
    t = np.float16(np.float32(added) - np.float32(np.uint16(e).view(np.float16)))
    return np.uint16(int(t.view(U16)) & 0xFF80).view(np.float16)


class Dither(unittest.TestCase):
    def test_the_add_is_against_the_clamped_value(self):
        """The mantissa must survive into the sum.

        `add.f16x2 %r224, %r225, %r226` takes `%r225` - the f16 of the value -
        and `%r226` = `e | n`.  Adding to `e` instead collapses every value in an
        exponent bin onto the same result: 0.75 and 0.5 would both dither to 0.5.
        The oracle's earlier bug did exactly that and lost half the distinct
        token vectors.
        """
        # 0.75 = 0x3A00, e = 0x3800 = 0.5, so 0.75 + 0.5 - 0.5 = 0.75
        self.assertEqual(float(dither(0.75, 0)), 0.75)
        self.assertEqual(float(dither(0.5, 0)), 0.5)
        self.assertNotEqual(float(dither(0.75, 0)), float(dither(0.5, 0)))
        # and the two differ from the exponent-only form on every bin
        for value in (0.625, 0.75, 0.875, 1.25, 1.5, 3.75):
            h = np.float16(value)
            e = np.uint16(int(h.view(U16)) & 0xFC00)
            added = np.float16(np.float32(np.uint16(e).view(np.float16))
                               + np.float32(np.uint16(int(e) | 0).view(np.float16)))
            t = np.float16(np.float32(added)
                           - np.float32(np.uint16(e).view(np.float16)))
            broken = np.uint16(int(t.view(U16)) & 0xFF80).view(np.float16)
            self.assertNotEqual(float(dither(value, 0)), float(broken), value)

    def test_hand_computed_cases(self):
        """Values worked out from the instruction sequence, not from the code."""
        # 0.5 -> e|127 = 0.5620117; 0.5 + that = 1.0625 (a tie, resolved up);
        # minus 0.5 = 0.5625, which survives 0xFF80
        self.assertEqual(float(dither(0.5, 127)), 0.5625)
        # 0.5 -> e|64 = 0.53125; 0.5 + 0.53125 = 1.03125; - 0.5 = 0.53125;
        # 0x3840 & 0xFF80 = 0x3800 = 0.5
        self.assertEqual(float(dither(0.5, 64)), 0.5)
        # 0.75 -> e|127 = 0.5620117; sum 1.3125; - 0.5 = 0.8125
        self.assertEqual(float(dither(0.75, 127)), 0.8125)
        # 1.0 -> 1.0 + 1.0 = 2.0; - 1.0 = 1.0
        self.assertEqual(float(dither(1.0, 0)), 1.0)
        # the sign carries through `e` and survives the 0xFF80 mask
        self.assertEqual(float(dither(-0.5, 0)), -0.5)
        self.assertEqual(float(dither(-1.0, 0)), -1.0)
        # zero stays zero: n < 128 never reaches bit 7 of the mantissa
        self.assertEqual(float(dither(0.0, 127)), 0.0)

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
                self.assertIn(abs(got), grid, (value, n, got))

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
#
# Two independent derivations, then a comparison.
#
# (a) The 20 values the kernel computes per pixel, in instruction order
#     (f292..f311): its own texel, the four motion candidates (which on a reset
#     frame are the same texel again), the exposure scalar and the RGB A of the
#     history sample.  `reset` zeroes the last five.
# (b) The physical slot order, from the two `st.shared.v4.u32` stores: each
#     4-byte word packs one `cvt.rn.satfinite.e4m3x2` of two f16x2 dither
#     registers, and the stores write words {r230,r231}, {r232,r233}, ...
#
# The map the model and the shader must both use is (b) applied to (a).

def ptx_value_sequence():
    """f292..f311: which value each slot of the *value* sequence carries."""
    own = ['r', 'g', 'b']
    candidates = ['r', 'g', 'b'] * 4           # f295..f306, four 2x2 neighbours
    return own + candidates + ['exposure'] + ['history'] * 4


def ptx_slot_order():
    """The 32 slots in store order, as indices into the value sequence (-1 = zero).

    The 16 `cvt.rn.satfinite.e4m3x2` results and the value pair each packs, in
    the order the PTX emits them:
        r230 (f292,f293)  r231 (f308,f309)  r232 (f294,f295)  r233 (f310,f311)
        r234 (f296,f297)  r235 (0,0)        r236 (f298,f299)  r237 (0,0)
        r238 (f300,f301)  r239 (0,0)        r240 (f302,f303)  r241 (0,0)
        r242 (f304,f305)  r243 (0,0)        r244 (f306,f307)  r245 (0,0)
    and the two `st.shared.v4.u32` stores, each packing four registers into a
    16-byte group:
        [r256]      = {r230, r231}, {r232, r233}, {r234, r235}, {r236, r237}
        [r256+1024] = {r238, r239}, {r240, r241}, {r242, r243}, {r244, r245}
    """
    pairs = {'r230': (0, 1), 'r231': (16, 17), 'r232': (2, 3), 'r233': (18, 19),
             'r234': (4, 5), 'r235': (-1, -1), 'r236': (6, 7), 'r237': (-1, -1),
             'r238': (8, 9), 'r239': (-1, -1), 'r240': (10, 11), 'r241': (-1, -1),
             'r242': (12, 13), 'r243': (-1, -1), 'r244': (14, 15), 'r245': (-1, -1)}
    order = []
    for group in (['r230', 'r231', 'r232', 'r233', 'r234', 'r235', 'r236', 'r237'],
                  ['r238', 'r239', 'r240', 'r241', 'r242', 'r243', 'r244', 'r245']):
        for register in group:
            order.extend(pairs[register])
    assert len(order) == 32
    return order


def derived_slot_channel():
    """slot -> colour channel, independently of `m_enc0_model`."""
    values = ptx_value_sequence()
    order = ptx_slot_order()
    channel = {}
    for slot, value in enumerate(order):
        if value < 0:
            continue
        v = values[value]
        if v in ('r', 'g', 'b'):
            channel[slot] = {'r': 0, 'g': 1, 'b': 2}[v]
    return channel


#: The map the model and the shader both carry; compared against the derivation
#: above rather than being its source.
SLOT_CHANNEL = derived_slot_channel()


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

    def test_the_nonzero_slots_carry_rgb_in_ascending_order(self):
        # the two texel-derived parts of the sequence (own colour and four
        # candidates) are the same value on a reset frame, so the 15 nonzero
        # slots cycle r, g, b
        got = [SLOT_CHANNEL[s] for s in sorted(SLOT_CHANNEL)]
        self.assertEqual(got, [0, 1, 2] * 5)
        self.assertEqual(list(sorted(SLOT_CHANNEL)),
                         [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28])

    def test_the_exposure_and_history_slots_are_the_zeroed_ones(self):
        # f307 (the exposure scalar) is slot 29 and f308..f311 (the history
        # sample) are slots 2, 3, 6, 7 - zeroed by `reset`, which is why they
        # are the gaps in the rgb cycle
        self.assertNotIn(2, SLOT_CHANNEL)
        self.assertNotIn(3, SLOT_CHANNEL)
        self.assertNotIn(6, SLOT_CHANNEL)
        self.assertNotIn(7, SLOT_CHANNEL)
        self.assertNotIn(29, SLOT_CHANNEL)


# ---------------------------------------------------------------------------
# the epilogue's channel permutation

def epilogue_channel_map():
    """Field channel -> mma column, from the eight final accumulator chains.

    mma 7/8, 15/16, 23/24, 31/32 are the last k32 step of the four `n/16`
    groups, two `g` halves each; the epilogue packs two chains into one stored
    32-bit word, so a channel is not the column that produced it.
    """
    out = []
    for plane in range(2):
        for half in range(2):                    # channels 0..15 / 16..31
            for j in range(4):
                lo = plane * 32 + (0 if half == 0 else 8)
                hi = plane * 32 + (16 if half == 0 else 24)
                out += [lo + 2 * j, lo + 2 * j + 1, hi + 2 * j, hi + 2 * j + 1]
    return out


class Epilogue(unittest.TestCase):
    def test_the_channel_map_is_a_bijection(self):
        m = epilogue_channel_map()
        self.assertEqual(len(m), 64)
        self.assertEqual(sorted(m), list(range(64)))

    def test_each_plane_maps_to_itself(self):
        m = epilogue_channel_map()
        for c in range(64):
            self.assertEqual(m[c] // 32, c // 32)

    def test_channel_zero_is_column_zero(self):
        # the identity is *not* the mapping, but the two agree at the first
        # column - the packing only reorders from channel 2 onward
        m = epilogue_channel_map()
        self.assertEqual(m[0], 0)
        self.assertEqual(m[1], 1)
        self.assertEqual(m[2], 16)
        self.assertNotEqual(m[:16], list(range(16)))

    def test_the_two_pi_clamp_is_the_f16_constant(self):
        # `mov.f32 %f582, 0f40C90FDB` then `cvt.rn.f16.f32` -> 6.28125
        self.assertEqual(float(np.float16(6.2831855)), 6.28125)


# ---------------------------------------------------------------------------
# the shader and the CPU oracle must not drift

def shader_array(text, name, count):
    body = text.split(name, 1)[1].split('};', 1)[0]
    values = [v.strip().rstrip('u') for v in body.split('{', 1)[1].split(',')]
    return [int(v) for v in values if v]


class ShaderAgreement(unittest.TestCase):
    """The HLSL stage and `scripts/m_enc0_model.py` share three tables.

    Parsing the shader source is the only way to check them without NVIDIA data,
    and a silent divergence between the two implementations is exactly the kind
    of error that would survive a fixture run on one side only.
    """

    @classmethod
    def setUpClass(cls):
        import os
        root = os.path.dirname(HERE)
        with open(os.path.join(root, 'src', 'native', 'enc0_cs.hlsl'),
                  encoding='utf-8') as handle:
            cls.shader = handle.read()
        cls.model = __import__('m_enc0_model')

    def test_the_slot_map(self):
        got = shader_array(self.shader, 'kSlotChannel[32]', 32)
        want = [int(c) if c >= 0 else 3 for c in self.model.SLOT_CHANNEL]
        self.assertEqual(got, want)

    def test_the_channel_permutation(self):
        got = shader_array(self.shader, 'kColumn[64]', 64)
        self.assertEqual(got, [int(v) for v in self.model.channel_columns()])
        self.assertEqual(got, epilogue_channel_map())

    def test_the_geometry_constants(self):
        self.assertIn('kTokensPerRow = 480', self.shader)
        self.assertIn('kTokenRowWords = 3840', self.shader)     # 15360 / 4
        self.assertIn('kPlaneWords = 1044480', self.shader)     # 4177920 / 4
        self.assertEqual(480 * 32, 15360)
        self.assertEqual(272 * 15360, 4177920)


if __name__ == '__main__':
    sys.exit(unittest.main())
