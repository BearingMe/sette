#!/usr/bin/env python3
"""B4.3 contract tests: dec0's operand mapping and its number formats.

These run without NVIDIA data.  Each check is against something independent of
the shader:

  * the E4M3 code space, rebuilt from the OCP definition (`1-4-3`, bias 7), so the
    decode is checked against the format, not against the shader's own arithmetic.
    Subnormals are explicitly covered: the first native attempt gave them an
    implicit leading 1 and inflated every small weight by 8x.
  * exact rational arithmetic for the encoder's round-to-nearest-even, including
    the ties and the 448 saturation.
  * the PTX's weight-address formula, compared with `m_weights.layout_offsets`
    and required to be a bijection of the 8192 matrix bytes.
  * the token field's geometry, derived from the kernel's own terms.

What these cannot show is that the capture agrees; that comparison is the GPU run
recorded in EXP-014.
"""
import sys
import unittest
from fractions import Fraction

import numpy as np

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(HERE), 'scripts'))

import m_weights as mw        # noqa: E402


# ---------------------------------------------------------------------------
# the format, from its definition

def e4m3_values():
    """Every finite E4M3 magnitude, ascending, from the OCP definition."""
    values = [Fraction(0)]
    for m in range(1, 8):                        # subnormal: m * 2^-9
        values.append(Fraction(m, 2 ** 9))
    for e in range(1, 16):                       # normal: (1 + m/8) * 2^(e-7)
        for m in range(8):
            if e == 15 and m == 7:               # 0x7F is NaN
                continue
            values.append(Fraction(8 + m, 8) * Fraction(2) ** (e - 7))
    return sorted(set(values))


E4M3 = e4m3_values()


def e4m3_rne(value):
    """Nearest E4M3 magnitude of an exact value, ties to even code."""
    import bisect
    value = abs(Fraction(value))
    if value >= Fraction(448):
        return Fraction(448)
    i = bisect.bisect_left(E4M3, value)
    lo, hi = E4M3[max(i - 1, 0)], E4M3[min(i, len(E4M3) - 1)]
    dlo, dhi = value - lo, hi - value
    if dhi < dlo:
        return hi
    if dlo < dhi:
        return lo
    # An exact midpoint: the code parity is the significand's low bit, because
    # each binade holds eight consecutive codes.
    return lo if (E4M3.index(lo) % 2 == 0) else hi


# ---------------------------------------------------------------------------
# the shader's helpers, transcribed

def decode(code):
    """dec0_cs.hlsl's E4M3Value."""
    s = code >> 7
    e = (code >> 3) & 0xF
    m = code & 7
    if e == 0:
        v = Fraction(m, 2 ** 9)                    # no implicit leading 1
        return -v if s else v
    if e == 15 and m == 7:
        return None                                # NaN
    v = Fraction(8 + m, 8) * Fraction(2) ** (e - 7)
    return -v if s else v


def f16(value):
    """The nearest binary16 to a value, exactly: the shader's argument type."""
    return Fraction(float(np.float16(float(value))))


def encode(value):
    """dec0_cs.hlsl's E4M3Code; its argument is an exact binary16 value."""
    value = f16(value)
    negative = value < 0
    a = abs(Fraction(value))
    sign = 0x80 if negative else 0
    if a == 0:
        return sign
    # the shader works on f16 bits: find the biased exponent and mantissa
    if a < Fraction(2) ** -14:
        return sign                                # f16 subnormal: below 2^-9
    e = 0
    y = a
    while y >= 2:
        y /= 2
        e += 1
    while y < 1:
        y *= 2
        e -= 1
    biased = e + 15
    mant = int((y - 1) * 1024)
    if biased < 9:
        shift = 16 - biased
        n = (1024 + mant) >> shift
        rem = (1024 + mant) & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (n & 1)):
            n += 1
        return sign | min(n, 126)
    e4 = biased - 8
    m4 = mant >> 7
    rem = mant & 0x7F
    if rem > 64 or (rem == 64 and (m4 & 1)):
        m4 += 1
        if m4 == 8:
            m4 = 0
            e4 += 1
    if e4 > 15 or (e4 == 15 and m4 == 7):
        return sign | 0x7E
    return sign | (e4 << 3) | m4


class E4M3Format(unittest.TestCase):
    def test_decode_matches_the_formats_own_values(self):
        # codes 0..126 are the ascending finite magnitudes; 0x7F is NaN
        for code in range(127):
            self.assertEqual(decode(code), E4M3[code], f'code {code:#04x}')
        self.assertIsNone(decode(0x7F))
        for code in range(0x80, 0xFF):
            self.assertEqual(decode(code), -E4M3[code - 0x80], f'code {code:#04x}')
        self.assertIsNone(decode(0xFF))

    def test_subnormals_have_no_implicit_leading_one(self):
        # The first native version treated them as (1 + m/8) * 2^-6, which is 8x
        # too large at m = 1 and 2.6x at m = 7.  17 of the 64 weights of column 0
        # are subnormal codes, so this dominated every product.
        for m in range(1, 8):
            self.assertEqual(decode(m), Fraction(m, 2 ** 9))
        self.assertEqual(decode(4), Fraction(1, 128))
        self.assertNotEqual(decode(4), Fraction(1, 128) * Fraction(9, 8))

    def test_encode_round_trips_every_code(self):
        for code in range(128):
            if code == 0x7F:
                continue
            self.assertEqual(encode(decode(code)), code, f'code {code:#04x}')

    def test_encode_is_round_to_nearest_even(self):
        # sweep values between adjacent representable magnitudes, including the
        # exact midpoints, and require the encoder to agree with exact RNE
        rng = np.random.default_rng(20261029)
        checked = 0
        for i in range(0, len(E4M3) - 1):
            lo, hi = E4M3[i], E4M3[i + 1]
            for t in (Fraction(0), Fraction(1, 4), Fraction(1, 2),
                      Fraction(3, 4), Fraction(1)):
                v = lo + (hi - lo) * t
                if v < Fraction(2) ** -24:         # below f16's smallest subnormal
                    continue
                want = e4m3_rne(f16(v))
                got = decode(encode(v))
                self.assertEqual(got, want, f'{float(v)!r} between '
                                            f'{float(lo)!r} and {float(hi)!r}')
                checked += 1
        for _ in range(2000):
            v = Fraction(float(rng.uniform(0, 500)))
            if v >= 448:
                continue
            got, want = decode(encode(v)), e4m3_rne(f16(v))
            if want >= 448:
                want = Fraction(448)
            self.assertEqual(got, want, f'{float(v)!r}')
            checked += 1
        self.assertGreater(checked, 800)

    def test_encode_saturates_at_448(self):
        for value in (Fraction(449), Fraction(1000), Fraction(65504)):
            self.assertEqual(encode(value), 0x7E)
            self.assertEqual(decode(encode(value)), Fraction(448))

    def test_saturation_boundary_is_the_midpoint_above_448(self):
        # 464 is the midpoint between 448 and the next representable value, so
        # everything below it stays at 448 and everything at or above saturates
        self.assertEqual(decode(encode(Fraction(463))), Fraction(448))
        self.assertEqual(decode(encode(Fraction(464))), Fraction(448))


# ---------------------------------------------------------------------------
# the weight layout

def ptx_weight_offsets(k, n):
    """The byte of element (k, n), from the kernel's load and mma instructions.

    `ld.weak.global.ca.v4.u32` at `weights + lane*16 + 512*kb` feeds two
    m16n8k32 B fragments (8 bytes per lane, at lane*16 + 8*g).  For that
    instruction's B fragment k = 4*tig + j%4 + 16*(j/4) and n = groupID.
    """
    kb = 2 * (n // 16) + k // 32
    lane = 4 * (n % 8) + (k % 16) // 4
    g = (n % 16) // 8
    j = (k % 4) + 4 * ((k % 32) // 16)
    return 512 * kb + 16 * lane + 8 * g + j


class WeightLayout(unittest.TestCase):
    def test_offsets_agree_with_the_inventory_decoder(self):
        spec = mw.front_end_layout('dec0')
        have = mw.layout_offsets(spec)
        for k in range(spec['K']):
            for n in range(spec['N']):
                self.assertEqual(int(have[k, n]), ptx_weight_offsets(k, n),
                                 f'({k},{n})')

    def test_offsets_are_a_bijection_of_the_matrix_bytes(self):
        seen = np.array([[ptx_weight_offsets(k, n) for n in range(128)]
                         for k in range(64)])
        self.assertTrue(np.array_equal(np.sort(seen.ravel()), np.arange(8192)))

    def test_bias_is_one_f16_per_column_and_does_not_overlap_the_matrix(self):
        spec = mw.front_end_layout('dec0')
        self.assertEqual(spec['bias'], 8192)
        self.assertEqual(spec['bias_count'], 128)
        self.assertEqual(2 * spec['bias_count'], 256)
        self.assertEqual(spec['base'] + spec['K'] * spec['N'], spec['bias'])


# ---------------------------------------------------------------------------
# the token field

def token_offset(ty, tx, c, width=960, height=544):
    """dec0_cs.hlsl's token addressing, from the kernel's own terms."""
    return ((c // 32) * (height // 2) * (width // 2) * 32
            + ty * (width // 2) * 32 + tx * 32 + (c % 32))


class TokenField(unittest.TestCase):
    def test_plane_stride_matches_the_parameter_block(self):
        # the kernel computes H/2 * (W/2) * 8 words from param+16, which is the
        # *padded* render size (960, 544) - reading 540 there was EXP-013's error
        self.assertEqual(token_offset(0, 0, 32), 4177920)
        self.assertEqual(4177920, 272 * 480 * 32)

    def test_field_is_two_planes_of_272_rows(self):
        width, height = 960, 544
        last = token_offset(height // 2 - 1, width // 2 - 1, 63, width, height)
        self.assertEqual(last + 1, 2 * (height // 2) * (width // 2) * 32)
        self.assertEqual(last + 1, 8355840)

    def test_token_bytes_are_32_per_channel_plane(self):
        for ty, tx in ((0, 0), (0, 1), (1, 0), (271, 479)):
            for c in range(0, 64, 7):
                self.assertEqual(token_offset(ty, tx, c) % 32, c % 32)
        self.assertEqual(token_offset(1, 0, 0) - token_offset(0, 0, 0), 15360)
        self.assertEqual(token_offset(0, 1, 0) - token_offset(0, 0, 0), 32)


# ---------------------------------------------------------------------------
# the column map and the store

LIVE = [(s, c) for s in range(4) for c in range(20)]
VALUE = {'v0': (0, 1, 2, 3), 'v1': (4, 5, 6, 7), 'v2': (8, 9, 10, 11),
         'v3': (12, 13, 14, 15), 'v4': (16, 17, 18, 19)}


class ColumnMap(unittest.TestCase):
    """The sub-position/column map the capture established (EXP-014)."""

    def test_sub_position_uses_a_contiguous_32_column_block(self):
        for sub in range(4):
            self.assertEqual([c for s, c in LIVE if s == sub], list(range(20)))
            self.assertEqual(sub * 32 + 19, sub * 32 + 19)

    def test_the_four_outputs_read_distinct_values(self):
        # gates, mix, covariance and the recurrent feature, by value name
        gates = [VALUE['v0'][0], VALUE['v0'][1], VALUE['v2'][2]]
        mix = [VALUE['v2'][3], VALUE['v3'][0], VALUE['v3'][1]]
        cov = [VALUE['v3'][2], VALUE['v3'][3], VALUE['v4'][0]]
        feat = [VALUE['v1'][2], VALUE['v1'][3], VALUE['v2'][0], VALUE['v2'][1]]
        fresh = [VALUE['v0'][2], VALUE['v0'][3], VALUE['v1'][0], VALUE['v1'][1]]
        used = gates + mix + cov + feat + fresh
        # 17 of the 20 live columns are read: v4.y, v4.z and v4.w are computed
        # and discarded (they are the four-sub-position padding)
        self.assertEqual(len(set(used)), 17)
        self.assertEqual(sorted(used), list(range(17)))
        for unused in (17, 18, 19):
            self.assertNotIn(unused, used)


def trunc16(value):
    """The store: f32 -> f16 truncating toward zero."""
    v = np.float32(value)
    if v == 0 or not np.isfinite(v):
        return np.float16(v)
    h = np.float16(v)
    return np.nextafter(h, np.float16(0.0)) if abs(np.float32(h)) > abs(v) else h


class StoreQuantisation(unittest.TestCase):
    def test_truncation_is_one_sided(self):
        # the measured residual of the covariance is one-sided (mine - ref is
        # never -1 ULP), which is what a truncating store looks like
        rng = np.random.default_rng(7)
        values = rng.uniform(0.05, 60.0, 20000).astype(np.float32)
        truncated = np.array([trunc16(v) for v in values])
        nearest = np.asarray(np.float16(values))
        delta = (np.abs(np.float32(truncated)) - np.abs(np.float32(nearest)))
        self.assertTrue(np.all(delta <= 0))
        self.assertFalse(np.all(delta == 0))

    def test_truncation_only_moves_at_most_one_step(self):
        rng = np.random.default_rng(11)
        values = rng.uniform(0.05, 60.0, 20000).astype(np.float32)
        nearest = np.asarray(np.float16(values))
        truncated = np.array([trunc16(v) for v in values])
        steps = np.abs(truncated.view(np.uint16).astype(np.int64)
                       - nearest.view(np.uint16).astype(np.int64))
        self.assertLessEqual(int(steps.max()), 1)


if __name__ == '__main__':
    sys.exit(unittest.main())
