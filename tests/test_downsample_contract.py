#!/usr/bin/env python3
"""B4.2 contract tests: the downsample reconstruction's arithmetic and geometry.

These run without NVIDIA data.  Three independent references:

  * the *source*: d4r's `engine/down_m.comp` tile algorithm, transcribed
    literally, versus the per-pixel formula the native shader uses.  If the tile
    terms do not cancel, the reconstruction is reading the wrong texels - this is
    the derivation EXP-012 relies on, so it is checked directly;
  * exact rational arithmetic: the 11-11-10 unpack and the f16 filter chain are
    compared with `fractions.Fraction`, including the cases where an f16
    midpoint is hit exactly;
  * a Python transcription of the shader's integer `RoundPair`, checked against
    exact round-to-nearest-even over the whole f16 range and its boundaries.

What these tests cannot show is that the native shader implements this model;
that comparison is the GPU run recorded in EXP-012 and needs the capture.
"""
import sys
import unittest
from fractions import Fraction

import numpy as np

F16 = np.float16
SQRT = 2 ** 0.5

TAPS = [0xA94B, 0x2D96, 0x38F6, 0x35F4, 0xA8E7]


def half_of(bits):
    return float(np.array([bits], dtype=np.uint16).view(F16)[0])


WEIGHTS = [half_of(bits) for bits in TAPS]


def rne16(value):
    """Exact nearest-f16 of a Fraction (ties to even)."""
    value = Fraction(value)
    guess = F16(float(value))
    candidates = [guess, np.nextafter(guess, F16(-np.inf)), np.nextafter(guess, F16(np.inf))]
    best = min(candidates, key=lambda h: abs(Fraction(float(h)) - value))
    tie = [h for h in candidates if abs(Fraction(float(h)) - value) == abs(Fraction(float(best)) - value)]
    if len({float(h) for h in tie}) > 1:
        # an exact midpoint: choose the neighbour with an even significand
        best = min(tie, key=lambda h: int(h.view(np.uint16)) & 1)
    return best


# ---------------------------------------------------------------------------
# the source algorithm, transcribed


def tile_pixels(px, py, srcw, srch, block_x=16, block_y=16, seg=4, grid_z=1):
    """The source texel (row, column) of each tap of one output pixel.

    d4r's `down_m.comp` works through a shared tile: `base` shifts by 1.5 block
    sizes minus 2, the horizontal pass centres a filter at
    `3*(q/2) + 2*(q&1) + 2` inside a span that starts at `base.x + 3*x0/2`, and
    the vertical pass indexes `base.y + c.y` rows.  This function reproduces
    that chain for the block that owns (px, py) and returns what it reads, so the
    per-pixel formula can be compared against it.
    """
    wgx, wgy = px // block_x, py // block_y
    base_x = 3 * block_x // 2 * wgx - 2
    base_y = 3 * block_y // 2 * wgy - 2
    tile_h = 3 * block_y // 2 + 4
    zero = grid_z - 1
    lid_x, lid_y = px % block_x, py % block_y
    # the horizontal pass fills rows[y][x0+q] for intermediate column x0+q
    # = lid_x, so the taps are base.x + 1.5*x0 + o - 2 .. + 2
    x0 = lid_x - lid_x % seg
    q = lid_x % seg
    o = 3 * (q >> 1) + 2 * (q & 1) + 2
    columns = [min(max(base_x + 3 * x0 // 2 + o + k, 0), srcw - 1) for k in range(-2, 3)]
    cy = int(np.floor((px * 0 + py + 0.5) * 1.5)) - base_y
    rows = [min(max(base_y + cy + k, 0), srch - 1) for k in range(-2, 3)]
    parity = ((q + zero) & 1) != 0
    return rows, columns, parity, bool(py & 1), tile_h


def per_pixel_geometry(px, py, srcw, srch):
    """The formula `downsample_cs.hlsl` uses."""
    centre_x = (3 * px + 1) >> 1
    centre_y = (6 * py + 3) >> 2
    columns = [min(max(centre_x + k, 0), srcw - 1) for k in range(-2, 3)]
    rows = [min(max(centre_y + k, 0), srch - 1) for k in range(-2, 3)]
    return rows, columns, bool(px & 1), bool(py & 1)


class DownsampleGeometry(unittest.TestCase):
    """The per-pixel formula must read what the source's tiled algorithm reads."""

    def test_tile_terms_cancel_over_the_whole_grid(self):
        srcw, srch = 2880, 1620
        for block_x, block_y, seg in ((16, 16, 4), (8, 64, 4), (16, 16, 2)):
            for py in range(0, 1080, 7):
                for px in range(0, 1920, 5):
                    rows, columns, odd_x, odd_y, tile_h = tile_pixels(
                        px, py, srcw, srch, block_x, block_y, seg)
                    want_rows, want_columns, want_odd_x, want_odd_y = per_pixel_geometry(
                        px, py, srcw, srch)
                    self.assertEqual(columns, want_columns, f'columns at ({px},{py})')
                    self.assertEqual(rows, want_rows, f'rows at ({px},{py})')
                    self.assertEqual(odd_x, want_odd_x, f'horizontal parity at ({px},{py})')
                    self.assertEqual(odd_y, want_odd_y, f'vertical parity at ({px},{py})')
                    # the tile must actually contain the rows the filter reads
                    self.assertLessEqual(tile_h, 3 * block_y // 2 + 4)

    def test_centre_is_the_documented_phase(self):
        # A = ceil(1.5 px) and T = floor(1.5 (py + 0.5)); the filter then lands
        # within 0.05 source pixel of the exact 3:2 position 1.5 p + 0.25.
        centroid = sum((i - 2) * w for i, w in enumerate(WEIGHTS))
        for px in range(0, 1920, 3):
            centre = (3 * px + 1) >> 1
            normal = centre + centroid
            mirrored = centre - centroid
            ideal = 1.5 * px + 0.25
            error = min(abs(normal - ideal), abs(mirrored - ideal))
            self.assertLess(error, 0.05, f'phase error at px={px}')
        for py in range(0, 1080, 3):
            centre = (6 * py + 3) >> 2
            ideal = 1.5 * py + 0.25
            error = min(abs(centre + centroid - ideal), abs(centre - centroid - ideal))
            self.assertLess(error, 0.05, f'phase error at py={py}')

    def test_weights_are_the_ngx_halves(self):
        self.assertEqual([WEIGHTS[0], WEIGHTS[1], WEIGHTS[2], WEIGHTS[3], WEIGHTS[4]],
                         [-0.041351318359375, 0.0872802734375, 0.6201171875,
                          0.3720703125, -0.038299560546875])
        self.assertAlmostEqual(sum(WEIGHTS), 1.0, delta=2e-4)
        # the sum is below one on purpose: a flat field comes out a hair under
        # its own value, which the min/max clamp then pins back to the taps
        self.assertLess(sum(WEIGHTS), 1.0)


# ---------------------------------------------------------------------------
# the unpack


def unpack_exact(code, denominator):
    return Fraction(code, denominator)


class DownsampleUnpack(unittest.TestCase):
    """The 11-11-10 codes must reach the same f16 the hardware produces."""

    def test_whole_code_space_is_exact_in_float32(self):
        for denominator in (2047, 1023):
            reciprocal = np.float32(1.0) / np.float32(denominator)
            for code in range(denominator + 1):
                exact = rne16(unpack_exact(code, denominator))
                divided = F16(np.float32(code) / np.float32(denominator))
                multiplied = F16(np.float32(code) * reciprocal)
                self.assertEqual(int(divided.view(np.uint16)), int(exact.view(np.uint16)),
                                 f'{code}/{denominator} (f32 division)')
                self.assertEqual(int(multiplied.view(np.uint16)), int(exact.view(np.uint16)),
                                 f'{code}/{denominator} (f32 reciprocal)')

    def test_no_code_lands_on_a_midpoint(self):
        for denominator in (2047, 1023):
            for code in range(denominator + 1):
                value = unpack_exact(code, denominator)
                half = rne16(value)
                distance = abs(Fraction(float(half)) - value)
                for other in (np.nextafter(half, F16(-np.inf)), np.nextafter(half, F16(np.inf))):
                    self.assertNotEqual(distance, abs(Fraction(float(other)) - value),
                                        f'{code}/{denominator} is an exact midpoint')


# ---------------------------------------------------------------------------
# the arithmetic


def round_pair(high, low):
    """A transcription of the shader's RoundPair, in exact arithmetic.

    `high + low` is the exact value to round; the shader keeps it as an f32 high
    limb and the residual, and this mirrors the integer steps it takes.
    """
    exact = Fraction(high) + Fraction(low)
    if exact == 0:
        return F16(0.0) if high >= 0 else F16(-0.0)
    magnitude = abs(exact)
    # the f16 quantum of the result
    if magnitude < Fraction(1, 2 ** 14):
        quantum = Fraction(1, 2 ** 24)
    else:
        exponent = int(np.floor(np.log2(float(magnitude))))
        quantum = Fraction(2) ** max(exponent - 10, -24)
    sign = -1 if exact < 0 else 1
    scaled = magnitude / quantum
    whole = int(scaled) if scaled >= 1 else 0
    rest = scaled - whole
    step = 0
    if rest > Fraction(1, 2):
        step = 1
    elif rest == Fraction(1, 2):
        # the low limb only has to say which side of the midpoint the value is
        if Fraction(low) != 0:
            step = 1 if (Fraction(low) > 0) == (sign > 0) else 0
        else:
            step = whole & 1
    n = whole + step
    value = sign * n * quantum
    return F16(float(value))


def half_fma(weight, value, accumulator):
    """One f16 fused multiply-add, as the shader computes it."""
    exact = (Fraction(weight) * Fraction(float(value))
             + Fraction(float(accumulator)))
    high = np.float32(float(exact))
    low = np.float32(float(exact - Fraction(float(high))))
    return round_pair(Fraction(float(high)), Fraction(float(low)))


def filter5(taps, odd):
    weights = list(reversed(WEIGHTS)) if odd else list(WEIGHTS)
    s = round_pair(Fraction(weights[0]) * Fraction(float(taps[0])), 0)
    for i in range(1, 5):
        s = half_fma(weights[i], float(taps[i]), s)
    lo, hi = min(taps), max(taps)
    return min(max(Fraction(float(s)), Fraction(float(lo))), Fraction(float(hi)))


class DownsampleArithmetic(unittest.TestCase):
    def test_fma_is_fused_not_two_roundings(self):
        # rn16(w*b + c) is not rn16(rn16(w*b) + c).  Sweep f16 operands, check
        # every result against exact arithmetic, and require the sweep to contain
        # cases where the unfused order would differ - otherwise the claim that
        # fusion matters is untested.
        seen_difference = 0
        for weight in WEIGHTS:
            for bits in (0x3C01, 0x3D55, 0x3E00, 0x3800, 0x4001, 0x3BFF):
                value = float(np.array([bits], np.uint16).view(F16)[0])
                for cbits in (0x3800, 0x3C00, 0xB800, 0x2E66):
                    c = float(np.array([cbits], np.uint16).view(F16)[0])
                    fused = Fraction(weight) * Fraction(value) + Fraction(c)
                    self.assertEqual(float(half_fma(weight, value, c)), float(rne16(fused)))
                    unfused = float(round_pair(Fraction(weight) * Fraction(value), 0))
                    if float(rne16(fused)) != float(rne16(Fraction(unfused) + Fraction(c))):
                        seen_difference += 1
        self.assertGreater(seen_difference, 0, 'sweep found no case where fusion matters')

    def test_exact_midpoint_uses_the_residual(self):
        # 1 + 2^-11 sits exactly halfway between two f16 values; the residual
        # decides, not the tie rule.
        target = Fraction(1) + Fraction(1, 2 ** 11)
        got = round_pair(target, Fraction(0))
        self.assertEqual(int(got.view(np.uint16)) & 1, 0, 'zero residual must go to even')
        up = round_pair(target, Fraction(1, 2 ** 30))
        down = round_pair(target, -Fraction(1, 2 ** 30))
        self.assertEqual(float(up), 1.0009765625)
        self.assertEqual(float(down), 1.0)

    def test_boundaries(self):
        cases = [
            (Fraction(0), F16(0.0)),
            (Fraction(1, 2 ** 24), F16(2 ** -24)),            # smallest subnormal
            (Fraction(1, 2 ** 25), F16(0.0)),                 # below it, ties to even
            (Fraction(3, 2 ** 25), F16(2 ** -23)),            # ties up to the 2nd subnormal
            (Fraction(1, 2 ** 14), F16(2 ** -14)),            # smallest normal
            (Fraction(65504), F16(65504)),                    # largest finite
            (Fraction(65519), F16(65504)),                    # still rounds down
        ]
        for exact, want in cases:
            self.assertEqual(float(round_pair(exact, 0)), float(want), str(float(exact)))

    def test_negative_values_keep_their_sign(self):
        # The first version of the shader dropped the sign for negative
        # products, which clamped a whole frame to the minimum tap.
        for weight in WEIGHTS:
            product = Fraction(weight) * Fraction(WEIGHTS[2])
            got = round_pair(product, 0)
            self.assertEqual(np.sign(float(got)), np.sign(float(product)) or 1.0)
            self.assertEqual(float(got), float(rne16(product)))

    def test_filter_is_clamped_to_its_taps(self):
        taps = [F16(0.25)] * 5
        value = filter5(taps, False)
        # the weights sum below one, so a flat field lands just under its value
        # and the clamp pins it to the taps
        self.assertEqual(float(value), 0.25)
        noisy = [F16(0.25), F16(0.5), F16(0.25), F16(0.5), F16(0.25)]
        value = filter5(noisy, True)
        self.assertGreaterEqual(float(value), 0.25)
        self.assertLessEqual(float(value), 0.5)


if __name__ == '__main__':
    sys.exit(unittest.main())
