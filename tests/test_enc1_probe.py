"""Capture-free coordinate regressions for the enc1 diagnostic probe."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m_enc1_probe import coordinates, field_offsets, select_windows


class Enc1ProbeMapping(unittest.TestCase):
    def setUp(self):
        self.p = SimpleNamespace(W=480, H=272, sx=2, sy=2, abase=0x100000,
                                 inp=0x200000, out=0x300000)

    def test_shifted_four_window_raster(self):
        x, y = coordinates(self.p, 1, 1)
        expected = []
        for wy in (0, 4):
            for wx in (0, 4):
                for row in range(4):
                    for col in range(4):
                        expected.append((6 + wx + col, 6 + wy + row))
        self.assertEqual(list(zip(x.tolist(), y.tolist())), expected)

    def test_raw_offset_includes_plane_and_channel(self):
        x, y = np.array([6, 7]), np.array([6, 6])
        offsets = field_offsets(self.p, x, y, self.p.inp)
        self.assertEqual(int(offsets[0, 0]), 0x100000 + (6 * 480 + 6) * 32)
        self.assertEqual(int(offsets[0, 31]), int(offsets[0, 0]) + 31)
        self.assertEqual(int(offsets[0, 32]), int(offsets[0, 0]) + 272 * 480 * 32)
        self.assertEqual(int(offsets[1, 0]), int(offsets[0, 0]) + 32)

    def test_selection_uses_complete_windows_not_matching_tokens(self):
        p = SimpleNamespace(W=40, H=40, sx=2, sy=2)
        ref = np.zeros((40, 40, 64), np.uint8)
        got = ref.copy()
        x, y = coordinates(p, 2, 2)
        got[y[48:64], x[48:64], :] = 1
        selected = select_windows(p, got, ref)
        self.assertEqual(selected['mismatch'], (2, 2, 3))
        bx, by, w = selected['control']
        x, y = coordinates(p, bx, by)
        sl = slice(16 * w, 16 * w + 16)
        self.assertTrue(np.array_equal(got[y[sl], x[sl]], ref[y[sl], x[sl]]))


if __name__ == '__main__':
    unittest.main()
