"""The merge contract: operand order, per-k32 rounding, and the stored layout."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m_layer_fixture as lf
import m_layer_oracle as lo
import m_model as mm
from m_enc1_merge_probe import (C, K32, NPM, CODE_LUT, decode_output, encode_codes,
                                encode_field, merged_positions, rebuild)


def test_codes_roundtrip_through_the_lookup():
    for code in range(256):
        value = lo.me.E4M3[code]
        if not np.isfinite(value):
            continue
        assert encode_codes(np.float16(value)) == code
    assert CODE_LUT[np.float16(np.nan).view(np.uint16)] == 0x7f


def test_every_stored_merged_token_reads_inside_the_field():
    W, H, sx, sy = 480, 272, 2, 2          # enc1's captured geometry
    MX, MY, xs, ys = merged_positions(W, H, sx, sy)
    assert MX.size == (W // 2) * (H // 2) == 32640
    assert xs.min() >= 0 and ys.min() >= 0
    assert xs.max() < W and ys.max() < H
    # the padded ring is exactly what the shift removes
    assert lf.padded_geometry(W, H, sx, sy) == (488, 280)


def naive(codes, W, H, merge_w, merge_b, group, rows, cols):
    """Independent loop form of the declared merge contract."""
    value = lo.me.E4M3[codes].astype(np.float64)
    out = np.empty((rows, cols, NPM), np.uint8)
    sx = sy = 0
    for MY in range(rows):
        for MX in range(cols):
            acc = merge_b.astype(np.float64).copy()
            for step in range(0, 4 * C, group):
                total = np.zeros(NPM, np.float64)
                for k in range(group):
                    kk = step + k
                    q, c = divmod(kk, C)
                    px, py = 2 * MX + (q & 1), 2 * MY + (q >> 1)
                    total += value[py, px, c] * merge_w[kk].astype(np.float64)
                acc = (total + acc).astype(np.float16).astype(np.float64)
            acc = np.clip(acc, -6.2831855, 6.2831855).astype(np.float32)
            out[MY, MX] = encode_codes(lo.me.q8(acc))
    return out


def test_the_vectorised_rebuild_matches_an_independent_loop():
    rng = np.random.default_rng(20261029)
    W = H = 16
    codes = rng.integers(0x10, 0x20, size=(H, W, C), dtype=np.uint8)
    # random e4m3 weights (finite codes) and f16 bias
    weights = lo.me.E4M3[rng.integers(0x10, 0x20, size=(4 * C, NPM), dtype=np.uint8)]
    weights = np.asarray(weights, np.float32)
    bias = np.asarray(rng.integers(-64, 64, size=NPM) / 64.0, np.float32)
    for group in (K32, 4 * C):
        fast = rebuild(codes, W, H, weights, bias, 0, 0, group)
        reference = naive(codes, W, H, weights, bias, group, H // 2, W // 2)
        assert np.array_equal(fast, reference), f'group={group}'


def test_the_storage_layout_roundtrips_and_the_group_matters():
    rng = np.random.default_rng(7)
    W = H = 16
    codes = rng.integers(0x10, 0x20, size=(H, W, C), dtype=np.uint8)
    field = encode_field(codes, W, H, C)
    assert field.size == W * H * C
    assert np.array_equal(decode_output(field, W, H), codes)
    weights = np.asarray(lo.me.E4M3[rng.integers(0x10, 0x20, size=(4 * C, NPM))], np.float32)
    bias = np.zeros(NPM, np.float32)
    per_k32 = rebuild(codes, W, H, weights, bias, 0, 0, K32)
    single = rebuild(codes, W, H, weights, bias, 0, 0, 4 * C)
    # One f16 rounding per k32 is part of the contract, not an implementation detail.
    assert not np.array_equal(per_k32, single)
    assert mm.LAYERS['enc1'] == (C, 2, NPM, 0)



