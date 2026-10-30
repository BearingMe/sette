"""M09 enc1 Q: the encoder contract, the integer product and the graph guard.

CPU-only.  The GPU half is `scripts/m09_enc1_q.py run`; what is testable without
hardware is that the encoder and the CPU oracle mean what EXP-023 says they mean.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m09_enc1_q as ex
import m_int8 as q8
import m_layer_suite as suite

F16, F32, F64 = np.float16, np.float32, np.float64


def row(values):
    return np.asarray([values], F16)


# ---- activation encoder ------------------------------------------------------

def test_the_scale_is_the_row_maximum_over_127():
    x = row([1.125, -2.25, 0.5, 0.0])
    codes, scale = ex.quantize_rows_float(x)
    assert scale.shape == (1, 1)
    assert float(scale[0, 0]) == pytest.approx(float(np.float32(2.25) / np.float32(127)))


def test_the_zero_row_is_encodable_and_not_a_division_by_zero():
    codes, scale = ex.quantize_rows_float(row([0.0, 0.0, 0.0, 0.0]))
    assert np.all(codes == 0)
    assert float(scale[0, 0]) == pytest.approx(float(np.float32(1.0) / np.float32(127)))


def test_the_row_maximum_lands_on_the_endpoint():
    codes, scale = ex.quantize_rows_float(row([2.25, -1.0, 0.25, 0.0]))
    assert codes[0, 0] == 127
    assert int(np.abs(codes).max()) == 127


def test_an_exact_tie_rounds_to_even():
    """1.125 with max 2.25 gives the ratio 63.5 exactly: half-to-even keeps 64."""
    x = row([1.125, 2.25, 0.0, 0.0])
    codes, _ = ex.quantize_rows_float(x)
    ratio = np.float32(1.125) * (np.float32(127) / np.float32(2.25))
    assert float(ratio) == 63.5
    assert codes[0, 0] == 64


def test_the_encoder_never_exceeds_half_a_step_plus_float_rounding():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (64, 64)).astype(F16)
    codes, scale = ex.quantize_rows_float(x)
    residual = np.abs(np.asarray(x, F64) - codes.astype(F64) * scale)
    half = np.broadcast_to(scale / 2, residual.shape)
    assert np.all(residual <= half * (1 + 1e-6))


def test_the_encoder_is_deterministic():
    x = np.random.default_rng(3).normal(0, 1, (16, 64)).astype(F16)
    a, sa = ex.quantize_rows_float(x)
    b, sb = ex.quantize_rows_float(x)
    assert np.array_equal(a, b) and np.array_equal(sa, sb)


# ---- weights and packing -----------------------------------------------------

def test_weight_scales_are_per_output_column_and_transposed():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.2, (64, 64)).astype(F16).astype(F64)
    codes, scales, enc = ex.quantize_weights(W)
    assert codes.shape == (64, 64) and scales.shape == (64,)
    assert np.allclose(scales, np.abs(W).max(axis=0) / 127, rtol=1e-6)
    assert int(np.abs(codes).max()) <= 127


def test_the_packed_weight_codes_round_trip():
    import m_layer_fixture as lf
    codes = np.arange(-128, 128, dtype=np.int8).reshape(8, 32)
    packed = np.frombuffer(lf.pack_int8(codes), np.uint8)
    assert np.array_equal(packed.reshape(8, 32).view(np.int8), codes)


# ---- the integer product -----------------------------------------------------

def test_the_gemm_matches_an_independent_scalar_dot():
    rng = np.random.default_rng(11)
    codes = rng.integers(-127, 128, (6, 16)).astype(np.int8)
    wcodes = rng.integers(-127, 128, (16, 5)).astype(np.int8)
    row_scale = np.linspace(0.01, 0.02, 6, dtype=F32)
    wscale = np.linspace(0.001, 0.002, 5, dtype=F32)
    y, acc = ex.gemm_int8(codes, row_scale, wcodes, wscale)
    for i in range(6):
        for j in range(5):
            want = (ex.scalar_dot(codes[i], np.ascontiguousarray(wcodes.T)[j])
                    * float(row_scale[i]) * float(wscale[j]))
            assert float(y[i, j]) == pytest.approx(float(F16(want)), rel=1e-6)


def test_the_accumulator_bound_is_a_k_bound():
    assert 64 * 127 * 127 < 2 ** 31
    assert 200000 * 127 * 127 > 2 ** 31


def test_the_epilogue_is_the_float32_pair_not_float64():
    """The kernel multiplies twice in f32; a f64 epilogue would be a different contract."""
    codes = np.full((1, 1), 127, np.int8)
    wcodes = np.full((1, 1), 127, np.int8)
    row_scale = np.array([np.float32(0.017716535)], F32)
    wscale = np.array([np.float32(0.002246)], F32)
    y, acc = ex.gemm_int8(codes, row_scale, wcodes, wscale)
    assert int(acc[0, 0]) == 127 * 127
    f32 = np.float32(np.float32(np.float32(acc[0, 0]) * row_scale[0]) * wscale[0])
    f64 = np.float32(np.float64(acc[0, 0]) * np.float64(row_scale[0])
                     * np.float64(wscale[0]))
    assert float(y[0, 0]) == float(F16(f32))
    # the two routes are not the same real number; the contract is the f32 pair
    assert float(f32) != float(f64) or pytest.skip('this case rounds identically')


# ---- graph mode --------------------------------------------------------------

def test_the_hybrid_chain_refuses_to_guess_the_int8_fixture(tmp_path):
    with pytest.raises(SystemExit):
        suite.build_chain(str(tmp_path), str(tmp_path / 'chain'), q_int8=True,
                          enc1_int8=None)


def test_the_e4m3_table_used_for_decoding_matches_the_conformance_one():
    import m_conformance as mc
    for code in (0, 1, 0x38, 0x40, 0x7E, 0xB8, 0x7F):
        a, b = ex.E4M3[code], mc.e4m3_value(code)
        assert (np.isnan(a) and np.isnan(b)) or a == b
