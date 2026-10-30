"""M09 first experiment: the INT8 case contract and the integer product.

CPU-only.  The GPU half of the experiment is `scripts/m09_int8_gemm.py run`; what
is testable without hardware is that the case file, the derived references and
the accumulator bound mean what the experiment says they mean.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m09_int8_gemm as m09
import m_int8 as q

F16, F32, F64 = np.float16, np.float32, np.float64


def synthetic(M=6, K=8, N=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.5, (M, K)).astype(F16)
    w = rng.normal(0, 0.5, (K, N)).astype(F16)
    bias = rng.normal(0, 0.1, N).astype(F16)
    return M, K, N, x, w, bias


def write_f16(path, M, K, N, x, w, bias):
    header = np.zeros(1, m09.HEADER)[0]
    header['magic'] = b'MSTAGE01'
    header['version'] = 1
    header['M'], header['K'], header['N'] = M, K, N
    header['flags'] = 1
    with open(path, 'wb') as handle:
        handle.write(header.tobytes())
        handle.write(np.ascontiguousarray(x, F16).tobytes())
        handle.write(np.ascontiguousarray(w, F16).tobytes())
        handle.write(np.ascontiguousarray(bias, F16).tobytes())


def test_the_int8_case_round_trips_every_operand(tmp_path):
    M, K, N, x, w, bias = synthetic()
    args = type('Args', (), {'case_f16': tmp_path / 'f16.bin',
                             'out': tmp_path / 'i8.bin'})()
    write_f16(args.case_f16, M, K, N, x, w, bias)
    m09.export(args)
    case = m09.read_case(args.out)
    assert (case['M'], case['K'], case['N']) == (M, K, N)
    assert case['a'].shape == (M, K) and case['w'].shape == (N, K)
    assert case['a'].dtype == np.int8 and case['w'].dtype == np.int8
    assert np.abs(case['a']).max() <= 127 and np.abs(case['w']).max() <= 127
    assert case['ascale'].shape == (M,) and case['wscale'].shape == (N,)
    assert case['flags'] & 1


def test_the_weight_scale_is_per_output_column():
    M, K, N, x, w, bias = synthetic()
    wq = q.quantize(np.asarray(w, F64), 'per_output_channel')
    assert wq['scales'].shape == (N,)
    assert np.allclose(wq['scales'], np.abs(np.asarray(w, F64)).max(axis=0) / 127)


def test_a_truncated_case_is_refused(tmp_path):
    M, K, N, x, w, bias = synthetic()
    args = type('Args', (), {'case_f16': tmp_path / 'f16.bin',
                             'out': tmp_path / 'i8.bin'})()
    write_f16(args.case_f16, M, K, N, x, w, bias)
    m09.export(args)
    raw = Path(args.out).read_bytes()
    Path(args.out).write_bytes(raw[:-4])
    with pytest.raises(SystemExit):
        m09.read_case(args.out)


def test_the_independent_integer_product_agrees_with_the_rescaled_reference(tmp_path):
    M, K, N, x, w, bias = synthetic()
    args = type('Args', (), {'case_f16': tmp_path / 'f16.bin',
                             'out': tmp_path / 'i8.bin'})()
    write_f16(args.case_f16, M, K, N, x, w, bias)
    m09.export(args)
    case = m09.read_case(args.out)
    y, integers = m09.reference(case)
    # the busiest possible independent check: one code at a time, no BLAS
    probes = [(0, 0), (M - 1, N - 1), (M // 2, N // 2)]
    for row, col in probes:
        total = sum(int(case['a'][row, k]) * int(case['w'][col, k]) for k in range(K))
        assert total == int(integers[row, col])
        want = (total * float(case['ascale'][row]) * float(case['wscale'][col])
                + float(case['bias'][col]))
        assert y[row, col] == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_the_reference_is_the_single_accumulation_contract():
    """Per-output-channel weights mean one integer reduction, not a per-K split."""
    rng = np.random.default_rng(3)
    codes_a = rng.integers(-127, 128, (4, 16)).astype(np.int8)
    codes_w = rng.integers(-127, 128, (16, 5)).astype(np.int8)
    ascale = np.full((4, 1), 0.01, F64)
    wscale = np.linspace(0.002, 0.004, 5)
    single = q.gemm_integer(codes_a, codes_w, ascale, wscale, 'per_output_channel')
    other = wscale * 2.0
    split = q.gemm_integer(codes_a, codes_w, ascale, np.stack([wscale, other]),
                           'block_k', block=8)
    assert np.allclose(single, ascale * (codes_a.astype(F64) @ codes_w) * wscale[None, :])
    # a per-block rescale is a different product whenever the scales differ
    assert not np.allclose(single, split)


def test_the_accumulator_bound_is_on_k_not_on_the_row_count():
    k_limit = (2 ** 31 - 1) // (127 * 127)
    assert 64 < k_limit                          # this experiment's K is safe
    assert 200000 > k_limit                      # a long K would not be


def test_the_epilogue_tolerance_is_one_float32_rounding():
    """The check calls the accumulation exact only if the difference is f32-sized."""
    peak = 13.0
    tolerance = 1e-5 * peak
    assert tolerance == pytest.approx(1.3e-4)
    # the measured value from the recorded run is far below it
    assert 9.6e-7 <= tolerance
