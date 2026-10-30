"""Nonproprietary operator-oracle, fragment ABI, and instrumentation regressions."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m_k32_probe import exact_dot, korder, ptx_inputs, decode_fragments
from m_k32_ptx_trace import instrument


def test_k16_groups_are_a_partition_not_contiguous_halves():
    first, second = korder(0), korder(1)
    assert first == [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23]
    assert sorted(first + second) == list(range(32))
    assert not set(first) & set(second)


@pytest.mark.parametrize('sign', [1, -1])
def test_integer_oracle_at_binary16_midpoint(sign):
    result = exact_dot([sign, sign / 32], [1, 1 / 64])
    assert result == sign * float(Fraction(2049, 2048))
    assert np.float16(result) == sign


def test_integer_oracle_cancellation_and_subnormal_products():
    assert exact_dot([448, -448, 1 / 512], [448, 448, 1 / 512], .5) == .5 + 2**-18
    assert exact_dot([1 / 512] * 32, [1 / 512] * 32) == 2**-13


@pytest.mark.parametrize('a,b,c', [([], [], 0), ([1], [1, 1], 0),
                                   ([.1], [1], 0), ([float('nan')], [1], 0),
                                   ([1], [1], .1)])
def test_integer_oracle_rejects_invalid_operands(a, b, c):
    with pytest.raises(ValueError):
        exact_dot(a, b, c)


def test_fragment_packing_roundtrip_preserves_sign_and_k_order():
    # All rows and columns replicate a dot; unique K entries detect permutation.
    import m_layer_oracle as lo
    a = lo.me.E4M3[np.arange(32)].astype(np.float16)
    b = -a[::-1].copy()
    packed = ptx_inputs([('synthetic', a, b, np.float16(.125))])[0]
    raw = np.concatenate([packed, packed[:, 6:8]], axis=1)
    ac, bc, c, d = decode_fragments(raw)
    assert np.array_equal(lo.me.E4M3[ac], np.broadcast_to(a, (16, 32)))
    assert np.array_equal(lo.me.E4M3[bc], np.broadcast_to(b[:, None], (32, 8)))
    assert np.all(c == np.float16(.125).view(np.uint16))
    assert np.array_equal(c, d)


def test_fragment_packing_rejects_non_e4m3():
    with pytest.raises(ValueError):
        ptx_inputs([('invalid', np.full(32, .1, np.float16), np.ones(32, np.float16), np.float16(0))])


def test_ptx_instrumentation_preserves_instruction_and_register_operands():
    op = ('mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 '
          '{%d0,%d1}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1};')
    text = '.version 8.7\n.visible .entry rrlite_enc1_4x4() {\n' + op + '\nret;\n}'
    traced, site = instrument(text)
    assert traced.count(op) == 1
    assert traced.count('mma.sync') == 1
    assert site['registers']['C'] == ['%c0', '%c1']
    assert 'st.global.u32 [%trace_addr+32], %d0;' in traced
    assert 'TRACE_DONE:' in traced


def test_ptx_instrumentation_fails_without_expected_mma():
    with pytest.raises(ValueError):
        instrument('.visible .entry rrlite_enc1_4x4() { ret; }')


def test_ptx_launcher_does_not_overwrite_existing_evidence(tmp_path, monkeypatch):
    from m_k32_ptx_trace import main
    evidence = tmp_path / 'execution.log'
    evidence.write_text('preserve this')
    monkeypatch.setattr(sys, 'argv', ['trace', '--ptx', 'unused.ptx', '--capture', 'unused',
                                     '--out', str(tmp_path)])
    with pytest.raises(FileExistsError):
        main()
    assert evidence.read_text() == 'preserve this'
