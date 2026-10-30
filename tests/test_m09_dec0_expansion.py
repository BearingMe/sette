"""Independent physical layout and INT8 contract checks; no captured fixtures."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import m09_dec0_expansion as ex
import m09_enc1_q as eq
import m_weights as mw


def test_all_physical_weights_are_a_bijection_and_match_independent_table():
    offsets = np.array([[ex.weight_offset(k,n) for n in range(128)] for k in range(64)])
    assert np.array_equal(np.sort(offsets.ravel()), np.arange(8192))
    assert np.array_equal(offsets, mw.layout_offsets(mw.front_end_layout('dec0')))


def test_nonuniform_layout_and_bias_across_boundaries():
    rows = 963
    # Byte space cannot uniquely label 8192 positions: use two independent
    # nonuniform patterns, plus a full offset bijection check above.
    for seed in (21, 37):
        rng = np.random.default_rng(seed)
        logical = rng.integers(0,127,(rows,64), dtype=np.uint8)
        physical = np.empty(rows*64, np.uint8)
        for row in range(rows):
            for c in range(64):
                physical[ex.token_offset(row,c,rows)] = logical[row,c]
        matrix = rng.integers(0,127,(64,128),dtype=np.uint8)
        blob = np.zeros(8448,np.uint8)
        for k in range(64):
            for n in range(128):
                blob[ex.weight_offset(k,n)] = matrix[k,n]
        bias = (np.arange(128)-64).astype('<f2')
        blob[8192:] = np.frombuffer(bias.tobytes(),np.uint8)
        a,w,b = ex.decode_operands(physical.tobytes(),blob.tobytes(),rows)
        assert np.array_equal(a, mw.e4m3_decode(logical))
        assert np.array_equal(w, mw.e4m3_decode(matrix[:,ex.LIVE]))
        assert np.array_equal(b,bias[ex.LIVE])
        for row in (0,1,479,480,rows//2,rows-1):
            for c in (0,15,16,31,32,47,48,63):
                assert physical[ex.token_offset(row,c,rows)] == logical[row,c]


@pytest.mark.parametrize('peak',[0,2**-9,1.125,2.25,448])
def test_signed_small_zero_endpoint_quantization_and_scalar_dot(peak):
    a = np.zeros((3,64),np.float16)
    a[:, :4] = [peak,-peak,peak/2,-peak/2]
    w = np.linspace(-0.5,0.5,64*80,dtype=np.float32).reshape(64,80)
    bias = np.linspace(-1,1,80,dtype=np.float32)
    c,s,wc,ws,acc,y,h = ex.cpu_reference(a,w,bias)
    assert np.isfinite(s).all() and (s>0).all()
    assert np.max(np.abs(c.astype(np.int16))) <= 127
    if peak: assert c[0,0]==127 and c[0,1]==-127
    else: assert np.all(c==0) and np.array_equal(h,np.broadcast_to(bias.astype(np.float16),h.shape))
    for n in (0,19,20,40,79):
        scalar = sum(int(c[0,k])*int(wc[k,n]) for k in range(64))
        assert scalar == acc[0,n]
        value = np.float32(np.float32(np.float32(scalar)*s[0,0])*ws[n])
        assert y[0,n] == np.float32(value+bias[n])
    assert np.max(np.abs(acc)) <= 64*127*127 < 2**31
    assert np.array_equal(c,eq.quantize_rows_float(a)[0])


def test_reject_nan_source_and_wrong_extents():
    with pytest.raises(ValueError): ex.decode_operands(b'',b'')
    blob = np.zeros(8448,np.uint8)
    tokens = np.zeros(64,np.uint8); tokens[0]=127
    with pytest.raises(ValueError): ex.decode_operands(tokens.tobytes(),blob.tobytes(),1)


def test_zero_weight_columns_and_worst_case_integer_bound():
    a = np.ones((1,64),np.float16)
    w = np.ones((64,80),np.float32); w[:,0]=0
    c,s,wc,ws,acc,y,h = ex.cpu_reference(a,w,np.zeros(80,np.float32))
    assert np.all(wc[:,0]==0) and ws[0]==1 and y[0,0]==0
    assert acc[0,1]==64*127*127
    assert acc[0,1] < 2**24  # exact int32 -> float32 conversion
    assert np.array_equal(h,y.astype(np.float16))


def test_positive_and_negative_ties_to_even():
    a = np.zeros((1,64),np.float16)
    a[0,:3] = [2.25,1.125,-1.125]
    c,_ = eq.quantize_rows_float(a)
    assert c[0,:3].tolist() == [127,64,-64]


def test_dispatch_geometry_stays_within_d3d12_axis_limits(tmp_path):
    fixture = tmp_path/'fixture'; fixture.mkdir()
    for name in ('in-tokens.bin','in-weights.bin','params.bin'):
        (fixture/name).write_bytes(b'\0'*32)
    out = tmp_path/'graph'
    ex.write_graph(fixture,out,np.zeros((64,80),np.int8),
                   np.ones(80,np.float32),np.zeros(80,np.float32))
    groups = [list(map(int,line.split()[1:])) for line in (out/'graph.txt').read_text().splitlines()
              if line.startswith('groups ')]
    assert groups == [[2040,40,1],[2040,1,1],[2040,40,1]]
    assert all(0<axis<=65535 for group in groups for axis in group)
