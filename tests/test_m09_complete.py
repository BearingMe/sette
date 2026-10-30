"""EXP-029 contracts, independently enumerated layouts and explicit fallbacks."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m09_complete as h
from test_m09_cumulative import fixture


@pytest.mark.parametrize('heads',[2,4])
def test_window_map_independently_enumerated(heads):
    g={'GW':16,'GH':8,'tokens':128,'heads':heads}
    expected=[]
    for y in range(0,8,4):
        for x in range(0,16,4):expected.append([(y+dy)*16+x+dx for dy in range(4) for dx in range(4)])
    assert np.array_equal(h.window_tokens(g),expected)
    assert sorted(h.window_tokens(g).ravel())==list(range(128))


@pytest.mark.parametrize('heads',[2,4])
def test_dynamic_contract_zero_rows_and_integer_dots(heads):
    g={'GW':8,'GH':8,'tokens':64,'heads':heads}
    rng=np.random.default_rng(29)
    q=(rng.integers(-32,33,(64,heads*32))/16).astype('<f2')
    v=(rng.integers(-64,65,q.shape)/16).astype('<f2')
    idx=h.window_tokens(g);q[idx[0]]=0;v[idx[0]]=0
    bias=np.zeros(heads*256,'<f4')
    t,r=h.attention_reference(q,v,bias,g)
    assert r['status']=='PASS'
    assert t['aq'].dtype==np.dtype('<f2')
    assert np.all(t['qc'][:heads]==0) and np.all(t['vc'][:heads]==0)
    assert np.all(t['qs'][:heads]==np.float32(1/127))
    assert np.all(t['vs'][:heads]==np.float32(1/127))
    assert np.all(t['prob'][:heads]==np.float16(1/16))
    for b,row,col in [(heads,0,1),(heads+1,15,31)]:
        qcol=col%16
        want=sum(int(t['qc'][b,row,k])*int(t['qc'][b,qcol,k]) for k in range(32))
        assert t['qacc'][b,row,qcol]==want
        want=sum(int(t['pc'][b,row,k])*int(t['vc'][b,col,k]) for k in range(16))
        assert t['pacc'][b,row,col]==want
        # V column scale is window-local, not one learned/global column scale.
        vv=t['av'][b,col];top=max(abs(float(a)) for a in vv) or 1
        scale=np.float32(np.float32(top)*np.float32(1/127))
        assert scale.tobytes()==t['vs'][b,col].tobytes()


def test_attention_replacement_and_explicit_fallback(tmp_path):
    source=fixture(tmp_path/'source',96,heads=4)
    target=tmp_path/'converted';h.cu.prepare_layer(source,target,'spatial')
    before=(target/'graph.txt').read_text()
    assert 'shader swin_attention.cso' in before
    h.prepare_attention(target);assert h.audit_attention(target)==2
    after=(target/'graph.txt').read_text()
    assert 'shader swin_attention.cso' not in after
    assert 'srv 0 att_pc' in after and 'srv 1 att_vc' in after
    assert 'srv 2 att_ps' in after and 'srv 3 att_vs' in after
    assert (source/'graph.txt').read_text().count('shader swin_attention.cso')==1
    with pytest.raises(ValueError,match='already'):h.prepare_attention(target)
    (target/'graph.txt').write_text(after.replace('shader swin_attention_pv.cso','shader swin_attention.cso'))
    with pytest.raises(ValueError,match='fallback'):h.audit_attention(target)


def test_attention_rejects_unknown_shape(tmp_path):
    source=fixture(tmp_path/'source');target=tmp_path/'converted'
    h.cu.prepare_layer(source,target,'spatial')
    p=target/'fixture.json';r=json.loads(p.read_text());r['geometry']['heads']=3;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='unvalidated'):h.prepare_attention(target)


def test_enc0_cpu_slot_mirror_and_field_permutation():
    colour=np.zeros((4,4,4),'<f4')
    for y in range(4):
        for x in range(4):colour[y,x,:3]=[x/8,y/8,(x+y)/16]
    f=h.enc.features(colour,np.zeros(2048,np.uint16),0,[4,8],[4,4])
    assert f.shape==(4,2,128)
    # First pixel of padded token row 2 is y=4, mirrored to y=2.
    assert f[2,0,1]==f[1,0,1]
    for sub in range(4):
        assert np.all(f[:,:,sub*32+np.flatnonzero(h.enc.SLOT_CHANNEL<0)]==0)
    assert sorted(h.enc.channel_columns())==list(range(64))


def test_enc0_conversion_preserves_original_and_checks_modes(tmp_path):
    d=tmp_path/'enc0';d.mkdir();blob=np.zeros(12416,np.uint8)
    (d/'in-weights.bin').write_bytes(blob.tobytes())
    original='version 2\nshader enc0_cs.cso\n'
    (d/'stage.txt').write_text(original)
    h.prepare_enc0(d)
    text=(d/'stage.txt').read_text()
    assert 'shader enc0_q8_epilogue.cso' in text
    assert 'enc0_int8 enc0-wc.bin enc0-ws.bin enc0-bias.bin' in text
    assert (d/'enc0-wc.bin').stat().st_size==8192
    assert (d/'enc0-ws.bin').stat().st_size==256
    with pytest.raises(ValueError,match='original'):h.prepare_enc0(d)


def test_current_inventory_overlay_preserves_mac_denominator():
    r=json.loads((h.ROOT/'docs/experiments/EXP-028-matrix-inventory.json').read_text())
    current=r['m09_h']
    assert current['int8_operators']==len(r['operators'])==76
    assert current['int8_macs']==current['macs']==r['totals']['macs']
    assert current['mac_coverage']==current['operation_coverage']==1.0
    assert set(current['overrides'])=={a[2] for a in r['operators'] if not a[7]}
