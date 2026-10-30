"""EXP-026 independent, non-proprietary numerical and layout contracts."""
from pathlib import Path
import sys
import json

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m09_mlp_up as mlp
import m_b6_frame as b6
import m_translated_layer_suite as translated
from types import SimpleNamespace


def test_scalar_integer_oracle_signs_and_bound():
    for sign in (-1,1):
        a=np.full(64,127*sign,np.int8); w=np.full(64,127,np.int8)
        assert mlp.eq.scalar_dot(a,w)==sign*1032256
    assert 64*127*127<2**31


def test_quantization_ties_negative_and_zero():
    a=np.zeros((3,64),np.float16)
    # max=127 gives inv=1: exact .5 ties on both sides, including odd integers.
    a[0,:9]=[127,-127,.5,1.5,2.5,-.5,-1.5,-2.5,0]
    a[2,:4]=[1,-1,np.nextafter(np.float16(.5),np.float16(0)),np.nextafter(np.float16(.5),np.float16(1))]
    codes,scales=mlp.eq.quantize_rows_float(a)
    assert codes[0,:9].tolist()==[127,-127,0,2,2,0,-2,-2,0]
    assert not codes[1].any()
    assert scales[1,0]==np.float32(1/127)
    for i in range(3):
        c,s=mlp.scalar_row(a[i]); assert np.array_equal(c,codes[i]); assert s==scales[i,0]
    assert codes.min()>=-127 and codes.max()<=127


def test_oracle_bias_precision_and_independent_scalar_dots():
    rng=np.random.default_rng(26)
    a=(rng.normal(size=(5,64))*2).astype(np.float16)
    w=(rng.normal(size=(64,256))*.1).astype(np.float32)
    bias=rng.normal(size=256).astype(np.float16).astype(np.float32)
    c,s,wc,ws,acc,y,half=mlp.cpu_oracle(a,w,bias)
    for row in range(5):
        sc,ss=mlp.scalar_row(a[row])
        assert np.array_equal(c[row],sc) and s[row]==ss
        for col in (0,7,31,32,127,255):
            total=sum(int(sc[k])*int(wc[k,col]) for k in range(64))
            assert total==acc[row,col]
            expected=np.float32(np.float32(np.float32(total)*ss)*ws[col])+bias[col]
            assert expected.tobytes()==y[row,col].tobytes()
            assert np.float16(expected).tobytes()==half[row,col].tobytes()
    assert mlp.cpu_oracle(a,w,bias)[-1].tobytes()==half.tobytes()


def test_scalar_weight_encoding_zero_and_signed_endpoints():
    w=np.zeros((64,256),np.float32)
    w[:,1]=127; w[:,2]=-127
    w[:8,3]=[127,-127,.5,1.5,2.5,-.5,-1.5,-2.5]
    wc,ws,_=mlp.eq.quantize_weights(w)
    for col in range(256):
        codes,scale=mlp.scalar_weight_column(w[:,col])
        assert np.array_equal(codes,wc[:,col]) and scale==ws[col]
    assert ws[0]==1 and not wc[:,0].any()
    assert wc[:,1].tolist()==[127]*64 and wc[:,2].tolist()==[-127]*64
    assert wc[:8,3].tolist()==[127,-127,0,2,2,0,-2,-2]


def test_chain_rejects_combination_and_missing_mlp_fixture(tmp_path):
    with pytest.raises(SystemExit):
        mlp.suite.build_chain(str(tmp_path),str(tmp_path/'out'),mlp_int8=True)
    with pytest.raises(SystemExit):
        mlp.suite.build_chain(str(tmp_path),str(tmp_path/'out'),q_int8=True,
                             mlp_int8=True,enc1_mlp=str(tmp_path))


def test_translated_refresh_fails_closed_without_mutating_history(tmp_path,monkeypatch):
    source=tmp_path/'archive';source.mkdir()
    summary='{"layers":{"enc1":{}},"chain":{}}'
    (source/'summary.json').write_text(summary)
    fixture=source/'enc1-fixture';fixture.mkdir()
    (fixture/'fixture.json').write_text('{"outputs":{"out":{}}}')
    old=source/'enc1-contract/sette-0';old.mkdir(parents=True)
    (old/'out-out.bin').write_bytes(b'old!')
    def changed(command,**kwargs):
        out=Path(command[command.index('--out')+1]);out.mkdir(parents=True)
        (out/'out-out.bin').write_bytes(b'new!')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(translated.subprocess,'run',changed)
    with pytest.raises(RuntimeError,match='changed numerically'):
        translated.refresh_sette(source,tmp_path/'refresh',tmp_path/'chain')
    assert (source/'summary.json').read_text()==summary
    assert not (tmp_path/'refresh/summary.json').exists()
    assert (tmp_path/'refresh/refresh-failure.json').is_file()


def test_physical_address_nonuniform_bijection():
    # First chunk's 4096 matrix bytes form two disjoint 2048-byte images;
    # W1's blocked positions span both K blocks without overlap.
    offsets=np.array([[mlp.raw_up_offset(14000,k,j) for j in range(32)] for k in range(64)])
    reference=mlp.lf.lo.sm.woff_table(14000,512,1024,64,32)[:,[mlp.gp(j) for j in range(32)]]
    assert np.array_equal(offsets,reference)
    assert np.unique(offsets).size==2048
    source=np.arange(20000,dtype=np.int64)*73+19
    for k in (0,1,15,16,31,32,63):
        for j in (0,1,2,15,16,31):
            assert source[offsets[k,j]]==source[reference[k,j]]


def test_mlp_mode_is_exclusive():
    b6.validate_precision_modes('fp16','fp16','int8-up')
    with pytest.raises(SystemExit): b6.validate_precision_modes('int8-q','fp16','int8-up')
    with pytest.raises(SystemExit): b6.validate_precision_modes('fp16','int8-expansion','int8-up')


@pytest.mark.parametrize('mode',['int8-up','probe','fp16-up'])
def test_fixture_descriptor_order_and_gpu_activation_contract(tmp_path,mode):
    source=tmp_path/'source';source.mkdir()
    # No activation payload: offline preparation may read weights only.
    record={'layer':'enc1','geometry':{'C':64,'tokens':2},'weights':{
        'offsets':{'w1':0,'b1':16384}},'q_mode':'fp16'}
    (source/'fixture.json').write_text(json.dumps(record))
    (source/'weights.bin').write_bytes(np.zeros(16640,'<f4').tobytes())
    text=('version 1\nresource h2 256\nresource x2 256\nresource weights 66560 weights.bin\n'
          'resource x2b 256\nresource scratch 16\nstage mlp\nshader swin_mlp.cso\n'
          'threads 64 1 1\ngroups 1 1 1\nparams params-mlp.bin 160\n'
          'srv 0 h2\nsrv 1 weights\nsrv 2 x2\nuav 0 x2b\nuav 1 scratch\n')
    (source/'graph.txt').write_text(text)
    out=mlp.prepare(source,tmp_path/mode,mode)
    _,stages,_=mlp.suite._parse_graph(out/'graph.txt')
    for stage in stages:
        for kind in ('srv','uav'):
            regs=[int(w[1]) for w in stage['lines'] if w[0]==kind]
            assert regs==list(range(len(regs)))
    original=[s for s in stages if s['name']=='mlp'][0]
    assert original['lines'][-2:]==[['uav','0','x2b'],['uav','1','scratch']]
    if mode!='probe': assert ['srv','3','mlp_up'] in original['lines']
    if mode=='int8-up':
        assert [s['name'] for s in stages]==['mlp-quant','mlp-up','mlp']
        assert ['srv','0','h2'] in stages[0]['lines']
    assert (source/'graph.txt').read_text()==text
