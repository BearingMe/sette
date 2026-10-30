"""EXP-028 matrix inventory, bounded family selection and fallback contracts."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m09_cumulative as cu


def fixture(path,C=64,label='enc1',merge=96,cin=0,heads=None):
    path.mkdir(parents=True)
    g={'C':C,'heads':heads or C//32,'NPM':merge,'CIN':cin,'W':16,'H':8,'GW':24,'GH':16,
       'sx':2,'sy':2,'lowW':8,'lowH':4,'tokens':384}
    Q=g['heads']*32
    shapes={'g1':C,'bo':C,'g2':C,'b2':C,'q':C*Q,'v':C*Q,'wo':Q*C,
            'w1':C*4*C,'b1':4*C,'w2':4*C*C,'mergeW':4*C*merge,'mergeB':merge,
            'expandW':cin*4*C,'expandB':4*C if cin else 0,'bias':g['heads']*256}
    offsets={};cursor=0
    for key,n in shapes.items(): offsets[key]=cursor;cursor+=n
    (path/'weights.bin').write_bytes(np.zeros(cursor,'<f4').tobytes())
    r={'layer':label,'geometry':g,'weights':{'offsets':offsets},'outputs':{
        'out':{'bytes':16*8*C,'geometry':[16,8,C],'kind':'e4m3'}}}
    if merge:r['outputs']['merge']={'bytes':8*4*merge,'geometry':[8,4,merge],'kind':'e4m3'}
    (path/'fixture.json').write_text(json.dumps(r))
    resources={'weights':cursor*4,'h1':384*C*2,'h2':384*C*2,'res':384*C*2,
        'q':384*Q*2,'v':384*Q*2,'o':384*Q*2,'x0':384*C*2,'x1':384*C*2,
        'x2':384*C*2,'x2b':384*C*2,'out':16*8*C,'scratch':16,'in':16*8*C}
    if merge:resources['merge']=8*4*merge
    if cin:resources.update({'low':8*4*cin,'skip':16*8*C,'expand':8*4*4*C*2})
    lines=['version 1']+[f'resource {n} {size}'+(' weights.bin' if n=='weights' else '') for n,size in resources.items()]
    names=(['expand'] if cin else [])+['prepare','norm1','q','v','attention','wo','norm2','mlp','store']+(['merge'] if merge else [])
    for name in names:
        (path/f'params-{name}.bin').write_bytes(bytes(160))
        kind={'q':'matmul','v':'matmul','wo':'matmul','expand':'matmul','norm1':'norm','norm2':'norm'}.get(name,name)
        lines += [f'stage {name}',f'shader swin_{kind}.cso','threads 64 1 1','groups 1 1 1',
                  f'params params-{name}.bin 160','srv 0 h1','srv 1 weights','srv 2 res','uav 0 x1','uav 1 scratch']
    (path/'graph.txt').write_text('\n'.join(lines)+'\n')
    for role,entry in r['outputs'].items():(path/f'ref-{role}.bin').write_bytes(bytes(entry['bytes']))
    return path


@pytest.mark.parametrize('C',[64,96,128])
@pytest.mark.parametrize('batch',list(cu.BATCHES))
def test_batch_reuses_existing_matrix_shaders_and_gpu_activations(tmp_path,C,batch):
    source=fixture(tmp_path/'source',C)
    out=tmp_path/'out';r=cu.prepare_layer(source,out,batch)
    expected=2+(batch!='qv')+2*(batch in ('mlp','spatial'))+(batch=='spatial')
    assert len(r['int8_operations'])==expected
    _,stages,dumps=cu.up.suite._parse_graph(out/'graph.txt')
    for op in r['int8_operations']:
        s=next(v for v in stages if v['name']==op['name'])
        assert ['shader',op['shader']] in s['lines']
        assert op['shader'] in ('swin_mlp_up_q8.cso','swin_mlp_down_q8.cso')
        assert op['K']*127*127<2**24
    for s in stages:
        for kind in ('srv','uav'):
            regs=[int(w[1]) for w in s['lines'] if w[0]==kind]
            assert regs==list(range(len(regs)))
    assert next(s for s in stages if s['name']=='attention')['lines'][0]==['shader','swin_attention.cso']
    if batch in ('mlp','spatial'):
        assert not any(s['name']=='mlp' for s in stages)
        assert next(op for op in r['int8_operations'] if op['name']=='mlp-up')['N']==4*C
        assert next(op for op in r['int8_operations'] if op['name']=='mlp-down')['K']==4*C


def test_complete_inventory_and_workload_denominator(tmp_path):
    for label in cu.up.suite.LABELS:
        fixture(Path(cu.up.suite.fixture(str(tmp_path),label)),label=label,
                merge=96 if label in ('enc1','enc2') else 0,cin=96 if label in ('dec2','dec1') else 0)
    inv=cu.inventory(tmp_path,'spatial',validated=True)
    assert inv['count']==76 and inv['learned_count']==56 and inv['dynamic_count']==20
    assert inv['int8_count']==55
    assert inv['macs']==sum(r['M']*r['K']*r['N']*r['instances'] for r in inv['operators'])
    assert all(not r['int8'] for r in inv['operators'] if r['kind']=='dynamic' or r['family']=='enc0')
    assert inv['mac_coverage']!=inv['operation_coverage']
    graph=tmp_path/'enc1/graph.txt';graph.write_text(graph.read_text()+'stage omitted-matrix\n')
    with pytest.raises(ValueError,match='omitted'):cu.inventory(tmp_path)


def test_decoder_unpack_is_required_only_in_spatial_batch(tmp_path):
    source=fixture(tmp_path/'source',96,'dec2',merge=0,cin=128)
    for batch in ('mlp','spatial'):
        r=cu.prepare_layer(source,tmp_path/batch,batch)
        assert any(o['name']=='expand' for o in r['int8_operations'])==(batch=='spatial')
        if batch=='spatial':
            op=next(o for o in r['int8_operations'] if o['name']=='expand')
            assert (op['M'],op['K'],op['N'])==(32,128,384)


def test_rectangular_qv_and_output_projection_contract(tmp_path):
    source=fixture(tmp_path/'source',96,heads=4)
    r=cu.prepare_layer(source,tmp_path/'out','output')
    assert [(o['K'],o['N']) for o in r['int8_operations']]==[(96,128),(96,128),(128,96)]


def test_committed_inventory_is_complete_and_weighted():
    p=cu.ROOT/'docs/experiments/EXP-028-matrix-inventory.json'
    r=json.loads(p.read_text());rows=r['operators']
    assert len(rows)==76 and len({(a[0],a[2]) for a in rows})==76
    assert sum(a[7] for a in rows)==55
    assert sum(a[3]*a[4]*a[5]*a[6] for a in rows)==r['totals']['macs']
    assert sum(a[3]*a[4]*a[5]*a[6] for a in rows if a[7])==r['totals']['int8_macs']
    assert all(a[2] in r['contracts'] for a in rows)
    assert all(not a[7] for a in rows if a[2] in ('enc0','qkt','pv'))


def test_cumulative_config_rejects_fp16_matrix_fallback(tmp_path):
    for label in cu.up.suite.LABELS:
        source=fixture(tmp_path/('src-'+label.replace('#','-')),label=label,merge=0)
        cu.prepare_layer(source,Path(cu.up.suite.fixture(str(tmp_path/'layers'),label)),'qv')
    assert cu.audit_config(tmp_path/'layers','qv')==20
    p=tmp_path/'layers/enc1/graph.txt';text=p.read_text()
    p.write_text(text.replace('shader swin_mlp_up_q8.cso','shader swin_matmul.cso',1))
    with pytest.raises(ValueError,match='fallback'):cu.audit_config(tmp_path/'layers','qv')


def test_unpack_reference_retains_negative_zero():
    raw=np.array([0,128,1,129,126,254],np.uint8)
    halves=cu.mw.e4m3_decode(raw).astype('<f2').view('<u2')
    assert halves[0]==0 and halves[1]==0x8000
    assert np.array_equal((halves>>15).astype(np.uint8),raw>>7)
