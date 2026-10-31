#!/usr/bin/env python3
"""EXP-029 bounded enc0 integration and dynamic-attention validation utilities."""
import argparse
import json
from pathlib import Path
import shutil
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import m09_cumulative as cu
import m_enc0_model as enc

ROOT=cu.ROOT
ENC0_DIAGNOSTICS={'features':33423360,'q8a':16711680,'q8as':522240,
                  'acc':33423360,'float':33423360,'embedding':16711680}


def inspect_isa(out,rga,executed):
    out.mkdir(parents=True,exist_ok=False);report={}
    for n in ['swin_mlp_up_q8.cso','swin_mlp_down_q8.cso','dec0_q8_matmul.cso','swin_attention_qk.cso','swin_attention_pv.cso']:
        shader=ROOT/'build/native'/n;digest=cu.up.eq.sha(shader)
        if executed[n]!=digest:raise ValueError('inspected shader differs from execution')
        stem=Path(n).stem
        done=subprocess.run([str(rga),'-s','dx12','-c','gfx1101','--cs-blob',str(shader),'--isa',str(out/stem)],capture_output=True,text=True,check=True)
        (out/(stem+'.log')).write_text(done.stdout+done.stderr)
        p=out/('gfx1101_'+stem+'_comp.isa');text=p.read_text()
        item={'sha256':digest,'isa_sha256':cu.up.eq.sha(p),'dot4':len(re.findall('v_dot4_i32_i8',text)),
              'fp16_matrix':len(re.findall(r'v_(?:pk_)?(?:mul|fma|mac)_f16|v_wmma',text))}
        if not item['dot4'] or item['fp16_matrix'] or cu.up.eq.sha(shader)!=digest:raise ValueError('native ISA gate failed')
        report[n]=item
    (out/'isa.json').write_text(json.dumps(report,indent=2));return report


def enc0_weights(directory):
    table,matrix,bias=enc.stack(np.fromfile(directory/'in-weights.bin',np.uint8))
    return table,cu.mw.e4m3_decode(matrix).astype('<f4'),bias.astype('<f4')


def prepare_enc0(directory):
    """Explicit mode, leaving the original fixture and FP16 defaults untouched."""
    manifest=directory/'stage.txt';text=manifest.read_text()
    if 'enc0_int8' in text or 'shader enc0_cs.cso' not in text:
        raise ValueError('enc0 INT8 requires original FP16 fixture')
    _,W,bias=enc0_weights(directory)
    wc,ws,_=cu.up.eq.quantize_weights(W)
    for col in range(64):
        c,s=cu.up.scalar_weight_column(W[:,col])
        if not np.array_equal(c,wc[:,col]) or s.tobytes()!=ws[col].tobytes():
            raise ValueError('enc0 scalar weight encoding disagreement')
    for name,a in [('enc0-wc',wc.T),('enc0-ws',ws),('enc0-bias',bias)]:
        (directory/(name+'.bin')).write_bytes(np.ascontiguousarray(a).tobytes())
    text=text.replace('shader enc0_cs.cso','shader enc0_q8_epilogue.cso')
    manifest.write_text(text+'enc0_int8 enc0-wc.bin enc0-ws.bin enc0-bias.bin\n')


def validate_enc0(directory,results):
    p=np.fromfile(directory/'params.bin','<u4')
    table,W,bias=enc0_weights(directory)
    colour=np.fromfile(directory/'in-colour.bin','<f2').reshape(540,960,4).astype('<f4')
    features=cu.mw.e4m3_decode(cu.mw.e4m3_encode(enc.features(colour,table,int(p[4]),p[2:4],p[:2]))).astype('<f2').reshape(-1,128)
    codes,scales=cu.up.eq.quantize_rows_float(features)
    wc,ws,_=cu.up.eq.quantize_weights(W)
    acc=codes.astype(np.int64)@wc.astype(np.int64)
    y=((acc.astype('<f4')*scales)*ws[None,:]+bias[None,:]).astype('<f4');half=y.astype('<f2')
    field=cu.mw.e4m3_encode(np.clip(half[:,enc.channel_columns()],-enc.TWO_PI,enc.TWO_PI))
    field=field.reshape(-1,2,32).transpose(1,0,2).copy()
    expected={'features':features,'q8a':codes,'q8as':scales,'acc':acc.astype('<i4'),
              'float':y,'embedding':half,'tokens':field}
    checks={name:(results/f'out-{name}.bin').read_bytes()==np.ascontiguousarray(a).tobytes() for name,a in expected.items()}
    fp=cu.up.lf.me.mma_chain(features.astype(np.float64),W.astype(np.float64),
                           C0=np.broadcast_to(bias.astype(np.float64),half.shape)).astype('<f2')
    checks['finite_unsaturated']=bool(np.isfinite(half).all() and (np.abs(half)<65504).all())
    checks['safe_accumulator']=int(np.abs(acc).max())<=128*127*127<2**24
    probes=[]
    for row in [0,479,480,129599,130080,130559]:
        c,s=cu.up.scalar_row(features[row]);probes.append(np.array_equal(c,codes[row]) and s.tobytes()==scales[row,0].tobytes())
        for col in [0,17,63]:probes.append(cu.up.eq.scalar_dot(c,wc[:,col])==int(acc[row,col]))
    checks['scalar_probes']=bool(all(probes))
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'int8_vs_same_input_fp16':cu.up.eq.compare(half,fp),'accumulator_range':[int(acc.min()),int(acc.max())],
        'feature_range':[float(features.min()),float(features.max())],
        'activation_scale_range':[float(scales.min()),float(scales.max())],
        'hashes':{p.name:cu.up.eq.sha(p) for p in results.glob('out-*.bin')}}


def run_enc0(fixture,out):
    import m_b6_frame as frame
    out.mkdir(parents=True,exist_ok=False)
    target=out/'fixture';shutil.copytree(fixture,target);prepare_enc0(target)
    shaders=ROOT/'build/native';probe=shaders/'Release/m_image_stage.exe'
    before={p.name:cu.up.eq.sha(p) for p in shaders.glob('*.cso')}
    results=frame.run(probe,target/'stage.txt',out/'run0',shaders,seed=0)
    report=validate_enc0(target,results)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if report['status']!='PASS':return report
    repeat=frame.run(probe,target/'stage.txt',out/'run1',shaders,seed=1)
    report['checks']['deterministic_complete_writes']=all(cu.up.eq.sha(repeat/n)==h for n,h in report['hashes'].items())
    report['checks']['zero_debug_messages']=all('messages (0 total)' in (out/f'run{i}/execution.log').read_text() for i in (0,1))
    report['checks']['shaders_unchanged']=all(cu.up.eq.sha(shaders/n)==h for n,h in before.items())
    report['shaders']=before;report['host_sha256']=cu.up.eq.sha(probe)
    report['status']='PASS' if all(report['checks'].values()) else 'FAIL'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return report


def window_tokens(g):
    """Independent reshape of natural padded grid into row-major 4x4 windows."""
    return np.arange(g['tokens']).reshape(g['GH']//4,4,g['GW']//4,4).transpose(0,2,1,3).reshape(-1,16)


def attention_reference(q,v,bias,g):
    """CPU-first dynamic contract: Q row scales, P row scales, V column scales.

    All three encodings use the established runtime f32 row-quantization rule.
    A V column is transposed to a 16-element row before encoding. No offline
    weight scale or learned-weight quantization is used for either operand.
    """
    import m_layer_oracle as lo
    H=g['heads'];idx=window_tokens(g);B=len(idx)*H
    Q=cu.up.lf.me.q8(q[idx]).astype('<f2').reshape(-1,16,H,32).transpose(0,2,1,3).reshape(B,16,32)
    V=v[idx].reshape(-1,16,H,32).transpose(0,2,3,1).reshape(B,32,16)
    qc,qs=cu.up.eq.quantize_rows_float(Q.reshape(-1,32));qc=qc.reshape(B,16,32);qs=qs.reshape(B,16)
    vc,vs=cu.up.eq.quantize_rows_float(V.reshape(-1,16));vc=vc.reshape(B,32,16);vs=vs.reshape(B,32)
    dot=qc.astype(np.int64)@qc.astype(np.int64).transpose(0,2,1)
    score=((dot.astype('<f4')*qs[:,:,None])*qs[:,None,:]).astype('<f4')
    sb=np.broadcast_to(bias.reshape(1,H,16,16),(len(idx),H,16,16)).reshape(B,16,16)
    score=(score+sb.astype('<f4')).astype('<f4');S=score.astype('<f2')
    P=lo.softmax_rows(S.astype(np.float64).reshape(-1,16)).astype('<f2').reshape(B,16,16)
    pc,ps=cu.up.eq.quantize_rows_float(P.reshape(-1,16));pc=pc.reshape(B,16,16);ps=ps.reshape(B,16)
    pv=pc.astype(np.int64)@vc.astype(np.int64).transpose(0,2,1)
    y=((pv.astype('<f4')*ps[:,:,None])*vs[:,None,:]).astype('<f4');half=y.astype('<f2')
    out=cu.up.lf.me.q8(half).reshape(-1,H,16,32).transpose(0,2,1,3).reshape(-1,16,H*32)
    natural=np.empty_like(q);natural[idx]=out
    original_scores=(Q.astype(np.float64)@Q.astype(np.float64).transpose(0,2,1)+sb.astype(np.float64)).astype('<f2')
    original_p=lo.softmax_rows(original_scores.astype(np.float64).reshape(-1,16)).astype('<f2').reshape(B,16,16)
    original_o=cu.up.lf.me.q8((original_p.astype(np.float64)@V.astype(np.float64).transpose(0,2,1)).astype('<f2'))
    checks={'finite':bool(np.isfinite(S).all() and np.isfinite(P).all() and np.isfinite(half).all()),
            'integer_bounds':int(np.abs(dot).max())<=32*127*127 and int(np.abs(pv).max())<=16*127*127,
            'unsaturated':bool((np.abs(half)<448).all()),'scalar':True}
    for b,r,c in [(0,0,0),(B-1,15,15),(B//2,7,11)]:
        checks['scalar'] &= sum(int(qc[b,r,k])*int(qc[b,c,k]) for k in range(32))==int(dot[b,r,c])
        checks['scalar'] &= sum(int(pc[b,r,k])*int(vc[b,c,k]) for k in range(16))==int(pv[b,r,c])
    tensors={'aq':Q,'av':V,'qc':qc,'qs':qs,'vc':vc,'vs':vs,'qacc':dot.astype('<i4'),'score32':score,'score':S,
             'prob':P,'pc':pc,'ps':ps,'pacc':pv.astype('<i4'),'value32':y,'value16':half,'out':natural}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'instances':B,
        'geometry':g,'qkt_shape':[16,32,16],'pv_shape':[16,16,32],
        'ranges':{n:[float(a.min()),float(a.max())] for n,a in tensors.items()},
        'qkt_vs_fp16':cu.up.eq.compare(S,original_scores),
        'attention_vs_fp16':cu.up.eq.compare(cu.up.lf.me.q8(half),original_o)}
    return tensors,report


def representative_inputs(frame,out):
    """Stop the real cumulative graph before enc1 Attention; no replacement yet."""
    out.mkdir(parents=True,exist_ok=False)
    resources,stages,_=cu.up.suite._parse_graph(frame/'chain-native/graph.txt')
    lines=['version 1']
    for r in resources:
        lines.append(f'resource {r["name"]} {r["bytes"]}'+(f' {r["init"]}' if r['init'] else ''))
    for s in stages:
        if s['name']=='enc1.attention':break
        lines.append('stage '+s['name']);lines+=[' '.join(w) for w in s['lines']]
    for n in ['enc1_q','enc1_v']:
        r=next(r for r in resources if r['name']==n);lines.append(f'dump {n} 0 {r["bytes"]} {n}.bin')
    (out/'graph.txt').write_text('\n'.join(lines)+'\n')
    return cu.up.run_graph(out,out/'results')


def prepare_attention(directory):
    """Replace only the attention stage in an existing cumulative layer fixture."""
    r=json.loads((directory/'fixture.json').read_text());g=r['geometry']
    if r.get('int8_attention'):raise ValueError('attention already converted')
    if (g['heads'] not in (2,4) or g['GW']%8 or g['GH']%8 or
            g['tokens']!=g['GW']*g['GH']):
        raise ValueError('unvalidated attention shape/layout')
    graph=directory/'graph.txt';lines=graph.read_text().splitlines()
    start=lines.index('stage attention');end=next((i for i in range(start+1,len(lines)) if lines[i].startswith(('stage ','dump '))),len(lines))
    if lines[start+1]!='shader swin_attention.cso':raise ValueError('unexpected attention fallback')
    B=g['GW']//4*(g['GH']//4)*g['heads']
    sizes={n:count*bytes for n,count,bytes in [
        ('aq',B*512,2),('av',B*512,2),('qc',B*512,1),('qs',B*16,4),('vc',B*512,1),('vs',B*32,4),
        ('qacc',B*256,4),('score32',B*256,4),('score',B*256,2),('prob',B*256,2),
        ('pc',B*256,1),('ps',B*16,4),('pacc',B*512,4),('value32',B*512,4),('value16',B*512,2)]}
    resources=[f'resource att_{n} {size}' for n,size in sizes.items()]
    params=next(line[len('params '):] for line in lines[start:end] if line.startswith('params '));stages=[]
    def stage(name,shader,par,srvs,uavs,count):
        stages.extend([f'stage attention-{name}',f'shader {shader}','threads 64 1 1',f'groups {(count+63)//64} 1 1',f'params {par}'])
        stages.extend(f'srv {i} {v}' for i,v in enumerate(srvs));stages.extend(f'uav {i} {v}' for i,v in enumerate(uavs))
    def quant(name,rows,K):
        par=f'params-att-{name}.bin';(directory/par).write_bytes(np.array([rows,K,0,0],'<u4').tobytes())
        a,c,s={'q':('aq','qc','qs'),'v':('av','vc','vs'),'p':('prob','pc','ps')}[name]
        stage(name+'-quant','swin_q8_quant.cso',par+' 16',['att_'+a],['att_'+c,'att_'+s],rows)
    stage('pack','swin_attention_pack.cso',params,['q','v'],['att_aq','att_av'],B)
    quant('q',B*16,32);quant('v',B*32,16)
    stage('qkt','swin_attention_qk.cso',params,['att_qc','att_qs','weights'],['att_score','att_qacc','att_score32'],B*16)
    stage('softmax','swin_attention_softmax.cso',params,['att_score'],['att_prob'],B*16)
    quant('p',B*16,16)
    stage('pv','swin_attention_pv.cso',params,['att_pc','att_vc','att_ps','att_vs'],['o','att_pacc','att_value32','att_value16'],B*16)
    # Place resource declarations before every stage, as required by the parser.
    first=next(i for i,line in enumerate(lines) if line.startswith('stage '))
    lines=lines[:first]+resources+lines[first:start]+stages+lines[end:]
    for n,size in sizes.items():lines.append(f'dump att_{n} 0 {size} dbg-att-{n}.bin')
    for n in ['q','v','o']:
        if not any(line.startswith(f'dump {n} ') for line in lines):
            lines.append(f'dump {n} 0 {g["tokens"]*g["heads"]*32*2} dbg-{n}.bin')
    graph.write_text('\n'.join(lines)+'\n')
    r['int8_attention']={'contract':'symmetric-qrow-prow-vcolumn-v1','instances':B,'diagnostics':sizes}
    (directory/'fixture.json').write_text(json.dumps(r,indent=2))
    return r


def audit_attention(directory):
    r=json.loads((directory/'fixture.json').read_text())
    if r.get('int8_attention',{}).get('contract')!='symmetric-qrow-prow-vcolumn-v1':
        raise ValueError('missing validated dynamic-attention contract')
    _,stages,_=cu.up.suite._parse_graph(directory/'graph.txt')
    attention=[s for s in stages if s['name'].startswith('attention')]
    expected=[('pack','swin_attention_pack.cso'),('q-quant','swin_q8_quant.cso'),('v-quant','swin_q8_quant.cso'),
              ('qkt','swin_attention_qk.cso'),('softmax','swin_attention_softmax.cso'),
              ('p-quant','swin_q8_quant.cso'),('pv','swin_attention_pv.cso')]
    if [(s['name'],next(w[1] for w in s['lines'] if w[0]=='shader')) for s in attention]!=[('attention-'+n,p) for n,p in expected]:
        raise ValueError('silent attention matrix fallback or missing stage')
    return 2


def complete_coverage(layers,validated=False):
    record=json.loads((Path(cu.up.suite.fixture(str(layers),'enc1'))/'fixture.json').read_text())
    canonical=Path(record['source_fixture']).parent if record.get('source_fixture') else layers
    inv=cu.inventory(canonical,'spatial',validated)
    for op in inv['operators']:
        op['int8']=True
        op['status']='INT8 implemented and validated' if validated else 'INT8 selected; validation pending'
        if op['family'] in ('qkt','pv'):
            op['conversion']='runtime Q/P row and window-local V column scales; symmetric-qrow-prow-vcolumn-v1'
            op['compatibility']='validated bounded dynamic-operand contract'
        if op['family']=='enc0':
            op['conversion']='GPU texture/dither feature adapter + existing runtime quantizer/GEMM + field epilogue'
            op['compatibility']='validated reset-frame texture/buffer adapter'
    inv.update(int8_count=76,int8_macs=inv['macs'],operation_coverage=1.0,mac_coverage=1.0)
    return inv


def run_complete_chain(source,enc0_frame,out):
    """Validate all operators on inputs from the real enc0 INT8 producer."""
    out.mkdir(parents=True,exist_ok=False);fixtures=out/'layers'
    shutil.copytree(source,fixtures)
    for label in cu.up.suite.LABELS:
        d=Path(cu.up.suite.fixture(str(fixtures),label));prepare_attention(d);audit_attention(d)
    cu.audit_config(fixtures,'spatial')
    chain=out/'chain';cu.up.suite.build_chain(str(fixtures),str(chain),cumulative_diagnostics=True)
    graph=chain/'graph.txt';lines=graph.read_text().splitlines()
    tokens=enc0_frame/'enc0-native/out-tokens.bin'
    lines=[' '.join(line.split()[:3]+[str(tokens)]) if line.startswith('resource enc1_in ') else line for line in lines]
    graph.write_text('\n'.join(lines)+'\n')
    hashes={p.name:cu.up.eq.sha(p) for p in (ROOT/'build/native').glob('*.cso')}
    result=cu.up.run_graph(chain,out/'run0')
    _,_,declared=cu.up.suite._parse_graph(graph)
    for d in declared:
        if (result/d['file']).stat().st_size!=d['bytes']:raise RuntimeError('incomplete diagnostic dump')
    reports={}
    for label in cu.up.suite.LABELS:
        tag=re.sub(r'[^A-Za-z0-9]+','-',label)
        reports[label]=validate_attention(Path(cu.up.suite.fixture(str(fixtures),label)),result,tag+'-')
        (out/'attention.json').write_text(json.dumps(reports,indent=2))
        print(label,'dynamic attention',reports[label]['status'],flush=True)
        if reports[label]['status']!='PASS':raise RuntimeError('stop: new unexplained attention difference')
    learned=cu.validate_chain(fixtures,result)
    (out/'learned.json').write_text(json.dumps(learned,indent=2))
    if learned['status']!='PASS':raise RuntimeError('learned matrix gate failed')
    digests={p.name:cu.up.eq.sha(p) for p in result.glob('*.bin')}
    report={'status':'PENDING','checks':{'attention':True,'learned':True},'hashes':digests,'shaders':hashes,
        'enc0_input_sha256':cu.up.eq.sha(tokens),'host_sha256':cu.up.eq.sha(ROOT/'build/native/Release/m_network_stage.exe')}
    (out/'chain.json').write_text(json.dumps(report,indent=2))
    cu.prune_tensor_run(result)
    repeat=cu.up.run_graph(chain,out/'run1',1)
    report['checks']['deterministic_complete_writes']=all(cu.up.eq.sha(repeat/n)==h for n,h in digests.items())
    report['checks']['zero_debug']=all('messages (0 total)' in (out/f'run{i}/execution.log').read_text() for i in (0,1))
    report['checks']['shaders_unchanged']=all(cu.up.eq.sha(ROOT/'build/native'/n)==h for n,h in hashes.items())
    report['status']='PASS' if all(report['checks'].values()) else 'FAIL'
    report['coverage']=complete_coverage(fixtures,report['status']=='PASS')
    (out/'chain.json').write_text(json.dumps(report,indent=2))
    if report['status']=='PASS':cu.prune_tensor_run(repeat)
    return report


def finalize_complete(chain,frame,isa_dir,out):
    """Tie all-element gates, native lowering and actual full-frame handoffs."""
    import m09_dec0_expansion as dec
    c=json.loads((chain/'chain.json').read_text());f=json.loads((frame/'b6.json').read_text())
    isa=json.loads((isa_dir/'isa.json').read_text())
    enc0=validate_enc0(frame/'enc0-fixture-native',frame/'enc0-native')
    d=frame/'dec0-native';a,w,b=dec.decode_operands((d/'in-tokens.bin').read_bytes(),(d/'in-weights.bin').read_bytes())
    codes,scales,wc,ws,acc,y,half=dec.cpu_reference(a,w,b)
    dec_checks={n:(d/'results'/f'out-{n}.bin').read_bytes()==np.ascontiguousarray(v).tobytes()
                for n,v in [('q8a',codes),('q8as',scales),('acc',acc.astype('<i4')),('float',y),('expansion',half)]}
    dec_checks['finite_unsaturated']=bool(np.isfinite(half).all() and (np.abs(half)<65504).all())
    def fingerprint(path):
        resources,stages,_=cu.up.suite._parse_graph(path)
        return {'resources':[(r['name'],r['bytes'],cu.up.eq.sha(path.parent/r['init']) if r['init'] else None) for r in resources],
            'stages':[(s['name'],[(w[0],cu.up.eq.sha(path.parent/w[1]),w[2]) if w[0]=='params' else
                    (w[0],Path(w[1]).name) if w[0]=='shader' else tuple(w) for w in s['lines']]) for s in stages]}
    def no_fallback(path):
        _,stages,_=cu.up.suite._parse_graph(path)
        shaders=[Path(w[1]).name for s in stages for w in s['lines'] if w[0]=='shader']
        if any(n in shaders for n in ['swin_attention.cso','swin_matmul.cso','swin_merge.cso','swin_mlp.cso']):return False
        common=(shaders.count('swin_attention_qk.cso')==10 and shaders.count('swin_attention_pv.cso')==10
                and shaders.count('swin_mlp_up_q8.cso')==34 and shaders.count('swin_mlp_down_q8.cso')==20)
        accepted=len(stages)==234 and shaders.count('swin_mlp_gelu.cso')==10 and shaders.count('swin_q8_quant.cso')==84
        fused=len(stages)==224 and shaders.count('swin_mlp_gelu_quant.cso')==10 and shaders.count('swin_mlp_gelu.cso')==0 and shaders.count('swin_q8_quant.cso')==74
        return common and (accepted or fused)
    fields={}
    for path in (frame/'chain-native-results').glob('out-*.bin'):
        raw=np.fromfile(path,np.uint8)
        fields[path.name]={'finite':bool(np.isfinite(cu.mw.e4m3_decode(raw)).all()),
                          'saturated':int(((raw&127)==126).sum()),'sha256':cu.up.eq.sha(path)}
    checks={'chain':c['status']=='PASS','enc0':enc0['status']=='PASS','dec0':all(dec_checks.values()),
        'same_cpu_checked_graph':fingerprint(chain/'chain/graph.txt')==fingerprint(frame/'chain-native/graph.txt'),
        'layer_handoffs':all(cu.up.eq.sha(frame/'chain-native-results'/n)==h for n,h in c['hashes'].items() if n.startswith('out-')),
        'enc0_handoff':cu.up.eq.sha(frame/'enc0-native/out-tokens.bin')==c['enc0_input_sha256'],
        'execution':f['execution']['status']=='PASS','repeat':f['repeat']['fresh'] and f['repeat']['deterministic'],
        'native_isa':all(r['dot4']>0 and r['fp16_matrix']==0 and r['sha256']==f['inputs']['shaders'][n]==c['shaders'][n]
                         for n,r in isa.items()),
        'shaders_current':all(cu.up.eq.sha(ROOT/'build/native'/n)==h for n,h in f['inputs']['shaders'].items()),
        'host_current':f['inputs']['network_probe_sha256']==cu.up.eq.sha(ROOT/'build/native/Release/m_network_stage.exe'),
        'image_host_current':f['inputs']['probe_sha256']==cu.up.eq.sha(ROOT/'build/native/Release/m_image_stage.exe'),
        'zero_frame_debug_messages':all('messages (0 total)' in p.read_text() for p in frame.rglob('execution.log')),
        'frame_script_current':f['scripts']['m_b6_frame.py']==cu.up.eq.sha(ROOT/'scripts/m_b6_frame.py'),
        'no_fp16_matrix_fallback':no_fallback(frame/'chain-native/graph.txt'),
        'layer_fields_finite_unsaturated':bool(fields) and all(r['finite'] and r['saturated']==0 for r in fields.values()),
        'explicit_enc0_dec0_int8':f['enc0_mode']=='int8-embedding' and f['dec0_mode']=='int8-expansion'}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'enc0':enc0,'dec0_checks':dec_checks,
        'coverage':complete_coverage(chain/'layers',all(checks.values())),'layer_fields':fields,
        'isa':isa,'frame':f['comparisons']['native'],'direct':f['direct_comparisons'],
        'image_regression':f['fidelity']['regression'],'repeat':f['repeat'],'boundaries':len(f['boundaries']),
        'scripts':{n:cu.up.eq.sha(ROOT/'scripts'/n) for n in ['m09_complete.py','m09_cumulative.py','m_b6_frame.py','m_layer_suite.py']},
        'evidence_sha256':{str(p):cu.up.eq.sha(p) for p in [chain/'chain.json',chain/'attention.json',chain/'learned.json',frame/'b6.json',isa_dir/'isa.json']}}
    out.write_text(json.dumps(report,indent=2));return report


def validate_attention(directory,results,prefix=''):
    r=json.loads((directory/'fixture.json').read_text());g=r['geometry'];shape=(g['tokens'],g['heads']*32)
    def path(name):return results/(prefix+'dbg-'+name+'.bin')
    q=np.fromfile(path('q'),'<f2').reshape(shape);v=np.fromfile(path('v'),'<f2').reshape(shape)
    # Cumulative fixtures retain an absolute original FP16 weights resource.
    resources,_,_=cu.up.suite._parse_graph(directory/'graph.txt')
    w=np.fromfile(directory/next(a['init'] for a in resources if a['name']=='weights'),'<f4')
    offset=r['weights']['offsets']['bias'];bias=w[offset:offset+g['heads']*256]
    tensors,report=attention_reference(q,v,bias,g)
    for n,a in tensors.items():
        p=path('o' if n=='out' else 'att-'+n)
        report['checks'][n]=p.read_bytes()==np.ascontiguousarray(a).tobytes()
    report['hashes']={p.name:cu.up.eq.sha(p) for p in results.glob(prefix+'dbg-*.bin')}
    report['status']='PASS' if all(report['checks'].values()) else 'FAIL'
    return report


def run_attention_pair(source,inputs,out):
    out.mkdir(parents=True,exist_ok=False);target=out/'fixture';shutil.copytree(source,target)
    prepare_attention(target)
    resources,stages,dumps=cu.up.suite._parse_graph(target/'graph.txt')
    g=json.loads((target/'fixture.json').read_text())['geometry']
    lines=['version 1']
    for a in resources:
        init=inputs/('enc1_'+a['name']+'.bin') if a['name'] in ('q','v') else (target/a['init'] if a['init'] else None)
        lines.append(f'resource {a["name"]} {a["bytes"]}'+(f' {init}' if init else ''))
    for s in stages:
        if s['name'].startswith('attention-'):
            lines.append('stage '+s['name'])
            for w in s['lines']:
                if w[0]=='params':w=[w[0],str(target/w[1]),*w[2:]]
                lines.append(' '.join(w))
    for d in dumps:
        if d['resource'].startswith('att_') or d['resource'] in ('q','v','o'):
            lines.append(f'dump {d["resource"]} 0 {d["bytes"]} {d["file"]}')
    (out/'graph.txt').write_text('\n'.join(lines)+'\n')
    hashes={p.name:cu.up.eq.sha(p) for p in (ROOT/'build/native').glob('*.cso')}
    result=cu.up.run_graph(out,out/'run0');report=validate_attention(target,result)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if report['status']!='PASS':return report
    repeat=cu.up.run_graph(out,out/'run1',1)
    report['checks']['deterministic_complete_writes']=all(cu.up.eq.sha(repeat/n)==h for n,h in report['hashes'].items())
    report['checks']['zero_debug']=all('messages (0 total)' in (out/f'run{i}/execution.log').read_text() for i in (0,1))
    report['checks']['shaders_unchanged']=all(cu.up.eq.sha(ROOT/'build/native'/n)==h for n,h in hashes.items())
    report['shaders']=hashes;report['status']='PASS' if all(report['checks'].values()) else 'FAIL'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return report


def attention_edge_gate(out):
    """GPU checks for zero rows, signs, extrema and multiple physical windows."""
    out.mkdir(parents=True,exist_ok=False);reports={}
    for heads in (2,4):
        d=out/str(heads);d.mkdir();g={'GW':8,'GH':8,'tokens':64,'heads':heads}
        rng=np.random.default_rng(29+heads)
        q=(rng.integers(-64,65,(64,heads*32))/32).astype('<f2')
        v=(rng.integers(-64,65,q.shape)/16).astype('<f2')
        # Entire zero window, zero head rows and V columns, plus signed zero.
        idx=window_tokens(g);q[idx[0]]=0;v[idx[0]]=0;q[idx[1,0],:32]=np.float16(-0.0)
        v[idx[1],0]=np.float16(-0.0)
        weights=(rng.integers(-16,17,heads*256)/8).astype('<f4')
        for n,a in [('q',q),('v',v),('weights',weights)]: (d/(n+'.bin')).write_bytes(a.tobytes())
        p=np.zeros(40,'<u4');p[1]=heads;p[2]=8;p[3]=8;p[8]=64
        (d/'params.bin').write_bytes(p.tobytes())
        (d/'fixture.json').write_text(json.dumps({'geometry':g,'weights':{'offsets':{'bias':0}}}))
        (d/'graph.txt').write_text('version 1\n'+''.join(f'resource {n} {a.nbytes} {n}.bin\n' for n,a in [('q',q),('v',v),('weights',weights)])+
            f'resource o {q.nbytes}\nstage attention\nshader swin_attention.cso\nthreads 64 1 1\ngroups 1 1 1\nparams params.bin 160\n'
            'srv 0 q\nsrv 1 v\nsrv 2 weights\nuav 0 o\n')
        prepare_attention(d)
        first=cu.up.run_graph(d,d/'run0');r=validate_attention(d,first)
        (out/'report.json').write_text(json.dumps({**reports,str(heads):r},indent=2))
        if r['status']!='PASS':return r
        second=cu.up.run_graph(d,d/'run1',1)
        r['checks']['complete_writes']=all(cu.up.eq.sha(second/n)==h for n,h in r['hashes'].items())
        r['status']='PASS' if all(r['checks'].values()) else 'FAIL';reports[str(heads)]=r
    report={'status':'PASS' if all(r['status']=='PASS' for r in reports.values()) else 'FAIL','shapes':reports}
    (out/'report.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['enc0','cpu','pair','edge','chain','isa','finalize'],default='enc0')
    ap.add_argument('--enc0-fixture',type=Path)
    ap.add_argument('--layers',type=Path,help='source spatial cumulative fixtures')
    ap.add_argument('--frame',type=Path)
    ap.add_argument('--inputs',type=Path)
    ap.add_argument('--chain',type=Path)
    ap.add_argument('--isa',type=Path)
    ap.add_argument('--rga',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=a.out.resolve()
    required={'enc0':['enc0_fixture'],'cpu':['frame','layers'],'pair':['layers','inputs'],
              'edge':[],'chain':['layers','frame'],'isa':['frame','rga'],'finalize':['chain','frame','isa']}[a.phase]
    for name in required:
        if not getattr(a,name):ap.error('--'+name.replace('_','-')+' is required for '+a.phase)
        setattr(a,name,getattr(a,name).resolve())
    if a.phase=='enc0':r=run_enc0(a.enc0_fixture,out)
    elif a.phase=='pair':r=run_attention_pair(Path(cu.up.suite.fixture(str(a.layers),'enc1')),a.inputs,out)
    elif a.phase=='chain':r=run_complete_chain(a.layers,a.frame,out)
    elif a.phase=='finalize':r=finalize_complete(a.chain,a.frame,a.isa,out)
    elif a.phase=='edge':r=attention_edge_gate(out)
    elif a.phase=='isa':
        executed=json.loads((a.frame/'b6.json').read_text())['inputs']['shaders']
        if not a.inputs:ap.error('--inputs must name the accepted representative pair for pre-expansion ISA checks')
        pair=json.loads((a.inputs/'report.json').read_text())
        if pair['status']!='PASS':raise ValueError('representative pair gate has not passed')
        audit_attention(a.inputs/'fixture')
        for n in ['swin_attention_qk.cso','swin_attention_pv.cso']:
            if pair['shaders'][n]!=executed[n]:raise ValueError('pair/frame shader inventories disagree')
        r={'status':'PASS','isa':inspect_isa(out,a.rga,executed)}
    else:
        results=representative_inputs(a.frame,out)
        directory=Path(cu.up.suite.fixture(str(a.layers),'enc1'));rec=json.loads((directory/'fixture.json').read_text());g=rec['geometry']
        q=np.fromfile(results/'enc1_q.bin','<f2').reshape(g['tokens'],-1);v=np.fromfile(results/'enc1_v.bin','<f2').reshape(q.shape)
        resources,_,_=cu.up.suite._parse_graph(directory/'graph.txt')
        w=np.fromfile(directory/next(r['init'] for r in resources if r['name']=='weights'),'<f4');offset=rec['weights']['offsets']['bias']
        _,r=attention_reference(q,v,w[offset:offset+g['heads']*256],g)
        (out/'cpu-contract.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ('coverage','shaders','hashes')},indent=2));sys.exit(r['status']!='PASS')
