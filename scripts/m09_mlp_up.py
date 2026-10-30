#!/usr/bin/env python3
"""EXP-026: one enc1 up projection, independent oracle and fixed graph integration.

All generated fixtures/diagnostics contain user-derived data and belong outside Git.
"""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m09_enc1_q as eq
import m_layer_fixture as lf
import m_layer_suite as suite

ROOT = Path(__file__).resolve().parents[1]


def gp(c):
    return 32*(c//32) + 2*((c%32)//4) + c%2 + 16*((c%4)//2)


def raw_up_offset(base, k, j, C=64):
    """Captured blocked E4M3 W1, natural output j, byte offset."""
    n = gp(j)
    return (base + 512*(k//32) + 16*C*(n//16) + 64*(n%8)
            + 16*((k%16)//4) + 8*((n//8)%2) + 4*((k%32)//16) + k%4)


def weights(directory):
    directory = Path(directory)
    r = json.loads((directory/'fixture.json').read_text())
    if r['layer'] != 'enc1' or r['geometry']['C'] != 64:
        raise ValueError('only enc1 C=64 is supported')
    off = r['weights']['offsets']
    data = np.fromfile(directory/'weights.bin', '<f4')
    w = data[off['w1']:off['w1']+16384].reshape(8,64,32).transpose(1,0,2).reshape(64,256)
    bias = data[off['b1']:off['b1']+256]
    if not (np.isfinite(w).all() and np.isfinite(bias).all()):
        raise ValueError('non-finite MLP weights')
    return r, w, bias


def cpu_oracle(a, w, bias):
    codes, scales = eq.quantize_rows_float(a)
    wc, ws, _ = eq.quantize_weights(w)
    acc = codes.astype(np.int64) @ wc.astype(np.int64)
    y = ((acc.astype(np.float32)*scales)*ws[None,:]).astype(np.float32)
    y = (y+np.asarray(bias,np.float32)[None,:]).astype(np.float32)
    return codes, scales.ravel(), wc, ws, acc, y, y.astype(np.float16)


def scalar_row(row):
    """Independent scalar encoder: Python ties-to-even after explicit f32 product."""
    top = max(abs(float(v)) for v in row) or 1.0
    inv = np.float32(np.float32(127)/np.float32(top))
    codes = [max(-127,min(127,round(float(np.float32(np.float32(v)*inv))))) for v in row]
    return np.array(codes,np.int8), np.float32(np.float32(top)*np.float32(1/127))


def scalar_weight_column(column):
    top=max(abs(float(v)) for v in column)
    scale=np.float32(top/127 if top else 1)
    codes=[max(-127,min(127,round(float(v)/float(scale)))) for v in column]
    return np.array(codes,np.int8),scale


def prepare(source, out, mode='int8-up'):
    """Clone an established FP16 fixture; no activation data is decoded on host.

    `probe` materializes the FP16 boundary for validation only, leaving Mlp intact.
    `fp16-up` supplies this boundary to the wrapper to validate the split exactly.
    """
    if mode not in ('int8-up','probe','fp16-up'):
        raise ValueError('unknown MLP mode')
    source, out = Path(source), Path(out)
    r,w,bias = weights(source)
    resources,stages,dumps = suite._parse_graph(source/'graph.txt')
    if r.get('q_mode','fp16')!='fp16' or r.get('mlp_mode','fp16')!='fp16':
        raise ValueError('MLP experiment requires an unmodified FP16 fixture')
    if [s['name'] for s in stages].count('mlp') != 1:
        raise ValueError('expected one fused MLP stage')
    out.mkdir(parents=True,exist_ok=False)
    for p in source.iterdir():
        if p.is_file(): shutil.copyfile(p,out/p.name)
    M=r['geometry']['tokens']; C=64; N=256
    additions = [f'resource mlp_up {M*N*2}']
    extra_dumps = [('h2',M*C*2),('x2',M*C*2),('x2b',M*C*2),('mlp_up',M*N*2)]
    def stage(name,shader,params,groups,srvs,uavs):
        return ([f'stage {name}',f'shader {shader}','threads 64 1 1',f'groups {groups} 1 1',
                 f'params {params}'] + [f'srv {i} {v}' for i,v in enumerate(srvs)]
                + [f'uav {i} {v}' for i,v in enumerate(uavs)])
    prefix=[]
    if mode=='int8-up':
        wc,ws,_=eq.quantize_weights(w)
        for name,data in [('mlp_wc',np.ascontiguousarray(wc.T).tobytes()),
                          ('mlp_ws',ws.astype('<f4').tobytes()),('mlp_bias',bias.astype('<f4').tobytes())]:
            (out/f'{name}.bin').write_bytes(data)
            additions.append(f'resource {name} {len(data)} {name}.bin')
        for name,size in [('mlp_a',M*C),('mlp_as',M*4),('mlp_acc',M*N*4),('mlp_float',M*N*4)]:
            additions.append(f'resource {name} {size}'); extra_dumps.append((name,size))
        (out/'params-mlp-q8.bin').write_bytes(np.array([M,C,N,0],'<u4').tobytes())
        prefix += stage('mlp-quant','swin_q8_quant.cso','params-mlp-q8.bin 16',(M+63)//64,
                        ['h2'],['mlp_a','mlp_as'])
        prefix += stage('mlp-up','swin_mlp_up_q8.cso','params-mlp-q8.bin 16',(M*(N//16)+63)//64,
                        ['mlp_a','mlp_wc','mlp_as','mlp_ws','mlp_bias'],['mlp_up','mlp_acc','mlp_float'])
    else:
        prefix += stage('mlp-up','swin_mlp_up_probe.cso','params-mlp.bin 160',(M*8+63)//64,
                        ['h2','weights','x2'],['mlp_up','scratch'])
    text=(source/'graph.txt').read_text()
    text=text.replace('version 1\n','version 1\n'+'\n'.join(additions)+'\n',1)
    text=text.replace('stage mlp\n','\n'.join(prefix)+'\nstage mlp\n',1)
    if mode!='probe':
        text=text.replace('shader swin_mlp.cso','shader swin_mlp_up_epilogue.cso',1)
        # The bounded runner builds descriptors in declaration order.
        before, mlp = text.split('stage mlp\n', 1)
        mlp=mlp.replace('srv 2 x2\n','srv 2 x2\nsrv 3 mlp_up\n',1)
        text=before+'stage mlp\n'+mlp
    for name,size in extra_dumps:
        text+=f'dump {name} 0 {size} dbg-{name}.bin\n'
    (out/'graph.txt').write_text(text)
    r['mlp_mode']=mode
    r['mlp_contract']={'M':M,'K':C,'N':N,'projection':'up','layout':'f16 row-major [M,256]'}
    parsed_resources, parsed_stages, _ = suite._parse_graph(out/'graph.txt')
    r['resources']=[[v['name'],v['bytes']] for v in parsed_resources]
    r['stages']=[{'name':s['name'],'shader':next(w[1] for w in s['lines'] if w[0]=='shader')}
                 for s in parsed_stages]
    (out/'fixture.json').write_text(json.dumps(r,indent=2))
    return out


def audit_layout(capture, fixture):
    """Raw capture vs exported weights, with a separate scalar address formula."""
    P,*_=lf.lo.read_layer(str(capture),'enc1')
    r,w,b=weights(fixture)
    L=lf.lo.sm.layout(64,2)
    checks=[]
    synthetic=np.arange(len(P.wbuf),dtype=np.int64)*73+19
    for chunk,base in enumerate(L['b1']):
        offsets=np.array([[raw_up_offset(base,k,j) for j in range(32)] for k in range(64)])
        table=lf.lo.sm.woff_table(base,512,1024,64,32)[:,[gp(j) for j in range(32)]]
        checks.append(bool(np.array_equal(offsets,table) and np.unique(offsets).size==2048
                           and np.array_equal(synthetic[offsets],synthetic[table])))
        decoded=lf.lo.me.E4M3[P.wbuf[offsets]]
        checks.append(bool(np.array_equal(decoded,w[:,chunk*32:(chunk+1)*32])))
        rawbias=np.array([P.f16vec(base+2048,32)[gp(j)] for j in range(32)])
        checks.append(bool(np.array_equal(rawbias,b[chunk*32:(chunk+1)*32])))
    return {'pass':all(checks),'checks':checks,'raw_chunk_bases':L['b1'],
            'capture':str(capture),'source_weights_sha256':eq.sha(Path(fixture)/'weights.bin')}


def run_graph(directory,out,seed=0):
    probe=ROOT/'build/native/Release/m_network_stage.exe'
    done=subprocess.run([str(probe),'--graph',str(directory/'graph.txt'),'--out',str(out),
                         '--shaders',str(ROOT/'build/native'),'--seed',str(seed)],capture_output=True,text=True)
    if done.returncode: raise RuntimeError(done.stdout[-1500:]+done.stderr[-1500:])
    (out/'execution.log').write_text(done.stdout+done.stderr)
    return out


def synthetic_gate(out):
    """Nonuniform signed inputs, zero rows/columns, ties and INT8 endpoints on GPU."""
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(26)
    a=(rng.normal(size=(8,64))*2).astype('<f2')
    a[0]=0; a[1]=127; a[2]=-127
    a[3,:9]=[127,-127,.5,1.5,2.5,-.5,-1.5,-2.5,0]
    w=(rng.normal(size=(64,256))*.2).astype('<f2').astype('<f4')
    w[:,0]=0; w[:,1]=1; w[:,2]=-1
    bias=rng.normal(size=256).astype('<f2').astype('<f4')
    c,s,wc,ws,acc,y,half=cpu_oracle(a,w,bias)
    lines=['version 1']
    for name,data in [('a',a.tobytes()),('w',np.ascontiguousarray(wc.T).tobytes()),
                      ('ws',ws.astype('<f4').tobytes()),('bias',bias.tobytes())]:
        (out/f'{name}.bin').write_bytes(data)
        lines.append(f'resource {name} {len(data)} {name}.bin')
    for name,size in [('codes',512),('scales',32),('acc',8192),('float',8192),('up',4096)]:
        lines.append(f'resource {name} {size}')
    (out/'params.bin').write_bytes(np.array([8,64,256,0],'<u4').tobytes())
    lines += ['stage quant','shader swin_q8_quant.cso','threads 64 1 1','groups 1 1 1',
              'params params.bin 16','srv 0 a','uav 0 codes','uav 1 scales',
              'stage up','shader swin_mlp_up_q8.cso','threads 64 1 1','groups 2 1 1',
              'params params.bin 16','srv 0 codes','srv 1 w','srv 2 scales','srv 3 ws','srv 4 bias',
              'uav 0 up','uav 1 acc','uav 2 float']
    expected={'codes':c,'scales':s.astype('<f4'),'acc':acc.astype('<i4'),'float':y,'up':half}
    for name,data in expected.items(): lines.append(f'dump {name} 0 {data.nbytes} {name}.bin')
    (out/'graph.txt').write_text('\n'.join(lines)+'\n')
    res=run_graph(out,out/'results'); repeat=run_graph(out,out/'repeat',1)
    checks={name:(res/f'{name}.bin').read_bytes()==np.ascontiguousarray(data).tobytes()
            and eq.sha(res/f'{name}.bin')==eq.sha(repeat/f'{name}.bin') for name,data in expected.items()}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}


def validate(source,out,capture):
    out.mkdir(parents=True,exist_ok=False)
    report={'layout':audit_layout(capture,source),'checks':{},'metrics':{}}
    if not report['layout']['pass']: raise RuntimeError('physical layout gate failed')
    report['synthetic']=synthetic_gate(out/'synthetic')
    if report['synthetic']['status']!='PASS': raise RuntimeError('synthetic GPU gate failed')
    r,w,bias=weights(source); M=r['geometry']['tokens']
    paths={}
    for mode in ('probe','fp16-up','int8-up'):
        fixture=prepare(source,out/mode,mode)
        paths[mode]=run_graph(fixture,out/(mode+'-results'))
    gpu=paths['int8-up']; fp=paths['probe']
    again=run_graph(out/'int8-up',out/'int8-up-repeat',1)
    checks=report['checks']
    for p in gpu.glob('*.bin'):
        checks['deterministic_'+p.name]=eq.sha(p)==eq.sha(again/p.name)
    for role in ('out-out','out-merge','dbg-x2b'):
        checks['fp16_split_'+role]=eq.sha(fp/f'{role}.bin')==eq.sha(paths['fp16-up']/f'{role}.bin')
    checks['same_h2']=eq.sha(fp/'dbg-h2.bin')==eq.sha(gpu/'dbg-h2.bin')
    checks['same_x2']=eq.sha(fp/'dbg-x2.bin')==eq.sha(gpu/'dbg-x2.bin')
    a=np.fromfile(gpu/'dbg-h2.bin','<f2').reshape(M,64)
    codes,scales,wc,ws,acc,y,half=cpu_oracle(a,w,bias)
    for name,want,dtype in [('mlp_a',codes,np.int8),('mlp_as',scales,'<f4'),
                            ('mlp_acc',acc.astype('<i4'),'<i4'),('mlp_float',y,'<f4'),('mlp_up',half,'<f2')]:
        data=(gpu/f'dbg-{name}.bin').read_bytes()
        checks['cpu_'+name]=data==np.ascontiguousarray(want,dtype=dtype).tobytes()
    checks['weight_codes']=(out/'int8-up/mlp_wc.bin').read_bytes()==np.ascontiguousarray(wc.T).tobytes()
    checks['weight_scales']=(out/'int8-up/mlp_ws.bin').read_bytes()==ws.astype('<f4').tobytes()
    checks['int32_safe']=int(np.abs(acc).max())<=64*127*127<2**31
    probes=[]
    rng=np.random.default_rng(26)
    for row,col in [(0,0),(M-1,255)]+[(int(rng.integers(M)),int(rng.integers(256))) for _ in range(46)]:
        c,s=scalar_row(a[row]); total=eq.scalar_dot(c,wc[:,col])
        value=np.float32(np.float32(np.float32(total)*s)*ws[col])+np.float32(bias[col])
        probes.append(bool(np.array_equal(c,codes[row]) and s.tobytes()==scales[row].tobytes()
                           and total==acc[row,col] and np.float16(value).tobytes()==half[row,col].tobytes()))
    checks['scalar_probes']=all(probes)
    checks['scalar_weight_encoding']=all(
        np.array_equal(scalar_weight_column(w[:,col])[0],wc[:,col])
        and scalar_weight_column(w[:,col])[1].tobytes()==ws[col].tobytes() for col in range(256))
    report['scalar_probes']=len(probes)
    report['activation']=eq.code_metrics(codes,scales)
    report['safety']={}
    report['accumulator_max_abs']=int(np.abs(acc).max())
    # FP16 oracle is the independently established k32 chain, not the new kernel.
    fpup=np.fromfile(fp/'dbg-mlp_up.bin','<f2').reshape(M,256)
    for chunk in range(8):
        sl=slice(chunk*32,(chunk+1)*32)
        expected=lf.me.mma_chain(a.astype(np.float64),w[:,sl].astype(np.float64),
                                C0=np.broadcast_to(bias[sl].astype(np.float64),(M,32))).astype('<f2')
        checks[f'fp16_cpu_chunk{chunk}']=expected.tobytes()==np.ascontiguousarray(fpup[:,sl]).tobytes()
    report['metrics']['projection']=eq.compare(half,fpup)
    for label,path in paths.items():
        for role in ('mlp_up','x2b'):
            values=np.fromfile(path/f'dbg-{role}.bin','<f2')
            checks[f'{label}_{role}_finite_unsaturated']=bool(np.isfinite(values).all() and (np.abs(values)<65504).all())
            report['safety'][label+'_'+role]={'non_finite':int((~np.isfinite(values)).sum()),
                'saturated':int((np.abs(values)>=65504).sum())}
    for role in ('out','merge'):
        baseline=np.fromfile(fp/f'out-{role}.bin',np.uint8)
        hybrid=np.fromfile(gpu/f'out-{role}.bin',np.uint8)
        ref=np.fromfile(source/f'ref-{role}.bin',np.uint8)
        # _field computes second minus first.
        report['metrics'][role]={'hybrid_vs_fp16':eq._field(baseline,hybrid),
            'fp16_vs_capture':eq._field(ref,baseline),'hybrid_vs_capture':eq._field(ref,hybrid)}
        report['safety'][role]={'non_finite':int((~np.isfinite(eq.E4M3[hybrid])).sum()),
                               'saturated':int(((hybrid&127)==126).sum())}
        checks[role+'_finite_unsaturated']=all(v==0 for v in report['safety'][role].values())
    report['metrics']['pre_store']=eq.compare(np.fromfile(gpu/'dbg-x2b.bin','<f2'),np.fromfile(fp/'dbg-x2b.bin','<f2'))
    report['shaders']={p.name:eq.sha(p) for p in (ROOT/'build/native').glob('*.cso')}
    report['probe_sha256']=eq.sha(ROOT/'build/native/Release/m_network_stage.exe')
    report['status']='PASS' if all(checks.values()) else 'FAIL'
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'failed':[k for k,v in checks.items() if not v],
                      'metrics':report['metrics']},indent=2))
    return report


def audit_frame(stage_dir, frame_dir, fp16_dir, q_dir, dec0_dir, rga):
    """Tie checked stage bytes, executed chain, ISA and unchanged control frames."""
    validation=json.loads((stage_dir/'validation.json').read_text())
    frame=json.loads((frame_dir/'b6.json').read_text())
    checks={'stage_gate':validation['status']=='PASS',
            'mlp_only':frame['mlp_mode']=='int8-up' and frame['q_mode']==frame['dec0_mode']=='fp16',
            'frame_execution':frame['execution']['status']=='PASS',
            'deterministic_boundaries':frame['repeat']['fresh'] and frame['repeat']['deterministic']}
    result={'checks':checks,'layer':validation['metrics'],'controls':{},'frame':{
        'canonical':frame['comparisons']['native'],'direct':frame['direct_comparisons']}}
    baseline_hashes={'fp16':'dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
        'enc1q':'0eb0185ea7e21f20617b67476519310ae2dfe8b402188f3207fb197b35299419',
        'dec0':'d7ef968356d95369a276edf3d4af3baefb1d46c7a3f98b292e3483637bb7075a'}
    for label,directory in [('fp16',fp16_dir),('enc1q',q_dir),('dec0',dec0_dir)]:
        rec=json.loads((directory/'b6.json').read_text())
        digest=eq.sha(directory/'downsample-native/results/out-out.bin')
        checks['control_'+label]=digest==baseline_hashes[label]==rec['comparisons']['native']['frame_sha256']
        result['controls'][label]={'frame_sha256':digest,'metrics':rec['comparisons']['native']}
    gpu=frame_dir/'chain-native-results'
    for p in gpu.glob('*.bin'):
        checks['chain_repeat_'+p.name]=eq.sha(p)==eq.sha(frame_dir/'chain-repeat-results'/p.name)
    for p in (stage_dir/'int8-up-results').glob('dbg-*.bin'):
        checks['frame_handoff_'+p.name]=eq.sha(p)==eq.sha(gpu/('enc1-'+p.name))
    for role in ('out','merge'):
        checks['frame_enc1_'+role]=eq.sha(stage_dir/'int8-up-results'/f'out-{role}.bin')==eq.sha(gpu/f'out-enc1-{role}.bin')
    for tag in ('native','repeat'):
        checks['debug_'+tag]='D3D12 debug: zero error/corruption messages (0 total)' in (
            frame_dir/f'chain-{tag}-results/execution.log').read_text()
    a=np.fromfile(fp16_dir/'chain-native-results/out-dec1.bin',np.uint8)
    b=np.fromfile(gpu/'out-dec1.bin',np.uint8)
    ref=np.fromfile(Path(frame['layers'])/'dec1/ref-out.bin',np.uint8)
    result['dec1']={'hybrid_vs_fp16':eq._field(a,b),'fp16_vs_capture':eq._field(ref,a),
                    'hybrid_vs_capture':eq._field(ref,b)}
    result['dec1_safety']={'non_finite':int((~np.isfinite(eq.E4M3[b])).sum()),
                          'saturated':int(((b&127)==126).sum())}
    checks['dec1_finite_unsaturated']=bool(np.isfinite(eq.E4M3[b]).all() and not np.any((b&127)==126))
    # Compare every non-MLP stage (shader, parameters and bindings), not just counts.
    _,baseline,_=suite._parse_graph(fp16_dir/'chain-native/graph.txt')
    _,hybrid,_=suite._parse_graph(frame_dir/'chain-native/graph.txt')
    def normalized(stages):
        out=[]
        for s in stages:
            if s['name'].startswith('enc1.mlp'): continue
            rows=[]
            for words in s['lines']:
                if words[0]=='params': rows.append(['params',eq.sha(words[1]),words[2]])
                elif words[0]=='shader': rows.append(['shader',Path(words[1]).name])
                else: rows.append(words)
            out.append({'name':s['name'],'lines':rows})
        return out
    checks['all_other_stages_unchanged']=normalized(baseline)==normalized(hybrid)
    checks['exactly_one_up_projection']=len(hybrid)==len(baseline)+2
    shaders=ROOT/'build/native'
    for name,digest in validation['shaders'].items():
        checks['shader_'+name]=eq.sha(shaders/name)==digest==frame['inputs']['shaders'][name]
    checks['probe_hash']=eq.sha(ROOT/'build/native/Release/m_network_stage.exe')==validation['probe_sha256']==frame['inputs']['network_probe_sha256']
    shader=shaders/'swin_mlp_up_q8.cso'
    digest=eq.sha(shader)
    done=subprocess.run([str(rga),'-s','dx12','-c','gfx1101','--cs-blob',str(shader),
                         '--isa',str(stage_dir/'executed-mlp-up')],capture_output=True,text=True)
    (stage_dir/'rga.log').write_text(done.stdout+done.stderr)
    if done.returncode: raise RuntimeError('RGA failed')
    path=stage_dir/'gfx1101_executed-mlp-up_comp.isa'
    text=path.read_text()
    result['isa']={'shader_sha256':digest,'isa_sha256':eq.sha(path),
        'dot4':len(re.findall('v_dot4_i32_i8',text)),
        'fp16_matrix':len(re.findall(r'v_(?:pk_)?(?:mul|fma|mac)_f16|v_wmma',text))}
    checks['native_int8']=result['isa']['dot4']>0 and result['isa']['fp16_matrix']==0
    checks['inspected_shader_unchanged']=eq.sha(shader)==digest
    result['provenance']={'validation_sha256':eq.sha(stage_dir/'validation.json'),
        'frame_report_sha256':eq.sha(frame_dir/'b6.json'),'network_probe_sha256':validation['probe_sha256']}
    result['status']='PASS' if all(checks.values()) else 'FAIL'
    (stage_dir/'integration.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':result['status'],'failed':[k for k,v in checks.items() if not v],
                      'isa':result['isa'],'dec1':result['dec1']},indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--capture',type=Path)
    ap.add_argument('--frame-run',type=Path,help='audit an executed frame against --out stage validation')
    ap.add_argument('--fp16-run',type=Path)
    ap.add_argument('--enc1q-run',type=Path)
    ap.add_argument('--dec0-run',type=Path)
    ap.add_argument('--rga',type=Path)
    args=ap.parse_args()
    if args.frame_run:
        if not all([args.fp16_run,args.enc1q_run,args.dec0_run,args.rga]):
            ap.error('--frame-run requires --fp16-run, --enc1q-run, --dec0-run and --rga')
        result=audit_frame(args.out,args.frame_run,args.fp16_run,args.enc1q_run,args.dec0_run,args.rga)
    else:
        if not all([args.source,args.capture]): ap.error('validation requires --source and --capture')
        result=validate(args.source.resolve(),args.out.resolve(),args.capture.resolve())
    sys.exit(0 if result['status']=='PASS' else 1)
