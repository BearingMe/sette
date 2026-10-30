#!/usr/bin/env python3
"""EXP-028: fixed Preset M matrix inventory and cumulative learned-weight INT8.

This is a bounded fixture transformation of the established ten-layer graph,
not an inference framework. Generated tensors/weights must remain outside Git.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m09_mlp_up as up
import m_weights as mw

ROOT = up.ROOT
BATCHES = {'qv': ('q', 'v'), 'output': ('q', 'v', 'wo'),
           'mlp': ('q', 'v', 'wo', 'up', 'down'),
           'spatial': ('q', 'v', 'wo', 'up', 'down', 'merge', 'expand', 'dec0')}


def inventory(layers, batch=None, validated=False):
    """Count logical matrix operators and MACs, including fused/batched products."""
    rows = []
    def add(layer, dispatch, family, M, K, N, instances=1, layout='row-f16 / natural K,N f32 weights'):
        dynamic = family in ('qkt', 'pv')
        converted = family in BATCHES.get(batch, ()) and family != 'enc0'
        rows.append({'id': f'{layer}.{family}', 'layer': layer, 'dispatch': dispatch,
            'family': family, 'M': M, 'K': K, 'N': N, 'instances': instances,
            'macs': int(M*K*N*instances), 'kind': 'dynamic' if dynamic else 'learned-weight',
            'operand_layout': layout, 'original_precision': 'E4M3 operands/f16 k32 accumulation' if not dynamic
            else ('E4M3 Q, f16 scores' if family == 'qkt' else 'f16 probabilities/V, f16 output'),
            'implementation': ('swin_layer_cs.hlsl::Attention' if dynamic else
                {'up':'swin_layer_cs.hlsl::Mlp', 'down':'swin_layer_cs.hlsl::Mlp',
                 'merge':'swin_layer_cs.hlsl::Merge', 'enc0':'enc0_cs.hlsl::Column',
                 'dec0':'dec0_cs.hlsl::Column'}.get(family, 'swin_layer_cs.hlsl::Matmul')),
            'compatibility': ('new dynamic-operand scale contract required' if dynamic else
                'texture-feature adapter not implemented' if family == 'enc0' else
                'existing symmetric per-row/per-column contract'),
            'conversion': ('unresolved dynamic RHS quantization/rescaling' if dynamic else
                {'merge':'GPU Q8 patch gather + original field epilogue',
                 'expand':'GPU plane E4M3-to-f16 unpack', 'enc0':'GPU dither/texture feature encoder + field epilogue',
                 'dec0':'existing fixed GPU image prefix', 'up':'naturalize chunked W1 offline; original GELU',
                 'down':'whole-K dot + original residual input'}.get(family, 'row encoder + existing dot4')),
            'status': ('INT8 implemented and validated' if validated else 'INT8 selected, validation pending') if converted else
                'unsupported or unresolved' if dynamic else 'INT8 feasible but not implemented',
            'int8': converted})
    for label in up.suite.LABELS:
        source = Path(up.suite.fixture(str(layers), label))
        r = json.loads((source / 'fixture.json').read_text())
        g = r['geometry']; C = g['C']; Q = g['heads']*32; M = g['tokens']
        _, stages, _ = up.suite._parse_graph(source / 'graph.txt')
        expected = ['expand'] if g['CIN'] else []
        expected += ['prepare','norm1','q','v','attention','wo','norm2','mlp','store']
        if g['NPM']: expected += ['merge']
        if [s['name'] for s in stages] != expected:
            raise ValueError(f'{label}: unknown or omitted baseline stages')
        for family in ('q','v'): add(label, family, family, M, C, Q)
        windows = (g['GW']//4)*(g['GH']//4)*g['heads']
        add(label, 'attention', 'qkt', 16, 32, 16, windows, 'per-head 4x4 windows, Q8(Q) x Q8(Q).T')
        add(label, 'attention', 'pv', 16, 16, 32, windows, 'f16 P per window/head x natural f16 V')
        add(label, 'wo', 'wo', M, Q, C)
        add(label, 'mlp', 'up', M, C, 4*C, layout='row-f16 h2 / chunked W1[C,32] naturalized to [C,4C]')
        add(label, 'mlp', 'down', M, 4*C, C, layout='GeluQ f16 [M,4C] / W2[4C,C], residual f16 [M,C]')
        if g['NPM']:
            add(label, 'merge', 'merge', (g['GW']//2)*(g['GH']//2), 4*C, g['NPM'],
                layout='Q8 patch gather of four padded tokens / natural [4C,NPM]')
        if g['CIN']:
            add(label, 'expand', 'expand', g['lowW']*g['lowH'], g['CIN'], 4*C,
                layout='plane-major E4M3 low field / natural [CIN,4C]')
    add('enc0','image','enc0',130560,128,64,layout='texture/dither-derived E4M3 [M,128] / blocked E4M3 [128,64]')
    add('dec0','image','dec0',130560,64,80,layout='two E4M3 planes / 80 active columns of 128 physical')
    if len(rows) != 76 or len({r['id'] for r in rows}) != 76:
        raise ValueError('inventory must cover 76 distinct logical matrix operators')
    total = sum(r['macs'] for r in rows)
    learned = [r for r in rows if r['kind']=='learned-weight']
    return {'operators': rows, 'count': len(rows), 'learned_count': len(learned),
        'dynamic_count': len(rows)-len(learned), 'macs': total,
        'int8_count': sum(r['int8'] for r in rows),
        'int8_macs': sum(r['macs'] for r in rows if r['int8']),
        'operation_coverage': sum(r['int8'] for r in rows)/len(rows),
        'mac_coverage': sum(r['macs'] for r in rows if r['int8'])/total,
        'non_gemm': [{'stage':'post','status':'explicit floating-point exception',
            'reason':'data-dependent 3x3 Gaussian filtering and pointwise covariance quadratic forms; not neural GEMMs'},
            {'stage':'downsample','status':'explicit floating-point exception',
             'reason':'5x5 reconstruction filter, not a learned/dynamic neural GEMM'}],
        'denominator':'76 logical layer/family operators, batched attention instances weighted by M*K*N; padded executed rows; one MAC, not two FLOPs'}


def prepare_layer(source, out, batch):
    """Fixed family transformations; original stages/parameters are the fallback."""
    source, out = Path(source).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = json.loads((source/'fixture.json').read_text())
    if record.get('q_mode','fp16')!='fp16' or record.get('mlp_mode','fp16')!='fp16':
        raise ValueError('cumulative input must be the canonical FP16 fixture')
    g = record['geometry']; C=g['C']; T=g['tokens']; NQ=g['heads']*32
    offsets=record['weights']['offsets']; raw=np.fromfile(source/'weights.bin','<f4')
    resources, original, _ = up.suite._parse_graph(source/'graph.txt')
    lines=['version 1']; stages=[]; sizes={}; dumps={}; operations=[]; layouts=[]
    for r in resources:
        lines.append(f'resource {r["name"]} {r["bytes"]}' + (f' {source/r["init"]}' if r['init'] else ''))
        sizes[r['name']]=r['bytes']
    def resource(name,size,data=None):
        if name in sizes: raise ValueError('duplicate resource '+name)
        sizes[name]=size
        if data is not None: (out/f'{name}.bin').write_bytes(data)
        lines.append(f'resource {name} {size}'+(f' {name}.bin' if data is not None else ''))
        return name
    def stage(name,shader,params,srvs,uavs,groups):
        stages.extend([f'stage {name}',f'shader {shader}','threads 64 1 1',f'groups {groups} 1 1',f'params {params}'])
        stages.extend(f'srv {i} {v}' for i,v in enumerate(srvs))
        stages.extend(f'uav {i} {v}' for i,v in enumerate(uavs))
    def keep(s):
        stages.append('stage '+s['name'])
        for words in s['lines']:
            w=list(words)
            if w[0]=='params': w[1]=str(source/w[1])
            stages.append(' '.join(w))
    def dump(name): dumps[name]=sizes[name]
    def weight(key,K,N): return raw[offsets[key]:offsets[key]+K*N].reshape(K,N)
    def gemm(name,A,Y,M,K,N,W,bias=None,residual=None):
        if K%4 or N%16 or M*(N//16)>65535*64: raise ValueError('unsupported dispatch shape')
        wc,ws,_=up.eq.quantize_weights(W)
        for col in range(N):
            c,s=up.scalar_weight_column(W[:,col])
            if not np.array_equal(c,wc[:,col]) or s.tobytes()!=ws[col].tobytes():
                raise ValueError('offline scalar weight encoding mismatch')
        prefix='i8_'+name
        codes=resource(prefix+'_w',K*N,np.ascontiguousarray(wc.T).tobytes())
        scales=resource(prefix+'_ws',N*4,ws.astype('<f4').tobytes())
        ac=resource(prefix+'_a',M*K); ass=resource(prefix+'_as',M*4)
        acc=resource(prefix+'_acc',M*N*4); flt=resource(prefix+'_float',M*N*4)
        if residual is None:
            b=np.zeros(N,'<f4') if bias is None else np.asarray(bias,'<f4')
            fifth=resource(prefix+'_bias',N*4,b.tobytes()); shader='swin_mlp_up_q8.cso'
        else: fifth=residual; shader='swin_mlp_down_q8.cso'
        params=f'params-{prefix}.bin'; (out/params).write_bytes(np.array([M,K,N,0],'<u4').tobytes())
        stage(name+'-quant','swin_q8_quant.cso',params+' 16',[A],[ac,ass],(M+63)//64)
        stage(name,shader,params+' 16',[ac,codes,ass,scales,fifth],[Y,acc,flt],(M*(N//16)+63)//64)
        for v in (A,Y,ac,ass,acc,flt): dump(v)
        if residual: dump(residual)
        wfile=prefix+'_oracle_w.bin'; (out/wfile).write_bytes(np.ascontiguousarray(W,'<f4').tobytes())
        operations.append({'name':name,'A':A,'Y':Y,'M':M,'K':K,'N':N,'W':wfile,
            'codes':ac,'scales':ass,'acc':acc,'float':flt,'weight_codes':codes,'weight_scales':scales,
            'bias':None if residual else fifth,'residual':residual,'shader':shader})
    active=BATCHES[batch]
    for s in original:
        name=s['name']; par=str(source/f'params-{name}.bin')+' 160'
        if name in ('q','v') and name in active:
            gemm(name,'h1',name,T,C,NQ,weight(name,C,NQ))
        elif name=='wo' and name in active:
            gemm(name,'o','x1',T,NQ,C,weight('wo',NQ,C),residual='res')
        elif name=='mlp' and 'up' in active:
            Z=resource('i8_z',T*4*C*2); G=resource('i8_g',T*4*C*2)
            W=weight('w1',4*C,C).reshape(C//8,C,32).transpose(1,0,2).reshape(C,4*C)
            bias=raw[offsets['b1']:offsets['b1']+4*C]
            gemm('mlp-up','h2',Z,T,C,4*C,W,bias=bias)
            stage('mlp-gelu','swin_mlp_gelu.cso',par,[Z,'weights','x2'],[G,'scratch'],(T+63)//64)
            dump(G); layouts.append({'kind':'gelu','A':Z,'Y':G,'shape':[T,4*C]})
            gemm('mlp-down',G,'x2b',T,4*C,C,weight('w2',4*C,C),residual='x2')
        elif name=='expand' and name in active:
            M=g['lowW']*g['lowH']; K=g['CIN']; N=4*C
            A=resource('i8_low',M*K*2)
            stage('expand-unpack','swin_cumulative_unpack.cso',par,['low','weights','weights'],[A,'scratch'],(M+63)//64)
            dump('low'); layouts.append({'kind':'unpack','A':'low','Y':A,'shape':[M,K]})
            gemm(name,A,'expand',M,K,N,weight('expandW',K,N),bias=raw[offsets['expandB']:offsets['expandB']+N])
        elif name=='merge' and name in active:
            M=g['GW']//2*(g['GH']//2); K=4*C; N=g['NPM']
            A=resource('i8_patch',M*K*2); Y=resource('i8_merge',M*N*2)
            stage('merge-gather','swin_cumulative_gather.cso',par,['x2b','weights','weights'],[A,'scratch'],(M+63)//64)
            dump('x2b'); layouts.append({'kind':'gather','A':'x2b','Y':A,'shape':[M,K]})
            gemm(name,A,Y,M,K,N,weight('mergeW',K,N),bias=raw[offsets['mergeB']:offsets['mergeB']+N])
            stage('merge-store','swin_cumulative_merge_store.cso',par,[Y,'weights','weights'],['merge','scratch'],(M+63)//64)
            dump('merge'); layouts.append({'kind':'merge-store','A':Y,'Y':'merge','shape':[M,N]})
        else: keep(s)
    lines.extend(stages)
    for role,entry in record['outputs'].items():
        lines.append(f'dump {role} 0 {entry["bytes"]} out-{role}.bin')
        shutil.copyfile(source/f'ref-{role}.bin',out/f'ref-{role}.bin')
    for name,size in dumps.items(): lines.append(f'dump {name} 0 {size} dbg-{name}.bin')
    (out/'graph.txt').write_text('\n'.join(lines)+'\n')
    record.update({'int8_batch':batch,'int8_operations':operations,'int8_layouts':layouts,'source_fixture':str(source)})
    (out/'fixture.json').write_text(json.dumps(record,indent=2))
    return record


def audit_config(fixtures,batch):
    """Reject silent matrix fallback or an omitted converted family."""
    total=0
    for label in up.suite.LABELS:
        directory=Path(up.suite.fixture(str(fixtures),label))
        r=json.loads((directory/'fixture.json').read_text());g=r['geometry']
        _,stages,_=up.suite._parse_graph(directory/'graph.txt')
        by_name={s['name']:s for s in stages}
        expected=['q','v']+(['wo'] if batch!='qv' else [])
        if batch in ('mlp','spatial'):expected+=['mlp-up','mlp-down']
        if batch=='spatial':
            if g['NPM']:expected+=['merge']
            if g['CIN']:expected+=['expand']
        if set(expected)!={op['name'] for op in r['int8_operations']}:
            raise ValueError('omitted or unexpected matrix operation: '+label)
        for op in r['int8_operations']:
            shader=next(w[1] for w in by_name[op['name']]['lines'] if w[0]=='shader')
            wanted='swin_mlp_down_q8.cso' if op['name'] in ('wo','mlp-down') else 'swin_mlp_up_q8.cso'
            if Path(shader).name!=wanted: raise ValueError('silent matrix fallback: '+label+'.'+op['name'])
        if ['shader','swin_attention.cso'] not in by_name['attention']['lines']:
            raise ValueError('dynamic attention contract changed unexpectedly')
        total+=len(expected)
    if total!={'qv':20,'output':30,'mlp':50,'spatial':54}[batch]:
        raise ValueError('unexpected cumulative matrix count')
    return total


def field_metrics(results,baseline):
    metrics={}
    for p in results.glob('out-*.bin'):
        if not (baseline/p.name).is_file(): continue
        a=np.fromfile(baseline/p.name,np.uint8);b=np.fromfile(p,np.uint8)
        metrics[p.name]={**up.eq._field(a,b),'non_finite':int((~np.isfinite(up.eq.E4M3[b])).sum()),
            'saturated':int(((b&127)==126).sum())}
    return metrics


def remeasure_fields(out,baseline):
    """Small, hash-tied rerun when disposable bulk outputs were already pruned."""
    checked=json.loads((out/'cpu.json').read_text())
    graph=out/'validated-chain/graph.txt';target=out/'field-check';target.mkdir(exist_ok=False)
    lines=[line for line in graph.read_text().splitlines()
           if not line.startswith('dump ') or line.split()[-1].startswith('out-')]
    (target/'graph.txt').write_text('\n'.join(lines)+'\n')
    results=up.run_graph(target,target/'results')
    checks={p.name:up.eq.sha(p)==checked['hashes'][p.name] for p in results.glob('*.bin')}
    metrics=field_metrics(results,baseline)
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'fields':metrics,
            'cpu_report_sha256':up.eq.sha(out/'cpu.json'),'probe_sha256':up.eq.sha(ROOT/'build/native/Release/m_network_stage.exe')}
    (out/'fields.json').write_text(json.dumps(report,indent=2))
    prune_tensor_run(results)
    return report


def validate_chain(fixtures, results):
    """All-element CPU validation on operands produced by this cumulative chain."""
    report={'checks':{},'operators':{},'layouts':{}}
    for label in up.suite.LABELS:
        directory=Path(up.suite.fixture(str(fixtures),label))
        record=json.loads((directory/'fixture.json').read_text()); g=record['geometry']
        tag=re.sub(r'[^A-Za-z0-9]+','-',label)
        def read(name,dtype='<f2',shape=None):
            a=np.fromfile(results/f'{tag}-dbg-{name}.bin',dtype)
            return a.reshape(shape) if shape else a
        for op in record['int8_operations']:
            name=label+'.'+op['name']; M,K,N=op['M'],op['K'],op['N']
            A=read(op['A'],shape=(M,K)); W=np.fromfile(directory/op['W'],'<f4').reshape(K,N)
            codes,scales=up.eq.quantize_rows_float(A); wc,ws,_=up.eq.quantize_weights(W)
            acc=codes.astype(np.int64)@wc.astype(np.int64)
            flt=((acc.astype(np.float32)*scales)*ws[None,:]).astype(np.float32)
            C0=(read(op['residual'],shape=(M,N)) if op['residual'] else
                np.broadcast_to(np.fromfile(directory/(op['bias']+'.bin'),'<f4'),(M,N)))
            flt=(flt+C0.astype(np.float32)).astype(np.float32); half=flt.astype('<f2')
            checks={}
            for key,want,dtype in [('codes',codes,np.int8),('scales',scales.ravel(),'<f4'),
                                   ('acc',acc.astype('<i4'),'<i4'),('float',flt,'<f4')]:
                checks[key]=read(op[key],dtype).tobytes()==np.ascontiguousarray(want).tobytes()
            checks['half']=read(op['Y']).tobytes()==half.tobytes()
            checks['weights']=(directory/(op['weight_codes']+'.bin')).read_bytes()==np.ascontiguousarray(wc.T).tobytes()
            checks['scales_offline']=(directory/(op['weight_scales']+'.bin')).read_bytes()==ws.astype('<f4').tobytes()
            checks['int32_safe']=int(np.abs(acc).max())<=K*127*127<2**24
            checks['finite_unsaturated']=bool(np.isfinite(half).all() and (np.abs(half)<65504).all())
            rng=np.random.default_rng(28); scalar=[]
            for row,col in [(0,0),(M-1,N-1)]+[(int(rng.integers(M)),int(rng.integers(N))) for _ in range(6)]:
                c,s=up.scalar_row(A[row]); total=up.eq.scalar_dot(c,wc[:,col])
                v=np.float32(np.float32(np.float32(total)*s)*ws[col])+np.float32(C0[row,col])
                scalar.append(bool(np.array_equal(c,codes[row]) and s.tobytes()==scales[row,0].tobytes()
                    and total==acc[row,col] and np.float16(v).tobytes()==half[row,col].tobytes()))
            checks['scalar']=all(scalar)
            for key,ok in checks.items(): report['checks'][name+'.'+key]=ok
            fp=up.lf.me.mma_chain(A.astype(np.float64),W.astype(np.float64),C0=C0.astype(np.float64)).astype('<f2')
            report['operators'][name]={'checks':checks,'shape':[M,K,N],'error_vs_same_input_fp16':up.eq.compare(half,fp),
                'acc_max_abs':int(np.abs(acc).max()),'scalar_probes':len(scalar),
                'activation_endpoints':int((np.abs(codes.astype(np.int16))==127).sum()),
                'source_A_sha256':up.eq.sha(results/f'{tag}-dbg-{op["A"]}.bin')}
            if not all(checks.values()):
                report['status']='FAIL';return report
            print(name, 'CPU/GPU exact', 'relative RMS',report['operators'][name]['error_vs_same_input_fp16']['relative_rms'],flush=True)
        for layout in record['int8_layouts']:
            kind=layout['kind']; M,N=layout['shape']
            if kind=='gelu':
                expected=up.lf.me.q8(up.lf.me.gelu_poly(read(layout['A'],shape=(M,N)))).astype('<f2')
                got=read(layout['Y'],shape=(M,N))
            elif kind=='unpack':
                raw=read(layout['A'],np.uint8).reshape(N//32,M,32).transpose(1,0,2).reshape(M,N)
                expected=mw.e4m3_decode(raw).astype('<f2');got=read(layout['Y'],shape=(M,N))
            elif kind=='gather':
                a=read(layout['A']).reshape(g['GH'],g['GW'],g['C'])
                expected=up.lf.me.q8(a.reshape(g['GH']//2,2,g['GW']//2,2,g['C']).transpose(0,2,1,3,4).reshape(M,N)).astype('<f2')
                got=read(layout['Y'],shape=(M,N))
            else:
                a=read(layout['A'],shape=(g['GH']//2,g['GW']//2,N))
                a=a[g['sy']//2:g['sy']//2+g['H']//2,g['sx']//2:g['sx']//2+g['W']//2]
                codes=mw.e4m3_encode(np.clip(a.astype(np.float32),-6.2831855,6.2831855).astype('<f2'))
                expected=codes.reshape(g['H']//2,g['W']//2,N//32,32).transpose(2,0,1,3).copy()
                got=read(layout['Y'],np.uint8)
            ok=got.tobytes()==np.ascontiguousarray(expected).tobytes()
            report['checks'][label+'.layout.'+kind]=ok;report['layouts'][label+'.'+kind]=ok
            if not ok: report['status']='FAIL';return report
    report['status']='PASS' if all(report['checks'].values()) else 'FAIL'
    return report


def prune_tensor_run(directory, keep=()):
    """Only disposable .bin outputs created by this runner; reports/hashes retained."""
    for p in directory.glob('*.bin'):
        if p.name not in keep: p.unlink()


def synthetic_unpack_gate(out):
    out.mkdir(parents=True,exist_ok=False)
    raw=(np.arange(32*128,dtype=np.uint32)*73+19).astype(np.uint8)
    raw[(raw&127)==127]=126;raw[::17]=128
    expected=mw.e4m3_decode(raw.reshape(4,32,32).transpose(1,0,2).reshape(32,128)).astype('<f2')
    (out/'input.bin').write_bytes(raw.tobytes())
    p=np.zeros(40,'<u4');p[0]=96;p[9]=8;p[10]=4;p[16]=128
    (out/'params.bin').write_bytes(p.tobytes())
    (out/'graph.txt').write_text('version 1\nresource input 4096 input.bin\nresource output 8192\nresource scratch 16\n'
        'stage unpack\nshader swin_cumulative_unpack.cso\nthreads 64 1 1\ngroups 1 1 1\nparams params.bin 160\n'
        'srv 0 input\nsrv 1 input\nsrv 2 input\nuav 0 output\nuav 1 scratch\ndump output 0 8192 output.bin\n')
    result=up.run_graph(out,out/'results')
    return (result/'output.bin').read_bytes()==expected.tobytes()


def run_batch(layers,out,batch,baseline,previous,rga,prune_bulk=False):
    out.mkdir(parents=True,exist_ok=False)
    fixtures=out/'layers';fixtures.mkdir()
    for label in up.suite.LABELS:
        prepare_layer(Path(up.suite.fixture(str(layers),label)),Path(up.suite.fixture(str(fixtures),label)),batch)
    audit_config(fixtures,batch)
    chain=out/'validated-chain'
    up.suite.build_chain(str(fixtures),str(chain),cumulative_diagnostics=True)
    results=up.run_graph(chain,out/'diagnostics-0')
    _,_,declared=up.suite._parse_graph(chain/'graph.txt')
    for d in declared:
        if (results/d['file']).stat().st_size!=d['bytes']:
            raise RuntimeError('incomplete diagnostic output: '+d['file'])
    checked=validate_chain(fixtures,results)
    checked['hashes']={p.name:up.eq.sha(p) for p in results.glob('*.bin')}
    checked['shaders']={p.name:up.eq.sha(p) for p in (ROOT/'build/native').glob('*.cso')}
    checked['probe_sha256']=up.eq.sha(ROOT/'build/native/Release/m_network_stage.exe')
    (out/'cpu.json').write_text(json.dumps(checked,indent=2))
    if checked['status']!='PASS': raise RuntimeError('CPU/layout gate failed; no batch advancement')
    prune_tensor_run(results)
    repeat=up.run_graph(chain,out/'diagnostics-1',1)
    checked['deterministic_complete_writes']=all(up.eq.sha(repeat/name)==h for name,h in checked['hashes'].items())
    checked['debug_zero']=all('D3D12 debug: zero error/corruption messages (0 total)' in (d/'execution.log').read_text()
                              for d in (results,repeat))
    if not checked['deterministic_complete_writes'] or not checked['debug_zero']:
        checked['status']='FAIL'
    (out/'cpu.json').write_text(json.dumps(checked,indent=2));prune_tensor_run(repeat)
    if checked['status']!='PASS': raise RuntimeError('determinism/resource gate failed')
    isa={}
    matrix_shaders=['swin_mlp_up_q8.cso','swin_mlp_down_q8.cso']
    if batch=='spatial': matrix_shaders.append('dec0_q8_matmul.cso')
    for name in matrix_shaders:
        shader=ROOT/'build/native'/name;digest=up.eq.sha(shader);stem=name.replace('.cso','')
        done=subprocess.run([str(rga),'-s','dx12','-c','gfx1101','--cs-blob',str(shader),
            '--isa',str(out/stem)],capture_output=True,text=True)
        (out/(stem+'-rga.log')).write_text(done.stdout+done.stderr)
        if done.returncode: raise RuntimeError('ISA inspection failed')
        text=(out/('gfx1101_'+stem+'_comp.isa')).read_text()
        isa[name]={'sha256':digest,'dot4':len(re.findall('v_dot4_i32_i8',text)),
                   'fp16_matrix':len(re.findall(r'v_(?:pk_)?(?:mul|fma|mac)_f16|v_wmma',text))}
        if not isa[name]['dot4'] or isa[name]['fp16_matrix'] or digest!=up.eq.sha(shader):
            raise RuntimeError('native ISA/hash gate failed')
    frame=out/'frame'
    cmd=[sys.executable,str(ROOT/'scripts/m_b6_frame.py'),'--out',str(frame),'--layers',str(layers),
         '--cumulative-mode',batch,'--cumulative-layers',str(fixtures),'--fp16-frame',str(baseline)]
    if previous: cmd+=['--previous-frame',str(previous)]
    done=subprocess.run(cmd,capture_output=True,text=True)
    (out/'frame.log').write_text(done.stdout+done.stderr)
    if not (frame/'b6.json').is_file(): raise RuntimeError('full frame failed: '+done.stderr[-1200:])
    f=json.loads((frame/'b6.json').read_text())
    checks={'cpu':checked['status']=='PASS','execution':f['execution']['status']=='PASS',
        'frame_deterministic':f['repeat']['fresh'] and f['repeat']['deterministic'],
        'enc0_input_identical':up.eq.sha(frame/'enc0-native/out-tokens.bin')==up.eq.sha(layers/'enc1/in.bin'),
        'probe_hash':f['inputs']['network_probe_sha256']==checked['probe_sha256'],
        'shader_hashes':f['inputs']['shaders']==checked['shaders']}
    for name,h in checked['hashes'].items():
        if name.startswith('out-'):
            checks['frame_handoff_'+name]=up.eq.sha(frame/'chain-native-results'/name)==h
    # The actual executed resources, bindings, parameters and shaders must match;
    # only diagnostics and the byte-identical enc0 input filename may differ.
    def fingerprint(path):
        resources,stages,_=up.suite._parse_graph(path)
        return {'resources':[(r['name'],r['bytes'],up.eq.sha(r['init']) if r['init'] else None) for r in resources],
            'stages':[(s['name'],[(w[0],up.eq.sha(w[1]),w[2]) if w[0]=='params' else
                (w[0],Path(w[1]).name) if w[0]=='shader' else tuple(w) for w in s['lines']]) for s in stages]}
    checks['same_validated_executed_graph']=fingerprint(chain/'graph.txt')==fingerprint(frame/'chain-native/graph.txt')
    if batch=='spatial':
        import m09_dec0_expansion as dec
        d=frame/'dec0-native';a,w,b=dec.decode_operands((d/'in-tokens.bin').read_bytes(),(d/'in-weights.bin').read_bytes())
        c,s,wc,ws,acc,y,half=dec.cpu_reference(a,w,b)
        for name,want in [('q8a',c),('q8as',s),('acc',acc.astype('<i4')),('float',y),('expansion',half)]:
            checks['dec0_cpu_'+name]=(d/'results'/f'out-{name}.bin').read_bytes()==np.ascontiguousarray(want).tobytes()
    report={'status':'PASS' if all(checks.values()) else 'FAIL','batch':batch,'checks':checks,
        'coverage':inventory(layers,batch,validated=all(checks.values())),'isa':isa,
        'frame':f['comparisons']['native'],'direct':f['direct_comparisons'],
        'image_regression':f['fidelity']['regression'],
        'cpu_report_sha256':up.eq.sha(out/'cpu.json'),'frame_report_sha256':up.eq.sha(frame/'b6.json'),
        'frame_boundaries':len(f['boundaries']),'repeat':f['repeat'],'artifact_policy':'bulk diagnostic .bin files pruned after recorded full CPU/digest checks; reports, graph, weights and raw frame retained'}
    (out/'batch.json').write_text(json.dumps(report,indent=2))
    if prune_bulk and report['status']=='PASS':
        keep={frame/'downsample-native/results/out-out.bin',frame/'chain-native-results/out-dec1.bin'}
        for p in frame.rglob('*.bin'):
            if p not in keep: p.unlink()
    print(json.dumps({'status':report['status'],'coverage':{k:v for k,v in report['coverage'].items() if k!='operators'},
        'frame':{k:report['frame'][k] for k in ('psnr_peak1_db','ssim_windowed_luma','frame_sha256')}},indent=2))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layers', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', choices=list(BATCHES))
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--previous', type=Path)
    ap.add_argument('--rga', type=Path)
    ap.add_argument('--prune-bulk', action='store_true', help='prune this batch\'s disposable frame tensor copies after PASS')
    a = ap.parse_args()
    if a.batch:
        if not (a.baseline and a.rga): ap.error('--batch requires --baseline and --rga')
        result=run_batch(a.layers.resolve(),a.out.resolve(),a.batch,a.baseline.resolve(),
                         a.previous.resolve() if a.previous else None,a.rga.resolve(),a.prune_bulk)
        sys.exit(0 if result['status']=='PASS' else 1)
    report = inventory(a.layers)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'inventory.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='operators'}, indent=2))
