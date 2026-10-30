#!/usr/bin/env python3
"""EXP-025 stage gate: validate the GPU handoff before attempting a hybrid frame.

The CPU-upload epilogue paths below are validation oracles only; the experimental
inference path uses the image executor's device-resident three-dispatch prefix.
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
import m09_dec0_expansion as ex
import m_b6_frame as b6
import m_stage_fixture as sf

ROLES = ('gate', 'mix', 'feat', 'cov')


def surface_metrics(path, reference, role):
    dtype = np.float16 if role == 'cov' else (np.int8 if role == 'feat' else np.uint8)
    a = np.fromfile(path, dtype).reshape(540,960,4)
    b = np.fromfile(reference, dtype).reshape(540,960,4)
    row = ex.metrics(a.reshape(-1,4), b.reshape(-1,4))
    row['identical_samples'] = float((a==b).mean())
    row['bytes'] = path.stat().st_size
    row['saturated_f16'] = int((np.abs(a.astype(np.float32))>=65504).sum()) if role=='cov' else 0
    return row


def stage_gate(fixture, out, probe, shaders):
    out.mkdir(parents=True, exist_ok=False)
    # Clone only authoritative fixture inputs/metadata, never historical dumps.
    def clone(label):
        dst = out/label; dst.mkdir()
        for name in ['stage.txt','fixture.json','params.bin','in-tokens.bin','in-weights.bin']:
            shutil.copyfile(fixture/name,dst/name)
        return dst
    fp16, hybrid = clone('fp16'), clone('int8')
    sf.prepare_dec0_mode(hybrid,'int8-expansion')
    outputs = {}
    checks = {}
    for mode,directory in [('fp16',fp16),('int8',hybrid)]:
        for seed in (0,1):
            outputs[mode,seed] = b6.run(probe,directory/'stage.txt',directory/f'run{seed}',shaders,seed=seed)
        checks[mode+'_complete_deterministic'] = all(
            (outputs[mode,0]/f'out-{r}.bin').stat().st_size == size and
            b6.sha(outputs[mode,0]/f'out-{r}.bin')==b6.sha(outputs[mode,1]/f'out-{r}.bin')
            for r,size in b6.declared_outputs(directory/'stage.txt').items())
    a,w,bias = ex.decode_operands((fixture/'in-tokens.bin').read_bytes(),(fixture/'in-weights.bin').read_bytes())
    codes,scales,wc,ws,acc,y,half = ex.cpu_reference(a,w,bias)
    gpu = outputs['int8',0]
    for name,want,dtype in [('q8a',codes,np.int8),('q8as',scales,np.float32),
                            ('acc',acc.astype(np.int32),np.int32),('float',y,np.float32),('expansion',half,np.float16)]:
        got = np.fromfile(gpu/f'out-{name}.bin',dtype).reshape(want.shape)
        checks['cpu_int8_'+name] = got.tobytes()==want.tobytes()
    # Exercise the same image epilogue independently of the new GPU handoff.
    cpu_fp16 = ex.me.mma_chain(a.astype(np.float64),w.astype(np.float64),
                               C0=np.broadcast_to(bias.astype(np.float64),(ex.M,ex.N))).astype(np.float16)
    for label,expansion,expected in [('oracle-int8',half,gpu),('oracle-fp16',cpu_fp16,outputs['fp16',0])]:
        directory = clone(label)
        (directory/'expansion.bin').write_bytes(expansion.tobytes())
        text = (directory/'stage.txt').read_text().replace('shader dec0_cs.cso','shader dec0_q8_epilogue.cso')
        text += f'srv buffer 2 expansion.bin {expansion.nbytes}\n'
        (directory/'stage.txt').write_text(text)
        res = b6.run(probe,directory/'stage.txt',directory/'results',shaders)
        checks[label+'_surfaces'] = all(b6.sha(res/f'out-{r}.bin')==b6.sha(expected/f'out-{r}.bin') for r in ROLES)
    surfaces = {}
    for r in ROLES:
        surfaces[r] = {'int8_vs_fp16': surface_metrics(gpu/f'out-{r}.bin',outputs['fp16',0]/f'out-{r}.bin',r),
                       'fp16_vs_capture': surface_metrics(outputs['fp16',0]/f'out-{r}.bin',fixture/f'ref-{r}.bin',r),
                       'int8_vs_capture': surface_metrics(gpu/f'out-{r}.bin',fixture/f'ref-{r}.bin',r)}
        checks[r+'_finite_unsaturated'] = surfaces[r]['int8_vs_fp16']['non_finite']==0 and surfaces[r]['int8_vs_fp16']['saturated_f16']==0
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
               'surfaces': surfaces, 'expansion_error': ex.metrics(half,cpu_fp16),
               'shaders': {n:b6.sha(shaders/n) for n in ['dec0_cs.cso','dec0_q8_epilogue.cso','dec0_q8_quant.cso','dec0_q8_matmul.cso']},
               'probe_sha256': b6.sha(probe),
               'inputs': {n:b6.sha(fixture/n) for n in ['in-tokens.bin','in-weights.bin','params.bin']}}
    (out/'stage.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'checks':checks,'surfaces':surfaces},indent=2))
    return report


def audit_frame(stage, stage_dir, frame_run, fp16_run, enc1q_run, shaders, rga):
    """Tie independently CPU-checked stage bytes to the actual full-frame run."""
    f = json.loads((frame_run/'b6.json').read_text())
    baseline = json.loads((fp16_run/'b6.json').read_text())
    q = json.loads((enc1q_run/'b6.json').read_text())
    res = frame_run/'dec0-native/results'
    checks = {'stage_gate': stage['status']=='PASS',
              'dec0_only': f['q_mode']=='fp16' and f['dec0_mode']=='int8-expansion',
              'frame_execution': f['execution']['status']=='PASS',
              'repeat': f['repeat']['fresh'] and f['repeat']['deterministic'],
              'same_neural_input': f['boundaries']['chain/out-dec1']==baseline['boundaries']['chain/out-dec1'],
              'fp16_baseline': baseline['comparisons']['native']['frame_sha256']=='dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
              'enc1q_baseline': q['comparisons']['native']['frame_sha256']=='0eb0185ea7e21f20617b67476519310ae2dfe8b402188f3207fb197b35299419',
              'same_tokens': b6.sha(frame_run/'dec0-native/in-tokens.bin')==stage['inputs']['in-tokens.bin']}
    for role in (*ROLES,*sf.DEC0_DIAGNOSTICS):
        checks['handoff_'+role] = b6.sha(res/f'out-{role}.bin')==b6.sha(stage_dir/'int8/run0'/f'out-{role}.bin')
    for name,digest in stage['shaders'].items():
        checks['shader_'+name] = digest==f['inputs']['shaders'][name]==b6.sha(shaders/name)
    graph = (frame_run/'chain-native/graph.txt').read_text()
    checks['chain_entirely_fp16'] = 'q8' not in graph and 'int8' not in graph
    logs = [(frame_run/f'dec0-{tag}/results/execution.log').read_text() for tag in ('native','repeat')]
    checks['d3d12_validation'] = all('D3D12 debug: zero error/corruption messages' in text for text in logs)
    subprocess.run([str(rga),'-s','dx12','-c','gfx1101','--cs-blob',str(shaders/'dec0_q8_matmul.cso'),
                    '--isa',str(stage_dir/'executed-matmul')],check=True,capture_output=True)
    path = stage_dir/'gfx1101_executed-matmul_comp.isa'
    text = path.read_text()
    isa = {'shader_sha256':b6.sha(shaders/'dec0_q8_matmul.cso'),'isa_sha256':b6.sha(path),
           'dot4':len(re.findall('v_dot4_i32_i8',text)),
           'fp16_matrix':len(re.findall(r'v_(?:pk_)?(?:mul|fma|mac)_f16|v_wmma',text))}
    checks['native_isa'] = isa['dot4']>0 and isa['fp16_matrix']==0
    report = {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'isa':isa,
              'frame_sha256':f['comparisons']['native']['frame_sha256'],
              'frame_report_sha256':b6.sha(frame_run/'b6.json'),
              'stage_report_sha256':b6.sha(stage_dir/'stage.json')}
    (stage_dir/'integration.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--probe',type=Path,default=ex.ROOT/'build/native/Release/m_image_stage.exe')
    ap.add_argument('--shaders',type=Path,default=ex.ROOT/'build/native')
    ap.add_argument('--frame-run',type=Path,help='audit this already-executed B6 dec0-only run')
    ap.add_argument('--fp16-run',type=Path)
    ap.add_argument('--enc1q-run',type=Path)
    ap.add_argument('--rga',type=Path)
    args=ap.parse_args()
    if args.frame_run and not all([args.fp16_run,args.enc1q_run,args.rga]):
        ap.error('--frame-run requires --fp16-run, --enc1q-run and --rga')
    row=stage_gate(args.fixture,args.out,args.probe,args.shaders)
    if row['status']=='PASS' and args.frame_run:
        row=audit_frame(row,args.out,args.frame_run,args.fp16_run,args.enc1q_run,args.shaders,args.rga)
    sys.exit(0 if row['status']=='PASS' else 1)
