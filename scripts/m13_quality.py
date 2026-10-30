#!/usr/bin/env python3
"""M13 quality recovery: run explicit mixed-precision configurations.

Every configuration is a declarative INT8/FP16 split (see `m13_selective.py`):
the layer graph is audited before execution, the frame is run through the
unmodified B6 harness, and the saved raw frames are re-measured here against the
captured reference and against the FP16 oracle frame.

The accepted full-INT8 frame and the FP16 oracle are configuration entries, not
assumptions: both are executed in the same run and their hashes are checked.

  m13_quality.py --out DIR --configs full-int8,fp16,family:q,...
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import m_b6_frame as b6
import m13_selective as selective

TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
INT8_LAYERS = TEMP / 'm09h-complete-chain-20261030/layers'
FP16_LAYERS = TEMP / 'm-capture-run/layers'
FP16_FRAME = TEMP / 'm09g-fp16-final-20261030/downsample-native/results/out-out.bin'
REFERENCE = TEMP / 'm-capture-run/fixtures/downsample/ref-out.bin'
PROBE = REPO / 'build/native/Release/m_image_stage.exe'
NETWORK = REPO / 'build/native/Release/m_network_stage.exe'
PROBE_TIMED = REPO / 'build/native/Release/m_image_stage_timed.exe'
NETWORK_TIMED = REPO / 'build/native/Release/m_network_stage_timed.exe'
SHADERS = REPO / 'build/native'

ACCEPTED = {
    'full-int8': 'f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00',
    'fp16': 'dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_config(name):
    """{'restore': {(label, family)}, 'enc0': bool, 'dec0': bool}.

    A trailing `+enc0-fp16` / `+dec0-fp16` (or both) restores that image stage
    independently of the layer graph.
    """
    cfg = {'restore': set(), 'enc0': True, 'dec0': True, 'timed': False,
           'asym': False}
    parts = name.split('+')
    base, modifiers = parts[0], parts[1:]
    if base == 'full-int8':
        pass
    elif base == 'fp16':
        cfg['restore'] = {(label, family) for label in selective.LABELS
                          for family in selective.available(INT8_LAYERS, label)}
        cfg['enc0'] = cfg['dec0'] = False
    elif base.startswith('family:'):
        family = base.split(':', 1)[1]
        if family not in selective.FAMILIES:
            raise SystemExit(f'unknown family {family}')
        cfg['restore'] = {(label, family) for label in selective.LABELS
                          if family in selective.available(INT8_LAYERS, label)}
    elif base.startswith('restore:'):
        cfg['restore'] = selective.parse_restore(base.split(':', 1)[1])
    elif base in ('enc0-fp16', 'image-fp16'):
        cfg['enc0'] = False
        if base == 'image-fp16':
            cfg['dec0'] = False
    elif base == 'dec0-fp16':
        cfg['dec0'] = False
    elif base == 'int8':
        pass
    else:
        raise SystemExit(f'unknown configuration {name}')
    for mod in modifiers:
        if mod in ('enc0-fp16', 'enc0fp16'):
            cfg['enc0'] = False
        elif mod in ('dec0-fp16', 'dec0fp16'):
            cfg['dec0'] = False
        elif mod in ('both-fp16', 'image-fp16'):
            cfg['enc0'] = cfg['dec0'] = False
        elif mod == 'asym':
            cfg['asym'] = True
        else:
            raise SystemExit(f'unknown configuration modifier {mod!r}')
    return cfg


def frame_metrics(ref, test):
    row = b6.metrics(ref, test)
    return {k: row[k] for k in ('psnr_peak1_db', 'psnr_peak_reference_db',
                                'ssim_windowed_luma', 'rms', 'mean_abs',
                                'p99_abs', 'max_abs', 'signed_mean',
                                'nan_or_inf_pairs', 'saturated_test')}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--int8-layers', type=Path, default=INT8_LAYERS)
    ap.add_argument('--fp16-layers', type=Path, default=FP16_LAYERS)
    ap.add_argument('--fp16-frame', type=Path, default=FP16_FRAME)
    ap.add_argument('--reference', type=Path, default=REFERENCE)
    ap.add_argument('--repeat', action='store_true',
                    help='also run the B6 fresh-repeat determinism check')
    ap.add_argument('--timed', action='store_true',
                    help='use the *_timed probes and record per-dispatch GPU time')
    ap.add_argument('--repeats', type=int, default=1,
                    help='timed repeats per configuration (implies --timed)')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    names = [x.strip() for x in a.configs.split(',') if x.strip()]
    fp16 = b6.frame(a.fp16_frame)
    reference = b6.frame(a.reference)
    report = {'int8_layers': str(a.int8_layers), 'fp16_layers': str(a.fp16_layers),
              'fp16_frame': str(a.fp16_frame), 'fp16_frame_sha256': sha(a.fp16_frame),
              'reference': str(a.reference), 'reference_sha256': sha(a.reference),
              'accepted': ACCEPTED, 'configs': {}, 'status': 'running'}
    timed = a.timed or a.repeats > 1
    for name in names:
        cfg = parse_config(name)
        safe = re.sub(r'[^A-Za-z0-9.+-]+', '-', name)[:120]
        store = a.out / 'configs' / safe
        # Always build a selective variant, even with an empty restore set, so
        # the full-INT8 baseline is audited through exactly the same path.
        selective.build(a.int8_layers, a.fp16_layers, store / 'layers', cfg['restore'],
                        asym=cfg['asym'])
        audited = selective.audit(store / 'layers')
        if audited['status'] != 'PASS':
            raise SystemExit(f'{name}: {audited["problems"]}')
        layers = store / 'layers'
        restores_attention = any(f == 'attention' for _, f in cfg['restore'])
        repeats = []
        for repeat in range(a.repeats if timed else 1):
            run = store / f'run-{repeat:02d}'
            command = [sys.executable, str(HERE / 'm_b6_frame.py'), '--out', str(run),
                       '--cumulative-mode', 'selective', '--cumulative-layers', str(layers),
                       '--attention-mode', 'fp16' if restores_attention else 'int8',
                       '--enc0-mode', 'int8-embedding' if cfg['enc0'] else 'fp16',
                       '--dec0-mode', 'int8-expansion' if cfg['dec0'] else 'fp16',
                       '--fp16-frame', str(a.fp16_frame)]
            if not a.repeat:
                command.append('--no-repeat')
            if timed:
                command += ['--probe', str(PROBE_TIMED), '--network-probe', str(NETWORK_TIMED),
                            '--timing-dir', str(run / 'timing')]
            done = subprocess.run(command, capture_output=True, text=True)
            frame_path = run / 'downsample-native/results/out-out.bin'
            if not frame_path.is_file():
                raise SystemExit(f'{name} run {repeat}: no frame\n{done.stderr[-1200:]}')
            body = json.loads((run / 'b6.json').read_text())
            produced = b6.frame(frame_path)
            entry = {
                'frame': str(frame_path), 'frame_sha256': sha(frame_path),
                'vs_reference': frame_metrics(reference, produced),
                'vs_fp16': frame_metrics(fp16, produced),
                'execution_status': body['execution']['status'],
                'regression_status': body['fidelity']['regression']['status'],
                'bounds': len(body['boundaries']),
                'non_finite': body['non_finite'], 'repeat': body.get('repeat'),
            }
            if timed:
                rows = []
                for path in sorted((run / 'timing').glob('*.json')):
                    payload = json.loads(path.read_text())
                    rows += [{'process': path.stem, 'label': d['label'], 'ms': d['ms']}
                             for d in payload['dispatches']]
                entry['gpu_ms'] = float(sum(d['ms'] for d in rows))
                entry['dispatches'] = len(rows)
                entry['gpu_ms_by_process'] = {
                    process: float(sum(d['ms'] for d in rows if d['process'] == process))
                    for process in sorted({d['process'] for d in rows})}
            repeats.append(entry)
            def fmt(value, digits=4):
                return 'n/a' if value is None else f'{value:.{digits}f}'
            print(f'{name} run {repeat}: ref {fmt(entry["vs_reference"]["psnr_peak1_db"])} dB, '
                  f'fp16 {fmt(entry["vs_fp16"]["psnr_peak1_db"])} dB, '
                  f'ssim {fmt(entry["vs_reference"]["ssim_windowed_luma"], 6)}'
                  + (f', gpu {entry["gpu_ms"]:.3f} ms' if timed else ''))
            sys.stdout.flush()
            if not timed:
                break
        best = dict(repeats[0])
        if timed and len(repeats) > 1:
            best['gpu_ms_median'] = float(np.median([r['gpu_ms'] for r in repeats]))
            best['gpu_ms_min'] = float(np.min([r['gpu_ms'] for r in repeats]))
            best['gpu_ms_max'] = float(np.max([r['gpu_ms'] for r in repeats]))
        coverage = selective.coverage(a.int8_layers, a.fp16_layers, cfg['restore'],
                                      enc0_int8=cfg['enc0'], dec0_int8=cfg['dec0'])
        report['configs'][name] = {
            'restore': sorted(f'{l}:{f}' for l, f in cfg['restore']),
            'asym': cfg['asym'],
            'enc0_int8': cfg['enc0'], 'dec0_int8': cfg['dec0'],
            'coverage': coverage, 'runs': repeats, 'best': best,
            'accepted_hash_match': entry['frame_sha256'] == ACCEPTED.get(name),
        }
        (a.out / 'm13-quality.json').write_text(json.dumps(report, indent=2))

    # ---- verdict summary ---------------------------------------------------
    base = report['configs'].get('full-int8', {}).get('best')
    rows = []
    for name, entry in report['configs'].items():
        best = entry['best']
        rows.append({
            'config': name,
            'psnr_reference': best['vs_reference']['psnr_peak1_db'],
            'psnr_fp16': best['vs_fp16']['psnr_peak1_db'],
            'ssim_reference': best['vs_reference']['ssim_windowed_luma'],
            'delta_vs_full_int8': (best['vs_reference']['psnr_peak1_db'] -
                                   base['vs_reference']['psnr_peak1_db']) if base else None,
            'int8_operators': entry['coverage']['int8_operators'],
            'int8_macs': entry['coverage']['int8_macs'],
            'gpu_ms': best.get('gpu_ms_median'),
            'accepted_hash_match': entry['accepted_hash_match'],
        })
    report['summary'] = rows
    report['status'] = 'PASS' if all(
        r['psnr_reference'] is not None and r['ssim_reference'] is not None
        for r in rows) else 'FAIL'
    (a.out / 'm13-quality.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(rows, indent=2))
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
