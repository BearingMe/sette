#!/usr/bin/env python3
"""M13 robustness: the same configurations across the M10 content sequence.

One canonical frame is not evidence of general quality.  This driver reuses the
M10 frame inputs and the FP16 oracle frames produced from identical inputs, and
measures every configuration on each selected content variant.  The captured
reference only exists for the canonical content (frames 0, 1, 10, 11), so those
are the only frames with an absolute PSNR; every frame has a direct
configuration-vs-FP16 measurement.

  m13_sequence.py --out DIR --configs full-int8,full-int8+asym --frames 0,2,6,7
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import m_b6_frame as b6
import m13_quality as quality
import m13_selective as selective

TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
M10 = TEMP / 'm10-sequence-20261030'
REFERENCE = TEMP / 'm-capture-run/fixtures/downsample/ref-out.bin'
CANONICAL = {0, 1, 10, 11}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(ref, test):
    row = b6.metrics(ref, test)
    return {k: row[k] for k in ('psnr_peak1_db', 'ssim_windowed_luma', 'rms',
                                'mean_abs', 'p99_abs', 'max_abs',
                                'nan_or_inf_pairs', 'saturated_test')}


def fp16_frame(index):
    root = M10 / 'fp16' / f'frame-{index:02d}'
    for relative in ('downsample-native/results/out-out.bin', 'frame.bin'):
        path = root / relative
        if path.is_file():
            return path
    raise SystemExit(f'no FP16 frame for {index}: {root}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--frames', default='0,2,3,4,5,6,7,8,9')
    ap.add_argument('--m10', type=Path, default=M10)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    sequence = json.loads((a.m10 / 'sequence.json').read_text())
    indices = [int(x) for x in a.frames.split(',') if x.strip()]
    reference = b6.frame(REFERENCE)
    oracles = {i: b6.frame(fp16_frame(i)) for i in indices}
    names = [x.strip() for x in a.configs.split(',') if x.strip()]
    report = {'m10': str(a.m10), 'frames': indices,
              'frame_inputs': {str(i): sequence['frames'][i]['colour'] for i in indices},
              'configs': {}, 'status': 'running'}
    for name in names:
        cfg = quality.parse_config(name)
        store = a.out / 'configs' / re.sub(r'[^A-Za-z0-9.+-]+', '-', name)[:120]
        selective.build(quality.INT8_LAYERS, quality.FP16_LAYERS, store / 'layers',
                        cfg['restore'], asym=cfg['asym'])
        checked = selective.audit(store / 'layers')
        if checked['status'] != 'PASS':
            raise SystemExit(f'{name}: {checked["problems"]}')
        attention = any(f == 'attention' for _, f in cfg['restore'])
        rows = {}
        for index in indices:
            run = store / f'frame-{index:02d}'
            command = [sys.executable, str(HERE / 'm_b6_frame.py'), '--out', str(run),
                       '--no-repeat', '--cumulative-mode', 'selective',
                       '--cumulative-layers', str(store / 'layers'),
                       '--attention-mode', 'fp16' if attention else 'int8',
                       '--enc0-mode', 'int8-embedding' if cfg['enc0'] else 'fp16',
                       '--dec0-mode', 'int8-expansion' if cfg['dec0'] else 'fp16',
                       '--colour', str(sequence['frames'][index]['colour'])]
            done = subprocess.run(command, capture_output=True, text=True)
            frame = run / 'downsample-native/results/out-out.bin'
            if not frame.is_file():
                raise SystemExit(f'{name} frame {index}: no frame\n{done.stderr[-900:]}')
            produced = b6.frame(frame)
            digest = sha(frame)
            kept = store / f'frame-{index:02d}.bin'
            shutil.move(str(frame), str(kept))
            for child in list(run.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            row = {'frame': str(kept), 'frame_sha256': digest,
                   'vs_fp16': metrics(oracles[index], produced)}
            if index in CANONICAL:
                row['vs_reference'] = metrics(reference, produced)
            rows[str(index)] = row
            psnr = row['vs_fp16']['psnr_peak1_db']
            print(f'{name} frame {index:02d}: vs fp16 {psnr:.4f} dB, '
                  f'ssim {row["vs_fp16"]["ssim_windowed_luma"]:.6f}'
                  + (f', vs ref {row["vs_reference"]["psnr_peak1_db"]:.4f} dB'
                     if index in CANONICAL else ''), flush=True)
        values = [r['vs_fp16']['psnr_peak1_db'] for r in rows.values()]
        canonical = [r['vs_reference']['psnr_peak1_db'] for i, r in rows.items()
                     if int(i) in CANONICAL]
        report['configs'][name] = {
            'asym': cfg['asym'],
            'restore': sorted(f'{l}:{f}' for l, f in cfg['restore']),
            'enc0_int8': cfg['enc0'], 'dec0_int8': cfg['dec0'],
            'coverage': selective.coverage(quality.INT8_LAYERS, quality.FP16_LAYERS,
                                           cfg['restore'], enc0_int8=cfg['enc0'],
                                           dec0_int8=cfg['dec0']),
            'per_frame': rows,
            'vs_fp16_median': statistics.median(values),
            'vs_fp16_min': min(values), 'vs_fp16_max': max(values),
            'vs_reference_canonical': canonical,
            'vs_reference_mean': statistics.mean(canonical) if canonical else None,
        }
        (a.out / 'm13-sequence.json').write_text(json.dumps(report, indent=2))
    base = report['configs'].get('full-int8')
    summary = []
    for name, entry in report['configs'].items():
        summary.append({
            'config': name,
            'vs_fp16_median': entry['vs_fp16_median'],
            'vs_fp16_min': entry['vs_fp16_min'],
            'vs_fp16_max': entry['vs_fp16_max'],
            'vs_reference_mean': entry['vs_reference_mean'],
            'delta_ref_vs_full_int8': (entry['vs_reference_mean'] -
                                       base['vs_reference_mean']) if base and
                                      entry['vs_reference_mean'] and
                                      base['vs_reference_mean'] else None,
            'int8_operators': entry['coverage']['int8_operators'],
            'int8_macs': entry['coverage']['int8_macs'],
        })
    report['summary'] = summary
    report['status'] = 'PASS'
    (a.out / 'm13-sequence.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
