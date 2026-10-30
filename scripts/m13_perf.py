#!/usr/bin/env python3
"""M13 performance: interleaved GPU timing for mixed-precision configurations.

Each configuration is built once, warmed once, then every configuration is run
once per round in the same order so clock/power drift is shared.  Timings are
D3D12 timestamp deltas from the `*_timed` probes; the final frame of every run
must stay stable across rounds (a changed result aborts the comparison).

  m13_perf.py --out DIR --configs full-int8,full-int8+asym --rounds 3
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
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import m13_quality as quality
import m13_selective as selective

TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
PROBE = REPO / 'build/native/Release/m_image_stage_timed.exe'
NETWORK = REPO / 'build/native/Release/m_network_stage_timed.exe'
SHADERS = REPO / 'build/native'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, layers, out, extra):
    out = Path(out)
    command = [sys.executable, str(HERE / 'm_b6_frame.py'), '--out', str(out),
               '--no-repeat', '--cumulative-mode', 'selective',
               '--cumulative-layers', str(layers),
               '--attention-mode', 'fp16' if extra['attention_fp16'] else 'int8',
               '--enc0-mode', 'int8-embedding' if extra['enc0'] else 'fp16',
               '--dec0-mode', 'int8-expansion' if extra['dec0'] else 'fp16',
               '--probe', str(PROBE), '--network-probe', str(NETWORK),
               '--timing-dir', str(out / 'timing'), '--shaders', str(SHADERS)]
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True)
    wall = time.perf_counter() - started
    frame = out / 'downsample-native/results/out-out.bin'
    if not frame.is_file():
        raise SystemExit(f'{config}: no frame\n{done.stderr[-1200:]}')
    digest = sha(frame)
    # Keep only the timing records, the report and the raw frame; the stage
    # tensors are reproducible and would otherwise fill the disk.
    shutil.move(str(frame), str(out / 'frame.bin'))
    for child in list(out.iterdir()):
        if child.name in ('timing', 'b6.json', 'frame.bin'):
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    rows = []
    for path in sorted((out / 'timing').glob('*.json')):
        payload = json.loads(path.read_text())
        rows += [{'process': path.stem, 'label': d['label'], 'ms': d['ms']}
                 for d in payload['dispatches']]
    if not rows:
        raise SystemExit(f'{config}: no timestamps')
    return {'frame_sha256': digest, 'gpu_ms': sum(r['ms'] for r in rows),
            'dispatches': len(rows), 'wall_seconds': wall,
            'by_process': {p: sum(r['ms'] for r in rows if r['process'] == p)
                           for p in sorted({r['process'] for r in rows})}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=1)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    names = [x.strip() for x in a.configs.split(',') if x.strip()]
    entries = {}
    for name in names:
        cfg = quality.parse_config(name)
        store = a.out / 'configs' / re.sub(r'[^A-Za-z0-9.+-]+', '-', name)[:120]
        selective.build(quality.INT8_LAYERS, quality.FP16_LAYERS, store / 'layers',
                        cfg['restore'], asym=cfg['asym'])
        checked = selective.audit(store / 'layers')
        if checked['status'] != 'PASS':
            raise SystemExit(f'{name}: {checked["problems"]}')
        entries[name] = {'cfg': cfg, 'layers': store / 'layers',
                         'attention_fp16': any(f == 'attention' for _, f in cfg['restore']),
                         'enc0': cfg['enc0'], 'dec0': cfg['dec0'], 'runs': []}
    for name in names:
        entry = entries[name]
        run(name, entry['layers'], a.out / 'warmup' / re.sub(
            r'[^A-Za-z0-9.+-]+', '-', name)[:80], entry)
        print(f'{name} warmup done', flush=True)
    for r in range(a.rounds):
        for name in names:
            entry = entries[name]
            safe = re.sub(r'[^A-Za-z0-9.+-]+', '-', name)[:100]
            row = run(name, entry['layers'], a.out / 'runs' / f'{safe}-{r:02d}', entry)
            entry['runs'].append(row)
            print(f'{name} round {r}: {row["gpu_ms"]:.3f} ms, '
                  f'{row["dispatches"]} dispatches, {row["wall_seconds"]:.1f} s wall',
                  flush=True)
    report = {'rounds': a.rounds, 'configs': {}}
    for name, entry in entries.items():
        values = [r['gpu_ms'] for r in entry['runs']]
        hashes = {r['frame_sha256'] for r in entry['runs']}
        report['configs'][name] = {
            'restore': sorted(f'{l}:{f}' for l, f in entry['cfg']['restore']),
            'asym': entry['cfg']['asym'], 'enc0_int8': entry['enc0'],
            'dec0_int8': entry['dec0'],
            'gpu_ms_median': statistics.median(values),
            'gpu_ms_min': min(values), 'gpu_ms_max': max(values),
            'dispatches': entry['runs'][0]['dispatches'],
            'frame_sha256': sorted(hashes),
            'deterministic': len(hashes) == 1,
            'runs': [{k: r[k] for k in ('gpu_ms', 'dispatches', 'wall_seconds')}
                     for r in entry['runs']],
            'by_process_median': {
                p: statistics.median([r['by_process'][p] for r in entry['runs']])
                for p in entry['runs'][0]['by_process']},
        }
    base = report['configs'].get('full-int8', {}).get('gpu_ms_median')
    for name, entry in report['configs'].items():
        entry['speedup_vs_full_int8'] = (base / entry['gpu_ms_median']) if base else None
        entry['quality'] = None
    (a.out / 'm13-perf.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {'gpu_ms_median': round(v['gpu_ms_median'], 3),
                          'min': round(v['gpu_ms_min'], 3),
                          'max': round(v['gpu_ms_max'], 3),
                          'dispatches': v['dispatches'],
                          'deterministic': v['deterministic']}
                      for k, v in report['configs'].items()}, indent=2))
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
