#!/usr/bin/env python3
"""M12: per-dispatch GPU timing of the complete frame, FP16 vs native INT8.

Runs the approved B6 harness with the `*_timed` probes, which record D3D12
timestamp queries around every dispatch.  Each repeat must reproduce the
accepted frame hash, so timing never hides a changed result.

Reported quantities are GPU-side timestamp deltas, not CPU dispatch time.  The
subprocess wall time is recorded for context only.

  m12_profile.py --out DIR [--repeats 5] [--cumulative-layers DIR]
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMP = Path.home() / 'AppData/Local/Temp/opencode'
ACCEPTED = {
    'fp16': 'dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
    'int8': 'f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00',
}
MODES = {
    'fp16': [],
    'int8': ['--enc0-mode', 'int8-embedding', '--attention-mode', 'int8',
             '--cumulative-mode', 'spatial'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = sorted(values)
    return {'median': statistics.median(values), 'min': values[0], 'max': values[-1],
            'p95': values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))) )],
            'samples': len(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cumulative-layers', type=Path,
                    default=TEMP / 'm09h-complete-chain-20261030' / 'layers')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=1,
                    help='unmeasured runs per mode before the measured repeats')
    ap.add_argument('--keep-runs', action='store_true',
                    help='keep every stage output (large); default prunes to timing+b6+frame')
    ap.add_argument('--modes', default='fp16,int8')
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f'unknown mode {mode}')
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    report = {'repeats': a.repeats, 'modes': modes, 'runs': {},
              'accepted_frame': ACCEPTED, 'status': 'running'}
    def execute(mode, out, timing):
        command = [sys.executable, str(HERE / 'm_b6_frame.py'), '--out', str(out),
                   '--no-repeat',
                   '--probe', str(REPO / 'build/native/Release/m_image_stage_timed.exe'),
                   '--network-probe', str(REPO / 'build/native/Release/m_network_stage_timed.exe'),
                   '--timing-dir', str(timing), *MODES[mode]]
        if mode != 'fp16':
            command += ['--cumulative-layers', str(a.cumulative_layers)]
        started = time.perf_counter()
        done = subprocess.run(command, capture_output=True, text=True)
        return time.perf_counter() - started, done

    def measure(mode, repeat):
        out = a.out / mode / f'run-{repeat:02d}'
        timing = out / 'timing'
        wall, done = execute(mode, out, timing)
        frame = out / 'downsample-native' / 'results' / 'out-out.bin'
        if not frame.is_file():
            raise RuntimeError(f'{mode} run {repeat}: no frame\n{done.stdout[-500:]}')
        digest = sha(frame)
        if digest != ACCEPTED[mode]:
            raise RuntimeError(f'{mode} run {repeat}: frame {digest} != accepted')
        dispatches = []
        for path in sorted(timing.glob('*.json')):
            payload = json.loads(path.read_text())
            for entry in payload['dispatches']:
                dispatches.append({'process': path.stem, 'label': entry['label'],
                                   'ms': entry['ms'], 'ticks': entry['ticks']})
        if not a.keep_runs:
            shutil.move(str(frame), str(out / 'frame.bin'))
            for child in list(out.iterdir()):
                if child.name in ('timing', 'frame.bin', 'b6.json'):
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        row = {'repeat': repeat, 'wall_seconds': wall,
               'frame_sha256': digest, 'dispatches': dispatches}
        print(f'{mode} repeat {repeat}: {len(dispatches)} dispatches, '
              f'{sum(d["ms"] for d in dispatches):.3f} ms GPU, {wall:.2f} s wall')
        sys.stdout.flush()
        return row

    for mode in modes:
        for warm in range(a.warmup):
            warm_dir = a.out / mode / f'warm-{warm:02d}'
            execute(mode, warm_dir, warm_dir / 'timing')
            shutil.rmtree(warm_dir)
            print(f'{mode} warmup {warm}', flush=True)

    # Interleave the modes so clock/power drift cannot favour one of them.
    rows_by_mode = {mode: [] for mode in modes}
    for repeat in range(a.repeats):
        for mode in modes:
            rows_by_mode[mode].append(measure(mode, repeat))

    for mode in modes:
        rows = rows_by_mode[mode]
        by_process = {}
        by_label = {}
        totals = []
        for row in rows:
            total = sum(d['ms'] for d in row['dispatches'])
            totals.append(total)
            grouped = {}
            for d in row['dispatches']:
                grouped.setdefault(d['process'], 0.0)
                grouped[d['process']] += d['ms']
                key = f'{d["process"]}:{d["label"]}'
                by_label.setdefault(key, []).append(d['ms'])
            for name, value in grouped.items():
                by_process.setdefault(name, []).append(value)
        entry = {
            'dispatch_count': len(rows[0]['dispatches']),
            'total_gpu_ms': stats(totals),
            'wall_seconds_median': statistics.median([r['wall_seconds'] for r in rows]),
            'process_gpu_ms': {k: stats(v) for k, v in sorted(by_process.items())},
            'dispatch_gpu_ms': {k: stats(v) for k, v in sorted(by_label.items())},
            'frame_sha256': sorted({r['frame_sha256'] for r in rows}),
            'runs': [{'repeat': r['repeat'], 'wall_seconds': r['wall_seconds'],
                      'gpu_ms': sum(d['ms'] for d in r['dispatches'])} for r in rows],
        }
        report['runs'][mode] = entry

    if {'fp16', 'int8'} <= set(modes):
        fp16 = report['runs']['fp16']
        int8 = report['runs']['int8']
        speedup = fp16['total_gpu_ms']['median'] / int8['total_gpu_ms']['median']
        report['comparison'] = {
            'fp16_gpu_ms_median': fp16['total_gpu_ms']['median'],
            'int8_gpu_ms_median': int8['total_gpu_ms']['median'],
            'speedup_fp16_over_int8': speedup,
            'classification': ('faster' if speedup > 1.05 else
                               'slower' if speedup < 0.95 else 'indistinguishable'),
            'per_process_speedup': {
                name: fp16['process_gpu_ms'][name]['median'] /
                      int8['process_gpu_ms'][name]['median']
                for name in fp16['process_gpu_ms'] if name in int8['process_gpu_ms']},
        }
        report['status'] = 'PASS'
    else:
        report['status'] = 'PASS'
    (a.out / 'm12-profile.json').write_text(json.dumps(report, indent=2))
    for mode in modes:
        entry = report['runs'][mode]
        print(f'{mode}: {entry["dispatch_count"]} dispatches, '
              f'GPU median {entry["total_gpu_ms"]["median"]:.3f} ms '
              f'(p95 {entry["total_gpu_ms"]["p95"]:.3f}, '
              f'min {entry["total_gpu_ms"]["min"]:.3f}, '
              f'max {entry["total_gpu_ms"]["max"]:.3f})')
    if 'comparison' in report:
        print(json.dumps(report['comparison'], indent=2))
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
