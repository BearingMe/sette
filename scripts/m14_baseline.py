#!/usr/bin/env python3
"""M14: complete performance baseline with GPU, host and VRAM scopes.

Runs the accepted FP16 and full-INT8 frames through the approved B6 harness with
the `*_timed` probes.  Each probe emits per-dispatch timestamps, a whole-command-
list GPU interval, host phase durations and a memory report.  The two modes are
interleaved so clock/power drift is shared, and every run must reproduce the
accepted frame hash.

Reported separately:
  * dispatch sum       - sum of per-dispatch timestamp deltas;
  * GPU interval       - one timestamp pair around the whole command list;
  * gap               - interval minus dispatch sum;
  * host phases       - startup, load/alloc, record, submit/wait, readback/write;
  * tracked memory    - committed allocation sizes by heap;
  * DXGI memory       - process-local usage/budget samples.

  m14_baseline.py --out DIR [--repeats 3] [--cumulative-layers DIR]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
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


def read_json(path):
    return json.loads(Path(path).read_text())


def run_mode(mode, out, layers, network_mode='diagnostic', network_arena=False,
             fuse_mlp=False):
    timing = out / 'timing'
    vram = out / 'vram'
    command = [sys.executable, str(HERE / 'm_b6_frame.py'), '--out', str(out),
               '--no-repeat',
               '--probe', str(REPO / 'build/native/Release/m_image_stage_timed.exe'),
               '--network-probe', str(REPO / 'build/native/Release/m_network_stage_timed.exe'),
               '--timing-dir', str(timing), '--vram-dir', str(vram),
               '--network-mode', network_mode, *MODES[mode]]
    if fuse_mlp and mode != 'fp16':
        command.append('--fuse-mlp')
    if network_arena:
        # The arena pays off on the INT8 chain's 4.39 GB working set. The FP16
        # chain is ~0.76 GB and the extra aliasing barriers cost ~1%, so the
        # production policy keeps FP16 on committed buffers.
        if mode != 'fp16':
            command.append('--network-arena')
    if mode != 'fp16':
        command += ['--cumulative-layers', str(layers)]
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True)
    wall = time.perf_counter() - started
    frame = out / 'downsample-native' / 'results' / 'out-out.bin'
    if not frame.is_file():
        raise RuntimeError(f'{mode}: no frame\n{done.stderr[-1200:]}')
    digest = sha(frame)
    if digest != ACCEPTED[mode]:
        raise RuntimeError(f'{mode}: frame {digest} != accepted')
    processes = {}
    for path in sorted(timing.glob('*.json')):
        gpu = read_json(path)
        memory = read_json(vram / path.name)
        dispatches = gpu['dispatches']
        processes[path.stem] = {
            'gpu': {
                'dispatch_count': len(dispatches),
                'dispatch_sum_ms': sum(d['ms'] for d in dispatches),
                'global_ms': gpu['global']['ms'],
                'gap_ms': gpu['global']['ms'] - sum(d['ms'] for d in dispatches),
                'max_dispatch_ms': max(d['ms'] for d in dispatches),
            },
            'host': memory['host'],
            'memory': {
                'tracked': memory['tracked'],
                'dxgi': {s['label']: s for s in memory['dxgi']},
                'adapter': memory['adapter'],
            },
        }
    shutil.move(str(frame), str(out / 'frame.bin'))
    for child in list(out.iterdir()):
        if child.name in ('timing', 'vram', 'frame.bin', 'b6.json'):
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    total = {
        'dispatch_count': sum(p['gpu']['dispatch_count'] for p in processes.values()),
        'dispatch_sum_ms': sum(p['gpu']['dispatch_sum_ms'] for p in processes.values()),
        'global_ms': sum(p['gpu']['global_ms'] for p in processes.values()),
        'gap_ms': sum(p['gpu']['gap_ms'] for p in processes.values()),
    }
    total['host'] = {k: sum(p['host'][k] for p in processes.values())
                     for k in ('startup_s', 'load_alloc_s', 'record_s',
                               'execute_wait_s', 'readback_write_s', 'total_s')}
    total['host']['outside_gpu_ms'] = (total['host']['execute_wait_s'] * 1000.0 -
                                       total['global_ms'])
    tracked = {
        'allocations': sum(p['memory']['tracked']['allocations'] for p in processes.values()),
        'total_bytes': sum(p['memory']['tracked']['total_bytes'] for p in processes.values()),
        'default_bytes': sum(p['memory']['tracked']['default_bytes'] for p in processes.values()),
        'upload_bytes': sum(p['memory']['tracked']['upload_bytes'] for p in processes.values()),
        'readback_bytes': sum(p['memory']['tracked']['readback_bytes'] for p in processes.values()),
        'transient_peak_bytes': max(p['memory']['tracked']['transient_peak_bytes']
                                    for p in processes.values()),
        'peak_bound_bytes': sum(p['memory']['tracked']['peak_bound_bytes']
                                for p in processes.values()),
        'largest_process': max(processes,
                               key=lambda p: processes[p]['memory']['tracked']['total_bytes']),
        'largest': max((entry for p in processes.values()
                        for entry in p['memory']['tracked']['largest']),
                       key=lambda e: e['bytes']),
    }
    dxgi_peak = max((p['memory']['dxgi']['after_execute']['local']['usage']
                     for p in processes.values()), default=None)
    dxgi_budget = min((p['memory']['dxgi']['after_execute']['local']['budget']
                       for p in processes.values()), default=None)
    total['memory'] = {'tracked': tracked, 'dxgi_peak_local_bytes': dxgi_peak,
                       'dxgi_min_budget_bytes': dxgi_budget,
                       'per_process': {name: p['memory'] for name, p in processes.items()}}
    return {'frame_sha256': digest, 'wall_seconds': wall, 'processes': processes,
            'total': total}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cumulative-layers', type=Path,
                    default=TEMP / 'm09h-complete-chain-20261030' / 'layers')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--network-mode', choices=['diagnostic', 'production'],
                    default='diagnostic')
    ap.add_argument('--network-arena', action='store_true')
    ap.add_argument('--fuse-mlp', action='store_true')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    report = {'repeats': a.repeats, 'warmup': a.warmup,
              'layers': str(a.cumulative_layers), 'accepted': ACCEPTED,
              'network_mode': a.network_mode, 'network_arena': a.network_arena,
              'fuse_mlp': a.fuse_mlp,
              'runs': {}, 'comparison': {}, 'status': 'running'}
    for mode in ('fp16', 'int8'):
        for warm in range(a.warmup):
            run_mode(mode, a.out / mode / f'warm-{warm:02d}', a.cumulative_layers,
                     a.network_mode, a.network_arena, a.fuse_mlp)
            print(f'{mode} warmup {warm} done', flush=True)
    for repeat in range(a.repeats):
        for mode in ('fp16', 'int8'):
            row = run_mode(mode, a.out / mode / f'run-{repeat:02d}', a.cumulative_layers,
                           a.network_mode, a.network_arena, a.fuse_mlp)
            report['runs'].setdefault(mode, []).append({
                'repeat': repeat, 'wall_seconds': row['wall_seconds'],
                'frame_sha256': row['frame_sha256'],
                'total': row['total'], 'processes': row['processes']})
            t = row['total']
            print(f'{mode} round {repeat}: GPU interval {t["global_ms"]:.3f} ms, '
                  f'dispatch sum {t["dispatch_sum_ms"]:.3f} ms, '
                  f'{t["dispatch_count"]} dispatches, host {t["host"]["total_s"]:.2f} s',
                  flush=True)
    summary = {}
    for mode, runs in report['runs'].items():
        globals_ = [r['total']['global_ms'] for r in runs]
        sums = [r['total']['dispatch_sum_ms'] for r in runs]
        summary[mode] = {
            'runs': len(runs),
            'dispatch_count': runs[0]['total']['dispatch_count'],
            'gpu_interval_ms_median': statistics.median(globals_),
            'gpu_interval_ms_min': min(globals_), 'gpu_interval_ms_max': max(globals_),
            'dispatch_sum_ms_median': statistics.median(sums),
            'gap_ms_median': statistics.median([r['total']['gap_ms'] for r in runs]),
            'window_ms_median': statistics.median(
                [(r['total']['host']['execute_wait_s'] * 1000.0) for r in runs]),
            'host_ms_median': statistics.median(
                [r['total']['host']['total_s'] * 1000.0 for r in runs]),
            'frame_hashes': sorted({r['frame_sha256'] for r in runs}),
            'deterministic': len({r['frame_sha256'] for r in runs}) == 1,
            'memory': runs[0]['total']['memory'],
        }
    report['summary'] = summary
    if {'fp16', 'int8'} <= set(summary):
        f, i = summary['fp16'], summary['int8']
        report['comparison'] = {
            'fp16_gpu_interval_ms': f['gpu_interval_ms_median'],
            'int8_gpu_interval_ms': i['gpu_interval_ms_median'],
            'speedup_interval': f['gpu_interval_ms_median'] / i['gpu_interval_ms_median'],
            'fp16_dispatch_sum_ms': f['dispatch_sum_ms_median'],
            'int8_dispatch_sum_ms': i['dispatch_sum_ms_median'],
            'speedup_dispatch_sum': f['dispatch_sum_ms_median'] / i['dispatch_sum_ms_median'],
            'fp16_host_total_ms': f['host_ms_median'],
            'int8_host_total_ms': i['host_ms_median'],
        }
    report['status'] = 'PASS' if all(v['deterministic'] for v in summary.values()) else 'FAIL'
    (a.out / 'm14-baseline.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in
                          ('dispatch_count', 'gpu_interval_ms_median',
                           'dispatch_sum_ms_median', 'gap_ms_median', 'window_ms_median',
                           'host_ms_median', 'deterministic')}
                      for k, v in summary.items()}, indent=2))
    print(json.dumps(report.get('comparison'), indent=2))
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
