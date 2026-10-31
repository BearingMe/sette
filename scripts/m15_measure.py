#!/usr/bin/env python3
"""M15-A: interleaved comparison of committed, arena and production graphs.

All three configurations execute the same accepted ten-layer chain graph with the
same seed and the same timed probe. Each run must reproduce the accepted final
tensor before its numbers are used:

  * committed  - M14 behaviour: every resource is a committed buffer;
  * arena      - diagnostic dumps, transient tensors sharing one placed heap;
  * production - only the final tensor is read back, transient tensors aliased.

The configuration order is rotated every round so clock drift is shared. GPU
interval, dispatch sum, host phases, DXGI process usage, tracked allocations and
arena statistics are reported separately; they are never combined.

  m15_measure.py --graph graph.txt --out DIR [--repeats 3]
  m15_measure.py --build-from-layers LAYERS --out DIR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
# The accepted chain tensor (B6 boundary `chain/out-dec1`, M14 run-00). This is
# not the full-frame hash: the image stages run after the neural chain.
ACCEPTED_CHAIN = 'c89208ed744acd71275f349d99d758c16bd55c77a6fb5d3da88109a3573c6b0d'
CONFIGS = {
    'committed': {'mode': 'diagnostic', 'arena': False},
    'arena': {'mode': 'diagnostic', 'arena': True},
    'production': {'mode': 'production', 'arena': True},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def run(probe, graph, out, cfg, shaders, seed=0):
    out.mkdir(parents=True)
    command = [str(probe), '--graph', str(graph), '--out', str(out),
               '--shaders', str(shaders), '--seed', str(seed), '--mode', cfg['mode'],
               '--timing', str(out / 'timing.json'), '--vram', str(out / 'vram.json'),
               '--arena-report', str(out / 'arena.json')]
    if cfg['arena']:
        command.append('--arena')
    env = dict(os.environ, M_STAGE_DEBUG='1')
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    (out / 'execution.log').write_text(done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError(f'{out}: {done.stdout[-800:]}{done.stderr[-800:]}')
    timing = read_json(out / 'timing.json')
    memory = read_json(out / 'vram.json')
    arena = read_json(out / 'arena.json')
    dispatches = timing['dispatches']
    return {
        'frame_sha256': sha(out / 'out-dec1.bin'),
        'dumps': {p.name: sha(p) for p in out.glob('*.bin')},
        'gpu': {
            'dispatch_count': len(dispatches),
            'dispatch_sum_ms': sum(d['ms'] for d in dispatches),
            'interval_ms': timing['global']['ms'],
            'gap_ms': timing['global']['ms'] - sum(d['ms'] for d in dispatches),
            'max_dispatch_ms': max(d['ms'] for d in dispatches),
            'frequency': timing['frequency'],
        },
        'host': memory['host'],
        'memory': {
            'tracked': memory['tracked'],
            'dxgi': {s['label']: s for s in memory['dxgi']},
        },
        'arena': arena['arena'],
        'barriers': arena['barriers'],
        'dumps_count': arena['dumps'],
        'dropped_dumps': arena['dropped_dumps'],
        'mode': cfg['mode'],
        'arena_enabled': cfg['arena'],
    }


def median(values):
    return statistics.median(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--graph', type=Path)
    ap.add_argument('--build-from-layers', type=Path)
    ap.add_argument('--enc1-input', type=Path,
                    help='enc0 token tensor used as the chain input (B6 graph)')
    ap.add_argument('--graph-dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--probe', type=Path,
                    default=REPO / 'build/native/Release/m_network_stage_timed.exe')
    ap.add_argument('--shaders', type=Path, default=REPO / 'build/native')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--fuse-mlp', action='store_true',
                    help='fuse the post-MLP GELU with the down-projection quantizer')
    ap.add_argument('--m14', type=Path,
                    default=TEMP / 'm14-baseline-20261030' / 'm14-baseline.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    graph = a.graph
    if a.build_from_layers:
        import m_b6_frame as b6
        if not a.enc1_input:
            raise SystemExit('--build-from-layers also needs --enc1-input')
        graph_dir = a.graph_dir or (a.out / 'chain')
        b6.build_chain(a.build_from_layers, graph_dir, a.enc1_input,
                       fuse_mlp=a.fuse_mlp)
        graph = graph_dir / 'graph.txt'
    if graph is None:
        raise SystemExit('--graph or --build-from-layers is required')

    report = {'graph': str(graph), 'repeats': a.repeats, 'accepted_chain': ACCEPTED_CHAIN,
              'fuse_mlp': a.fuse_mlp, 'configs': CONFIGS, 'runs': {}, 'status': 'running'}
    # warm-up once per configuration, then interleave the measured rounds
    for name, cfg in CONFIGS.items():
        run(a.probe, graph, a.out / name / 'warm', cfg, a.shaders)
        print(f'{name} warm-up done', flush=True)
    for repeat in range(a.repeats):
        order = list(CONFIGS)
        order = order[repeat % len(order):] + order[:repeat % len(order)]
        for name in order:
            row = run(a.probe, graph, a.out / name / f'run-{repeat}', CONFIGS[name],
                      a.shaders)
            report['runs'].setdefault(name, []).append(row)
            g = row['gpu']
            print(f'{name} round {repeat}: interval {g["interval_ms"]:.3f} ms, '
                  f'sum {g["dispatch_sum_ms"]:.3f} ms, '
                  f'local {row["memory"]["dxgi"]["after_execute"]["local"]["usage"] / 1e6:.1f} MB',
                  flush=True)

    baseline = report['runs']['committed'][0]['dumps']
    summary = {}
    for name, runs in report['runs'].items():
        final = {r['frame_sha256'] for r in runs}
        if final != {ACCEPTED_CHAIN}:
            raise RuntimeError(f'{name}: final hash {sorted(final)} != accepted')
        diagnostic = {}
        if name != 'production':
            for r in runs:
                for filename, digest in baseline.items():
                    if r['dumps'].get(filename) != digest:
                        raise RuntimeError(f'{name}: dump {filename} differs')
            diagnostic = {'dumps': len(baseline), 'byte_identical': True}
        summary[name] = {
            'runs': len(runs),
            'interval_ms_median': median([r['gpu']['interval_ms'] for r in runs]),
            'interval_ms_min': min(r['gpu']['interval_ms'] for r in runs),
            'interval_ms_max': max(r['gpu']['interval_ms'] for r in runs),
            'dispatch_sum_ms_median': median([r['gpu']['dispatch_sum_ms'] for r in runs]),
            'gap_ms_median': median([r['gpu']['gap_ms'] for r in runs]),
            'dispatch_count': runs[0]['gpu']['dispatch_count'],
            'load_alloc_ms_median': median([r['host']['load_alloc_s'] * 1000 for r in runs]),
            'record_ms_median': median([r['host']['record_s'] * 1000 for r in runs]),
            'execute_wait_ms_median': median(
                [r['host']['execute_wait_s'] * 1000 for r in runs]),
            'readback_write_ms_median': median(
                [r['host']['readback_write_s'] * 1000 for r in runs]),
            'host_total_ms_median': median([r['host']['total_s'] * 1000 for r in runs]),
            'dxgi_local_peak_bytes': max(
                r['memory']['dxgi']['after_execute']['local']['usage'] for r in runs),
            'dxgi_local_budget_bytes': min(
                r['memory']['dxgi']['after_execute']['local']['budget'] for r in runs),
            'tracked_total_bytes': runs[0]['memory']['tracked']['total_bytes'],
            'tracked_default_bytes': runs[0]['memory']['tracked']['default_bytes'],
            'tracked_readback_bytes': runs[0]['memory']['tracked']['readback_bytes'],
            'tracked_transient_peak_bytes': runs[0]['memory']['tracked']
                                            ['transient_peak_bytes'],
            'arena': runs[0]['arena'],
            'barriers': runs[0]['barriers'],
            'dumps': runs[0]['dumps_count'],
            'dropped_dumps': runs[0]['dropped_dumps'],
            'final_sha256': sorted(final)[0],
            'diagnostic': diagnostic,
        }
    report['summary'] = summary
    c, ar, pr = summary['committed'], summary['arena'], summary['production']
    report['comparison'] = {
        'arena_interval_saving_ms': c['interval_ms_median'] - ar['interval_ms_median'],
        'arena_interval_ratio': c['interval_ms_median'] / ar['interval_ms_median'],
        'production_interval_saving_ms': c['interval_ms_median']
                                         - pr['interval_ms_median'],
        'arena_local_peak_saving_bytes': c['dxgi_local_peak_bytes']
                                         - ar['dxgi_local_peak_bytes'],
        'arena_local_peak_ratio': c['dxgi_local_peak_bytes'] / ar['dxgi_local_peak_bytes'],
        'arena_load_alloc_saving_ms': c['load_alloc_ms_median']
                                      - ar['load_alloc_ms_median'],
        'production_readback_saving_ms': ar['readback_write_ms_median']
                                         - pr['readback_write_ms_median'],
        'production_readback_bytes_removed': ar['tracked_readback_bytes']
                                             - pr['tracked_readback_bytes'],
    }
    if a.m14.is_file():
        m14 = read_json(a.m14)
        chain = [r['processes']['chain-native']['gpu']['global_ms']
                 for r in m14['runs']['int8']]
        report['m14_reference'] = {
            'chain_interval_ms_median': median(chain),
            'full_frame_interval_ms_median': m14['summary']['int8']['gpu_interval_ms_median'],
            'chain_local_peak_bytes': max(
                r['processes']['chain-native']['memory']['dxgi']['after_execute']
                ['local']['usage'] for r in m14['runs']['int8']),
            'chain_load_alloc_ms_median': median(
                [r['processes']['chain-native']['host']['load_alloc_s'] * 1000
                 for r in m14['runs']['int8']]),
        }
        ref = report['m14_reference']['chain_interval_ms_median']
        report['comparison']['arena_vs_m14_chain_ms'] = ref - ar['interval_ms_median']
        report['comparison']['arena_vs_m14_chain_ratio'] = ref / ar['interval_ms_median']
        report['comparison']['production_vs_m14_chain_ms'] = (
            ref - pr['interval_ms_median'])
        report['comparison']['production_vs_m14_chain_ratio'] = (
            ref / pr['interval_ms_median'])
    report['status'] = 'PASS'
    (a.out / 'm15a-measure.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'summary': {k: {kk: vv for kk, vv in v.items()
                                      if kk in ('interval_ms_median', 'interval_ms_min',
                                                'interval_ms_max', 'dispatch_sum_ms_median',
                                                'load_alloc_ms_median', 'host_total_ms_median',
                                                'dxgi_local_peak_bytes', 'tracked_total_bytes',
                                                'tracked_readback_bytes', 'dumps',
                                                'dropped_dumps')}
                                for k, v in summary.items()},
                      'comparison': report['comparison'],
                      'm14_reference': report.get('m14_reference')}, indent=2))
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
