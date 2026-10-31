#!/usr/bin/env python3
"""M15-B: interleaved full-frame control/fused measurement.

Uses the established `m14_baseline.run_mode` path (same B6 harness, production
network mode, INT8 arena) and alternates the control and fused chain on every
round.  The accepted frame hash is checked by the harness on every run.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import m14_baseline as baseline

FAMILIES = ('mlp-up', 'mlp-gelu', 'mlp-down-quant', 'mlp-fused', 'mlp-down')


def family_ms(path):
    timing = json.loads(Path(path).read_text())
    out = {name: 0.0 for name in FAMILIES}
    for dispatch in timing['dispatches']:
        tail = dispatch['label'].rsplit('.', 1)[-1]
        if tail in out:
            out[tail] += dispatch['ms']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layers', type=Path,
                    default=Path.home() / 'AppData/Local/Temp/opencode/'
                    'm09h-complete-chain-20261030/layers')
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    configs = {'control': False, 'fused': True}
    runs = {name: [] for name in configs}
    for name, fuse in configs.items():
        baseline.run_mode('int8', a.out / name / 'warm', a.layers,
                          'production', True, fuse)
        print(f'{name} warm-up done', flush=True)
    for repeat in range(a.repeats):
        order = list(configs)
        order = order[repeat % len(order):] + order[:repeat % len(order)]
        for name in order:
            out = a.out / name / f'run-{repeat}'
            result = baseline.run_mode('int8', out, a.layers, 'production', True,
                                       configs[name])
            total = result['total']
            chain = result['processes']['chain-native']
            runs[name].append({
                'frame_sha256': result['frame_sha256'],
                'gpu_interval_ms': total['global_ms'],
                'dispatch_sum_ms': total['dispatch_sum_ms'],
                'dispatch_count': total['dispatch_count'],
                'gap_ms': total['gap_ms'],
                'host_ms': total['host']['total_s'],
                'load_alloc_ms': total['host']['load_alloc_s'] * 1000,
                'dxgi_peak_local_bytes': total['memory']['dxgi_peak_local_bytes'],
                'tracked_total_bytes': total['memory']['tracked']['total_bytes'],
                'chain_interval_ms': chain['gpu']['global_ms'],
                'chain_local_peak_bytes': chain['memory']['dxgi']['after_execute'][
                    'local']['usage'],
                'families_ms': family_ms(out / 'timing/chain-native.json'),
            })
            print(f'{name} round {repeat}: frame {total["global_ms"]:.3f} ms, '
                  f'chain {chain["gpu"]["global_ms"]:.3f} ms, '
                  f'{total["dispatch_count"]} dispatches', flush=True)
    summary = {}
    for name, rows in runs.items():
        hashes = {r['frame_sha256'] for r in rows}
        if hashes != {baseline.ACCEPTED['int8']}:
            raise RuntimeError(f'{name}: frame hash {sorted(hashes)} != accepted')
        summary[name] = {
            'runs': len(rows),
            'frame_interval_ms_median': statistics.median(
                [r['gpu_interval_ms'] for r in rows]),
            'frame_interval_ms_min': min(r['gpu_interval_ms'] for r in rows),
            'frame_interval_ms_max': max(r['gpu_interval_ms'] for r in rows),
            'dispatch_sum_ms_median': statistics.median(
                [r['dispatch_sum_ms'] for r in rows]),
            'dispatch_count': rows[0]['dispatch_count'],
            'chain_interval_ms_median': statistics.median(
                [r['chain_interval_ms'] for r in rows]),
            'chain_local_peak_bytes': max(r['chain_local_peak_bytes'] for r in rows),
            'dxgi_peak_local_bytes': max(r['dxgi_peak_local_bytes'] for r in rows),
            'tracked_total_bytes': rows[0]['tracked_total_bytes'],
            'host_ms_median': statistics.median([r['host_ms'] for r in rows]),
            'load_alloc_ms_median': statistics.median(
                [r['load_alloc_ms'] for r in rows]),
            'families_ms_median': {
                name: statistics.median([r['families_ms'][name] for r in rows])
                for name in FAMILIES},
            'frame_sha256': sorted(hashes)[0],
        }
    c, f = summary['control'], summary['fused']
    report = {'repeats': a.repeats, 'layers': str(a.layers),
              'summary': summary,
              'comparison': {
                  'frame_saving_ms': c['frame_interval_ms_median']
                                     - f['frame_interval_ms_median'],
                  'frame_ratio': c['frame_interval_ms_median']
                                 / f['frame_interval_ms_median'],
                  'chain_saving_ms': c['chain_interval_ms_median']
                                     - f['chain_interval_ms_median'],
                  'dispatch_reduction': c['dispatch_count'] - f['dispatch_count'],
                  'gelu_quant_control_ms': c['families_ms_median']['mlp-gelu']
                                           + c['families_ms_median']['mlp-down-quant'],
                  'fused_stage_ms': f['families_ms_median']['mlp-fused'],
                  'gelu_quant_saving_ms': (c['families_ms_median']['mlp-gelu']
                                           + c['families_ms_median']['mlp-down-quant']
                                           - f['families_ms_median']['mlp-fused']),
              },
              'status': 'PASS'}
    (a.out / 'm15b-frame.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'summary': summary, 'comparison': report['comparison']},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
