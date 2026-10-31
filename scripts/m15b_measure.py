#!/usr/bin/env python3
"""M15-B: interleaved chain comparison of the accepted and fused MLP graphs.

Both graphs are the same ten-layer INT8 chain with the same seed and input.
The only difference is that the fused graph replaces every
(GELU, down-quantizer) pair with one dispatch.  Runs are rotated so clock drift
is shared; every run must reproduce the accepted chain hash.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import m15_measure as measure

FAMILIES = {
    'mlp-up': ('mlp-up',),
    'mlp-gelu': ('mlp-gelu',),
    'mlp-down-quant': ('mlp-down-quant',),
    'mlp-fused': ('mlp-fused',),
    'mlp-down': ('mlp-down',),
}


def family_ms(dispatches):
    out = {name: 0.0 for name in FAMILIES}
    for dispatch in dispatches:
        tail = dispatch['label'].rsplit('.', 1)[-1]
        if tail in out:
            out[tail] += dispatch['ms']
    return out


def load(path):
    return json.loads(Path(path).read_text())


def row(timing, vram, arena, mode, graph):
    dispatches = timing['dispatches']
    return {
        'graph': str(graph),
        'mode': mode,
        'gpu': {
            'dispatch_count': len(dispatches),
            'dispatch_sum_ms': sum(d['ms'] for d in dispatches),
            'interval_ms': timing['global']['ms'],
            'max_dispatch_ms': max(d['ms'] for d in dispatches),
            'frequency': timing['frequency'],
        },
        'families_ms': family_ms(dispatches),
        'host': vram['host'],
        'memory': {
            'tracked': vram['tracked'],
            'local_peak_bytes': next((s['local']['usage'] for s in vram['dxgi']
                                      if s['label'] == 'after_execute'), None),
            'dxgi': {s['label']: s for s in vram['dxgi']},
        },
        'arena': arena['arena'],
        'barriers': arena['barriers'],
        'dumps': arena['dumps'],
        'dropped_dumps': arena['dropped_dumps'],
    }


def median(values):
    return statistics.median(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--control-graph', type=Path, required=True)
    ap.add_argument('--fused-graph', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--probe', type=Path,
                    default=REPO / 'build/native/Release/m_network_stage_timed.exe')
    ap.add_argument('--shaders', type=Path, default=REPO / 'build/native')
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    graphs = {'control': a.control_graph, 'fused': a.fused_graph}
    report = {'graphs': {k: str(v) for k, v in graphs.items()},
              'repeats': a.repeats, 'accepted_chain': measure.ACCEPTED_CHAIN,
              'runs': {}, 'status': 'running'}
    cfg = {'mode': 'production', 'arena': True}
    for name, graph in graphs.items():
        measure.run(a.probe, graph, a.out / name / 'warm', cfg, a.shaders)
        print(f'{name} warm-up done', flush=True)
    for repeat in range(a.repeats):
        order = list(graphs)
        order = order[repeat % len(order):] + order[:repeat % len(order)]
        for name in order:
            out = a.out / name / f'run-{repeat}'
            result = measure.run(a.probe, graphs[name], out, cfg, a.shaders)
            timing = load(out / 'timing.json')
            vram = load(out / 'vram.json')
            arena = load(out / 'arena.json')
            entry = row(timing, vram, arena, cfg['mode'], graphs[name])
            entry['frame_sha256'] = result['frame_sha256']
            entry['dumps'] = result['dumps_count']
            entry['dropped_dumps'] = result['dropped_dumps']
            report['runs'].setdefault(name, []).append(entry)
            print(f'{name} round {repeat}: interval {entry["gpu"]["interval_ms"]:.3f} ms, '
                  f'sum {entry["gpu"]["dispatch_sum_ms"]:.3f} ms, '
                  f'{entry["gpu"]["dispatch_count"]} dispatches, '
                  f'local {entry["memory"]["local_peak_bytes"] / 1e6:.1f} MB',
                  flush=True)
    summary = {}
    for name, runs in report['runs'].items():
        hashes = {r['frame_sha256'] for r in runs}
        if hashes != {measure.ACCEPTED_CHAIN}:
            raise RuntimeError(f'{name}: final hash {sorted(hashes)} != accepted')
        summary[name] = {
            'runs': len(runs),
            'interval_ms_median': median([r['gpu']['interval_ms'] for r in runs]),
            'interval_ms_min': min(r['gpu']['interval_ms'] for r in runs),
            'interval_ms_max': max(r['gpu']['interval_ms'] for r in runs),
            'dispatch_sum_ms_median': median([r['gpu']['dispatch_sum_ms'] for r in runs]),
            'dispatch_count': runs[0]['gpu']['dispatch_count'],
            'families_ms_median': {k: median([r['families_ms'][k] for r in runs])
                                   for k in FAMILIES},
            'local_peak_bytes': max(r['memory']['local_peak_bytes'] for r in runs),
            'tracked_total_bytes': runs[0]['memory']['tracked']['total_bytes'],
            'tracked_readback_bytes': runs[0]['memory']['tracked']['readback_bytes'],
            'tracked_transient_peak_bytes': runs[0]['memory']['tracked'][
                'transient_peak_bytes'],
            'arena': runs[0]['arena'],
            'barriers': runs[0]['barriers'],
            'dumps': runs[0]['dumps'],
            'dropped_dumps': runs[0]['dropped_dumps'],
            'final_sha256': sorted(hashes)[0],
        }
    c, f = summary['control'], summary['fused']
    report['summary'] = summary
    report['comparison'] = {
        'interval_saving_ms': c['interval_ms_median'] - f['interval_ms_median'],
        'interval_ratio': c['interval_ms_median'] / f['interval_ms_median'],
        'dispatch_reduction': c['dispatch_count'] - f['dispatch_count'],
        'dispatch_sum_saving_ms': c['dispatch_sum_ms_median']
                                  - f['dispatch_sum_ms_median'],
        'gelu_quant_control_ms': c['families_ms_median']['mlp-gelu']
                                 + c['families_ms_median']['mlp-down-quant'],
        'fused_stage_ms': f['families_ms_median']['mlp-fused'],
        'gelu_quant_saving_ms': (c['families_ms_median']['mlp-gelu']
                                 + c['families_ms_median']['mlp-down-quant']
                                 - f['families_ms_median']['mlp-fused']),
        'local_peak_saving_bytes': c['local_peak_bytes'] - f['local_peak_bytes'],
        'barrier_reduction': {
            key: c['barriers'].get(key, 0) - f['barriers'].get(key, 0)
            for key in set(c['barriers']) | set(f['barriers'])},
    }
    report['status'] = 'PASS'
    (a.out / 'm15b-measure.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'summary': summary, 'comparison': report['comparison']},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
