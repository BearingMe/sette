#!/usr/bin/env python3
"""M12-C: join per-dispatch timing with declared buffer traffic.

The graph declares every resource size and every stage binding.  Summing the
bound resources gives an upper bound on bytes touched per dispatch; dividing by
the measured GPU time gives an *effective* bandwidth.  It is not a hardware
counter: a low number means the dispatch did not saturate memory, not which
limit it hit.

  m12_analyze.py --profile DIR [--graph GRAPH] [--out JSON]
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_graph(path):
    resources, stages, stage = {}, [], None
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == 'resource':
            resources[words[1]] = int(words[2])
        elif words[0] == 'stage':
            stage = {'name': words[1], 'srv': [], 'uav': [], 'groups': None}
            stages.append(stage)
        elif stage is not None and words[0] == 'srv':
            stage['srv'].append(words[2])
        elif stage is not None and words[0] == 'uav':
            stage['uav'].append(words[2])
        elif stage is not None and words[0] == 'groups':
            stage['groups'] = [int(x) for x in words[1:4]]
    return resources, stages


def family(label):
    return label.split('.', 1)[1] if '.' in label else label


def stats(values):
    values = sorted(values)
    return {'median': statistics.median(values), 'min': values[0], 'max': values[-1],
            'samples': len(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--graph', type=Path, default=None,
                    help='chain graph.txt (default: rebuild from the cumulative layers)')
    ap.add_argument('--cumulative-layers', type=Path,
                    default=Path.home() / 'AppData/Local/Temp/opencode/m09h-complete-chain-20261030/layers')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    report = json.loads((a.profile / 'm12-profile.json').read_text())
    graph = a.graph
    if graph is None:
        import tempfile
        import m_b6_frame as b6
        temp = Path(tempfile.mkdtemp(prefix='m12-graph-'))
        b6.build_chain(a.cumulative_layers, temp, temp / 'dummy-tokens.bin')
        graph = temp / 'graph.txt'
    resources, stages = parse_graph(graph)
    traffic = {}
    for s in stages:
        read = sum(resources.get(name, 0) for name in s['srv'])
        write = sum(resources.get(name, 0) for name in s['uav'])
        traffic[s['name']] = {'read_bytes': read, 'write_bytes': write,
                              'groups': s['groups'], 'srv': len(s['srv']),
                              'uav': len(s['uav'])}
    result = {'profile': str(a.profile), 'graph': str(graph),
              'resources': len(resources), 'stages': len(stages),
              'peak_effective_gbps': None, 'by_stage': {}, 'by_family': {},
              'unmatched_stages': {}}
    rows = []
    unmatched = {}
    for mode in ('fp16', 'int8'):
        if mode not in report['runs']:
            continue
        unmatched[mode] = []
        for label, ms in report['runs'][mode]['dispatch_gpu_ms'].items():
            if not label.startswith('chain-native:'):
                continue
            name = label.split(':', 1)[1]
            t = traffic.get(name)
            if not t:
                unmatched[mode].append(name)
            moved = (t['read_bytes'] + t['write_bytes']) if t else None
            seconds = ms['median'] / 1000.0
            gbps = (moved / 1e9) / seconds if moved and seconds else None
            rows.append({'mode': mode, 'label': name, 'ms': ms['median'],
                         'moved_bytes': moved, 'effective_gbps': gbps,
                         'groups': t['groups'] if t else None, 'family': family(name),
                         'traffic_known': bool(t)})
    if not rows:
        raise SystemExit('no chain dispatches matched the graph')
    peak = max((r['effective_gbps'] for r in rows if r['effective_gbps']), default=None)
    result['peak_effective_gbps'] = peak
    for row in rows:
        row['fraction_of_peak'] = (row['effective_gbps'] / peak
                                   if row['effective_gbps'] and peak else None)
        key = f'{row["mode"]}:{row["label"]}'
        result['by_stage'][key] = row
    for mode in ('fp16', 'int8'):
        fam = {}
        for row in rows:
            if row['mode'] != mode:
                continue
            entry = fam.setdefault(row['family'], {'ms': 0.0, 'moved_bytes': 0,
                                                   'dispatches': 0, 'traffic_known': 0})
            entry['ms'] += row['ms']
            entry['dispatches'] += 1
            if row['moved_bytes'] is not None:
                entry['moved_bytes'] += row['moved_bytes']
                entry['traffic_known'] += 1
        total = sum(v['ms'] for v in fam.values())
        for entry in fam.values():
            entry['share'] = entry['ms'] / total if total else None
            entry['effective_gbps'] = (((entry['moved_bytes'] / 1e9) /
                                        (entry['ms'] / 1000.0))
                                       if entry['ms'] and entry['traffic_known']
                                       else None)
        result['by_family'][mode] = dict(sorted(fam.items(),
                                                key=lambda kv: -kv[1]['ms']))
        result[f'total_chain_ms_{mode}'] = total
    result['unmatched_stages'] = unmatched
    out = a.out or (a.profile / 'm12-analysis.json')
    out.write_text(json.dumps(result, indent=2))
    for mode in ('fp16', 'int8'):
        if f'total_chain_ms_{mode}' not in result:
            continue
        print(f'{mode}: chain {result[f"total_chain_ms_{mode}"]:.3f} ms, '
              f'peak effective {peak:.1f} GB/s')
        for name, entry in list(result['by_family'][mode].items())[:8]:
            gbps = entry['effective_gbps']
            print(f'  {name:26s} {entry["ms"]:8.3f} ms {100*entry["share"]:5.1f}% '
                  f'{(f"{gbps:7.1f} GB/s" if gbps else "      n/a")}')
    print('out:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
