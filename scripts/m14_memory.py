#!/usr/bin/env python3
"""M14: declared-resource lifetime and aliasing bound for the ten-layer chain.

The host creates every graph resource up front and keeps it until process exit,
so its footprint is the sum of all declarations.  This script computes what an
arena allocator could achieve from the same graph: each resource is live from
its first binding to its last binding (or to the end if it is dumped), and the
peak of the overlapping live sets is a lower bound on the arena size.

Two peaks are reported:
  * as executed - every diagnostic dump is live to the end;
  * production  - write-only resources that exist only to be dumped are removed.

  m14_memory.py --graph GRAPH [--out JSON]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_graph(path):
    resources, stages, dumps = {}, [], []
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == 'resource':
            resources[words[1]] = {'name': words[1], 'bytes': int(words[2]),
                                   'init': bool(words[3:])}
        elif words[0] == 'stage':
            stages.append({'name': words[1], 'srv': [], 'uav': []})
        elif words[0] == 'srv' and stages:
            stages[-1]['srv'].append(words[2])
        elif words[0] == 'uav' and stages:
            stages[-1]['uav'].append(words[2])
        elif words[0] == 'dump':
            dumps.append({'resource': words[1], 'bytes': int(words[3]),
                          'file': words[4]})
    return resources, stages, dumps


def analyse(graph):
    resources, stages, dumps = parse_graph(graph)
    last = len(stages) - 1
    info = {name: {'first': None, 'last': None, 'srv': 0, 'uav': 0, **r}
            for name, r in resources.items()}
    for index, stage in enumerate(stages):
        for role, names in (('srv', stage['srv']), ('uav', stage['uav'])):
            for name in names:
                if name not in info:
                    raise SystemExit(f'{stage["name"]}: unknown resource {name}')
                entry = info[name]
                entry[role] += 1
                entry['first'] = index if entry['first'] is None else min(entry['first'], index)
                entry['last'] = index if entry['last'] is None else max(entry['last'], index)
    dumped = {d['resource'] for d in dumps}
    for name, entry in info.items():
        if entry['first'] is None:
            # declared but never bound: allocated by the host, not live in the graph
            entry['bound'] = False
            continue
        entry['bound'] = True
        if entry['init']:
            entry['first'] = 0
        if name in dumped:
            entry['last'] = last
    total = sum(r['bytes'] for r in resources.values())
    peak_all, peak_all_stage = 0, -1
    peak_prod, peak_prod_stage = 0, -1
    live_at = []
    for index in range(len(stages)):
        live = [e for e in info.values()
                if e.get('bound') and e['first'] <= index <= e['last']]
        live_bytes = sum(e['bytes'] for e in live)
        live_at.append(live_bytes)
        if live_bytes > peak_all:
            peak_all, peak_all_stage = live_bytes, index
        production = [e for e in live
                      if not (e['srv'] == 0 and e['name'] in dumped)]
        prod_bytes = sum(e['bytes'] for e in production)
        if prod_bytes > peak_prod:
            peak_prod, peak_prod_stage = prod_bytes, index
    diagnostics = [e for e in info.values()
                   if e.get('bound') and e['srv'] == 0 and e['name'] in dumped]
    longest = sorted(info.values(), key=lambda e: (-e['bytes'], e['name']))[:15]
    spans = sorted((e for e in info.values()
                    if e.get('bound') and e['last'] - e['first'] >= 10),
                   key=lambda e: -(e['bytes']))
    return {
        'graph': str(graph),
        'stages': len(stages), 'resources': len(resources), 'dumps': len(dumps),
        'declared_bytes': total,
        'peak_live_bytes': peak_all,
        'peak_live_stage': peak_all_stage,
        'peak_live_stage_name': stages[peak_all_stage]['name'] if stages else None,
        'peak_production_bytes': peak_prod,
        'peak_production_stage': peak_prod_stage,
        'peak_production_stage_name': stages[peak_prod_stage]['name'] if stages else None,
        'alias_saving_bytes': total - peak_all,
        'production_saving_bytes': total - peak_prod,
        'diagnostic_bytes': sum(e['bytes'] for e in diagnostics),
        'diagnostic_resources': len(diagnostics),
        'largest': [{k: e[k] for k in ('name', 'bytes', 'first', 'last', 'srv', 'uav')}
                    for e in longest],
        'long_lived': [{k: e[k] for k in ('name', 'bytes', 'first', 'last', 'srv', 'uav')}
                       for e in spans[:15]],
        'live_curve': live_at,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--graph', type=Path, default=None)
    ap.add_argument('--build-from-layers', type=Path, default=None,
                    help='build the production chain graph from the accepted layer fixtures')
    ap.add_argument('--graph-dir', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    graph = a.graph
    if a.build_from_layers:
        import m_layer_suite as suite
        graph_dir = a.graph_dir or (a.out.parent / 'm14-production-chain'
                                    if a.out else Path('m14-production-chain'))
        if graph_dir.exists():
            raise SystemExit(f'{graph_dir} exists; use a fresh directory')
        suite.build_chain(str(a.build_from_layers), str(graph_dir))
        graph = graph_dir / 'graph.txt'
    if graph is None:
        raise SystemExit('--graph or --build-from-layers is required')
    report = analyse(graph)
    out = a.out or (graph.parent / 'm14-memory.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('largest', 'long_lived', 'live_curve')}, indent=2))
    print('largest allocations:')
    for e in report['largest'][:8]:
        print(f'  {e["bytes"] / 1e6:8.1f} MB  stage {e["first"]:3d}-{e["last"]:3d} '
              f'srv {e["srv"]} uav {e["uav"]}  {e["name"]}')
    print('longest-lived large resources:')
    for e in report['long_lived'][:8]:
        print(f'  {e["bytes"] / 1e6:8.1f} MB  stage {e["first"]:3d}-{e["last"]:3d} '
              f'{e["name"]}')
    print('out:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
