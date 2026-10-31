#!/usr/bin/env python3
"""M15-A: graph lifetime inventory and static arena plan.

The native host (``m_network_stage``) is authoritative for what actually runs;
this script reproduces its classification and best-fit interval allocation from
the graph text so the plan can be inspected, tested and compared with the
runtime arena report.

For every resource it records the first writer, last reader, first/last use,
initialization requirement, access flags, size, placement alignment, lifetime
kind and whether it is diagnostic. Allocations are 64 KiB aligned (a D3D12
placed-resource requirement) and use best-fit over disjoint live intervals.

  m15_arena.py --graph graph.txt --out inventory.json
  m15_arena.py --build-from-layers LAYERS --out inventory.json
  m15_arena.py --graph graph.txt --out inventory.json --check-report arena.json
"""
import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m14_memory as memory

PLACEMENT = 65536


def align(value):
    return (value + PLACEMENT - 1) // PLACEMENT * PLACEMENT


def select_production_dumps(stages, dumps):
    """Keep only the final stage's UAV dump, one per resource, last file wins."""
    produced = set(stages[-1]['uav'])
    kept, seen = [], set()
    for dump in reversed(dumps):
        if dump['resource'] in produced and dump['resource'] not in seen:
            seen.add(dump['resource'])
            kept.append(dump)
    if not kept and dumps:
        kept = [dumps[-1]]
    return list(reversed(kept))


def analyse(graph, mode='diagnostic'):
    resources, stages, dumps = memory.parse_graph(graph)
    dropped = 0
    if mode == 'production':
        selected = select_production_dumps(stages, dumps)
        dropped = len(dumps) - len(selected)
        dumps = selected
    n = len(stages)
    names = list(resources)
    info = {name: {
        'name': name, 'bytes': resources[name]['bytes'],
        'format': 'R32_UINT buffer view', 'alignment': PLACEMENT,
        'init_required': bool(resources[name]['init']),
        'first_write': None, 'last_read': None,
        'first': None, 'last': None,
        'srv_bindings': 0, 'uav_bindings': 0,
        'srv': False, 'uav': False,
        'both_states_one_stage': False, 'reads_before_write': False,
        'dumped': False, 'offset': 0, 'aligned_bytes': 0,
    } for name in names}

    # first write / last read / first-last use
    for index, stage in enumerate(stages):
        both = {}
        for role, bound in (('srv', stage['srv']), ('uav', stage['uav'])):
            for name in bound:
                if name not in info:
                    raise SystemExit(f'{stage["name"]}: unknown resource {name}')
                entry = info[name]
                entry['srv'] |= role == 'srv'
                entry['uav'] |= role == 'uav'
                entry['srv_bindings'] += role == 'srv'
                entry['uav_bindings'] += role == 'uav'
                entry['first'] = index if entry['first'] is None else min(entry['first'], index)
                entry['last'] = index if entry['last'] is None else max(entry['last'], index)
                if role == 'srv':
                    entry['last_read'] = index if entry['last_read'] is None \
                        else max(entry['last_read'], index)
                else:
                    entry['first_write'] = index if entry['first_write'] is None \
                        else min(entry['first_write'], index)
                both[name] = both.get(name, set()) | {role}
        for name, roles in both.items():
            if roles == {'srv', 'uav'}:
                info[name]['both_states_one_stage'] = True

    for dump in dumps:
        entry = info[dump['resource']]
        entry['dumped'] = True
        entry['last'] = n

    for entry in info.values():
        if entry['first'] is None and not entry['dumped']:
            # Unbound per-layer inputs are placeholders; the chain rewires them.
            entry['lifetime'] = 'dead'
            entry['required_init'] = False
        elif entry['init_required']:
            entry['lifetime'] = 'persistent'
            entry['first'] = 0
            if entry['last'] is None:
                entry['last'] = 0
            entry['required_init'] = True
        elif entry['first'] is None:
            raise SystemExit(f'dumped resource is never bound: {entry["name"]}')
        else:
            entry['lifetime'] = 'transient'
            entry['required_init'] = False
            entry['reads_before_write'] = entry['name'] not in stages[entry['first']]['uav']
        entry['diagnostic_only'] = bool(entry.get('uav') and not entry.get('srv')
                                        and not entry['dumped'])
        entry['persistent'] = entry['lifetime'] == 'persistent'

    # peak live sums
    peak = peak_transient = 0
    for index in range(n + 1):
        live = [e for e in info.values()
                if e['lifetime'] != 'dead' and e['first'] is not None
                and e['first'] <= index <= e['last']]
        peak = max(peak, sum(e['bytes'] for e in live))
        peak_transient = max(peak_transient, sum(e['bytes'] for e in live
                                                 if e['lifetime'] != 'persistent'))

    # static best-fit placement, exactly the native schedule
    persistent = [e for e in info.values() if e['lifetime'] == 'persistent']
    dead = [e for e in info.values() if e['lifetime'] == 'dead']
    transient = []
    for entry in info.values():
        if entry['lifetime'] in ('persistent', 'dead'):
            continue
        if not (entry['uav'] and not entry['reads_before_write']
                and not entry['both_states_one_stage']):
            entry['lifetime'] = 'dedicated'
        else:
            transient.append(entry)

    for entry in persistent + [e for e in info.values() if e['lifetime'] == 'dedicated']:
        entry['aligned_bytes'] = align(entry['bytes'])
        entry['offset'] = 0
    for entry in transient:
        entry['aligned_bytes'] = align(entry['bytes'])
        entry['offset'] = 0

    starts = {}
    ends = {}
    for entry in transient:
        starts.setdefault(entry['first'], []).append(entry)
        ends.setdefault(entry['last'], []).append(entry)
    free = []  # [offset, size]
    high = 0
    bump = 0

    def add_free(offset, size):
        free.append([offset, size])
        free.sort()
        merged = []
        for start, length in free:
            if merged and start <= merged[-1][0] + merged[-1][1]:
                merged[-1][1] = max(merged[-1][0] + merged[-1][1], start + length) \
                    - merged[-1][0]
            else:
                merged.append([start, length])
        return merged

    for index in range(n + 1):
        if index:
            for entry in ends.get(index - 1, []):
                free = add_free(entry['offset'], entry['aligned_bytes'])
        batch = sorted(starts.get(index, []),
                       key=lambda e: (-e['aligned_bytes'], e['name']))
        for entry in batch:
            need = entry['aligned_bytes']
            candidates = [k for k, (_, size) in enumerate(free) if size >= need]
            if candidates:
                best = min(candidates, key=lambda k: (free[k][1], free[k][0]))
                entry['offset'] = free[best][0]
                free[best][0] += need
                free[best][1] -= need
                if not free[best][1]:
                    free.pop(best)
            else:
                entry['offset'] = high
                high += need
                bump += 1

    heap = align(high)
    largest = max((e for e in transient), key=lambda e: e['bytes'], default=None)
    return {
        'graph': str(graph),
        'mode': mode, 'stage_count': n, 'resource_count': len(info), 'dumps': len(dumps),
        'dropped_dumps': dropped,
        'declared_bytes': sum(e['bytes'] for e in info.values()),
        'arena': {
            'heap_bytes': heap,
            'persistent_bytes': sum(e['aligned_bytes'] for e in persistent),
            'dedicated_bytes': sum(e['aligned_bytes'] for e in info.values()
                                   if e['lifetime'] == 'dedicated'),
            'placed_resources': len(transient),
            'persistent_resources': len(persistent),
            'dedicated_resources': sum(1 for e in info.values()
                                       if e['lifetime'] == 'dedicated'),
            'dead_resources': len(dead),
            'bump_regions': bump,
            'peak_live_bytes': peak,
            'peak_transient_bytes': peak_transient,
            'largest_transient_bytes': largest['bytes'] if largest else 0,
            'largest_transient_name': largest['name'] if largest else '',
            'utilization': peak_transient / heap if heap else 0.0,
        },
        'resources': sorted(info.values(), key=lambda e: e['name']),
    }


def resolve(args):
    if args.build_from_layers:
        import m_layer_suite as suite
        graph_dir = args.graph_dir or (args.out.parent / 'm15-production-chain')
        if graph_dir.exists():
            raise SystemExit(f'{graph_dir} exists; use a fresh directory')
        info = suite.build_chain(str(args.build_from_layers), str(graph_dir))
        return graph_dir / 'graph.txt', info
    if not args.graph:
        raise SystemExit('--graph or --build-from-layers is required')
    return args.graph, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--graph', type=Path)
    ap.add_argument('--build-from-layers', type=Path)
    ap.add_argument('--graph-dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['diagnostic', 'production'], default='diagnostic')
    ap.add_argument('--check-report', type=Path,
                    help='compare with a native --arena-report JSON')
    a = ap.parse_args()
    graph, _ = resolve(a)
    report = analyse(graph, a.mode)
    a.out.write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k != 'resources'}
    print(json.dumps(summary, indent=2))
    if a.check_report:
        native = json.loads(a.check_report.read_text())
        checks = {
            'heap_bytes': native['arena']['heap_bytes'] == report['arena']['heap_bytes'],
            'persistent_bytes': native['arena']['persistent_bytes']
                                == report['arena']['persistent_bytes'],
            'placed_resources': native['arena']['placed_resources']
                                == report['arena']['placed_resources'],
            'peak_live_bytes': native['arena']['peak_live_bytes']
                               == report['arena']['peak_live_bytes'],
            'peak_transient_bytes': native['arena']['peak_transient_bytes']
                                    == report['arena']['peak_transient_bytes'],
            'largest_transient_bytes': native['arena']['largest_transient_bytes']
                                       == report['arena']['largest_transient_bytes'],
        }
        print(json.dumps({'native_agreement': checks,
                          'status': 'PASS' if all(checks.values()) else 'FAIL'}, indent=2))
        if not all(checks.values()):
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
