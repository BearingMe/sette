#!/usr/bin/env python3
"""Run the contract comparison for every layer that has a translated-PTX capture.

Each translated capture is one kernel excluded from d4r's native replacement
list. The layer's fixture is exported from that same capture, so Sette, the
pinned conservative/fast modules and the translated PTX all consume identical
input bytes. This is mode A (independent layers): the captured input is the
real network's, not a substituted intermediate.

With `--chain-*` it also measures mode B: the chained boundaries of a capture in
which every network kernel ran translated, against Sette's own chain and against
the canonical capture. The acceptance criteria live in `m_conformance.py`.

  m_translated_layer_suite.py --enc2 DIR --tube DIR --dec2 DIR --dec1 DIR --out DIR
  ... --chain-enc1 DIR --chain-enc2 DIR --chain-tube DIR --chain-dec2 DIR --chain-dec1 DIR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import m_layer_fixture as lf
import m_layer_oracle as lo
from m_enc1_contract import metrics

LAYERS = {'enc1': ('rrlite_enc1_4x4', ['enc1']),
          'enc2': ('rrlite_enc2_4x4', ['enc2']),
          'tube': ('rrlite_enc3_tube_4x4', [f'tube#{i}' for i in range(1, 7)]),
          'dec2': ('rrlite_dec2_4x4', ['dec2']),
          'dec1': ('rrlite_dec1_4x4', ['dec1'])}
KERNELS = {'enc1': 'rrlite_enc1_4x4', 'enc2': 'rrlite_enc2_4x4',
           'enc3_tube': 'rrlite_enc3_tube_4x4', 'dec2': 'rrlite_dec2_4x4',
           'dec1': 'rrlite_dec1_4x4'}
CHAIN = {'enc1': ('enc1', 'chain_enc1'),
         'enc2': ('enc2', 'chain_enc2'),
         'tube#1': ('enc3_tube', 'chain_tube'),
         'tube#2': ('enc3_tube', 'chain_tube'),
         'tube#3': ('enc3_tube', 'chain_tube'),
         'tube#4': ('enc3_tube', 'chain_tube'),
         'tube#5': ('enc3_tube', 'chain_tube'),
         'tube#6': ('enc3_tube', 'chain_tube'),
         'dec2': ('dec2', 'chain_dec2'),
         'dec1': ('dec1', 'chain_dec1')}
PAIRS = ('sette-vs-translated', 'conservative-vs-translated',
         'conservative-vs-sette', 'fast-vs-translated')
DEFAULT_LAYERS = os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp',
                              'opencode', 'm-capture-run', 'layers')


def replays(root, kernel):
    return sorted((Path(root) / 'capture').glob(f'replay-*-{kernel}'))


def module_sha(root, kernel):
    """The dumped module's SHA-256, from run.json or by matching the entry point."""
    run = Path(root) / 'run.json'
    if run.is_file():
        modules = json.loads(run.read_text()).get('ptx_modules') or []
        if modules:
            return modules[0]['sha256']
    for path in (Path(root) / 'ptx').glob('*.ptx'):
        if f'.entry {kernel}(' in path.read_text():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    raise SystemExit(f'{root}: no PTX module for {kernel}')


def describe(rows):
    text = f'{100 * rows["output"]["exact"]:.4f}% output'
    if 'merge' in rows:
        text += f', {100 * rows["merge"]["exact"]:.4f}% merge'
    return text


def chain_boundaries(a, summary):
    """Mode B: chained agreement at every boundary, three pairings."""
    roots = {key: getattr(a, value) for key, (layer, value) in CHAIN.items()}
    if any(value is None for value in roots.values()):
        return False
    summary['chain'] = {}
    for label, (layer, field) in CHAIN.items():
        index = 1
        if label.startswith('tube#'):
            index = int(label.split('#')[1])
        found = replays(roots[label], KERNELS[layer])
        if len(found) < index:
            raise SystemExit(f'chain {label}: no replay {index} of {KERNELS[layer]}')
        replay = found[index - 1]
        P, raw = lo.raw_layer(str(replay), layer)
        run = replay.parent.parent / 'run.json'
        run = json.loads(run.read_text()) if run.is_file() else {}
        tag = label.replace('#', '-')
        sette = Path(a.sette_chain) / f'out-{tag}-out.bin'
        if not sette.is_file():
            raise SystemExit(f'chain {label}: missing Sette boundary {sette}')
        canonical = Path(a.canonical) / tag / 'ref-out.bin'
        sette_codes = np.fromfile(sette, np.uint8)
        rows = {'translated-vs-sette': {'output': metrics(raw['output'], sette_codes)}}
        if canonical.is_file():
            canon = np.fromfile(canonical, np.uint8)
            rows['canonical-vs-translated'] = {'output': metrics(canon, raw['output'])}
            rows['canonical-vs-sette'] = {'output': metrics(canon, sette_codes)}
        summary['chain'][label] = {
            'layer': layer, 'replay': str(replay), 'pairs': rows,
            'translated_dispatches': run.get('translated_dispatches'),
            'native_dispatches': run.get('native_dispatches'),
            'excluded_kernels': run.get('excluded_kernels'),
            'translated_ptx_sha256': [m['sha256'] for m in run.get('ptx_modules', [])]}
        print(f'chain {label:7s} {describe(rows["translated-vs-sette"])} translated-vs-sette'
              + (f'; canonical-vs-translated '
                 f'{100 * rows["canonical-vs-translated"]["output"]["exact"]:.4f}%'
                 if 'canonical-vs-translated' in rows else ''), flush=True)
    return True


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    for key in LAYERS:
        ap.add_argument(f'--{key}', type=Path, required=True)
    for key in ('enc1', 'enc2', 'tube', 'dec2', 'dec1'):
        ap.add_argument(f'--chain-{key}', type=Path, default=None)
    ap.add_argument('--sette-chain', type=Path, default=Path(DEFAULT_LAYERS, 'chain', 'results'))
    ap.add_argument('--canonical', type=Path, default=Path(DEFAULT_LAYERS))
    ap.add_argument('--resume', action='store_true',
                    help='reuse completed layer contracts already in --out')
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=a.resume)
    summary = {'layers': {}, 'status': 'running'}
    for key, (kernel, labels) in LAYERS.items():
        found = replays(getattr(a, key), kernel)
        if len(found) != len(labels):
            raise SystemExit(f'{key}: {len(found)} replay(s) for {len(labels)} layer(s)')
        for label, replay in zip(labels, found):
            name = label.replace('#', '-')
            layer = 'enc3_tube' if label.startswith('tube') else label
            fixture = a.out / f'{name}-fixture'
            layer_out = a.out / f'{name}-contract'
            report_path = layer_out / 'report.json'
            if not (a.resume and report_path.is_file()):
                lf.export(str(replay), label, str(fixture), 1)
                command = [sys.executable, repo / 'scripts/m_enc1_contract.py',
                           '--out', layer_out, '--kernel', kernel, '--layer', layer,
                           '--capture', replay, '--translated', replay,
                           '--fixture', fixture]
                done = subprocess.run([str(x) for x in command], capture_output=True, text=True)
                (a.out / f'{name}-contract.log').write_text(done.stdout + done.stderr)
                if done.returncode:
                    summary['layers'][label] = {'status': 'failed',
                                                'returncode': done.returncode}
                    continue
            report = json.loads(report_path.read_text())
            rows = {pair: report['comparisons'][pair] for pair in PAIRS
                    if pair in report['comparisons']}
            if 'sette_execution' not in summary:
                summary['sette_execution'] = {
                    'probe': report['sette_execution']['probe'],
                    'probe_sha256': report['sette_execution']['probe_sha256'],
                    'shaders': report['sette_execution']['shaders']}
            summary['layers'][label] = {
                'status': 'pass', 'layer': layer, 'kernel': kernel,
                'contract': str(layer_out / 'report.json'),
                'translated_dispatch': report['translated_control']['dispatch'],
                'translated_ptx_sha256': module_sha(getattr(a, key), kernel),
                'input_sha256': report['input_sha256'],
                'module_sha256': {m: b['module_sha256'] for m, b in report['builds'].items()},
                'deterministic': {m: report['runs'][m + '-deterministic']
                                  for m in ('fast', 'conservative', 'sette')},
                'pairs': rows}
            print(f'{label:7s} sette-vs-translated {describe(rows["sette-vs-translated"])}; '
                  f'conservative-vs-translated {describe(rows["conservative-vs-translated"])}',
                  flush=True)
    summary['chain_measured'] = chain_boundaries(a, summary)
    summary['status'] = 'pass' if all(v['status'] == 'pass' for v in summary['layers'].values()) else 'fail'
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
