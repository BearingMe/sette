#!/usr/bin/env python3
"""Run the contract comparison for every layer that has a translated-PTX capture.

Each translated capture is one kernel excluded from d4r's native replacement
list. The layer's fixture is exported from that same capture, so Sette, the
pinned conservative/fast modules and the translated PTX all consume identical
input bytes. This is mode A (independent layers): the captured input is the
real network's, not a substituted intermediate.

  m_translated_layer_suite.py --enc2 DIR --tube DIR --dec2 DIR --dec1 DIR --out DIR
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import m_layer_fixture as lf

LAYERS = {'enc2': ('rrlite_enc2_4x4', ['enc2']),
          'tube': ('rrlite_enc3_tube_4x4', [f'tube#{i}' for i in range(1, 7)]),
          'dec2': ('rrlite_dec2_4x4', ['dec2']),
          'dec1': ('rrlite_dec1_4x4', ['dec1'])}
PAIRS = ('sette-vs-translated', 'conservative-vs-translated',
         'conservative-vs-sette', 'fast-vs-translated')


def replays(root, kernel):
    return sorted((Path(root) / 'capture').glob(f'replay-*-{kernel}'))


def describe(rows):
    text = f'{100 * rows["output"]["exact"]:.4f}% output'
    if 'merge' in rows:
        text += f', {100 * rows["merge"]["exact"]:.4f}% merge'
    return text


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    for key in LAYERS:
        ap.add_argument(f'--{key}', type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    summary = {'layers': {}, 'status': 'running'}
    for key, (kernel, labels) in LAYERS.items():
        found = replays(getattr(a, key), kernel)
        if len(found) != len(labels):
            raise SystemExit(f'{key}: {len(found)} replay(s) for {len(labels)} layer(s)')
        for label, replay in zip(labels, found):
            name = label.replace('#', '-')
            layer = 'enc3_tube' if label.startswith('tube') else label
            fixture = a.out / f'{name}-fixture'
            lf.export(str(replay), label, str(fixture), 1)
            layer_out = a.out / f'{name}-contract'
            command = [sys.executable, repo / 'scripts/m_enc1_contract.py',
                       '--out', layer_out, '--kernel', kernel, '--layer', layer,
                       '--capture', replay, '--translated', replay,
                       '--fixture', fixture]
            done = subprocess.run([str(x) for x in command], capture_output=True, text=True)
            (a.out / f'{name}-contract.log').write_text(done.stdout + done.stderr)
            if done.returncode:
                summary['layers'][label] = {'status': 'failed', 'returncode': done.returncode}
                continue
            report = json.loads((layer_out / 'report.json').read_text())
            rows = {pair: report['comparisons'][pair] for pair in PAIRS
                    if pair in report['comparisons']}
            summary['layers'][label] = {
                'status': 'pass', 'layer': layer, 'kernel': kernel,
                'contract': str(layer_out / 'report.json'),
                'translated_dispatch': report['translated_control']['dispatch'],
                'translated_ptx_sha256': report['translated_control']['configuration']['ptx_modules'][0]['sha256'],
                'input_sha256': report['input_sha256'],
                'module_sha256': {m: b['module_sha256'] for m, b in report['builds'].items()},
                'deterministic': {m: report['runs'][m + '-deterministic']
                                  for m in ('fast', 'conservative', 'sette')},
                'pairs': rows}
            print(f'{label:7s} sette-vs-translated {describe(rows["sette-vs-translated"])}; '
                  f'conservative-vs-translated {describe(rows["conservative-vs-translated"])}',
                  flush=True)
    summary['status'] = 'pass' if all(v['status'] == 'pass' for v in summary['layers'].values()) else 'fail'
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
