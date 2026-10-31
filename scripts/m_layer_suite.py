#!/usr/bin/env python3
"""B5.4-B5.10: the native ten-layer suite.

Exports every captured Swin layer as its own graph fixture, executes each one
independently, compares it with the captured output, and then rebuilds the whole
network as one native D3D12 graph whose intermediate tensors stay on the GPU.

  suite.py export CAPTURE --out DIR
  suite.py run DIR [--probe EXE] [--shaders DIR]
  suite.py check DIR [--json OUT]
  suite.py chain DIR [--json OUT]

`check` is the independent-replay mode (mode A): each layer reads its captured
input.  `chain` is mode B: enc1 reads the captured enc1 input, every later layer
reads the field its native predecessor produced, and the final dec1 field is
compared with the capture.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import m_layer_fixture as lf      # noqa: E402
import m_layer_oracle as lo       # noqa: E402
import m_model as mm              # noqa: E402

LABELS = ['enc1', 'enc2'] + [f'tube#{i}' for i in range(1, 7)] + ['dec2', 'dec1']
STAGE_NAMES = ['enc1', 'enc2'] + ['enc3_tube'] * 6 + ['dec2', 'dec1']


def fixture(directory, label):
    return os.path.join(directory, re.sub(r'[^A-Za-z0-9]+', '-', label))


def export_all(capture, out):
    os.makedirs(out, exist_ok=True)
    for label in LABELS:
        target = fixture(out, label)
        print(f'== {label} -> {target}', flush=True)
        lf.export(capture, label, target, 1)
    return 0


def run_one(path, probe, shaders, seed=0, results='results'):
    return lf.run(path, probe, shaders, seed, os.path.join(path, results))


def run_all(directory, probe, shaders):
    for label in LABELS:
        path = fixture(directory, label)
        print(f'== run {label}', flush=True)
        run_one(path, probe, shaders)
    return 0


#: Declared before the run: every layer output must reproduce at least this
#: fraction of codes (the measured minima are 91.1% merge / 93.6% tube, so the
#: floor is not the claim, only the regression bound).
LAYER_FLOOR = 0.90
#: The chain acceptance: a native ten-layer chain must reproduce the captured
#: dec1 field at this fraction.  It currently does not - the suite therefore
#: reports B5 as failed, not as a regression (EXP-019).
CHAIN_FLOOR = 0.99


def check_all(directory, probe, shaders, json_out):
    rows, problems = [], []
    for label, name in zip(LABELS, STAGE_NAMES):
        path = fixture(directory, label)
        record = json.load(open(os.path.join(path, 'fixture.json')))
        run_one(path, probe, shaders, 0, 'results')
        run_one(path, probe, shaders, 0, 'results-repeat')
        row = {'layer': label, 'outputs': {}, 'deterministic': True}
        for role, entry in record['outputs'].items():
            got = np.fromfile(os.path.join(path, 'results', f'out-{role}.bin'), np.uint8)
            again = np.fromfile(os.path.join(path, 'results-repeat', f'out-{role}.bin'),
                                np.uint8)
            ref = np.fromfile(os.path.join(path, f'ref-{role}.bin'), np.uint8)
            if got.size != entry['bytes'] or ref.size != entry['bytes']:
                problems.append(f'{label}.{role}: size {got.size}/{ref.size}, '
                                f'declared {entry["bytes"]}')
                continue
            if not np.array_equal(got, again):
                row['deterministic'] = False
                problems.append(f'{label}.{role}: two runs differ')
            delta = np.abs(got.astype(np.int16) - ref.astype(np.int16))
            row['outputs'][role] = {
                'bytes': int(got.size),
                'identical': float((got == ref).mean()),
                'differing': int((got != ref).sum()),
                'max_code_delta': int(delta.max()),
            }
        rows.append(row)
        summary = '  '.join(f'{r} {100 * v["identical"]:7.3f}% max {v["max_code_delta"]}'
                            for r, v in row['outputs'].items())
        print(f'{label:9s} {summary}  deterministic={row["deterministic"]}')
        for role, value in row['outputs'].items():
            if value['identical'] < LAYER_FLOOR:
                problems.append(f'{label}.{role}: {100 * value["identical"]:.3f}% '
                                f'below the declared floor {100 * LAYER_FLOOR:.1f}%')
    result = {'layers': rows, 'problems': problems,
              'pass': not problems and all(r['deterministic'] for r in rows)}
    if json_out:
        json.dump(result, json_out, indent=2)
    print(f'\n{len(problems)} problem(s)')
    for p in problems:
        print(f'  {p}')
    return 0 if result['pass'] else 1


# ---------------------------------------------------------------------------
# mode B: the native chain


#: producer field -> consumer input (the low-resolution field of a decoder is
#: its patch-expand source, not a `prepare` binding)
CHAIN_EDGES = {
    'enc1.merge': ['enc2.in'],
    'enc2.merge': ['tube#1.in'],
    'tube#1.out': ['tube#2.in'],
    'tube#2.out': ['tube#3.in'],
    'tube#3.out': ['tube#4.in'],
    'tube#4.out': ['tube#5.in'],
    'tube#5.out': ['tube#6.in'],
    'tube#6.out': ['dec2.in'],
    'dec2.out': ['dec1.in'],
}


def _parse_graph(path):
    resources, stages, dumps = [], [], []
    stage = None
    for line in open(path, encoding='utf-8'):
        words = line.split()
        if not words:
            continue
        if words[0] == 'version':
            continue
        if words[0] == 'resource':
            resources.append({'name': words[1], 'bytes': int(words[2]),
                              'init': words[3] if len(words) > 3 else None})
        elif words[0] == 'stage':
            stage = {'name': words[1], 'lines': []}
            stages.append(stage)
        elif words[0] == 'dump':
            dumps.append({'resource': words[1], 'offset': int(words[2]),
                          'bytes': int(words[3]), 'file': words[4]})
        elif stage is not None:
            stage['lines'].append(words)
    return resources, stages, dumps


def validate_mlp_stages(stages, mode):
    """Bounded MLP mode gate: reject missing projections and FP16 matrix fallback."""
    selected = [s for s in stages if s['name'].startswith('enc1.mlp')]
    expected = [('enc1.mlp-quant', 'swin_q8_quant.cso'), ('enc1.mlp-up', 'swin_mlp_up_q8.cso')]
    if mode == 'int8-up':
        expected += [('enc1.mlp', 'swin_mlp_up_epilogue.cso')]
    elif mode == 'int8-pair':
        expected += [('enc1.mlp-gelu', 'swin_mlp_gelu.cso'),
                     ('enc1.mlp-down-quant', 'swin_q8_quant.cso'),
                     ('enc1.mlp-down', 'swin_mlp_down_q8.cso')]
    else:
        raise SystemExit('unsupported MLP mode')
    actual = [(s['name'], os.path.basename(next(w[1] for w in s['lines'] if w[0] == 'shader')))
              for s in selected]
    if actual != expected:
        raise SystemExit('unexpected MLP stage sequence or matrix shader fallback')


def fuse_mlp_stages(stages, shader='swin_mlp_gelu_quant.cso', layers=None):
    """Replace each (GELU, down-quantizer) pair with one fused dispatch.

    The row maximum is still reduced by the quantizer's four lanes inside one
    threadgroup, so no token crosses a threadgroup boundary.  The fused stage
    reads the MLP-up f16 output and writes the GELU diagnostic tensor plus the
    down-projection codes and scales; the GELU and quantizer stages disappear.
    """
    def line(stage, key, reg=None):
        found = [w for w in stage['lines'] if w and w[0] == key
                 and (reg is None or w[1] == str(reg))]
        if len(found) != 1:
            raise SystemExit(f'{stage["name"]}: expected one {key} line'
                             + (f' for register {reg}' if reg is not None else ''))
        return found[0]

    fused, count, index = [], 0, 0
    while index < len(stages):
        stage = stages[index]
        if not stage['name'].endswith('.mlp-gelu'):
            fused.append(stage)
            index += 1
            continue
        prefix = stage['name'][:-len('.mlp-gelu')]
        if layers is not None and prefix not in layers:
            fused.append(stage)
            index += 1
            continue
        if index + 1 >= len(stages) or not stages[index + 1]['name'].endswith(
                '.mlp-down-quant'):
            raise SystemExit(f'{stage["name"]}: no following down-quantizer')
        gelu, quant = stage, stages[index + 1]
        z = line(gelu, 'srv', 0)[2]
        g = line(gelu, 'uav', 0)[2]
        if line(quant, 'srv', 0)[2] != g:
            raise SystemExit(f'{quant["name"]}: does not consume {g}')
        codes, scales = line(quant, 'uav', 0)[2], line(quant, 'uav', 1)[2]
        params = line(quant, 'params')
        groups = line(quant, 'groups')
        fused.append({'name': prefix + '.mlp-fused', 'lines': [
            ['shader', shader], ['threads', '256', '1', '1'], groups, params,
            ['srv', '0', z], ['uav', '0', g], ['uav', '1', codes],
            ['uav', '2', scales]]})
        count += 1
        index += 2
    if count == 0:
        raise SystemExit('chain has no MLP GELU/quantizer pair to fuse')
    return fused, count


def build_chain(directory, out, q_int8=False, enc1_int8=None, mlp_int8=False, enc1_mlp=None,
                mlp_mode=None, cumulative_diagnostics=False, fuse_mlp=False,
                fuse_mlp_layers=None):
    """Concatenate the ten layer graphs, remapping every produced field.

    A resource `<layer>_<name>` is local to that layer, except the fields named
    in `CHAIN_EDGES`: the consumer reads the producer's resource directly, so no
    intermediate leaves the GPU.

    `q_int8` (M09 I4) takes enc1's graph from `enc1_int8` instead of
    `directory/enc1`, and changes nothing else: the fixture it points at already
    contains exactly one replaced projection (EXP-023).
    """
    os.makedirs(out, exist_ok=True)
    resources, stages, dumps = [], [], []
    produced = {}
    cumulative_aliases = {}
    overrides = {}
    if mlp_int8:
        if q_int8 or not enc1_mlp:
            raise SystemExit('MLP INT8 requires its own enc1 fixture and cannot combine modes')
        record = json.load(open(os.path.join(enc1_mlp, 'fixture.json')))
        selected_mlp_mode = record.get('mlp_mode')
        if selected_mlp_mode not in ('int8-up', 'int8-pair') or record.get('q_mode', 'fp16') != 'fp16':
            raise SystemExit('expected an enc1 INT8 MLP-only fixture')
        if mlp_mode is not None and selected_mlp_mode != mlp_mode:
            raise SystemExit('requested MLP mode does not match its fixture')
        overrides['enc1'] = enc1_mlp
    if q_int8:
        if not enc1_int8:
            raise SystemExit('q_int8 needs the INT8 enc1 fixture directory (enc1_int8)')
        overrides['enc1'] = enc1_int8
    for label, name in zip(LABELS, STAGE_NAMES):
        path = overrides.get(label) or fixture(directory, label)
        local_dir = path
        layer_resources, layer_stages, local_dumps = _parse_graph(os.path.join(path, 'graph.txt'))
        record = json.load(open(os.path.join(path, 'fixture.json')))
        outputs = record['outputs']
        # names this layer produces, keyed by role
        role_resource = {'out': None, 'merge': None}
        for stage in layer_stages:
            for words in stage['lines']:
                if words[0] == 'uav' and words[2] == 'out':
                    role_resource['out'] = words[2]
                if words[0] == 'uav' and words[2] == 'merge':
                    role_resource['merge'] = words[2]
        rename = lambda n: f'{re.sub(r"[^A-Za-z0-9]+", "_", label)}_{n}'
        if cumulative_diagnostics and record.get('int8_batch'):
            for d in local_dumps:
                if d['file'].startswith('dbg-'):
                    tag = re.sub(r'[^A-Za-z0-9]+', '-', label)
                    dumps.append(f'dump {rename(d["resource"])} {d["offset"]} {d["bytes"]} {tag}-{d["file"]}')
        if mlp_int8 and label == 'enc1':
            for d in local_dumps:
                if d['file'].startswith('dbg-'):
                    dumps.append(f'dump {rename(d["resource"])} {d["offset"]} {d["bytes"]} enc1-{d["file"]}')
        for r in layer_resources:
            entry = {'name': rename(r['name']), 'bytes': r['bytes'], 'init': None}
            if r['init']:
                entry['init'] = os.path.join(local_dir, r['init'])
            resources.append(entry)
        for stage in layer_stages:
            new = {'name': f'{label}.{stage["name"]}', 'lines': []}
            for words in stage['lines']:
                if words[0] in ('srv', 'uav'):
                    new['lines'].append([words[0], words[1], rename(words[2])])
                elif words[0] == 'params':
                    new['lines'].append([words[0], os.path.join(local_dir, words[1]),
                                         words[2]])
                elif words[0] == 'shader':
                    new['lines'].append([words[0], os.path.join(local_dir, words[1])])
                else:
                    new['lines'].append(words)
            stages.append(new)
        produced[f'{label}.out'] = rename('out')
        if outputs.get('merge'):
            produced[f'{label}.merge'] = rename('merge')

    # rewire the consumer inputs: encoder `prepare` reads `in`, a decoder's
    # `expand` matmul reads the low-resolution `low` field
    for key, targets in CHAIN_EDGES.items():
        source = produced.get(key)
        if source is None:
            raise SystemExit(f'chain: {key} has no producer')
        for target in targets:
            layer, _ = target.split('.')
            safe = re.sub(r'[^A-Za-z0-9]+', '_', layer)
            for stage in stages:
                if not stage['name'].startswith(layer + '.'):
                    continue
                for words in stage['lines']:
                    if words[0] != 'srv':
                        continue
                    if words[2] == f'{safe}_in' and stage['name'].endswith('.prepare'):
                        words[2] = source
                    if words[2] == f'{safe}_low' and stage['name'].endswith(('.expand', '.expand-unpack')):
                        words[2] = source
                        cumulative_aliases[f'{safe}_low'] = source
    # decoder skip fields
    for consumer, producer in (('dec2', 'enc2.out'), ('dec1', 'enc1.out')):
        safe = re.sub(r'[^A-Za-z0-9]+', '_', consumer)
        for stage in stages:
            if stage['name'].startswith(consumer + '.') and stage['name'].endswith('.prepare'):
                for words in stage['lines']:
                    if words[0] == 'srv' and words[2] == f'{safe}_skip':
                        words[2] = produced[producer]

    if q_int8:
        # the replaced projection must be the only difference in the graph
        replaced = [s['name'] for s in stages
                    if s['name'].startswith('enc1.') and s['name'].endswith(('q-quant', 'q'))]
        if replaced != ['enc1.q-quant', 'enc1.q']:
            raise SystemExit(f'hybrid chain: unexpected enc1 stages {replaced}')

    final = produced['dec1.out']
    if mlp_int8:
        validate_mlp_stages(stages, selected_mlp_mode)
    fused_mlp = 0
    if fuse_mlp:
        stages, fused_mlp = fuse_mlp_stages(stages, layers=fuse_mlp_layers)
    lines = ['version 1']
    for r in resources:
        lines.append(f'resource {r["name"]} {r["bytes"]}' +
                     (f' {r["init"]}' if r['init'] else ''))
    for stage in stages:
        lines.append(f'stage {stage["name"]}')
        for words in stage['lines']:
            lines.append(' '.join(words))
    # every produced field is exported so propagation can be measured stage by stage
    for label, name in zip(LABELS, STAGE_NAMES):
        record = json.load(open(os.path.join(fixture(directory, label), 'fixture.json')))
        for role, entry in record['outputs'].items():
            key = f'{label}.{role}'
            if key in produced:
                tag = re.sub(r'[^A-Za-z0-9]+', '-', key)
                lines.append(f'dump {produced[key]} 0 {entry["bytes"]} out-{tag}.bin')
    bytes_ = json.load(open(os.path.join(fixture(directory, 'dec1'), 'fixture.json')))[
        'outputs']['out']['bytes']
    lines.append(f'dump {final} 0 {bytes_} out-dec1.bin')
    for line in dumps:
        words = line.split()
        words[1] = cumulative_aliases.get(words[1], words[1])
        lines.append(' '.join(words))
    open(os.path.join(out, 'graph.txt'), 'w').write('\n'.join(lines) + '\n')
    print(f'chain: {len(stages)} dispatches, {len(resources)} resources'
          + (f', {fused_mlp} fused MLP stage(s)' if fused_mlp else '') + f' -> {out}')
    return {'stages': len(stages), 'resources': len(resources),
            'final': final, 'bytes': bytes_, 'fused_mlp': fused_mlp}


def run_chain(directory, probe, shaders, json_out, q_int8=False, enc1_int8=None,
              fuse_mlp=False):
    out = os.path.join(directory, 'chain-int8q' if q_int8 else 'chain')
    info = build_chain(directory, out, q_int8=q_int8, enc1_int8=enc1_int8,
                       fuse_mlp=fuse_mlp)
    results = os.path.join(out, 'results')
    lf.run(out, probe, shaders, 0, results)
    again = os.path.join(out, 'results-repeat')
    lf.run(out, probe, shaders, 0, again)
    got = np.fromfile(os.path.join(results, 'out-dec1.bin'), np.uint8)
    repeat = np.fromfile(os.path.join(again, 'out-dec1.bin'), np.uint8)
    ref = np.fromfile(os.path.join(fixture(directory, 'dec1'), 'ref-out.bin'), np.uint8)
    problems = []
    if got.size != info['bytes'] or ref.size != info['bytes']:
        problems.append(f'dec1: size {got.size}/{ref.size}, declared {info["bytes"]}')
    if not np.array_equal(got, repeat):
        problems.append('chain: two runs differ')
    identical = float((got == ref).mean())
    delta = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    result = {'chain': info, 'identical': identical,
              'differing': int((got != ref).sum()), 'samples': int(got.size),
              'max_code_delta': int(delta.max()),
              'deterministic': bool(np.array_equal(got, repeat)),
              'floor': CHAIN_FLOOR, 'problems': problems}
    if identical < CHAIN_FLOOR:
        problems.append(f'chain: {100 * identical:.3f}% identical, below the declared '
                        f'chain floor {100 * CHAIN_FLOOR:.1f}%')
    if json_out:
        json.dump(result, json_out, indent=2)
    print(f'chain dec1: identical {100 * identical:8.4f}%  differ {result["differing"]}  '
          f'max {result["max_code_delta"]}  deterministic={result["deterministic"]}')
    for p in problems:
        print(f'  {p}')
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('capture')
    e.add_argument('--out', required=True)
    r = sub.add_parser('run')
    r.add_argument('directory')
    r.add_argument('--probe', default=os.path.join('build', 'native', 'Release',
                                                   'm_network_stage.exe'))
    r.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    c = sub.add_parser('check')
    c.add_argument('directory')
    c.add_argument('--probe', default=os.path.join('build', 'native', 'Release',
                                                   'm_network_stage.exe'))
    c.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    c.add_argument('--json', type=argparse.FileType('w'), default=None)
    h = sub.add_parser('chain')
    h.add_argument('directory')
    h.add_argument('--probe', default=os.path.join('build', 'native', 'Release',
                                                   'm_network_stage.exe'))
    h.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    h.add_argument('--json', type=argparse.FileType('w'), default=None)
    h.add_argument('--q-mode', choices=['fp16', 'int8-q'], default='fp16')
    h.add_argument('--enc1-int8', default=None,
                   help='enc1 fixture with the INT8 Q projection (int8-q mode)')
    h.add_argument('--fuse-mlp', action='store_true',
                   help='fuse the post-MLP GELU with the down-projection quantizer')
    a = ap.parse_args()
    if a.command == 'export':
        return export_all(a.capture, a.out)
    if a.command == 'run':
        return run_all(a.directory, a.probe, a.shaders)
    if a.command == 'check':
        return check_all(a.directory, a.probe, a.shaders, a.json)
    if a.q_mode == 'int8-q':
        return run_chain(a.directory, a.probe, a.shaders, a.json,
                         q_int8=True, enc1_int8=a.enc1_int8,
                         fuse_mlp=a.fuse_mlp)
    return run_chain(a.directory, a.probe, a.shaders, a.json,
                     fuse_mlp=a.fuse_mlp)


if __name__ == '__main__':
    sys.exit(main())
