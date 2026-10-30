#!/usr/bin/env python3
"""M13 selective precision: build and audit mixed INT8/FP16 layer fixtures.

The accepted full-INT8 graph is the baseline.  This module produces *explicit*
mixed configurations by restoring one named family in one layer to its original
FP16 stage, leaving every other stage exactly as the accepted fixture declares.

A family is the smallest substitution the existing shaders support:

    q, v, wo        one projection
    attention       the whole QK/softmax/PV block (no FP16 half exists)
    mlp             the fused FP16 MLP (up and down together)
    merge           the merge projection
    expand          the decoder expansion projection

The variant directory contains only `graph.txt` and `fixture.json`; resources
and parameters are referenced by absolute path from the source fixtures, so a
variant costs kilobytes rather than gigabytes and cannot silently drift from
the accepted weights.

The audit re-derives the expected stage sequence and shader for every family, so
a variant cannot pass with an unexpected fallback or an omitted INT8 operator.

  m13_selective.py --int8 LAYERS --fp16 LAYERS --out DIR --restore enc1:mlp,...
  m13_selective.py --int8 LAYERS --fp16 LAYERS --out DIR --restore-family mlp
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m_layer_suite as suite

LABELS = suite.LABELS

#: family -> (INT8 stage names, original FP16 stage name, expected FP16 shader,
#:            INT8 shader for the operator)
FAMILIES = {
    'q': (['q-quant', 'q'], 'q', 'swin_matmul.cso', 'swin_mlp_up_q8.cso'),
    'v': (['v-quant', 'v'], 'v', 'swin_matmul.cso', 'swin_mlp_up_q8.cso'),
    'wo': (['wo-quant', 'wo'], 'wo', 'swin_matmul.cso', 'swin_mlp_down_q8.cso'),
    'attention': (['attention-pack', 'attention-q-quant', 'attention-v-quant',
                   'attention-qkt', 'attention-softmax', 'attention-p-quant',
                   'attention-pv'],
                  'attention', 'swin_attention.cso', None),
    'mlp': (['mlp-up-quant', 'mlp-up', 'mlp-gelu', 'mlp-down-quant', 'mlp-down'],
            'mlp', 'swin_mlp.cso', None),
    'merge': (['merge-gather', 'merge-quant', 'merge', 'merge-store'],
              'merge', 'swin_merge.cso', 'swin_mlp_up_q8.cso'),
    'expand': (['expand-unpack', 'expand-quant', 'expand'],
               'expand', 'swin_matmul.cso', 'swin_mlp_up_q8.cso'),
}

#: INT8 operator entry produced by `m09_cumulative` -> family.
OPERATOR_FAMILY = {'q': 'q', 'v': 'v', 'wo': 'wo', 'mlp-up': 'mlp',
                   'mlp-down': 'mlp', 'merge': 'merge', 'expand': 'expand'}
#: family -> operator stage names (attention is tracked by its own record).
OPERATOR_STAGES = {'q': {'q'}, 'v': {'v'}, 'wo': {'wo'},
                   'mlp': {'mlp-up', 'mlp-down'}, 'merge': {'merge'},
                   'expand': {'expand'}}
#: operator stage -> accepted INT8 shader.
OPERATOR_SHADER = {'q': 'swin_mlp_up_q8.cso', 'v': 'swin_mlp_up_q8.cso',
                   'wo': 'swin_mlp_down_q8.cso', 'mlp-up': 'swin_mlp_up_q8.cso',
                   'mlp-down': 'swin_mlp_down_q8.cso',
                   'merge': 'swin_mlp_up_q8.cso', 'expand': 'swin_mlp_up_q8.cso'}
#: diagnostics that only exist while a family is INT8.
FAMILY_LAYOUTS = {'mlp': {'gelu'}, 'merge': {'gather', 'merge-store'},
                  'expand': {'unpack'}}
#: inventory family -> restoration family.
INVENTORY_FAMILY = {'q': 'q', 'v': 'v', 'wo': 'wo', 'up': 'mlp', 'down': 'mlp',
                    'qkt': 'attention', 'pv': 'attention', 'merge': 'merge',
                    'expand': 'expand'}

DENOMINATOR_MACS = 34518597632
DENOMINATOR_OPERATORS = 76


def split_graph(lines):
    """[('raw', line) | ('stage', name, body)] preserving every non-stage line."""
    items, i = [], 0
    while i < len(lines):
        words = lines[i].split()
        if words and words[0] == 'stage':
            name, body = words[1], []
            i += 1
            while i < len(lines):
                nxt = lines[i].split()
                if nxt and nxt[0] in ('stage', 'dump', 'resource', 'version'):
                    break
                body.append(lines[i])
                i += 1
            items.append(('stage', name, body))
        else:
            items.append(('raw', lines[i]))
            i += 1
    return items


def stage_blocks(lines):
    """[(name, [lines])] for every `stage` block in graph order."""
    return [(item[1], item[2]) for item in split_graph(lines) if item[0] == 'stage']


def flatten(items):
    out = []
    for item in items:
        if item[0] == 'raw':
            out.append(item[1])
        else:
            out.append('stage ' + item[1])
            out.extend(item[2])
    return out


def read_graph(path):
    return Path(path).read_text(encoding='utf-8').splitlines()


def absolute(lines, base):
    """Point relative `params` and `resource` init paths at `base`."""
    out = []
    for line in lines:
        words = line.split()
        index = None
        if words and words[0] == 'params' and len(words) > 1:
            index = 1
        elif words and words[0] == 'resource' and len(words) > 3:
            index = 3
        if index is not None:
            p = Path(words[index])
            words[index] = str(p if p.is_absolute() else (Path(base) / p))
            line = ' '.join(words)
        out.append(line)
    return out


def available(int8_layers, label):
    """Families the INT8 fixture actually converted in this layer."""
    record = json.loads((Path(suite.fixture(str(int8_layers), label)) /
                         'fixture.json').read_text())
    names = {op['name'] for op in record.get('int8_operations', [])}
    families = {OPERATOR_FAMILY[n] for n in names if n in OPERATOR_FAMILY}
    if record.get('int8_attention'):
        families.add('attention')
    return families


def build(int8_layers, fp16_layers, out, restore, asym=False):
    """Write a mixed ten-layer fixture tree; `restore` is {(label, family)}."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists; use a fresh directory')
    out.mkdir(parents=True)
    restore = set(restore)
    int8_layers, fp16_layers = Path(int8_layers), Path(fp16_layers)
    known = {(label, family) for label in LABELS for family in FAMILIES}
    unknown = restore - known
    if unknown:
        raise SystemExit(f'unknown restore targets: {sorted(unknown)}')
    manifest = {'int8_batch': 'selective', 'restored': sorted(f'{l}:{f}' for l, f in restore),
                'int8_layers': str(int8_layers), 'fp16_layers': str(fp16_layers),
                'layers': {}}
    for label in LABELS:
        source = Path(suite.fixture(str(int8_layers), label))
        original = Path(suite.fixture(str(fp16_layers), label))
        record = json.loads((source / 'fixture.json').read_text())
        fp16_record = json.loads((original / 'fixture.json').read_text())
        if record.get('int8_batch') != 'spatial':
            raise SystemExit(f'{label}: expected the accepted spatial INT8 fixture')
        if record['geometry'] != fp16_record['geometry']:
            raise SystemExit(f'{label}: source fixtures disagree on geometry')
        lines = absolute(read_graph(source / 'graph.txt'), source)
        original_blocks = dict(stage_blocks(absolute(read_graph(original / 'graph.txt'),
                                                     original)))
        declared = available(int8_layers, label)
        selected = {family for l, family in restore if l == label}
        missing = selected - declared
        if missing:
            raise SystemExit(f'{label}: cannot restore non-INT8 family {sorted(missing)}')

        items = split_graph(lines)
        owners = {}
        for family in selected:
            names, fp16_name, shader, _ = FAMILIES[family]
            if fp16_name not in original_blocks:
                raise SystemExit(f'{label}: FP16 fixture has no {fp16_name} stage')
            expected_shader = next(w[1] for w in
                                   (line.split() for line in original_blocks[fp16_name])
                                   if w and w[0] == 'shader')
            if Path(expected_shader).name != shader:
                raise SystemExit(f'{label}.{family}: unexpected FP16 shader {expected_shader}')
            for name in names:
                owners[name] = family
        emitted, seen = [], set()
        for item in items:
            if item[0] == 'stage' and item[1] in owners:
                family = owners[item[1]]
                if family not in seen:
                    seen.add(family)
                    _, fp16_name, _, _ = FAMILIES[family]
                    emitted.append(('stage', fp16_name, original_blocks[fp16_name]))
                continue
            emitted.append(item)
        missing = selected - seen
        if missing:
            raise SystemExit(f'{label}: INT8 stages not found for {sorted(missing)}')
        new_lines = flatten(emitted)
        (out / re.sub(r'[^A-Za-z0-9]+', '-', label)).mkdir()
        target = out / re.sub(r'[^A-Za-z0-9]+', '-', label)

        record = dict(record)
        record['int8_batch'] = 'selective'
        record['int8_restored'] = sorted(f'{label}:{f}' for f in selected)
        record['int8_selective'] = True
        record['int8_operations'] = [op for op in record.get('int8_operations', [])
                                     if OPERATOR_FAMILY.get(op['name']) not in selected]
        restore_layouts = set()
        for family in selected:
            restore_layouts |= FAMILY_LAYOUTS.get(family, set())
        record['int8_layouts'] = [x for x in record.get('int8_layouts', [])
                                  if x['kind'] not in restore_layouts]
        if 'attention' in selected:
            record.pop('int8_attention', None)
        if asym:
            attention_int8 = bool(record.get('int8_attention'))
            new_lines, sizes = asym_rewrite(new_lines, target, source, label, record,
                                            attention_int8)
            record['int8_asym'] = {
                'mlp-down': ('swin_q8_quant_asym.cso', 'swin_mlp_down_q8_asym.cso',
                             'i8_mlp-down_off', 'i8_mlp-down_wsum')
                if any(op['name'] == 'mlp-down' for op in record['int8_operations']) else None,
                'attention-p': ('swin_q8_quant_asym.cso', 'swin_attention_pv_asym.cso',
                                'att_po') if attention_int8 else None,
                'resources': sizes,
            }
        (target / 'graph.txt').write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
        (target / 'fixture.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        manifest['layers'][label] = {'restored': sorted(f'{label}:{f}' for f in selected),
                                    'int8_operations': [op['name'] for op in
                                                        record['int8_operations']],
                                    'int8_attention': bool(record.get('int8_attention')),
                                    'int8_asym': bool(record.get('int8_asym')) if asym else False}
    (out / 'selective.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def asym_rewrite(lines, target, source, label, record, attention_int8):
    """Rewrite the two non-negative-operand matmuls to the asymmetric shaders.

    Returns the new line list and the resource sizes; the caller updates the
    fixture record.  Only stages that are still INT8 are touched.
    """
    items = split_graph(lines)
    ops = {op['name']: op for op in record.get('int8_operations', [])}
    added = {}
    sizes = {}

    def resource(name, nbytes, init=None):
        if name in sizes:
            return
        sizes[name] = nbytes
        added[name] = f'resource {name} {nbytes}' + (f' {init}' if init else '')

    resources = []
    if 'mlp-down' in ops:
        op = ops['mlp-down']
        M, K, N = op['M'], op['K'], op['N']
        resource('i8_mlp-down_off', M * 4)
        codes = np.fromfile(Path(source) / (op['weight_codes'] + '.bin'), np.uint8)
        if codes.size != K * N:
            raise SystemExit(f'{label}: weight code file has {codes.size} bytes, '
                             f'expected {K * N}')
        # stored column-major (wc.T): one contiguous row per output column
        signed = codes.astype(np.int16)
        signed = np.where(signed > 127, signed - 256, signed)
        wsum = signed.reshape(N, K).astype(np.int64).sum(axis=1).astype(np.int32)
        (Path(target) / 'i8_mlp-down_wsum.bin').write_bytes(wsum.astype('<i4').tobytes())
        resource('i8_mlp-down_wsum', N * 4, 'i8_mlp-down_wsum.bin')
    if attention_int8:
        B = int(record['int8_attention']['instances'])
        resource('att_po', B * 16 * 4)

    for index, item in enumerate(list(items)):
        if item[0] == 'stage':
            name, body = item[1], list(item[2])
            if 'mlp-down' in ops:
                if name == 'mlp-down-quant':
                    body = [line for line in body
                            if not line.startswith(('shader ', 'uav 2 '))]
                    body.insert(0, 'shader swin_q8_quant_asym.cso')
                    body.append('uav 2 i8_mlp-down_off')
                elif name == 'mlp-down':
                    body = [line for line in body
                            if not line.startswith(('shader ', 'srv 5 ', 'srv 6 '))]
                    body.insert(0, 'shader swin_mlp_down_q8_asym.cso')
                    body.append('srv 5 i8_mlp-down_wsum')
                    body.append('srv 6 i8_mlp-down_off')
            if attention_int8:
                if name == 'attention-p-quant':
                    body = [line for line in body
                            if not line.startswith(('shader ', 'uav 2 '))]
                    body.insert(0, 'shader swin_q8_quant_asym.cso')
                    body.append('uav 2 att_po')
                elif name == 'attention-pv':
                    body = [line for line in body
                            if not line.startswith(('shader ', 'srv 4 '))]
                    body.insert(0, 'shader swin_attention_pv_asym.cso')
                    body.append('srv 4 att_po')
            items[index] = ('stage', name, body)
    # resources must precede the first stage
    first = next(i for i, item in enumerate(items) if item[0] == 'stage')
    for line in added.values():
        items.insert(first, ('raw', line))
        first += 1
    # diagnostics for the added tensors (only exported by cumulative dumps)
    tail = []
    if 'mlp-down' in ops:
        op = ops['mlp-down']
        tail.append(f'dump i8_mlp-down_off 0 {op["M"] * 4} dbg-i8_mlp-down_off.bin')
        tail.append(f'dump i8_mlp-down_wsum 0 {op["N"] * 4} dbg-i8_mlp-down_wsum.bin')
    if attention_int8:
        B = int(record['int8_attention']['instances'])
        tail.append(f'dump att_po 0 {B * 16 * 4} dbg-att-po.bin')
    return flatten(items) + tail, sizes


def audit(layers, manifest=None):
    """Verify the variant is exactly the declared INT8/FP16 split."""
    layers = Path(layers)
    if manifest is None:
        manifest = json.loads((layers / 'selective.json').read_text())
    restore = {tuple(x.split(':')) for x in manifest['restored']}
    problems = []
    for label in LABELS:
        record = json.loads((Path(suite.fixture(str(layers), label)) /
                             'fixture.json').read_text())
        if record.get('int8_batch') != 'selective':
            problems.append(f'{label}: batch is {record.get("int8_batch")}')
        selected = {f for l, f in restore if l == label}
        if set(record.get('int8_restored', [])) != {f'{label}:{f}' for f in selected}:
            problems.append(f'{label}: restored manifest disagrees')
        blocks = stage_blocks(read_graph(Path(suite.fixture(str(layers), label)) / 'graph.txt'))
        names = [n for n, _ in blocks]
        for family in selected:
            int8_names, fp16_name, shader, _ = FAMILIES[family]
            if fp16_name not in names:
                problems.append(f'{label}.{family}: FP16 stage missing')
            # q/v/wo/merge/expand reuse the operator's own stage name, so only
            # the auxiliary INT8 stages (quant/unpack/gather/...) distinguish it;
            # the FP16 shader check below verifies the reused name.
            auxiliary = [n for n in int8_names if n != fp16_name]
            if any(n in names for n in auxiliary):
                problems.append(f'{label}.{family}: INT8 stage survived restoration')
            body = dict(blocks).get(fp16_name, [])
            use = next((w[1] for w in (line.split() for line in body)
                        if w and w[0] == 'shader'), '')
            if Path(use).name != shader:
                problems.append(f'{label}.{family}: FP16 shader is {use}')
        expected_ops = set()
        for family, stages in OPERATOR_STAGES.items():
            if family not in selected:
                expected_ops |= stages & set(names)
        actual_ops = {op['name'] for op in record.get('int8_operations', [])}
        if actual_ops != expected_ops:
            problems.append(f'{label}: int8_operations {sorted(actual_ops)} '
                            f'!= {sorted(expected_ops)}')
        asym = record.get('int8_asym') or {}
        for op in record.get('int8_operations', []):
            if op['name'] not in OPERATOR_SHADER:
                problems.append(f'{label}.{op["name"]}: unknown operator')
                continue
            body = dict(blocks).get(op['name'], [])
            use = next((w[1] for w in (line.split() for line in body)
                        if w and w[0] == 'shader'), '')
            expected = OPERATOR_SHADER[op['name']]
            if op['name'] == 'mlp-down' and asym.get('mlp-down'):
                expected = 'swin_mlp_down_q8_asym.cso'
            if Path(use).name != expected:
                problems.append(f'{label}.{op["name"]}: INT8 shader is {use}')
        if asym:
            resources = {line.split()[1]: int(line.split()[2])
                         for line in read_graph(Path(suite.fixture(str(layers), label)) /
                                                'graph.txt')
                         if line.split()[:1] == ['resource']}
            if asym.get('mlp-down'):
                op = next(o for o in record['int8_operations'] if o['name'] == 'mlp-down')
                for name, size in (('i8_mlp-down_off', op['M'] * 4),
                                   ('i8_mlp-down_wsum', op['N'] * 4)):
                    if resources.get(name) != size:
                        problems.append(f'{label}: {name} is {resources.get(name)}, '
                                        f'expected {size}')
                body = dict(blocks).get('mlp-down-quant', [])
                use = next((w[1] for w in (line.split() for line in body)
                            if w and w[0] == 'shader'), '')
                if Path(use).name != 'swin_q8_quant_asym.cso':
                    problems.append(f'{label}.mlp-down-quant: shader is {use}')
            if asym.get('attention-p'):
                body = dict(blocks).get('attention-pv', [])
                use = next((w[1] for w in (line.split() for line in body)
                            if w and w[0] == 'shader'), '')
                if Path(use).name != 'swin_attention_pv_asym.cso':
                    problems.append(f'{label}.attention-pv: shader is {use}')
                body = dict(blocks).get('attention-p-quant', [])
                use = next((w[1] for w in (line.split() for line in body)
                            if w and w[0] == 'shader'), '')
                if Path(use).name != 'swin_q8_quant_asym.cso':
                    problems.append(f'{label}.attention-p-quant: shader is {use}')
        if bool(record.get('int8_attention')) == ('attention' in selected):
            problems.append(f'{label}: attention contract disagrees with restoration')
    return {'status': 'PASS' if not problems else 'FAIL', 'problems': problems}


def coverage(int8_layers, fp16_layers, restore, enc0_int8=True, dec0_int8=True):
    """Recompute operator/MAC coverage for a mixed configuration."""
    ledger = coverage_ledger(fp16_layers)
    restore = set(restore)
    rows = []
    for row in ledger:
        family = INVENTORY_FAMILY.get(row['family'], row['family'])
        layer = row['layer']
        if row['family'] == 'enc0':
            int8 = bool(enc0_int8)
        elif row['family'] == 'dec0':
            int8 = bool(dec0_int8)
        else:
            int8 = (layer, family) not in restore
        rows.append({**row, 'precision': 'int8' if int8 else 'fp16', 'int8': int8})
    int8_rows = [r for r in rows if r['int8']]
    redesigned = [r for r in rows if not r['int8']]
    return {
        'denominator_operators': DENOMINATOR_OPERATORS,
        'denominator_macs': DENOMINATOR_MACS,
        'int8_operators': len(int8_rows),
        'int8_macs': sum(r['macs'] for r in int8_rows),
        'operation_coverage': len(int8_rows) / DENOMINATOR_OPERATORS,
        'mac_coverage': sum(r['macs'] for r in int8_rows) / DENOMINATOR_MACS,
        'fp16_operators': len(redesigned),
        'fp16_macs': sum(r['macs'] for r in redesigned),
        'fp16_matrix_operations': [r['id'] for r in redesigned],
        'restored': sorted(f'{l}:{f}' for l, f in restore),
    }


def coverage_ledger(fp16_layers):
    """The established 76-row inventory, read from the canonical FP16 fixtures."""
    import m09_cumulative as cumulative
    inventory = cumulative.inventory(Path(fp16_layers))
    return inventory['operators']


def parse_restore(text):
    out = set()
    for item in re.split(r'[,|]', text or ''):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise SystemExit(f'restore target {item!r} must be label:family')
        label, family = item.split(':', 1)
        if label not in LABELS:
            raise SystemExit(f'unknown layer {label}')
        if family not in FAMILIES:
            raise SystemExit(f'unknown family {family}')
        out.add((label, family))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--int8', type=Path, required=True)
    ap.add_argument('--fp16', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--restore', default='')
    ap.add_argument('--restore-family', default='')
    ap.add_argument('--asym', action='store_true',
                    help='asymmetric activation encoding for mlp-down and attention P')
    a = ap.parse_args()
    restore = parse_restore(a.restore)
    if a.restore_family:
        for label in LABELS:
            for family in a.restore_family.split(','):
                family = family.strip()
                if family and family in available(a.int8, label):
                    restore.add((label, family))
    manifest = build(a.int8, a.fp16, a.out, restore, asym=a.asym)
    checked = audit(a.out, manifest)
    print(json.dumps({'restored': manifest['restored'], 'audit': checked}, indent=2))
    return 0 if checked['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
