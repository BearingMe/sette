#!/usr/bin/env python3
"""M07 (C11-C15): INT8 feasibility over the *complete* Preset M weight inventory.

Every matrix the network owns is extracted from one captured frame - the front-end
embedding (`enc0`, 128x64), the ten Swin layers, and the expansion (`dec0`,
64x80) - and encoded to symmetric INT8 at five scaling scopes.  Weight-only
error, activation ranges, and the error an INT8 GEMM would actually produce on
the captured activations are reported separately, because they fail for
different reasons.

The arithmetic is `scripts/m_int8.py` (CPU only, integer accumulation, explicit
rescales).  Nothing here runs on a GPU and nothing here uses the reference model
to compute the error it reports.

usage: m07_int8_weight_quant.py CAPTURE_DIR [--json OUT] [--stride N]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import m_int8 as qi          # noqa: E402
import m_weights as mw       # noqa: E402

F64 = np.float64
SCOPES = ('per_tensor', 'per_output_channel', 'per_input_channel', 'block_k:16', 'block_k:32')


def split(scope):
    if scope.startswith('block_k:'):
        return 'block_k', int(scope.split(':')[1])
    return scope, None


def encode(W, scope):
    name, block = split(scope)
    return qi.quantize(W, name, block)


def weight_table(stages):
    """Element-weighted aggregate error per scope, over every captured matrix.

    Stages that appear more than once (the six tube blocks) keep their own row,
    keyed by the hook's launch sequence.
    """
    labels = {}
    counts = {}
    for stage in stages:
        counts[stage.stage] = counts.get(stage.stage, 0) + 1
    for stage in stages:
        labels[id(stage)] = (stage.stage if counts[stage.stage] == 1
                             else f'{stage.stage}#{stage.sequence}')
    totals = {s: [0.0, 0, 0.0] for s in SCOPES}     # sum(err^2), elements, worst
    per_stage = {}
    for stage, row in mw.matrices(stages):
        W = np.asarray(row['W'], F64)
        entry = per_stage.setdefault(labels[id(stage)], {s: [0.0, 0] for s in SCOPES})
        for scope in SCOPES:
            stats = qi.metrics(W, encode(W, scope))
            totals[scope][0] += stats['rel_frobenius'] ** 2 * W.size
            totals[scope][1] += W.size
            totals[scope][2] = max(totals[scope][2], stats['rel_frobenius'])
            entry[scope][0] += stats['rel_frobenius'] ** 2 * W.size
            entry[scope][1] += W.size
    summary = {s: {'rms_rel': float(np.sqrt(v[0] / v[1])), 'worst_matrix': v[2],
                   'elements': v[1]} for s, v in totals.items()}
    return summary, per_stage


def activation_table(acts):
    rows = {}
    for info in acts:
        x = info['tensor']
        if x is None:
            continue
        values = np.asarray(x, F64)
        rows[info['label']] = {'shape': list(values.shape),
                               'absmax': float(np.abs(values).max()),
                               'rms': float(np.sqrt((values ** 2).mean())),
                               'peak_over_rms': float(np.abs(values).max() / max(np.sqrt((values ** 2).mean()), 1e-30)),
                               'source': info['source']}
    return rows


def propagate(x, W, stride=1):
    """INT8 GEMM error on a real captured operand, per weight scope.

    Error is measured on the matrix product alone.  The stage's f16 bias is an
    additive constant that no quantization scope touches, so folding it into the
    comparison would only rescale the denominator.
    """
    x = np.asarray(x, F64).reshape(-1, np.asarray(W).shape[0])[::stride]
    W = np.asarray(W, F64)
    reference = qi.gemm_exact(x, W)
    norm = np.linalg.norm(reference)
    rows = {}
    aq = qi.quantize_rows(x, 'per_row')
    activation_only = qi.gemm_exact(qi.dequantize_rows(aq), W)
    for scope in SCOPES:
        name, block = split(scope)
        wq = qi.quantize(W, name, block)
        y = qi.gemm_integer(aq['codes'], wq['codes'], aq['scales'], wq['scales'], name, block)
        rows[scope] = {
            'weight_only_rel': float(np.linalg.norm(qi.gemm_exact(x, qi.dequantize(wq)) - reference) / norm),
            'activation_only_rel': float(np.linalg.norm(activation_only - reference) / norm),
            'weights_and_activation_rel': float(np.linalg.norm(y - reference) / norm),
            'weights_and_activation_max_abs': float(np.abs(y - reference).max()),
        }
    return {'rows': int(x.shape[0]), 'cols': int(W.shape[1]), 'k': int(W.shape[0]),
            'activation_at_endpoint': float(np.mean(np.abs(qi.quantize_rows(x, 'per_row')['codes']) == 127)),
            'scopes': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    ap.add_argument('--stride', type=int, default=16,
                    help='row stride for the large dec0 operand (default 16)')
    a = ap.parse_args()

    stages = mw.inventory(a.capture)
    problems = mw.self_check(stages)
    if problems:
        for problem in problems:
            print(f'FAIL {problem}')
        return 1

    # ---- C11: inventory ---------------------------------------------------
    print('C11 inventory (one captured frame)')
    total = 0
    for stage in stages:
        count = sum(int(r['W'].size) for r in stage.matrices)
        total += count
        variant = (f"{stage.variant['motion']}/{stage.variant['range']}"
                   if stage.variant['motion'] else '')
        print(f'  {stage.stage:10s} {stage.kernel:36s} {variant:10s} '
              f'{"matrices=" + str(len(stage.matrices)):12s} codes={count:7d} '
              f'blob={stage.blob_bytes:7d}B')
    print(f'  total: {sum(len(s.matrices) for s in stages)} matrices, {total} e4m3 values, '
          f'decode/layout checks OK')

    front = {}
    for stage, row in mw.matrices(stages):
        if stage.stage not in ('enc0', 'dec0'):
            continue
        W = np.asarray(row['W'], F64)
        front[row['name']] = {
            'shape': row['shape'], 'absmax': float(np.abs(W).max()),
            'rms': float(np.sqrt((W ** 2).mean())),
            'zeros': int((W == 0).sum()), 'bias': row['bias'].astype(F64).tolist(),
            'layout': row['layout'],
        }
        print(f'  {row["name"]:18s} {str(row["shape"]):10s} absmax {np.abs(W).max():.4f} '
              f'rms {np.sqrt((W ** 2).mean()):.4f} zeros {int((W == 0).sum()):5d} '
              f'bias[{len(row["bias"])}] first {float(row["bias"][0]):+.4f}')

    # ---- C12/C13: encoding and scope comparison ---------------------------
    summary, per_stage = weight_table(stages)
    print('\nC12/C13 weight encoding: element-weighted RMS relative error')
    print(f'  {"scope":22s} {"rms":>8s} {"worst":>8s} {"elements":>10s}')
    for scope in SCOPES:
        row = summary[scope]
        print(f'  {scope:22s} {row["rms_rel"]:8.4f} {row["worst_matrix"]:8.4f} {row["elements"]:10d}')

    print('\n  per stage (rms relative error)')
    header = '  ' + f'{"stage":14s}' + ''.join(f'{s:>20s}' for s in SCOPES)
    print(header)
    for name, entry in per_stage.items():
        cells = ''.join(f'{np.sqrt(v[0] / v[1]):20.4f}' for v in entry.values())
        print(f'  {name:14s}{cells}')

    split_changed = split_total = 0
    for _, row in mw.matrices(stages):
        changed, size = qi.d4r_split_agreement(np.asarray(row['W'], F64))
        split_changed += changed
        split_total += size
    print(f'\nC12 determinism: repeat encode identical='
          f'{all(qi.determinism(np.asarray(r["W"], F64)) for _, r in mw.matrices(stages))}')
    print(f'  d4r float64-vs-float32 rounding divisor: {split_changed}/{split_total} codes differ '
          f'over the whole inventory')

    storage = {}
    print('\nC13 storage: INT8 data plus scales, against the FP8 bytes it replaces')
    for scope in SCOPES:
        bytes_data = bytes_scale = 0
        for _, row in mw.matrices(stages):
            stats = qi.metrics(np.asarray(row['W'], F64), encode(np.asarray(row['W'], F64), scope))
            bytes_data += stats['data_bytes']
            bytes_scale += stats['scale_bytes']
        storage[scope] = {'data': bytes_data, 'scales': bytes_scale,
                          'total': bytes_data + bytes_scale,
                          'over_fp8': float((bytes_data + bytes_scale) / bytes_data)}
        print(f'  {scope:22s} data {bytes_data:8d} B  scales {bytes_scale:7d} B  '
              f'total {bytes_data + bytes_scale:8d} B  x{storage[scope]["over_fp8"]:.4f} of FP8')
    print('  (the FP8 source is already one byte per value: INT8 cannot shrink it, only add scales)')

    # ---- C14: captured activations ---------------------------------------
    acts = mw.activations(stages)
    ranges = activation_table(acts)
    print('\nC14 captured activation operands')
    for name, row in ranges.items():
        print(f'  {name:14s} {str(row["shape"]):16s} absmax {row["absmax"]:8.3f} '
              f'rms {row["rms"]:7.4f} peak/rms {row["peak_over_rms"]:5.2f}  [{row["source"]}]')
    for info in acts:
        if info['tensor'] is None:
            print(f'  {info["label"]:14s} NO captured operand: {info["source"]}')
    enc0_note = 'enc0: its 128-feature operand is computed inside the kernel from texture samples'
    print(f'  {enc0_note}; the capture stores no such tensor, so no enc0 GEMM propagation is reported.')

    print('\nC14 INT8 GEMM on the captured operands (weights+activations, int32 accumulation)')
    cases = []
    by_name = {row['name']: (stage, row) for stage, row in mw.matrices(stages)}
    for stages_name, matrix in (('enc1', 'enc1.q0'), ('enc2', 'enc2.q0'),
                                ('enc3_tube', 'enc3_tube.q0'), ('dec2', 'dec2.expand_w'),
                                ('dec1', 'dec1.expand_w'), ('dec0', 'dec0.expansion')):
        if matrix not in by_name:
            continue
        stage, row = by_name[matrix]
        info = next((a for a in acts if a['stage'] == stage.stage and a['tensor'] is not None), None)
        if info is None:
            continue
        stride = a.stride if stage.stage == 'dec0' else 1
        result = propagate(info['tensor'], row['W'], stride)
        cases.append({'matrix': matrix, **result})
        print(f'  {matrix:16s} [{result["rows"]}x{result["k"]}] x [{result["k"]}x{result["cols"]}]'
              f' activation@endpoint {100 * result["activation_at_endpoint"]:.1f}%')
        for scope in SCOPES:
            r = result['scopes'][scope]
            print(f'    {scope:22s} weights {r["weight_only_rel"]:6.4f}  '
                  f'activation {r["activation_only_rel"]:6.4f}  '
                  f'both {r["weights_and_activation_rel"]:6.4f}  '
                  f'max|d| {r["weights_and_activation_max_abs"]:9.4f}')

    # d4r's own published INT8 convention on dec0: per-column weights, per-token
    # activation, one int32 accumulation, f32 rescale, bias added afterwards.
    dec0 = next((c for c in cases if c['matrix'] == 'dec0.expansion'), None)
    d4r_convention = None
    if dec0 is not None:
        row = by_name['dec0.expansion'][1]
        dec0_act = next(a for a in acts if a['stage'] == 'dec0')
        x = np.asarray(dec0_act['tensor'], F64).reshape(-1, 64)[::a.stride]
        W = np.asarray(row['W'], F64)
        reference = qi.gemm_exact(x, W)
        wq = qi.quantize(W, 'per_output_channel')
        aq = qi.quantize_rows(x, 'per_row')
        y = qi.gemm_integer(aq['codes'], wq['codes'], aq['scales'], wq['scales'],
                            'per_output_channel')
        d4r_convention = {'rel_frobenius': float(np.linalg.norm(y - reference) / np.linalg.norm(reference)),
                          'max_abs': float(np.abs(y - reference).max()),
                          'rows': int(x.shape[0])}
        print(f'\n  d4r dec0 INT8 convention (per-column weights, per-token activation, one int32 sum):'
              f' rel {d4r_convention["rel_frobenius"]:.4f}, max|d| {d4r_convention["max_abs"]:.4f}')

    # ---- C15 inventory reconciliation ------------------------------------
    stage_groups = {}
    for stage, row in mw.matrices(stages):
        stage_groups.setdefault(stage.stage, {'matrices': 0, 'codes': 0, 'kinds': set()})
        entry = stage_groups[stage.stage]
        entry['matrices'] += 1
        entry['codes'] += int(row['W'].size)
        entry['kinds'].add(row['name'].split('.')[1].rstrip('0123456789'))
    print('\nC15 matrix inventory by stage')
    for name, entry in stage_groups.items():
        kinds = ','.join(sorted(entry['kinds']))
        print(f'  {name:10s} {entry["matrices"]:3d} matrices {entry["codes"]:7d} codes  [{kinds}]')
    total_codes = sum(e['codes'] for e in stage_groups.values())
    fp8_bytes = total_codes
    per_channel = storage['per_output_channel']['total']
    print(f'  total {total_codes} codes = {fp8_bytes} B FP8; per-output-channel INT8 = '
          f'{per_channel} B ({per_channel / fp8_bytes:.3f}x)')
    print('  retained floating point: RMS norms, softmax, e4m3 requantisation, residuals, '
          'GELU, and the whole reconstruction (post/downsample own no matrices)')

    if a.json:
        json.dump({'capture': a.capture,
                   'inventory': mw.summary(stages),
                   'front_end': front,
                   'scopes': summary,
                   'per_stage': {k: {s: float(np.sqrt(v[0] / v[1])) for s, v in e.items()}
                                 for k, e in per_stage.items()},
                   'storage': storage,
                   'activations': ranges,
                   'propagation': cases,
                   'd4r_dec0_convention': d4r_convention,
                   'stride': a.stride}, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
