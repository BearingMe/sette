#!/usr/bin/env python3
"""M11-C: controlled, diagnostic-only precision experiments on real tensors.

One change at a time, identical inputs, the accepted arithmetic as the control.
Nothing here is implemented on the GPU; a variant that is not the accepted
contract is labelled `diagnostic_only` and cannot be promoted to the INT8 path.

Inputs are the real enc1 Q/V operands exported by the M09 attention contract and
the real enc1 Q weight/bias.  The Q-weighted product below uses Q as the
activation matrix for the Q weight; that *pairing* is synthetic (the real Q
input is the prepared token tensor), so conclusions are about the scheme's
relative error, not about network quality.

  m11_precision.py --out DIR [--attention DIR] [--layers DIR]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m09_enc1_q as eq
import m_weights as mw

F16, F32, F64 = np.float16, np.float32, np.float64
QMAX = 127


def rel(a, b):
    a = np.asarray(a, F64).ravel()
    b = np.asarray(b, F64).ravel()
    d = b - a
    rms = float(np.sqrt(np.mean(d ** 2)))
    ref = float(np.sqrt(np.mean(a ** 2)))
    return {'rms': rms, 'relative_rms': rms / ref if ref else None,
            'mean_abs': float(np.mean(np.abs(d))), 'max_abs': float(np.max(np.abs(d)))}


def per_tensor_rows(x):
    """Symmetric per-tensor activation encoding with the accepted row arithmetic."""
    x32 = np.asarray(x, F16).astype(F32)
    top = np.float32(np.abs(x32).max())
    if top == 0:
        top = np.float32(1.0)
    inv = np.float32(QMAX) / top
    codes = np.clip(np.rint((x32 * inv).astype(F32)), -QMAX, QMAX).astype(np.int8)
    return codes, np.float32(top * np.float32(1.0 / QMAX))


def per_tensor_weights(w):
    w64 = np.asarray(w, F64)
    top = float(np.abs(w64).max()) or 1.0
    scale = np.float32(top / QMAX)
    codes = np.clip(np.rint(w64 / float(scale)), -QMAX, QMAX).astype(np.int8)
    return codes, scale


def clipped_rows(x, percentile):
    """Per-row scale after clipping |x| at a percentile of the row maxima."""
    x32 = np.asarray(x, F16).astype(F32)
    limit = np.percentile(np.abs(x32), percentile, axis=1, keepdims=True).astype(F32)
    limit = np.where(limit > 0, limit, np.float32(1.0))
    clipped = np.clip(x32, -limit, limit)
    return eq.quantize_rows_float(clipped)


def gemm(x, w, bias, act, weight):
    if act == 'row':
        ac, ascale = eq.quantize_rows_float(x)
    else:
        ac, ascale = per_tensor_rows(x)
    if weight == 'column':
        wc, wscale, _ = eq.quantize_weights(w)
    else:
        wc, wscale = per_tensor_weights(w)
    y, acc = eq.gemm_int8(ac, ascale, wc, np.atleast_1d(wscale))
    if bias is not None:
        y = (y.astype(F32) + np.asarray(bias, F32)[None, :]).astype(F16)
    return y


def qk(windows, scheme):
    """One 4x4 window/head: QK^T with per-row or per-tensor Q quantisation."""
    q = windows.astype(F16)
    exact = (q.astype(F64) @ q.astype(F64).T).astype(F16)
    if scheme == 'row':
        codes, scale = eq.quantize_rows_float(q)
        acc = codes.astype(np.int64) @ codes.astype(np.int64).T
        got = ((acc.astype(F32) * scale) * scale.T).astype(F16)
    else:
        codes, scale = per_tensor_rows(q)
        acc = codes.astype(np.int64) @ codes.astype(np.int64).T
        got = (acc.astype(F32) * np.float32(scale) * np.float32(scale)).astype(F16)
    return got, exact


def pv(windows, scheme):
    """One 4x4 window/head: P·V with V scales local to the window or global."""
    v = windows.astype(F16)
    p = np.full((16, 16), 1.0 / 16.0, F16)
    exact = (p.astype(F64) @ v.astype(F64)).astype(F16)
    if scheme == 'window':
        cols, scale = eq.quantize_rows_float(np.ascontiguousarray(v.T))
        acc = p.astype(F64) @ cols.astype(np.int64).T  # exact P, V quantised only
        got = (acc.astype(F32) * scale.reshape(1, -1)).astype(F16)
    else:
        codes, scale = per_tensor_rows(np.ascontiguousarray(v.T))
        acc = p.astype(F64) @ codes.astype(np.int64).T
        got = (acc.astype(F32) * np.float32(scale)).astype(F16)
    return got, exact


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--attention', type=Path,
                    default=Path.home() / 'AppData/Local/Temp/opencode/m09h-attention-cpu-20261030')
    ap.add_argument('--layers', type=Path,
                    default=Path.home() / 'AppData/Local/Temp/opencode/m09h-complete-chain-20261030/layers')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    q = np.fromfile(a.attention / 'results' / 'enc1_q.bin', F16)
    v = np.fromfile(a.attention / 'results' / 'enc1_v.bin', F16)
    if q.size != 136640 * 64 or v.size != 136640 * 64:
        raise SystemExit(f'unexpected Q/V sizes {q.size}/{v.size}')
    q = q.reshape(136640, 64)
    v = v.reshape(136640, 64)
    weight = np.fromfile(a.layers / 'enc1' / 'i8_q_oracle_w.bin', F32).reshape(64, 64)
    bias = np.fromfile(a.layers / 'enc1' / 'i8_q_bias.bin', F32)

    exact_q = (q.astype(F64) @ weight.astype(F64) + bias.astype(F64)[None, :]).astype(F16)
    report = {'inputs': {'attention': str(a.attention), 'layers': str(a.layers),
                         'q_shape': list(q.shape), 'v_shape': list(v.shape),
                         'weight_shape': list(weight.shape)},
              'diagnostic_only': True,
              'accepted_contract': 'per-row activation, per-output-column weight, '
                                   'symmetric signed INT8, exact INT32, f32 rescale',
              'gemm': {}, 'attention': {}}

    variants = {
        'accepted: row act / column weight': ('row', 'column'),
        'act per-tensor / column weight': ('tensor', 'column'),
        'row act / weight per-tensor': ('row', 'tensor'),
        'act and weight per-tensor': ('tensor', 'tensor'),
    }
    for name, (act, weight_scope) in variants.items():
        got = gemm(q, weight, bias, act, weight_scope)
        report['gemm'][name] = {**rel(exact_q, got), 'activation': act,
                                'weight': weight_scope, 'diagnostic_only': act != 'row' or weight_scope != 'column'}

    for percentile in (99.99, 99.9, 99.5):
        codes, scale = clipped_rows(q, percentile)
        wc, ws, _ = eq.quantize_weights(weight)
        y, _ = eq.gemm_int8(codes, scale, wc, ws)
        y = (y.astype(F32) + bias.astype(F32)[None, :]).astype(F16)
        report['gemm'][f'accepted + activation clip p{percentile}'] = {
            **rel(exact_q, y), 'diagnostic_only': True}

    # attention: 4x4 token windows inside a head, 32 channels
    wins = 4096
    idx = np.arange(136640).reshape(-1, 16)[:wins]
    for label, tensor in (('QK', q), ('PV', v)):
        for scheme in ('row', 'tensor') if label == 'QK' else ('window', 'global'):
            rows = []
            for head in range(2):
                block = tensor[:, head * 32:(head + 1) * 32]
                for w in idx:
                    got, exact = qk(block[w], scheme) if label == 'QK' else pv(block[w], scheme)
                    rows.append(rel(exact, got))
            report['attention'][f'{label} {scheme}'] = {
                'windows': len(rows), 'per_window': rows[0],
                'relative_rms_mean': float(np.mean([r['relative_rms'] for r in rows])),
                'relative_rms_max': float(np.max([r['relative_rms'] for r in rows])),
                'diagnostic_only': scheme not in ('row', 'window')}

    (a.out / 'precision.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (round(v['relative_rms'], 6) if isinstance(v, dict)
                          and 'relative_rms' in v else v)
                      for k, v in report['gemm'].items()}, indent=2))
    for k, v in report['attention'].items():
        print(k, 'mean', round(v['relative_rms_mean'], 6), 'max', round(v['relative_rms_max'], 6))
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
