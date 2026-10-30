#!/usr/bin/env python3
"""M13 quantization probe: bounded scale-granularity alternatives on real data.

Two independent questions, both diagnostic CPU arithmetic (no GPU change):

1. Static weights.  Every learned-weight operator in the accepted INT8 graph is
   read from its `*_oracle_w.bin` (the exact f32 matrix the accepted encoder
   quantized) and re-encoded per column, per K block (32/64) and per tensor.
   This isolates weight representation error from activation error.

2. One real GEMM.  The captured enc1 Q operand and Q weight are used with the
   accepted arithmetic as control, then per-block activation and weight scales
   are added one at a time.  Block scales change the accumulation contract
   (scale before the sum), so they are measured as a scheme, not assumed to be
   equivalent.

Outputs relative RMS versus the exact product; nothing here is promoted to the
GPU path.

  m13_quant_probe.py --out DIR [--layers LAYERS] [--attention DIR]
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m09_enc1_q as eq
import m_layer_suite as suite
import m_weights as mw

F32, F64 = np.float32, np.float64
QMAX = 127


def rel(a, b):
    a = np.asarray(a, F64).ravel()
    b = np.asarray(b, F64).ravel()
    d = b - a
    rms = float(np.sqrt(np.mean(d ** 2)))
    ref = float(np.sqrt(np.mean(a ** 2)))
    return {'rms': rms, 'relative_rms': rms / ref if ref else None,
            'mean_abs': float(np.mean(np.abs(d))), 'max_abs': float(np.max(np.abs(d)))}


def quant_column(W):
    """Accepted weight encoding: one symmetric scale per output column."""
    return eq.quantize_weights(W)[:2]


def quant_block(W, block):
    """One symmetric scale per (output column, K block)."""
    K, N = W.shape
    if K % block:
        raise ValueError(f'K={K} not divisible by block {block}')
    codes = np.empty_like(W, dtype=np.int8)
    scales = np.empty((K // block, N), np.float32)
    for b in range(K // block):
        slab = np.asarray(W[b * block:(b + 1) * block], F64)
        top = np.abs(slab).max(axis=0)
        scale = np.where(top > 0, top / QMAX, 1.0 / QMAX).astype(F32)
        codes[b * block:(b + 1) * block] = np.clip(
            np.rint(slab / scale.astype(F64)), -QMAX, QMAX).astype(np.int8)
        scales[b] = scale
    return codes, scales


def dequant_block(codes, scales, block):
    out = np.empty(codes.shape, F64)
    for b in range(scales.shape[0]):
        out[b * block:(b + 1) * block] = (codes[b * block:(b + 1) * block].astype(F64) *
                                          scales[b].astype(F64))
    return out


def quant_tensor(W):
    top = float(np.abs(np.asarray(W, F64)).max()) or 1.0
    scale = np.float32(top / QMAX)
    codes = np.clip(np.rint(np.asarray(W, F64) / float(scale)), -QMAX, QMAX).astype(np.int8)
    return codes, np.float32(scale)


def gemm_block(A, W, bias, act_block=None, weight_block=None):
    """Accepted dot arithmetic with optional per-K-block scales.

    With no blocks this is exactly `eq.gemm_int8` (int64 accumulation, f32
    scale pair, f16 store).  With blocks it sums `int_dot * act_scale *
    weight_scale` per block in f64, then rounds once to f16 - the best case a
    block-scaled kernel could achieve.
    """
    if act_block is None and weight_block is None:
        ac, ascale = eq.quantize_rows_float(A)
        wc, wscale = quant_column(W)
        y, _ = eq.gemm_int8(ac, ascale, wc, wscale)
        if bias is not None:
            y = (y.astype(F32) + np.asarray(bias, F32)[None, :]).astype(np.float16)
        return y

    K = A.shape[1]
    block = act_block or weight_block
    if K % block:
        raise ValueError(f'K={K} not divisible by block {block}')
    if act_block is not None:
        ac = np.empty_like(A, dtype=np.int8)
        ascale = np.empty((A.shape[0], K // block), F32)
        for b in range(K // block):
            slab = np.asarray(A[:, b * block:(b + 1) * block], F64)
            top = np.abs(slab).max(axis=1)
            scale = np.where(top > 0, top / QMAX, 1.0 / QMAX).astype(F32)
            ac[:, b * block:(b + 1) * block] = np.clip(
                np.rint(slab / scale[:, None].astype(F64)), -QMAX, QMAX).astype(np.int8)
            ascale[:, b] = scale
    else:
        ac, row_scale = eq.quantize_rows_float(A)
        ascale = np.repeat(row_scale, K // block, axis=1)
    if weight_block is not None:
        wc, wscale = quant_block(W, weight_block)
    else:
        wc, column_scale = quant_column(W)
        wscale = np.repeat(column_scale[None, :], K // block, axis=0)
    out = np.zeros((A.shape[0], W.shape[1]), F64)
    for b in range(K // block):
        acc = (ac[:, b * block:(b + 1) * block].astype(np.int64) @
               wc[b * block:(b + 1) * block].astype(np.int64))
        out += acc.astype(F64) * ascale[:, b:b + 1].astype(F64) * wscale[b].astype(F64)[None, :]
    y = out.astype(F32)
    if bias is not None:
        y = (y + np.asarray(bias, F32)[None, :]).astype(np.float16)
    else:
        y = y.astype(np.float16)
    return y


def quant_row_asym(x, mode):
    """Per-row encoding in [-128,127] with an explicit offset.

    `unsigned` clips the lower bound to zero (valid for probabilities and
    approximately for GELU); `asym` uses the row's true minimum.
    """
    x = np.asarray(x, F64)
    hi = x.max(axis=1, keepdims=True)
    lo = np.zeros_like(hi) if mode == 'unsigned' else x.min(axis=1, keepdims=True)
    scale = ((hi - lo) / 255.0).astype(F32)
    scale = np.where(scale > 0, scale, np.float32(1.0 / 255.0)).astype(F32)
    code = np.clip(np.rint((x - lo) / scale.astype(F64)) - 128.0,
                   -128, 127).astype(np.int8)
    return code, scale, lo.astype(F32)


def activation_schemes(x):
    """Dequantized activation error for the accepted and asymmetric schemes."""
    x = np.asarray(x, F64)
    ac, ascale = eq.quantize_rows_float(np.asarray(x, np.float16))
    out = {'symmetric': rel(x, ac.astype(F64) * ascale.astype(F64))}
    for mode in ('unsigned', 'asym'):
        code, scale, lo = quant_row_asym(x, mode)
        deq = (code.astype(F64) + 128.0) * scale.astype(F64) + lo.astype(F64)
        out[mode] = rel(x, deq)
    return out


def gemm_schemes(A, W, mode):
    """Accepted int8 dot plus the asymmetric variants, one scheme at a time."""
    A16 = np.asarray(A, np.float16)
    wc, wscale = quant_column(W)
    if mode == 'symmetric':
        ac, ascale = eq.quantize_rows_float(A16)
        # A is already f16; use the exact integer reduction.
        acc = ac.astype(np.int64) @ wc.astype(np.int64)
        y = (acc.astype(F32) * ascale * wscale[None, :]).astype(F32)
    else:
        code, scale, lo = quant_row_asym(A16, mode)
        acc = code.astype(np.int64) @ wc.astype(np.int64)
        wsum = wc.astype(np.int64).sum(axis=0)
        # dequant(A) = (code + 128) * scale + lo, so the +128 term must be
        # included once per column before applying the weight scale.
        y = ((acc.astype(F64) + 128.0 * wsum.astype(F64)[None, :]) *
             scale.astype(F64) * wscale.astype(F64)[None, :] +
             lo.astype(F64) * wscale.astype(F64)[None, :] * wsum.astype(F64)[None, :])
        y = y.astype(F32)
    return y.astype(np.float16)


def pv_schemes(P, V, mode):
    """One 16x16 probability window times 16x32 values, P quantised only."""
    P64, V64 = np.asarray(P, F64), np.asarray(V, F64)
    exact = (P64 @ V64).astype(np.float16)
    if mode == 'symmetric':
        codes, scale = eq.quantize_rows_float(np.asarray(P, np.float16))
        deq = codes.astype(F64) * scale.astype(F64)
    else:
        code, scale, lo = quant_row_asym(P, mode)
        deq = (code.astype(F64) + 128.0) * scale.astype(F64) + lo.astype(F64)
    got = (deq @ V64).astype(np.float16)
    return exact, got


def real_tensor_probe(probe_dir, layers, sample_rows=8192, sample_windows=2048):
    """Post-GELU (MLP-down) and probability (PV) tensors from an executed chain."""
    probe_dir = Path(probe_dir)
    out = {'source': str(probe_dir), 'mlp_down': {}, 'pv': {}}
    for label, rows in (('enc1', 136640), ('dec1', 130560)):
        path = probe_dir / f'{label}-dbg-i8_g.bin'
        directory = Path(suite.fixture(str(layers), label))
        record = json.loads((directory / 'fixture.json').read_text())
        op = next((o for o in record.get('int8_operations', [])
                   if o['name'] == 'mlp-down'), None)
        if not path.is_file() or op is None:
            out['mlp_down'][label] = 'missing'
            continue
        G = np.fromfile(path, np.float16).reshape(rows, op['K'])
        W = np.fromfile(directory / op['W'], '<f4').reshape(op['K'], op['N'])
        sample = G[:sample_rows]
        exact = (sample.astype(F64) @ W.astype(F64)).astype(np.float16)
        entry = {'rows': rows, 'sampled_rows': int(sample.shape[0]),
                 'activation': activation_schemes(sample),
                 'output': {mode: rel(exact, gemm_schemes(sample, W, mode))
                            for mode in ('symmetric', 'unsigned', 'asym')}}
        out['mlp_down'][label] = entry
    for label, instances in (('enc1', 17080), ('dec1', 16320)):
        prob = probe_dir / f'{label}-dbg-att-prob.bin'
        value = probe_dir / f'{label}-dbg-att-av.bin'
        if not prob.is_file() or not value.is_file():
            out['pv'][label] = 'missing'
            continue
        P = np.fromfile(prob, np.float16).reshape(instances, 16, 16)[:sample_windows]
        V = np.fromfile(value, np.float16).reshape(instances, 16, 32)[:sample_windows]
        entry = {'instances': instances, 'sampled_windows': int(P.shape[0]),
                 'activation': activation_schemes(P.reshape(-1, 16)),
                 'output': {}}
        for mode in ('symmetric', 'unsigned', 'asym'):
            rows = []
            for window in range(P.shape[0]):
                exact, got = pv_schemes(P[window], V[window], mode)
                rows.append(rel(exact, got)['relative_rms'])
            entry['output'][mode] = {
                'relative_rms_mean': float(np.mean(rows)),
                'relative_rms_max': float(np.max(rows)),
                'windows': len(rows)}
        out['pv'][label] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layers', type=Path,
                    default=Path(os.environ.get('LOCALAPPDATA', '.')) /
                            'Temp/opencode/m09h-complete-chain-20261030/layers')
    ap.add_argument('--attention', type=Path,
                    default=Path(os.environ.get('LOCALAPPDATA', '.')) /
                            'Temp/opencode/m09h-attention-cpu-20261030')
    ap.add_argument('--diagnostics', type=Path,
                    default=Path(os.environ.get('LOCALAPPDATA', '.')) /
                            'Temp/opencode/m13-diagnostics-results')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    # ---- 1. static weights -------------------------------------------------
    weights = {}
    for label in suite.LABELS:
        directory = Path(suite.fixture(str(a.layers), label))
        record = json.loads((directory / 'fixture.json').read_text())
        for op in record.get('int8_operations', []):
            path = directory / op['W']
            W = np.fromfile(path, '<f4').reshape(op['K'], op['N'])
            weights[f'{label}.{op["name"]}'] = W
    if not weights:
        raise SystemExit(f'no oracle weights under {a.layers}')
    rows = {}
    for name, W in weights.items():
        column = dequant_block(*quant_column(W), block=W.shape[0])
        entry = {'shape': list(W.shape),
                 'per_column': rel(W, column)}
        for block in (32, 64):
            if W.shape[0] % block == 0:
                entry[f'block_{block}'] = rel(W, dequant_block(*quant_block(W, block), block))
        entry['per_tensor'] = rel(W, float(quant_tensor(W)[1]) *
                                  quant_tensor(W)[0].astype(F64))
        rows[name] = entry

    summary = {}
    for scheme in ('per_column', 'block_32', 'block_64', 'per_tensor'):
        values = [r[scheme]['relative_rms'] for r in rows.values() if scheme in r]
        summary[scheme] = {
            'operators': len(values),
            'relative_rms_median': float(np.median(values)),
            'relative_rms_max': float(np.max(values)),
            'relative_rms_min': float(np.min(values)),
        }

    # ---- 2. real GEMM ------------------------------------------------------
    q = np.fromfile(a.attention / 'results' / 'enc1_q.bin', np.float16)
    v = np.fromfile(a.attention / 'results' / 'enc1_v.bin', np.float16)
    if q.size != 136640 * 64:
        raise SystemExit(f'unexpected Q size {q.size}')
    q = q.reshape(136640, 64)
    v = v.reshape(136640, 64)
    W = np.fromfile(a.layers / 'enc1' / 'i8_q_oracle_w.bin', '<f4').reshape(64, 64)
    bias = np.fromfile(a.layers / 'enc1' / 'i8_q_bias.bin', '<f4')

    def exact(A):
        return (A.astype(F64) @ W.astype(F64) + bias.astype(F64)[None, :]).astype(np.float16)

    variants = {
        'accepted: row act / column weight': (q, W, None, None),
        'block32 activation / column weight': (q, W, 32, None),
        'row activation / block32 weight': (q, W, None, 32),
        'block32 activation / block32 weight': (q, W, 32, 32),
        'block64 activation / block64 weight': (q, W, 64, 64),
        'per-tensor activation / column weight': (None, W, 'tensor', None),
    }
    gemm = {}
    exact_q = exact(q)
    for name, (A, Wm, ab, wb) in variants.items():
        if A is None:
            # explicit per-tensor activation encoding
            top = float(np.abs(np.asarray(q, F64)).max()) or 1.0
            scale = np.float32(top / QMAX)
            codes = np.clip(np.rint(np.asarray(q, F64) / float(scale)), -QMAX, QMAX).astype(np.int8)
            wc, wscale = quant_column(Wm)
            acc = codes.astype(np.int64) @ wc.astype(np.int64)
            got = (acc.astype(F32) * scale * wscale[None, :]).astype(F32)
            got = (got + np.asarray(bias, F32)[None, :]).astype(np.float16)
        else:
            got = gemm_block(A, Wm, bias, ab, wb)
        gemm[name] = rel(exact_q, got)

    # separate activation-only and weight-only error, same product
    ac, ascale = eq.quantize_rows_float(q)
    wc, wscale = quant_column(W)
    act_only = (ac.astype(F64) * ascale.astype(F64)) @ W.astype(F64)
    weight_only = q.astype(F64) @ dequant_block(wc, wscale[None, :].repeat(64, 0), 64)
    exact_no_bias = q.astype(F64) @ W.astype(F64)
    report = {
        'weights': rows, 'weight_summary': summary,
        'gemm_inputs': {'attention': str(a.attention), 'layers': str(a.layers),
                        'rows': int(q.shape[0]), 'K': int(q.shape[1]),
                        'weight_shape': list(W.shape),
                        'pairing_note': 'real Q operand with the real Q weight; the '
                                        'real Q-projection input is the prepared token '
                                        'tensor, so this compares schemes only'},
        'gemm': gemm,
        'real_tensors': real_tensor_probe(a.diagnostics, a.layers),
        'separated': {
            'activation_only': rel(exact_no_bias, act_only),
            'weight_only': rel(exact_no_bias, weight_only),
            'accepted_total_no_bias': rel(exact_no_bias,
                                          (ac.astype(F64) * ascale.astype(F64)) @
                                          dequant_block(wc, wscale[None, :].repeat(64, 0), 64)),
        },
    }
    (a.out / 'quant-probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(json.dumps({k: round(v['relative_rms'], 6) for k, v in gemm.items()}, indent=2))
    print(json.dumps({k: round(v['relative_rms'], 6)
                      for k, v in report['separated'].items()}, indent=2))
    real = report['real_tensors']
    for group in ('mlp_down', 'pv'):
        for label, entry in real[group].items():
            if isinstance(entry, str):
                print(group, label, entry)
                continue
            out = entry['output']
            print(group, label, {k: round((v['relative_rms'] if 'relative_rms' in v
                                           else v['relative_rms_mean']), 6)
                                 for k, v in out.items()})
    print('out:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
