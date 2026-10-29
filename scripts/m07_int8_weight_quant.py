#!/usr/bin/env python3
"""M07: INT8 quantization feasibility for Preset M's real weights and activations.

Weights come from the captured layer launches through d4r's `plain_layer()`,
which decodes NGX's blocked e4m3 storage back to exact f16 matrices (every
e4m3 value is an f16 value, so this step is lossless and is not part of the
error being measured).

For every matrix it compares symmetric INT8 conversion at four scaling
granularities against the exact f16 tensor, and then propagates the best and
worst schemes through a real activation matrix taken from the same capture.

Metrics are relative Frobenius error (||W - Wq|| / ||W||), the worst single
element, and the outlier cost: how much the largest channel's scale inflates
every other channel in that scope.

usage: m07_int8_weight_quant.py CAPTURE_DIR [--json OUT]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'engine'))

import m_model as mm          # noqa: E402
import model_enc3 as me       # noqa: E402

F16, F64 = np.float16, np.float64
QMAX = 127.0


def quantise_per_tensor(w):
    scale = np.abs(w).max() / QMAX
    if scale == 0:
        return w.copy(), 0.0
    return np.round(w / scale) * scale, float(scale)


def quantise_axis(w, block):
    """Symmetric INT8 with one scale per `block` rows (block=None: per output column)."""
    out = np.empty_like(w)
    if block is None:                       # one scale per output channel (column)
        for n in range(w.shape[1]):
            scale = np.abs(w[:, n]).max() / QMAX
            out[:, n] = np.round(w[:, n] / scale) * scale if scale else w[:, n]
        return out
    for start in range(0, w.shape[0], block):
        tile = w[start:start + block]
        scale = np.abs(tile).max() / QMAX
        out[start:start + block] = np.round(tile / scale) * scale if scale else tile
    return out


def rel_error(w, q):
    denominator = np.linalg.norm(w)
    return float(np.linalg.norm(w - q) / denominator) if denominator else 0.0


def outlier_cost(w, axis):
    """Ratio of the largest to the median per-slice magnitude (scale inflation)."""
    per_slice = np.abs(w).max(axis=axis)
    return float(per_slice.max() / max(np.median(per_slice), 1e-30))


def collect(capture):
    dirs = sorted(glob.glob(os.path.join(capture, 'replay-*-rrlite_*_4x4')))
    layers = []
    for directory in dirs:
        kernel = open(os.path.join(directory, 'manifest.txt')).readline().split()[1]
        name = kernel[len('rrlite_'):-len('_4x4')]
        if name in mm.LAYERS:
            layers.append((mm.ORDER.index(name), name, directory))
    if len(layers) < 10:
        raise SystemExit(f'{capture}: found {len(layers)} layer launches, expected at least 10')
    layers.sort()
    result = []
    for _, name, directory in layers[:10]:
        P = me.Params(directory)
        t = mm.plain_layer(P, name)
        C, heads, NPM, CIN = mm.LAYERS[name]
        matrices = {f'q{h}': m for h, m in enumerate(t['q'])}
        matrices.update({f'v{h}': m for h, m in enumerate(t['v'])})
        matrices['wo'] = t['wo']
        matrices.update({f'w1_{i}': m for i, m in enumerate(t['w1'])})
        matrices.update({f'w2_{i}': m for i, m in enumerate(t['w2'])})
        if NPM:
            matrices['merge_w'] = t['merge_w']
        if CIN:
            matrices['expand_w'] = t['expand_w']
        if CIN:
            x = mm.tokens(P.arena, P.inp - P.abase, P.W // 2, P.H // 2, CIN)
        else:
            x = mm.tokens(P.arena, P.inp - P.abase, P.W, P.H, C)
        result.append({'layer': name, 'matrices': matrices, 'activation': x})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    layers = collect(a.capture)
    schemes = ('per_tensor', 'per_channel', 'block16', 'block32')
    totals = {s: [0.0, 0.0, 0.0] for s in schemes}   # sum(err^2*elems), elems, worst

    print('weights (relative Frobenius error of symmetric INT8 vs exact f16)')
    print(f'{"layer":10s} {"matrix":9s} {"shape":11s} {"|w|max":>8s} '
          f'{"tensor":>8s} {"channel":>8s} {"blk16":>8s} {"blk32":>8s} {"chanInfl":>8s}')
    for entry in layers:
        for name, w in entry['matrices'].items():
            w = np.asarray(w, F64)
            values = {'per_tensor': rel_error(w, quantise_per_tensor(w)[0])}
            values['per_channel'] = rel_error(w, quantise_axis(w, None))
            values['block16'] = rel_error(w, quantise_axis(w, 16))
            values['block32'] = rel_error(w, quantise_axis(w, 32))
            for scheme in schemes:
                totals[scheme][0] += values[scheme] ** 2 * w.size
                totals[scheme][1] += w.size
                totals[scheme][2] = max(totals[scheme][2], values[scheme])
            print(f'{entry["layer"]:10s} {name:9s} {str(w.shape):11s} {np.abs(w).max():8.3f} '
                  f'{values["per_tensor"]:8.4f} {values["per_channel"]:8.4f} '
                  f'{values["block16"]:8.4f} {values["block32"]:8.4f} {outlier_cost(w, 0):8.2f}')

    print('\naggregate over all captured weight elements')
    summary = {}
    for scheme in schemes:
        squared, count, worst = totals[scheme]
        rms = float(np.sqrt(squared / count))
        summary[scheme] = {'rms_rel_error': rms, 'worst_rel_error': worst}
        print(f'  {scheme:12s} rms {rms:6.4f}  worst matrix {worst:6.4f}')

    print('\nactivation ranges (as the reference feeds them, e4m3-rounded)')
    activations = {}
    for entry in layers:
        x = np.asarray(entry['activation'], F64)
        span = {'layer': entry['layer'], 'absmax': float(np.abs(x).max()),
                'rms': float(np.sqrt((x ** 2).mean())),
                'absmax_over_rms': float(np.abs(x).max() / max(np.sqrt((x ** 2).mean()), 1e-30))}
        activations.setdefault(entry['layer'], span)
        print(f'  {entry["layer"]:10s} shape {str(x.shape):14s} absmax {span["absmax"]:7.3f} '
              f'rms {span["rms"]:7.4f} peak/rms {span["absmax_over_rms"]:6.2f}')

    print('\nGEMM propagation on enc1 q (x: real captured tokens, exact float64 accumulation)')
    enc1 = layers[0]
    x = np.asarray(enc1['activation'], F64).reshape(-1, mm.LAYERS['enc1'][0])
    w = np.asarray(enc1['matrices']['q0'], F64)
    reference = x @ w
    propagation = {}
    for scheme in schemes:
        if scheme == 'per_tensor':
            wq = quantise_per_tensor(w)[0]
        elif scheme == 'per_channel':
            wq = quantise_axis(w, None)
        else:
            wq = quantise_axis(w, int(scheme[5:]))
        produced = x @ wq
        rel = float(np.linalg.norm(produced - reference) / np.linalg.norm(reference))
        propagation[scheme] = rel
        print(f'  {scheme:12s} output relative error {rel:6.4f}')

    if a.json:
        json.dump({'weights': summary, 'activations': activations,
                   'gemm_enc1_q0': propagation}, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
