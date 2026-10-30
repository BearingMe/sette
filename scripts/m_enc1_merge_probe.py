#!/usr/bin/env python3
"""Isolate enc1's patch merge: identical operands, four arithmetic paths.

The merge's operands are the layer's own e4m3 output field: the store stage writes
`enc8(x2b)` in natural channel order and the merge re-quantizes the same `x2b` with
`q8`, so the stored codes are the merge's input codes. Every stored merged token
consumes only tokens that the field contains (the shift removes the padded ring), so
this probe needs no extra instrumentation and no proprietary dump.

For one capture it rebuilds the merge from that capture's own output codes and
weights with the declared kernel contract (exact 32-product sum, one f16 rounding per
k32, clamp to +-2*pi, e4m3 RNE) and compares the result with the capture's own merge
field. Agreement near 100% means the path is reproduced; disagreement localizes an
operand, ordering or epilogue error.

  m_enc1_merge_probe.py CAPTURE [CAPTURE ...] --json OUT
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

import m_layer_fixture as lf
import m_layer_oracle as lo
import m_model as mm

C, HEADS, NPM, CIN = mm.LAYERS['enc1']
K32 = 32
LIMIT = np.float32(6.2831855)


def _code_lut():
    """f16 bit pattern -> e4m3 code, from the format's own decode table."""
    lut = np.zeros(1 << 16, np.uint8)
    for code in range(256):
        value = lo.me.E4M3[code]
        if not np.isfinite(value):
            continue
        bits = int(np.float16(value).view(np.uint16))
        lut[bits] = code
    lut[np.float16(np.nan).view(np.uint16)] = 0x7f
    return lut


CODE_LUT = _code_lut()


def encode_codes(values):
    """e4m3 codes of f16 values, including the sign of zero and NaN."""
    return CODE_LUT[np.asarray(values, np.float16).view(np.uint16)]


def decode_output(field, W, H):
    """The layer output field as (H, W, C) natural e4m3 codes."""
    return field.reshape(C // 32, H, W, 32).transpose(1, 2, 0, 3).reshape(H, W, C)


def encode_field(codes, W, H, channels):
    """The stored byte layout of a field whose natural channels are `channels`."""
    return codes.reshape(H, W, channels // 32, 32).transpose(2, 0, 1, 3).reshape(-1)


def merged_positions(W, H, sx, sy):
    """Padded-grid source coordinates of every stored merged token and its 4C slots."""
    GW, GH = lf.padded_geometry(W, H, sx, sy)
    halfW, halfH = W // 2, H // 2
    MX, MY = np.meshgrid(np.arange(halfW), np.arange(halfH), indexing='xy')
    mx2 = MX.ravel() + sx // 2
    my2 = MY.ravel() + sy // 2
    bx, mx = mx2 >> 2, mx2 & 3
    by, my = my2 >> 2, my2 & 3
    xs = np.empty((mx2.size, 4), np.int64)
    ys = np.empty((mx2.size, 4), np.int64)
    for q in range(4):
        xs[:, q] = 8 * bx + 2 * mx + (q & 1)
        ys[:, q] = 8 * by + 2 * my + (q >> 1)
    return MX.ravel(), MY.ravel(), xs - sx, ys - sy


def rebuild(codes, W, H, merge_w, merge_b, sx, sy, group=K32):
    """The merge contract on the supplied e4m3 operands; returns (H/2, W/2, NPM) codes.

    `group` is the accumulation group: 32 rounds once per k32 (the kernel contract),
    4*C rounds once for the whole sum (the mathematical reference).
    """
    MX, MY, xs, ys = merged_positions(W, H, sx, sy)
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= W or ys.max() >= H:
        raise ValueError('a stored merged token reads outside the captured field')
    tokens = xs.shape[0]
    acc = np.empty((tokens, NPM), np.float64)
    acc[:] = merge_b.astype(np.float64)[None, :]
    value = lo.me.E4M3[codes].astype(np.float64)          # (H, W, C), exact
    for step in range(0, 4 * C, group):
        a = np.empty((tokens, group), np.float64)
        for k in range(group):
            kk = step + k
            q, c = divmod(kk, C)
            a[:, k] = value[ys[:, q], xs[:, q], c]
        w = merge_w[step:step + group].astype(np.float64)
        acc = np.add(a @ w, acc).astype(np.float16).astype(np.float64)
    acc = np.clip(acc, -LIMIT, LIMIT).astype(np.float32)
    codes_out = encode_codes(lo.me.q8(acc))
    out = np.empty((H // 2, W // 2, NPM), np.uint8)
    out[MY, MX] = codes_out
    return out


def compare(reference, actual, W, H):
    """Byte, per-plane, per-token and per-channel identity of two merge fields."""
    a = decode_merge(reference, W, H)
    b = decode_merge(actual, W, H)
    differ = a != b
    planes = [1.0 - differ[:, :, 32 * p:32 * p + 32].mean() for p in range(NPM // 32)]
    tokens = 1.0 - differ.reshape(-1, NPM).mean(axis=1)
    channels = 1.0 - differ.reshape(-1, NPM).mean(axis=0)
    delta = lo.me.E4M3[b].astype(np.float64) - lo.me.E4M3[a].astype(np.float64)
    return {'bytes': int(a.size), 'exact': float((a == b).mean()),
            'differing_codes': int(differ.sum()),
            'relative_rms': float(np.sqrt((delta ** 2).mean() /
                                          (lo.me.E4M3[a].astype(np.float64) ** 2).mean())),
            'max_abs_decoded': float(np.abs(delta).max()),
            'signed_mean_decoded': float(delta.mean()),
            'nonfinite_pairs': int((~np.isfinite(lo.me.E4M3[a]) |
                                    ~np.isfinite(lo.me.E4M3[b])).sum()),
            'saturated_reference': int(np.isin(a, (0x7e, 0xfe)).sum()),
            'saturated_actual': int(np.isin(b, (0x7e, 0xfe)).sum()),
            'plane_exact': planes,
            'tokens_exact': float((tokens == 1.0).mean()),
            'token_exact_min': float(tokens.min()),
            'channel_exact_min': float(channels.min()),
            'channel_exact_worst': int(np.argmax(differ.reshape(-1, NPM).sum(axis=0))),
            'mismatching_tokens': int((tokens < 1.0).sum())}


def decode_merge(field, W, H):
    return field.reshape(NPM // 32, H // 2, W // 2, 32).transpose(1, 2, 0, 3).reshape(
        H // 2, W // 2, NPM)


def named_tokens(reference, actual, W, H):
    """One mismatching and one exact merged token, plus one per output plane."""
    a = decode_merge(reference, W, H)
    b = decode_merge(actual, W, H)
    differ = a != b
    picks = {}
    for label, mask in (('mismatch', differ), ('control', ~differ)):
        if mask.any():
            y, x = (int(v[0]) for v in np.nonzero(mask.any(axis=2)))
            picks[label] = sample(a, b, y, x)
    for plane in range(NPM // 32):
        sub = differ[:, :, 32 * plane:32 * plane + 32]
        if sub.any():
            y, x = (int(v[0]) for v in np.nonzero(sub.any(axis=2)))
            picks[f'mismatch-plane{plane}'] = sample(a, b, y, x)
    return picks


def sample(a, b, y, x):
    return {'merged_yx': [y, x],
            'differing_channels': int((a[y, x] != b[y, x]).sum()),
            'reference_codes': a[y, x].tolist(),
            'actual_codes': b[y, x].tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', metavar='NAME=CAPTURE',
                    help='NAME=replay directory to rebuild and compare')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    ap.add_argument('--sette', type=Path, default=None,
                    help='results directory with out-out.bin/out-merge.bin, decoded with '
                         'the first capture geometry and weights')
    a = ap.parse_args()
    report = {'contract': {'k32': K32, 'rounding': 'one f16 RNE per k32',
                           'clamp': float(LIMIT), 'encoding': 'q8_exact'},
              'runs': {}}
    geometry = None
    for item in a.runs:
        name, _, directory = item.partition('=')
        if not directory:
            raise SystemExit(f'expected NAME=CAPTURE, got {item}')
        P, raw = lo.raw_layer(directory, 'enc1')
        geometry = geometry or P
        rows = rebuild_and_compare(name, P, raw['output'], raw['merge'], report)
        print_rows(name, rows)
    if a.sette is not None:
        if geometry is None:
            raise SystemExit('--sette needs a capture for weights and geometry')
        output = np.fromfile(a.sette / 'out-out.bin', np.uint8)
        merge = np.fromfile(a.sette / 'out-merge.bin', np.uint8)
        print_rows('sette', rebuild_and_compare('sette', geometry, output, merge, report))
    if a.json:
        json.dump(report, a.json, indent=2)


def print_rows(name, rows):
    for label in ('k32', 'single-rounding'):
        row = rows[label]
        print(f'{name:14s} {label:16s} {100 * row["exact"]:8.4f}% exact  '
              f'rms {row["relative_rms"]:.5f}  max {row["max_abs_decoded"]:.2f}  '
              f'bias {row["signed_mean_decoded"]:+.6f}  '
              f'planes {[round(100 * p, 3) for p in row["plane_exact"]]}')
    for label, pick in rows['tokens'].items():
        print(f'  {label:18s} merged {tuple(pick["merged_yx"])} '
              f'{pick["differing_channels"]} differing channels')


def rebuild_and_compare(name, P, output, merge, report):
    packed = lf.WeightPack(P, 'enc1')
    merge_w = packed.parts['mergeW'].astype(np.float32)
    merge_b = packed.parts['mergeB'].astype(np.float32)
    codes = decode_output(output, P.W, P.H)
    rows = {}
    for label, group in (('k32', K32), ('single-rounding', 4 * C)):
        rebuilt = rebuild(codes, P.W, P.H, merge_w, merge_b, P.sx, P.sy, group)
        rebuilt_field = encode_field(rebuilt, P.W // 2, P.H // 2, NPM)
        rows[label] = compare(merge, rebuilt_field, P.W, P.H)
        if group == K32:
            rows['tokens'] = named_tokens(merge, rebuilt_field, P.W, P.H)
    report['runs'][name] = {'fields': rows,
                            'weights_sha256': __import__('hashlib').sha256(
                                P.wbuf).hexdigest()}
    return rows

if __name__ == '__main__':
    main()

