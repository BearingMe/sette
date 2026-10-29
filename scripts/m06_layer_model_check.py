#!/usr/bin/env python3
"""M06: run d4r's independent numpy Preset M layer model against a layer capture.

The capture produced by the Windows ZLUDA replay hook stores, for every layer
launch, the raw argument buffer and the device allocations as they were *before*
the launch (`alloc-N.bin`) plus `gpu-output/alloc-N.bin` as they were after the
kernel ran.  d4r's `engine/m_model.py` expects the Linux bridge's
`dump/launch-NNN-...-arg024-*` snapshots instead, so this script feeds it the
equivalent bytes from `gpu-output/`.

It deliberately reuses d4r's model rather than re-deriving the layer: the point
of the checkpoint is to compare an independent implementation (numpy, exact
accumulation, explicit NVIDIA roundings) with what the GPU actually produced.

usage: m06_layer_model_check.py CAPTURE_DIR [--json OUT]

CAPTURE_DIR holds `replay-<seq>-rrlite_<layer>_4x4` directories with a
`gpu-output/` subdirectory.  Nothing is written unless --json is given.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ENGINE = os.path.join(REPO, 'references', 'sources', 'd4r', 'engine')
sys.path.insert(0, ENGINE)   # m_model inserts kernels/tools itself

import m_model as mm          # noqa: E402
import model_enc3 as me       # noqa: E402


def post_launch_arena(directory, address):
    """Bytes of the gpu-output allocation that contains `address`."""
    allocs = {}
    for line in open(os.path.join(directory, 'manifest.txt')):
        q = line.split()
        if q and q[0] == 'alloc':
            allocs[int(q[1])] = (int(q[2], 16), int(q[3]) if len(q) > 3 else None)
    for index, (base, _) in allocs.items():
        path = os.path.join(directory, 'gpu-output', f'alloc-{index}.bin')
        if not os.path.exists(path):
            continue
        data = np.fromfile(path, np.uint8)
        if base <= address < base + data.size:
            return base, data
    raise ValueError(f'{directory}: no post-launch allocation contains {address:#x}')


def compare(directory, name):
    P = me.Params(directory)
    C, heads, NPM, CIN = mm.LAYERS[name]
    keys = [l.split()[1] for l in open(os.path.join(directory, 'manifest.txt')) if l.startswith('kernel')]
    assert keys and keys[0] == f'rrlite_{name}_4x4', (name, keys)

    t = mm.plain_layer(P, name)
    if CIN:
        x = mm.tokens(P.arena, P.inp - P.abase, P.W // 2, P.H // 2, CIN)
        skip = mm.tokens(P.arena, P.p32 - P.abase, P.W, P.H, C)
    else:
        x = mm.tokens(P.arena, P.inp - P.abase, P.W, P.H, C)
        skip = None

    abase, after = post_launch_arena(directory, P.out)
    out, merged = mm.forward(t, name, x, skip, P.sx, P.sy)
    ref = mm.tokens(after, P.out - abase, P.W, P.H, C)

    row = {
        'layer': name,
        'grid': list(P.grid),
        'extent': [P.W, P.H, C],
        'shift': [P.sx, P.sy],
        'psnr_db': float(mm.psnr(out, ref)),
        'identical_pct': float(100.0 * (out == ref).mean()),
        'max_abs_diff': float(np.abs(np.asarray(out, np.float64) - ref).max()),
        'rms_diff': float(np.sqrt(((np.asarray(out, np.float64) - ref) ** 2).mean())),
    }
    if NPM:
        mref = mm.tokens(after, P.p48 - abase, P.W // 2, P.H // 2, NPM)
        row['merged_psnr_db'] = float(mm.psnr(merged, mref))
        row['merged_identical_pct'] = float(100.0 * (merged == mref).mean())
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.capture, 'replay-*-rrlite_*_4x4')))
    launches = []
    for directory in dirs:
        kernel = open(os.path.join(directory, 'manifest.txt')).readline().split()[1]
        name = kernel[len('rrlite_'):-len('_4x4')]
        if name in mm.LAYERS:
            launches.append((mm.ORDER.index(name), name, directory))
    if len(launches) < 10:
        raise SystemExit(f'{a.capture}: found {len(launches)} layer launches, expected at least 10')
    launches.sort()

    rows = []
    for _, name, directory in launches[:10]:
        row = compare(directory, name)
        rows.append(row)
        line = (f"{row['layer']:10s} {row['extent'][0]}x{row['extent'][1]}x{row['extent'][2]} "
                f"shift {row['shift'][0]},{row['shift'][1]}: "
                f"{row['psnr_db']:5.1f} dB, identical {row['identical_pct']:5.1f}%, "
                f"max|d| {row['max_abs_diff']:.4g}")
        if 'merged_psnr_db' in row:
            line += f"; merged {row['merged_psnr_db']:5.1f} dB, identical {row['merged_identical_pct']:5.1f}%"
        print(line, flush=True)

    if a.json:
        json.dump(rows, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
