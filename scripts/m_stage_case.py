#!/usr/bin/env python3
"""C18/C19 groundwork: export one captured matrix stage as a native GPU case, and
compare the native result with the reference arithmetic.

`export` writes `MSTAGE01` cases (see `src/native/m_stage_probe.cpp`) from a
capture:

  dec0.expansion  x = the captured token field (real E4M3 values, one row per
                  2x2 render-pixel block), W = the captured expansion matrix,
                  bias = the stage's f16 bias.

`check` compares the probe's float32 output with three independent references:
exact float64, the float16-step chain (`model_enc3.mma_chain`, one rounding per
32 products), and the error the step rounding contributes.  No rounding model
is fitted to the GPU result: the chain is computed from the published contract.

usage:
  m_stage_case.py export CAPTURE --matrix dec0.expansion --out case.bin [--rows N] [--stride S]
  m_stage_case.py check --case case.bin --result result.bin [--json OUT]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'engine'))

import m_weights as mw       # noqa: E402
import model_enc3 as me      # noqa: E402

F16, F32, F64 = np.float16, np.float32, np.float64
HEADER = np.dtype([('magic', 'S8'), ('version', '<u4'), ('M', '<u4'),
                   ('K', '<u4'), ('N', '<u4'), ('flags', '<u4')])


def find_matrix(stages, name):
    for stage, row in mw.matrices(stages):
        if row['name'] == name:
            return stage, row
    raise SystemExit(f'{name} is not in this capture')


def export(a):
    stages = mw.inventory(a.capture)
    stage, row = find_matrix(stages, a.matrix)
    acts = mw.activations(stages)
    info = next((x for x in acts if x['stage'] == stage.stage and x['tensor'] is not None), None)
    if info is None:
        raise SystemExit(f'{stage.stage} has no captured operand to export')

    W = np.asarray(row['W'], F16)
    K, N = W.shape
    x = np.asarray(info['tensor'], F16).reshape(-1, K)[::a.stride][:a.rows]
    if x.shape[0] < a.rows:
        raise SystemExit(f'only {x.shape[0]} rows available')
    bias = (np.asarray(row['bias'], F16) if row['bias'] is not None
            else np.zeros(N, F16))

    header = np.zeros(1, HEADER)[0]
    header['magic'] = b'MSTAGE01'
    header['version'] = 1
    header['M'], header['K'], header['N'] = x.shape[0], K, N
    header['flags'] = 1 if row['bias'] is not None else 0
    with open(a.out, 'wb') as handle:
        handle.write(header.tobytes())
        handle.write(np.ascontiguousarray(x, F16).tobytes())
        handle.write(np.ascontiguousarray(W, F16).tobytes())
        handle.write(np.ascontiguousarray(bias, F16).tobytes())
    print(f'{a.matrix}: {x.shape[0]}x{K} x {K}x{N} from {stage.kernel}, '
          f'{os.path.getsize(a.out)} bytes -> {a.out}')
    return 0


def check(a):
    with open(a.case, 'rb') as handle:
        header = np.frombuffer(handle.read(HEADER.itemsize), HEADER)[0]
        assert header['magic'] == b'MSTAGE01'
        M, K, N = int(header['M']), int(header['K']), int(header['N'])
        x = np.frombuffer(handle.read(M * K * 2), F16).reshape(M, K)
        w = np.frombuffer(handle.read(K * N * 2), F16).reshape(K, N)
        bias = np.frombuffer(handle.read(N * 2), F16)
    gpu = np.fromfile(a.result, F32).reshape(M, N)

    exact = np.asarray(x, F64) @ np.asarray(w, F64)
    if int(header['flags']) & 1:
        exact = exact + np.asarray(bias, F64)[None, :]
    chained = me.mma_chain(np.asarray(x, F64), np.asarray(w, F64),
                           C0=None if not int(header['flags']) & 1
                           else np.broadcast_to(np.asarray(bias, F64), (M, N)))
    chained = np.asarray(chained, F64)

    def ulp_stats(got, want):
        got = np.asarray(got, F64)
        want = np.asarray(want, F64)
        diff = got - want
        peak = max(float(np.abs(want).max()), 1e-30)
        return {'max_abs': float(np.abs(diff).max()),
                'rms': float(np.sqrt((diff ** 2).mean())),
                'max_rel_to_peak': float(np.abs(diff).max() / peak),
                'identical_f16': float(np.mean(np.asarray(got, F16) == np.asarray(want, F16)))}

    result = {
        'case': {'rows': M, 'k': K, 'n': N},
        'gpu_vs_exact_f64': ulp_stats(gpu, exact),
        'gpu_vs_step_chain': ulp_stats(gpu, chained),
        'chain_vs_exact_f64': ulp_stats(chained, exact),
        'gpu_finite': bool(np.isfinite(gpu).all()),
        'gpu_absmax': float(np.abs(gpu).max()),
    }
    print(f'case {M}x{K} x {K}x{N}, bias={bool(int(header["flags"]) & 1)}')
    for key in ('gpu_vs_step_chain', 'gpu_vs_exact_f64', 'chain_vs_exact_f64'):
        row = result[key]
        print(f'  {key:22s} max|d| {row["max_abs"]:9.5f}  rms {row["rms"]:9.6f}  '
              f'/peak {row["max_rel_to_peak"]:7.5f}  identical f16 {100 * row["identical_f16"]:5.1f}%')
    print(f'  finite={result["gpu_finite"]}  |y|max={result["gpu_absmax"]:.4f}')
    if a.json:
        json.dump(result, a.json, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('capture')
    e.add_argument('--matrix', default='dec0.expansion')
    e.add_argument('--out', required=True)
    e.add_argument('--rows', type=int, default=8192)
    e.add_argument('--stride', type=int, default=1)
    c = sub.add_parser('check')
    c.add_argument('--case', required=True)
    c.add_argument('--result', required=True)
    c.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()
    return export(a) if a.command == 'export' else check(a)


if __name__ == '__main__':
    sys.exit(main())
