#!/usr/bin/env python3
"""M09: one native INT8 matrix operation on gfx1101, against independent references.

The operation is `dec0`'s expansion GEMM with real captured operands:

    Y[8160 x 80] = A[8160 x 64] * W[64 x 80] + bias[80]

`A` is the captured token field (a deterministic stride-16 sample of the 130 560
tokens), `W` the captured E4M3 expansion matrix widened to f16, `bias` the
stage's f16 bias.  The same case already drives the FP16 stage
(`gemm_f16_cs.hlsl`, EXP-010), so the INT8 result is compared against a
reusable FP16 operator output on identical operands, not against a new capture.

Subcommands:

  export CASE_F16 --out CASE_INT8
      Quantise both operands through the M07 contract (`m_int8`): symmetric
      INT8, per-output-channel weight scale, per-row activation scale, no zero
      point.  Writes the version-2 case and a JSON record of the quantisation.

  check --case-int8 ... --f16-case ... --f16-result ... --int8-result ...
        [--repeat INT8_RESULT] [--json OUT]
      Compares the GPU INT8 output with (a) an independent CPU INT8 reference
      computed from the same codes and scales, (b) the FP16 operator output on
      the same operands, and (c) the exact float64 product.  Checks finite,
      determinism (when a repeat is given) and the integer-accumulation bound.

  run --out DIR --f16-case CASE
      export, execute the FP16 and INT8 stages twice, and check.

The integer reduction is exact for K <= 131071 (|sum| <= K * 127 * 127), so any
difference from the CPU reference must come from the f32 epilogue - that is what
`gpu_vs_cpu_int8` is for.  Everything else measured here is quantisation error,
which is not a defect.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import m_int8 as q                      # noqa: E402

F16, F32, F64 = np.float16, np.float32, np.float64
HEADER = np.dtype([('magic', 'S8'), ('version', '<u4'), ('M', '<u4'),
                   ('K', '<u4'), ('N', '<u4'), ('flags', '<u4')])
PROBE = REPO / 'build/native/Release/m_stage_probe.exe'
SHADERS = REPO / 'build/native'
QMAX = q.QMAX


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_case(path):
    raw = Path(path).read_bytes()
    header = np.frombuffer(raw[:HEADER.itemsize], HEADER)[0]
    if header['magic'] != b'MSTAGE01':
        raise SystemExit(f'{path}: not a MSTAGE01 case')
    M, K, N, version = int(header['M']), int(header['K']), int(header['N']), int(header['version'])
    offset = HEADER.itemsize
    if version == 1:
        expected = offset + M * K * 2 + K * N * 2 + N * 2
        if len(raw) != expected:
            raise SystemExit(f'{path}: {len(raw)} bytes, header implies {expected}')
        x = np.frombuffer(raw, F16, M * K, offset).reshape(M, K)
        offset += M * K * 2
        w = np.frombuffer(raw, F16, K * N, offset).reshape(K, N)
        offset += K * N * 2
        bias = np.frombuffer(raw, F16, N, offset)
        return {'version': 1, 'M': M, 'K': K, 'N': N, 'flags': int(header['flags']),
                'x': x, 'w': w, 'bias': bias}
    if version == 2:
        expected = offset + M * K + N * K + M * 4 + N * 4 + N * 4
        if len(raw) != expected:
            raise SystemExit(f'{path}: {len(raw)} bytes, header implies {expected}')
        a = np.frombuffer(raw, np.int8, M * K, offset).reshape(M, K)
        offset += M * K
        w = np.frombuffer(raw, np.int8, N * K, offset).reshape(N, K)
        offset += N * K
        ascale = np.frombuffer(raw, F32, M, offset)
        offset += M * 4
        wscale = np.frombuffer(raw, F32, N, offset)
        offset += N * 4
        bias = np.frombuffer(raw, F32, N, offset)
        return {'version': 2, 'M': M, 'K': K, 'N': N, 'flags': int(header['flags']),
                'a': a, 'w': w, 'wscale': wscale, 'ascale': ascale, 'bias': bias}
    raise SystemExit(f'{path}: unknown case version {version}')


def write_case(path, M, K, N, a, w, wscale, ascale, bias):
    header = np.zeros(1, HEADER)[0]
    header['magic'] = b'MSTAGE01'
    header['version'] = 2
    header['M'], header['K'], header['N'] = M, K, N
    header['flags'] = 1
    with open(path, 'wb') as handle:
        handle.write(header.tobytes())
        handle.write(np.ascontiguousarray(a, np.int8).tobytes())
        handle.write(np.ascontiguousarray(w, np.int8).tobytes())
        handle.write(np.ascontiguousarray(ascale, F32).tobytes())
        handle.write(np.ascontiguousarray(wscale, F32).tobytes())
        handle.write(np.ascontiguousarray(bias, F32).tobytes())


def export(a):
    case = read_case(a.case_f16)
    M, K, N = case['M'], case['K'], case['N']
    if case['version'] != 1:
        raise SystemExit('expected a version-1 FP16 case to quantise')
    x = np.asarray(case['x'], F16)
    w = np.asarray(case['w'], F16)
    bias = np.asarray(case['bias'], F16).astype(F32)

    wq = q.quantize(np.asarray(w, F64), 'per_output_channel')
    aq = q.quantize_rows(np.asarray(x, F64), 'per_row')
    write_case(a.out, M, K, N, aq['codes'], np.ascontiguousarray(wq['codes'].T),
               wq['scales'], aq['scales'].ravel().astype(F32), bias)

    exact = np.asarray(x, F64) @ np.asarray(w, F64) + np.asarray(bias, F64)[None, :]
    weight_only = np.asarray(x, F64) @ q.dequantize(wq) + np.asarray(bias, F64)[None, :]
    activation_only = q.dequantize_rows(aq) @ np.asarray(w, F64) + np.asarray(bias, F64)[None, :]
    both = q.dequantize_rows(aq) @ q.dequantize(wq) + np.asarray(bias, F64)[None, :]
    norm = float(np.linalg.norm(exact))
    record = {
        'case_f16': str(a.case_f16), 'case_f16_sha256': sha(a.case_f16),
        'case_int8': str(a.out), 'case_int8_sha256': sha(a.out),
        'shape': {'M': M, 'K': K, 'N': N},
        'weight': q.metrics(np.asarray(w, F64), wq),
        'activation': {
            'scope': 'per_row',
            'at_endpoint': float(np.mean(np.abs(aq['codes']) == QMAX)),
            'distinct_codes': int(np.unique(aq['codes']).size),
            'scale_min': float(np.min(aq['scales'])), 'scale_max': float(np.max(aq['scales'])),
        },
        'quantization_only': {
            'exact_frobenius': norm,
            'weight_rel_frobenius': float(np.linalg.norm(weight_only - exact) / norm),
            'activation_rel_frobenius': float(np.linalg.norm(activation_only - exact) / norm),
            'both_rel_frobenius': float(np.linalg.norm(both - exact) / norm),
        },
        'accumulator_bound': {'max_abs_sum': int(K) * QMAX * QMAX,
                              'int32_limit': 2 ** 31 - 1,
                              'safe': int(K) * QMAX * QMAX < 2 ** 31},
    }
    Path(a.out).with_suffix('.json').write_text(json.dumps(record, indent=2))
    print(f'{M}x{K} x {K}x{N} -> {a.out} ({os.path.getsize(a.out)} bytes)')
    print(f"  weights: per_output_channel rmse {record['weight']['rmse']:.5f}, "
          f"rel_frobenius {record['weight']['rel_frobenius']:.5f}")
    print(f"  activations: per_row endpoint {record['activation']['at_endpoint']:.4f}")
    print(f"  quantisation-only relative error (weights, acts, both): "
          f"{record['quantization_only']['weight_rel_frobenius']:.5f}, "
          f"{record['quantization_only']['activation_rel_frobenius']:.5f}, "
          f"{record['quantization_only']['both_rel_frobenius']:.5f}")
    return 0


def run_probe(case, out, shader, probe=PROBE):
    done = subprocess.run([str(probe), '--case', str(case), '--out', str(out),
                           '--shader', str(SHADERS / shader)],
                          capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'{shader}: {done.stdout[-500:]}{done.stderr[-500:]}')
    return done.stdout.strip()


def reference(case, bias_dtype=F64):
    """The independent CPU INT8 product: integer reduction, then the rescale."""
    aq = {'codes': case['a'], 'scales': np.asarray(case['ascale'], F64).reshape(-1, 1),
          'scope': 'per_row'}
    wq = {'codes': np.ascontiguousarray(case['w'].T), 'scales': case['wscale'],
          'scope': 'per_output_channel'}
    exact_integer = q.independent_integer_product(aq['codes'], wq['codes'])
    y = q.gemm_integer(aq['codes'], wq['codes'], aq['scales'], wq['scales'],
                       'per_output_channel')
    if int(case['flags']) & 1:
        y = y + np.asarray(case['bias'], F64)[None, :]
    return y, exact_integer


def stats(got, want, label, exact_integer=None):
    got = np.asarray(got, F64)
    want = np.asarray(want, F64)
    diff = got - want
    norm = float(np.linalg.norm(want))
    row = {'label': label,
           'max_abs': float(np.abs(diff).max()),
           'mean_abs': float(np.abs(diff).mean()),
           'rms': float(np.sqrt((diff ** 2).mean())),
           'rel_frobenius': float(np.linalg.norm(diff) / norm) if norm else 0.0,
           'max_rel_to_peak': float(np.abs(diff).max() / max(np.abs(want).max(), 1e-30)),
           'identical_f16': float(np.mean(np.asarray(got, F16) == np.asarray(want, F16)))}
    return row


def check(a):
    case = read_case(a.case_int8)
    f16 = read_case(a.f16_case)
    if (case['M'], case['K'], case['N']) != (f16['M'], f16['K'], f16['N']):
        raise SystemExit('INT8 and FP16 cases disagree on the shape')
    M, N = case['M'], case['N']
    gpu = np.fromfile(a.int8_result, F32).reshape(M, N).astype(F64)
    fp16 = np.fromfile(a.f16_result, F32).reshape(M, N).astype(F64)

    cpu, exact_integer = reference(case)
    # the integer reduction is only correct if the f32 epilogue is the sole
    # difference: compare the GPU result with the CPU's own rescale
    integer_check = float(np.abs(gpu - cpu).max())

    exact = np.asarray(f16['x'], F64) @ np.asarray(f16['w'], F64)
    if int(f16['flags']) & 1:
        exact = exact + np.asarray(f16['bias'], F64)[None, :]

    result = {
        'case': {'M': M, 'K': case['K'], 'N': N,
                 'int8_sha256': sha(a.case_int8), 'f16_sha256': sha(a.f16_case)},
        'gpu_vs_cpu_int8': stats(gpu, cpu, 'GPU INT8 vs independent CPU INT8'),
        'gpu_vs_fp16_stage': stats(gpu, fp16, 'GPU INT8 vs FP16 stage (same operands)'),
        'fp16_stage_vs_exact_f64': stats(fp16, exact, 'FP16 stage vs exact float64'),
        'gpu_vs_exact_f64': stats(gpu, exact, 'GPU INT8 vs exact float64'),
        'cpu_vs_exact_f64': stats(cpu, exact, 'CPU INT8 vs exact float64'),
        'integer_accumulation': {
            'max_abs_gpu_minus_cpu': integer_check,
            'f32_epilogue_only': bool(integer_check <= 1e-5 * max(float(np.abs(cpu).max()), 1.0)),
            'int32_overflow_free': bool(np.abs(exact_integer).max() < 2 ** 31),
        },
        'gpu_finite': bool(np.isfinite(gpu).all()),
        'gpu_absmax': float(np.abs(gpu).max()),
    }
    if a.repeat is not None:
        again = np.fromfile(a.repeat, F32).reshape(M, N).astype(F64)
        result['deterministic'] = bool(np.array_equal(again, gpu))
        result['repeat_sha256'] = sha(a.repeat)
    result['int8_result_sha256'] = sha(a.int8_result)
    for key in ('gpu_vs_cpu_int8', 'gpu_vs_fp16_stage', 'gpu_vs_exact_f64',
                'fp16_stage_vs_exact_f64'):
        row = result[key]
        print(f"  {row['label']:42s} rel {row['rel_frobenius']:.5f}  rms {row['rms']:.5f}  "
              f"max {row['max_abs']:.5f}  identical f16 {100 * row['identical_f16']:5.1f}%")
    print(f"  integer accumulation: max|gpu-cpu| {integer_check:.3e}, "
          f"f32-epilogue-only={result['integer_accumulation']['f32_epilogue_only']}, "
          f"int32-safe={result['integer_accumulation']['int32_overflow_free']}")
    print(f"  finite={result['gpu_finite']}  |y|max={result['gpu_absmax']:.4f}"
          + (f"  deterministic={result.get('deterministic')}" if a.repeat else ''))
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2))
    return 0


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    case_int8 = out / 'case-int8.bin'
    export(type('Args', (), {'case_f16': a.f16_case, 'out': case_int8})())
    f16_a = out / 'result-f16-a.bin'
    f16_b = out / 'result-f16-b.bin'
    int8_a = out / 'result-int8-a.bin'
    int8_b = out / 'result-int8-b.bin'
    print(run_probe(a.f16_case, f16_a, 'gemm_f16_cs.cso'))
    print(run_probe(a.f16_case, f16_b, 'gemm_f16_cs.cso'))
    print(run_probe(case_int8, int8_a, 'gemm_int8_cs.cso'))
    print(run_probe(case_int8, int8_b, 'gemm_int8_cs.cso'))
    class Check:
        pass
    c = Check()
    c.case_int8, c.f16_case = case_int8, a.f16_case
    c.int8_result, c.repeat = int8_a, int8_b
    c.f16_result = f16_a
    c.json = out / 'm09.json'
    print(f'f16 repeat identical: {sha(f16_a) == sha(f16_b)}')
    return check(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('case_f16')
    e.add_argument('--out', required=True)
    c = sub.add_parser('check')
    c.add_argument('--case-int8', required=True)
    c.add_argument('--f16-case', required=True)
    c.add_argument('--f16-result', required=True)
    c.add_argument('--int8-result', required=True)
    c.add_argument('--repeat', default=None)
    c.add_argument('--json', default=None)
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--f16-case', required=True)
    a = ap.parse_args()
    return {'export': export, 'check': check, 'run': run}[a.command](a)


if __name__ == '__main__':
    sys.exit(main())
