#!/usr/bin/env python3
"""M15-B: operator-level oracle for the MLP GELU + quantizer fusion.

Builds one small synthetic fixture twice:

  * control - the accepted `swin_mlp_gelu` then `swin_q8_quant` pair;
  * fused   - one `swin_mlp_gelu_quant` dispatch.

Both must reproduce the same GELU/E4M3 tensor, INT8 codes and per-row scales,
and the independent CPU model must reproduce the GPU tensors.  This is the
B2 gate; the full chain is only touched after it passes.
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
sys.path.insert(0, str(REPO / 'references/sources/d4r/kernels/tools'))
import m09_enc1_q as eq
import model_enc3 as model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_graph(directory, rows, k, fused):
    params = np.array([rows, k, k, 0], '<u4')
    (directory / 'params-q8.bin').write_bytes(params.tobytes())
    lines = ['version 1',
             f'resource z {rows * k * 2} z.bin',
             f'resource g {rows * k * 2}',
             f'resource codes {rows * k}',
             f'resource scales {rows * 4}',
             'resource scratch 16']
    if fused:
        lines += ['stage mlp-fused', 'shader swin_mlp_gelu_quant.cso',
                  'threads 256 1 1', f'groups {(rows + 63) // 64} 1 1',
                  'params params-q8.bin 16', 'srv 0 z',
                  'uav 0 g', 'uav 1 codes', 'uav 2 scales']
    else:
        gelu = np.zeros(40, '<u4')
        gelu[8] = rows
        gelu[11] = k // 32
        (directory / 'params-gelu.bin').write_bytes(gelu.tobytes())
        lines += ['stage mlp-gelu', 'shader swin_mlp_gelu.cso',
                  'threads 64 1 1', f'groups {(rows + 63) // 64} 1 1',
                  'params params-gelu.bin 160', 'srv 0 z', 'srv 1 scratch',
                  'srv 2 scratch', 'uav 0 g', 'uav 1 scratch',
                  'stage mlp-down-quant', 'shader swin_q8_quant.cso',
                  'threads 64 1 1', f'groups {(rows + 63) // 64} 1 1',
                  'params params-q8.bin 16', 'srv 0 g',
                  'uav 0 codes', 'uav 1 scales']
    lines += [f'dump g 0 {rows * k * 2} g.bin',
              f'dump codes 0 {rows * k} codes.bin',
              f'dump scales 0 {rows * 4} scales.bin']
    (directory / 'graph.txt').write_text('\n'.join(lines) + '\n')


def make_input(rows=8, k=256):
    rng = np.random.default_rng(15)
    z = rng.normal(size=(rows, k)).astype('<f2')
    z[0] = 0
    z[1] = 2.0
    z[2] = -2.0
    z[3, :16] = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
                 0.001953125, -0.001953125, 448.0, -448.0, 1.5, -1.5,
                 0.0009765625, -0.0009765625]
    return z


def run(probe, shaders, directory, results):
    done = subprocess.run([str(probe), '--graph', str(directory / 'graph.txt'),
                           '--out', str(results), '--shaders', str(shaders),
                           '--seed', '0'], capture_output=True, text=True)
    results.mkdir(parents=True, exist_ok=True)
    (results / 'execution.log').write_text(done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError(f'{directory}: {done.stdout[-800:]}{done.stderr[-800:]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--probe', type=Path,
                    default=REPO / 'build/native/Release/m_network_stage.exe')
    ap.add_argument('--shaders', type=Path, default=REPO / 'build/native')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)

    rows, k = 8, 256
    z = make_input(rows, k)
    cpu_g = model.q8(model.gelu_poly(z)).astype('<f2')
    cpu_codes, cpu_scales = eq.quantize_rows_float(cpu_g)
    cpu_scales = cpu_scales.ravel().astype('<f4')

    results = {}
    for name, fused in (('control', False), ('fused', True)):
        directory = a.out / name
        directory.mkdir()
        (directory / 'z.bin').write_bytes(z.tobytes())
        write_graph(directory, rows, k, fused)
        run(a.probe, a.shaders, directory, a.out / f'{name}-results')
        g = np.fromfile(a.out / f'{name}-results/g.bin', '<f2').reshape(rows, k)
        codes = np.fromfile(a.out / f'{name}-results/codes.bin', np.int8).reshape(rows, k)
        scales = np.fromfile(a.out / f'{name}-results/scales.bin', '<f4')
        results[name] = {
            'g_sha256': sha(a.out / f'{name}-results/g.bin'),
            'codes_sha256': sha(a.out / f'{name}-results/codes.bin'),
            'scales_sha256': sha(a.out / f'{name}-results/scales.bin'),
            'g_cpu_exact': bool(np.array_equal(g, cpu_g)),
            'codes_cpu_exact': bool(np.array_equal(codes, cpu_codes)),
            'scales_cpu_exact': bool(np.array_equal(scales, cpu_scales)),
            'g_finite': bool(np.isfinite(g).all()),
            'codes_range': [int(codes.min()), int(codes.max())],
        }
    checks = {
        'g_identical': results['fused']['g_sha256'] == results['control']['g_sha256'],
        'codes_identical': results['fused']['codes_sha256'] == results['control']['codes_sha256'],
        'scales_identical': results['fused']['scales_sha256'] == results['control']['scales_sha256'],
        **{f'{name}_{key}': value for name, row in results.items()
           for key, value in row.items() if key.endswith('_exact')},
    }
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL',
              'rows': rows, 'k': k, 'checks': checks, 'results': results,
              'shader_sha256': sha(a.shaders / 'swin_mlp_gelu_quant.cso'),
              'probe_sha256': sha(a.probe)}
    (a.out / 'oracle.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'],
                      'failed': [k for k, v in checks.items() if not v],
                      'results': results}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
