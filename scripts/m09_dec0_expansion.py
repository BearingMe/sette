#!/usr/bin/env python3
"""EXP-024: isolated full-size dec0 expansion, physical E4M3 -> GPU INT8.

python scripts/m09_dec0_expansion.py --fixture <dec0 image fixture> --out <temp dir>
No stage/frame integration: the FP16 probe invokes the existing Column function.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m09_enc1_q as eq
import m_int8
import m_weights as mw
import model_enc3 as me

ROOT = Path(__file__).resolve().parents[1]
M, K, N = 130560, 64, 80
LIVE = np.array([32*s+i for s in range(4) for i in range(20)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_offset(row, c, rows=M):
    return (c // 32) * rows * 32 + row * 32 + c % 32


def weight_offset(k, n):
    kb = 2*(n//16) + k//32
    lane = 4*(n % 8) + (k % 16)//4
    return 512*kb + 16*lane + 8*((n % 16)//8) + k % 4 + 4*((k % 32)//16)


def decode_operands(tokens, blob, rows=M):
    """Bulk decode independent of the shader address formulas (reshape/table).

    Full physical 128 columns are decoded before selecting the 80-column map.
    """
    tokens = np.frombuffer(tokens, np.uint8)
    blob = np.frombuffer(blob, np.uint8)
    if tokens.size != rows*64 or blob.size != 8448:
        raise ValueError('expected two 32-channel planes and an 8448-byte weight blob')
    a = mw.e4m3_decode(tokens.reshape(2, rows, 32).transpose(1, 0, 2).reshape(rows, 64))
    offsets = mw.layout_offsets(mw.front_end_layout('dec0'))
    w = mw.e4m3_decode(blob[offsets])[:, LIVE]
    bias = blob[8192:].copy().view('<f2')[LIVE].astype(np.float32)
    if not (np.isfinite(a).all() and np.isfinite(w).all() and np.isfinite(bias).all()):
        raise ValueError('non-finite source operands')
    return a.astype(np.float16), w.astype(np.float32), bias


def cpu_reference(a, w, bias):
    codes, scales = eq.quantize_rows_float(a)
    wq = m_int8.quantize(w.astype(np.float64), 'per_output_channel')
    wc, ws = wq['codes'], wq['scales'].astype(np.float32)
    acc = codes.astype(np.int64) @ wc.astype(np.int64)
    y = (acc.astype(np.float32)*scales)*ws[None, :]
    y = (y + bias[None, :]).astype(np.float32)
    return codes, scales, wc, ws, acc, y, y.astype(np.float16)


def metrics(got, want):
    got, want = got.astype(np.float64), want.astype(np.float64)
    d = got-want
    rms = np.sqrt(np.mean(d*d))
    norm = np.sqrt(np.mean(want*want))
    return {'relative_rms': float(rms/norm) if norm else None,
            'rms': float(rms), 'max_abs': float(np.max(np.abs(d))),
            'signed_bias': float(d.mean()), 'mean_abs': float(np.abs(d).mean()),
            'per_channel_rms': np.sqrt(np.mean(d*d, axis=0)).tolist(),
            'per_channel_max': np.max(np.abs(d), axis=0).tolist(),
            'non_finite': int((~np.isfinite(got)).sum())}


def write_graph(fixture, out, wc, ws, bias):
    out.mkdir(parents=True, exist_ok=True)
    for name in ('tokens', 'weights'):
        (out/f'{name}.bin').write_bytes((fixture/f'in-{name}.bin').read_bytes())
    (out/'params.bin').write_bytes((fixture/'params.bin').read_bytes())
    for name, a in [('wc', wc.T.astype(np.int8)), ('ws', ws), ('bias', bias)]:
        (out/f'{name}.bin').write_bytes(np.ascontiguousarray(a).tobytes())
    sizes = {'tokens': M*K, 'weights': 8448, 'wc': N*K, 'ws': N*4, 'bias': N*4,
             'a': M*K, 'scale': M*4, 'f16': M*N*2, 'half': M*N*2,
             'acc': M*N*4, 'float': M*N*4, 'scratch': 16}
    initialized = {'tokens', 'weights', 'wc', 'ws', 'bias'}
    lines = ['version 1']
    for name, size in sizes.items():
        lines.append(f'resource {name} {size}' + (f' {name}.bin' if name in initialized else ''))
    for name, shader, count, srvs, uavs in [
        ('fp16', 'dec0_expansion_f16.cso', M*40, ['weights', 'tokens'], ['scratch']*4+['f16']),
        ('quant', 'dec0_q8_quant.cso', M, ['weights', 'tokens'], ['scratch']*4+['a', 'scale']),
        ('gemm', 'dec0_q8_matmul.cso', M*40, ['a', 'wc', 'scale', 'ws', 'bias'], ['half', 'acc', 'float'])]:
        groups_y = count//M
        lines += [f'stage {name}', f'shader {shader}', 'threads 64 1 1',
                  f'groups {M//64} {groups_y} 1', 'params params.bin 32']
        lines += [f'srv {i} {v}' for i, v in enumerate(srvs)]
        lines += [f'uav {i} {v}' for i, v in enumerate(uavs)]
    for name in ('a', 'scale', 'wc', 'ws', 'f16', 'half', 'acc', 'float'):
        lines.append(f'dump {name} 0 {sizes[name]} {name}.bin')
    (out/'graph.txt').write_text('\n'.join(lines)+'\n')


def run(fixture, out, rga=None):
    a, w, bias = decode_operands((fixture/'in-tokens.bin').read_bytes(),
                                 (fixture/'in-weights.bin').read_bytes())
    codes, scales, wc, ws, acc, y, half = cpu_reference(a, w, bias)
    write_graph(fixture, out, wc, ws, bias)
    shader_dir = ROOT/'build/native'
    shaders = {p.name: sha(p) for p in shader_dir.glob('*.cso')}
    for seed in (0, 1):
        subprocess.run([str(shader_dir/'Release/m_network_stage.exe'), '--graph', str(out/'graph.txt'),
                        '--out', str(out/f'run{seed}'), '--shaders', str(shader_dir),
                        '--seed', str(seed)], check=True, capture_output=True)
    def load(name, dtype, shape):
        return np.fromfile(out/'run0'/f'{name}.bin', dtype).reshape(shape)
    gpu_codes = load('a', np.int8, (M, K))
    gpu_scale = load('scale', np.float32, (M, 1))
    gpu_wc = load('wc', np.int8, (N,K))
    gpu_ws = load('ws', np.float32, (N,))
    gpu_acc = load('acc', np.int32, (M, N))
    gpu_y = load('float', np.float32, (M, N))
    gpu_half = load('half', np.float16, (M, N))
    fp16 = load('f16', np.float16, (M, N))
    # Reuse the established k32 oracle, not a newly reconstructed FP16 model.
    cpu_fp16 = me.mma_chain(a.astype(np.float64), w.astype(np.float64),
                            C0=np.broadcast_to(bias.astype(np.float64), (M, N)))
    probes = []
    scalar_encoding = True
    for row in (0, 1, 479, 480, M//2, M-1):
        top = max(abs(float(v)) for v in a[row]) or 1.0
        inv = np.float32(np.float32(127)/np.float32(top))
        scale = np.float32(np.float32(top)*np.float32(1/127))
        scalar_encoding &= scale.tobytes() == scales[row,0].tobytes()
        for k in range(K):
            ratio = float(np.float32(np.float32(a[row,k])*inv))
            scalar_encoding &= int(codes[row,k]) == max(-127,min(127,round(ratio)))
        for col in (0, 19, 20, 39, 40, 59, 60, 79):
            dot = sum(int(codes[row,k])*int(wc[k,col]) for k in range(K))
            probes.append({'row': row, 'col': col, 'scalar': dot,
                           'match': dot == int(acc[row,col]) == int(gpu_acc[row,col])})
    scalar_weights = True
    for col in range(N):
        top = max(abs(float(v)) for v in w[:,col])
        scale = np.float32(top/127 if top else 1.0)
        scalar_weights &= scale.tobytes() == ws[col].tobytes()
        for k in range(K):
            code = max(-127,min(127,round(float(w[k,col])/float(scale))))
            scalar_weights &= code == int(wc[k,col])
    deterministic = {name: sha(out/'run0'/f'{name}.bin') == sha(out/'run1'/f'{name}.bin')
                     for name in ('a', 'scale', 'wc', 'ws', 'acc', 'float', 'half', 'f16')}
    checks = {'codes': np.array_equal(gpu_codes, codes),
              'scales': np.array_equal(gpu_scale.view(np.uint32), scales.view(np.uint32)),
              'weight_codes': np.array_equal(gpu_wc, wc.T),
              'weight_scales': np.array_equal(gpu_ws.view(np.uint32), ws.view(np.uint32)),
              'integer': np.array_equal(gpu_acc, acc),
              'float32': np.array_equal(gpu_y.view(np.uint32), y.view(np.uint32)),
              'half': np.array_equal(gpu_half.view(np.uint16), half.view(np.uint16)),
              'fp16_oracle': np.array_equal(fp16, cpu_fp16),
              'scalar': all(p['match'] for p in probes),
              'scalar_activation_encoding': bool(scalar_encoding),
              'scalar_weight_encoding': bool(scalar_weights),
              'shaders_unchanged_during_run': all(sha(shader_dir/name)==digest for name,digest in shaders.items()),
              'deterministic_complete_writes': all(deterministic.values()),
              'finite': bool(np.isfinite(gpu_half).all()),
              'int32_safe': int(np.abs(acc).max()) < 2**31}
    isa = None
    if rga:
        subprocess.run([str(rga), '-s', 'dx12', '-c', 'gfx1101', '--cs-blob',
                        str(shader_dir/'dec0_q8_matmul.cso'), '--isa', str(out/'matmul')],
                       check=True, capture_output=True)
        path = out/'gfx1101_matmul_comp.isa'
        text = path.read_text()
        isa = {'shader_sha256': sha(shader_dir/'dec0_q8_matmul.cso'),
               'rga_sha256': sha(rga), 'target': 'gfx1101', 'isa_sha256': sha(path),
               'static_dot4_instructions': len(re.findall('v_dot4_i32_i8',text)),
               'fp16_matrix_instructions': len(re.findall(r'v_(?:pk_)?(?:mul|fma|mac)_f16|v_wmma',text))}
        checks['native_isa'] = isa['static_dot4_instructions']>0 and isa['fp16_matrix_instructions']==0
        checks['inspected_shader_is_executed_shader'] = isa['shader_sha256']==shaders['dec0_q8_matmul.cso']
    # Isolate reconstruction/accumulation rounding from quantization itself.
    exact = a.astype(np.float64) @ w.astype(np.float64) + bias.astype(np.float64)[None,:]
    quantized_exact = acc.astype(np.float64)*scales.astype(np.float64)*ws.astype(np.float64)[None,:]
    quantized_exact += bias.astype(np.float64)[None,:]
    result = {'checks': {k: bool(v) for k,v in checks.items()},
              'status': 'PASS' if all(checks.values()) else 'FAIL',
              'shape': [M,K,N], 'shaders': shaders,
              'isa': isa,
              'probe_sha256': sha(shader_dir/'Release/m_network_stage.exe'),
              'graph_sha256': sha(out/'graph.txt'),
              'script_sha256': sha(__file__),
              'inputs': {name: sha(out/f'{name}.bin') for name in ('tokens','weights','wc','ws','bias')},
              'gpu_vs_cpu_float': metrics(gpu_y, y),
              'gpu_vs_cpu_half': metrics(gpu_half, half),
              'int8_vs_fp16': metrics(gpu_half, fp16),
              'cpu_int8_vs_fp16': metrics(half, fp16),
              'fp16_vs_exact': metrics(fp16, exact),
              'quantization_only_vs_exact': metrics(quantized_exact, exact),
              'int8_output_rounding_vs_quantized_exact': metrics(gpu_half, quantized_exact),
              'accumulator_range': [int(acc.min()), int(acc.max())],
              'activation_scale_range': [float(scales.min()), float(scales.max())],
              'weight_scale_range': [float(ws.min()), float(ws.max())],
              'endpoint_codes': int((np.abs(codes.astype(np.int16))==127).sum()),
              'half_overflow': int(np.isinf(gpu_half).sum()),
              'e4m3_saturation_after_half': int((np.abs(gpu_half.astype(np.float32))>448).sum()),
              'zero_rows': int((np.abs(a).max(axis=1)==0).sum()),
              'activation_codes_differing': int((codes!=gpu_codes).sum()),
              'half_saturation_count': int((np.abs(gpu_half.astype(np.float32))>=65504).sum()),
              'probes': probes, 'deterministic': deterministic}
    (out/'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'checks': result['checks'],
                      'error': {k:v for k,v in result['int8_vs_fp16'].items() if not k.startswith('per_channel')},
                      'accumulator_range': result['accumulator_range']}, indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rga', type=Path, help='RGA executable; inspect and gate native gfx1101 ISA')
    args = parser.parse_args()
    sys.exit(run(args.fixture.resolve(), args.out.resolve(), args.rga))
