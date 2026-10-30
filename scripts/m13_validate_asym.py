#!/usr/bin/env python3
"""M13 asymmetric-path validation: executed GPU vs independent CPU arithmetic.

The asymmetric quantizer and its two consumers are new arithmetic, so they get
the same treatment as the accepted INT8 path: every exported element is compared
with a CPU reference that re-derives the contract from the GPU's own inputs.

For each checked layer:

  mlp-down   quantizer codes / scales / offsets, then the GEMM's int32
             accumulator, f32 epilogue and f16 store, against `i8_g`, the
             weight codes, the offline column sums and the residual;
  attention  P codes / scales / offsets and the PV int32 accumulator, f32
             epilogue and f16 store, against the probed P and V code buffers.

The flags are also inspected with RGA: signed `dot4` must be present and no
FP16 matrix instruction may appear.

  m13_validate_asym.py --layers VARIANT --out DIR [--labels enc1,dec1] [--rga EXE]
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import m_layer_suite as suite

F16, F32, F64 = np.float16, np.float32, np.float64
ONE_255 = F32(1.0 / 255.0)


def sha(path):
    return __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()


def f16_bits(values):
    return np.asarray(values, F16).view(np.uint16)


def quantize_asym(a):
    """The GPU contract, operation for operation, in f32."""
    a = np.asarray(a, F16)
    x = a.astype(F64)
    lo = x.min(axis=1).astype(F32)
    hi = x.max(axis=1).astype(F32)
    rng = (hi - lo).astype(F32)
    rng = np.where(rng > 0, rng, F32(1.0)).astype(F32)
    scale = (rng * ONE_255).astype(F32)
    invf = (F32(1.0) / scale).astype(F32)
    residual = (F64(1.0) - invf.astype(F64) * scale.astype(F64))
    inv = (invf.astype(F64) + residual / scale.astype(F64)).astype(F32)
    ratio = ((a.astype(F32) - lo[:, None]) * inv[:, None]).astype(F32)
    codes = (np.rint(ratio).astype(np.int32) - 128)
    codes = np.clip(codes, -128, 127).astype(np.int8)
    offset = (lo + (F32(128.0) * scale)).astype(F32)
    return codes, scale, offset


def check_mlp(directory, results, label, tag):
    record = json.loads((Path(suite.fixture(str(directory), label)) /
                         'fixture.json').read_text())
    op = next(o for o in record['int8_operations'] if o['name'] == 'mlp-down')
    M, K, N = op['M'], op['K'], op['N']
    read = lambda name, dtype='<f2', shape=None: (
        lambda a: a.reshape(shape) if shape else a)(
        np.fromfile(Path(results) / f'{tag}-dbg-{name}.bin', dtype))
    A = read('i8_g', '<f2', (M, K))
    codes = read('i8_mlp-down_a', np.uint8, (M, K))
    scales = read('i8_mlp-down_as', '<f4', (M,))
    offsets = read('i8_mlp-down_off', '<f4', (M,))
    wsum = read('i8_mlp-down_wsum', '<i4', (N,))
    acc = read('i8_mlp-down_acc', '<i4', (M, N))
    flt = read('i8_mlp-down_float', '<f4', (M, N))
    out = read('x2b', '<f2', (M, N))
    residual = read('x2', '<f2', (M, N))
    graph = Path(suite.fixture(str(directory), label)) / 'graph.txt'
    resources = {w[1]: w[3] for w in (line.split() for line in graph.read_text().splitlines())
                 if w and w[0] == 'resource' and len(w) > 3}
    wcodes = np.fromfile(resources[op['weight_codes']], np.uint8)
    if wcodes.size != K * N:
        raise SystemExit(f'{label}: weight codes {wcodes.size} != {K * N}')
    wcodes = wcodes.reshape(N, K).T.astype(np.int8)  # stored column-major

    want_codes, want_scale, want_offset = quantize_asym(A)
    checks = {
        'codes': codes.tobytes() == want_codes.astype(np.uint8).tobytes(),
        'scales': scales.tobytes() == want_scale.tobytes(),
        'offsets': offsets.tobytes() == want_offset.tobytes(),
        'weight_column_sums': True,
    }
    # Independent column sums from the same codes the GPU read.
    want_wsum = wcodes.astype(np.int64).sum(axis=0).astype(np.int32)
    checks['weight_column_sums'] = wsum.tobytes() == want_wsum.tobytes()

    want_acc = codes.astype(np.int8).astype(np.int64) @ wcodes.astype(np.int64)
    checks['int32_accumulator'] = acc.tobytes() == want_acc.astype(np.int32).tobytes()
    # Epilogue: ((acc*scale + offset*wsum) * wscale) + residual, f32, one f16 store.
    t = (acc.astype(F32) * scales[:, None]).astype(F32)
    corr = (offsets[:, None] * wsum.astype(F32)[None, :]).astype(F32)
    wscales = np.fromfile(resources[op['weight_scales']], '<f4').reshape(N)
    y = ((t + corr) * wscales[None, :]).astype(F32)
    y = (y + residual.astype(F32)).astype(F32)
    checks['f32_epilogue'] = flt.tobytes() == y.tobytes()
    checks['f16_store'] = f16_bits(out).tobytes() == f16_bits(y).tobytes()
    checks['finite'] = bool(np.isfinite(out).all())
    checks['saturated'] = int((np.abs(out.astype(F64)) >= 65504).sum()) == 0
    return {'shape': [M, K, N], 'checks': checks,
            'max_abs_diff_f16': float(np.max(np.abs(out.astype(F64) - y.astype(F64))))}


def check_pv(directory, results, label, tag):
    record = json.loads((Path(suite.fixture(str(directory), label)) /
                         'fixture.json').read_text())
    B = int(record['int8_attention']['instances'])
    read = lambda name, dtype='<f2', shape=None: (
        lambda a: a.reshape(shape) if shape else a)(
        np.fromfile(Path(results) / f'{tag}-dbg-{name}.bin', dtype))
    P = read('att-prob', '<f2', (B, 16, 16))
    pc = read('att-pc', np.uint8, (B, 16, 16))
    ps = read('att-ps', '<f4', (B, 16))
    po = read('att-po', '<f4', (B, 16))
    vc = read('att-vc', np.uint8, (B, 32, 16))
    vs = read('att-vs', '<f4', (B, 32))
    pacc = read('att-pacc', '<i4', (B, 16, 32))
    value32 = read('att-value32', '<f4', (B, 16, 32))
    value16 = read('att-value16', '<f2', (B, 16, 32))

    want_pc, want_ps, want_po = quantize_asym(P.reshape(B * 16, 16))
    checks = {
        'codes': pc.reshape(-1, 16).tobytes() == want_pc.astype(np.uint8).tobytes(),
        'scales': ps.reshape(-1).tobytes() == want_ps.tobytes(),
        'offsets': po.reshape(-1).tobytes() == want_po.tobytes(),
    }
    signed_v = vc.view(np.int8).astype(np.int64)
    want_acc = np.einsum('brk,bck->brc', pc.view(np.int8).astype(np.int64), signed_v)
    checks['int32_accumulator'] = pacc.tobytes() == want_acc.astype(np.int32).tobytes()
    vsum = signed_v.sum(axis=2).astype(F32)
    t = (pacc.astype(F32) * ps[:, :, None]).astype(F32)
    corr = (po[:, :, None] * vsum[:, None, :]).astype(F32)
    y = ((t + corr) * vs[:, None, :]).astype(F32)
    checks['f32_epilogue'] = value32.tobytes() == y.tobytes()
    checks['f16_store'] = f16_bits(value16).tobytes() == f16_bits(y).tobytes()
    checks['finite'] = bool(np.isfinite(value16).all())
    checks['saturated'] = int((np.abs(value16.astype(F64)) >= 65504).sum()) == 0
    return {'instances': B, 'checks': checks,
            'max_abs_diff_f16': float(np.max(np.abs(value16.astype(F64) -
                                                    y.astype(F64))))}


def inspect_isa(shaders, out, rga):
    report = {}
    for name in shaders:
        shader = REPO / 'build/native' / name
        stem = Path(name).stem
        done = subprocess.run([str(rga), '-s', 'dx12', '-c', 'gfx1101',
                               '--cs-blob', str(shader), '--isa', str(out / stem)],
                              capture_output=True, text=True)
        (out / f'{stem}-rga.log').write_text(done.stdout + done.stderr)
        if done.returncode:
            raise SystemExit(f'RGA failed for {name}: {done.stderr[-400:]}')
        candidates = sorted(Path(out).glob(f'gfx1101_{stem}*.isa'))
        if not candidates:
            raise SystemExit(f'no ISA for {name}')
        text = '\n'.join(p.read_text(errors='replace') for p in candidates)
        # The quantizer performs no matrix product; the two consumers must
        # show signed dot4.
        need_dot4 = 'quant' not in stem
        report[name] = {'sha256': sha(shader), 'need_dot4': need_dot4,
                        'dot4': len(re.findall('v_dot4_i32_i8', text)),
                        'fp16_matrix': len(re.findall(
                            r'v_(?:pk_)?(?:mul|fma|mac)_f16|v_wmma', text))}
    report['status'] = ('PASS' if all((v['dot4'] or not v['need_dot4'])
                                      and not v['fp16_matrix']
                                      for k, v in report.items() if k != 'status')
                        else 'FAIL')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layers', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--labels', default='enc1,dec1')
    ap.add_argument('--rga', type=Path,
                    default=Path(os.environ.get('LOCALAPPDATA', '.')) /
                            'Temp/opencode/rga-2.15/rga.exe')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    chain = a.out / 'chain'
    suite.build_chain(str(a.layers), str(chain), cumulative_diagnostics=True)
    keep = set()
    for label in a.labels.split(','):
        tag = re.sub(r'[^A-Za-z0-9]+', '-', label)
        for name in ('i8_g', 'i8_mlp-down_a', 'i8_mlp-down_as', 'i8_mlp-down_off',
                     'i8_mlp-down_acc', 'i8_mlp-down_float', 'i8_mlp-down_wsum',
                     'x2b', 'x2', 'att-prob', 'att-pc', 'att-ps', 'att-po',
                     'att-vc', 'att-vs', 'att-pacc', 'att-value32', 'att-value16'):
            keep.add(f'{tag}-dbg-{name}.bin')
    lines = (chain / 'graph.txt').read_text().splitlines()
    kept = [line for line in lines
            if not line.startswith('dump ') or line.split()[4] in keep]
    (chain / 'graph.txt').write_text('\n'.join(kept) + '\n')
    results = a.out / 'results'
    done = subprocess.run([str(REPO / 'build/native/Release/m_network_stage.exe'),
                           '--graph', str(chain / 'graph.txt'), '--out', str(results),
                           '--shaders', str(REPO / 'build/native'), '--seed', '0'],
                          capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'chain failed: {done.stdout[-800:]}{done.stderr[-800:]}')
    (results / 'execution.log').write_text(done.stdout + done.stderr)

    report = {'layers': str(a.layers), 'checked': {}, 'isa': {},
              'chain': str(chain), 'results': str(results)}
    ok = True
    for label in a.labels.split(','):
        tag = re.sub(r'[^A-Za-z0-9]+', '-', label)
        entry = {'mlp_down': check_mlp(a.layers, results, label, tag),
                 'pv': check_pv(a.layers, results, label, tag)}
        entry['status'] = ('PASS' if all(entry['mlp_down']['checks'].values())
                           and all(entry['pv']['checks'].values()) else 'FAIL')
        ok = ok and entry['status'] == 'PASS'
        report['checked'][label] = entry
        print(label, entry['status'], entry['mlp_down']['checks'],
              entry['pv']['checks'])
    report['isa'] = inspect_isa(['swin_q8_quant_asym.cso',
                                 'swin_mlp_down_q8_asym.cso',
                                 'swin_attention_pv_asym.cso'], a.out, a.rga)
    ok = ok and report['isa']['status'] == 'PASS'
    report['status'] = 'PASS' if ok else 'FAIL'
    (a.out / 'validation.json').write_text(json.dumps(report, indent=2))
    print('isa', {k: v for k, v in report['isa'].items() if k != 'status'})
    print('status:', report['status'])
    print('out:', a.out)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
