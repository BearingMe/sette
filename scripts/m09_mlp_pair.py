#!/usr/bin/env python3
"""EXP-027: complete enc1's INT8 MLP pair inside the existing real graph.

GELU/E4M3 and the norm2 residual remain floating-point. The down projection
replaces eight FP16 k32 accumulation steps with one exact K=256 integer dot,
two f32 rescale multiplies, a f32 residual add, and one f16 RNE store.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m09_mlp_up as up

ROOT = up.ROOT


def down_weights(source):
    record, _, _ = up.weights(source)
    raw = np.fromfile(Path(source) / 'weights.bin', '<f4')
    start = record['weights']['offsets']['w2']
    w = raw[start:start + 16384].reshape(256, 64)
    if not np.isfinite(w).all():
        raise ValueError('non-finite down-projection weights')
    return w


def raw_down_offset(base, k, c):
    n = up.gp(c)
    return (base + 512*(n//16) + 64*(n%8) + 16*((k%16)//4)
            + 8*((n//8)%2) + 4*((k%32)//16) + k%4)


def audit_layout(capture, source):
    P, *_ = up.lf.lo.read_layer(str(capture), 'enc1')
    bases = up.lf.lo.sm.layout(64, 2)['b1']
    w = down_weights(source)
    checks = []
    synthetic = np.arange(len(P.wbuf), dtype=np.int64)*73 + 19
    for s, base in enumerate(bases):
        base += 2112
        offsets = np.array([[raw_down_offset(base, k, c) for c in range(64)] for k in range(32)])
        table = up.lf.lo.sm.woff_table(base, 0, 512, 32, 64)[:, [up.gp(c) for c in range(64)]]
        checks.append(bool(np.array_equal(offsets, table) and np.unique(offsets).size == 2048
                           and np.array_equal(synthetic[offsets], synthetic[table])))
        checks.append(bool(np.array_equal(up.lf.lo.me.E4M3[P.wbuf[offsets]], w[s*32:(s+1)*32])))
    return {'pass': all(checks), 'checks': checks, 'down_chunk_bases': [v+2112 for v in bases]}


def cpu_down(g, w, residual):
    codes, scales = up.eq.quantize_rows_float(g)
    wc, ws, _ = up.eq.quantize_weights(w)
    acc = codes.astype(np.int64) @ wc.astype(np.int64)
    product = ((acc.astype(np.float32)*scales)*ws[None, :]).astype(np.float32)
    y = (product + np.asarray(residual, np.float32)).astype(np.float32)
    return codes, scales.ravel(), wc, ws, acc, product, y, y.astype('<f2')


def prepare(source, out, fp16_down=False):
    """Offline weights only; all up/GELU/down activations stay on the GPU."""
    source, out = Path(source), Path(out)
    up.prepare(source, out, 'int8-up')
    record = json.loads((out / 'fixture.json').read_text())
    M = record['geometry']['tokens']
    additions = [f'resource mlp_g {M*256*2}']
    dumps = [('mlp_g', M*256*2)]
    def stage(name, shader, params, groups, srvs, uavs):
        return ([f'stage {name}', f'shader {shader}', 'threads 64 1 1',
                 f'groups {groups} 1 1', f'params {params}']
                + [f'srv {i} {v}' for i, v in enumerate(srvs)]
                + [f'uav {i} {v}' for i, v in enumerate(uavs)])
    stages = stage('mlp-gelu', 'swin_mlp_gelu.cso', 'params-mlp.bin 160', (M+63)//64,
                   ['mlp_up', 'weights', 'x2'], ['mlp_g', 'scratch'])
    if fp16_down:
        stages += stage('mlp-down', 'swin_mlp_down_probe.cso', 'params-mlp.bin 160', (M+63)//64,
                        ['mlp_g', 'weights', 'x2'], ['x2b', 'scratch'])
    else:
        w = down_weights(source)
        wc, ws, _ = up.eq.quantize_weights(w)
        for name, data in [('down_wc', np.ascontiguousarray(wc.T).tobytes()),
                           ('down_ws', ws.astype('<f4').tobytes())]:
            (out / f'{name}.bin').write_bytes(data)
            additions.append(f'resource {name} {len(data)} {name}.bin')
        for name, size in [('down_a', M*256), ('down_as', M*4),
                           ('down_acc', M*64*4), ('down_float', M*64*4)]:
            additions.append(f'resource {name} {size}')
            dumps.append((name, size))
        (out / 'params-down-q8.bin').write_bytes(np.array([M, 256, 64, 0], '<u4').tobytes())
        stages += stage('mlp-down-quant', 'swin_q8_quant.cso', 'params-down-q8.bin 16', (M+63)//64,
                        ['mlp_g'], ['down_a', 'down_as'])
        stages += stage('mlp-down', 'swin_mlp_down_q8.cso', 'params-down-q8.bin 16', (M*4+63)//64,
                        ['down_a', 'down_wc', 'down_as', 'down_ws', 'x2'],
                        ['x2b', 'down_acc', 'down_float'])
    text = (out / 'graph.txt').read_text()
    start = text.index('stage mlp\n')
    end = text.index('\nstage ', start+1)
    text = text[:start] + '\n'.join(stages) + text[end:]
    text = text.replace('version 1\n', 'version 1\n' + '\n'.join(additions) + '\n', 1)
    for name, size in dumps:
        text += f'dump {name} 0 {size} dbg-{name}.bin\n'
    (out / 'graph.txt').write_text(text)
    record['mlp_mode'] = 'validation-fp16-down' if fp16_down else 'int8-pair'
    record['mlp_contract']['down'] = {'M': M, 'K': 256, 'N': 64, 'residual': 'x2 f16, f32 post-rescale addition'}
    resources, parsed, _ = up.suite._parse_graph(out / 'graph.txt')
    record['resources'] = [[v['name'], v['bytes']] for v in resources]
    record['stages'] = [{'name': s['name'], 'shader': next(w[1] for w in s['lines'] if w[0] == 'shader')}
                        for s in parsed]
    (out / 'fixture.json').write_text(json.dumps(record, indent=2))
    return out


def synthetic_gate(out):
    """K=256 GPU edge cases, including zero rows/columns and signed residuals."""
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(27)
    g = rng.normal(size=(8, 256)).astype('<f2')
    g[0] = 0; g[1] = 127; g[2] = -127
    g[3, :8] = [127, -127, .5, 1.5, 2.5, -.5, -1.5, -2.5]
    w = (rng.normal(size=(256, 64))*.2).astype('<f2').astype('<f4')
    w[:, 0] = 0; w[:, 1] = 1; w[:, 2] = -1
    residual = rng.normal(size=(8, 64)).astype('<f2')
    codes, scales, wc, ws, acc, _, y, half = cpu_down(g, w, residual)
    lines = ['version 1']
    for name, data in [('g', g.tobytes()), ('w', np.ascontiguousarray(wc.T).tobytes()),
                       ('ws', ws.astype('<f4').tobytes()), ('residual', residual.tobytes())]:
        (out / f'{name}.bin').write_bytes(data)
        lines.append(f'resource {name} {len(data)} {name}.bin')
    expected = {'codes': codes, 'scales': scales.astype('<f4'), 'acc': acc.astype('<i4'),
                'float': y, 'half': half}
    for name, data in expected.items():
        lines.append(f'resource {name} {data.nbytes}')
    (out / 'params.bin').write_bytes(np.array([8, 256, 64, 0], '<u4').tobytes())
    lines += ['stage quant', 'shader swin_q8_quant.cso', 'threads 64 1 1', 'groups 1 1 1',
              'params params.bin 16', 'srv 0 g', 'uav 0 codes', 'uav 1 scales',
              'stage down', 'shader swin_mlp_down_q8.cso', 'threads 64 1 1', 'groups 1 1 1',
              'params params.bin 16', 'srv 0 codes', 'srv 1 w', 'srv 2 scales', 'srv 3 ws',
              'srv 4 residual', 'uav 0 half', 'uav 1 acc', 'uav 2 float']
    for name, data in expected.items():
        lines.append(f'dump {name} 0 {data.nbytes} {name}.bin')
    (out / 'graph.txt').write_text('\n'.join(lines) + '\n')
    first = up.run_graph(out, out / 'results')
    repeat = up.run_graph(out, out / 'repeat', 1)
    checks = {name: (first / f'{name}.bin').read_bytes() == np.ascontiguousarray(data).tobytes()
        and up.eq.sha(first / f'{name}.bin') == up.eq.sha(repeat / f'{name}.bin')
        for name, data in expected.items()}
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks}


def validate(source, out, capture):
    out.mkdir(parents=True, exist_ok=False)
    synthetic = synthetic_gate(out / 'synthetic-down')
    if synthetic['status'] != 'PASS':
        raise RuntimeError('synthetic down gate failed')
    # Reuse the full established up-projection gate before validating the pair.
    first = up.validate(source, out / 'up-validation', capture)
    if first['status'] != 'PASS':
        raise RuntimeError('up-projection gate failed')
    layout = audit_layout(capture, source)
    if not layout['pass']:
        raise RuntimeError('down-layout gate failed')
    fixture = prepare(source, out / 'pair')
    gpu = up.run_graph(fixture, out / 'pair-results')
    repeat = up.run_graph(fixture, out / 'pair-repeat', 1)
    split = prepare(source, out / 'fp16-down', fp16_down=True)
    split_result = up.run_graph(split, out / 'fp16-down-results')
    checks = {'up_gate': True, 'down_layout': layout['pass']}
    for p in gpu.glob('*.bin'):
        checks['repeat_' + p.name] = up.eq.sha(p) == up.eq.sha(repeat / p.name)
    for role in ('out-out', 'out-merge', 'dbg-x2b'):
        checks['original_gelu_down_split_' + role] = up.eq.sha(split_result / f'{role}.bin') == up.eq.sha(
            out / 'up-validation/int8-up-results' / f'{role}.bin')
    checks['up_handoff'] = up.eq.sha(gpu / 'dbg-mlp_up.bin') == up.eq.sha(
        out / 'up-validation/int8-up-results/dbg-mlp_up.bin')
    checks['residual_unchanged'] = up.eq.sha(gpu / 'dbg-x2.bin') == up.eq.sha(
        out / 'up-validation/int8-up-results/dbg-x2.bin')
    M = up.weights(source)[0]['geometry']['tokens']
    z = np.fromfile(gpu / 'dbg-mlp_up.bin', '<f2').reshape(M, 256)
    g = np.fromfile(gpu / 'dbg-mlp_g.bin', '<f2').reshape(M, 256)
    cpu_g = up.lf.me.q8(up.lf.me.gelu_poly(z)).astype('<f2')
    checks['original_gelu_cpu_exact'] = g.tobytes() == cpu_g.tobytes()
    residual = np.fromfile(gpu / 'dbg-x2.bin', '<f2').reshape(M, 64)
    w = down_weights(source)
    codes, scales, wc, ws, acc, product, y, half = cpu_down(cpu_g, w, residual)
    for name, want in [('down_a', codes), ('down_as', scales.astype('<f4')),
                       ('down_acc', acc.astype('<i4')), ('down_float', y), ('x2b', half)]:
        checks['cpu_' + name] = (gpu / f'dbg-{name}.bin').read_bytes() == np.ascontiguousarray(want).tobytes()
    checks['weight_codes'] = (fixture / 'down_wc.bin').read_bytes() == np.ascontiguousarray(wc.T).tobytes()
    checks['weight_scales'] = (fixture / 'down_ws.bin').read_bytes() == ws.astype('<f4').tobytes()
    checks['scalar_weights'] = all(np.array_equal(up.scalar_weight_column(w[:, col])[0], wc[:, col])
        and up.scalar_weight_column(w[:, col])[1].tobytes() == ws[col].tobytes() for col in range(64))
    rng = np.random.default_rng(27)
    probes = []
    for row, col in [(0, 0), (M-1, 63)] + [(int(rng.integers(M)), int(rng.integers(64))) for _ in range(46)]:
        c, s = up.scalar_row(cpu_g[row])
        total = up.eq.scalar_dot(c, wc[:, col])
        value = np.float32(np.float32(np.float32(total)*s)*ws[col]) + np.float32(residual[row, col])
        probes.append(bool(np.array_equal(c, codes[row]) and s.tobytes() == scales[row].tobytes()
            and total == acc[row, col] and value.tobytes() == y[row, col].tobytes()
            and np.float16(value).tobytes() == half[row, col].tobytes()))
    checks['scalar_down_probes'] = all(probes)
    checks['int32_safe'] = int(np.abs(acc).max()) <= 256*127*127 < 2**31
    # Same G and residual: the original k32 CPU oracle must reproduce the FP16 probe.
    fp_down = up.lf.me.mma_chain(cpu_g.astype(np.float64), w.astype(np.float64),
                               C0=residual.astype(np.float64)).astype('<f2')
    checks['fp16_down_cpu_exact'] = fp_down.tobytes() == (split_result / 'dbg-x2b.bin').read_bytes()
    safety = {}
    for name, values in [('gelu', g), ('x2b', half), ('up', z)]:
        safety[name] = {'non_finite': int((~np.isfinite(values)).sum()),
                        'saturated': int((np.abs(values) >= 65504).sum())}
        checks[name + '_finite_unsaturated'] = all(v == 0 for v in safety[name].values())
    metrics = {'up': first['metrics']['projection'], 'down_vs_same_input_fp16': up.eq.compare(half, fp_down)}
    for role in ('out', 'merge'):
        baseline = np.fromfile(out / 'up-validation/probe-results' / f'out-{role}.bin', np.uint8)
        up_only = np.fromfile(out / 'up-validation/int8-up-results' / f'out-{role}.bin', np.uint8)
        pair = np.fromfile(gpu / f'out-{role}.bin', np.uint8)
        ref = np.fromfile(source / f'ref-{role}.bin', np.uint8)
        metrics[role] = {'hybrid_vs_fp16': up.eq._field(baseline, pair),
            'pair_vs_up_only': up.eq._field(up_only, pair), 'fp16_vs_capture': up.eq._field(ref, baseline),
            'hybrid_vs_capture': up.eq._field(ref, pair)}
        checks[role + '_finite_unsaturated'] = bool(np.isfinite(up.eq.E4M3[pair]).all()
                                                   and not np.any((pair & 127) == 126))
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'mode': 'int8-pair', 'hybrid_results': 'pair-results', 'layout': layout, 'synthetic': synthetic,
        'metrics': metrics, 'safety': safety, 'activation': up.eq.code_metrics(codes, scales),
        'accumulator_max_abs': int(np.abs(acc).max()), 'scalar_probes': len(probes),
        'shaders': {p.name: up.eq.sha(p) for p in (ROOT / 'build/native').glob('*.cso')},
        'probe_sha256': up.eq.sha(ROOT / 'build/native/Release/m_network_stage.exe')}
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'failed': [k for k,v in checks.items() if not v],
                      'metrics': metrics}, indent=2))
    return report


def audit_frame(out, frame, fp16, q, dec0, up_only, rga):
    report = up.audit_frame(out, frame, fp16, q, dec0, rga)
    report['status'] = 'PENDING_DOWN_ISA'
    (out / 'integration.json').write_text(json.dumps(report, indent=2))
    shader = ROOT / 'build/native/swin_mlp_down_q8.cso'
    digest = up.eq.sha(shader)
    done = subprocess.run([str(rga), '-s', 'dx12', '-c', 'gfx1101', '--cs-blob', str(shader),
                           '--isa', str(out / 'executed-mlp-down')], capture_output=True, text=True)
    (out / 'rga-down.log').write_text(done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError('down ISA inspection failed')
    path = out / 'gfx1101_executed-mlp-down_comp.isa'
    text = path.read_text()
    report['down_isa'] = {'shader_sha256': digest, 'isa_sha256': up.eq.sha(path),
        'dot4': len(re.findall('v_dot4_i32_i8', text)),
        'fp16_matrix': len(re.findall(r'v_(?:pk_)?(?:mul|fma|mac)_f16|v_wmma', text))}
    checks = report['checks']
    checks['down_native_int8'] = report['down_isa']['dot4'] > 0 and report['down_isa']['fp16_matrix'] == 0
    checks['down_inspected_executed_hash'] = digest == up.eq.sha(shader)
    previous = json.loads((up_only / 'b6.json').read_text())
    previous_hash = up.eq.sha(up_only / 'downsample-native/results/out-out.bin')
    checks['up_only_control_unchanged'] = previous_hash == previous['comparisons']['native']['frame_sha256'] == (
        '00a438dc2d8c74be5ccf514c5846a8484ce91cfbae5b722409d45bd6c663949c')
    report['controls']['mlp_up_only'] = {'frame_sha256': previous_hash,
                                       'metrics': previous['comparisons']['native']}
    report['status'] = 'PASS' if all(checks.values()) else 'FAIL'
    (out / 'integration.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'down_isa': report['down_isa']}, indent=2))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--capture', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frame-run', type=Path)
    ap.add_argument('--fp16-run', type=Path)
    ap.add_argument('--enc1q-run', type=Path)
    ap.add_argument('--dec0-run', type=Path)
    ap.add_argument('--up-run', type=Path)
    ap.add_argument('--rga', type=Path)
    a = ap.parse_args()
    if a.frame_run:
        if not all([a.fp16_run, a.enc1q_run, a.dec0_run, a.up_run, a.rga]):
            ap.error('frame audit requires FP16/Q/dec0/up-only runs and RGA')
        result = audit_frame(a.out, a.frame_run, a.fp16_run, a.enc1q_run, a.dec0_run, a.up_run, a.rga)
    else:
        if not (a.source and a.capture):
            ap.error('validation requires --source and --capture')
        result = validate(a.source.resolve(), a.out.resolve(), a.capture.resolve())
    sys.exit(0 if result['status'] == 'PASS' else 1)
