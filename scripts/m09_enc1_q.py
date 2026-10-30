#!/usr/bin/env python3
"""M09: enc1's Q projection in native INT8, validated against an independent CPU oracle.

The experiment replaces exactly one dispatch in the enc1 graph (I4) and then
compares, in order:

  Q      the projection itself: GPU FP16 vs GPU INT8 vs CPU INT8 vs CPU FP16 chain
  layer  enc1 with the INT8 Q: attention output, x1, the e4m3 output field, merge
  chain  the ten-layer graph with enc1's Q replaced (fp16 vs int8-q)
  frame  the B6 static frame with the hybrid graph (optional)

Arithmetic contract (I1/I2, EXP-023):

  A = h1  f16 [tokens][K], every value exactly E4M3-representable
  W = weights[pOffQ]  f32 [K][N], E4M3-widened
  Y = q   f16 [tokens][N]

  activation encoding   per row:   inv = f32(127)/f32(max|a|)   (correctly rounded)
                        ratio = f32(f32(a) * inv);  code = rint_half_even(ratio)
                        scale = f32(f32(max) * f32(1/127))
  weight encoding       m_int8.quantize(W, 'per_output_channel'): codes [K][N],
                        scale[n] = f32(max_k |W[k][n]| / 127)
  product               int32 dot4 over K (exact), y = f32(f32(acc*scale_a) * scale_w)
  output                one f16 RNE conversion

`quantize_rows_float` below reproduces the GPU encoder in the *same* f32
arithmetic, so a code-level comparison is meaningful rather than approximate;
EXP-023 records the two encoders that failed that bar (an f32 quotient with a
1-ulp reciprocal, and a f64 quotient with the device's 1-ulp f64 divider).

usage:
  m09_enc1_q.py run --out DIR --layers LAYER_DIR [--capture DIR] [--frame]
  m09_enc1_q.py oracle --h1 FILE --weights FILE --weight-offset N [--tokens N]
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

import m_conformance as mc               # noqa: E402
import m_int8 as q8                      # noqa: E402
import m_layer_fixture as lf             # noqa: E402
import m_layer_oracle as lo              # noqa: E402
import m_layer_suite as suite            # noqa: E402
import m_model as mm                     # noqa: E402
import model_enc3 as me                  # noqa: E402

F16, F32, F64 = np.float16, np.float32, np.float64
QMAX = 127


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- encoders

def quantize_rows_float(x):
    """The GPU encoder, step for step: f32 inverse, f32 product, RNE, clamp.

    `m_int8.quantize_rows` uses the same rules but forms the product in f64 with
    an f32 inverse; that is a different rounding of the same contract, and
    EXP-023 measures the difference rather than assuming it away.
    """
    x32 = np.asarray(x, F16).astype(F32)
    top = np.abs(x32).max(axis=1, keepdims=True)
    top = np.where(top > 0, top, np.float32(1.0)).astype(F32)
    inv = (np.float32(QMAX) / top).astype(F32)          # correctly rounded quotient
    ratio = (x32 * inv).astype(F32)
    codes = np.clip(np.rint(ratio), -QMAX, QMAX).astype(np.int8)
    scale = (top * np.float32(1.0 / QMAX)).astype(F32)
    return codes, scale


def quantize_weights(W):
    W = np.asarray(W, F64)
    enc = q8.quantize(W, 'per_output_channel')
    return enc['codes'], np.asarray(enc['scales'], F32), enc


def gemm_int8(codes, row_scale, wcodes, wscale):
    """The kernel's arithmetic: exact int64 reduction, f32 epilogue, f16 store."""
    acc = codes.astype(np.int64) @ wcodes.astype(np.int64)
    rows = np.asarray(row_scale, F32).reshape(-1, 1)
    y = (acc.astype(F32) * rows) * np.asarray(wscale, F32)[None, :]
    return y.astype(F16), acc


def scalar_dot(a_codes, w_codes):
    """Independent scalar integer dot product, one multiply-add at a time."""
    total = 0
    for a, w in zip(a_codes, w_codes):
        total += int(a) * int(w)
    return total


# ---------------------------------------------------------------- metrics

def compare(got, want, scale_peak=None):
    got = np.asarray(got, F64)
    want = np.asarray(want, F64)
    diff = got - want
    norm = float(np.linalg.norm(want)) or 1.0
    peak = float(scale_peak if scale_peak is not None else np.abs(want).max()) or 1.0
    return {'samples': int(diff.size),
            'identical_f16': float((np.asarray(got, F16) == np.asarray(want, F16)).mean()),
            'relative_rms': float(np.sqrt((diff ** 2).mean()) / np.sqrt((want ** 2).mean()))
            if np.any(want) else 0.0,
            'relative_frobenius': float(np.linalg.norm(diff) / norm),
            'rms': float(np.sqrt((diff ** 2).mean())),
            'mean_abs': float(np.abs(diff).mean()),
            'max_abs': float(np.abs(diff).max()),
            'max_relative_to_peak': float(np.abs(diff).max() / peak),
            'signed_mean': float(diff.mean()),
            'finite': bool(np.isfinite(got).all())}


def per_channel_error(got, want):
    got, want = np.asarray(got, F64), np.asarray(want, F64)
    diff = np.abs(got - want)
    return {'mean_abs_per_channel': [float(v) for v in diff.mean(axis=0)],
            'max_abs_per_channel': [float(v) for v in diff.max(axis=0)]}


def code_stats(got, want):
    got, want = np.asarray(got, np.int8), np.asarray(want, np.int8)
    delta = got.astype(np.int16) - want.astype(np.int16)
    return {'samples': int(got.size), 'identical': float((delta == 0).mean()),
            'differing': int((delta != 0).sum()),
            'max_delta': int(np.abs(delta).max())}


def code_metrics(codes, scale):
    """Saturation and coverage of one encoding, from the codes themselves."""
    return {'samples': int(codes.size),
            'at_positive_endpoint': int((codes == QMAX).sum()),
            'at_negative_endpoint': int((codes == -QMAX).sum()),
            'zero_codes': int((codes == 0).sum()),
            'distinct': int(np.unique(codes).size),
            'scale_min': float(np.min(scale)), 'scale_max': float(np.max(scale)),
            'zero_scale_rows': int((np.asarray(scale) == 0).sum())}


# ---------------------------------------------------------------- paths

def run_probe(probe, graph_dir, out, shaders):
    done = subprocess.run([str(probe), '--graph', str(Path(graph_dir) / 'graph.txt'),
                           '--out', str(out), '--shaders', str(shaders), '--seed', '0'],
                          capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'{graph_dir}: {done.stdout[-600:]}{done.stderr[-600:]}')
    return Path(out)


def load_layer(fixture_dir, results):
    rec = json.loads((Path(fixture_dir) / 'fixture.json').read_text())
    tokens = rec['geometry']['tokens']
    C = rec['geometry']['C']
    N = 32 * rec['geometry']['heads']
    W = np.fromfile(Path(fixture_dir) / 'weights.bin', F32)
    Wq = W[rec['weights']['offsets']['q']:][:C * N].reshape(C, N)
    out = {'record': rec, 'dir': Path(fixture_dir), 'tokens': tokens, 'C': C, 'N': N,
           'W': Wq.astype(F64)}
    for role in ('h1', 'q', 'o', 'x1'):
        path = Path(results) / f'dbg-{role}.bin'
        if path.is_file():
            out[role] = np.fromfile(path, F16).reshape(tokens, C if role in ('h1', 'x1') else N)
    for role in ('q8a', 'q8as'):
        path = Path(results) / f'dbg-{role}.bin'
        if path.is_file():
            raw = np.fromfile(path, np.uint8 if role == 'q8a' else F32)
            out[role] = (raw.reshape(tokens, C).view(np.int8) if role == 'q8a'
                         else raw.reshape(tokens))
    return out


def q_report(fp16, int8):
    """Q-only comparison: FP16 vs INT8, GPU vs CPU, codes and scales."""
    if not (np.array_equal(fp16['h1'].view(np.uint16), int8['h1'].view(np.uint16))):
        raise SystemExit('the two runs disagree on h1; the INT8 change is not isolated')
    tokens, C, N = fp16['tokens'], fp16['C'], fp16['N']
    W = fp16['W']
    report = {'h1': {'identical_between_modes': True, 'shape': [tokens, C]},
              'weight': {}, 'activation': {}, 'arithmetic': {}, 'output': {}}

    wcodes, wscale, enc = quantize_weights(W)
    report['weight'] = {'source_sha256': hashlib.sha256(W.astype(F32).tobytes()).hexdigest(),
                        'scope': 'per_output_channel',
                        'metrics': q8.metrics(W, enc)}
    # weight codes as the graph holds them: [N][K] little-endian words
    graph_w = np.fromfile(int8['dir'] / 'q8w.bin', np.uint8)
    graph_w = graph_w.reshape(N, C).view(np.int8)
    report['weight']['codes_vs_cpu'] = code_stats(graph_w, np.ascontiguousarray(wcodes.T))
    graph_ws = np.fromfile(int8['dir'] / 'q8ws.bin', F32)
    report['weight']['scales_identical'] = bool(np.array_equal(graph_ws, wscale))
    report['weight']['scale_min'] = float(np.min(graph_ws))
    report['weight']['scale_max'] = float(np.max(graph_ws))

    # activation encoding: the GPU's dumped codes against the CPU encoder
    cpu_codes, cpu_scale = quantize_rows_float(int8['h1'])
    report['activation'] = {
        'gpu_vs_cpu_codes': code_stats(int8['q8a'], cpu_codes),
        'gpu_vs_cpu_scales': bool(np.array_equal(int8['q8as'], cpu_scale.ravel())),
        'cpu_encoding': code_metrics(cpu_codes, cpu_scale.ravel()),
        'max_abs_residual': float(np.abs(
            np.asarray(int8['h1'], F64) - cpu_codes.astype(F64) * cpu_scale).max()),
        'half_step': float(np.abs(cpu_scale).max() / 2),
    }

    # the two CPU products: from the GPU's codes (isolates the GEMM) and from the
    # CPU's own codes (the end-to-end reference)
    y_gpu_codes, acc_gpu = gemm_int8(int8['q8a'], int8['q8as'], wcodes, wscale)
    y_cpu_codes, acc_cpu = gemm_int8(cpu_codes, cpu_scale.ravel(), wcodes, wscale)
    fp16_chain = me.mma_chain(np.asarray(fp16['h1'], F64), W)
    report['arithmetic'] = {
        'gpu_codes_to_cpu_gemm_vs_gpu_q': compare(y_gpu_codes, int8['q']),
        'cpu_codes_to_cpu_gemm_vs_gpu_q': compare(y_cpu_codes, int8['q']),
        'fp16_gpu_vs_cpu_chain': compare(fp16['q'], fp16_chain),
        'scalar_dot_probes': [],
        'accumulator_max_abs': int(np.abs(acc_cpu).max()),
        'accumulator_int32_safe': bool(np.abs(acc_cpu).max() < 2 ** 31),
    }
    rng = np.random.default_rng(0)
    for row, col in [(int(rng.integers(tokens)), int(rng.integers(N))) for _ in range(8)]:
        exact = scalar_dot(cpu_codes[row], np.ascontiguousarray(wcodes.T)[col])
        report['arithmetic']['scalar_dot_probes'].append(
            {'row': row, 'col': col, 'scalar': exact, 'matmul': int(acc_cpu[row, col]),
             'match': exact == int(acc_cpu[row, col])})

    fp16_peak = float(np.abs(np.asarray(fp16['q'], F64)).max())
    report['output'] = {
        'fp16_vs_int8': {**compare(int8['q'], fp16['q'], scale_peak=fp16_peak),
                         **per_channel_error(int8['q'], fp16['q'])},
        'cpu_int8_vs_fp16': compare(y_cpu_codes, fp16['q'], scale_peak=fp16_peak),
        'fp16_peak': fp16_peak,
        'q_absmax': float(np.abs(np.asarray(int8['q'], F64)).max()),
        'q_shape': list(np.asarray(int8['q']).shape),
        'q_format': 'f16, tok*N + n',
    }
    return report


E4M3 = np.array([mc.e4m3_value(int(c)) for c in range(256)], F64)


def _field(a, b):
    """e4m3 code agreement plus the error it stands for after decoding.

    Code deltas are not proportional to error: near zero an e4m3 step is
    2^-9, so a small absolute difference moves several codes while a large
    value moves none.  Both readings are reported rather than the flattering
    one.
    """
    delta = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    da, db = E4M3[a], E4M3[b]
    finite = np.isfinite(da) & np.isfinite(db)
    diff = db[finite] - da[finite]
    return {'samples': int(a.size), 'identical': float((np.asarray(a) == np.asarray(b)).mean()),
            'differing': int((np.asarray(a) != np.asarray(b)).sum()),
            'max_code_delta': int(delta.max()),
            'mean_code_delta': float(delta.mean()),
            'finite_samples': int(finite.sum()),
            'decoded_relative_rms': float(np.sqrt((diff ** 2).mean())
                                          / np.sqrt((da[finite] ** 2).mean())),
            'decoded_max_abs': float(np.abs(diff).max()),
            'decoded_signed_mean': float(diff.mean())}


def layer_report(fp16_results, int8_results, fixture_dir):
    out = {}
    for role in ('o', 'x1'):
        out[role] = compare(np.fromfile(int8_results / f'dbg-{role}.bin', F16),
                            np.fromfile(fp16_results / f'dbg-{role}.bin', F16))
    for role in ('out', 'merge'):
        a = np.fromfile(int8_results / f'out-{role}.bin', np.uint8)
        b = np.fromfile(fp16_results / f'out-{role}.bin', np.uint8)
        ref = np.fromfile(Path(fixture_dir) / f'ref-{role}.bin', np.uint8)
        out[role] = {'hybrid_vs_fp16': _field(a, b),
                     'fp16_vs_capture': _field(b, ref),
                     'hybrid_vs_capture': _field(a, ref)}
    return out


def chain_report(layers, out, enc1_int8, probe, shaders, translated_suite=None):
    results = {}
    for mode in ('fp16', 'int8-q'):
        chain_dir = out / f'chain-{mode}'
        info = suite.build_chain(str(layers), str(chain_dir),
                                 q_int8=mode == 'int8-q', enc1_int8=enc1_int8)
        res = run_probe(probe, chain_dir, out / f'chain-{mode}-results', shaders)
        repeat = run_probe(probe, chain_dir, out / f'chain-{mode}-results-repeat', shaders)
        dec1 = np.fromfile(res / 'out-dec1.bin', np.uint8)
        dec1_again = np.fromfile(repeat / 'out-dec1.bin', np.uint8)
        results[mode] = {'info': info, 'dec1': dec1, 'path': str(res),
                         'deterministic': bool(np.array_equal(dec1, dec1_again)),
                         'dec1_sha256': hashlib.sha256(dec1.tobytes()).hexdigest()}
    a, b = results['fp16']['dec1'], results['int8-q']['dec1']
    ref = np.fromfile(Path(layers) / 'dec1' / 'ref-out.bin', np.uint8)
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    out_report = {'dispatching': {'fp16': results['fp16']['info']['stages'],
                                  'int8-q': results['int8-q']['info']['stages']},
                  'deterministic': {k: v['deterministic'] for k, v in results.items()},
                  'dec1_sha256': {k: v['dec1_sha256'] for k, v in results.items()},
                  'dec1': {'samples': int(a.size), 'identical': float((a == b).mean()),
                           'differing': int((a != b).sum()),
                           'max_code_delta': int(delta.max()),
                           'mean_code_delta': float(delta.mean())},
                  'vs_capture': {'fp16': _field(a, ref), 'int8_q': _field(b, ref)},
                  'decoded': {'rms': None}}
    # decoded error on the e4m3 table
    table = np.array([mc.e4m3_value(int(c)) for c in range(256)], F64)
    da, db = table[a], table[b]
    finite = np.isfinite(da) & np.isfinite(db)
    out_report['decoded'] = {
        'finite_samples': int(finite.sum()),
        'non_finite_codes': int((~finite).sum()),
        'relative_rms': float(np.sqrt(((da[finite] - db[finite]) ** 2).mean())
                              / np.sqrt((da[finite] ** 2).mean())),
        'max_abs': float(np.abs(da[finite] - db[finite]).max()),
        'signed_mean': float((db[finite] - da[finite]).mean())}
    return out_report


def frame_report(a, out, layers, enc1_int8, translated, fixtures):
    command = [sys.executable, str(HERE / 'm_b6_frame.py'),
               '--out', str(out / 'frame-int8q'), '--layers', str(layers),
               '--fixtures', str(fixtures), '--q-mode', 'int8-q',
               '--enc1-int8', str(enc1_int8)]
    if translated:
        command += ['--translated-dec1-capture', str(translated)]
    done = subprocess.run(command, capture_output=True, text=True)
    record = out / 'frame-int8q' / 'b6.json'
    if done.returncode and not record.is_file():
        return {'error': f'{done.stdout[-800:]}{done.stderr[-800:]}'}
    return json.loads(record.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--layers', required=True, help='directory of ten fp16 layer fixtures')
    r.add_argument('--capture', default=None, help='export fresh enc1 fixtures from here')
    r.add_argument('--fp16-fixture', default=None)
    r.add_argument('--int8-fixture', default=None)
    r.add_argument('--fixtures', default=str(
        Path(os.path.expanduser('~')) / 'AppData/Local/Temp/opencode/m-capture-run/fixtures'),
        help='B6 image-stage fixtures')
    r.add_argument('--translated-dec1', default=None)
    r.add_argument('--frame', action='store_true')
    r.add_argument('--probe', default=str(REPO / 'build/native/Release/m_network_stage.exe'))
    r.add_argument('--shaders', default=str(REPO / 'build/native'))
    o = sub.add_parser('oracle')
    o.add_argument('--h1', required=True)
    o.add_argument('--weights', required=True)
    o.add_argument('--weight-offset', type=int, required=True)
    o.add_argument('--tokens', type=int, required=True)
    o.add_argument('--channels', type=int, default=64)
    a = ap.parse_args()

    if a.command == 'oracle':
        h1 = np.fromfile(a.h1, F16).reshape(a.tokens, a.channels)
        W = np.fromfile(a.weights, F32)[a.weight_offset:][:a.channels * a.channels]
        W = W.reshape(a.channels, a.channels).astype(F64)
        codes, scale = quantize_rows_float(h1)
        wcodes, wscale, _ = quantize_weights(W)
        y, acc = gemm_int8(codes, scale.ravel(), wcodes, wscale)
        print(json.dumps({'codes': code_metrics(codes, scale.ravel()),
                          'output_absmax': float(np.abs(y.astype(F64)).max()),
                          'accumulator_max_abs': int(np.abs(acc).max())}, indent=2))
        return 0

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fixtures = Path(a.fixtures)
    fp16_dir = Path(a.fp16_fixture) if a.fp16_fixture else out / 'enc1-fp16'
    int8_dir = Path(a.int8_fixture) if a.int8_fixture else out / 'enc1-int8'
    if a.capture:
        lf.export(a.capture, 'enc1', str(fp16_dir), 1, 'fp16', True)
        lf.export(a.capture, 'enc1', str(int8_dir), 1, 'int8-q', True)

    results = {}
    paths = {}
    for mode, directory in (('fp16', fp16_dir), ('int8-q', int8_dir)):
        paths[mode] = run_probe(a.probe, directory, out / f'{mode}-results', a.shaders)
    fp16 = load_layer(fp16_dir, paths['fp16'])
    int8 = load_layer(int8_dir, paths['int8-q'])
    results['shaders'] = {name: sha(Path(a.shaders) / name)
                          for name in ('swin_q8_quant.cso', 'swin_q8_matmul.cso',
                                       'swin_matmul.cso', 'swin_attention.cso')}
    results['q'] = q_report(fp16, int8)
    results['layer'] = layer_report(paths['fp16'], paths['int8-q'], fp16_dir)
    results['chain'] = chain_report(Path(a.layers), out, int8_dir, a.probe, a.shaders)
    if a.frame:
        results['frame'] = frame_report(out, out, Path(a.layers), int8_dir,
                                        Path(a.translated_dec1) if a.translated_dec1 else None,
                                        fixtures)
    (out / 'm09-enc1q.json').write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({'q': {'fp16_vs_int8': results['q']['output']['fp16_vs_int8'],
                            'gpu_vs_cpu_codes': results['q']['activation']['gpu_vs_cpu_codes'],
                            'gpu_codes_to_cpu_gemm': results['q']['arithmetic'][
                                'gpu_codes_to_cpu_gemm_vs_gpu_q'],
                            'cpu_codes_to_cpu_gemm': results['q']['arithmetic'][
                                'cpu_codes_to_cpu_gemm_vs_gpu_q']},
                      'layer': results['layer'],
                      'chain_vs_capture': results['chain']['vs_capture'],
                      'chain': {'dec1': results['chain']['dec1'],
                                'decoded': results['chain']['decoded'],
                                'deterministic': results['chain']['deterministic']}},
                     indent=2, default=str))
    print('out:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

