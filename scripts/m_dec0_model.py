#!/usr/bin/env python3
"""M05 B4.3 instrument: a transcription of `dec0_m.comp`'s epilogue.

**This model does not reproduce the reference.**  It exists because it is the instrument
that localised the blocker recorded in [EXP-013]: the epilogue is fully determined by the
20 expansion values, the activation table and the normalised stores, and all of that is
reproduced here - but the expansion values themselves cannot be obtained from the captured
weight blob with any decode tried so far.  Running it prints how far the four surfaces are
from the capture, which is the evidence that the *matrix mapping* (not the epilogue) is open.

The pipeline is:

    token field (E4M3, 64 per token)
      -> expansion matrix [64][80] + f16 bias, one f16 rounding per 32 products
      -> E4M3 round-trip of each of the 20 outputs per (token, 2x2 sub-position)
      -> activation table lookups (sigmoid / tanh / gate) by E4M3 code
      -> gates, softmax mix, 2x2 covariance, recurrent feature
      -> UNORM8 / SNORM8 / RGBA16F surfaces

Frame 1 is a reset frame: `featTex` is zero and `p.reset` short-circuits the history sample,
so no texture is read and the epilogue is a pure function of the expansion values.

usage: m_dec0_model.py CAPTURE [--json OUT]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ENGINE = os.path.join(REPO, 'references', 'sources', 'd4r', 'engine')
TOOLS = os.path.join(REPO, 'references', 'sources', 'd4r', 'kernels', 'tools')
for path in (HERE, ENGINE, TOOLS):
    sys.path.insert(0, path)

import m_weights as mw        # noqa: E402
import model_enc3 as me       # noqa: E402
import gen_activation_m as ga  # noqa: E402

F16, F32, F64 = np.float16, np.float32, np.float64
W, H = 960, 540
# dec0_m.comp's parameter block, from the argument slots (EXP-013): rnd2hi, hiSize, reset,
# the render size, and the two f32 parameters that fit its use of sigma and rho.
PARAMS = {'rnd2hi': [3.0, 3.0], 'hiSize': [2880, 1620], 'reset': 1,
          'size': [960, 540], 'sigma': 8.0, 'rho': 0.07863261550664902}
SIGMA = F32(PARAMS['sigma'])
RHO = F32(PARAMS['rho'])
SURFACES = {'feat': 2, 'gate': 3, 'mix': 4, 'cov': 5}


def half(bits):
    return np.array(bits, np.uint16).view(F16)


def act(code):
    """(sigmoid, tanh, gate) f16 bit patterns of one E4M3 code (generated, not tabulated)."""
    packed, gate = ga.row(int(code))
    return half(packed & 0xFFFF), half((packed >> 16) & 0xFFFF), half(gate)


ACT_SIG = np.array([act(c)[0] for c in range(256)], F16)
ACT_TAN = np.array([act(c)[1] for c in range(256)], F16)
ACT_GATE = np.array([act(c)[2] for c in range(256)], F16)


def fma(a, b, c):
    """Fused f32 multiply-add: the f64 product is exact, so this rounds once."""
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def unorm8(values):
    v = np.clip(np.asarray(values, F32), 0.0, 1.0)
    return np.floor(v * F32(255.0) + F32(0.5)).astype(np.uint8)


def snorm8(values):
    v = np.clip(np.asarray(values, F32), -1.0, 1.0)
    code = np.floor(np.abs(v) * F32(127.0) + F32(0.5))
    return (np.sign(v) * code).astype(np.int32).astype(np.uint8)


def expansion(capture):
    """The 20 values per (token, sub-position), from the captured blob and token field."""
    stages = mw.inventory(capture)
    row = next(r for s, r in mw.matrices(stages) if r['name'] == 'dec0.expansion')
    dec1 = next(s for s in stages if s.stage == 'dec1')
    P = me.Params(dec1.directory)
    # dec0_m.comp's non-PLANES branch reads 64 contiguous E4M3 channels per token; its
    # PLANES branch - and m_model.tokens - read two planes of 32.  EXP-013 records that
    # neither reproduces the captured surfaces with this matrix decode.
    offset = P.out - P.abase
    codes = np.frombuffer(P.arena, np.uint8)[offset:offset + P.W * P.H * 64]
    x = me.E4M3[codes.reshape(-1, 64)].astype(F64)
    matrix = np.asarray(row['W'], F16)
    bias = np.asarray(row['bias'], F16)
    out = me.mma_chain(x, matrix.astype(F64),
                       C0=np.broadcast_to(bias.astype(F64), (x.shape[0], matrix.shape[1])))
    out = np.asarray(out, F16).reshape(P.H, P.W, 4, 20)
    return out, P.W, P.H


def epilogue(values, tokw, tokh):
    """The four surfaces, from the 20 values per (token, sub-position)."""
    codes = mw.e4m3_encode(values.astype(F32)).reshape(tokh, tokw, 4, 20)
    quantised = mw.e4m3_decode(codes).astype(F16)         # q8: f16 -> E4M3 -> f16

    py, px = np.arange(H), np.arange(W)
    sub = ((py & 1)[:, None] << 1) | (px & 1)[None, :]
    v = quantised[py[:, None] >> 1, px[None, :] >> 1, sub]
    c = codes[py[:, None] >> 1, px[None, :] >> 1, sub]

    sig = lambda k: ACT_SIG[c[..., k]]
    tan = lambda k: ACT_TAN[c[..., k]]
    gate = lambda k: ACT_GATE[c[..., k]]

    gates = np.stack([sig(0), sig(1), sig(10), np.zeros((H, W), F16)], -1)

    m = np.stack([v[..., 11], v[..., 12], v[..., 13]], -1).astype(F32)
    ex = np.exp2((m - m.max(-1, keepdims=True)) * F32(1.442695), dtype=F32)
    total = ex[..., 2] + (ex[..., 0] + ex[..., 1])
    mix = np.concatenate([ex * (F32(1.0) / total)[..., None], np.zeros((H, W, 1), F32)], -1)

    a = SIGMA * ACT_SIG[c[..., 14]].astype(F32)
    b = SIGMA * ACT_SIG[c[..., 15]].astype(F32)
    t = v[..., 16].astype(F32) * RHO
    s, mag = t * t, np.abs(t)
    big = fma(F32(1.0) / (np.exp2(mag * F32(2.88539), dtype=F32) + F32(1.0)), F32(-2.0), F32(1.0))
    big = np.where(mag >= F32(9.0109138), F32(1.0), big)
    poly = fma(F32(0.01573968306183815), s, F32(-0.05230396240949631))
    poly = fma(poly, s, F32(0.13315297663211823))
    poly = fma(poly, s, F32(-0.33332768082618713))
    poly = fma(fma(poly, s, F32(0.0)), t, t)
    rho = np.where(mag >= F32(0.6), np.where(t < 0.0, -big, big), poly).astype(F32)
    cov = np.stack([a * a, b * b, a * b * rho, np.zeros((H, W), F32)], -1).astype(F16)

    # reset frame: (1 - g) * 0 is exactly zero, so the blend is one f16 product
    g = np.stack([gate(6), gate(7), gate(8), gate(9)], -1)
    fresh = np.stack([tan(2), tan(3), tan(4), tan(5)], -1)
    blend = (g.astype(F32) * fresh.astype(F32)).astype(F16)

    return {'feat': snorm8(blend), 'gate': unorm8(gates), 'mix': unorm8(mix), 'cov': cov,
            'values': values, 'codes': codes}


def captured(capture, name):
    directory = None
    for entry in os.listdir(capture):
        if entry.startswith('replay-') and 'dec0' in entry:
            directory = os.path.join(capture, entry)
            break
    if directory is None:
        raise SystemExit('the capture has no dec0 launch')
    path = os.path.join(directory, 'gpu-output', f'image-{SURFACES[name]}.bin')
    return np.fromfile(path, F16 if name == 'cov' else np.uint8).reshape(H, W, 4)


def compare(native, reference):
    if native.dtype == F16:
        same = native.view(np.uint16) == reference.view(np.uint16)
        delta = np.abs(native.astype(F64) - reference.astype(F64))
        return {'identical': float(same.mean()), 'max_abs': float(delta.max()),
                'rms': float(np.sqrt((delta ** 2).mean())),
                'identical_per_channel': [float(same[..., c].mean()) for c in range(4)]}
    same = native == reference
    delta = np.abs(native.astype(np.int16) - reference.astype(np.int16))
    return {'identical': float(same.mean()), 'max_code_delta': int(delta.max()),
            'identical_per_channel': [float(same[..., c].mean()) for c in range(4)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    values, tokw, tokh = expansion(a.capture)
    surfaces = epilogue(values, tokw, tokh)
    result = {'capture': a.capture, 'params': PARAMS,
              'token_layout': '64 contiguous E4M3 channels per token (dec0_m.comp non-PLANES)',
              'surfaces': {}, 'reproduces_reference': False}
    print(f'{PARAMS}')
    for name in ('feat', 'gate', 'mix', 'cov'):
        row = compare(surfaces[name], captured(a.capture, name))
        result['surfaces'][name] = row
        print(f'{name:5s} identical {100 * row["identical"]:6.2f}%  '
              + (f'max|d| {row["max_abs"]:.5g}  rms {row["rms"]:.4g}' if 'max_abs' in row
                 else f'max code delta {row["max_code_delta"]}'))
    print('\nThis model does not reproduce the reference: see EXP-013. The open question is '
          'the expansion matrix/token mapping, not the epilogue.')
    if a.json:
        json.dump(result, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
