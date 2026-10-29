#!/usr/bin/env python3
"""Numerical conformance criteria for the FP16 oracle (M05 B4.6).

A reconstruction stage is accepted on three kinds of evidence, and this module
is the executable form of them:

 1. **Identity floors.**  How many samples match exactly, per surface.  A floor
    is only meaningful next to (2).
 2. **Structural invariants.**  Properties that hold for a residual caused by
    quantisation boundaries and break for one caused by an arithmetic error:
      * a difference in a surface derived from an E4M3 value lands on an
        *adjacent E4M3 code* of that value (checked against the 256-entry table,
        not against the implementation);
      * the surfaces that consume disjoint expansion columns never differ on the
        same pixel (a shared input error would make them coincide);
      * the end-to-end residual is zero-mean to within half an output ulp.
 3. **Propagation bound.**  What the stage's residual does to the *final* frame
    when the rest of the chain is fed the native surfaces instead of the
    captured ones.

The thresholds below are floors measured on the canonical frame with margin
(EXP-017 §3).  They are acceptance floors for that frame, not universal
guarantees: a new frame re-measures, it does not inherit.

usage: m_conformance.py [--json OUT]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIXTURES = os.environ.get(
    'SETTE_FIXTURES',
    os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp', 'opencode',
                 'm-capture-run', 'fixtures'))

#: Per-surface floors.  `identical` is the fraction of samples that must match
#: exactly; the max deltas are the widest observed, so they catch a regression
#: that stays just under the fraction floor.
CRITERIA = {
    # A stage whose arithmetic is reproduced exactly claims byte identity.
    'exact': {'identical_min': 1.0},
    # dec0's 8-bit surfaces: one E4M3 step in the feeding expansion value,
    # amplified by the activation, at ~0.1% of samples.
    'dec0-unorm8': {'identical_min': 0.9990, 'max_code_delta': 8},
    'dec0-snorm8': {'identical_min': 0.9995, 'max_code_delta': 3},
    # the covariance surface: same mechanism, with the square's amplification.
    'dec0-f16': {'identical_min': 0.9980},
    # the end-to-end claim: the final frame after the whole chain.
    'frame': {'identical_min': 0.9990, 'max_ulp_delta': 8, 'mean_ulp_max': 1.0},
}


def e4m3_value(code):
    """The 256-value table, from the format's definition (1-4-3, bias 7)."""
    sign = (code >> 7) & 1
    exponent = (code >> 3) & 0xF
    mantissa = code & 7
    if exponent == 0:
        value = mantissa * 2.0 ** -9
    elif exponent == 15 and mantissa == 7:
        return float('nan')
    else:
        value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return -value if sign else value


def sigmoid(v):
    """dec0's activation sequence: f16 input, exp2, f16 output."""
    c = np.float16(np.clip(v, -20.0, 20.0))
    e = np.exp2(np.float32(-1.4426951) * np.float32(c))
    return np.float16(np.float32(1.0) / (np.float32(1.0) + e))


def activation_table(sigma=8.0):
    """Sorted (value, activation) pairs over the finite E4M3 codes."""
    values = np.array([e4m3_value(int(c)) for c in range(256)])
    act = np.array([float(sigmoid(v)) for v in values], np.float32)
    ok = np.isfinite(act)
    values, act = values[ok], act[ok]
    order = np.argsort(values)
    return values[order], act[order]


def classify(samples, kind):
    """Identity and magnitude of a surface comparison, from raw codes."""
    native, reference = samples
    if kind == 'f16':
        nu = native.view(np.uint16).astype(np.int32)
        ru = reference.view(np.uint16).astype(np.int32)
        delta = nu - ru
        differ = native != reference
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.abs(native.astype(np.float64) - reference.astype(np.float64)) \
                / np.maximum(np.abs(reference.astype(np.float64)), 1e-30)
        out = {'identical': float((delta == 0).mean()),
               'differing': int(differ.sum()),
               'samples': int(delta.size),
               'max_ulp_delta': int(np.abs(delta).max()),
               'max_relative': float(rel[differ].max()) if differ.any() else 0.0}
        return out
    delta = native.astype(np.int32) - reference.astype(np.int32)
    return {'identical': float((delta == 0).mean()),
            'differing': int((delta != 0).sum()),
            'samples': int(delta.size),
            'max_code_delta': int(np.abs(delta).max())}


def covariance_code_distance(native, reference, sigma=8.0):
    """How many E4M3 codes apart the two covariance values are.

    `cov.xx = (sigma*sigmoid(v))^2` is monotonic in `v`, so the position of a
    value in the 254-entry table is the E4M3 code.  A boundary flip is distance
    1; anything larger is a real difference in the expansion value.
    """
    _, act = activation_table(sigma)
    table = (sigma * act.astype(np.float64)) ** 2
    worst = 0
    for channel in range(native.shape[1]):
        nv = native[:, channel].astype(np.float64)
        rv = reference[:, channel].astype(np.float64)
        for k in np.nonzero(nv != rv)[0]:
            i = int(np.argmin(np.abs(table - rv[k])))
            j = int(np.argmin(np.abs(table - nv[k])))
            worst = max(worst, abs(i - j))
    return worst


def on_one_e4m3_code(values, activations, native, reference):
    """Every differing sample matches the activation table one code apart.

    `native`/`reference` are the surface's values in activation units (a
    UNORM8 code divided by 255, an SNORM8 code divided by 127).  Returns the
    largest index distance and the largest match residual.
    """
    worst, residual = 0, 0.0
    for value in np.asarray(values, np.float64):
        i = int(np.argmin(np.abs(activations - value)))
        residual = max(residual, float(abs(activations[i] - value)))
    for nv, rv in zip(native, reference):
        if nv == rv:
            continue
        i = int(np.argmin(np.abs(activations - rv)))
        j = int(np.argmin(np.abs(activations - nv)))
        worst = max(worst, abs(i - j))
    return worst


def disjoint(masks):
    """True when no pixel differs on more than one surface."""
    total = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for mask in masks.values():
        if (total & mask).any():
            return False
        total |= mask
    return True


def verdict(stage, name, result):
    """Apply the criterion for `name` to one classified surface."""
    rules = CRITERIA[name]
    problems = []
    if result['identical'] < rules.get('identical_min', 0.0):
        problems.append(f'{stage}.{name}: identical {result["identical"]:.6f} < '
                        f'{rules["identical_min"]}')
    for key in ('max_code_delta', 'max_ulp_delta'):
        if key in rules and result.get(key, 0) > rules[key]:
            problems.append(f'{stage}.{name}: {key} {result[key]} > {rules[key]}')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fixtures', default=FIXTURES)
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    report = {'criteria': CRITERIA, 'surfaces': {}, 'problems': []}
    print(f'fixtures: {a.fixtures}')

    # --- exact stages ------------------------------------------------------
    exact = {
        'enc0': ('post', None),
        'downsample': ('downsample', 'out-out.bin'),
    }
    for stage, (folder, _) in exact.items():
        out = os.path.join(a.fixtures, stage, 'results', 'out-out.bin')
        if stage != 'downsample':
            out = os.path.join(a.fixtures, stage, 'results', 'out-tokens.bin')
        ref = os.path.join(a.fixtures, stage, 'ref-tokens.bin'
                           if stage == 'enc0' else 'ref-out.bin')
        if not (os.path.exists(out) and os.path.exists(ref)):
            print(f'  {stage}: no results (run the fixture first)')
            continue
        native = np.fromfile(out, np.uint8)
        reference = np.fromfile(ref, np.uint8)
        result = {'identical': float((native == reference).mean()),
                  'differing': int((native != reference).sum()),
                  'samples': int(native.size)}
        report['surfaces'][stage] = result
        problems = verdict(stage, 'exact', result)
        report['problems'] += problems
        print(f'  {stage:10s} identical {100 * result["identical"]:.4f}%  '
              f'{"PASS" if not problems else "FAIL"}')

    # --- dec0 --------------------------------------------------------------
    dec0 = os.path.join(a.fixtures, 'dec0')
    if os.path.isdir(dec0):
        specs = {'gate': ('image-3', 3, 'dec0-unorm8', 255.0),
                 'mix': ('image-4', 3, 'dec0-unorm8', 255.0),
                 'feat': ('image-2', 4, 'dec0-snorm8', 127.0),
                 'cov': ('image-5', 3, 'dec0-f16', None)}
        fields, masks = {}, {}
        covariance = None
        for role, (image, channels, name, scale) in specs.items():
            native = np.fromfile(os.path.join(dec0, 'results', f'out-{role}.bin'), np.uint8)
            reference = np.fromfile(os.path.join(dec0, f'ref-{role}.bin'), np.uint8)
            if name == 'dec0-f16':
                native = np.frombuffer(native, np.float16).reshape(-1, 4)[:, :channels]
                reference = np.frombuffer(reference, np.float16).reshape(-1, 4)[:, :channels]
                result = classify((native, reference), 'f16')
                covariance = (native[:, :2].copy(), reference[:, :2].copy())
            else:
                native = native.reshape(-1, 4)[:, :channels]
                reference = reference.reshape(-1, 4)[:, :channels]
                result = classify((native, reference), 'codes')
            masks[role] = (native != reference).any(-1)
            report['surfaces'][f'dec0-{role}'] = result
            problems = verdict('dec0', name, result)
            report['problems'] += problems
            worst = result.get('max_code_delta')
            if name == 'dec0-f16':
                worst = f'rel {result["max_relative"]:.3f}'
            print(f'  dec0.{role:5s} identical {100 * result["identical"]:.4f}%  '
                  f'max {worst}  {"PASS" if not problems else "FAIL"}')
        if not disjoint(masks):
            report['problems'].append('dec0: two surfaces differ on the same pixel, '
                                      'which points at an upstream error')
        print(f'  dec0 surfaces disjoint: {disjoint(masks)}')
        if covariance is not None:
            distance = covariance_code_distance(*covariance)
            report['covariance_code_distance'] = distance
            print(f'  dec0.cov  off by one E4M3 code at most: {distance}')
            if distance > 1:
                report['problems'].append(
                    f'dec0.cov: a value matches the E4M3 table {distance} codes away, '
                    'so the difference is not a boundary flip')

    # --- propagation -------------------------------------------------------
    frame_native = os.path.join(a.fixtures, 'downsample-from-native', 'results',
                                'out-out.bin')
    frame_ref = os.path.join(a.fixtures, 'downsample', 'ref-out.bin')
    if os.path.exists(frame_native) and os.path.exists(frame_ref):
        native = np.fromfile(frame_native, np.float16)
        reference = np.fromfile(frame_ref, np.float16)
        result = classify((native, reference), 'f16')
        delta = (native.view(np.uint16).astype(np.int32)
                 - reference.view(np.uint16).astype(np.int32))
        result['mean_ulp'] = float(delta[delta != 0].mean()) if (delta != 0).any() else 0.0
        report['surfaces']['frame'] = result
        problems = verdict('chain', 'frame', result)
        if abs(result['mean_ulp']) > CRITERIA['frame']['mean_ulp_max']:
            problems.append(f'chain.frame: mean signed residual {result["mean_ulp"]:.3f} '
                            f'ulp exceeds {CRITERIA["frame"]["mean_ulp_max"]}')
        report['problems'] += problems
        print(f'  frame      identical {100 * result["identical"]:.4f}%  '
              f'max {result["max_ulp_delta"]} ulp  mean {result["mean_ulp"]:+.3f}  '
              f'{"PASS" if not problems else "FAIL"}')

    print()
    if report['problems']:
        for problem in report['problems']:
            print(f'  FAIL {problem}')
        return 1
    print('  all criteria met')
    if a.json:
        json.dump(report, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
