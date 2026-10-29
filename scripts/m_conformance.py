#!/usr/bin/env python3
"""The regression gate for the reconstructed stages (M05 B4.7).

One command runs every stage from its captured fixture, compares every declared
output against its captured reference, rebuilds the dec0 -> post -> downsample
propagation chain, and applies the numerical criteria of EXP-017.

The gate fails on anything it did not actually compare.  Each stage declares its
outputs up front; a missing result, a result whose size does not match the
fixture's own declaration, a reference of the wrong size, or an empty comparison
is a failure, never a skip.  With `--run` (the default) the results are produced
by this invocation, so a stale file cannot pass either.

usage:
  m_conformance.py                 # run every stage, then gate
  m_conformance.py --no-run        # gate existing results (still requires them)
  m_conformance.py --json OUT
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIXTURES = os.environ.get(
    'SETTE_FIXTURES',
    os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp', 'opencode',
                 'm-capture-run', 'fixtures'))

#: Every stage and every output it must produce.  `kind` selects the comparison:
#: `bytes` is a literal byte comparison, `codes`/`f16` compare per sample after
#: dropping the unused channels.  `criterion` names the floor in CRITERIA.
STAGES = {
    'downsample': [{'role': 'out', 'kind': 'bytes', 'criterion': 'exact'}],
    'enc0': [{'role': 'tokens', 'kind': 'bytes', 'criterion': 'exact'}],
    'dec0': [{'role': 'gate', 'kind': 'codes', 'channels': 3,
              'criterion': 'dec0-unorm8'},
             {'role': 'mix', 'kind': 'codes', 'channels': 3,
              'criterion': 'dec0-unorm8'},
             {'role': 'feat', 'kind': 'codes', 'channels': 4,
              'criterion': 'dec0-snorm8'},
             {'role': 'cov', 'kind': 'f16', 'channels': 3,
              'criterion': 'dec0-f16'}],
    'post': [{'role': 'packed', 'kind': 'bytes', 'criterion': 'exact'},
             {'role': 'history', 'kind': 'bytes', 'criterion': 'exact'},
             {'role': 'half', 'kind': 'bytes', 'criterion': 'exact'}],
}

#: Floors measured on the canonical frame with margin (EXP-017 §4).
CRITERIA = {
    'exact': {'identical_min': 1.0},
    'dec0-unorm8': {'identical_min': 0.9990, 'max_code_delta': 8},
    'dec0-snorm8': {'identical_min': 0.9995, 'max_code_delta': 3},
    'dec0-f16': {'identical_min': 0.9980},
    'frame': {'identical_min': 0.9990, 'max_ulp_delta': 8, 'mean_ulp_max': 1.0},
}

#: The propagation chain.  Each entry clones a *consumer* fixture and overwrites
#: the inputs that the producer stage generates, so the chain is rebuilt from
#: native outputs rather than from the capture.
CHAIN = [{'base': 'post', 'target': 'post-from-native',
          'subs': [('gate', 'dec0'), ('mix', 'dec0'), ('cov', 'dec0')]},
         {'base': 'downsample', 'target': 'downsample-from-native',
          'subs': [('src', 'post-from-native')]}]


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


def sigmoid(v, sigma_unused=None):
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


def classify(native, reference, kind):
    """Identity and magnitude of one comparison, from raw bytes."""
    if kind == 'bytes':
        same = native == reference
        return {'identical': float(same.mean()), 'differing': int((~same).sum()),
                'samples': int(same.size)}
    if kind == 'f16':
        nu = native.view(np.uint16).astype(np.int32)
        ru = reference.view(np.uint16).astype(np.int32)
        delta = nu - ru
        differ = delta != 0                  # by bits: +0 and -0 are different bytes
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.abs(native.astype(np.float64) - reference.astype(np.float64)) \
                / np.maximum(np.abs(reference.astype(np.float64)), 1e-30)
        return {'identical': float((delta == 0).mean()),
                'differing': int(differ.sum()), 'samples': int(delta.size),
                'max_ulp_delta': int(np.abs(delta).max()),
                'max_relative': float(rel[differ].max()) if differ.any() else 0.0}
    delta = native.astype(np.int32) - reference.astype(np.int32)
    return {'identical': float((delta == 0).mean()),
            'differing': int((delta != 0).sum()), 'samples': int(delta.size),
            'max_code_delta': int(np.abs(delta).max())}


def covariance_code_distance(worst_native, worst_reference, sigma=8.0):
    """How many E4M3 codes apart the two covariance values are.

    `cov.xx = (sigma*sigmoid(v))^2` is monotonic in `v`, so a value's position in
    the 254-entry table is its E4M3 code.  A boundary flip is distance 1.
    """
    _, act = activation_table(sigma)
    table = (sigma * act.astype(np.float64)) ** 2
    worst = 0
    for channel in range(worst_native.shape[1]):
        nv = worst_native[:, channel].astype(np.float64)
        rv = worst_reference[:, channel].astype(np.float64)
        for k in np.nonzero(nv != rv)[0]:
            i = int(np.argmin(np.abs(table - rv[k])))
            j = int(np.argmin(np.abs(table - nv[k])))
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


def verdict(scope, criterion, result):
    rules = CRITERIA[criterion]
    problems = []
    if result['identical'] < rules['identical_min']:
        problems.append(f'{scope}: identical {result["identical"]:.6f} < '
                        f'{rules["identical_min"]}')
    for key in ('max_code_delta', 'max_ulp_delta'):
        if key in rules and result.get(key, 0) > rules[key]:
            problems.append(f'{scope}: {key} {result[key]} > {rules[key]}')
    return problems


def declared_bytes(fixture):
    """role -> byte count the fixture's own manifest declares for each output."""
    path = os.path.join(fixture, 'fixture.json')
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        record = json.load(handle)
    return {role: entry['bytes'] for role, entry in record['bindings'].items()
            if entry.get('output')}


def audit_stage(fixtures, stage):
    """Compare every output the stage declares.  Missing files are failures."""
    fixture = os.path.join(fixtures, stage)
    problems, results = [], {}
    declared = declared_bytes(fixture)
    if declared is None:
        return {}, [f'{stage}: no fixture.json under {fixture}']
    for spec in STAGES[stage]:
        role = spec['role']
        scope = f'{stage}.{role}'
        out = os.path.join(fixture, 'results', f'out-{role}.bin')
        ref = os.path.join(fixture, f'ref-{role}.bin')
        for label, path in (('result', out), ('reference', ref)):
            if not os.path.exists(path):
                problems.append(f'{scope}: missing {label} {os.path.basename(path)}')
        if problems and any(scope in p for p in problems):
            continue
        native = np.fromfile(out, np.uint8)
        reference = np.fromfile(ref, np.uint8)
        if native.size != reference.size:
            problems.append(f'{scope}: result is {native.size} B, the reference is '
                            f'{reference.size} B')
            continue
        if role not in declared:
            problems.append(f'{scope}: the fixture does not declare this output')
            continue
        if native.size != declared[role]:
            problems.append(f'{scope}: result is {native.size} B, the fixture declares '
                            f'{declared[role]} B')
            continue
        if native.size == 0:
            problems.append(f'{scope}: empty comparison')
            continue
        if spec['kind'] in ('codes', 'f16'):
            dtype = np.float16 if spec['kind'] == 'f16' else np.uint8
            channels = spec['channels']
            native = np.frombuffer(native.tobytes(), dtype).reshape(-1, 4)[:, :channels]
            reference = np.frombuffer(reference.tobytes(), dtype).reshape(-1, 4)[:, :channels]
        result = classify(native, reference, spec['kind'])
        results[role] = result
        problems += verdict(scope, spec['criterion'], result)
    return results, problems


def execute(fixtures, stage, probe, shaders, suffix=''):
    """Run one fixture from scratch, so a stale result cannot be compared."""
    fixture = os.path.join(fixtures, stage)
    results = os.path.join(fixture, f'results{suffix}')
    if os.path.isdir(results):
        shutil.rmtree(results)
    done = subprocess.run([probe, '--fixture', os.path.join(fixture, 'stage.txt'),
                           '--out', results, '--shaders', shaders],
                          capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f'{stage}: {done.stdout[-500:]}{done.stderr[-500:]}')


def audit_determinism(fixtures, stage):
    """Two runs of the same fixture must produce the same bytes."""
    problems = []
    for spec in STAGES[stage]:
        role = spec['role']
        first = os.path.join(fixtures, stage, 'results', f'out-{role}.bin')
        second = os.path.join(fixtures, stage, 'results-repeat', f'out-{role}.bin')
        for path in (first, second):
            if not os.path.exists(path):
                problems.append(f'{stage}.{role}: no output for the determinism check')
                break
        else:
            a = np.fromfile(first, np.uint8)
            b = np.fromfile(second, np.uint8)
            if a.size != b.size or not np.array_equal(a, b):
                problems.append(f'{stage}.{role}: two runs differ '
                                f'({int((a != b).sum()) if a.size == b.size else "size"} '
                                'bytes)')
    return problems


def rebuild_chain(fixtures, probe, shaders):
    """Rebuild each downstream fixture from the upstream stage's native output."""
    for link in CHAIN:
        base = os.path.join(fixtures, link['base'])
        dst = os.path.join(fixtures, link['target'])
        if not os.path.isdir(base):
            return f'{link["target"]}: {link["base"]} is missing'
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(base, dst)
        shutil.rmtree(os.path.join(dst, 'results'), ignore_errors=True)
        for role, producer in link['subs']:
            if producer == 'dec0':
                produced = os.path.join(fixtures, 'dec0', 'results', f'out-{role}.bin')
            else:
                # a producer stage's only exported buffer is its packed output
                produced = os.path.join(fixtures, producer, 'results', 'out-packed.bin')
            if not os.path.exists(produced):
                return f'{link["target"]}: {producer} produced no {os.path.basename(produced)}'
            shutil.copyfile(produced, os.path.join(dst, f'in-{role}.bin'))
        execute(fixtures, link['target'], probe, shaders)
    return None


def audit_frame(fixtures):
    """The end-to-end criterion: the final frame from the native dec0 surfaces."""
    native_path = os.path.join(fixtures, 'downsample-from-native', 'results',
                               'out-out.bin')
    ref_path = os.path.join(fixtures, 'downsample', 'ref-out.bin')
    for path in (native_path, ref_path):
        if not os.path.exists(path):
            return None, [f'frame: missing {os.path.basename(path)} '
                          f'(the propagation chain did not run)']
    native = np.fromfile(native_path, np.float16)
    reference = np.fromfile(ref_path, np.float16)
    if native.size != reference.size:
        return None, [f'frame: result is {native.size} samples, the reference is '
                      f'{reference.size}']
    result = classify(native, reference, 'f16')
    delta = (native.view(np.uint16).astype(np.int32)
             - reference.view(np.uint16).astype(np.int32))
    result['mean_ulp'] = float(delta[delta != 0].mean()) if (delta != 0).any() else 0.0
    problems = verdict('frame', 'frame', result)
    if abs(result['mean_ulp']) > CRITERIA['frame']['mean_ulp_max']:
        problems.append(f'frame: mean signed residual {result["mean_ulp"]:.3f} ulp '
                        f'exceeds {CRITERIA["frame"]["mean_ulp_max"]}')
    return result, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fixtures', default=FIXTURES)
    ap.add_argument('--probe', default=os.path.join('build', 'native', 'Release',
                                                    'm_image_stage.exe'))
    ap.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    ap.add_argument('--no-run', action='store_true',
                    help='gate the existing results instead of running the stages')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    report = {'fixtures': a.fixtures, 'ran': not a.no_run,
              'criteria': CRITERIA, 'surfaces': {}, 'problems': []}
    print(f'fixtures {a.fixtures}')

    if not a.no_run:
        for stage in STAGES:
            try:
                execute(a.fixtures, stage, a.probe, a.shaders)
                execute(a.fixtures, stage, a.probe, a.shaders, suffix='-repeat')
            except RuntimeError as error:
                report['problems'].append(str(error))
            report['problems'] += audit_determinism(a.fixtures, stage)
        chain_error = rebuild_chain(a.fixtures, a.probe, a.shaders)
        if chain_error:
            report['problems'].append(chain_error)
    else:
        print('  determinism not checked (--no-run)')

    for stage, specs in STAGES.items():
        results, problems = audit_stage(a.fixtures, stage)
        report['problems'] += problems
        for spec in specs:
            role = spec['role']
            scope = f'{stage}.{role}'
            result = results.get(role)
            if result is None:
                print(f'  {scope:22s} NOT COMPARED')
                continue
            report['surfaces'][scope] = result
            extra = result.get('max_code_delta', result.get('max_ulp_delta'))
            shown = f'max {extra}' if spec['kind'] != 'f16' else \
                f'max rel {result["max_relative"]:.3f}'
            print(f'  {scope:22s} identical {100 * result["identical"]:8.4f}%  '
                  f'{shown:14s} {result["differing"]:>6d} differ  '
                  f'{"ok" if not any(scope in p for p in problems) else "FAIL"}')

    # structural invariants
    dec0 = {role: np.frombuffer(
        np.fromfile(os.path.join(a.fixtures, 'dec0', f'ref-{role}.bin'), np.uint8)
        .tobytes(), np.float16 if role == 'cov' else np.uint8
    ).reshape(-1, 4)[:, :(3 if role != 'feat' else 4)]
        for role in ('gate', 'mix', 'feat', 'cov')}
    native = {role: np.frombuffer(
        np.fromfile(os.path.join(a.fixtures, 'dec0', 'results', f'out-{role}.bin'),
                    np.uint8).tobytes(),
        np.float16 if role == 'cov' else np.uint8
    ).reshape(-1, 4)[:, :(3 if role != 'feat' else 4)]
        for role in ('gate', 'mix', 'feat', 'cov')}
    masks = {role: ((native[role].view(np.uint16) != dec0[role].view(np.uint16))
                    if role == 'cov' else (native[role] != dec0[role])).any(-1)
             for role in dec0}
    overlap = not disjoint(masks)
    report['disjoint'] = not overlap
    if overlap:
        report['problems'].append('dec0: two surfaces differ on the same pixel')
    distance = covariance_code_distance(native['cov'][:, :2], dec0['cov'][:, :2])
    report['covariance_code_distance'] = distance
    if distance > 1:
        report['problems'].append(f'dec0.cov: {distance} E4M3 codes apart, not a '
                                  'boundary flip')

    frame, problems = audit_frame(a.fixtures)
    report['problems'] += problems
    if frame is None:
        print('  frame                  NOT COMPARED')
    else:
        report['surfaces']['frame'] = frame
        print(f'  {"frame":22s} identical {100 * frame["identical"]:8.4f}%  '
              f'max {frame["max_ulp_delta"]} ulp  mean {frame["mean_ulp"]:+.3f}  '
              f'{"ok" if not problems else "FAIL"}')
    print(f'  {"dec0 disjoint":22s} {report["disjoint"]}   '
          f'covariance off by <= 1 E4M3 code: {distance == 1}')

    print()
    if report['problems']:
        for problem in report['problems']:
            print(f'  FAIL {problem}')
        if a.json:
            json.dump(report, a.json, indent=2)
        return 1
    print('  gate passed: every declared output was compared')
    if a.json:
        json.dump(report, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
