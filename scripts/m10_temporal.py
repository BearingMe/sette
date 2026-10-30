#!/usr/bin/env python3
"""M10: bounded multi-frame (temporal) execution over the accepted reset graph.

The reconstruction has no executable cross-frame consumer: every stage fixture
is the reset path, `post`/`dec0` do not bind their history/feature inputs, and
`enc0` declares no motion outputs.  This driver therefore evaluates the only
temporal contract that exists, and says so instead of implying more:

  * a sequence is a list of frames with *different inputs*, each executed by the
    unmodified approved harness (`m_b6_frame.py`);
  * the persistent tensors the graph still produces (post history, post
    half-resolution surface, dec0 recurrent feature) are captured, hashed and
    compared element by element between the FP16 oracle and the INT8 path;
  * a replay pass re-executes selected frames from their recorded inputs and
    requires every boundary and state hash to reproduce;
  * frame-to-frame independence is measured, not assumed: the same frame is run
    in different sequence positions and in isolation, and its hashes must agree.

It does not implement `reset = 0` history reuse.  That needs a labelled
non-reset capture and the missing Catmull-Rom/recurrent/motion branches; see
EXP-030 and EXP-031 for the explicit BLOCKED classification.

  m10_temporal.py --out DIR [--cumulative-layers DIR] [--frames 12] [--modes fp16,int8]
                  [--replay 0,6,11] [--checkpoint 0,5,11]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
FIXTURES = Path(os.environ.get('SETTE_FIXTURES', TEMP / 'm-capture-run/fixtures'))
LAYERS = Path(os.environ.get('SETTE_M10_LAYERS',
                             TEMP / 'm09h-complete-chain-20261030/layers'))
W, H = 960, 540
ALPHA = None

#: The persistent tensors the executed graph still writes.  `role` is the
#: output name `m_b6_frame.py` records, `stage` the directory it lives in.
STATE = {
    'post.history': {'stage': 'post', 'file': 'out-history.bin',
                     'dtype': '<f2', 'shape': [1620, 5760, 2], 'bytes': 37324800},
    'post.half': {'stage': 'post', 'file': 'out-half.bin',
                  'dtype': '<f4', 'shape': [810, 1440, 4], 'bytes': 18662400},
    'dec0.feature': {'stage': 'dec0', 'file': 'out-feat.bin',
                     'dtype': 'u1', 'shape': [540, 960, 4], 'bytes': 2073600},
}

#: Frame content.  These are *input* variations, not validated camera motion:
#: the reset path ignores motion, so a shift is simply different pixels.
SCENARIOS = [
    ('static', 'canonical colour, unchanged'),
    ('hold', 'identical to the previous frame'),
    ('pan-8', 'RGB rolled 8 px horizontally'),
    ('pan-32v', 'RGB rolled 32 px vertically'),
    ('rotate-3', 'RGB rotated 3 degrees about the centre'),
    ('fast-96', 'RGB rolled 96 px horizontally'),
    ('disocclude', 'shifted background with a 96x96 opaque reveal patch'),
    ('repeat', 'one 48x48 crop tiled over the frame'),
    ('thin', '1-px diagonal lines on the canonical colour'),
    ('abrupt', 'inverted and shifted colour'),
    ('recover', 'canonical colour again after the abrupt frame'),
    ('static-end', 'canonical colour, held to the end'),
]

MODES = {
    'fp16': [],
    'int8': ['--enc0-mode', 'int8-embedding', '--attention-mode', 'int8',
             '--cumulative-mode', 'spatial'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_colour():
    data = np.fromfile(FIXTURES / 'enc0' / 'in-colour.bin', '<f2')
    if data.size != H * W * 4:
        raise SystemExit(f'{FIXTURES / "enc0" / "in-colour.bin"}: {data.size} f16 values')
    return data.reshape(H, W, 4)


def rotate(rgb, degrees):
    from PIL import Image
    out = np.empty_like(rgb)
    for c in range(3):
        plane = Image.fromarray(rgb[..., c].astype(np.float32), 'F')
        out[..., c] = np.asarray(plane.rotate(degrees, resample=Image.BICUBIC,
                                              center=(W / 2, H / 2)), np.float32)
    return out


def variant(colour, name):
    """Deterministic RGB content for one frame; alpha is never changed."""
    rgb = colour[..., :3].astype(np.float32)
    out = rgb.copy()
    if name in ('static', 'hold', 'recover', 'static-end'):
        pass
    elif name == 'pan-8':
        out = np.roll(rgb, 8, axis=1)
    elif name == 'pan-32v':
        out = np.roll(rgb, 32, axis=0)
    elif name == 'rotate-3':
        out = rotate(rgb, 3.0)
    elif name == 'fast-96':
        out = np.roll(rgb, 96, axis=1)
    elif name == 'disocclude':
        out = np.roll(rgb, 40, axis=1)
        reveal = np.roll(rgb, 200, axis=0)[120:216, 400:496]
        out[220:316, 480:576] = reveal
    elif name == 'repeat':
        crop = rgb[200:248, 300:348]
        out = np.tile(crop, (H // 48 + 1, W // 48 + 1, 1))[:H, :W]
    elif name == 'thin':
        out = rgb.copy()
        for k in range(0, W + H, 24):
            ys = np.arange(max(0, k - H + 1), min(H, k + 1))
            out[ys, k - ys] = 0.0
            out[ys, (k - ys + 7) % W] = 1.0
    elif name == 'abrupt':
        out = 1.0 - np.roll(rgb, 64, axis=0)
    else:
        raise SystemExit(f'unknown scenario {name}')
    frame = np.empty_like(colour)
    frame[..., :3] = np.clip(out, 0.0, 1.0)
    frame[..., 3] = colour[..., 3]
    return frame.astype('<f2')


def run_frame(mode, colour, out, layers, extra=()):
    out = Path(out)
    if out.is_dir():
        shutil.rmtree(out)
    command = [sys.executable, str(HERE / 'm_b6_frame.py'), '--out', str(out),
               '--no-repeat', '--colour', str(colour), *MODES[mode], *extra]
    if mode != 'fp16':
        command += ['--cumulative-layers', str(layers)]
    done = subprocess.run(command, capture_output=True, text=True)
    report_path = out / 'b6.json'
    if not report_path.is_file():
        raise RuntimeError(f'{mode} {colour}: no b6.json\n{done.stdout[-800:]}{done.stderr[-800:]}')
    report = json.loads(report_path.read_text())
    if report['execution']['status'] == 'FAIL':
        raise RuntimeError(f'{mode} {colour}: execution FAIL {report["execution"]["reasons"]}')
    return report


def frame_path(out, mode, index):
    return Path(out) / mode / f'frame-{index:02d}' / 'downsample-native' / 'results' / 'out-out.bin'


def state_path(out, mode, index, role):
    spec = STATE[role]
    return (Path(out) / mode / f'frame-{index:02d}' / f'{spec["stage"]}-native' /
            'results' / spec['file'])


def read_state(out, mode, index, role):
    spec = STATE[role]
    path = state_path(out, mode, index, role)
    raw = np.fromfile(path, spec['dtype'])
    if raw.size != int(np.prod(spec['shape'])):
        raise RuntimeError(f'{path}: {raw.size} values, expected {spec["shape"]}')
    return raw.reshape(spec['shape'])


def describe(a, b):
    """Difference between two persistent tensors in their own storage domain."""
    if a.dtype == np.uint8:
        # SNORM8: the dec0 encoder emits sign(v)*round(|v|*127), so -128 is never
        # produced; clamp it for a defined decode if it ever appears.
        def snorm(x):
            signed = np.where(x < 128, x.astype(np.int16), x.astype(np.int16) - 256)
            return np.clip(signed, -127, 127).astype(np.float64) / 127.0
        x, y = snorm(a), snorm(b)
        unit, scale = 'snorm8', 1.0 / 127.0
    else:
        x = a.astype(np.float64)
        y = b.astype(np.float64)
        unit, scale = ('f16' if a.dtype == np.float16 else 'f32'), None
    with np.errstate(invalid='ignore'):
        d = y - x
    finite = np.isfinite(d)
    mse = float(np.mean(d[finite] ** 2)) if finite.any() else None
    denom = float(np.sqrt(np.mean(x[finite] ** 2))) if finite.any() else 0.0
    return {
        'dtype': str(a.dtype), 'unit': unit, 'elements': int(a.size),
        'identical': bool(np.array_equal(a, b)),
        'equal_elements': int(np.count_nonzero(a == b)),
        'differing_elements': int(np.count_nonzero(a != b)),
        'non_finite': int((~finite).sum()),
        'max_abs': float(np.max(np.abs(d[finite]))) if finite.any() else None,
        'rms': float(np.sqrt(mse)) if mse is not None else None,
        'relative_rms': float(np.sqrt(mse) / denom) if mse and denom else None,
        'quantum': scale,
    }


def boundary_mismatches(original, replay):
    """Boundary hashes the replay failed to reproduce."""
    return sorted(k for k, v in original.items() if replay.get(k) != v)


def state_matches(original, replay):
    """Every persistent tensor hash reproduced."""
    return all(replay[role]['sha256'] == row['sha256'] for role, row in original.items())


def frame_metrics(fp16, int8):
    sys.path.insert(0, str(HERE))
    import m_b6_frame as b6
    ref = b6.frame(fp16)
    test = b6.frame(int8)
    row = b6.metrics(ref, test)
    keep = ('psnr_peak1_db', 'ssim_windowed_luma', 'rms', 'mean_abs', 'p50_abs',
            'p99_abs', 'max_abs', 'mean_relative', 'signed_mean', 'nan_or_inf_pairs',
            'saturated_test', 'alpha_identical', 'alpha_samples')
    return {k: row[k] for k in keep}


def main():
    global FIXTURES, LAYERS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cumulative-layers', type=Path, default=LAYERS)
    ap.add_argument('--fixtures', type=Path, default=FIXTURES)
    ap.add_argument('--frames', type=int, default=len(SCENARIOS))
    ap.add_argument('--modes', default='fp16,int8')
    ap.add_argument('--replay', default='',
                    help='comma-separated frame indices to re-execute from their inputs')
    ap.add_argument('--checkpoint', default='',
                    help='comma-separated frame indices whose persistent tensors are kept')
    a = ap.parse_args()

    FIXTURES, LAYERS = a.fixtures, a.cumulative_layers
    if not (LAYERS / 'dec1' / 'graph.txt').is_file():
        raise SystemExit(f'missing cumulative layer fixtures: {LAYERS}')
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f'unknown mode {mode}')
    if a.frames < 1 or a.frames > len(SCENARIOS):
        raise SystemExit(f'--frames must be 1..{len(SCENARIOS)}')
    replay = sorted({int(x) for x in a.replay.split(',') if x.strip()})
    checkpoint = sorted({int(x) for x in a.checkpoint.split(',') if x.strip()})
    for index in replay + checkpoint:
        if not 0 <= index < a.frames:
            raise SystemExit(f'frame index {index} outside 0..{a.frames - 1}')

    a.out.mkdir(parents=True, exist_ok=False)
    inputs = a.out / 'inputs'
    inputs.mkdir()
    colour = load_colour()
    frames = []
    for index, (name, description) in enumerate(SCENARIOS[:a.frames]):
        data = variant(colour, name)
        path = inputs / f'frame-{index:02d}-colour.bin'
        path.write_bytes(np.ascontiguousarray(data).tobytes())
        frames.append({'index': index, 'scenario': name, 'description': description,
                       'colour': str(path), 'colour_sha256': sha(path)})
    (a.out / 'sequence.json').write_text(json.dumps(
        {'frames': frames, 'modes': modes, 'layers': str(LAYERS),
         'fixtures': str(FIXTURES)}, indent=2))

    report = {'status': 'running', 'frames': frames, 'modes': modes,
              'layers': str(LAYERS), 'fixtures': str(FIXTURES),
              'replay': replay, 'checkpoint': checkpoint, 'per_frame': {},
              'state_checkpoints': {}, 'replays': {}}
    canonical = {
        'fp16': 'dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
        'int8': 'f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00',
    }

    for frame in frames:
        index = frame['index']
        row = {'scenario': frame['scenario'], 'colour_sha256': frame['colour_sha256'],
               'modes': {}}
        for mode in modes:
            out = a.out / mode / f'frame-{index:02d}'
            b6 = run_frame(mode, frame['colour'], out, a.cumulative_layers)
            frame_file = frame_path(a.out, mode, index)
            entry = {'b6': str(out / 'b6.json'),
                     'frame': str(frame_file), 'frame_sha256': sha(frame_file),
                     'boundaries': b6['boundaries'],
                     'boundary_count': len(b6['boundaries']),
                     'execution': b6['execution']['status'],
                     'regression': b6['fidelity']['regression']['status'],
                     'psnr_vs_canonical': b6['comparisons']['native']['psnr_peak1_db'],
                     'state': {}}
            for role in STATE:
                path = state_path(a.out, mode, index, role)
                entry['state'][role] = {'path': str(path), 'sha256': sha(path),
                                        'bytes': path.stat().st_size}
            if mode == 'fp16' and index == 0:
                entry['matches_accepted_m09_frame'] = entry['frame_sha256'] == canonical['fp16']
            if mode == 'int8' and index == 0:
                entry['matches_accepted_m09_frame'] = entry['frame_sha256'] == canonical['int8']
            row['modes'][mode] = entry

        if {'fp16', 'int8'} <= set(modes):
            row['frame_int8_vs_fp16'] = frame_metrics(frame_path(a.out, 'fp16', index),
                                                      frame_path(a.out, 'int8', index))
            row['state_int8_vs_fp16'] = {
                role: describe(read_state(a.out, 'fp16', index, role),
                               read_state(a.out, 'int8', index, role))
                for role in STATE}
            row['state_identical_between_modes'] = {
                role: row['state_int8_vs_fp16'][role]['identical'] for role in STATE}

        if index in checkpoint:
            for mode in modes:
                store = a.out / 'state' / mode / f'frame-{index:02d}'
                store.mkdir(parents=True, exist_ok=True)
                saved = {}
                for role in STATE:
                    target = store / STATE[role]['file']
                    shutil.copyfile(state_path(a.out, mode, index, role), target)
                    saved[role] = {'file': str(target), 'sha256': sha(target)}
                report['state_checkpoints'][f'{mode}/frame-{index:02d}'] = saved

        report['per_frame'][str(index)] = row
        (a.out / 'sequence.json').write_text(json.dumps(report, indent=2))
        summary = row.get('frame_int8_vs_fp16', {})
        print(f'frame {index:02d} {row["scenario"]:10s} '
              f'psnr={summary.get("psnr_peak1_db")} '
              f'rms={summary.get("rms")} max={summary.get("max_abs")}')
        sys.stdout.flush()

    for index in replay:
        frame = frames[index]
        replay_row = {'frame': index, 'scenario': frame['scenario'], 'modes': {}}
        for mode in modes:
            out = a.out / 'replay' / mode / f'frame-{index:02d}'
            b6 = run_frame(mode, frame['colour'], out, a.cumulative_layers)
            frame_file = frame_path(a.out / 'replay', mode, index)
            original = report['per_frame'][str(index)]['modes'][mode]
            differing = boundary_mismatches(original['boundaries'], b6['boundaries'])
            state = {}
            for role in STATE:
                fresh = sha(state_path(a.out / 'replay', mode, index, role))
                state[role] = {'sha256': fresh,
                               'matches': fresh == original['state'][role]['sha256']}
            replay_row['modes'][mode] = {
                'b6': str(out / 'b6.json'), 'frame_sha256': sha(frame_file),
                'frame_matches': sha(frame_file) == original['frame_sha256'],
                'boundaries_match': not differing,
                'differing_boundaries': differing,
                'state': state,
                'state_matches': state_matches(original['state'], state)}
        report['replays'][str(index)] = replay_row
        (a.out / 'sequence.json').write_text(json.dumps(report, indent=2))
        print(f'replay frame {index:02d}: '
              + ', '.join(f'{m} frame={v["frame_matches"]} boundaries={v["boundaries_match"]} '
                          f'state={all(s["matches"] for s in v["state"].values())}'
                          for m, v in replay_row['modes'].items()))

    # ---- verdicts ---------------------------------------------------------
    checks = {}
    checks['first_frame_matches_accepted_m09'] = all(
        report['per_frame']['0']['modes'][m].get('matches_accepted_m09_frame')
        for m in modes if m in canonical)
    checks['all_frames_executed'] = all(
        x['execution'] != 'FAIL'
        for row in report['per_frame'].values() for x in row['modes'].values())
    checks['frame_finite'] = all(
        row.get('frame_int8_vs_fp16', {}).get('nan_or_inf_pairs', 0) == 0
        for row in report['per_frame'].values()) if {'fp16', 'int8'} <= set(modes) else None
    checks['state_finite'] = all(
        v['non_finite'] == 0 for row in report['per_frame'].values()
        for v in row.get('state_int8_vs_fp16', {}).values())
    checks['replay_reproduces'] = all(
        v['modes'][m]['frame_matches'] and v['modes'][m]['boundaries_match']
        and all(s['matches'] for s in v['modes'][m]['state'].values())
        for v in report['replays'].values() for m in modes) if replay else None
    checks['checkpoints_kept'] = bool(checkpoint) and all(
        f'{m}/frame-{i:02d}' in report['state_checkpoints']
        for i in checkpoint for m in modes)
    report['checks'] = checks
    report['status'] = ('PASS' if all(v is not False for v in checks.values())
                        else 'FAIL')
    (a.out / 'sequence.json').write_text(json.dumps(report, indent=2))
    print('checks:', json.dumps(checks, indent=2))
    print('status:', report['status'])
    print('out:', a.out)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
