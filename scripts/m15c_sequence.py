#!/usr/bin/env python3
"""M15-C: persistent multi-frame validation and measurement.

Two independent paths execute the same twelve M10 content variants:

  * reference - the accepted fixture runner (`m_b6_frame.py`), one process per
    stage per frame, on disk, exactly as M15-B measured it;
  * persistent - `m_inference_stage`, one process for all twelve frames, live
    GPU textures in, one caller-owned output texture out, no per-frame
    allocation or rebuild.

Every frame must match the reference byte for byte.  Frame 0 must match the
accepted M09 frame.  A thirteenth execution replays frame 0 to prove exact
recovery.  A separate production run measures steady state; an interleaved
block compares it with M15-B on the canonical frame.

  m15c_sequence.py --out DIR [--repeats 5] [--reuse]
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
DEFAULT_LAYERS = TEMP / 'm09h-complete-chain-20261030/layers'
DEFAULT_FIXTURES = Path(os.environ.get('SETTE_FIXTURES', TEMP / 'm-capture-run/fixtures'))

import m10_temporal as temporal  # noqa: E402
import m14_baseline as baseline  # noqa: E402
import m15c_compose  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, allow_fail=False):
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode and not allow_fail:
        raise RuntimeError('command failed:\n' + ' '.join(map(str, command)) +
                           '\n' + done.stdout[-2000:] + done.stderr[-2000:])
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layers', type=Path, default=DEFAULT_LAYERS)
    ap.add_argument('--fixtures', type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument('--engine', type=Path,
                    default=REPO / 'build/native/Release/m_inference_stage.exe')
    ap.add_argument('--shaders', type=Path, default=REPO / 'build/native')
    ap.add_argument('--manifest', type=Path, help='use an existing composed manifest')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--reuse', action='store_true',
                    help='reuse compose/engine.txt and references.json from --out')
    a = ap.parse_args()
    if a.out.exists():
        if not a.reuse:
            raise SystemExit(f'{a.out} exists; use a fresh directory')
        if not (a.out / 'inputs').is_dir():
            raise SystemExit(f'{a.out} has no inputs to reuse')
    else:
        a.out.mkdir(parents=True)

    temporal.FIXTURES = a.fixtures
    colour = temporal.load_colour()
    inputs = []
    (a.out / 'inputs').mkdir(exist_ok=True)
    frames = []
    for index, (name, description) in enumerate(temporal.SCENARIOS):
        data = temporal.variant(colour, name)
        path = a.out / 'inputs' / f'frame-{index:02d}.bin'
        if not path.exists():
            path.write_bytes(data.tobytes())
        inputs.append(path)
        frames.append({'index': index, 'scenario': name, 'description': description,
                       'input': str(path), 'input_sha256': sha(path)})

    if a.manifest:
        manifest = a.manifest
    elif a.reuse and (a.out / 'compose' / 'engine.txt').is_file():
        manifest = a.out / 'compose' / 'engine.txt'
    else:
        composed = a.out / 'compose'
        m15c_compose.compose(composed, a.layers, a.fixtures, fused=True)
        manifest = composed / 'engine.txt'

    report = {'status': 'running', 'manifest': str(manifest), 'frames': frames,
              'reference': {}, 'persistent': {}, 'benchmark': {}, 'm15b': {}}
    references = {}
    reference_file = a.out / 'references.json'
    if a.reuse and reference_file.is_file():
        references = {int(k): v for k, v in
                      json.loads(reference_file.read_text())['hashes'].items()}
        for frame in frames:
            frame['reference_sha256'] = references[frame['index']]
    else:
        ref_root = a.out / 'reference'
        ref_root.mkdir(exist_ok=True)
        for index, frame in enumerate(frames):
            target = ref_root / f'frame-{index:02d}'
            run([sys.executable, str(HERE / 'm_b6_frame.py'), '--out', str(target),
                 '--no-repeat', '--colour', frame['input'],
                 '--enc0-mode', 'int8-embedding', '--attention-mode', 'int8',
                 '--cumulative-mode', 'spatial', '--cumulative-layers', str(a.layers),
                 '--fuse-mlp', '--network-arena', '--network-mode', 'production'],
                allow_fail=True)
            frame_file = target / 'downsample-native' / 'results' / 'out-out.bin'
            if not frame_file.is_file():
                raise SystemExit(f'reference frame {index} was not produced')
            report_file = target / 'b6.json'
            if report_file.is_file():
                execution = json.loads(report_file.read_text()).get('execution', {})
                if execution.get('status') == 'FAIL':
                    raise SystemExit(f'reference frame {index} execution FAIL: '
                                     f'{execution.get("reasons")}')
            references[index] = sha(frame_file)
            frame['reference_sha256'] = references[index]
            print(f'reference {index:02d} {frame["scenario"]:10s} {references[index]}',
                  flush=True)
            shutil.rmtree(target)
        reference_file.write_text(json.dumps(
            {'hashes': {str(k): v for k, v in references.items()}}, indent=2))
    if references[0] != baseline.ACCEPTED['int8']:
        raise SystemExit(f'frame 0 reference {references[0]} != accepted')

    # ---------------------------------------------------------- persistent run
    dump_root = a.out / 'engine-dumps'
    command = [str(a.engine), '--manifest', str(manifest), '--shaders', str(a.shaders),
               '--output', 'frame_out', '--frames', str(len(inputs)),
               '--mode', 'validation', '--dump-dir', str(dump_root),
               '--dump-frame', '0', '--recover',
               '--report', str(a.out / 'engine-validation.json')]
    for frame in frames:
        command += ['--input', f'enc0_in_colour={frame["input"]}']
    run(command)
    validation = json.loads((a.out / 'engine-validation.json').read_text())
    matches = {}
    for index, frame in enumerate(frames):
        got = sha(dump_root / f'frame-{index:02d}' / 'out-frame.bin')
        frame['engine_sha256'] = got
        frame['match'] = got == references[index]
        matches[index] = frame['match']
    recovered = sha(dump_root / 'recover' / 'out-frame.bin')
    report['reference'] = {'hashes': {str(k): v for k, v in references.items()},
                           'frame0_accepted': references[0] == baseline.ACCEPTED['int8']}
    report['persistent'] = {
        'hashes': {str(k): f['engine_sha256'] for k, f in enumerate(frames)},
        'matches': {str(k): v for k, v in matches.items()},
        'recovered_sha256': recovered,
        'recovery_matches_frame0': recovered == references[0],
        'validation_report': validation,
    }
    # Caller-owned command recording: the engine records on a list it neither
    # closes, executes nor waits on.  The harness submits and reads the output
    # back itself.
    caller = a.out / 'caller-list'
    run([str(a.engine), '--manifest', str(manifest), '--shaders', str(a.shaders),
         '--output', 'frame_out', '--frames', '1', '--caller-list',
         '--caller-readback', str(caller / 'frame.bin'),
         '--input', f'enc0_in_colour={frames[0]["input"]}',
         '--report', str(caller / 'report.json'), '--debug'])
    caller_hash = sha(caller / 'frame.bin')
    report['caller_list'] = {'sha256': caller_hash,
                             'matches_frame0': caller_hash == references[0]}
    report['status'] = ('PASS' if all(matches.values()) and
                        recovered == references[0] and
                        caller_hash == references[0] else 'FAIL')

    # -------------------------------------------------------------- benchmark
    # Steady state: production mode, no readback, all twelve inputs in one
    # process after one warm-up frame.
    warm = a.out / 'warm'
    warm_command = [str(a.engine), '--manifest', str(manifest), '--shaders', str(a.shaders),
                    '--output', 'frame_out', '--frames', '1',
                    '--input', f'enc0_in_colour={frames[0]["input"]}',
                    '--report', str(warm / 'report.json')]
    run(warm_command)
    steady = a.out / 'steady'
    steady_command = [str(a.engine), '--manifest', str(manifest), '--shaders', str(a.shaders),
                      '--output', 'frame_out', '--frames', str(len(inputs)),
                      '--report', str(steady / 'report.json')]
    for frame in frames:
        steady_command += ['--input', f'enc0_in_colour={frame["input"]}']
    run(steady_command)
    steady_report = json.loads((steady / 'report.json').read_text())
    gpu = [row['gpu_ms'] for row in steady_report['runs']]
    record = [row['record_ms'] for row in steady_report['runs']]
    wait = [row['submit_wait_ms'] for row in steady_report['runs']]
    report['benchmark'] = {
        'init_ms': steady_report['init']['total_ms'],
        'init': steady_report['init'],
        'memory': steady_report['memory'],
        'frames': len(gpu),
        'gpu_ms_median': statistics.median(gpu),
        'gpu_ms_min': min(gpu),
        'gpu_ms_max': max(gpu),
        'record_ms_median': statistics.median(record),
        'submit_wait_ms_median': statistics.median(wait),
        'per_frame': gpu,
    }

    # ------------------------------------------------- interleaved M15-B block
    rounds = {'m15b': [], 'persistent': []}
    canonical = a.fixtures / 'enc0' / 'in-colour.bin'
    for repeat in range(a.repeats):
        order = ['m15b', 'persistent'][repeat % 2:] + ['m15b', 'persistent'][:repeat % 2]
        for name in order:
            if name == 'm15b':
                result = baseline.run_mode('int8', a.out / 'm15b' / f'run-{repeat}',
                                           a.layers, 'production', True, True)
                rounds['m15b'].append(result['total']['global_ms'])
                print(f'm15b round {repeat}: {result["total"]["global_ms"]:.3f} ms',
                      flush=True)
            else:
                target = a.out / 'persistent' / f'run-{repeat}'
                run([str(a.engine), '--manifest', str(manifest), '--shaders', str(a.shaders),
                     '--output', 'frame_out', '--frames', '1',
                     '--input', f'enc0_in_colour={canonical}',
                     '--report', str(target / 'report.json')])
                row = json.loads((target / 'report.json').read_text())['runs'][0]
                rounds['persistent'].append(row['gpu_ms'])
                print(f'persistent round {repeat}: {row["gpu_ms"]:.3f} ms', flush=True)
    m15b = statistics.median(rounds['m15b'])
    persistent = statistics.median(rounds['persistent'])
    report['m15b'] = {
        'rounds': rounds,
        'gpu_ms_median': m15b,
        'persistent_gpu_ms_median': persistent,
        'saving_ms': m15b - persistent,
        'ratio': m15b / persistent,
    }
    report['status'] = 'PASS' if report['status'] == 'PASS' else 'FAIL'
    (a.out / 'sequence.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'],
                      'matches': report['persistent']['matches'],
                      'recovery': report['persistent']['recovery_matches_frame0'],
                      'steady_gpu_ms': report['benchmark']['gpu_ms_median'],
                      'm15b_gpu_ms': m15b,
                      'persistent_gpu_ms': persistent,
                      'saving_ms': report['m15b']['saving_ms']}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
