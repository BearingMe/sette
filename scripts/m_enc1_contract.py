#!/usr/bin/env python3
"""Build pinned fast/conservative enc1, replay identical bytes, compare contracts.

All generated code objects, captures and reports stay in a fresh local directory.
No installed runtime, pinned source, or historical fixture is modified.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

import m_layer_fixture as lf
import m_layer_oracle as lo
from m_capture_provenance import verify_translated

KERNEL = 'rrlite_enc1_4x4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, log, env=None):
    result = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, timeout=240)
    Path(log).write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f'exit={result.returncode}: {args}; see {log}')
    return result.stdout.decode(errors='replace')


def verify_macros(text, mode):
    names = set(re.findall(r'^#define (\w+)', text, re.M))
    exact = 'SWIN_EXACT' in names and 'D4R_ACCURACY' in names
    fast = 'SWIN_NO_Q8' in names and 'SWIN_F32ACC' in names
    if mode == 'conservative':
        valid = exact and not ({'SWIN_NO_Q8', 'SWIN_F32ACC'} & names)
    else:
        valid = fast and not ({'SWIN_EXACT', 'D4R_ACCURACY'} & names)
    if not valid:
        raise ValueError(f'effective macro mismatch: {mode}')
    return sorted(n for n in names if n.startswith(('SWIN_', 'D4R_')))


def verify_replay(text, kernel=KERNEL):
    required = [f'hipModuleLoad(replay) -> 0 (hipSuccess)',
                'hipModuleLaunchKernel(replay prep) -> 0 (hipSuccess)',
                'hipModuleLaunchKernel(replay) -> 0 (hipSuccess)',
                f'PASS NATIVE_REPLAY architecture=gfx1101 kernel={kernel} iterations=1']
    geometry = re.search(rf'REPLAY kernel={re.escape(kernel)} '
                         r'grid=\d+,\d+,\d+ block=\d+,\d+,\d+', text)
    if not geometry or not all(x in text for x in required):
        raise ValueError('native module load/dispatch/geometry not proven')


def metrics(reference, actual):
    if reference.size == 0 or reference.shape != actual.shape:
        raise ValueError('empty or unequal field sizes')
    x, y = lo.me.E4M3[reference].astype(np.float64), lo.me.E4M3[actual].astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    delta = y[finite] - x[finite]
    denom = float(np.mean(x[finite] ** 2)) if finite.any() else 0.0
    return {'bytes': int(reference.size), 'exact': float((reference == actual).mean()),
            'differing_codes': int((reference != actual).sum()),
            'nonfinite_pairs': int((~finite).sum()),
            'signed_mean_decoded': float(np.mean(delta)) if delta.size else None,
            'saturated_reference': int(np.isin(reference, (0x7e, 0xfe)).sum()),
            'saturated_actual': int(np.isin(actual, (0x7e, 0xfe)).sum()),
            'relative_rms': float(np.sqrt(np.mean(delta ** 2) / denom)) if denom else None,
            'max_abs_decoded': float(np.max(np.abs(delta))) if delta.size else None}


def natural(codes, W, H, C):
    """The stored e4m3 field as (H, W, C) natural channel order."""
    if codes.size != W * H * C:
        raise ValueError(f'{codes.size} B does not hold {W}x{H}x{C}')
    a = codes.reshape(C // 32, H, W, 32)
    return a.transpose(1, 2, 0, 3).reshape(H, W, C)


def distribution(reference, actual, W, H, C):
    """Where the differing codes sit: channels, tokens, and 8x8 sub-positions."""
    r, a = natural(reference, W, H, C), natural(actual, W, H, C)
    differ = r != a
    channel = 1.0 - differ.reshape(-1, C).mean(axis=0)
    token = 1.0 - differ.reshape(H * W, C).mean(axis=1)
    channel_differ = differ.reshape(-1, C).sum(axis=0)
    token_differ = differ.reshape(H * W, C).sum(axis=1)
    delta = lo.me.E4M3[a].astype(np.float64) - lo.me.E4M3[r].astype(np.float64)
    bias = delta.reshape(-1, C).mean(axis=0)
    ys = (np.arange(H) % 8)[:, None]
    xs = (np.arange(W) % 8)[None, :]
    grid = np.array([[1.0 - differ[(ys == dy) & (xs == dx)].mean()
                      for dx in range(8)] for dy in range(8)])
    worst_channel = int(np.argmax(channel_differ))
    worst_token = np.unravel_index(int(np.argmax(token_differ)), (H, W))
    return {'channel_exact_min': float(channel.min()),
            'channel_exact_worst': worst_channel,
            'channels_below_99': int((channel < 0.99).sum()),
            'channel_signed_bias_max': float(bias[int(np.argmax(np.abs(bias)))]),
            'token_exact_min': float(token.min()),
            'token_exact_median': float(np.median(token)),
            'token_exact_worst_yx': [int(worst_token[0]), int(worst_token[1])],
            'spatial_exact_min': float(grid.min()),
            'spatial_exact_max': float(grid.max())}


def fields(P, raw, directory):
    candidates = [p for p in (Path(directory) / 'gpu-output').glob('*.bin')
                  if p.stat().st_size == P.arena.size]
    if len(candidates) != 1:
        raise ValueError(f'{len(candidates)} arena-sized allocations in {directory}')
    arena = np.fromfile(candidates[0], np.uint8)
    return {role: arena[ptr - P.abase:ptr - P.abase + raw[role].size]
            for role, ptr in [('output', P.out), ('merge', P.p48)] if role in raw}


def main():
    repo = Path(__file__).resolve().parents[1]
    temp = Path(os.environ['LOCALAPPDATA']) / 'Temp/opencode'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kernel', default=KERNEL, help=f'kernel to replay (default {KERNEL})')
    ap.add_argument('--layer', default='enc1', help='network layer name (default enc1)')
    ap.add_argument('--hip-root', type=Path, default=Path(r'C:\Program Files\AMD\ROCm\7.2'))
    ap.add_argument('--probe', type=Path, default=temp / 'd4r-win-src/build/probe2/d4r_native_replay_probe.exe')
    ap.add_argument('--capture', type=Path, default=None,
                    help='canonical native replay for the same launch (default: enc1 fixture)')
    ap.add_argument('--translated', type=Path, default=None,
                    help='translated-PTX replay of this kernel (default: enc1 capture)')
    ap.add_argument('--fixture', type=Path, default=None,
                    help='Sette fixture built from the same input bytes')
    a = ap.parse_args()
    kernel, layer = a.kernel, a.layer
    if a.capture is None:
        a.capture = temp / f'm-capture-run/capture-1080b/replay-000001-{kernel}'
    if a.translated is None:
        if kernel != KERNEL:
            ap.error('--translated is required for a kernel other than the enc1 control')
        a.translated = temp / f'enc1-translated-20261029/capture/replay-000001-{kernel}'
    if a.fixture is None:
        if layer != 'enc1':
            ap.error('--fixture is required for a layer other than enc1')
        a.fixture = temp / 'm-capture-run/layers/enc1'
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    source = repo / 'references/sources/d4r'
    hip = a.hip_root
    compiler = hip / 'bin/clang++.exe'
    kernel_source = source / f'kernels/m/{kernel}.hip'
    base = [compiler, '-x', 'hip', '-std=c++17', '--offload-arch=gfx1101',
            '--offload-device-only', '-O3', '-DSWIN_BUFFER_BLOAD_FAST',
            # ROCm 7.2 spelling compatibility; verify the emitted minnum/maxnum IR below.
            '-D__builtin_elementwise_maximumnum=__builtin_elementwise_max',
            '-D__builtin_elementwise_minimumnum=__builtin_elementwise_min',
            f'-I{hip / "include"}', f'--rocm-path={hip}',
            f'--rocm-device-lib-path={hip / "amdgcn/bitcode"}']
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    if dirty:
        raise RuntimeError('pinned d4r source is dirty')
    report = {'source_commit': commit, 'source_dirty': False, 'compiler_sha256': sha(compiler),
              'compiler_version': command([compiler, '--version'], a.out / 'compiler.log'),
              'probe_sha256': sha(a.probe), 'builds': {}, 'runs': {}, 'comparisons': {}}
    report['sources'] = {str(p.relative_to(source)): sha(p) for directory in ('kernels/m', 'kernels/common')
                         for p in (source / directory).glob('*') if p.is_file()}
    # Independently check the compatibility spelling's LLVM semantics (not fast-math).
    check = a.out / 'builtin-check.hip'
    check.write_text('typedef _Float16 hv2 __attribute__((ext_vector_type(2)));\n'
                     'extern "C" __attribute__((device, used)) hv2 check(hv2 x, hv2 y) {\n'
                     'return __builtin_elementwise_minimumnum('
                     '__builtin_elementwise_maximumnum(x, -y), y); }\n')
    command(base + ['-S', '-emit-llvm', check, '-o', a.out / 'builtin-check.ll'], a.out / 'builtin-check.log')
    ir = (a.out / 'builtin-check.ll').read_text()
    if 'llvm.minnum.v2f16' not in ir or 'llvm.maxnum.v2f16' not in ir:
        raise RuntimeError('compatibility builtins did not lower to minnum/maxnum')
    P, raw = lo.raw_layer(str(a.capture), layer)
    report['capture_files'] = {p.name: sha(p) for p in a.capture.glob('*.bin')}
    report['input_sha256'] = hashlib.sha256(raw['input']).hexdigest()
    report['weights_sha256'] = hashlib.sha256(P.wbuf).hexdigest()
    data = {}
    if a.capture.resolve() != a.translated.resolve():
        data['capture'] = {k: raw[k] for k in ('output', 'merge')}
    for mode in ('conservative', 'fast'):
        build = a.out / mode
        build.mkdir()
        flags = base + (['-DD4R_ACCURACY'] if mode == 'conservative' else [])
        macros = command(flags + ['-E', '-dM', kernel_source], build / 'macros.txt')
        effective = verify_macros(macros, mode)
        module = build / f'{kernel}.hsaco'
        cmd = flags + [kernel_source, '-o', module, '--no-gpu-bundle-output']
        command(cmd, build / 'build.log')
        report['builds'][mode] = {'command': list(map(str, cmd)), 'effective_macros': effective,
                                 'module_sha256': sha(module), 'module': str(module)}
        before = sha(module)
        for repeat in ('first', 'second'):
            output = a.out / f'{mode}-{repeat}'
            output.mkdir()
            cmd = [a.probe, '--hip-root', hip, '--module', module, '--fixture-dir', a.capture,
                   '--output-dir', output / 'gpu-output', '--iterations', '1']
            text = command(cmd, output / 'replay.log')
            verify_replay(text, kernel)
            if sha(module) != before:
                raise RuntimeError('module changed during replay')
            report['runs'][f'{mode}-{repeat}'] = {'command': list(map(str, cmd)),
                'module_sha256': before, 'log_sha256': sha(output / 'replay.log'),
                'output_hashes': {p.name: sha(p) for p in (output / 'gpu-output').glob('*.bin')}}
        first, second = [fields(P, raw, a.out / f'{mode}-{r}') for r in ('first', 'second')]
        report['runs'][mode + '-deterministic'] = all(np.array_equal(first[k], second[k]) for k in first)
        data[mode] = first
    # Fresh Sette runs; historical references are not overwritten.
    sette_probe = repo / 'build/native/Release/m_network_stage.exe'
    shaders = repo / 'build/native'
    report['sette_execution'] = {'probe': str(sette_probe), 'probe_sha256': sha(sette_probe),
        'shaders': {p.name: sha(p) for p in shaders.glob('swin_*.cso')},
        'fixture_files': {p.name: sha(p) for p in a.fixture.iterdir() if p.is_file()}, 'commands': []}
    for seed in (0, 1):
        output = a.out / f'sette-{seed}'
        cmd = [os.sys.executable, repo / 'scripts/m_layer_fixture.py', 'run', a.fixture,
               '--seed', str(seed), '--out', output, '--probe', sette_probe, '--shaders', shaders]
        report['sette_execution']['commands'].append(list(map(str, cmd)))
        command(cmd, a.out / f'sette-{seed}.log')
    roles = [('output', 'out'), ('merge', 'merge')]
    roles = [(k, r) for k, r in roles if (a.out / f'sette-0/out-{r}.bin').exists()]
    report['runs']['sette-deterministic'] = all(
        sha(a.out / f'sette-0/out-{r}.bin') == sha(a.out / f'sette-1/out-{r}.bin') for k, r in roles)
    data['sette'] = {k: np.fromfile(a.out / f'sette-0/out-{r}.bin', np.uint8) for k, r in roles}
    fixture = json.loads((a.fixture / 'fixture.json').read_text())
    geometry = fixture['geometry']
    # The packed fixture must originate from this exact input and captured weight image.
    packed = lf.WeightPack(P, layer)
    if geometry['CIN']:
        same_input = (np.array_equal(np.fromfile(a.fixture / 'low.bin', np.uint8), raw['low']) and
                      np.array_equal(np.fromfile(a.fixture / 'skip.bin', np.uint8), raw['skip']))
    else:
        same_input = np.array_equal(np.fromfile(a.fixture / 'in.bin', np.uint8), raw['input'])
    if (not same_input or
            not np.array_equal(np.fromfile(a.fixture / 'weights.bin', np.float32), packed.data)):
        raise RuntimeError('Sette fixture input or weights differ from capture')
    report['sette_fixture_sha256'] = sha(a.fixture / 'fixture.json')
    report['sette_geometry'] = geometry
    PT, translated = lo.raw_layer(str(a.translated), layer)
    if not np.array_equal(P.wbuf, PT.wbuf) or not np.array_equal(raw['input'], translated['input']):
        raise RuntimeError('translated input or weights differ')
    data['translated'] = {k: translated[k] for k in ('output', 'merge') if k in translated}
    translated_root = a.translated.parent.parent
    log = translated_root / 'probe.log'
    expected = 1
    if (translated_root / 'run.json').is_file():
        expected = json.loads((translated_root / 'run.json').read_text()).get(
            'translated_dispatches', 1)
    report['translated_control'] = {'dispatch': verify_translated(
            log.read_text(errors='replace'), kernel, expected),
        'log_sha256': sha(log), 'run_sha256': sha(translated_root / 'run.json'),
        'configuration': json.loads((translated_root / 'run.json').read_text())}
    shapes = {'output': (geometry['W'], geometry['H'], geometry['C'])}
    if 'merge' in translated:
        shapes['merge'] = (geometry['W'] // 2, geometry['H'] // 2, geometry['NPM'])
    for left, right in itertools.combinations(data, 2):
        rows = {}
        for field, values in data[left].items():
            row = metrics(values, data[right][field])
            row['distribution'] = distribution(values, data[right][field], *shapes[field])
            rows[field] = row
        report['comparisons'][left + '-vs-' + right] = rows
    report['field_hashes'] = {name: {k: hashlib.sha256(v).hexdigest() for k, v in value.items()} for name, value in data.items()}
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))
    if not all(report['runs'][k + '-deterministic'] for k in ('fast', 'conservative', 'sette')):
        raise RuntimeError('nondeterministic execution; see report.json')
    print(json.dumps({k: {f: {x: v[x] for x in ('exact', 'differing_codes',
                                               'relative_rms', 'max_abs_decoded',
                                               'signed_mean_decoded')}
                          for f, v in rows.items()}
                      for k, rows in report['comparisons'].items()}, indent=2))


if __name__ == '__main__':
    main()
