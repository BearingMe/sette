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


def verify_replay(text):
    required = [f'REPLAY kernel={KERNEL} grid=61,35,1 block=32,1,4',
                'hipModuleLoad(replay) -> 0 (hipSuccess)',
                'hipModuleLaunchKernel(replay prep) -> 0 (hipSuccess)',
                'hipModuleLaunchKernel(replay) -> 0 (hipSuccess)',
                f'PASS NATIVE_REPLAY architecture=gfx1101 kernel={KERNEL} iterations=1']
    if not all(x in text for x in required):
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
            'relative_rms': float(np.sqrt(np.mean(delta ** 2) / denom)) if denom else None,
            'max_abs_decoded': float(np.max(np.abs(delta))) if delta.size else None}


def fields(P, raw, directory):
    arena = np.fromfile(Path(directory) / 'gpu-output/alloc-1.bin', np.uint8)
    if arena.size != P.arena.size:
        raise ValueError('replay arena size mismatch')
    return {role: arena[ptr - P.abase:ptr - P.abase + raw[role].size]
            for role, ptr in [('output', P.out), ('merge', P.p48)]}


def main():
    repo = Path(__file__).resolve().parents[1]
    temp = Path(os.environ['LOCALAPPDATA']) / 'Temp/opencode'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hip-root', type=Path, default=Path(r'C:\Program Files\AMD\ROCm\7.2'))
    ap.add_argument('--probe', type=Path, default=temp / 'd4r-win-src/build/probe2/d4r_native_replay_probe.exe')
    ap.add_argument('--capture', type=Path, default=temp / f'm-capture-run/capture-1080b/replay-000001-{KERNEL}')
    ap.add_argument('--translated', type=Path, default=temp / f'enc1-translated-20261029/capture/replay-000001-{KERNEL}')
    ap.add_argument('--fixture', type=Path, default=temp / 'm-capture-run/layers/enc1')
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    source = repo / 'references/sources/d4r'
    hip = a.hip_root
    compiler = hip / 'bin/clang++.exe'
    kernel = source / f'kernels/m/{KERNEL}.hip'
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
    P, raw = lo.raw_layer(str(a.capture), 'enc1')
    report['capture_files'] = {p.name: sha(p) for p in a.capture.glob('*.bin')}
    report['input_sha256'] = hashlib.sha256(raw['input']).hexdigest()
    report['weights_sha256'] = hashlib.sha256(P.wbuf).hexdigest()
    data = {'capture': {k: raw[k] for k in ('output', 'merge')}}
    for mode in ('conservative', 'fast'):
        build = a.out / mode
        build.mkdir()
        flags = base + (['-DD4R_ACCURACY'] if mode == 'conservative' else [])
        macros = command(flags + ['-E', '-dM', kernel], build / 'macros.txt')
        effective = verify_macros(macros, mode)
        module = build / f'{KERNEL}.hsaco'
        cmd = flags + [kernel, '-o', module, '--no-gpu-bundle-output']
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
            verify_replay(text)
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
    report['runs']['sette-deterministic'] = all(
        sha(a.out / f'sette-0/out-{r}.bin') == sha(a.out / f'sette-1/out-{r}.bin') for r in ('out', 'merge'))
    data['sette'] = {k: np.fromfile(a.out / f'sette-0/out-{r}.bin', np.uint8)
                     for k, r in [('output', 'out'), ('merge', 'merge')]}
    fixture = json.loads((a.fixture / 'fixture.json').read_text())
    # The packed fixture must originate from this exact input and captured weight image.
    packed = lf.WeightPack(P, 'enc1')
    if (not np.array_equal(np.fromfile(a.fixture / 'in.bin', np.uint8), raw['input']) or
            not np.array_equal(np.fromfile(a.fixture / 'weights.bin', np.float32), packed.data)):
        raise RuntimeError('Sette fixture input or weights differ from capture')
    report['sette_fixture_sha256'] = sha(a.fixture / 'fixture.json')
    report['sette_geometry'] = fixture['geometry']
    PT, translated = lo.raw_layer(str(a.translated), 'enc1')
    if not np.array_equal(P.wbuf, PT.wbuf) or not np.array_equal(raw['input'], translated['input']):
        raise RuntimeError('translated input or weights differ')
    data['translated'] = {k: translated[k] for k in ('output', 'merge')}
    translated_root = a.translated.parent.parent
    log = translated_root / 'probe.log'
    report['translated_control'] = {'dispatch': verify_translated(log.read_text(errors='replace'), KERNEL),
        'log_sha256': sha(log), 'run_sha256': sha(translated_root / 'run.json'),
        'configuration': json.loads((translated_root / 'run.json').read_text())}
    for left, right in itertools.combinations(data, 2):
        report['comparisons'][left + '-vs-' + right] = {k: metrics(data[left][k], data[right][k]) for k in data[left]}
    report['field_hashes'] = {name: {k: hashlib.sha256(v).hexdigest() for k, v in value.items()} for name, value in data.items()}
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))
    if not all(report['runs'][k + '-deterministic'] for k in ('fast', 'conservative', 'sette')):
        raise RuntimeError('nondeterministic execution; see report.json')
    print(json.dumps(report['comparisons'], indent=2))


if __name__ == '__main__':
    main()
