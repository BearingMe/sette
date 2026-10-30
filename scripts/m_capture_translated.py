#!/usr/bin/env python3
"""Capture one original network kernel via ZLUDA, excluding only its native replacement.

Creates a private native directory and fresh capture; never edits the installed
runtime or an existing capture. The result is translated execution on AMD, not
execution on NVIDIA hardware. Logs must prove backend=translated at dispatch.
`--kernel` selects the kernel (default: the enc1 control); the other native
modules stay in place, so the captured input is whatever the real network
produced for that launch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from m_capture_provenance import verify_translated


KERNEL = 'rrlite_enc1_4x4'
KERNEL_DEFAULT = KERNEL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    temp = Path(os.environ['LOCALAPPDATA']) / 'Temp/opencode'
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kernel', default=KERNEL_DEFAULT,
                    help=f'captured/verified kernel (default {KERNEL_DEFAULT})')
    ap.add_argument('--exclude', action='append', default=None,
                    help='kernel(s) removed from the native allowlist; repeatable. '
                         'Defaults to --kernel. Excluding every network kernel runs '
                         'the whole chain translated (mode B).')
    ap.add_argument('--limit', type=int, default=1,
                    help='captured dispatches of this kernel (tube kernel: 6)')
    ap.add_argument('--runtime', type=Path, default=temp / 'd4r-runtime/d4r')
    ap.add_argument('--probe-build', type=Path, default=temp / 'd4r-win-src/build/probe2')
    ap.add_argument('--hip-root', type=Path, default=Path(r'C:\Program Files\AMD\ROCm\7.2'))
    ap.add_argument('--cuda', type=Path)
    ap.add_argument('--timeout', type=int, default=900)
    a = ap.parse_args()
    a.out = a.out.resolve()
    KERNEL = a.kernel
    excluded = sorted(set(a.exclude or [KERNEL]))
    a.out.mkdir(parents=True, exist_ok=False)
    native_source = a.runtime / 'native/gfx1101'
    native = a.out / 'native-except-translated'
    native.mkdir()
    manifest = (native_source / 'd4r-kernels.txt').read_text()
    lines = [line for line in manifest.splitlines()
             if not line.split() or line.split()[0] not in excluded]
    present = {line.split()[0] for line in manifest.splitlines() if line.split()}
    assert set(excluded) <= present, sorted(set(excluded) - present)
    (native / 'd4r-kernels.txt').write_text('\n'.join(lines) + '\n')
    for source in native_source.glob('*.hsaco'):
        if source.stem not in excluded:
            shutil.copy2(source, native / source.name)
    assert not any((native / (name + '.hsaco')).exists() for name in excluded)
    cuda = a.cuda or a.runtime / 'zluda-c10/nvcuda.dll'
    env = os.environ.copy()
    # Inherited EXTRA_BC, FAST_MATH or another native/cache selection must not
    # silently alter this control. Only the explicit settings below are allowed.
    removed = sorted(k for k in env if k.startswith(('D4R_', 'ZLUDA_')))
    for key in removed:
        del env[key]
    settings = {
        'D4R_ZLUDA_NATIVE_DIR': str(native),
        'D4R_ZLUDA_VERBOSE': '1',
        'D4R_CUDA_REPLAY_DUMP_DIR': str(a.out / 'capture'),
        'D4R_CUDA_REPLAY_DUMP_FILTER': KERNEL,
        'D4R_CUDA_REPLAY_DUMP_LIMIT': str(a.limit),
        'D4R_ZLUDA_DUMP_PTX_DIR': str(a.out / 'ptx'),
        'D4R_NVAPI_BACKEND': str(a.runtime / 'nvapi/backend/nvapi64.dll'),
        'D4R_DIAG_DIR': str(a.out / 'frame'),
        'ZLUDA_CACHE_DIR': str(a.out / 'jit-cache'),
        'D4R_ZLUDA_WMMA': '1',
        'D4R_ZLUDA_WMMA_FP8': '1',
        'D4R_ZLUDA_WMMA_FP8_NATIVE': '0',
        'D4R_ZLUDA_WMMA_F16_REFERENCE': '1',
    }
    env.update(settings)
    command = [str(a.probe_build / 'd4r_d3d12_dlss_probe.exe'),
               '--hip-root', str(a.hip_root), '--cuda-dll', str(cuda),
               '--module', str(a.probe_build / 'd4r_nvngx.dll'),
               '--ngx-core', str(a.runtime / 'ngx/_nvngx.dll'),
               '--dlss-dll', str(repo / 'references/binaries/nvngx_dlss.dll'),
               '--nvapi-dll', str(a.runtime / 'nvapi/gfx1101/nvapi64.dll'),
               '--ngx-output-resolution', '1920x1080', '--preset', '13',
               '--iterations', '1']
    report = {'command': command, 'environment': settings, 'kernel': KERNEL,
              'excluded_kernels': excluded,
              'removed_environment_keys': removed,
              'native_manifest_sha256': sha(native / 'd4r-kernels.txt'),
              'source_native_manifest_sha256': sha(native_source / 'd4r-kernels.txt'),
              'binaries': {str(p): sha(p) for p in (
                  Path(command[0]), cuda, a.probe_build / 'd4r_nvngx.dll',
                  repo / 'references/binaries/nvngx_dlss.dll')},
              'status': 'running'}
    report_path = a.out / 'run.json'
    report_path.write_text(json.dumps(report, indent=2))
    try:
        with (a.out / 'probe.log').open('wb') as log:
            done = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=a.timeout)
        report['returncode'] = done.returncode
    except subprocess.TimeoutExpired:
        report['status'] = 'timeout'
        report_path.write_text(json.dumps(report, indent=2))
        print('FAIL: timed out; log retained at', a.out / 'probe.log')
        return 1
    text = (a.out / 'probe.log').read_text(errors='replace')
    report['translated_dispatches'] = text.count(
        f'[d4r-launch] kernel="{KERNEL}" backend=translated')
    report['native_dispatches'] = text.count(
        f'[d4r-launch] kernel="{KERNEL}" backend=native')
    report['native_direct'] = f'module_direct kernel="{KERNEL}"' in text
    try:
        report['verified_dispatch'] = verify_translated(text, KERNEL, a.limit)
        verified = True
    except ValueError as error:
        report['dispatch_error'] = str(error)
        verified = False
    captures = sorted((a.out / 'capture').glob('*' + KERNEL))
    report['captures'] = [str(p) for p in captures]
    complete = False
    if len(captures) == a.limit:
        outputs = {}
        for capture in captures:
            allocations = {}
            for line in (capture / 'manifest.txt').read_text().splitlines():
                words = line.split()
                if words and words[0] == 'alloc':
                    allocations[words[1]] = int(words[3])
            for i, size in allocations.items():
                outputs[f'{capture.name}/alloc-{i}'] = (
                    capture / 'gpu-output' / f'alloc-{i}.bin', size)
        complete = bool(outputs) and all(p.exists() and p.stat().st_size == size
                                         for p, size in outputs.values())
        report['output_allocations'] = {i: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                                        for i, (p, size) in outputs.items() if p.exists()}
    report['ptx_modules'] = [
        {'path': str(p), 'sha256': sha(p)} for p in (a.out / 'ptx').glob('*.ptx')
        if f'.entry {KERNEL}(' in p.read_text()]
    passed = (done.returncode == 0 and verified and
              report['translated_dispatches'] == a.limit and
              not report['native_dispatches'] and not report['native_direct'] and complete
              and len(report['ptx_modules']) == 1)
    report['status'] = 'pass' if passed else 'fail'
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('status', 'returncode',
          'translated_dispatches', 'native_dispatches', 'captures')}, indent=2))
    print('Log:', a.out / 'probe.log')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
