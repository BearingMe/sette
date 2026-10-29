#!/usr/bin/env python3
"""Replay captured preset-M layers with d4r's native probe and compare exactly.

For every `replay-*-rrlite_*` directory in CAPTURE, run
`d4r_native_replay_probe.exe` with the matching `.hsaco` module and compare
each reproduced allocation byte-for-byte with the capture's `gpu-output/`
files. With `--repeat`, run every layer twice and also require the two native
runs to agree.

usage: replay_m_layers.py CAPTURE --probe PROBE.exe --modules DIR --hip-root DIR --output DIR
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_of(directory):
    with open(directory / "manifest.txt", encoding="utf-8") as stream:
        first = stream.readline().strip()
    if not first.startswith("kernel "):
        raise SystemExit(f"{directory}: manifest has no kernel line")
    return first.split(None, 1)[1]


def run_layer(probe, modules, hip_root, directory, output):
    kernel = kernel_of(directory)
    module = modules / f"{kernel}.hsaco"
    if not module.is_file():
        raise SystemExit(f"missing module: {module}")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(probe), "--hip-root", str(hip_root), "--module", str(module),
        "--fixture-dir", str(directory), "--output-dir", str(output),
        "--iterations", "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{directory.name}: replay exit={result.returncode}")
    return kernel


def compare(directory, output):
    expected = sorted((directory / "gpu-output").glob("alloc-*.bin"))
    if not expected:
        raise SystemExit(f"{directory}: capture has no gpu-output allocations")
    mismatches = []
    for source in expected:
        target = output / source.name
        if not target.is_file() or sha(source) != sha(target):
            mismatches.append(source.name)
    return len(expected), mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--modules", required=True, type=Path)
    parser.add_argument("--hip-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()

    directories = sorted(d for d in args.capture.iterdir() if d.is_dir() and "-rrlite_" in d.name)
    if not directories:
        raise SystemExit("no rrlite capture directories")
    if args.output.exists():
        raise SystemExit(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)

    failures = 0
    for directory in directories:
        first = args.output / directory.name / "first"
        kernel = run_layer(args.probe, args.modules, args.hip_root, directory, first)
        count, mismatches = compare(directory, first)
        repeat = ""
        if args.repeat:
            second = args.output / directory.name / "second"
            run_layer(args.probe, args.modules, args.hip_root, directory, second)
            deterministic = all(
                sha(first / name) == sha(second / name) for name in (p.name for p in first.glob("alloc-*.bin"))
            )
            repeat = f" repeat={'exact' if deterministic else 'DIFF'}"
            failures += not deterministic
        status = "EXACT" if not mismatches else "MISMATCH " + ",".join(mismatches)
        failures += bool(mismatches)
        print(f"{kernel:24s} files={count} {status}{repeat}")
    print(f"layers={len(directories)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
