# EXP-001: Reproduce the pinned quattro D3D12/WMMA reference on this workstation

**Milestone:** M01, M02
**Status:** PASS

## Question

Can the pinned quattro reference (`ad0482a`) be built from a clean source archive and execute its native INT8 and FP16 WMMA experiments on this workstation using the installed Windows/D3D12/DXC toolchain?

Acceptance criteria, defined before execution:

1. CMake configures with the Visual Studio 2022 x64 generator and the pinned SDK/DXC paths.
2. The Release build completes.
3. `rdna3-int8.exe run wmma-int8` and `run wmma-fp16` both exit 0 and report `PASS`.

## Method

- Source: `references/sources/quattro` at `ad0482a952b1af80b8be2dcaa6a49e098aa57fc1`, exported with `git archive` to a temporary directory. The pinned submodule was not modified and no build output was written inside it.
- Build: `cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release`, then `cmake --build ... --config Release`.
- Compiler/runtime as reported by the build and probe:

| Component | Observed |
|---|---|
| OS | Windows 11 Pro, build 10.0.26200 |
| GPU | AMD Radeon RX 7700 XT, `gfx1101`, 12,080 MiB |
| Driver | 32.0.32015.7029 (from the M01 inventory) |
| MSVC | 19.44.35229.0 (VS 2022 Community 17.14.43) |
| Windows SDK | 10.0.26100.0 |
| DXC | 1.8.2502.11 at `Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe` (not on `PATH`) |
| CMake | 4.4.3 |

Commands, using a temporary working directory:

```powershell
$tmp = Join-Path $env:TEMP 'sette-quattro-verify'
git -C references/sources/quattro archive --format=zip -o "$tmp/src.zip" HEAD
Expand-Archive "$tmp/src.zip" "$tmp/src"
cmake -S "$tmp/src" -B "$tmp/build" -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release
cmake --build "$tmp/build" --config Release --parallel
& "$tmp/src/build/bin/rdna3-int8.exe" run wmma-int8
& "$tmp/src/build/bin/rdna3-int8.exe" run wmma-fp16
```

CMake reported that `FSR4_REFERENCE_MODEL` was not found, so the `fsr4-oracle` experiment is unavailable. This was expected because quattro's nested reference submodules are uninitialized and does not affect the two WMMA experiments.

## Results

The Release build completed and produced `rdna3-int8.exe`. Both acceptance experiments exited 0.

INT8 WMMA (`wmma-int8`):

```text
adapter: AMD Radeon RX 7700 XT (vid=0x1002 did=0x747E, 12080 MiB)
amd extensions: intrinsics yes, analyzer yes
intrinsics: waveSize 32 (min 32, max 64), WaveMatrix support: yes
wmma i8*i8             | correct OK (0 mismatches) | idempotent OK | wmma x 1 dot4 x 0
wmma u8*u8             | correct OK (0 mismatches) | idempotent OK | wmma x 1 dot4 x 0
wmma i8*u8             | correct OK (0 mismatches) | idempotent OK | wmma x 1 dot4 x 0
wmma u8*i8             | correct OK (0 mismatches) | idempotent OK | wmma x 1 dot4 x 0
dp4a i8*i8 (control)   | correct OK (0 mismatches) | idempotent OK | wmma x 0 dot4 x 4
diagnostics: no unexplained error-severity messages (128 known-benign GBAV reports)
int8 wmma probe: PASS
```

FP16 WMMA (`wmma-fp16`):

```text
fp16 wmma ->f32 acc    | maxdiff 1.19209e-07 (tol 0.002) OK   | wmma_f32 x1 wmma_f16 x0
fp16 wmma ->f16 acc    | maxdiff 0 (tol 0.06) OK              | wmma_f32 x0 wmma_f16 x1
int8 wmma  min@K64 73.96us  min@K256 189.36us  scale 2.56x
fp16 wmma  min@K64 71.64us  min@K256 253.48us  scale 3.54x
int8 dp4a  min@K64 16.52us  min@K256  80.52us  scale 4.87x
fp16 wmma + comparison: PASS
```

The measured minima differ slightly from quattro's recorded EXP-004/EXP-005 values (for example, INT8 i8×i8 13.920 µs here versus 13.720 µs recorded). Correctness, instruction classification, and PASS/FAIL outcomes agree; the timing difference is consistent with run-to-run clock state and is not treated as a performance result.

## Interpretation

- `[M]` The pinned quattro reference is buildable and executable on this workstation with the installed VS 2022, Windows SDK 10.0.26100.0, DXC 1.8.2502.11, and driver 32.0.32015.7029. The Windows/D3D12 reference path is reproducible here.
- `[M]` Native INT8 WMMA (`v_wmma_i32_16x16x16_iu8`) and FP16 WMMA (`v_wmma_f32_16x16x16_f16`, `v_wmma_f16_16x16x16_f16`) execute correctly on this RX 7700 XT, with zero mismatches for INT8 and 1 ULP / bit-exact results for FP16, independently reproducing quattro's recorded outcomes.
- `[M]` The DP4A control contains no WMMA instructions, so the instruction-class check is not vacuous.
- `[S]` The experiment does not test d4r, the Preset M network, FP8 weights, or 1920×1080 output. It does not satisfy any later milestone gate.
- `[U]` d4r's Linux/Vulkan/HIP oracle path remains unbuilt and unverified on this host. That is deferred to M05.

## Next

Build and execute d4r's standalone M path in M05 when the Linux/ROCm environment is available; use this verified Windows/D3D12 path for operator and INT8 kernel work in later milestones.
