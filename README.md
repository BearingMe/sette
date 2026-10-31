# DLSS 4.5 Preset M INT8 Reconstruction

This project investigates and reconstructs DLSS 4.5 Super Resolution Preset M, targeting native INT8 matrix arithmetic on an AMD Radeon RX 7700 XT (project target: gfx1101) under Windows, D3D12, HLSL, and DXC. Trace's evidence classifications, milestone gates, and validation rules remain authoritative; see [Research Conventions](docs/CONVENTIONS.md).

The goal is an independently validated reconstruction—not a claim of bit-exact equivalence to NVIDIA hardware.

## Ordered objectives

1. **Execution:** Reconstruct and execute the complete network, using native INT8 matrix operations for its designated matrix work. Unacceptable image quality is acceptable at this stage; a working but ugly output is valid progress.
2. **Quality:** Improve quantization fidelity until outputs are meaningfully comparable with the reference execution.
3. **Performance:** Optimize specifically for gfx1101, measuring latency, memory traffic, resource use, and complete inference time.

Do not trade away execution completeness for early optimization. Floating-point operations may remain where justified; every FP16/FP32 exception must be explicit. The goal is native INT8 matrix arithmetic, not forcing every operation into integer precision.

## Research target and current evidence

The target values below are project requirements from the initialization brief. They are not, by themselves, evidence about NVIDIA's model or the reference implementation.

| Area | Project target | Current evidence |
|---|---|---|
| Network | DLSS 4.5 Super Resolution, Preset M | [R] The local DLL 310.7.0 contains `rrlite_*` M code; [M] all 360 of its matrices (1 553 408 e4m3 values) are extracted and decoded from a captured frame ([EXP-009](docs/experiments/EXP-009-m07-complete-inventory.md)), while the `mvlo` variant the capture cannot produce remains [U]. d4r implements a documented M reference pipeline and pins the same DLL build. |
| Original weights | FP8 is the stated candidate representation | [S] d4r documents FP8 (e4m3) M weights and activations; [M] the captured weights are e4m3 and decode losslessly to f16 ([EXP-009](docs/experiments/EXP-009-m07-complete-inventory.md)). They are derived from the user's DLL and are not redistributable. |
| GPU | AMD Radeon RX 7700 XT, target architecture gfx1101 | [M] Enumerated and exercised in EXP-001; native INT8 and FP16 WMMA confirmed on the device. The complete ISA/feature inventory remains partial. |
| Platform/API | Windows x64, D3D12 | [M] D3D12 compute and native WMMA executed in EXP-001; a full feature-level inventory has not been recorded separately. |
| Shader toolchain | HLSL and DXC | [M] DXC 1.8.2502.11 is installed with Windows SDK 10.0.26100.0 and compiled the EXP-001 shaders; it is not on `PATH`. |
| Host orchestration | Prefer Rust where practical | [M] Rust/Cargo 1.98.1, CMake, MSVC, and Python are installed; the pinned quattro reference builds with CMake/MSVC/Python. Rust integration remains unproven. |

Evidence tags follow `docs/CONVENTIONS.md`: `[D]` documentation, `[S]` source observation, `[R]` reverse engineering, `[M]` measurement, `[H]` hypothesis, and `[U]` unknown. For current local environment observations, see [Environment and setup](#environment-and-setup). Distinguish original NVIDIA behavior, d4r behavior, conservative FP16 oracle behavior, optimized FP16 behavior, and this project's INT8 behavior; they are not presumed numerically identical.

## Scope

**In scope:** Preset M architecture and weights; graph, tensor layouts, and execution dependencies; FP8/FP16 numerical behavior; INT8 weights and activations; RDNA 3 WMMA and DP4A; HLSL/D3D12 kernels; differential tensor validation; temporal state; standalone inference; and reproducible memory, scheduling, and performance analysis.

**Out of scope:** Other DLSS presets, Ray Reconstruction, Frame Generation, FSR, NGX/DXGI interception, game-specific integration, unrelated GPU architectures, and a generalized ML framework.

The work is organized in the [milestone roadmap](docs/MILESTONES.md). Phase I (M01–M04) and the Phase II probes M06–M08 are complete. M05's capture is complete at 1920×1080 for both resource kinds: every stage group including the `enc0`/`dec0` weight blobs ([EXP-008](docs/experiments/EXP-008-m05-full-capture.md)), and every texture and surface the image-bound stages bind, 33 targets whose contents reproduce the stage edges and the final frame byte for byte ([EXP-011](docs/experiments/EXP-011-m05-image-capture.md)). The layer-scope oracle replays bit-exactly ([EXP-003](docs/experiments/EXP-003-m05-layer-capture.md)). The native reconstruction now covers every stage: the fixture path and `downsample` reproduce their captures byte for byte ([EXP-012](docs/experiments/EXP-012-m05-native-downsample.md)), `dec0`'s operand and weight mapping comes from the kernel's own PTX ([EXP-014](docs/experiments/EXP-014-m05-dec0-mapping.md)), `enc0` and `post` are byte-exact ([EXP-015](docs/experiments/EXP-015-m05-enc0-contract.md), [EXP-016](docs/experiments/EXP-016-m05-post-reconstruction.md)), and the ten-layer network runs natively but is not yet byte-faithful: the chained `dec1` field reproduces 21.3% of the capture, so B5 remains in progress ([EXP-019](docs/experiments/EXP-019-m05-native-network.md)). M07 has closed the weight side: all 360 matrices (1 553 408 e4m3 values) are extracted and quantized at five scopes ([EXP-009](docs/experiments/EXP-009-m07-complete-inventory.md)). M09 and M10 are complete for the bounded matrix and reset-frame execution
scope; cross-frame temporal reuse and the 45 dB image gate remain open. M12/M13
measured the static quantization alternatives: per-K-block weight scaling was
rejected, and asymmetric activation encoding for the two non-negative operands
is accepted as an opt-in configuration, raising the canonical frame to 44.4026 dB
with 76/76 INT8 operators ([EXP-041](docs/experiments/EXP-041-quality-recovery.md));
the 45 dB gate still fails. M14 now has a reproducible GPU-interval baseline,
host-phase and VRAM measurements ([EXP-042](docs/experiments/EXP-042-performance-baseline.md));
M15-A adds a production/diagnostic graph split and a bounded placed-resource
arena with byte-identical outputs ([EXP-043](docs/experiments/EXP-043-graph-lifetime-memory-reuse.md)).

## Trace research workflow

Follow Trace's sequence: define a bounded question, inspect primary sources, run reproducible experiments where needed, reconstruct only behavior supported by evidence or explicit assumptions, validate against independent expectations, and consolidate durable findings. Evidence tags express provenance, not confidence. Compilation or plausible-looking output alone is not correctness or fidelity evidence. The unchanged [`AGENTS.md`](AGENTS.md) and [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) remain authoritative for agent behavior, evidence, experiments, and validation.

## Initial experiment constraints

- Preset M only, with one model version and deterministic inputs where possible.
- Proposed output resolution: **1920×1080**.
- Proposed quality setting: **Quality**.
- These are provisional assumptions, not a fixed test configuration. The reference execution path must first be inspected to verify that it supports them. The corresponding input dimensions and any required settings will not be inferred.
- Keep the conservative FP16 reference path unchanged. Do not optimize for visual appearance alone or silently change numerical behavior.
- Do not introduce game integration. Until M10 passes, prioritize complete execution over image quality and performance.

## Environment and setup

### Target requirements

- Windows x64 with an AMD Radeon RX 7700 XT and a driver/runtime that supports the required D3D12 and shader features.
- D3D12 development/runtime components and a compatible Windows SDK.
- DXC with the HLSL, shader-model, and AMD intrinsic support established by the pinned references.
- Git for source provenance and pinned read-only references.
- Rust stable is preferred for host orchestration, subject to the reference implementation's actual requirements.
- Additional dependencies must be justified by inspected source and recorded with versions in the reference/build procedure.

### M01 workstation inventory

Observed on **2026-10-28** on the current Windows workstation (local inventory, `[M]`):

- Windows 11 Pro, version 10.0.26200, 64-bit.
- Device enumeration reports AMD Radeon RX 7700 XT; installed AMD driver version `32.0.32015.7029` (driver date reported as 2026-10-17).
- Windows SDK include directories include `10.0.26100.0` (other older SDK directories are also present).
- `rustc` and `cargo` report version `1.98.1`.
- `dxc.exe` is not on `PATH`. `[M]` DXC 1.8.2502.11 is present at `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe`; EXP-001 used it to compile the reference shaders.
- D3D12 shader compilation and GPU dispatch were exercised by EXP-001; a full D3D12 feature-level query and Preset M runtime validation have not been performed.

Repeat the host inventory in PowerShell with:

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture
Get-PnpDevice -Class Display | Select-Object Status, FriendlyName, InstanceId
Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -eq 'AMD Radeon RX 7700 XT' } | Select-Object DeviceName, DriverVersion, DriverDate, Manufacturer
Get-ChildItem "$env:ProgramFiles(x86)\Windows Kits\10\Include" -Directory | Select-Object -ExpandProperty Name
rustc --version
cargo --version
Get-Command dxc -ErrorAction SilentlyContinue
```

The Windows/D3D12 path is no longer only an inventory: EXP-001 configured and built the pinned quattro reference from a clean archive with VS 2022, Windows SDK 10.0.26100.0, and DXC 1.8.2502.11, then executed its INT8 and FP16 WMMA probes successfully on the RX 7700 XT. d4r's Linux/Vulkan/HIP oracle build remains unverified. Do not infer Preset M execution from these WMMA probes.

## Repository organization

| Location | Responsibility |
|---|---|
| `AGENTS.md` | Generic Trace operating rules; preserved unchanged. |
| `docs/MILESTONES.md` | Authoritative progress, checkpoints, dependencies, and validation gates. |
| `docs/CONVENTIONS.md` | Evidence, experiment, finding, and validation rules. |
| `docs/experiments/` | Reproducible probes and observations. |
| `docs/findings/` | Consolidated technical knowledge. |
| `patches/` | Project-owned deltas to a pinned reference, applied outside it. |
| `references/` | Reference index, provenance, and pinned sources. |
| `scripts/` | Reusable investigation utilities. |
| `src/` | Reconstruction code, only when supported by evidence. |
| `tests/` | Independent numerical and regression validation. |

The [reference index](references/README.md) records four pinned source submodules and the local DLL, with inspection status and limitations. No redistributable Preset M model has been acquired: the weights this project measures are extracted at run time from the user's own `nvngx_dlss.dll` and stay out of Git. Repository state and evidence must remain sufficient to resume without AgentMemory, MemPalace, or chat history.
