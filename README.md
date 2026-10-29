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

| Area | Project target | M01 status |
|---|---|---|
| Network | DLSS 4.5 Super Resolution, Preset M | Exact model/build and artifact availability are [U] unknown. |
| Original weights | FP8 is the stated candidate representation | [U] Must be verified against source/artifacts; do not treat as established. |
| GPU | AMD Radeon RX 7700 XT, target architecture gfx1101 | [M] Windows enumerates an RX 7700 XT. The architecture/ISA mapping and required feature support remain [U] until source/documentation and hardware probes verify them. |
| Platform/API | Windows x64, D3D12 | Windows 11 Pro x64 is present; D3D12 device/feature support has not been tested. |
| Shader toolchain | HLSL and DXC | DXC is not found on the current PATH; exact compiler and shader-model versions are [U]. |
| Host orchestration | Prefer Rust where practical | Rust/Cargo 1.98.1 are installed on the current workstation; compatibility with the eventual reference/build path is not yet established. |

Evidence tags follow `docs/CONVENTIONS.md`: `[D]` documentation, `[S]` source observation, `[R]` reverse engineering, `[M]` measurement, `[H]` hypothesis, and `[U]` unknown. For current local environment observations, see [Environment and setup](#environment-and-setup). Distinguish original NVIDIA behavior, d4r behavior, conservative FP16 oracle behavior, optimized FP16 behavior, and this project's INT8 behavior; they are not presumed numerically identical.

## Scope

**In scope:** Preset M architecture and weights; graph, tensor layouts, and execution dependencies; FP8/FP16 numerical behavior; INT8 weights and activations; RDNA 3 WMMA and DP4A; HLSL/D3D12 kernels; differential tensor validation; temporal state; standalone inference; and reproducible memory, scheduling, and performance analysis.

**Out of scope:** Other DLSS presets, Ray Reconstruction, Frame Generation, FSR, NGX/DXGI interception, game-specific integration, unrelated GPU architectures, and a generalized ML framework.

The work is organized in the [milestone roadmap](docs/MILESTONES.md). M01 is in progress. This initialization does **not** implement neural kernels, quantization, or inference scaffolding, and does not authorize starting M02.

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
- `dxc.exe` was not found by `Get-Command` or `where.exe` on PATH. This does not establish whether another non-PATH copy exists.
- No D3D12 feature query, shader compilation, GPU dispatch, or target-runtime validation has been run.

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

This is an inventory, not a validated build environment. Exact compatible SDK/DXC/driver versions, source-specific dependencies, and reproducible build/run commands remain open until the relevant reference paths are inspected and pinned. Do not infer successful D3D12 execution from device enumeration.

## Repository organization

| Location | Responsibility |
|---|---|
| `AGENTS.md` | Generic Trace operating rules; preserved unchanged. |
| `docs/MILESTONES.md` | Authoritative progress, checkpoints, dependencies, and validation gates. |
| `docs/CONVENTIONS.md` | Evidence, experiment, finding, and validation rules. |
| `docs/experiments/` | Reproducible probes and observations. |
| `docs/findings/` | Consolidated technical knowledge. |
| `references/` | Reference index, provenance, and pinned sources. |
| `scripts/` | Reusable investigation utilities. |
| `src/` | Reconstruction code, only when supported by evidence. |
| `tests/` | Independent numerical and regression validation. |

The [reference index](references/README.md) currently records candidate sources and their unpinned status; no upstream repository or model artifact has been acquired in M01. Repository state and evidence must remain sufficient to resume without AgentMemory, MemPalace, or chat history.
