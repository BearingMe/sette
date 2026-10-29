# EXP-002: Preset M FP16 Oracle Blocker

**Milestone:** M05
**Status:** BLOCKED

## Question

Can a reproducible Preset M FP16 reference execution be established on this workstation, and does d4r's Windows branch remove the requirement for a captured NGX weight upload?

## Method

- Inspected the pinned d4r submodule at `05e19d4` and its remote refs.
- Found `origin/windows` at `7da2f9ac399404c23b2afe2dc26f245fca16fffb` (2026-10-18) and cloned that branch, depth 1, into a temporary workspace. The pinned `references/sources/d4r` checkout was not modified.
- Read `docs/windows-native-engine.md`, `docs/windows-local-package.md`, `docs/windows-sr-optimization.md`, `docs/windows-gpu-support.md`; the M capture/replay scripts (`test-m-captures.ps1`, `test-m-replay.ps1`, `stage-native-m.ps1`); the native build scripts; and the CMake target list for `d4r_d3d12_dlss_probe`.
- Downloaded the public Windows release `v0.1.5a` (`d4r-0.1.5a-windows-x64.zip`, 146,259,930 bytes) and listed every archive entry.
- Searched the local temporary workspace for `weights8.bin` and `hipnet.bin`.
- Checked the WSL2 Ubuntu environment for CMake, glslang, Vulkan tools, and ROCm.
- Started `scripts/windows/setup-windows-tools.ps1 -RuntimeProfile therock`; it was stopped before any build. The log is empty and no build result is claimed.

## Results

- [S] The Windows branch provides native Windows runners (`d4r-k.exe`, `d4r-m.exe`), a D3D12 DLSS probe target (`d4r_d3d12_dlss_probe`), TheRock HIP build scripts, and per-layer M capture validation scripts.
- [S] The branch's own documentation states: "A complete model needs a real captured NGX upload. The current native Windows runner does not provide a model-capture front end or D3D12 game adapter." It also states that Super Resolution weights are packed inside `nvngx_dlss.dll` and are "read where NGX unpacks them"; the harness renders a frame per preset to obtain them.
- [S] The public Windows release contains 1,375 entries: the d4r shim/architecture loader, ZLUDA, HIP runtime, `rrlite_*` M kernels for each gfx target, and model tools (`compile_k.py`, `compile_m.py`, Python, NumPy, glslang). It contains no `weights8.bin`, no capture, no `d4r-m.exe`, and no DLSS probe executable.
- [S] `stage-native-m.ps1` only copies native M kernels and writes a manifest; `test-m-captures.ps1` requires an existing `CaptureDirectory` with `replay-*-rrlite_*` directories. Neither produces weights.
- [U] No `weights8.bin` or `hipnet.bin` exists anywhere in the local temporary workspace.
- [S] WSL2 Ubuntu exists but has no CMake, glslang, Vulkan tools, or ROCm; this does not matter because the missing weights would block a Linux build too.
- [U] The setup/build/probe path was not completed, so neither the Windows engine build nor NGX execution of preset 13 on gfx1101 was tested.

## Interpretation

- [S] The Windows branch changes build feasibility, not data availability. It still requires an NGX weight upload captured from a real DLSS execution.
- [U] Whether `d4r_d3d12_dlss_probe.exe` can run preset 13 on gfx1101 through ZLUDA and produce a usable capture is untested.
- [U] Compatibility between the local 310.7.0 feature DLL, the Windows release's pinned 310.9.1 tooling, and the 617.14 NGX core is untested.
- **M05 cannot establish a Preset M oracle from the available artifacts.** No substitute oracle was created from our own reconstruction, because that would derive the reference from the implementation under test.

## Next

To resume M05, in order:

1. Provision the pinned tools: `scripts/windows/setup-windows-tools.ps1 -RuntimeProfile therock`.
2. Build the probe, shim, and native engine: `scripts/windows/build-windows-rdna4.ps1 -RuntimeProfile therock -GpuArch gfx1101` (installs under `dist/windows-rdna4-therock-gfx1101`).
3. Obtain an NGX core with `scripts/windows/install-dlss.ps1` (about 945 MiB driver archive; NVIDIA terms; keep local and outside version control).
4. Run `d4r_d3d12_dlss_probe.exe` with `--preset 13` and the M capture environment (`D4R_CUDA_REPLAY_DUMP_DIR`, `D4R_CUDA_REPLAY_DUMP_FILTER=rrlite_`, a dump limit high enough for all launches).
5. Compile the capture with `compile_m.py CAPTURE OUT` using the packaged Python/NumPy/glslang, then run `d4r-m.exe` on the model as the oracle.

Risks: NGX may refuse the AMD adapter, the 310.7 feature DLL may not match the probe/core, and the capture may miss weight allocations. Captures and weights are NVIDIA-derived, must stay local, and must never be committed.

## Files

- d4r Windows branch: `origin/windows` at `7da2f9a` (temporary clone, not part of this repository).
- Public release: `d4r-0.1.5a-windows-x64.zip` (temporary local artifact).
- No repository files were produced by the aborted build.
