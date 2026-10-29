# Reference Inventory and Preset M Artifact Evidence

## Summary

M02 acquired and inspected the four requested source references and the local NVIDIA DLL. d4r is the closest implementation oracle for Preset M, quattro supplies validated gfx1101 D3D12 INT8/FP16 WMMA mechanisms, the NVIDIA DLSS SDK supplies only API headers, and TransformerEngine supplies NVIDIA FP8 recipe context but no DLSS material. The local DLL contains Preset M PTX entry points and matches d4r's pinned 310.7.0 build, but no weights were extracted, and no redistributable model or weights were found in any acquired source.

## Findings

### Acquired references

| Reference | Pin | Scope acquired | Relevant result |
|---|---|---|---|
| d4r | `05e19d402781083118d106f2a21d0cf2bc43f744` (2026-10-27) | Full submodule | Preset M engine, stage algorithms, FP8/f16 WMMA kernels, weight-extraction tools |
| quattro | `ad0482a952b1af80b8be2dcaa6a49e098aa57fc1` (2026-10-28) | Full submodule; nested `.references/sources/*` submodules uninitialized | Validated gfx1101 D3D12 INT8 and FP16 WMMA probes, ISA inspection, quantization and fixed-network experiments |
| NVIDIA DLSS | `9981c6350dc58c0ce21f024fd65e31e851c31689` (`v310.9.16`) | Shallow, sparse `include/` only | NGX API headers and license; no model, weights, or inference runner |
| NVIDIA TransformerEngine | `dfc4158d1597f72972fe2863adac436fa3fe1d4d` (2026-10-28) | Shallow, sparse `transformer_engine/common/{recipe,cast,include}` | CUDA FP8 scaling recipes only; no DLSS or AMD relevance established |
| Local `nvngx_dlss.dll` | SHA-256 `be6e434a...b6ee6e`, file version `310,7,0,0` | Local, gitignored | Matches d4r's pinned 310.7.0 build; contains Preset M (`rrlite_*`) PTX; weights not extracted or verified |

### d4r

- `[S]` d4r states that it supports DLSS 4.5 transformer preset M and lists gfx1101 (RX 7700 XT/7800 XT/7700, PRO W7700) with native WMMA kernels in `README.md`. It also reports RX 7700 XT runtime testing for preset M.
- `[S]` Preset M is documented as an input/tokenization stage, ten Swin-style FP8 layers, token expansion, reconstruction at 1.5× the output size, and a 3→2 downsample in `engine/README.md` §"Preset M" and `engine/M_NOTES.md`. Captured-launch reproduction is reported byte-identical for the ten network layers.
- `[S]` On gfx11, d4r's preset M kernels use wave32 WMMA with binary16 operands/accumulation; `kernels/m/swin_common.h` states that no native FP8 instruction is used and quantized intermediates remain binary16. Native FP8 WMMA is gated to gfx12 (`D4R_FP8_WMMA`).
- `[S]` The standalone path is the native Vulkan engine: `MModel`/`MEngine` in `engine/runtime.h` and `engine/runtime_m.inc`, the `d4r-m` runner (`engine/replay_m.cpp`), and the D3D12 game shim in `engine/game_m.cpp`. Build requirements are Linux, Vulkan, ROCm 7.2/HIP for the native network, and the d4r toolchain listed in `README.md`.
- `[S]` M model directories are built from a captured frame's `rrlite_enc0`, `rrlite_dec0`, and ten layer launches. `engine/compile_m.py` writes `weights8.bin` (FP8), `weights16.bin` (f16 vectors/dither), `offsets.bin`, and compiled shaders. Its docstring and `engine/README.md` state that the weights are read from NVIDIA's DLL and are not redistributable with d4r.
- `[S]` d4r's release install pins Super Resolution `nvngx_dlss.dll` 310.7.0 with SHA-256 `be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e` (`install.sh`). The local DLL now matches that version and hash; extraction and runtime compatibility were not executed.

### quattro

- `[S]` quattro targets the same GPU, OS, API, and shader language as this project: RX 7700 XT / gfx1101, Windows 11, D3D12, HLSL with native INT8 WMMA and INT32 accumulation (`README.md`).
- `[M]` EXP-004 reports a hardware PASS: the AMD shader-intrinsics path (`AmdWaveMatrix*`, `src/gpu/shaders/int8_wmma.hlsl`) generated `v_wmma_i32_16x16x16_iu8`, matched an independent CPU INT32 reference with zero mismatches in all four signedness modes, and ran on driver `32.0.32015.7029` — the driver recorded for this workstation.
- `[M]` EXP-005 reports a hardware PASS for FP16 WMMA (`src/gpu/shaders/fp16_wmma.hlsl`): `v_wmma_f32_16x16x16_f16` matched the CPU reference within 1 ULP, and `v_wmma_f16_16x16x16_f16` was bit-exact. It also records that the driver-generated operand-gather prologue dominates these first-cut kernels; this is not evidence about optimized throughput.
- `[S]` The reusable mechanisms are the D3D12 device/graph path (`src/gpu/device.cpp`, `graph.cpp`, `matmul.cpp`), AMD extension loading (`src/gpu/amdext.cpp`), vendored AMD intrinsic headers with `PROVENANCE.md`, and the experiment CLI (`src/main.cpp`, `src/experiments/experiments.cpp`).
- `[S]` quattro's model is FSR 4.1, not DLSS. Its validated kernels and numerical methodology transfer as mechanisms; its model graph and weights do not.
- `[S]` The four upstream repositories under `quattro/.references/sources/` are registered but uninitialized in this checkout. quattro's vendored AMD headers and experiment records remain present and pinned by the quattro commit.

### NVIDIA DLSS SDK

- `[S]` The pinned SDK checkout contains `include/` NGX headers, `LICENSE.txt`, and a pre-release license PDF. The sparse worktree does not include `lib/`, `doc/`, or `utils/`.
- `[S]` No model definition, weights, or standalone inference runner is present in the acquired subset. The SDK documents the integration API, not the Preset M network.

### NVIDIA TransformerEngine

- `[S]` The acquired subset contains CUDA FP8 scaling recipes (`current_scaling.cu`, `delayed_scaling.cu`, `fp8_block_scaling.cu`, `mxfp8_scaling.cu`, `nvfp4.cu`) and cast/include headers.
- `[S]` TransformerEngine targets NVIDIA CUDA hardware and does not implement DLSS or AMD WMMA. It is retained as requested numerical context; no Preset M claim should rest on it without separate evidence.

### Local DLL

- `[R]` Extracting the DLL's embedded fatbins with d4r's `kernels/tools/extract_dlss_ptx.py` produced 169 PTX modules. Thirty-five modules each contain one Preset M/L entry; 35 unique `rrlite_*` entry points were found, including `rrlite_enc0_4x4_{mvlo,mvhi}_{hdr,ldr}[_folded]`, `rrlite_enc1_4x4`, `rrlite_enc2_4x4`, `rrlite_enc3_tube_4x4`, `rrlite_dec0_4x4[_folded]`, `rrlite_dec1_4x4`, `rrlite_dec2_4x4`, `rrlite_post_3_{1,2}_*`, and the static/dynamic downsample kernels.
- `[R]` The presence of `_folded` and unfolded entry points is consistent with d4r's statement that preset M uses the folded input/expansion kernels while preset L uses the unfolded ones. The local DLL therefore appears to carry both M and L code paths, but this inference was not validated at runtime.
- `[U]` Weight location, layout, and numerical behavior of this DLL are unknown. It matches d4r's pinned 310.7.0 build, but d4r's extractor and model-package format were not executed against it here.

## Limitations and Unknowns

- `[U]` No model or weight file exists in any acquired source repository. d4r's weights are generated locally from a user-supplied DLL and are explicitly non-redistributable.
- `[U]` The exact Preset M model variant and weight set within 310.7.0 are not established. Code presence and the d4r version match do not prove extractability or numerical behavior.
- `[U]` d4r's standalone engine is a Linux/Vulkan path with an optional HIP network. It was not built or executed here, and its compatibility with this Windows/D3D12 host is unverified. WSL2 Ubuntu is installed, but no ROCm/HIP environment was validated.
- `[M]` quattro's INT8 and FP16 WMMA experiments were rebuilt and rerun here in EXP-001 and both passed. Their recorded driver version matches this workstation; the measured single-tile minima differ slightly, consistent with run-to-run clock state.
- `[U]` quattro's nested upstream references are uninitialized; no claim here depends on their contents.
- `[M]` DXC 1.8.2502.11 is present at `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe` but is not on `PATH`. EXP-001 compiled and dispatched D3D12 shaders; a full D3D12 feature-level query and Preset M runtime validation were not performed.
- `[U]` Whether TransformerEngine's FP8 recipes correspond to NVIDIA's DLSS weight encoding is unresolved and should not be assumed.

## References

- d4r: `references/sources/d4r/README.md`, `engine/README.md`, `engine/M_NOTES.md`, `engine/compile_m.py`, `kernels/m/swin_common.h`, `install.sh` at `05e19d4`.
- quattro: `references/sources/quattro/README.md`, `docs/experiments/EXP-004-int8-wmma.md`, `docs/experiments/EXP-005-fp16-wmma-comparison.md`, `src/gpu/shaders/`, `src/gpu/amdext.cpp`, `src/main.cpp` at `ad0482a`.
- NVIDIA DLSS: `references/sources/dlss/include/` at `9981c63`.
- NVIDIA TransformerEngine: `references/sources/transformerengine/transformer_engine/common/recipe/` at `dfc4158`.
- Local binary: `references/binaries/README.md`.
