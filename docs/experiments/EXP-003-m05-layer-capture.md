# EXP-003: Preset M Layer Capture and FP16 Replay Oracle

**Milestone:** M05
**Status:** PASS (layer scope) / full-network gate still BLOCKED

## Question

Can the D3D12 NGX probe produce a reproducible, legitimate Preset M weight and
tensor capture on this workstation, and can d4r's FP16 kernels reproduce the
captured layer outputs exactly?

## Method

- Pinned d4r source `05e19d4`; Windows tooling from `origin/windows` `7da2f9a`.
- Built only the required Windows targets with CMake/Ninja, llvm-mingw, and the
  installed ROCm 7.2 HIP SDK: `d4r_d3d12_dlss_probe`, `d4r_nvngx_windows`,
  `d4r_debug_launcher`, `d4r_native_replay_probe`. The full release was not rebuilt.
- NGX core: `_nvngx.dll` 32.0.16.1714, extracted from the pinned NVIDIA 617.14
  driver by `scripts/windows/install-dlss.ps1` (local only, not committed).
- Feature DLLs: local pinned `nvngx_dlss.dll` 310.7.0 (SHA-256 `be6e434a...`)
  and the installer's 310.9.1, compared side by side.
- Probe invocation: `--preset 13 --ngx-app-id 100152211 --ngx-abi driver
  --ngx-frontend d4r --interop-mode command-list --iterations 1`.
- Capture: `D4R_CUDA_REPLAY_DUMP_DIR`, `D4R_CUDA_REPLAY_DUMP_FILTER=rrlite_`,
  `D4R_CUDA_REPLAY_DUMP_LIMIT=40`; the dump is implemented by the release ZLUDA.
- Weight validation: `scripts/validate_m_capture.py` parses each launch with
  d4r's `model_enc3.Params` and extracts plain tensors with `m_model.plain_layer`.
- Replay: `scripts/replay_m_layers.py` runs `d4r_native_replay_probe.exe` on each
  capture with the matching gfx1101 `.hsaco`, then compares every reproduced
  allocation with the capture's `gpu-output/`.

## Results

- [M] Both feature DLL versions completed preset 13 on gfx1101:
  `D4R_CAPABILITIES available=1 feature_init=0x00000001 needs_driver=0`,
  `PASS D3D12_DLSS architecture=gfx1101 preset=13 frames=1`, finite output.
  The 512x288 output mean/variance were identical for 310.7.0 and 310.9.1
  (`mean=0.492251686 variance=0.0386891064`).
- [M] The capture contains exactly ten replay directories: `enc1`, `enc2`,
  six `enc3_tube`, `dec2`, `dec1`. The six tube launches have distinct weight
  allocations, distinct tensor hashes, and the expected alternating shifts
  `(2,2)`, `(0,0)`, `(2,2)`, `(0,0)`, `(2,2)`, `(0,0)`.
- [M] All ten layers parse into the expected d4r tensor sets (37/55/65 tensors
  for C=64/96/128), all finite, with the expected C, heads, grids, and shifts.
- [M] The ten layer weight sets are byte-identical between the 310.7.0 and
  310.9.1 captures (`identical_layers=10/10`), so the pinned DLL and the newer
  feature DLL carry the same Preset M layer weights for this model.
- [M] Native FP16 replay is bit-exact for every captured layer: 10/10 layers,
  both allocations each (`files=2 EXACT`), and a second complete run is
  byte-identical (`repeat=exact`).
- [M] The NGX frame output is deterministic for this synthetic input:
  `D3D12_OUTPUT preset=13 frame=0 finite=1`; `output-0.rgba16f` is 1,179,648
  bytes (512x288 RGBA16F).

### Discrepancy retained

`m_replay_validate.py validate --real-capture` reports 95.06% identical,
maximum 49 FP8 code steps, 47.7 dB, and fails its `<=1 step / >=50 dB` gate for
enc1. The native probe's output is nevertheless byte-identical to the captured
NVIDIA `gpu-output`, and a second native run is byte-identical. The failing
check compares against d4r's approximate logical `swin_check` reference, not
the captured GPU result; it does not contradict the exact replay.

## Interpretation

- [M] A legitimate, reproducible source of real Preset M layer weights and
  tensors now exists on this workstation: the user-supplied NVIDIA DLL executed
  through the d4r probe, with the capture kept local.
- [M] d4r's FP16 layer kernels reproduce the captured NVIDIA layer outputs
  bit-exactly on gfx1101 for all ten network layers.
- [S] The Windows capture hook is explicitly whitelisted to the ten
  `rrlite_*_4x4` layer kernels (`d4r_replay.rs`); `enc0`, `dec0`, `post`, and
  `downsample` launches are not dumped. A complete `compile_m.py` model
  therefore cannot be built from this Windows capture alone.
- [U] The full-network FP16 oracle, full-frame 1920x1080 Quality execution, and
  the encode/decode/reconstruction stages remain unestablished. M05 stays
  BLOCKED at the milestone gate; this experiment establishes a layer-level
  oracle, not a complete network oracle.

## Next

- Use the ten captured layers as M06 operator fixtures and as the reference for
  INT8 layer comparisons (M07/M09).
- To close M05, capture `enc0` and `dec0` weights. The Windows ZLUDA hook must
  be extended (or the Linux/Wine bridge used) because the current whitelist
  omits those kernels; rebuilding ZLUDA is the remaining cost.
- Reproduce with:
  `scripts/validate_m_capture.py CAPTURE --compare OTHER_CAPTURE` and
  `scripts/replay_m_layers.py CAPTURE --probe ... --modules ... --hip-root ... --output ... --repeat`.

## Files

- `scripts/validate_m_capture.py`
- `scripts/replay_m_layers.py`
- Captures, weights, modules, and NVIDIA DLLs remain local temporary artifacts
  and are not committed.
