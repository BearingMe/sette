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

## Addendum (2026-10-29): replay independence limitation

The enc1 investigation in [EXP-019's addendum](EXP-019-m05-native-network.md)
finds that the later canonical `capture-1080b` log explicitly selects d4r-native
replacement kernels and skips translated JIT. Consequently native replay
identity alone cannot establish independent NVIDIA/PTX fidelity. The earlier
byte comparisons remain historical observations; their attribution to original
NVIDIA execution requires backend evidence, not merely an `rrlite_*` kernel
name or the recorded original launch dimensions.

## Addendum (2026-10-29): canonical launch audit and numerical-mode control

[M] The canonical 1920x1080 `capture-1080b` is a **mixed-backend** capture,
not fourteen independent original-network executions. The UTF-16
`%TEMP%/opencode/m-capture-run/probe-1080b.log` has SHA-256
`82a811f20e2c00c15ea3e3a793ef4729e8df17d1b38eaf25a7442e1429a9288c`.
Its launch records, bound to each manifest and capture completion by
`scripts/m_capture_provenance.py`, establish:

| Execution order | Captured kernel (tube occurrence) | Actual backend | Log line | Numerical-mode evidence |
|---:|---|---|---:|---|
| 1 | `rrlite_enc0_4x4_mvhi_ldr_folded` | translated PTX | 635 | ZLUDA lowering, not NVIDIA hardware |
| 2 | `rrlite_enc1_4x4` | d4r native | 640 | Conservative-equivalent output in the new controlled experiment; historical build flags not recorded |
| 3 | `rrlite_enc2_4x4` | d4r native | 645 | Fast/conservative mode not established by historical metadata |
| 4 | `rrlite_enc3_tube_4x4` #1 | d4r native | 650 | Same limitation |
| 5 | `rrlite_enc3_tube_4x4` #2 | d4r native | 655 | Same limitation |
| 6 | `rrlite_enc3_tube_4x4` #3 | d4r native | 660 | Same limitation |
| 7 | `rrlite_enc3_tube_4x4` #4 | d4r native | 665 | Same limitation |
| 8 | `rrlite_enc3_tube_4x4` #5 | d4r native | 670 | Same limitation |
| 9 | `rrlite_enc3_tube_4x4` #6 | d4r native | 675 | Same limitation |
| 10 | `rrlite_dec2_4x4` | d4r native | 680 | Same limitation |
| 11 | `rrlite_dec1_4x4` | d4r native | 685 | Same limitation |
| 12 | `rrlite_dec0_4x4_folded` | translated PTX | 690 | ZLUDA lowering, not NVIDIA hardware |
| 13 | `rrlite_post_3_1_mvhi_ldr_folded` | translated PTX | 695 | ZLUDA lowering, not NVIDIA hardware |
| 14 | `rrlite_downsample_kernel_static_ldr` | translated PTX | 700 | ZLUDA lowering, not NVIDIA hardware |

[M] The five unique neural modules log `module_direct ... translated_jit=skipped`
at lines 513, 520, 527, 534 and 541. Their selected module paths are under
`%TEMP%/opencode/d4r-runtime/d4r/native/gfx1101/`. The audit report records those
paths and **audit-time** binary hashes. These are not retrospective proof that
the files were unchanged since capture. Original recorded compiler flags are
unavailable; neither the directory name nor a current source default establishes
the installed binary's arithmetic mode.

[S] At pinned d4r `05e19d402781083118d106f2a21d0cf2bc43f744`, all five
`kernels/m/rrlite_*.hip` entry points define `SWIN_NO_Q8` and `SWIN_F32ACC`.
However, `swin_common.h:40-57` overrides both when `D4R_ACCURACY` selects
`SWIN_EXACT`. `kernels/build.sh:79-107` requires a separate accuracy output and
passes `-DD4R_ACCURACY`; `docs/native-kernels.md` describes native substitution,
and patches `0004` and `0014` implement replacement/lazy translation. These are
source facts, **not evidence that this capture selected the fast build**.

[M] In the new experiment, separately compiled pinned fast and conservative enc1
binaries replay identical captured bytes. Conservative output **and merge are
100% identical to the canonical capture**; fast is only 30.537265% / 27.768650%.
Therefore the hypothesis that enc1's old 96.924% Sette agreement is explained by
a fast-reference/conservative-Sette mismatch is not supported. This does not
certify the historical binaries' flags or all other layers' contracts.

Full metrics, binary identities, first-operation localization, build/replay
commands, and limitations are in [EXP-019's contract-validation addendum](EXP-019-m05-native-network.md#addendum-oracle-provenance-and-controlled-contract-validation-2026-10-29).
The original replay measurements above remain valid as self-replay measurements;
their wording as captured **NVIDIA layer fidelity** is superseded. The separate
translated enc1 control is GPU execution through ZLUDA on AMD; the PTX interpreter
remains supplementary semantic evidence only.
