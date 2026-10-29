# EXP-008: Full Preset M Capture Coverage (M05 C10)

**Milestone:** M05
**Status:** PASS

## Question

Can the Windows capture path be extended from the ten Swin layers to every
Preset M stage, so that a complete model can be built from one frame?
Acceptance criteria, defined before execution:

1. The rebuilt hook dumps `enc0`, `dec0`, `post_3_*` and `downsample_kernel`
   launches in the same fixture format the layer captures use.
2. The `enc0`/`dec0` weight blobs decode with d4r's own layout code into the
   shapes `compile_m.py` expects, with non-zero values.
3. The capture runs at the project's target geometry (1920x1080 output).

## Method

The probe-side MinHook route was already ruled out ([EXP-004](EXP-004-m05-capture-hook-attempt.md):
zero launches observed), so the ZLUDA replay hook itself was extended.

Rebuild, entirely inside the temporary workspace:

1. ZLUDA at `ee2f25a` (`v7-preview.11`) plus d4r's 17 `patches/zluda` and the
   audited K output-capture delta (`hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel`).
2. Four whitelist groups added to `d4r_replay.rs`, with the PTX parameter-block
   sizes from [the coverage audit](../findings/2026-10-29-m05-capture-coverage.md):
   `enc0_*_{hdr,ldr}_folded` 360 B, `dec0_4x4[_folded]` 168 B, `post_3_{1,2}_*` 248 B,
   `downsample_kernel_*` 96 B.
3. Pinned LLVM (`ff4dc1f`) built with the audited `0001-backport-gfx1154.patch`
   (12 jobs, ~15 min), PTX helper bitcode (4 variants), then the ZLUDA runtime
   with the pinned Rust 1.98.1 toolchain.
4. Probe run against the rebuilt `nvcuda.dll` (sha256 `71b45190…b98219`), preset 13,
   at 512x288 and again at 1920x1080.

Two harness obstacles were found and worked around, both in the temporary
workspace only: `ext/HiGHS` is a second submodule the build needs, and
`prepare-zluda-source.ps1` aborts under Windows PowerShell 5.1 because a native
command writing progress to stderr becomes terminating; the same audited patch
sequence was applied explicitly instead.

## Results

### Coverage

[M] Every stage group is dumped, in one frame, at both resolutions:

```text
rrlite_enc0_4x4_mvhi_ldr_folded      13 allocations, args 360
rrlite_enc1_4x4                       2 allocations, args 56
rrlite_enc2_4x4                       2 allocations, args 56
rrlite_enc3_tube_4x4                  2 allocations, args 88
rrlite_dec2_4x4                       2 allocations, args 56
rrlite_dec1_4x4                       2 allocations, args 56
rrlite_dec0_4x4_folded                8 allocations, args 168
rrlite_post_3_1_mvhi_ldr_folded      14 allocations, args 248
rrlite_downsample_kernel_static_ldr    4 allocations, args 96
```

[S] The argument sizes match the PTX `.param` blocks exactly, which is what the
whitelist entries were derived from.

### enc0 / dec0 weights decode with d4r's own layout code

[M] Using `compile_m.py`'s own extraction calls (`blob_of` + `swin_model.woff_table`)
against the new capture:

| Stage | Blob | Blob bytes | Tensor | Decoded range | RMS |
|---|---|---:|---|---|---|
| enc0 | `rrlite_enc0_4x4_mvhi_ldr_folded` | 12 416 | embedding 128x64 e4m3 | -0.688 … 0.688 | 0.194 |
| enc0 | same | | 64 f16 bias | 0.494, 0.109, -0.260, 0.220, -0.028 | |
| enc0 | same | | dither table 4096 B | 3 927 non-zero bytes | |
| dec0 | `rrlite_dec0_4x4_folded` | 8 448 | expansion 64x80 e4m3 | -0.750 … 0.625 | 0.095 |
| dec0 | same | | 80 f16 bias | -0.268, -0.227, 0.294, 0.175, 0.087 | |

The two weight captures that blocked the full model now exist and carry real,
non-degenerate values.

### Target geometry

[M] `--ngx-output-resolution 1920x1080` runs with input 960x540 (quality mode,
`PerfQualityValue=2`), preset 13, and writes `output-0.rgba16f` (16 588 800 B)
plus `output-0.bmp`. This is the first capture at the project's target output
resolution rather than the 512x288 convenience size.

## Limitations and Unknowns

- [S] **Variant naming.** The probe supplies full-resolution motion vectors, so
  NGX selects the `mvhi_*` stages (`enc0_4x4_mvhi_ldr_folded`,
  `post_3_1_mvhi_ldr_folded`, `downsample_kernel_static_ldr`). `compile_m.py`
  looks for `rrlite_enc0_4x4_mvlo_{hdr,ldr}_folded` by name, so it rejects this
  capture at its preset-detection step even though the weights are present. The
  `mvlo` variant is what d4r's game captures use; producing it needs the probe to
  submit reduced-resolution motion vectors, which it cannot do today.
- [S] **One launch per kernel per frame.** The hook names each dump
  `replay-000001-<kernel>` and does not increment within a frame, so a
  six-frame run overwrites its own layer dumps. A single frame is enough to
  build the model (weights are shared) but not to collect per-shift tensors;
  the earlier ten-layer capture (EXP-003) remains the source for those.
- [M] Running the probe through `d4r_debug_launcher.exe` with the rebuilt DLL
  makes `NVSDK_NGX_CUDA_Init` return `0x1` (feature init `0xBAD00004`), while the
  identical command run directly initializes normally. The capture path
  therefore invokes the probe directly. The cause was not investigated; it does
  not affect the capture contents.
- [U] Whether the `mvlo` stage uses bit-identical weights to `mvhi` is untested.
  Nothing in this experiment depends on the answer, but the full-model oracle
  will, if it is built from a `mvlo` game capture instead.

## Addendum (2026-10-29): images are captured too

This experiment covers buffers and weights. The textures and surfaces that `enc0`, `dec0`,
`post` and `downsample` bind are now dumped by the same hook as well (33 image targets per
frame, with geometry, format, pitch and pre/post contents), and the second bullet above is
resolved by `D4R_CUDA_REPLAY_DUMP_LIMIT` (see the addendum below and
[EXP-011](EXP-011-m05-image-capture.md)). One consequence worth knowing: at ~0.95 GB of
images per 1080p frame, a temporal capture must bound its frame count or it fills the
volume.

## Files

- Capture workspaces stay under the temp directory; no capture data is
  committed (weights are NVIDIA's).

## Addendum (2026-10-29): the two harness defects, diagnosed

1. **"Capture filename collision" is really a default limit.** [S] The hook
   names each dump `replay-{sequence:06}-{kernel}` where `sequence` counts
   launches *per kernel name*, and returns early once that count reaches
   `D4R_CUDA_REPLAY_DUMP_LIMIT`, which defaults to **1**.  Six tube launches
   therefore produce one directory, and different kernels legitimately share
   `replay-000001` because their names differ.  Every complete capture in this
   project uses `D4R_CUDA_REPLAY_DUMP_LIMIT=40`; the earlier 1080p frame is kept
   as-is and is not treated as evidence for the tube blocks.
   (`zluda/src/impl/d4r_replay.rs`, `begin()`.)

2. **`prepare-zluda-source.ps1` aborts under PowerShell 5.1.** [M] The script
   sets `$ErrorActionPreference = 'Stop'`, and PowerShell 5.1 turns a *native*
   command's stderr output into a terminating `NativeCommandError` - git writes
   ordinary progress there ("Switched to a new branch", patch statistics), so
   the script died before its own `$LASTEXITCODE` checks could run.  [M] Fixed
   by routing the native calls through a helper that relaxes the preference only
   around the invocation, keeps `$LASTEXITCODE` authoritative, and throws with
   the captured output on a non-zero exit.  Verified in PowerShell 5.1 both
   ways: stderr with exit 0 no longer terminates, and exit 3 still throws
   (`cmd /c ... failed (exit 3)`).  The build that produced this capture used
   the manual patch sequence before the fix; the fix changes no build product.

## References

- [Coverage audit](../findings/2026-10-29-m05-capture-coverage.md)
- [EXP-004](EXP-004-m05-capture-hook-attempt.md) (rejected probe-side hook)
- [EXP-003](EXP-003-m05-layer-capture.md) (ten-layer capture and replay)
