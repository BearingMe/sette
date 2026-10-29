# Preset M Stage Graph and Capture Contents

**Milestone:** M05 (checkpoints B1/C16 and B2/C17)
**Status:** graph confirmed from the capture; image-bound tensors are not captured

## Summary

A 1920x1080 Preset M frame is fourteen launches.  Every edge between the eleven
buffer-mediating stages is confirmed by **byte-identical tensor content**
(14 of 14), the arena's addresses and sizes are mapped, and the stages whose
operands are *textures or surfaces* are identified: `enc0`, `post_3_1` and
`downsample` bind no tensors that this capture format records.

The capture therefore contains: all 360 weight matrices, the whole token chain
(enc0's output through dec1's output), and the probe's final frame readback -
but **no** texture or surface image.  Stage-level image validation of `enc0`,
`dec0`, `post` and `downsample` cannot be done from this capture; the final
frame can.

## Method

`scripts/m_capture_audit.py` reads each launch's manifest (parameter bytes,
grid, resolved pointers), compares every pre-launch allocation image with its
`gpu-output/` counterpart to find what the kernel actually wrote, and matches
each stage's candidate outputs (token field, patch-merge field, reconstruction
field) against every other stage's candidate inputs (token field, skip field)
by address **and size**, then verifies the bytes.

Address overlap alone is not evidence: the arena is a pool, so several stages
write the same address at different times, and a parameter block contains output
pointers too.  Sizes and content are what make an edge real.

## The graph

[M] 14 launches, all confirmed:

| # | Stage | Kernel | Grid | Block | Args | Role |
|---:|---|---|---|---|---:|---|
| 0 | enc0 | `rrlite_enc0_4x4_mvhi_ldr_folded` | 121x69 | 32 | 360 | textures -> token field |
| 1 | enc1 | `rrlite_enc1_4x4` | 61x35 | 32x2 | 56 | tokens -> tokens + merge |
| 2 | enc2 | `rrlite_enc2_4x4` | 31x18 | 32x4 | 56 | merge -> tokens + merge |
| 3-8 | enc3_tube x6 | `rrlite_enc3_tube_4x4` | 16x9 / 15x9 | 32x4 | 88 | ping-pong tokens |
| 9 | dec2 | `rrlite_dec2_4x4` | 30x17 | 32x4 | 56 | tokens + skip -> tokens |
| 10 | dec1 | `rrlite_dec1_4x4` | 60x34 | 32x2 | 56 | tokens + skip -> tokens |
| 11 | dec0 | `rrlite_dec0_4x4_folded` | 120x68 | 32 | 168 | tokens -> 4 surfaces |
| 12 | post_3_1 | `rrlite_post_3_1_mvhi_ldr_folded` | 180x51 | 16x16 | 248 | textures + dec0 surfaces -> reconstruction |
| 13 | downsample | `rrlite_downsample_kernel_static_ldr` | 120x68 | 16x16 | 96 | reconstruction -> history image |

[M] Content-verified edges (producer post-launch tensor == consumer pre-launch
tensor, byte for byte):

| Producer | Field | Consumer | Bytes |
|---|---|---|---:|
| enc0 | token out | enc1 token in | 8 355 840 |
| enc1 | merge out | enc2 token in | 3 133 440 |
| enc1 | token out | dec1 skip in | 8 355 840 |
| enc2 | merge out | enc3_tube#1 token in | 1 044 480 |
| enc2 | token out | dec2 skip in | 3 133 440 |
| enc3_tube#1..#5 | token out | enc3_tube#2..#6 token in | 1 044 480 each |
| enc3_tube#6 | token out | dec2 token in | 1 044 480 |
| dec2 | token out | dec1 token in | 3 133 440 |
| dec1 | token out | dec0 token in | 8 355 840 |
| post_3_1 | reconstruction out | downsample recon in | 18 662 400 |

[S] The roles are the ones the stage sources declare: `layer_m.comp` writes
`out` and its patch-merge result to `p48` and reads `inp`/`p32`; `dec0_m.comp`
reads the token field at binding 7; `down_m.comp` reads `post`'s packed buffer at
binding 3.

## Arena map (one frame, 960x540 render)

[M] All token traffic lives in one 36 556 800-byte allocation.  Offsets, from
the captured parameter blocks:

| Offset | Size | Tensor | Written by | Read by |
|---:|---:|---|---|---|
| 16 711 680 | 8 355 840 | 480x272x64 | enc0, dec1 | enc1, dec0 |
| 16 711 680 | 3 133 440 | 240x136x96 | enc2 | dec2 (skip) |
| 19 845 120 | 1 044 480 | 120x68x128 | enc2 (merge) | tube#1 |
| 20 889 600 | 1 044 480 | 120x68x128 | tube #1/#3/#5 | tube #2/#4/#6 |
| 21 934 080 | 1 044 480 | 120x68x128 | tube #2/#4/#6 | tube #3/#5, dec2 |
| 25 067 520 | 8 355 840 | 480x272x64 | enc1 | dec1 (skip) |
| 33 423 360 | 3 133 440 | 240x136x96 | enc1 (merge), dec2 | enc2, dec1 |

[M] The first 16.7 MB of the arena is untouched in every frame; a clean frame
starts with it entirely zero.  Overlapping rows are buffer reuse across
lifetimes, not aliasing within one dispatch - the size column is what
disambiguates them.

## What the capture does not contain

**Superseded by the addendum below (2026-10-29, EXP-011).** The text in this section
describes the buffer-only capture that produced this graph. Texture and surface storage is
now captured; see the addendum for the measured inventory.

[M] `enc0` binds eight descriptor objects and `post_3_1` eleven; `dec0` and
`downsample` bind two or three.  Those objects are 96 or 272 bytes and contain
host addresses (`0x2f14...`, `0x7ff9...`), so they are ZLUDA's texture/surface
descriptors, not the images.  No allocation in the dump is large enough to be
those images, and `dec0`/`downsample` write **zero bytes** into any captured
allocation.

Consequently the following are *not* in the capture:

| Stage | Missing |
|---|---|
| enc0 | colour, depth, motion, previous motion, half-res history and recurrent-feature inputs; its three motion outputs |
| dec0 | the four reconstruction surfaces (feature, gates, mix, covariance) |
| post_3_1 | dec0's surfaces, history, motion inputs, and its history output |
| downsample | its output image |

The final frame is available, but as the probe's readback (`output-0.rgba16f`),
not as a captured allocation.

**Consequence for the oracle.** Weights and the whole token chain are available
for stage-level validation, so the ten layers, `dec0`'s matrix operand and
`enc0`'s *output* are testable.  `enc0`, `dec0`, `post` and `downsample` cannot
be compared stage-by-stage against captured tensors; only their effect on the
final frame can be.  Closing that gap needs a capture extension (dump the image
memory behind `cuArrayCreate`/`cuMemAllocPitch`, or the surface's backing
buffer), which is a ZLUDA-side change and is recorded as an open task rather
than approximated.

## Addendum (2026-10-29): the image gap is closed

[M] The replay hook now dumps the array or pitched storage behind every texture and surface
object a launch binds (`patches/zluda/0018-m05-image-capture.patch`,
[EXP-011](../experiments/EXP-011-m05-image-capture.md)). For the same frame:

| Stage | Images | Written by the dispatch |
|---|---:|---|
| `enc0` | 11 (8 textures, 3 surfaces) | none - only its token buffer changes |
| `dec0` | 6 (2 textures, 4 surfaces) | 4 surfaces |
| `post_3_1` | 13 (11 textures, 2 surfaces) | history and half-resolution surfaces |
| `downsample` | 3 (2 textures, 1 surface) | the final 1920x1080 image |

[M] Two claims in the body of this note are corrected:

1. `enc0`'s three 960x540 RG16F surfaces are **zero before and after** the dispatch, so in
   the `mvhi` (display-motion) variant they do not receive motion candidates, or they are
   rewritten with identical zeros. They were not merely "uncaptured".
2. Stage-level validation of `enc0`, `dec0`, `post` and `downsample` no longer depends only
   on the final frame: their operands and results are captured, and `downsample`'s output
   array is byte-identical to the probe's independent `output-0.rgba16f` readback.

[M] New edges confirmed by array identity and content: `dec0` gates/mix/covariance to
`post` (3/3 identical), plus four temporal chains (`dec0` feature, `post` history, `post`
half-resolution, `enc0` motion A) over 24 frame pairs.

## Variant handling (B2/C17)

[S] `scripts/m_weights.py` parses kernel names into explicit fields -
`{family, stage, motion, range, folded}` - and never matches a prefix.  The
capture's `mvhi` names are kept as `mvhi`; nothing is renamed to `mvlo`.

[S] Evidence that `mvhi` and `mvlo` are different stages that share a weight
format, from the local 310.7.0 DLL (PTX extracted with
`extract_dlss_ptx.py`, modules `dlss-0154-00` and `dlss-0155-00`):

- both declare the same 360-byte parameter block;
- both reference exactly the offsets 4096, 8192 and 12288 into the weight block
  once, i.e. the same dither/matrix/bias geometry `compile_m.py` assumes;
- their bodies differ substantially (2966 vs 4038 PTX lines, different register
  counts), so the motion-reduction change is a real code difference, not a
  rename.

That is why `compile_m.py`'s `mvlo`-only name check rejects this capture: the
*weights* are interchangeable, the *stages* are not.  A Windows pipeline must
select the variant the frame actually ran.

[U] Whether the two variants carry byte-identical weight blobs is still
untested: only `mvhi` exists in this capture and the probe cannot submit
reduced-resolution motion.

## Limitations

- [M] One frame, one adapter, one DLL version (310.7.0), preset 13, quality,
  1920x1080 output.  No temporal evidence.
- [U] The parameter blocks are opaque 360/248/168/96-byte arrays in the PTX, so
  the mapping from argument offset to binding was taken from the payloads
  (which pointers resolve, which bytes change) plus the stage sources.  It is
  consistent for every stage but was not derived from a debug symbol.
- [U] `post_3_1`'s 18 662 400-byte output buffer is 9 bytes per output pixel;
  its internal layout is not established here.
- [U] `enc0`'s descriptor count (eight) exceeds its six samplers; the two extra
  objects are not identified.

## References

- `scripts/m_capture_audit.py` (this audit)
- `references/sources/d4r/engine/{layer_m.comp,dec0_m.comp,post_m.comp,down_m.comp}`
- [EXP-008](EXP-008-m05-full-capture.md) (capture coverage)
- [EXP-009](EXP-009-m07-complete-inventory.md) (weights from the same capture)
