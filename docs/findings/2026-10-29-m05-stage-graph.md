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

## Addendum (2026-10-29): the reconstruction buffer's layout

[M] The 18 662 400-byte buffer `post_3_1` writes and `downsample` reads is 2880x1620
32-bit words - 4 bytes per *source* pixel, i.e. 9 bytes per output pixel at 1.5 x 1.5
source pixels per output pixel. Each word is the LDR-packed display colour in 11-11-10 bits
using all 32: r = bits 0..10 over 2047, g = bits 11..21 over 2047, b = bits 22..31 over
1023. The stage that consumes it, `downsample`, is reconstructed natively and reproduces its
captured surface byte for byte
([EXP-012](../experiments/EXP-012-m05-native-downsample.md)), so this is a measured layout
rather than a reading of `down_m.comp`.

## Addendum (2026-10-29, later): `dec0`'s operands, from the kernel's own PTX

[S] [EXP-014](../experiments/EXP-014-m05-dec0-mapping.md) reads the address arithmetic of
`rrlite_dec0_4x4_folded` directly and fixes what this finding previously described only from
d4r's sources:

- the token field is **two planes of 32 channels**, 32 contiguous bytes per token, 480 tokens
  per row, **272 rows per plane** (the parameter block's height is the *padded* 544), so a plane
  is 4 177 920 bytes and the field 8 355 840 - exactly what `dec1` writes;
- the expansion matrix is the blocked layout this finding lists as 8 192 E4M3 bytes, and the
  PTX's offset formula is identical to `m_weights.layout_offsets`; the bias is one f16 per
  output column at 8 192;
- the four surfaces are written from 20 columns per 2x2 sub-position, `s*32 + 0..19`, with d4r's
  `v0..v4` value order.

[M] With that mapping the stage is reproduced natively at 99.9991% / 99.9348% / 99.9787% /
99.9883% byte agreement on the four surfaces.

## Addendum (2026-10-29, later still): `enc0`'s contract, byte-exact

[S] [EXP-015](../experiments/EXP-015-m05-enc0-contract.md) now derives `enc0` from the PTX too,
and the stage reproduces its 8 355 840-byte output byte for byte.  What the pipeline needs to
know about it:

- **Bindings it never reads.**  The two 1x1 textures (`param+112`, `param+120`), the depth
  texture (`param+136`) and the pointer at `param+40` are not loaded by this variant.  The
  depth texture's absence identifies the display-motion path, which takes one motion vector per
  pixel.  The three RG16F surfaces at `param+280/288/296` *are* written (`sust.b.2d.v2.b16.zero`),
  and the token output is `param+48` = `alloc1 + 0xFF0000`, not `param+40`.
- **The blob** is 4 096 B of dither table, 8 192 B of blocked 128x64 E4M3 matrix (`layout_offsets`
  is the PTX's own formula) and 128 B of f16 bias, one per output column.
- **The feature vector** is one point-sampled texel per pixel: a reset frame makes the history,
  previous-feature and motion-change slots exactly zero, leaving 15 nonzero slots of the 32,
  which carry the texel's r, g, b cyclically in ascending slot order.  A token is a 2x2 pixel
  block and its four pixels are the embedding's four k32 steps in (0,0), (0,1), (1,0), (1,1)
  order.  The kernel's shared-memory re-store is a no-op, so `k` is the slot index.
- **The epilogue** clamps the f16 accumulator to +-2*pi and then applies a channel permutation
  (channel `c` is not matrix column `c`) before the E4M3 conversion.  Two of the three errors
  that had blocked `enc0` were of this kind: orders that look like bookkeeping but change every
  byte.

[M] Consequence for B5: `enc0` is no longer an unknown in the native path.  The remaining input
stage is `post` (B4.5), and the ten-layer net still has to run inside this graph.

## Addendum (2026-10-29, later still again): `post_3_1`'s contract, byte-exact

[S] [EXP-016](../experiments/EXP-016-m05-post-reconstruction.md) derives the reconstruction
from `dlss-0145-00.ptx` and reproduces **all three** of its outputs byte for byte.  What the
pipeline needs to know:

- **It is the only stage with three outputs.**  `param+96` is the 2880x1620 RGBA16F
  reconstruction, `param+104` the 2880x1620 packed 11-11-10 word buffer that `downsample`
  reads, and `param+112` a 1440x810 half-resolution mean.  The reconstruction is 2880x1620
  because the render is 960x540 and the ratio is 3: a thread owns a column and a **row pair**.
- **The inputs are dec0's surfaces.**  post's image 3/4/5 are dec0's gate/mix/covariance by
  content, so the `dec0 -> post` edge is byte-exact at three of the four surfaces.
- **Three conversions decide the bytes and none of them is in the PTX.**  The RGBA16F surface
  truncates toward zero (round-to-nearest agrees on 55.8%, truncation on 99.998%); the
  half-resolution surface holds 10-bit codes as exact `k/1023` quotients; and the mean is
  quantised before that store.
- **The motion texture is sampled at display resolution.**  `trunc(pos/recon*display)` reaches
  (1919, 1079) while the captured texture is 960x540.  On this frame the texture is zero, so
  the question stays open - the same one EXP-015's E7 records.

[M] Consequence for B5: every input/output stage is now reconstructed and exact on the
canonical frame, so the ten-layer net is the only part of the native path that is still an
unknown.

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
  the addendum below resolves this.
- [U] `enc0`'s descriptor count (eight) exceeds its six samplers; the two extra
  objects are not identified.
  *Resolved:* the two extra are the 1x1 textures at `param+112`/`+120`, which the
  kernel never loads (see the `enc0` addendum).

## References

- `scripts/m_capture_audit.py` (this audit)
- `references/sources/d4r/engine/{layer_m.comp,dec0_m.comp,post_m.comp,down_m.comp}`
- [EXP-008](EXP-008-m05-full-capture.md) (capture coverage)
- [EXP-009](EXP-009-m07-complete-inventory.md) (weights from the same capture)
