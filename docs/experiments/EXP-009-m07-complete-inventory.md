# EXP-009: Complete Weight Inventory and INT8 Feasibility (M07 extension)

**Milestone:** M07 (checkpoints A1–A5 below; the task's C11–C15)
**Status:** PASS for the weight side; stage-level image quality is out of scope

## Question

Once `enc0` and `dec0` are captured, does the INT8 feasibility conclusion from
[EXP-007](EXP-007-m07-int8-feasibility.md) still hold for *every* matrix Preset M
owns, and which scale scope should the reconstruction use?  Acceptance criteria,
fixed before execution:

1. Every matrix of one frame is extracted from the capture, decoded with an
   independently checked layout, and accounted for by shape and precision.
2. The INT8 encoder is deterministic, independently verified, and its saturation
   and rounding behaviour is characterised.
3. Weight-scope alternatives are compared with the same metrics on the same data.
4. Where a captured activation exists, the error an INT8 GEMM would produce on it
   is measured with explicitly stated arithmetic.
5. The weight inventory reconciles every matrix; remaining unknowns stay visible.

## Method

New and extended tooling (all CPU-only, no GPU and no reference model in the
error path):

- `scripts/m_weights.py` - capture -> inventory.  Parses each kernel name into
  explicit `{family, stage, motion, range, folded}` fields, maps launches onto
  network positions by name and per-kernel sequence (never by directory sort),
  extracts `enc0`/`dec0` through the blocked layout described below and the ten
  Swin layers through d4r's `plain_layer()`, and keeps both the e4m3 **codes** and
  the exact f16 values.  `activations()` returns the captured operand of every
  stage that reads one from a buffer, including `dec0`'s, identified by pointer
  identity with `dec1`'s output allocation rather than a guessed offset.
- `scripts/m_int8.py` - symmetric INT8 encoders (per tensor, per output channel,
  per input channel, per 16/32-row K block), error/storage metrics, and
  `gemm_integer()`, which runs the product the way a kernel must: exact integer
  products, int32 accumulation, one floating-point rescale per scale group.
- `scripts/m07_int8_weight_quant.py` - extended from the four-scope layer study
  to the full inventory, five scopes, activation attribution and storage cost.
- `tests/test_m07_quantization_contract.py` - 24 contract tests (independent
  e4m3 decode, layout bijection and 32k x 2n unit grammar, scalar-vs-vector
  encoder agreement, half-to-even rounding, saturation, zero handling, integer
  product vs an element-wise loop, scope/e2e consistency, determinism).

Capture: one frame, preset 13, **1920x1080 output / 960x540 render**, quality
mode, `D4R_CUDA_REPLAY_DUMP_LIMIT=40` so all six tube launches survive (the
default limit of 1 keeps only the first launch of each kernel name - the cause
of the earlier "filename collision", see the harness note at the end).

```powershell
python scripts/m07_int8_weight_quant.py $env:TEMP\opencode\m-capture-run\capture-1080b --json m07.json
```

## Results

### A1 (task C11) - the complete inventory

[M] One frame yields 14 captured launches covering 14 positions of the graph:
360 matrices, **1 553 408 e4m3 values**, every one decoded, finite and
round-tripped through the code table.

| Stage | Kernel | Variant | Matrices | Codes | Weight blob |
|---|---|---|---:|---:|---:|
| enc0 | `rrlite_enc0_4x4_mvhi_ldr_folded` | mvhi/ldr | 1 | 8 192 | 12 416 B |
| enc1 | `rrlite_enc1_4x4` | - | 22 | 69 632 | 71 872 B |
| enc2 | `rrlite_enc2_4x4` | - | 34 | 159 744 | 163 584 B |
| enc3_tube x6 | `rrlite_enc3_tube_4x4` | - | 41 each | 180 224 each | 184 320 B each |
| dec2 | `rrlite_dec2_4x4` | - | 34 | 159 744 | 164 096 B |
| dec1 | `rrlite_dec1_4x4` | - | 22 | 69 632 | 72 192 B |
| dec0 | `rrlite_dec0_4x4_folded` | - | 1 | 5 120 | 8 448 B |
| post_3_1 | `rrlite_post_3_1_mvhi_ldr_folded` | mvhi/ldr | 0 | 0 | - |
| downsample | `rrlite_downsample_kernel_static_ldr` | static/ldr | 0 | 0 | - |

The ten Swin layers alone are 358 matrices / 1 540 096 values - the figure
EXP-007 reported as 312 matrices / 122 880 values (see the correction there).

**Front-end layouts.** [S] Both blobs fit their stage's own declaration
(`enc0_m.comp` binding 10 is `[128][64]`, binding 11 a 16 x 64 bias tile,
binding 12 the 8x16x16 dither; `dec0_m.comp` binding 8 is `[64][80]`, binding 9 a
16 x 80 bias tile):

| Stage | Store | Matrix | Bytes | Bias | Auxiliary |
|---|---|---|---|---|---|
| enc0 | blocked e4m3 | `[128][64]` | 4096..12288 | 64 f16 @ 12288 | dither 4096 B @ 0 |
| dec0 | blocked e4m3 | `[64][128]`, 80 live columns | 0..8192 | 128 f16 @ 8192 | - |

[M] Decoded values are real and non-degenerate: `enc0` embedding absmax 0.6875,
RMS 0.1937, 24 zeros, bias[0] = +0.4939; `dec0` expansion absmax 0.75, RMS
0.0953, 67 zeros, bias[0] = -0.2676.  The 4096-byte dither block is 3 927/4096
non-zero bytes.

[M] Three independent checks on the extraction:
* the offset formula is a **bijection** onto exactly `[4096, 12288)` (enc0) and
  `[0, 8192)` (dec0) - no aliasing, no hole;
* every 64-byte unit of the layout holds exactly 32 k rows x 2 columns **8
  columns apart**, i.e. the m16n8k32 fragment grammar the shaders index;
* a second, bit-level e4m3 decoder (OCP definition) reproduces d4r's table and
  the stored values exactly.

[M] Reproducibility across captures: two independent 1080p runs give **155/155
identical** shared matrices (the rest differ only because the first run kept one
tube launch); the 1080p frame and an earlier differently-sized frame give
**358/358 identical** layer matrices, with only `enc0`/`dec0` absent from the
older capture.  Weights are therefore frame- and resolution-independent, and the
extraction is deterministic.

### A2 (task C12) - the encoder

[M] Repeat encoding is bit-identical (codes and float32 scale bits) for all
360 matrices.  [M] d4r's own asymmetry (round against a float64 scale, store the
float32 one) changes **0 of 1 553 408** codes, so reproducing d4r's convention
exactly is free rather than approximate.

[S] Rounding is half-to-even (`nearbyint`), saturation is a clamp to +-127, and
a zero weight always produces code 0 (the tests pin all three).

### A3 (task C13) - scale scopes

[M] Element-weighted RMS relative weight error over all 1 553 408 values:

| Scope | RMS | Worst matrix | Scales | Storage vs FP8 |
|---|---:|---:|---:|---:|
| per tensor | 0.0095 | 0.0172 | 1 440 | 1.001x |
| per output channel | 0.0062 | 0.0074 | 97 472 | 1.063x |
| per input channel | 0.0061 | 0.0087 | 121 472 | 1.078x |
| per K block of 16 (per column) | **0.0048** | **0.0052** | 388 352 | 1.250x |
| per K block of 32 (per column) | 0.0055 | 0.0061 | 194 176 | 1.125x |

[M] Per stage the ordering is uniform; `dec0` is the only matrix where per-tensor
is clearly the worst (0.0172 vs 0.0058 per column), because its distribution is
broad rather than spiked.

[M] Outlier structure, worst cases: `dec0.expansion` 0.0172 -> 0.0058 (per column);
`enc2.w1_10` 0.0148 -> 0.0064; `enc1.w1_5` (matrix max 2.0) 0.0135 -> 0.0061.  No
matrix exceeds 0.0074 under per-column scaling.  The largest per-column error is
`enc2.merge_w` (0.0074), a wide `[384][128]` matrix.

[S] This refines EXP-007's ranking rather than confirming it.  EXP-007's
"block16/32" put **one** scale on a whole 16-row slab (shared across columns);
the K-blocked scope here gives every column its own scale inside each 16-row
block, which is what an INT8 GEMM can actually consume.  The old scope was
worse than per-channel; the new one is better.

**Trade-off.** Per-input-channel and K-blocked scales need the integer reduction
restarted per scale group (4/8/16 float32 rescales for K=64/128/256) instead of
one.  Per output channel keeps a single int32 accumulation.  Measured on the real
operands (next section), K-block:16 buys 1.5-2x lower weight error for that
restart cost; per-input-channel buys less and needs the most rescales.

### A4 (task C14) - INT8 GEMM on captured operands

[M] Activation operands at the captured 960x540 render extent (token fields read
from the arena, values decoded from e4m3):

| Stage | Shape | absmax | RMS | peak/RMS |
|---|---|---:|---:|---:|
| enc1 | 272x480x64 | 1.375 | 0.386 | 3.56 |
| enc2 | 136x240x96 | 5.500 | 1.518 | 3.62 |
| tube #1..#6 | 68x120x128 | 6.5 -> 16.0 | 3.36 -> 3.84 | 1.93 -> 4.16 |
| dec2 | 68x120x128 | 7.500 | 1.804 | 4.16 |
| dec1 | 136x240x96 | 13.000 | 3.139 | 4.14 |
| dec0 | 272x480x64 | 16.000 | 2.116 | **7.56** |

[M] `dec0`'s operand is the worst-conditioned in the network (peak/RMS 7.56,
against 1.9-4.2 elsewhere), and `dec0` also owns the worst-conditioned weight
matrix.  Both ends of that stage are outlier-heavy, which makes it the first
place to lose quality.

[M] INT8 GEMM error on the real operands, integer accumulation, per-token
activation scale, bias excluded (it is an additive constant no scope touches):

| Matrix | Shape | Weights (per col) | Activation | Both | Both (K-block 16) |
|---|---|---:|---:|---:|---:|
| enc1.q0 | 130560x64x32 | 0.0096 | 0.0108 | 0.0149 | 0.0137 |
| enc2.q0 | 32640x96x32 | 0.0089 | 0.0106 | 0.0138 | 0.0127 |
| tube#1.q0 | 8160x128x32 | 0.0087 | 0.0081 | 0.0119 | 0.0105 |
| dec2.expand_w | 8160x128x384 | 0.0049 | 0.0047 | 0.0068 | 0.0059 |
| dec1.expand_w | 32640x96x256 | 0.0068 | 0.0065 | 0.0094 | 0.0082 |
| dec0.expansion | 8160x64x80 | 0.0066 | 0.0074 | 0.0100 | 0.0091 |

[M] With per-output-channel weights the **activation term is as large as or
larger than** the weight term (0.0108 vs 0.0096 on `enc1.q0`).  Weight
quantization is no longer the binding constraint at that scope; the per-token
activation scale is.  This is the main new conclusion for M09/M11.

[M] d4r's own published INT8 convention for `dec0` (per-column weights,
per-token activation, one int32 accumulation, one float32 rescale) reproduces
0.0100 relative / 0.3088 max absolute error on the captured tokens.  That is the
same number as the per-column row above, so the convention is consistent - and
it is ~10% worse than the K-block:16 variant, not orders of magnitude.

[U] `enc0` has no captured GEMM operand: its 128 features are computed inside
the kernel from texture samples (colour, depth, motion, previous feature and
history), so no `enc0` propagation is reported rather than a substituted input.
`dec0` rows are a deterministic stride-16 sample (8 160 of 130 560 tokens).

### A5 (task C15) - reconciliation

[M] Matrix count by stage: enc0 1, enc1 22, enc2 34, enc3_tube 246 (6 x 41),
dec2 34, dec1 22, dec0 1 = **360**.  Kinds: `q`, `v`, `wo`, `w1`, `w2`, `merge_w`,
`expand_w`, `embedding`, `expansion`.  No other matrix exists in the graph: post
and downsample are pure image stages, and exposure is a two-pass curve.

[M] Storage: FP8 is already one byte per value, so INT8 can never shrink the
weights - it replaces 1 553 408 B of FP8 with 1 553 408 B of INT8 plus scales.
Per-output-channel costs +97 472 B (1.063x), K-block:16 +388 352 B (1.25x).
Framing INT8 as a size optimization would be wrong; it is a *throughput*
representation.

[S] Retained floating point (unchanged by any weight scope): RMS norms with the
`2^-13` floor, softmax with the exponent trick, e4m3 requantisation at every
operand boundary, the two f16 residual streams, GELU, exposure, and the whole
reconstruction (`post`, `downsample`).

**Recommended policy** (supersedes EXP-007's wording): per-output-channel INT8
weights as the baseline because it needs a single int32 accumulation; per
(column, 16-row K block) where the extra rescales can be hidden, which is every
matrix except the small projections.  Activations need their own investigation:
at per-token scaling they already dominate the error.

## Limitations and Unknowns

- [M] Statistics are from one frame.  Weight statistics are frame-independent
  (verified across two geometries); activation ranges are not.
- [U] Asymmetric/zero-point quantization and clipping sweeps remain untested
  (deferred to M12 by design).
- [U] These are per-GEMM errors, not image quality: no block-level, residual or
  temporal propagation is measured here (M11).
- [U] `enc0`'s operand is unavailable in this capture format (see above), so the
  input stage's INT8 error is unmeasured.
- [U] Whether the `mvlo` variant's weight blob is byte-identical to `mvhi`'s is
  still untested; only `mvhi` exists in this capture.

## Addendum (2026-10-29): the front-end matrix order is not confirmed

The inventory's *counts*, *codes* and *layer matrices* are unaffected, but two claims above
depend on a decode of the two front-end weight blobs that has now been tested against an
independent reference and did not survive it:

- [M] [EXP-013](EXP-013-m05-dec0-epilogue.md) tried to reproduce `dec0`'s four captured
  surfaces from `front_end_layout('dec0')`'s `[64][128]` decode and the published epilogue.
  No column of the decoded matrix has the sign, magnitude or spatial pattern the surfaces
  imply, under either candidate token layout, nor for five other row-major decodings of the
  same blob. The blob's *size* (8192 + 256 bytes) agrees with the decode; its element order
  does not.
- [U] Consequently `dec0`'s activation statistics quoted here (peak 7.56) and `enc0`'s
  matrix values should be treated as decoded-but-unverified until the order is established.
  The layer matrices are validated independently by the ten-layer check
  ([EXP-006](EXP-006-m06-operator-contract.md)), so the INT8 scope comparison - which is
  dominated by them - stands.

## Harness note

The hook's `D4R_CUDA_REPLAY_DUMP_LIMIT` defaults to **1 per kernel name**, and
the per-name counter is what appears in `replay-<seq>-<kernel>`.  Six tube
launches therefore collapse to one directory unless the limit is raised; it is
not a race between different kernels.  The verified 1080p capture keeps all six
with `D4R_CUDA_REPLAY_DUMP_LIMIT=40`.

## Files

- `scripts/m_weights.py`, `scripts/m_int8.py`, `scripts/m07_int8_weight_quant.py`
- `tests/test_m07_quantization_contract.py`
- Captures, weights and JSON reports remain local; nothing NVIDIA-derived is
  committed.

## References

- [EXP-007](EXP-007-m07-int8-feasibility.md) (layer-scope study and its correction)
- [EXP-008](EXP-008-m05-full-capture.md) (capture coverage)
- [EXP-006](EXP-006-m06-operator-contract.md) (e4m3/f16 losslessness)
- [EXP-005](EXP-005-m08-native-int8-probes.md) (INT32 accumulation, wraparound)
- `references/sources/d4r/engine/compile_m.py`, `dec0_m.comp`, `enc0_m.comp`,
  `runtime_m.inc` (`widenM` INT8 convention)
