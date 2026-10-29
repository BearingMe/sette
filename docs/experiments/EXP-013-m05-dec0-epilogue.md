# EXP-013: `dec0`'s Expansion Epilogue - Operand/Weight Mapping Unresolved (M05 B4.3)

**Milestone:** M05 (checkpoint B4.3/C19)
**Status:** BLOCKED - the stage's *values* cannot be reproduced, and the question is now localised
**Depends on:** [EXP-010](EXP-010-m05-native-fp16-stage.md) (the step chain), [EXP-011](EXP-011-m05-image-capture.md)
(the four captured surfaces), [EXP-012](EXP-012-m05-native-downsample.md) (the native image-stage path)

## Addendum (2026-10-29): the blocker is resolved

**Status:** the question below is answered; see [EXP-014](EXP-014-m05-dec0-mapping.md). This
report is kept as the record of the failed attempt.

[⇒] The failure was **not** in the matrix decode. `m_weights.layout_offsets('dec0')` is identical,
element for element, to the address formula in the kernel's own PTX (all 64x128 elements), and
the `live` column permutation (`s*32 + 0..19`, values `v0..v4`) is d4r's and correct. Two other
things were wrong:

1. [M] `param+16` is the **padded** render size, `(960, 544)` - the captured `args.bin` reads
   `0x220` = 544, not 540. The token plane is therefore 272 rows and 4 177 920 bytes, not 270
   rows and 4 147 200. The two-row shift put the second channel plane in the wrong place.
2. [M] The E4M3 **subnormal** decode needs no implicit leading 1: the value is `m * 2^-9`, not
   `(1 + m/8) * 2^-6`. 17 of the 64 weights of column 0 are subnormal codes, so this inflated a
   quarter of every product by up to 8x.

[M] With both corrected, the same binary reproduces the four surfaces at 99.9991% / 99.9348% /
99.9787% / 99.9883% and the native D3D12 stage matches. [⇒] The "next" list below is complete:
the blob search it proposed was unnecessary, because the mapping was already correct and the
operand was not.

## Question

`dec0` expands the last decoder's 64-channel E4M3 token field to 20 values per (token, 2x2
sub-position) with one matrix product, then turns them into four render-resolution surfaces:
a recurrent feature (SNORM8), three gates (UNORM8), three softmax weights (UNORM8) and a 2x2
covariance (RGBA16F). Its matrix stage is the one native operator that already matched a
published contract (EXP-010), the capture holds all four written surfaces, and frame 1 is a
reset frame - `featTex` is all zeros and `p.reset` short-circuits the history sample, so the
epilogue is a **pure function of the 20 values per (token, sub-position)** with no texturing.

Can the stage therefore be reproduced exactly: matrix chain, the E4M3 re-quantisation of its
20 outputs, the tabulated activations, and the four normalised stores?

Acceptance criteria, fixed before the comparison:

- A. The GEMM's 20 outputs per (token, sub-position) must be established, by reproducing the
  captured surfaces through the published epilogue.
- B. Each of the four surfaces must then be compared byte for byte, and any difference
  characterised (ULP distribution, spatial extent).
- C. Whatever is unresolved must be named as a blocker rather than approximated.

## Method

### Environment

As EXP-012: Windows 11, RX 7700 XT, DXC 1.8.2502.11, capture frame 1 of
`replay-000001-rrlite_dec0_4x4_folded` (`nvngx_dlss.dll` 310.7.0, preset 13, 1920x1080 from a
960x540 render). Reference: the four post-launch surfaces (`image-2` feature, `image-3` gates,
`image-4` mix, `image-5` covariance), which EXP-011 showed feed `post` byte for byte.

### The model

`scripts/m_dec0_model.py` is a transcription of `dec0_m.comp`'s non-DOT2 path:

- the token field as 64 values per token, decoded from E4M3;
- the expansion matrix and its bias from the captured weight blob
  (`m_weights.front_end_layout('dec0')`: `[64][128]` E4M3 at 0, 128 f16 bias at 8192);
- `mma_chain`: bias as the initial f16 accumulator, one f16 rounding per 32 products;
- `q8`: E4M3 round-trip of each of the 20 outputs;
- `activation_m.glsl`'s `ACT` table (regenerated from `gen_activation_m.py`, which derives it
  from the formulas alone) for sigmoid, tanh and the gate;
- the epilogue's arithmetic and the `unorm8Value` / `snorm8Value` encodings.

The parameter block is read from the argument slots: `rnd2hi = (3,3)`, `hiSize = (2880,1620)`,
`reset = 1`, render size 960x540, `sigma = 8.0`, `rho = 0.0786`. Two of these (`sigma`, `rho`)
are identified by plausibility, not by a symbol - see the limitations.

### Inverting the reference

The four surfaces are *activations* of the 20 values, so the capture can be inverted to a
coarse but dense target for several of them. The gates are `unorm8(sig16(v))`, and `ACT` holds
only 256 codes, so each captured byte pins the E4M3 code of the value (up to codes that share a
byte). Inverting the whole frame gives, over 518 400 pixels:

| Column | Recovered quantity | Mean | Std | Range |
|---|---|---|---:|---:|---:|
| 0 | `v0.x` (gate 0) | +1.99 | 1.03 | -0.81 .. 5.00 |
| 1 | `v0.y` (gate 1) | +8.75 | 0.96 | +1.75 .. 9.00 (saturated) |
| 10 | `v2.z` (gate 2) | +8.71 | 1.06 | +1.88 .. 9.00 (saturated) |

The last two saturate (their sigmoid rounds to 1.0 for every code above 9), but the sign and
the lower bound are unambiguous: those two values are **large and positive** everywhere.

## Results

[M] The model does not reproduce any surface. With the token field read as 64 contiguous
channels per token:

| Surface | Identical | Max code delta |
|---|---:|---:|
| feature (SNORM8) | 0.52% | 255 |
| gates (UNORM8) | 25.15% | 255 |
| mix (UNORM8) | 25.37% | 255 |
| covariance (RGBA16F) | 25.23% | 255 |

(The 25% is the constant fourth channel, which the stage writes as zero in all four surfaces -
that agreement is trivially right.) The model's own columns are near zero-mean while the
recovered targets are strongly positive (`v0.x` +2.0, `v0.y` and `v2.z` at the saturation
floor), so the mismatch is not a rounding difference: the *values* are wrong.

[M] The following were eliminated, each by comparing the analytic mean and standard deviation
of every decoded column (`mean = E[x] W`, `var = W^T cov(x) W`, so no dispatch is needed)
against the recovered targets:

| Variant | Best column for `v0.x` |
|---|---|
| token field, 64 contiguous channels per token | column 14, \|corr\| < 0.06 |
| token field, two planes of 32 channels | column 16, \|corr\| < 0.10 |
| either of the above with the channel-pair permutation `gp` or its inverse | no column above 0.07 |
| matrix decoded as `[64][128]` with `layout_offsets` and the 80 "live" columns | std 1.4-2.0 where the target needs 1.0 |
| matrix decoded as row-major `[64][80]` / `[80][64]` at byte offsets 0, 768, 1024, 2560, 5120 of the blob | no column with mean > 0.5 |

No combination produced a column with the recovered mean, let alone a correlated spatial
pattern. A least-squares fit of the token field to the recovered `v0.x` (which would recover the
true weight column up to the quantisation noise of the inversion) also failed to correlate with
any decoded column, so the failure is not merely a column permutation.

[S] One layout question *is* settled, and it corrects an assumption the model started with:
`dec0_m.comp`'s `#if PLANES` branch - offset `hf * 16 * numWorkGroups.x * numWorkGroups.y * 32`
- is the two-plane layout d4r calls "NGX's layout", and it is the layout `m_model.tokens()`
decodes and that [EXP-006](EXP-006-m06-operator-contract.md)'s ten-layer check validates for
layer-to-layer edges. The non-PLANES branch (64 contiguous channels per token) is the one d4r
compiles for their own HIP network. Neither interpretation reproduces `dec0`'s surfaces here.

[U] Therefore the expansion matrix's element order - or the identity of the buffer the kernel
reads as `w8` - is not established for this capture. The blob is 8448 bytes, exactly
8192 (a `[64][128]` E4M3 matrix) + 256 (128 f16 biases), so its *size* agrees with
`m_weights`' decode; its *element order* does not, because no decode tried yields values whose
sign and magnitude match the surfaces.

[U] `sigma = 8.0` and `rho = 0.0786` are the two f32 parameters in the captured argument block
that fit `dec0_m.comp`'s use; they are not confirmed against a symbol. They do not affect the
`v0.x` target, which needs no scale, so the conclusion above does not depend on them.

[U] The captured frame is a reset frame, so the history/texture half of the epilogue is
untested - only the "no history" path is exercised by any model here.

## Interpretation

- [M] The four surfaces are not reproducible from the published epilogue and the decoded
  weights, so `dec0`'s *matrix operand/weight mapping* is the blocker, not the epilogue: the
  epilogue is fully specified by the captured data once the 20 values are known.
- [S] `w8` and `w16` are separate bindings in `dec0_m.comp` while the captured blob is a single
  8448-byte allocation. Either the host writes the two buffers elsewhere or the blob's internal
  order differs from `layout_offsets`; the arena region the launch also binds is all zeros
  before the dispatch, so it is not the matrix.
- [U] Whether the same mapping error affects the other front-end stage (`enc0`) and the
  ten-layer matrices: those were validated *through their own kernels' contracts* (EXP-006,
  EXP-003), which does not exercise a front-end blob decode. EXP-009's weight inventory uses
  the same decode, so its per-matrix values for the two front-end stages carry the same risk;
  its *counts* and the layer matrices are unaffected.
- [M] Nothing here contradicts EXP-010: that experiment compared the native scalar chain
  against the published step chain on the same operand, which is a property of the arithmetic,
  not of the operand's provenance.

## Next

1. Resolve `w8`'s layout. The cheapest independent handle is a *search inside the blob*: solve
   for the true weight column from the recovered `v0.x` by least squares on the token field,
   then locate that 64-element pattern in the 8448-byte blob at every offset and stride. A hit
   identifies both the layout and the operand layout unambiguously. (Attempted here on the
   plane and interleaved decodes only; the inversion's quantisation makes the pattern coarse,
   so the search must allow for that.)
2. Confirm the operand layout from `layer_m.comp`'s token read rather than from `dec0_m.comp`
   alone: the ten-layer check fixes the layout of the producer, and `dec0` must read what
   `dec1` wrote.
3. Only then port the epilogue, which is already transcribed and needs no further reverse
   engineering.

## Files

- `scripts/m_dec0_model.py` (the transcription; documented as an instrument that does *not*
  reproduce the reference)
- Fixtures and results stay in the temporary workspace.

## References

- `references/sources/d4r/engine/dec0_m.comp`, `activation_m.glsl`, `gen_activation_m.py`,
  `normalized_store.glsl`, `round_half.glsl` (d4r `05e19d4`)
- [EXP-006](EXP-006-m06-operator-contract.md) (the ten-layer check that fixes the token layout)
- [EXP-009](EXP-009-m07-complete-inventory.md) (the weight inventory built on the same decode)
- [EXP-010](EXP-010-m05-native-fp16-stage.md), [EXP-011](EXP-011-m05-image-capture.md)
