# EXP-015: `enc0`'s Resource Contract, Blob Layout and Feature Structure (M05 B4.4, E1-E7)

**Milestone:** M05 (checkpoint B4.4/C19)
**Status:** **PASS** — completed by the addendum at the end of this report, which
resolves the E4 failure below: the CPU oracle and the native D3D12 stage both
reproduce the captured token field byte for byte (8 355 840 / 8 355 840).
Sections 1-8 are the original investigation and are preserved unchanged; §9
records what was wrong and how it was established.
**Depends on:** [EXP-011](EXP-011-m05-image-capture.md) (the captured resources),
[EXP-014](EXP-014-m05-dec0-mapping.md) (the method), [EXP-009](EXP-009-m07-complete-inventory.md)

## Question

`rrlite_enc0_4x4_mvhi_ldr_folded` turns the render-resolution inputs into the 64-channel
E4M3 token field that `enc1` consumes. Three questions, in dependency order:

1. Which captured resources does it actually read, and at which parameter offsets (E1)?
2. What does its weight allocation contain, and how is it addressed (E3)?
3. What are the 128 input features of one token, and does a CPU reconstruction of them
   reproduce the captured token field (E2, E4)?

## Method

Kernel: `.visible .entry rrlite_enc0_4x4_mvhi_ldr_folded`, `.target sm_89`, 2 967 lines, from
the module **`dlss-0154-00.ptx`** of `nvngx_dlss.dll` 310.7.0 (extracted with the project's
existing PTX utility; the `mvlo` modules were not consulted). Capture: frame 1 of
`replay-000001-rrlite_enc0_4x4_mvhi_ldr_folded`, launch `121 x 69 x 1`, 32 threads, parameters
360 bytes, weight allocation 12 416 bytes. Reference: `gpu-output/alloc-1.bin` at the launch's
own output pointer.

## E1 — the resource and parameter contract: PASS

[S] Every parameter the module loads, and the value the capture recorded. `[%rd1+N]` in the PTX
is `param+(N+8)`, which is worth stating because misreading it by 8 attributes every handle to
the wrong slot.

| param | kind | captured value | role (from its use) |
|---|---|---|---|
| +0 | u64 | alloc 0 | weights (noise table, matrix, bias) |
| +8 | 2 u32 | (2, 2) | block origin: `r1 = ctaid.x*8 - origin.x` |
| +16 | 2 u32 | **(960, 544)** | padded size: sizes the token field |
| +40 | u64 | alloc 1 + 0 | **never loaded** |
| +48 | u64 | alloc 1 + 0xFF0000 | token output (the stores' base) |
| +56 | 2 u32 | (960, 540) | visible size (the mirror bound) |
| +64 | 2 f32 | (1/960, 1/540) | render inverse size |
| +80 | u64 | image 0 | colour texture, f16x4 |
| +88 | 2 u32 | (0, 0) | colour window origin |
| +96 | u64 | image 1 | motion texture, f16x2 |
| +104 | 2 u32 | (0, 0) | motion window origin |
| +112 | u64 | image 2 | **1x1 texture, never loaded** |
| +120 | u64 | image 3 | **1x1 texture, never loaded** |
| +128 | u64 | image 4 | previous motion (`prevMvTex`), f16x2, sampled with `tex.level` |
| +136 | u64 | image 5 | **depth, never loaded** |
| +152 | u64 | image 6 | history (`halfTex`), f32x4 at 1.5x render (1440x810) |
| +160 | u64 | image 7 | previous recurrent feature (`featTex`), snorm8x4 |
| +168 | 2 f32 | (0, 0) | jitter |
| +176 | 2 f32 | (0, 0) | — |
| +184 | 2 f32 | (1/3, 1/3) | — |
| +192 | 2 f32 | (3, 3) | — |
| +200 | 2 f32 | (1.5, 1.5) | motion scale |
| +208 | 2 f32 | (0, 0) | motion offset |
| +224 | 2 f32 | (2, 2) | display-motion pixel scale |
| +232 | 2 u32 | (959, 539) | colour window maximum |
| +240 | 2 u32 | (1919, 1079) | motion window maximum (display resolution) |
| +248 | 2 u32 | (2880, 1620) | history size |
| +256 | 2 f32 | (1/2880, 1/1620) | history inverse size |
| +268 | f32 | 9.045065 | a scale on the feature values |
| +272 | f32 | 0.2235685 | — |
| +280/288/296 | u64 | alloc 10/11/12 | the three RG16F motion surfaces (written) |
| +312 | u32 | 0 | frame (the dither table's frame index) |
| +316 | u8 | 1 | reset |

[M] **Four captured bindings are never loaded**, which settles the questions EXP-011 left open:
the two 1x1 textures at `param+112` and `param+120` are not read at all, and neither is the
depth texture at `param+136` nor the pointer at `param+40`. The depth texture's absence is the
signature of the display-motion variant: [S] d4r's `enc0_features.glsl` documents the
`ENC0_DISPLAY_MV` branch as taking "one vector per pixel, no depth", and this kernel has exactly
that shape - one `tex.base.2d` fetch of the motion texture at
`floor((pixel + 0.5 - jitter) * 2)` clamped to `[0, 1919] x [0, 1079]`.

[M] The three RG16F surfaces **are** written: `sust.b.2d.v2.b16.zero` at `param+280/288/296`,
guarded by `outside = raw.x < 0 || raw.y < 0 || |x| >= size.x || |y| >= size.y`. They hold the
three motion candidates, and they are zero on this frame because the motion texture and the
motion offset are zero - not because the kernel never writes them.

[U] The motion texture captured here is 960x540 while the sampling window is the display size
(1919x1079). On this frame the texture is all zeros, so the discrepancy cannot be observed; a
non-zero-motion capture would settle whether the real binding is display-resolution.

## E3 — the blob: PASS

[M] The 12 416-byte allocation is exactly 4 096 (dither) + 8 192 (matrix) + 128 (bias), and the
PTX addresses it as three consecutive regions.

[S] Matrix load: `ld.weak.global.ca.v4.u32` at `weights + lane*16 + 4096 + 512*kb`, `kb = 0..15`,
16 bytes per lane, feeding two `m16n8k32` B fragments (`{r,r+1}` and `{r+2,r+3}`). With `K = 128`
the four k-chunks of 32 step through blocks `4*(n/16) + k/32`:

```
byte(k, n) = 4096 + 512*kb + 16*lane + 8*g + j
kb = 4*(n/16) + k/32 ; lane = 4*(n%8) + (k%16)/4 ; g = (n%16)/8 ; j = k%4 + 4*((k%32)/16)
```

[M] This is **identical, element for element, to `m_weights.layout_offsets('enc0')`** (all
128x64 elements) and is a bijection of bytes `[4096, 12288)`. So the inventory's enc0 matrix
decode is now source-grounded, as dec0's was.

[M] Bias: `ld.global.u32 [weights + 12288 + 16*bi + 4*(lane&3)]` into the mma's C operand as
`{r,r}`; the word holds the f16 pair of columns `8*bi + 2*(lane&3)` and the next, so the f16
index of column *n* is *n*. 64 columns.

[M] **452 of the 8 192 weight codes are E4M3 subnormals** (5.5%): the decode bug EXP-014 found
would corrupt 5.5% of every product here, so it matters for this stage too.

[M] Dither. The table is 4 096 bytes = 2 048 u16 entries, indexed
`(frame << 8) | ((|y| & 15) << 4) | (|x| & 15)` - the PTX builds it with
`bfi(y&15, x&15, 4, 4)` then `prmt.b32 d, frame, tile, 0x2104`, so `d = tile | (frame << 8)`.
The entry is read as `ld.global.u16` and shifted `>> 9` to give `n` in `0..127`, which is then
OR-ed into the value's f16 exponent and added/subtracted:

```
h = f16(clamp(v, 0, 1024)) ; e = bits(h) & 0xFC00
t = f16(f16(e | n) + h) - f16(e) ; r = f16(bits(t) & 0xFF80)
```

so `r` is a *truncation* of `h` displaced by a position-dependent fraction of one E4M3 step.
Over a 16x16 tile the captured table gives 123 distinct `n`, so the dither is very much active.
[M] The overshoot above 448 is not clamped here: `cvt.rn.satfinite.e4m3x2.f16x2` is what
saturates.

## E2 — the features: partly established

[M] Structure, from the PTX:

- a workgroup is 8x8 pixels (32 threads, two passes of 8x4, `r1208` and `r1208 + 32`);
- each pixel computes **32 feature slots** and stores them as two 16-byte `st.shared.v4.u32`
  writes, at `smem + 16*(8*(tid/8) + tid%8)` and `+1024` - so the shared tile is two 1 024-byte
  planes of 16 slots x 64 pixels;
- a token is a 2x2 pixel block, so it gathers **4 x 32 = 128** features, and each pixel is one
  k32 step of the embedding.

[M] On a reset frame the feature vector collapses to the colour texel. `reset = 1` makes the
three texture-dependent terms vanish: the history sample (`halfTex`, guarded by
`!(... || reset != 0)`), the previous recurrent feature (`featTex`, guarded by `reset == 0`) and
the motion-change scalar `mf` (also guarded). All the `h[k]` then equal the pixel's own colour,
so every nonzero slot is one of the three colour channels - 15 of the 32 slots, the other 17
zero.

[M] The reference field is correspondingly degenerate: **24 distinct 64-byte token vectors** over
130 560 tokens, 117 distinct E4M3 codes, 99.8% of bytes nonzero, values in `[-1.0, 1.375]`.

## E4 — the CPU model: FAIL (does not reproduce)

`scripts/m_enc0_model.py` implements everything above: the padded-plane token field, the mirror,
the 32-slot vector with the dither, the embedding (bias as the initial f16 accumulator, one f16
rounding per 32 products), and the E4M3 output.

[M] It produces **12 distinct token vectors** against the reference's 24, values in
`[-0.9375, 1.625]` against `[-1.0, 1.375]`, and **1.3% byte agreement** - i.e. chance.

[M] The failure is not a permutation of anything already established. The quarter order (which
pixel supplies which k32 step) and the output-column order (the `gp` permutation
`front_end_layout` describes) were both searched exhaustively - 4! orders x 2 channel directions
= 48 combinations - and the best is **1.32%**. Nor is it a scale: the sorted-value multisets of
the 24 reference vectors and the model's vectors share no member, so the *values* differ, not
just their arrangement.

[U] Therefore the earliest divergence is upstream of the embedding: either the slot map (which
of the 32 slots carries which value) or a feature constant. The candidates, in order of
likelihood, are (a) the 15-slot map, which is taken from d4r's pinned file rather than derived
from the PTX's shared-memory transpose; (b) `param+268`'s scale, which is applied to the feature
values somewhere but whose exact use was not traced; (c) the smem transpose between the two
`st.shared` groups, which reorders the slots before the mma reads them.

## E5 — the native stage: not attempted

No `enc0_cs.hlsl` exists. Implementing it before E4 converges would only reproduce a wrong
feature vector on the GPU.

## E6 — the output field and the `enc1` edge: PASS (structural part)

[M] The output pointer is `param+48` = `alloc 1 + 0xFF0000`, the same address `dec0` later reads
as its token input and `dec1` overwrites. The field is 8 355 840 bytes = two 4 177 920-byte
planes, i.e. **272 padded rows**, and the launch changed 8 339 370 of those bytes (the remainder
are padding bytes that were already equal).

[S] The store's arithmetic is the mirror image of `dec0`'s load: `r47 = (param+16.x >> 1) << 3`
words per token row, `r48 = (param+16.y >> 1) * r47` for the plane, so
`byte(ty, tx, c) = (c/32)*4177920 + ty*15360 + tx*32 + (c%32)`. **The padded size at `param+16`
drives both terms**, exactly as in EXP-014 - using the visible 540 would shift channel 32..63 by
two rows.

[M] The established edge holds: `enc1`'s pre-launch image of that region is **byte-identical** to
`enc0`'s post-launch image, and `enc1`'s parameter block points at the same address
(`+24 = 0x270e1c0000`). So the field this report describes is genuinely what `enc1` reads. This
does *not* say anything about the reconstruction's fidelity, which is E4's failure above.

## E7 — limitations

- [U] Only the canonical frame: `mvhi`, LDR, frame 0, `reset = 1`, zero motion, a synthetic
  checkerboard with constant depth. The history, previous-feature and motion-change paths are
  all *unexercised*: on this frame they are exactly zero, so nothing here validates them.
- [U] Frames 2-3 of the capture have `reset = 0` and would exercise the history path; they were
  not used.
- [U] The `mvlo` variant was not consulted. `HDR`, dynamic resolution and other presets are out
  of scope.
- [U] The motion texture's resolution ambiguity (E1) cannot be resolved on a zero-motion frame.
- [U] The 1x1 textures and the depth texture are not read *by this kernel*. That is a statement
  about this variant only; it does not say the pipeline ignores them elsewhere.

## Interpretation

- [M] `enc0`'s resource contract, blob layout and output geometry are established and traceable
  to instructions, and the matrix decode the inventory used is confirmed. That part of B4.4 is
  done.
- [M] The feature vector on a reset frame is a pure function of one colour texel per pixel, which
  makes the canonical frame far more tractable than it first appeared - and the reference's 24
  distinct token vectors confirm the degeneracy.
- [M] The CPU model nevertheless fails, and the failure is localised to the slot map: not the
  weight layout, not the bias, not the dither table, not the accumulator contract, not the token
  geometry, and not any of the 48 permutations examined.
- [⇒] B4.4 stays **IN PROGRESS**, not COMPLETE. The remaining work is bounded and concrete: derive
  the 32-slot order from the PTX's shared-memory transpose rather than from d4r's file, then
  trace `param+268`'s scale. Both are inside one bounded instruction slice.

## Next

1. Read the smem transpose (PTX lines 1780-2010): 8 `ld.shared` pairs are re-stored at new
   addresses before the mma reads them, and that is exactly the slot reorder. Deriving it
   replaces the assumption that d4r's slot map holds for this variant.
2. Trace `param+268` = 9.045 065 and `param+272` = 0.2236 to their uses.
3. Only then implement `enc0_cs.hlsl` and run the E5 comparison (native vs the captured field
   vs `enc1`'s input).

## Files

- `scripts/m_enc0_model.py` (the instrument; documented as not reproducing the reference)
- `tests/test_enc0_contract.py` (blob, dither, dither index, token field, slot invariants)
- No shader, no fixture: E5 was not attempted.

## References

- `nvngx_dlss.dll` 310.7.0, module `dlss-0154-00.ptx`
- `references/sources/d4r/engine/enc0_m.comp`, `enc0_features.glsl`, `fp8_soft.glsl`
  (d4r `05e19d4`) - the same 32 slots and the `ENC0_DISPLAY_MV` note
- [EXP-011](EXP-011-m05-image-capture.md), [EXP-014](EXP-014-m05-dec0-mapping.md),
  [EXP-009](EXP-009-m07-complete-inventory.md)

---

# Addendum (same milestone, same capture): the E4 failure resolved, E2/E5/E6/E7 closed

Baseline `4bd0bcd`. Everything below is measured on the canonical capture
(`replay-000001-rrlite_enc0_4x4_mvhi_ldr_folded`, frame 1, `reset = 1`) against
`gpu-output/alloc-1.bin` at the launch's own output pointer.

**Correction to §7 of the original text.** The script's docstring claimed the model produced
"exactly as many distinct token vectors as the capture (24)". The measured run produced **12**
distinct vectors. The docstring was wrong; the number in this report was right. The corrected
model now produces 24 and every one of them equals a captured vector.

## 1. The dither transcription error (F1): confirmed, and it was real

The original §E3 formula was right and the code did not implement it. The PTX is
(`dlss-0154-00.ptx` 824-855, and eight more copies of the same block):

```
and.b32   %r229, %r225, -67044352   // 0xFC00FC00: the exponent of h
or.b32    %r226, %r251, %r229       // (e | n)
add.f16x2 %r224, %r225, %r226       // h + (e|n)      <- %r225 is h
sub.f16x2 %r227, %r224, %r229       // - e
and.b32   %r245, %r227, -8323200    // 0xFF80FF80
```

`%r225` is `mov.b32 {%rs62, %rs63}` where both halves are `cvt.rn.f16.f32` of the clamped feature
value. The model had `base` (the exponent-only value) in that slot, so it computed
`f16(e + (e|n)) - e` instead of `f16(h + (e|n)) - e` and every value in an exponent bin collapsed
onto the same dither result. The masks in the PTX are exactly `0xFC00FC00` and `0xFF80FF80`
(-67044352 and -8323200 as PTX prints them), which is what pins the register roles: only the
`and 0xFF80…` keeps a sign-preserving truncation, and the `add` must carry the mantissa or the
mask has nothing to truncate.

Measured effect in isolation, with the slot map and channel order still wrong:

| dither | byte agreement | distinct vectors | value range |
|---|---|---|---|
| `e` in the add (before) | 1.3158% | 12 | [-0.938, 1.625] |
| `h` in the add (after) | 0.9784% | **24** | [-1.625, 2.250] |

The byte agreement *fell*. The distinct-vector count reached the reference's 24, which was the
first evidence that and the value range was still wrong for a second reason: the slot map.

## 2. The feature slots (F2/F3): 15 positions were right, 8 channel assignments were not

The original §E2 took d4r's 32-slot map on trust. The PTX's own derivation, in three steps:

**(a) The 20 values per pixel.** `f292..f311` are selected from clamped floats: `f292..f294` are
the pixel's own texel, `f295..f306` are the four motion candidates' RGB (`f679..f690`, which on
this frame are four more point-samples of the *same* texel at `{r92,r93}` - the LDR/reset branch
of the candidate loop), `f307` is `f337` (the motion-change scalar) and `f308..f311` are the
history RGBA (`f691..f694`).

**(b) Which are zero on a reset frame.** `rs237` carries `reset` on this path (`ld.param.u8
%rs237, [%rd1+308]` at 350), so `p55 = (rs237 != 0)` and `p63 = (rs234 == 0)` are both decided by
it: the history slots become 0 (`f691..f694 = 0` at 365-369) and `f337` becomes 0 (line 424).
Exactly 15 of the 20 survive.

**(c) The physical order.** The two `st.shared.v4.u32` stores pack the dither results two per
32-bit word:

```
[r256]      = {r230,r231} {r232,r233} {r234,r235} {r236,r237}
[r256+1024] = {r238,r239} {r240,r241} {r242,r243} {r244,r245}
```

with `r230..r245` coming from `(f292,f293) (f308,f309) (f294,f295) (f310,f311) (f296,f297) (0,0)
(f298,f299) (0,0) (f300,f301) … (f306,f307) (0,0)`. Applying that packing to (a) gives the slot
map, and because `f292..f306` cycle r,g,b, the 15 nonzero slots cycle r,g,b in ascending slot
order:

```
slot  0  1  4  5  8  9 12 13 16 17 20 21 24 25 28
ch    r  g  b  r  g  b  r  g  b  r  g  b  r  g  b
```

The *positions* agree with d4r's map. The *channels* do not: d4r assigns slot 8 = r where this
kernel has g, slot 12 = g where this has r, and so on for 8 of the 15. That difference is why
the model could produce 24 plausible vectors and match none of them.

**The shared-memory "transpose" is a no-op.** The eight `ld.shared` pairs at 1784-1881 are
re-stored at 1909-1964 to the *same* addresses: the load and store address blocks are the same
expression (`r940 = lane>>2`, `r941 = lane&3`, `r942 = r940&3`, …) with the store's copy reading
`r420 = %laneid` instead of `r419`. Emulating the address block for all 32 lanes confirms
`store_word(i) == load_word(i)` for every i. So the mma reads exactly what the pixel wrote, and
the k order follows from the read addresses:

| k | pixel | field token |
|---|---|---|
| 0-31 | `p` | `(2 ty, 2 tx)` |
| 32-63 | `p+1` | `(2 ty, 2 tx+1)` |
| 64-95 | `p+8` | `(2 ty+1, 2 tx)` |
| 96-127 | `p+9` | `(2 ty+1, 2 tx+1)` |

and `k = slot index`, with slots 0-15 the first shared-memory plane and 16-31 the second. The
k-quarter order is the natural 2x2 one, which the earlier exhaustive search had already
confirmed was not the problem.

## 3. The epilogue's channel permutation (the last 87%)

`mma 7, 8, 15, 16, 23, 24, 31, 32` are the four-k32-step chains' final accumulators - eight
chains of eight columns. The epilogue clamps each to `+-2*pi` (`max.f16x2`/`min.f16x2` with
`0f40C90FDB`, whose f16 value is 6.28125), converts with `cvt.rn.satfinite.e4m3x2`, and packs
*two chains into one stored word*:

```
word at byte 4j        <- chain (n/16=0, g=0) columns 2j, 2j+1, then chain (n/16=1, g=0)
word at byte 16+4j     <- chain (n/16=0, g=1), then chain (n/16=1, g=1)
```

so field channel `c` is not mma column `c`:

```
c:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
n:  0  1 16 17  2  3 18 19  4  5 20 21  6  7 22 23
c: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
n:  8  9 24 25 10 11 26 27 12 13 28 29 14 15 30 31
```

(and the same shifted by 32 in the second plane). A bijection of 0..63 that preserves the plane,
*not* the identity - the model's "column n is channel n" was wrong for 62 of the 64 channels.

## 4. `param+268` and `param+272` (F4): both are motion-only, both are inert on this frame

| parameter | value | only use | reaches |
|---|---|---|---|
| +268 | 9.045065 | 203: `f222 = f9 * 540.0`, then 204: `f31 = f222 / 540.0` (f29 is `cvt.rn.f32.u32` of the visible height, 540) | multiplies the previous-motion residual at 205-208 |
| +272 | 0.2235685 | 259: `f52 = f659 * 0.6931472 * log2(f678)` | the motion-change scalar |

They compose into one quantity: `f31 = 9.045065` (the 540s cancel),
`f32 = |9.045065 * dmv|^2`, `f678 = 1 + sqrt(f32)`, `f52 = 0.2235685 * ln(f678)`, and `f52` is
then `selp`-ed against zero by `p63` (line 424). `f52` has exactly one consumer: the slot-29
feature. On a reset frame `p63` is false, the slot is zero, and neither constant can affect the
output - which the byte-exact reproduction below confirms empirically.

## 5. Result: E4 and E5, byte for byte

CPU oracle (`scripts/m_enc0_model.py`, rewritten from the above):

| metric | before | after |
|---|---|---|
| identical bytes | 1.3158% | **100.0000%** (8 355 840 / 8 355 840) |
| fully identical tokens | 0.0000% | **100.0000%** (130 560 / 130 560) |
| distinct token vectors | 12 | **24** (the reference's 24) |
| value range | [-0.938, 1.625] | **[-1.000, 1.375]** |
| per-plane agreement | - | 100% / 100% |

Native D3D12 (`src/native/enc0_cs.hlsl`, fixture `fixtures/enc0`, RX 7700 XT gfx1101):

| comparison | result |
|---|---|
| native vs captured field | **100.0000%** (0 differing bytes of 8 355 840) |
| native vs the CPU oracle | **100.0000%** |
| native vs `enc1`'s captured input | **100.0000%** (0 of 8 355 840) |
| run 1 vs run 2 | **0 bytes differ** |
| per-plane / per-token | 100% / 100% |
| `m_stage_fixture.py check` | 0 problems, `identical_min = 1.0` met |

The `expect.identical_min = 1.0` was declared in the stage's fixture map before the run, so the
criterion was not chosen after seeing the result. Note that both comparisons in the last row are
distinct claims: the captured `enc0` output equals `enc1`'s input because that is the established
edge (original to original), while the native output equals both because the reconstruction is
byte-exact.

## 6. E7: what the canonical frame still does not exercise

Unchanged from §E7, with two clarifications the reconstruction now supports:

- The history slots (`f308..f311`), the previous-feature slots and the motion-change scalar are
  **provably zero** on this frame, and the native shader implements the reset path only. Its
  fixture validator rejects `reset != 0` so the gap cannot be crossed silently.
- `param+268` / `param+272`, the previous-motion texture and the three motion surfaces are
  implemented by *nothing* in the native stage, and cannot affect its output on this frame (§4).
- Still unknown and out of scope: `mvlo`, HDR, non-zero motion, frames 2+ (history), the motion
  texture's real resolution (960x540 captured against a 1919x1079 sampling window), and the two
  1x1 textures the kernel never loads.

## 7. Files

- `scripts/m_enc0_model.py` - the CPU oracle, now byte-exact
- `src/native/enc0_cs.hlsl` - the native stage (scalar, no WMMA)
- `scripts/m_stage_fixture.py` - `enc0` stage map, `enc0_params`, the stage's validators, and a
  buffer-UAV export path (the field is written with `st.global`, not through a surface)
- `src/native/CMakeLists.txt` - the shader
- `tests/test_enc0_contract.py` - dither semantics and hand-computed cases, the slot derivation,
  the epilogue permutation, and a shader/model table cross-check
- `tests/test_stage_fixture.py` - the `enc0` fixture validator
- Reproduce: `python scripts/m_stage_fixture.py export <capture> --stage enc0 --out <dir>` then
  `run <dir>`, and `python scripts/m_enc0_model.py <capture>`.

