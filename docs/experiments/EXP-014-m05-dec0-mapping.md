# EXP-014: `dec0`'s Operand and Weight Mapping (M05 B4.3 / D1-D7)

**Milestone:** M05 (checkpoint B4.3/C19)
**Status:** PASS for D1-D7. B4.3 passes with a measured, bounded residual.
**Supersedes:** the blocker recorded in [EXP-013](EXP-013-m05-dec0-epilogue.md)
**Depends on:** [EXP-010](EXP-010-m05-native-fp16-stage.md), [EXP-011](EXP-011-m05-image-capture.md),
[EXP-012](EXP-012-m05-native-downsample.md)

## Question

EXP-013 could not reproduce any of `dec0`'s four captured surfaces and localised the failure
to the expansion matrix's element order. It also stated the next handle: *derive* the mapping
from the kernel itself instead of testing decodes of d4r's model. Three questions:

1. Which bytes does `rrlite_dec0_4x4_folded` read as tokens, as weights, as bias, and how does
   it address them (D1)?
2. Does the decoded matrix reproduce the captured surfaces once the operand mapping is right
   (D2-D5)?
3. Can the stage then be executed natively and compared surface by surface (D6, D7)?

Acceptance criteria, fixed before the GPU comparison:

- A. Every operand element's byte offset is traceable to an instruction in the captured
  kernel's PTX, not to a read of d4r's model.
- B. The CPU model reproduces all four surfaces with a measured agreement, and every remaining
  difference is attributed to a named arithmetic property.
- C. The native D3D12 stage reproduces the same four surfaces on the RX 7700 XT, deterministically.
- D. Byte equality is claimed only if achieved; otherwise it is reported as not achieved.

## Method

### Environment

As EXP-012/EXP-013. PTX source: the module `dlss-0162-00.ptx` of `nvngx_dlss.dll` 310.7.0,
`.visible .entry rrlite_dec0_4x4_folded`, `.target sm_89`, 1996 lines, extracted with the
existing PTX utility. Capture: frame 1 of `replay-000001-rrlite_dec0_4x4_folded` from the
EXP-011 run. Reference: the four post-launch surfaces.

### D1 - the operand addresses

The kernel's parameter block says (`ld.param` at fixed offsets):

| slot | PTX load | content |
|---|---|---|
| `param+0` | `ld.param.u64` | weights pointer |
| `param+16` | `ld.param.v2.u32` | render size **(960, 544)** |
| `param+24` | `ld.param.u64` | token field pointer |
| `param+56`, `+64` | `ld.param.u64` | motion texture, history texture |
| `param+120` | `ld.param.u8` | reset flag |
| `param+128/136/144/152` | `ld.param.u64` | feature, gate, mix, covariance surfaces |
| `param+160`, `+164` | `ld.param.v2.f32` | sigma (8.0), rho (0.0786) |

[S] `param+16` holds **544**, not 540: the captured `args.bin` reads `20 02 00 00` = `0x220` =
544. That is the *padded* render height, and it is what makes the token plane 272 rows tall.
Reading it as 540 was EXP-013's arithmetic error, and it shifted the second channel plane by two
rows - enough to make the whole stage look unreproducible.

[S] Token addressing, from `ld.global.u32 [%rd25 + 4*(row*3840 + col*8 + w)]` with
`3840 = (W/2)*8` words and a plane offset of `4*(272*3840)` bytes:

```
byte(ty, tx, c) = (c/32)*4177920 + ty*15360 + tx*32 + (c%32)
```

i.e. **two planes of 32 channels, 32 contiguous bytes per token, 480 tokens per row, 272 rows
per plane** - d4r's "NGX layout", 8 355 840 bytes, which is exactly what `dec1` wrote (the
capture shows `dec1`'s launch changing bytes `[16 711 680, 25 067 520)`, and that is the region
`dec0`'s token pointer targets).

Worked example, token (0,0), channels 0..3 and 32..35:

| logical (ty,tx,c) | byte offset | E4M3 code | value |
|---|---|---:|---:|
| (0,0,0) | 0 | `0xC9` | -4.5 |
| (0,0,1) | 1 | `0xC6` | -3.5 |
| (0,0,2) | 2 | `0xB8` | -1.0 |
| (0,0,3) | 3 | `0x3B` | 1.375 |
| (0,0,32) | 4 177 920 | `0xC1` | -2.25 |
| (0,0,35) | 4 177 923 | `0x40` | 2.0 |

### D2 - the weight instruction

[S] The whole matrix is one repeated load and 32 `mma` instructions:

```
rd42 = laneid*16 ;  rd4 = weights + rd42
ld.weak.global.ca.v4.u32 {r18,r19,r20,r21}, [rd4 + 512*kb]        kb = 0..15
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {d0,d1},
    {a0,a1,a2,a3}, {b0,b1}, {c0,c1}
```

The A operand (4 registers = 16 bytes per lane) is the token fragment; the B operand (2
registers = 8 bytes per lane) is the weight fragment, and each 16-byte load feeds two B
fragments (`{r18,r19}` and `{r20,r21}`). For `m16n8k32` with 8-bit operands the B fragment is
`k = 4*tig + j%4 + 16*(j/4)`, `n = groupID`. With `lane = 4*groupID + tig` this gives

```
byte(k, n) = 512*kb + 16*lane + 8*g + j
kb   = 2*(n/16) + k/32
lane = 4*(n%8) + (k%16)/4
g    = (n%16)/8
j    = k%4 + 4*((k%32)/16)
```

[M] That formula is **identical, element for element, to `m_weights.layout_offsets('dec0')`**
(64x128 comparisons) and is a bijection of bytes `[0, 8192)`. So the weight decode EXP-009 and
EXP-013 used was right all along; the operand it was fed was not.

[S] The A-operand derivation independently confirms the token layout: its four 32-byte regions
are `row groupID` and `row groupID+8` with channels 0..31, then the same at `+4177920` for
channels 32..63 - a plain row-major fragment, which is the two-plane layout above.

[S] Bias: `ld.global.u32 [weights + 8192 + 16*bi + 4*(lane&3)]` into the mma's C operand as
`{r,r}`, so the f16 at index `n` of the 128 is column `n`'s bias. The blob is
8 192 (matrix) + 256 (bias) = 8 448 bytes, and it does not overlap.

### D3/D4 - the column map and the activation inversion

[M] Rather than inverting the captured activations (EXP-013's approach, which was misled by
saturation), the four surfaces were reproduced **forward**: with the operands established, each
of the 128 columns was tested against every sub-position of every captured surface. The result
is unambiguous and exact:

| output | value | column (within a sub-position's block) | agreement |
|---|---|---|---:|
| gate 0 | `v0.x` | 0 | 99.999% |
| gate 1 | `v0.y` | 1 | **100.000%** |
| gate 2 | `v2.z` | 10 | **100.000%** |
| mix | `v2.w, v3.x, v3.y` | 11, 12, 13 | 99.93% |
| covariance | `v3.z, v3.w, v4.x` | 14, 15, 16 | (below) |
| feature gates | `v1.z..v2.y` | 6, 7, 8, 9 | 99.98% |
| feature fresh | `v0.z, v0.w, v1.x, v1.y` | 2, 3, 4, 5 | 99.98% |

[M] Sub-position *s* (a pixel's `(py&1, px&1)`) reads columns `s*32 + 0..19`, and the value order
within those 20 is d4r's `v0..v4` exactly. 17 of the 20 are read; `v4.y`, `v4.z`, `v4.w` are
computed and discarded. So the `live` permutation in `m_weights.front_end_layout` was also
right - the blocker was never in the weights or the column map.

### D5/D6 - the epilogue, and the two arithmetic properties it needs

The epilogue is transcribed from the PTX instructions, not from d4r. Two things the
transcription had to discover by measurement:

[M] **The RGBA16F stores truncate toward zero.** The residual was one-sided - `mine - ref` was
`+0` or `+1` ULP and never `-1` - which is the signature of a truncating conversion. Rounding to
nearest reproduces 47.08% of the covariance values; truncating reproduces **99.98%**. d4r's
`output_store.glsl` records the same behaviour for RADV's rgba16f stores. The native shader
therefore truncates explicitly (`HalfTrunc`).

[M] **The E4M3 subnormal decode has no implicit leading 1.** The first native version built
subnormals as `(1 + m/8) * 2^-6`, inflating code `m` by `8/m`. 17 of the 64 weights of column 0
are subnormal codes, so this corrupted a quarter of every product and left the surfaces at
92%/57%/35%/51%. With `m * 2^-9` the same binary reaches 99.999%/99.93%/99.98%/99.99%.

Everything else is a plain transcription: `sigmoid(v) = f16(1/(1+ex2(-1.4426951*f16(clamp(v,-20,20)))))`,
`mix` = an f32 `exp2` softmax normalised by a reciprocal, `cov = f16trunc((sigma*s)^2, (sigma*t)^2,
sigma^2*s*t*rho_eff(t))`, and the feature `(1-g)*history + g*tanh(v)` with `g = 0.5*tanh(0.5*v)+0.5`
in f16.

## Results

### CPU model (D6), frame 1

| Surface | identical | max step | residual |
|---|---:|---:|---|
| feature (SNORM8x4) | 99.9787% | 3 codes | 442 of 2 073 600 bytes |
| gates (UNORM8x4) | 99.9991% | 3 codes | 19 of 2 073 600 bytes |
| mix (UNORM8x4) | 99.9348% | 8 codes | 1 353 of 2 073 600 bytes |
| covariance (RGBA16F) | 99.9836% | 1 ULP outside 144 samples | rms 0.004 |

### Native D3D12 stage (D7), frame 1

`src/native/dec0_cs.hlsl`, one dispatch of 2040x1 groups of 64 threads (one thread per token),
executed twice in separate processes:

| Surface | identical | detail |
|---|---:|---|
| feature | **99.9787%** | max 3 codes, 442 differing bytes of 2 073 600 |
| gates | **99.9991%** | max 3 codes, 19 differing bytes |
| mix | **99.9348%** | max 8 codes, 592 differing pixels of 518 400 |
| covariance | **99.9883%** | f16 values identical; max |d| 1.1875, rms 0.00357, relative rms 6.2e-5; ≤1 ULP 99.9883%; 0 NaN/Inf |
| repeat run | identical | both processes byte-identical |

The native stage is therefore **as accurate as the CPU model and, on the covariance, slightly
better** (99.9883% vs 99.9836%) - it applies the truncating store the model only discovered
afterwards. Both reach the same answer from independent implementations (NumPy f64 vs HLSL
f64/f32), so the agreement also cross-validates the model.

### The residual, attributed

Every remaining difference falls into one of three classes, none of them a mapping error:

1. **Quantisation boundaries in the accumulator.** 86 covariance samples (of 518 400) differ by
   more than one step in channel 0, and 144 are one ULP off. They are scattered (no border, no
   row band, no sub-position preference) and appear where the accumulator sits exactly on an
   E4M3 midpoint, so the reference's `cvt.rn.satfinite.e4m3x2.f16x2` and the reconstruction take
   different sides of it. [U] The tie rule itself is round-to-nearest-even in both; what is not
   established is whether the hardware's tie differs for a handful of exact midpoints.
2. **`±0`.** 47 covariance samples were `+0` where with the reference has `-0`. They are
   numerically equal; the comparison now normalises them, because a raw bit-pattern ULP count
   calls `0x0000` vs `0x8000` the largest possible difference.
3. **The reference's own approximations.** `ex2.approx`, `rcp.approx` and `div.approx` are
   specified to a couple of ULPs, and the surfaces quantise to 8 bits or f16, so samples sitting
   on an output boundary can land one step apart. That is 19 gate bytes, 592 mix pixels and 442
   feature bytes.

**Byte equality is not achieved and is not claimed.**

## Interpretation

- [S][M] `dec0`'s operand and weight mapping is now traceable to the kernel's own instructions:
  the token field is two 272-row planes of 32 bytes per token at a 4 177 920-byte plane stride,
  the matrix is the blocked layout `layout_offsets` already described, and the bias is one f16
  per column. EXP-013's failure was its reading of `param+16` as 540 and, once that was fixed,
  the E4M3 subnormal decode.
- [M] The column map is d4r's: sub-position *s* uses columns `s*32 + 0..19` and the value order
  inside them is `v0..v4`. This was not assumed - it was established by forward comparison at
  100.000% on two gate channels.
- [M] The reference's f16 surface store truncates. This is an arithmetic property of the
  reference, not of the reconstruction, and it is worth 53 percentage points of covariance
  agreement.
- [U] Frames 2 and 3 of the capture have `reset = 0`, so they exercise the history path
  (motion texture, warped previous feature, `(1-g)*history`). The reconstruction implements the
  reset branch exactly and does not implement the sampling, so those two frames are
  **not** validated. This is the one part of the stage that remains open.
- [U] The two ULP-level classes above are the input to B4.6 (numerical conformance): the
  reference's `div.approx`/`rcp.approx`/`ex2.approx` lowerings on this stack are not modelled
  exactly, only bounded.

### Corrections to earlier documents

- [M] [EXP-013](EXP-013-m05-dec0-epilogue.md)'s blocker is resolved: its measurement that "no
  decode of the blob yields a matching column" was correct *for the operand it used*, and the
  decode it doubted was right. Its "token layout is settled" conclusion was right too; its
  plane-stride arithmetic was not.
- [M] [EXP-009](EXP-009-m07-complete-inventory.md)'s addendum doubted the front-end decode. The
  decode is now confirmed against the PTX, so the doubt is withdrawn: `dec0`'s and `enc0`'s
  matrix values stand as decoded. The **counts** and the layer matrices were never in question.
- [S] The M07 INT8 measurements do not change. The front-end matrices are two of 360, and their
  values are unchanged.

## Next

1. **B4.6** is now the natural place for the ULP classes above: model the reference's
   `div.approx`/`rcp.approx`/`ex2.approx` lowerings on this stack and see whether the boundary
   samples collapse. This is a bounded question with a clear oracle (the same four surfaces).
2. The history path (frames 2-3): the PTX carries it (`tex.base.2d` on the motion texture,
   `tex.level.2d` on the history, the `(1-g)*history` term). It needs its own fixture and its
   own comparison; the reset branch is done.
3. B4.4 `enc0`, then B4.5 `post`. `enc0`'s blob layout is the same instruction family (its
   `w8`/`w16` split and the 4 096-byte dither table are already in `front_end_layout`), so the
   method transfers directly.

## Files

- `src/native/dec0_cs.hlsl` (the stage), `src/native/m_image_stage.cpp` (the `R32_UINT` view),
  `src/native/CMakeLists.txt`
- `scripts/m_stage_fixture.py` (the `dec0` binding map, its parameter block, `validate_stage`)
- `tests/test_dec0_contract.py`, `tests/test_stage_fixture.py`
- Fixtures and results stay in the temporary workspace.

## References

- `nvngx_dlss.dll` 310.7.0, module `dlss-0162-00.ptx` (`.visible .entry
  rrlite_dec0_4x4_folded`), and `dlss-0160-00.ptx` (`rlite_dec1_4x4`) for the producer side
- `references/sources/d4r/engine/dec0_m.comp`, `activation_m.glsl`, `output_store.glsl`,
  `round_half.glsl` (d4r `05e19d4`) - the reference transcription, and `output_store.glsl` for
  the truncating-store note
- [EXP-011](EXP-011-m05-image-capture.md) (the captured surfaces), [EXP-012](EXP-012-m05-native-downsample.md)
  (the fixture path and the exact-f16 machinery), [EXP-013](EXP-013-m05-dec0-epilogue.md) (the
  failed attempt and its next handle), [EXP-009](EXP-009-m07-complete-inventory.md),
  [EXP-010](EXP-010-m05-native-fp16-stage.md)
