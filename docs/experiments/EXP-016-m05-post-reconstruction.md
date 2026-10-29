# EXP-016: `post`'s Reconstruction Contract, CPU Oracle and Native Stage (M05 B4.5)

**Milestone:** M05 (checkpoint B4.5)
**Status:** **PASS** — the CPU oracle reproduces the captured reconstruction to
99.998% and the native D3D12 stage reproduces **all three of its outputs byte for
byte** (18 662 400 / 18 662 400 packed words, 37 324 800 / 37 324 800 surface
bytes, 18 662 400 / 18 662 400 half-resolution bytes), deterministically.
**Depends on:** [EXP-011](EXP-011-m05-image-capture.md) (the captured images),
[EXP-014](EXP-014-m05-dec0-mapping.md) (the method), [EXP-015](EXP-015-m05-enc0-contract.md)
(the same method one stage earlier), [EXP-009](EXP-009-m07-complete-inventory.md)
(the stage inventory).
**Kernel:** `rrlite_post_3_1_mvhi_ldr_folded`, module `dlss-0145-00.ptx` of
`nvngx_dlss.dll` 310.7.0, `.target sm_89`.
**Capture:** `replay-000001-rrlite_post_3_1_mvhi_ldr_folded`, frame 1, `reset = 1`,
zero motion, 960x540 render -> 2880x1620 reconstruction.

## 1. What the kernel is

`post_3_1` is the reconstruction: it resamples the low-resolution colour and the
network's per-pixel outputs with the anisotropic Gaussians the network predicted,
picks the two most likely motion candidates, fetches the previous result along
them, and blends.  It runs 180x51 blocks of 16x16 threads; each thread owns one
reconstruction column and a **pair of rows**.  It writes three surfaces, which is
what makes it measurable stage-by-stage:

| output | binding | geometry | format | captured post-launch |
|---|---|---|---|---|
| reconstruction | `param+96` | 2880x1620 | RGBA16F | yes (37 324 800 B) |
| packed LDR | `param+104` | 2880x1620 | 32-bit words | yes (18 662 400 B) |
| half-resolution | `param+112` | 1440x810 | RGBA32F | yes (18 662 400 B) |

The packed buffer is the one `downsample` consumes, and its format was already
established by EXP-012.

## 2. The contract, from the PTX

All offsets are `ld.param` slots of `dlss-0145-00.ptx`; the numbers in brackets
are line numbers.

**Inputs.** `param+0` colour (960x540, point sample), `param+32` gate and
`param+40` mix (RGBA8, read as f16), `param+48` covariance (RGBA16F),
`param+56` history (2880x1620), `param+176` display motion (RG16F).  The
one-byte format tags match dec0's outputs by content: post's image 3 is dec0's
gate (sha `f3c8973baa53`), image 4 its mix (`4f3caebc8872`), image 5 its
covariance (`56e7ab40f41d`).

**Parameters.** `+120` gain = 2.753011 (0f40303156); `+124/+128` scale = 1/3
(0f3EAAAAAB); `+132/+136` offset = 0; `+140` reconstruction 2880x1620; `+148`
render 960x540; `+192` display 1920x1080; `+200/+208` motion scale 1.5 and
offset 0; `+168` reset = 1.

**The tile** (36-282).  The block's first 160 threads (`r52 = tid.x + 16*tid.y`,
`if r52 > 159` skip) each fetch one render texel at
`clamp((r1 + col - 2, r2 + row - 2))`, `r1 = trunc(scale_x*16*bx)`,
`r2 = trunc(scale_y*32*by)`, and write four f16 into each of three 1280-byte
shared arrays:

```
t0 = [colour.r, colour.g, colour.b, mix.x]
t1 = [mix.y,    mix.z,    gate.x,   gate.y]
t2 = [gate.z,   cov.x*k,  cov.y*k,  cov.z*2k]      k = -0.7213249802589417 (0fBF38A8C1)
```

**The Gaussians** (298-1333).  Tap `(i, j)` samples `base + (i-1, j-1)` with
`base = floor(pos - 0.5)`, and the exponent is f16 at every step

```
e = f16( f16(dx^2*cov.xx) + f16(dy^2*cov.yy) ) + f16( f16(dx*dy)*cov.xy )
w = ex2(f32(e));   w16 = f16(w)
```

The nine accumulators are f16 (`fma.rn.f16x2`, one rounding per tap), and the
normaliser `W` is the f32 sum of the *unrounded* weights in tap order.  Taps are
visited column-major (`[base]`, `[base+80]`, `[base+160]`, `[base+8]`, ...: 80
bytes is one tile row of ten entries).

**The paired rows** (301-332).  The odd row of a thread's pair keeps the *even*
row's three texel rows and measures its own residual from the even row's base:
`dy1 = f16((b1+0.5) - pos1)`, minus 1 when `b1 != b0`.  This is the detail that
makes a per-pixel model wrong by up to a f16 ULP on half the frame.

**The blend** (1843-1873).  `g0 = sat(gain*W*gate.x)`, `ga = fma(g0, 1-gate.y,
gate.y)`, `outA = fma(cur-hist, ga, hist)`, and the same with `gate.z` for
`outB`; a non-finite sum falls back to `cur`.

**The outputs** (1902-2373).  `outA` goes to the RGBA16F surface, `outB` is
packed as `trunc(clamp(fma(c, 2047, 0.5), 0, 2047))` with 1023 for blue, and the
half-resolution surface gets `(own pair + neighbouring column's pair) * 0.25`.

## 3. Three conversions the capture fixes

These are not in the PTX's arithmetic, but they decide the bytes:

1. **The f16 surface truncates.**  Comparing round-to-nearest against the
   capture gives 55.80% identical samples; truncating toward zero gives 99.998%
   with mean error 7.6e-9.  The captured RGBA16F surface is therefore the
   truncation of the f32 result, and the native stage writes the f16 bits
   itself through an `R32_UINT` view.
2. **The half-resolution surface is 10-bit.**  Its captured values are exactly
   `k/1023` for integers `k` (0.25024438 = 256/1023), so the mean is quantised
   before the store.
3. **That surface holds the exact IEEE quotient.**  `k * (1/1023)` differs from
   `k / 1023` for 0.08% of codes; the capture is the division, and dxc
   strength-reduces a float division by a constant even under `precise`, so the
   shader forms the quotient in double and rounds once.

## 4. The CPU oracle

`scripts/m_post_model.py` implements §2 and §3 and compares against the
captured outputs.  On the canonical frame:

| output | identical | differing | note |
|---|---|---|---|
| packed words | 99.99769% | 108 of 4 665 600 | 60 / 29 / 19 split across the channels, all ±1 |
| reconstruction (f16) | 99.99800% | 280 of 13 996 800 | ±1 f16 ULP |
| half-resolution codes | 99.99917% | 29 of 3 499 200 | ±1 code |

Every packed difference sits exactly on a quantisation boundary: the scaled value
`outB*2047 + 0.5` has fractional part 0.0, so the reference's own
`ex2.approx`/`rcp.approx` — which are specified to a couple of f32 ulps — decide
the code.  The mechanism is identified, so the residual is attributed rather than
tolerated: 108 knife-edge words out of 4.67M is 2.3e-5.

## 5. The native stage

`src/native/post_cs.hlsl`: scalar, one thread per column and row pair, the
captured 16x16x(180x51) dispatch, three `groupshared` tiles, no WMMA.

| comparison | result |
|---|---|
| packed words vs capture | **100.0000%** (0 differing bytes of 18 662 400) |
| reconstruction surface vs capture | **100.0000%** (0 of 18 662 400) |
| half-resolution surface vs capture | **100.0000%** (0 of 18 662 400) |
| run 1 vs run 2 | **0 bytes differ** on all three |
| `m_stage_fixture.py check` | 0 problems, `identical_min = 0.9999` met |

*Correction (EXP-018):* these figures were taken over two runs, which is not
enough to establish determinism.  A 16-run sample later showed that 4 of 16 runs
differed, in a single block, at pixels whose display-motion fetch was out of
bounds; the dead fetch was removed and the stage is now 16/16 identical and exact.
The identity claims above are unchanged, the determinism claim is now backed by
the regression gate's two-runs-per-stage check.

The GPU is *more* exact than the CPU oracle because its `ex2.approx` and
`rcp.approx` are the same hardware operations NVIDIA's kernel used; the 108
knife-edge values the oracle cannot decide are decided identically by the GPU.

The fixture declares `identical_min = 0.9999` in the stage map — a floor chosen
from the oracle's measured residual with its mechanism identified, before the
native run, and the run then met the stronger claim of exactness.

## 6. What the frame does not exercise

- **History.**  `reset = 1` makes the history position unreachable: the kernel
  tests `q < 0 || q > recon || reset != 0` and falls back to the colour texel at
  `floor(pos)`.  The Catmull-Rom five-tap path (1613-1646) never runs.  The
  native shader implements the reset path only and the fixture validator rejects
  `reset != 0`.
- **Motion.**  The motion texture is zero on this frame (all four RG16F images
  are bit-identical zeros), so `q = pos + 1.5*mv` collapses to `pos` and the two
  candidates coincide.  The kernel samples that texture at
  `trunc(pos/recon * display)`, i.e. up to (1919, 1079), while the captured
  texture is **960x540** — a contradiction that zero motion hides.  EXP-015's
  E7 leaves the same question open; it is still unresolved here.
- **Two of the five textures.**  `param+64/72/80` (three RG16F images) are bound
  but never loaded.
- **The half-resolution surface's storage format.**  The PTX stores f32 to
  `param+112`, but the captured values are 10-bit codes, so either the surface is
  RGB10A2 (and the capture expanded it) or the kernel's store quantises.  The
  bytes are reproduced; the format is not settled.

## 7. Files

- `scripts/m_post_model.py` — the CPU oracle (§4)
- `src/native/post_cs.hlsl` — the native stage (§5)
- `scripts/m_stage_fixture.py` — the `post_3_1` stage map, its parameter block,
  its validators, and a typed-view geometry override (the RGBA16F surface is
  written as `R32_UINT` 5760x1620)
- `src/native/CMakeLists.txt` — the shader
- `tests/test_post_contract.py` — tile geometry, paired rows, packing, surface
  conversions, constants, shader/model agreement
- `tests/test_stage_fixture.py` — the `post_3_1` fixture validator
- Reproduce: `python scripts/m_stage_fixture.py export <capture> --stage post_3_1
  --out <dir>` then `run <dir>`, and `python scripts/m_post_model.py <capture>`

## 8. References

- `nvngx_dlss.dll` 310.7.0, module `dlss-0145-00.ptx`
- `references/sources/d4r/engine/post_m.comp` (d4r `05e19d4`) — consulted as a
  checklist for the algorithm's shape; the constants, the sorting, the surface
  conversions and the paired-row rounding above are read from the PTX, and two
  of the three conversions in §3 are not in that file
- [EXP-012](EXP-012-m05-native-downsample.md) — the consumer of the packed buffer
- [EXP-014](EXP-014-m05-dec0-mapping.md), [EXP-015](EXP-015-m05-enc0-contract.md)
