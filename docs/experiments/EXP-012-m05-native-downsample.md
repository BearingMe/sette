# EXP-012: Native `downsample` Stage (M05 B4.1, B4.2)

**Milestone:** M05 (checkpoints B4.1/C19 fixture infrastructure and B4.2/C19 native downsample)
**Status:** PASS for B4.1 and B4.2; B4 as a whole is IN PROGRESS (`dec0`, `enc0`, `post` remain)

## Question

`rrlite_downsample_kernel_static_ldr` is the last launch of a Preset M frame and the only
stage whose output has an oracle independent of the capture path (`downsample`'s CUDA
surface equals the probe's D3D12 readback byte for byte, EXP-011). Two questions:

1. Can a captured launch be executed natively on Windows/D3D12 from the capture alone -
   formats, geometry, pitch, parameter block, dispatch - without reinterpreting bytes?
2. Does a reconstruction from `down_m.comp` plus the captured parameter block reproduce
   the captured surface exactly, including the 11-11-10 unpack, the alternating 5-tap
   phases, the per-pass clamp and the f16 arithmetic?

Acceptance criteria, fixed before the GPU comparison:

- A. Every binding of the stage resolves to a declared format and a byte count that agrees
  with the captured geometry; a mismatch is an error, not a reinterpretation.
- B. Each of a representative set of captured resources survives upload, dispatch and
  readback unchanged (B4.1's transport gate), across every format the stages use.
- C. Native `downsample` output equals the captured reference surface **byte for byte**;
  if that is unattainable, characterise the difference (ULP distribution, spatial
  localisation) and explain it.
- D. Two separate runs produce identical output.

## Method

### Environment

Windows 11, RX 7700 XT (gfx1101), driver 32.0.32015.7029, DXC 1.8.2502.11, CMake 4.4.3 /
VS 2022. Capture: `nvngx_dlss.dll` 310.7.0, preset 13, 1920x1080 output from a 960x540
render, frame 1 of the EXP-011 run (`capture-img`), ZLUDA patch 0018 build. Reference: the
captured `gpu-output/image-0.bin` of
`replay-000001-rrlite_downsample_kernel_static_ldr`, which EXP-011 showed is
byte-identical to the probe's `output-0.rgba16f` readback.

### B4.1 - fixtures

`scripts/m_stage_fixture.py` turns a captured launch into a fixture the new
`src/native/m_image_stage.cpp` executes. The manifest is text and every line is a
declaration, not an inference:

```text
version 2
shader  downsample_cs.cso
threads 16 16 1          # the recorded block
groups  120 68 1         # the recorded grid
params  params.bin 64
srv buffer 0 in-src.bin 18662400
srv tex2d  1 R16G16B16A16_FLOAT 960 540 in-colour.bin
uav tex2d  2 R16G16B16A16_FLOAT 1920 1080 out-out.bin
```

The exporter (a) resolves each role from the argument slot the capture recorded,
(b) maps the driver's backing-storage code to a D3D12 format and checks the captured row
stride against that format, (c) rebuilds the parameter block from the slots d4r's own
`test_down_m.py` reads, and (d) refuses a launch whose geometry contradicts the stage
(`validate_stage`). `fixture.json` records every mapping so the fixture can be audited
without rerunning the capture.

`roundtrip` builds a copy fixture per resource and runs `image_copy_cs.hlsl` (identity,
one typed view, no reinterpretation) through the same upload/dispatch/readback path.

### B4.2 - reconstruction

`src/native/downsample_cs.hlsl`. Three things were derived rather than assumed:

1. **Geometry.** d4r's `down_m.comp` filters through a shared tile with `base =
   1.5*BLOCK - 2`, a horizontal centre at `3*(q/2) + 2*(q&1) + 2` inside a span that
   starts at `base.x + 1.5*x0`, and a vertical centre of `floor(1.5*(py+0.5)) - base.y`
   rows. Cancelling the tile terms gives, for the captured block (16x16) and `grid.z = 1`,
   `A = ceil(1.5 px)` and `T = floor(1.5 (py+0.5))` with the weights mirrored by the
   parity of the output coordinate. `tests/test_downsample_contract.py` transcribes the
   tile algorithm literally and checks the two agree for every (px,py) sampled from three
   block/SEG configurations, so the cancellation is a tested claim, not a simplification.
2. **Packing.** The 18 662 400-byte source is 2880x1620 32-bit words: 11-11-10 bits over
   the full code range, i.e. 4 bytes per *source* pixel and 9 bytes per *output* pixel
   (1.5 x 1.5 = 2.25 source pixels per output pixel). The LDR variant reads the codes as
   display-referred values (`code/2047`, `code/2047`, `code/1023`); the HDR branch
   (range expansion, clamp to [1e-4,3], exposure division) does not run.
3. **Arithmetic.** The stage is f16 end to end. A fused f16 multiply-add rounds once;
   f32 rounds twice and native `half` on this stack is worse (a divide-and-add pair plus a
   truncating f32->f16 conversion, EXP-010). The shader therefore reproduces each f16
   operation exactly: an f16 x f16 product has a 22-bit significand, so it is exact in
   f32; `TwoSum` returns the exact sum of two f32 values as `(high, low)`; and rounding to
   f16 needs only the high limb's position relative to the midpoint plus, when the high
   limb is *on* the midpoint, the sign of the low limb. `RoundPair` implements that in
   integer arithmetic.

Commands (fixtures stay outside the repository - they contain NVIDIA-derived data):

```powershell
cmake -S src/native -B build/native -G "Visual Studio 17 2022" -A x64
cmake --build build/native --config Release
python scripts/m_stage_fixture.py roundtrip capture-img --out fixtures/roundtrip
python scripts/m_stage_fixture.py export capture-img --stage downsample --out fixtures/downsample
python scripts/m_stage_fixture.py run fixtures/downsample --json down-check.json
```

## Results

### B4.1 - transport round trip (criterion B)

[M] 8 of 8 resources returned unchanged, covering every format the four image stages bind
and two sizes that exercise a nontrivial pitch:

| Resource | Bytes | Differing |
|---|---:|---:|
| `downsample` packed source (buffer, 18.6 MB) | 18 662 400 | 0 |
| `downsample` colour 960x540 f16x4 | 4 147 200 | 0 |
| `downsample` frame 1920x1080 f16x4 | 16 588 800 | 0 |
| `enc0` colour 960x540 f16x4 | 4 147 200 | 0 |
| `enc0` depth 960x540 f32 | 2 073 600 | 0 |
| `enc0` motion 960x540 f16x2 | 2 073 600 | 0 |
| `dec0` feature 960x540 snorm8x4 (UNORM view) | 2 073 600 | 0 |
| `post` history 2880x1620 f16x4 (37 MB) | 37 324 800 | 0 |

The runtime reported the copyable row pitch for each texture, so the file/pitch conversion
is visible rather than assumed: the 960-wide f16x4 surfaces use 7680-byte rows and the
2880-wide one 23 040.

### B4.2 - downsample (criteria C and D)

[M] Native output versus the captured surface, first frame:

| Metric | Value |
|---|---|
| Byte agreement | **16 588 800 / 16 588 800 (100.0000%)** |
| f16 agreement | **8 294 400 / 8 294 400 (100.0000%)** |
| ULP histogram | `{0: 8294400}` |
| max abs error, RMS, relative RMS | 0, 0, 0 |
| NaN / Inf | 0 |
| Repeat run (separate process) | byte-identical |
| Runtime | one dispatch, 120x68 groups of 16x16 |

The alpha channel is also exact: it is the render-resolution colour's alpha
(`texelFetch` at `floor((px+0.5)*0.5)`) passed through, and `finalOutputValue` is the
identity for an rgba16f target.

### Two defects the comparison found

Recorded because both were invisible to inspection and would silently survive a
"compiles and looks plausible" check:

1. [M] `RoundPair` first built the f16 sign bit from the f32 sign position
   (`u & 0x80000000`), so every negative intermediate came back positive. A flat field
   then summed to ~1% above its own taps and the min/max clamp pinned the entire frame to
   the maximum tap: 1 925 860 of 2 073 600 pixels wrong, max error 0.48. Fixed by moving
   the bit to position 15.
2. [M] 51 of 8 294 400 halves were one code low, all at exact f16 midpoints. The compiler
   contracted `w * b` into the `TwoSum` add as a fused multiply-add, which yields a
   *different* exactly-rounded value, so the recovered residual had the wrong sign and the
   tie went the wrong way. Marking the product and the `TwoSum` steps `precise` fixed all
   51. This is why the arithmetic is documented as "one rounding per f16 operation": the
   shader has to prevent the compiler from being more accurate than the reference.

## Interpretation

- [M] The capture is sufficient to execute this stage natively: formats, geometry, pitch,
  parameter block and dispatch all come from the recorded launch, and the transport test
  shows none of them is being reinterpreted.
- [M] The reconstruction reproduces the reference **bit-exactly**, so for this stage the
  FP16 oracle is established at full precision rather than within a tolerance. The f16
  operation order (multiply, then four fused multiply-adds, then the tap clamp) and the
  alternating weight phase are therefore confirmed, not merely consistent.
- [S] The 9-bytes-per-output-pixel packing recorded as unknown in the stage graph is the
  11-11-10 word (4 bytes) at 1.5 x 1.5 source pixels per output pixel.
- [U] The captured block is 16x16 rather than the 8x64 `down_m.comp` declares. The filter
  is tile-independent (checked by the contract test against the source's own tile
  algorithm), so the result should not depend on the block shape; only 16x16 has been
  captured, so the claim that the block shape does not matter rests on the source
  transcription, not on a second capture.
- [U] Only the LDR variant is implemented. The HDR branch (`#ifndef LDR` in the source)
  needs `p.expo` - which the launch reads from the 1x1 f16 texture at argument 48 - and no
  HDR frame has been captured.
- [U] Only frame 1 was compared. Frames 2-3 exist in the same capture; the stage is stateless
  apart from its inputs, so no temporal behaviour is expected, but this is untested.

## Next

- B4.3 `dec0`: the matrix stage already matches the published step chain (EXP-010); what
  remains is the four-surface epilogue.
- The same fixture path serves `dec0`/`enc0`/`post`; only the per-stage binding map and a
  shader are missing.

## Files

- `src/native/m_image_stage.cpp`, `src/native/image_copy_cs.hlsl`,
  `src/native/downsample_cs.hlsl`, `src/native/CMakeLists.txt`
- `scripts/m_stage_fixture.py`
- `tests/test_downsample_contract.py`, `tests/test_stage_fixture.py`
- Fixtures, native outputs and JSON results stay in the temporary workspace.

## References

- [EXP-011](EXP-011-m05-image-capture.md) (the captured operands, edges and independent
  oracles this stage is validated against)
- [EXP-010](EXP-010-m05-native-fp16-stage.md) (the f32->f16 conversion finding and the
  first native stage)
- [stage graph](../findings/2026-10-29-m05-stage-graph.md) (`downsample`'s bindings)
- `references/sources/d4r/engine/down_m.comp`, `output_store.glsl`, `test_down_m.py`
  (d4r `05e19d4`)
