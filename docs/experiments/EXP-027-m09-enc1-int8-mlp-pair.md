# EXP-027 — Complete enc1's native INT8 MLP pair

**Milestone:** M09-F · **Status:** PASS for the enc1 MLP pair; M09 IN PROGRESS.
**Baseline:** `8c5a373` — committed M09-E, 314 tests. Nothing pushed.
**Target:** RX 7700 XT / gfx1101, Windows/D3D12, SDK DXC, RGA 2.15.

## Objective and acceptance

Complete the real enc1 MLP pair, not another detached projection probe. Reuse
the validated INT8 up projection and convert its down projection while retaining
GELU/E4M3, the norm2 residual, and all unrelated operations. Require independent
CPU integer/conversion checks, original FP16 split equivalence, native signed
dot4 tied to executed shader hashes, complete writes, finite deterministic
stage/chain/frame output, and unchanged previous control hashes.

## Source-grounded down contract

[S] The up/bias/activation and residual boundaries are defined in
[EXP-026](EXP-026-m09-enc1-int8-mlp-up.md). The down path comes from
`src/native/swin_layer_cs.hlsl::Mlp`,
`scripts/m_layer_oracle.py::natural_matrices`, and pinned d4r
`05e19d402781083118d106f2a21d0cf2bc43f744`,
`kernels/tools/swin_model.py::{layout,woff_table,swin_block}`.

* **M=136640, K=256, N=64**. Eight physical 32×64 W2 chunks form one logical
  256×64 down matrix in natural channel order.
* Input G is the **unchanged `GeluQ`** of the GPU INT8 up output. It is
  E4M3-representable f16, `[M,256]`, with byte offset `2*(256*row+k)`.
* Original exported W2 f32 element index:
  `pOffW2+(k//32)*2048+(k%32)*64+c`.
* Captured W2 chunk bases are
  **15808, 20096, 24256, 28416, 32576, 36736, 40896, 45056**.
  Let `n=gp(c)` (EXP-026), `j=k%32`; the raw E4M3 byte offset is:
  `base[k//32]+512*(n//16)+64*(n%8)+16*((j%16)//4)`
  `+8*((n//8)%2)+4*((j%32)//16)+j%4`.
* Offline INT8 W2 codes are `[64,256]`, byte offset `256*c+k`; four signed
  bytes per R32_UINT word. Scale byte offsets: `4*row` and `4*c`.
* There is **no down bias**. The original norm2 `x2=Add16(x1,b2)` remains f16
  `[M,64]`, byte offset `2*(64*row+c)`. It is not quantized.
* Output remains the original `x2b` resource, f16 `[M,64]` at the same byte
  offsets. Original Store/Merge consume it unchanged.

[M] Independent scalar offsets match the pinned table, nonuniform synthetic
address identities, and every real captured W2 value; each chunk has 2,048
unique addresses. The up contract is revalidated before accepting the pair.

## Arithmetic and integration

[S][M] Both projections reuse the established symmetric INT8 contract: f32
per-row activation inverse/product, half-even codes clamped ±127, f32 row
scale, and offline per-output-column weight scales/half-even codes. Zero
activation rows substitute max=1; zero weight columns use scale=1.

The down projection performs:

```text
acc[row,c] = exact INT32 sum over all 256 signed products
product = f32(f32(f32(acc)*scaleA[row])*scaleW[c])
y = f32(product + f32(x2[row,c]))
x2b[row,c] = f16_RNE(y)
```

The whole-K bound is **256×127²=4,129,024**, below both 2²⁴ and 2³¹; the
integer-to-f32 conversion is lossless. Measured maximum absolute accumulator:
**26,614**.

**Numerical change is explicit:** the original down path adds each k32 chunk
to the residual with a separate f16 rounding. INT8 uses one whole-K reduction,
the established f32 rescale, one f32 residual addition and one f16 RNE store.
It does not preserve those eight original accumulation-rounding points. The
norm2 residual itself, its producer, scale and floating-point representation
remain unchanged. This is a conversion contract, not a new FP16 investigation.

One existing D3D12 graph now executes:

```text
norm2 → up quantize → INT8 up+bias → original GELU/E4M3
      → down quantize → INT8 down+residual → original Store/Merge
      → remaining nine unchanged FP16 layers → original image stages → frame
```

`swin_mlp_gelu_cs.hlsl::Gelu` calls the original `GeluQ` directly. The two matrix
shaders have no floating-point matrix multiplication fallback. Activations,
GELU and residual stay GPU-resident; only weight conversion is offline.
Readbacks occur after consumers execute. No new executor/framework or backend.

* New mode: `--mlp-mode int8-pair --enc1-mlp DIR`.
* FP16 remains default; `int8-up` is preserved. Q/dec0 combinations remain
  rejected. Requested mode must match fixture metadata.
* The chain validates exact MLP stage **and shader** identities, rejecting an
  FP16-down fallback even if a fixture claims `int8-pair`.
* Chain: **98 dispatches / 161 resources** versus FP16 94/146 and up-only 96/154.
* Added G/codes/scales: 69,959,680 / 34,979,840 / 546,560 B.
  Down integer/f32 diagnostics are 34,979,840 B each; output reuses x2b.
* All dispatches are 1D and bounded: GELU/down quantizer 2135 groups, down
  matrix 8540 groups, 64 threads each.

## Independent validation and handoff

[M] `scripts/m09_mlp_pair.py` first runs the complete EXP-026 up gate, then:

1. Validates the K=256 encoder/down kernel on nonuniform synthetic signed
   values, ties, zero rows/columns, signed extrema and nonzero residuals under
   two different sentinel seeds.
2. Requires every materialized GELU value to equal the independent pinned CPU
   cubic/E4M3 reference bit for bit.
3. Requires **all 34,979,840 down codes**, scales, weight codes/scales,
   **8,744,960 INT32 accumulators**, f32 residual-added values and f16 outputs
   to match the independent CPU oracle bit for bit.
4. Checks 48 independent scalar row encodings/dots/epilogues and every weight
   column with the separate scalar encoder.
5. Runs a validation-only original FP16 down shader on the same G and residual.
   It agrees bit for bit with the CPU eight-k32 oracle, and its x2b, enc1 output
   and merge reproduce the previous **up-only original fused remainder** exactly.
6. Ties all checked stage diagnostics/surfaces to the actual executed pair
   frame, and verifies every non-MLP stage's shader, parameters and bindings
   against the FP16 graph.

**Results:** stage gate 39 checks PASS; integration gate 87 checks PASS.
The inherited up gate also passes. Both matrices' arithmetic is bit-exact.
There are zero non-finite/saturated up, G, x2b, enc1, merge, dec1 or frame values.
Stage outputs and every exported chain field/diagnostic are deterministic under
different sentinel seeds. All **23 B6 boundaries** repeat byte for byte.
Final pair stage/frame network logs contain **zero D3D12 debug messages**.

Down activation scale range: 0.0551181100–0.0629921257; 136,640 positive
INT8 endpoints, zero negative endpoints, 7,454,868 zero codes. These scale
extrema are not f16 saturation; all encodings remain in range.

## Numerical effects

[M] Bias is hybrid minus reference. The same-input down comparison includes
the residual in both outputs, isolating the new down conversion from up error.

| Boundary | Relative RMS vs FP16 | Max abs | Signed bias |
|---|---:|---:|---:|
| Up projection (unchanged) | 1.00019% | 0.14355469 | -0.000899925 |
| Down+residual, identical G/x2 | **2.79560%** | 0.18969727 | +0.001399166 |
| Pair enc1 output | **5.59811%** | 0.5 | -0.000703217 |
| Pair enc1 merge | **7.02243%** | 0.5 | -0.002301009 |
| Pair chained dec1 | **13.75546%** | 3.0625 | -0.005160854 |

Decoded E4M3 field comparisons separate existing reconstruction error:

| Against canonical captured field | FP16 relative RMS | Pair relative RMS |
|---|---:|---:|
| enc1 output | 1.24000% | 5.64828% |
| enc1 merge | 1.94231% | 7.03799% |
| Chained dec1 | 12.55491% | 14.12504% |

Pair-vs-up-only output/merge relative RMS is 4.16970% / 5.72824%, respectively.
The added down conversion is therefore not hidden inside pre-existing FP16 error.

## Full-frame measurements

[M] Existing raw RGB f16/peak-1 PSNR and windowed luma SSIM; alpha unchanged.
Fresh double-run controls retain all prior hashes and metrics.

| Mode vs canonical | PSNR dB | Windowed SSIM |
|---|---:|---:|
| FP16 | 47.051389 | 0.999258964 |
| Q-only INT8 | 45.421826 | 0.998961386 |
| dec0-only INT8 | 47.024874 | 0.999255693 |
| MLP up-only INT8 | 45.920786 | 0.999078068 |
| **MLP pair INT8** | **45.681534** | **0.999018153** |

| Pair against | PSNR dB | RMS | Mean abs | P99 abs | Max abs | Signed bias |
|---|---:|---:|---:|---:|---:|---:|
| Canonical | 45.681534 | 0.00519904 | 0.00122856 | 0.0263672 | 0.209473 | +0.00012260 |
| FP16 | 45.775350 | 0.00514319 | 0.00121080 | 0.0261230 | 0.139404 | +0.00003446 |
| Q-only | 44.798483 | 0.00575540 | 0.00132288 | 0.0292969 | 0.176758 | -0.00001392 |
| dec0-only | 45.640664 | 0.00522356 | 0.00122557 | 0.0266113 | 0.142334 | +0.00005124 |
| Up-only | 45.149383 | 0.00552753 | 0.00124538 | 0.0275879 | 0.143066 | +0.00008143 |

Direct pair-vs-FP16 SSIM: **0.999043843**. The canonical PSNR drop from up-only
is 0.239252 dB; it is not the direct pair-vs-up-only PSNR.
Weighted spatial pair-vs-FP16 absolute error: border/interior means
0.000749425 / 0.000962162; 16×16 block means 0.000173110–0.002826824;
worst block (y,x) coordinates (2,4), (16,7), (48,7), (32,7), (24,7).
Weighted spatial statistics differ from raw RGB means in definition.

## ISA and regression evidence

[M] Both executed matrix shaders inspected with RGA DX12 gfx1101 contain
**16 static `v_dot4_i32_i8` instructions** and zero FP16 matrix mul/FMA/MAC/WMMA
instructions. The loop reduction is K=64 for up and K=256 for down; static
instruction counts are not dynamic counts. Stage, frame and inspected shader
hashes match; this is lowering evidence, not a runtime instruction trace.

| Artifact | SHA-256 |
|---|---|
| New INT8 down shader | `115764abfb6531673d3db2f21fff4eca005b13b6cec76e273a81fb87014d7f75` |
| Original-GELU materializer | `fb16bc3a49819bbefd12627f64f06abe89d6168cc400048d9dbd4535f6ada2bc` |
| Validation-only FP16 down probe | `592deae9fb6c0ce0505bbc66638235d3112c653c594db2e6a25123ffa5256066` |
| Pair frame | `cb4d00b5608d32ccc72f4fa9fcd8f1a0ab19e6c728fd4dfb62b2bd7e2a7784aa` |

The up shader, network executor and **all earlier shader hashes remain unchanged**
from EXP-026. FP16, Q-only, dec0-only and up-only frame hashes also remain unchanged.
321 tests pass, preserving all 314 M09-E tests. Native build and whitespace
checks pass. B4 remains PASS (99.9369%, max 7 ULP); B5 fails only its documented
enc1.merge thresholds (89.8379%, relative RMS 0.02111; expected exit 1).

The first new Q control run stopped with Windows error 112 (disk full) while
copying image-control fixtures. It did not produce an accepted report. Removed
9.13 GiB of redundant `.bin` copies from explicitly named provisional M09-E
roots, retaining their JSON/log/graph failure records and all **verified** E
evidence/canonical captures. The resumed Q control and all subsequent pair
checks pass. Those provisional tensor copies are no longer retained.

## Reproduction and evidence roots

PowerShell; fresh output directories; T is `%LOCALAPPDATA%\Temp\opencode`:

```powershell
cmake --build build/native --config Release
python -m pytest tests -q
$env:M_STAGE_DEBUG = '1'
python scripts/m09_mlp_pair.py --source "$L/enc1" --capture $ENC1_CAPTURE --out "$T/pair-stage"
# Run existing FP16, Q-only, dec0-only and up-only controls with m_b6_frame.py.
python scripts/m_b6_frame.py --out "$T/pair-frame" --mlp-mode int8-pair `
    --enc1-mlp "$T/pair-stage/pair" `
    --fp16-frame "$T/fp16/downsample-native/results/out-out.bin" `
    --enc1q-frame "$T/q/downsample-native/results/out-out.bin" `
    --dec0-frame "$T/dec0/downsample-native/results/out-out.bin" `
    --mlpup-frame "$T/up/downsample-native/results/out-out.bin"
python scripts/m09_mlp_pair.py --out "$T/pair-stage" --frame-run "$T/pair-frame" `
    --fp16-run "$T/fp16" --enc1q-run "$T/q" --dec0-run "$T/dec0" --up-run "$T/up" --rga $RGA
```

Final local roots, all outside Git:

* `m09f-pair-stage-20261030/{validation,integration}.json`: CPU/layout/synthetic
  checks, real frame handoffs, both executed ISA files, split probes and debug logs.
* `m09f-pair-frame-20261030/b6.json`: pair frame, 23 deterministic boundaries,
  canonical and all direct comparisons; reconstructed/reference/difference PNGs
  use prefixes `frame-native` and `hybrid-vs-{fp16,enc1q,dec0,mlpup}`.
* `m09f-{fp16,enc1q-resumed,dec0,up}-20261030/b6.json`: controls.
* `m09f-conformance-20261030.json`: unchanged B4/B5 using the existing
  `m09e-translated-refresh-20261030` reference-equivalence record.

PNG display mapping is unchanged; all measurements use raw f16. Captures,
weights, tensors, images and generated shader binaries are not committed.

## Limits and next priority

The **enc1 MLP pair is complete**, not all MLPs or the full INT8 network. Only
one reset-frame/content/resolution/adapter is validated; no temporal or performance
claim. Floating-point GELU/E4M3, residuals and normalization are explicit exceptions;
the other nine layers remain FP16 in this pair-only mode. B6's existing file
boundaries between major image/neural stages remain as documented in EXP-025;
the MLP pair and neural chain are the GPU-resident scope. General static fidelity
remains INCONCLUSIVE. B5 is intentionally not repaired by this work.

The next priority is **cumulative full-network INT8 matrix coverage toward M10**:
track every matrix family and layer in the actual graph, eliminate unconverted
matrix fallbacks systematically, and retain independent CPU/ISA/handoff gates.
Do not treat four independent experimental modes as cumulative coverage. The
authoritative family coverage gaps and execution priorities are in M09 of
`docs/MILESTONES.md`; no new standalone experiment track or M10 completion is implied.
