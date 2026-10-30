# EXP-024 — Native INT8 dec0 expansion from physical E4M3 layouts

**Milestone:** M09-C · **Status:** PASS for isolated C1–C4; C5/C6 deferred.
**Baseline:** `41e460c` (282 tests). **Environment:** Windows/D3D12, RX 7700 XT,
gfx1101, Windows SDK DXC, RGA 2.15. No performance or full-stage fidelity claim.

## Question and acceptance criteria

Can the second operand layout be handled without CPU activation conversion?
Require independently checked addresses, a deterministic CPU INT8 reference,
GPU/CPU code/scale/integer/epilogue/output comparisons, native dot4 ISA tied to the
executed shader hash, and measured error on identical native FP16 operands.
Only after those pass may stage integration be considered. FP16 contracts and
B4/B5 thresholds must not change.

## C1 — Physical contract

[S] Sources: `src/native/dec0_cs.hlsl` (`TokenValue`, `WeightByte`, `BiasValue`,
`Column`, `main`), `scripts/m_weights.py` (`front_end_layout`, `layout_offsets`),
and the independent PTX derivation preserved in [EXP-014](EXP-014-m05-dec0-mapping.md).
No new binary reconstruction or change to pinned sources was needed.

* M = 480×272 = **130,560** tokens; K = **64**.
* Logical N = **80**, selected from **128 physical columns**, not the first 80.
  Logical column `n=20*s+i` maps to physical column `32*s+i`, for s=0..3,
  i=0..19. No channel-pair (`gp`) permutation is applied.
* Token byte offset, with row `r=480*ty+tx`:

  `offset(r,c) = floor(c/32)*4,177,920 + r*32 + c%32`.

  Two 32-channel planes, 15,360-byte spatial row stride, 4,177,920-byte plane
  stride, 8,355,840-byte field. Codes are E4M3, **not** f16 or signed INT8.
* Blocked/interleaved weight byte offset (k=0..63, physical n=0..127):

  ```text
  kb   = 2*floor(n/16) + floor(k/32)
  lane = 4*(n%8) + floor((k%16)/4)
  g    = floor((n%16)/8)
  j    = k%4 + 4*floor((k%32)/16)
  offset = 512*kb + 16*lane + 8*g + j
  ```

  The 16-column blocking is lane-interleaved, not plain row-major 16×16 tiles.
  The full 64×128 table bijects bytes [0,8192). Bias is f16 at byte
  `8192+2*physical_column`; the blob totals 8,448 bytes.
* Existing `Column`: f16 bias seeds the accumulator, exact double sum for each
  32 products, one f16 RNE rounding per step. The isolated FP16 entry calls this
  **existing function verbatim**, rather than constructing another FP16 model.
* Expansion output is row-major f16 `[130560][80]` before dec0's existing E4M3
  round-trip and nonlinear epilogue. Packed address is `row*80+n`, two f16 per
  uint. Sub-position s targets pixel `(2*tx+(s&1),2*ty+(s>>1))`. Only 17 of the
  20 columns influence surfaces, but this probe preserves all 20.

[M] Eleven new CPU tests include full weight-offset bijection/table agreement,
two nonuniform synthetic pattern reconstructions, first/middle/last tokens,
479→480 row boundaries, channels 15→16 and 31→32, and live-column/bias gaps.
Bulk CPU token decoding uses reshape/transpose; weight decoding uses the separate
existing layout table. Synthetic physical buffers are filled by scalar addresses,
so transposition or wrong plane strides do not agree accidentally.

## C2/C3 — Arithmetic and implementation

[S] Reuses EXP-023's encoder contract and `swin_q8_common.hlsli` helpers:

```text
max = max(abs(decoded E4M3 row)); zero row substitutes max=1
inv = f32(127)/f32(max)
ratio = f32(f32(value)*inv)
activation_code = clamp(round_half_even(ratio), -127,127)
activation_scale = f32(max * f32(1/127))

weight_scale[n] = f32(max_k(abs(W[k,n]))/127), or 1 for a zero column
weight_code[k,n] = clamp(round_half_even(f64(W[k,n])/f64(weight_scale[n])), -127,127)

acc = sum_k activation_code[k]*weight_code[k,n]       exact int32
y = f32(f32(f32(acc)*activation_scale)*weight_scale)
y = f32(y + bias[n])
out = f16_RNE(y)
```

The GPU inverse uses the same candidate plus double residual correction as the
actual enc1 shader source; `round()` supplies DXIL half-to-even. Weight encoding
calls `m_int8.quantize(...,'per_output_channel')`. No alternate scale scheme,
clipping, calibration, or per-K rescaling was introduced.

`dec0_q8_layout_cs.hlsl:Quantize` reads the physical plane-major bytes through
the established `TokenValue`/E4M3 decoder, writing packed row-major INT8 codes and
f32 scales. Activations stay GPU-resident between encoder and GEMM. Offline
weight decoding/quantization into `[80][64]` packed INT8 is explicit and permitted;
the GEMM does **not** claim to decode blocked FP8 weights at runtime.

`dec0_q8_matmul_cs.hlsl` reuses the `dot4add_i8packed` reduction pattern of
`gemm_int8_cs.hlsl`; it adds bias once in f32 and writes separate integer, f32,
and f16 diagnostics. These outputs distinguish implementation error from
quantization error. The test graph uses the existing `m_network_stage` runner.
No new graph framework or host executable was added.

Review caught an oversized initial 1D dispatch (81,600 groups, beyond D3D12's
65,535-per-axis limit). Final probes use **2040×40×1 groups** for the matrix
and **2040×1×1** for encoding, with explicit 2D flattening. A regression test
pins these dimensions. The final compliant run reproduced every preliminary
buffer byte for byte; only the compliant run supplies the final shader/ISA hashes.

### Independent comparisons

[M] Real inputs: the existing EXP-014 dec0 fixture, not the stride-16 sample of
[EXP-021](EXP-021-m09-int8-gemm.md). `report.json` records all source/converted
operand, runner, script, graph, and shader hashes.

| Check | Measured result |
|---|---|
| Activation codes | 8,355,840/8,355,840 identical |
| Activation f32 scales | 130,560/130,560 bit-identical |
| Offline weight encoding | All 5,120 codes and 80 scales checked by separate scalar max/divide/Python-round loops |
| GPU weight code/scale readback | All 5,120 codes and 80 scales bit-identical to CPU encoding |
| Scalar activation encoder | Six boundary/middle rows × 64 channels, codes/scales agree |
| Integer accumulators | All **10,444,800** GPU outputs equal exact CPU int64 reduction |
| Independent scalar dots | 48/48 agree with both CPU matrix product and GPU integer dump |
| f32 epilogue | All 10,444,800 values bit-identical |
| f16 output | All 10,444,800 values bit-identical |
| Existing FP16 Column vs established CPU k32 oracle | All outputs numerically identical |
| Two position-dependent sentinel seeds | All eight declared buffers byte-identical; CPU comparisons also verify every element |

[M] Accumulator range **[-82,690,132,025]**; worst possible bound
64×127² = 1,032,256 < 2²⁴ < 2³¹, so both integer accumulation and int32→f32 are
exact. Activation scales 0.04330709–0.11023622, weight scales
0.0006151575–0.005905512. Zero rows: 0; 170,110 activation codes at ±127
(expected extrema, not evidence of out-of-range clipping). Zero half overflow,
zero half saturation, zero values exceeding the E4M3 finite range, zero NaN/Inf.
Synthetic tests separately exercise zero rows, zero weight columns, signed ties,
smallest E4M3 magnitudes, ±448, and the worst-case accumulator.

### ISA provenance

[M] RGA DX12 compilation for gfx1101 on the same `.cso` named by the executed
graph reports **12 static `v_dot4_i32_i8` instructions** (partially unrolled loop;
the logical reduction is 16 dot4 operations per output), zero FP16 matrix
multiply/FMA/MAC or WMMA instructions. Example:

```text
v_dot4_i32_i8 v4, v6, v7, v22 // 0000000001B4
v_dot4_i32_i8 v5, v6, v9, v5  // 0000000001BC
```

SHA-256:

| Artifact | Hash |
|---|---|
| Executed/inspected INT8 GEMM | `c7fc7fa6a3aa67c540edaea7a76d0d2c3f5d1646c3b581609e8efa062e405a89` |
| GPU physical-layout encoder | `36a9e300c98e9fc78e3b77c49536b54398c79d6427eb25df21bdab1d2ec58d15` |
| FP16 Column probe | `dc7dc640b2f4d4cc084eaf014c6b567cac2b2a21062013b0e8f9d609876e2004` |
| RGA ISA text | `b0a2cc1e9eda6747a12244e0fafb57eae3f3a6fc0be2013e9539982f9eee1725` |
| Physical token field | `f56912b68bc2f4ca06e626bcc52e5f723c3ff1ea0e68e617a8e629ef51a62c71` |
| Physical weight/bias blob | `e4311d9e936c27d41718318adb70e4d688b1d50ea90ed22580e611313bdf60ef` |

The runner executes DXIL; RGA separately compiles that identical hashed DXIL for
gfx1101. This is static ISA evidence, not a runtime instruction trace. Shader
hashes are checked unchanged across execution and inspection.

## C4 — Differential error

[M] Identical physical inputs and source weights for all three paths:

| Comparison | Relative RMS | Absolute RMS | Max absolute | Signed bias |
|---|---:|---:|---:|---:|
| GPU INT8 vs CPU INT8 (f32 and f16) | 0 | 0 | 0 | 0 |
| INT8 f16 vs native FP16 Column | **0.005258823** | **0.01983132** | **0.21875** | +0.00004234 |
| Quantized exact rescale vs original exact f64 product+bias | 0.005247526 | 0.01978876 | 0.2171958 | +0.00004094 |
| Native FP16 vs original exact f64 product+bias | 0.000249078 | 0.00093929 | 0.01135254 | -0.00000171 |
| INT8 f16 vs quantized exact rescale | 0.000217311 | 0.00082031 | 0.00778794 | -0.00000031 |

All 80 per-channel RMS/max values are retained in JSON; RMS spans
0.00694304–0.05561788. The 0.526% FP16 difference includes quantization **and**
the intentional replacement of two bias-seeded f16 k32 roundings by one integer
reduction/f32 epilogue/f16 store. It is not implementation error: that error is
zero against the independent INT8 contract. The exact-product diagnostics isolate
quantization from the existing FP16 rounding model; no thresholds were relaxed.

## C5/C6 — Integration boundary and limitations

[S] `m_image_stage.cpp` executes one dispatch with texture surfaces. The current
graph runner (`m_network_stage.cpp`) supports buffer graphs, not those typed
output textures. An encoder→GEMM→unchanged image epilogue requires a device-side
buffer/texture orchestration extension; reading this experiment's output back to
Python and uploading it would not satisfy GPU-resident integration.

Following the optional-integration stop rule, **no dec0 stage mode or hybrid frame
was added**. No surface-comparison or image-quality result is inferred from the
isolated GEMM. Existing FP16 defaults, dec0 nonlinear operations, attention,
other operators, and temporal behavior remain untouched. Neither enc1-Q nor dec0
INT8 was combined with the other.

[U] One full captured token field, one device/driver; arbitrary shapes, runtime
blocked-weight conversion, subsequent nonlinear/surface amplification, and dec0
hybrid frame quality remain unvalidated. C02 gains a second proven layout path;
M09 C05 still lacks representative complete INT8 layer behavior. M09 remains
**IN PROGRESS**, its full gate **PENDING**.

## Regressions

[M] `python -m pytest tests -q`: **293 passed** (all 282 existing + 11 new).
Conformance rerun: B4 PASS, 99.9369%, max 7 ULP; B5 still FAIL **only**
`enc1.merge`, 89.8379% / 0.02111. Expected conformance exit code: 1.

Fresh FP16 B6: execution PASS, static fidelity INCONCLUSIVE, 47.051389 dB,
SSIM 0.999258964. Frame SHA-256 remains
`dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b`.
Every pre-existing shader hash matches the prior FP16 B6 manifest.
Fresh enc1-Q CPU/GPU/layer/chain rerun: Q report and both dec1 chain hashes
unchanged from EXP-023 (including bit-exact Q arithmetic). Its frame was not
rerun; no claim of a new enc1-Q frame measurement.

## Reproduction and retained evidence

```powershell
cmake --build build/native --config Release
python -m pytest tests -q
python scripts/m09_dec0_expansion.py --fixture <existing-dec0-fixture> `
    --out <temporary-output> --rga <rga.exe>
```

The fixture supplies `in-tokens.bin`, `in-weights.bin`, and `params.bin` from the
established 960×544 padded render geometry. `--rga` enables the ISA gate in addition
to numerical checks. A failed comparison exits nonzero and is preserved in JSON.
No captures, proprietary weights, tensors, or generated shaders are tracked.

Local evidence root: `%LOCALAPPDATA%\Temp\opencode\`:

* `m09-dec0-final-20261030/`: `report.json`, graph, two seeded result sets, RGA
  ISA, source and converted operand files.
* `m09-dec0-20261030/b4-b5.json`: unchanged conformance results (the other files
  in that directory belong to the preliminary, superseded dispatch geometry).
* `b6-fp16-after-dec0-20261030/b6.json`: independently executed FP16 frame.
* `enc1q-after-dec0-20261030/m09-enc1q.json`: enc1-Q regression run.

**Next independently testable step:** extend orchestration narrowly to let the
unchanged dec0 image epilogue consume a GPU-resident f16 expansion buffer; then
validate all four surfaces, sentinel coverage, and repeats before a dec0-only
hybrid frame. This is not authorization to convert another operator.
