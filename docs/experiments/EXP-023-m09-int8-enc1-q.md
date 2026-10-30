# EXP-023 — M09: enc1's Q projection in native INT8

**Milestone:** M09 (I1–I6) · **Status:** PASS on the declared gates for this one
projection; M09 C05 (representative layers) stays open.

## 1. Question

Can one matrix projection inside the working Preset M graph be replaced by native
INT8 arithmetic, with the integer arithmetic validated independently, the rest of the
layer untouched, and the numerical cost measured through the layer, the ten-layer chain
and the frame?

## 2. Contract (I1), established before the replacement

| Item | Established | Source |
|---|---|---|
| Projection | enc1 `q`: M = 136 640 tokens, K = 64, N = 64 (2 heads × 32) | `fixture.json` geometry, `swin_layer_cs.hlsl` `Matmul` |
| `q` dispatch | 8 540 groups × 64 threads, 16 output columns per thread tile | graph manifest |
| Operand `h1` | f16 buffer, row-major `tok*64 + c`, 17 489 920 B; **124 distinct values, all exactly E4M3-representable** (min −1.625, max 2.75; per-row maximum 2.25–2.75) | dumped from the FP16 run (`dbg-h1.bin`) and checked against the E4M3 table |
| Weight | `weights[pOffQ]`, f32, row-major `k*64 + n`, values are captured E4M3 codes widened to f32 (`rel_frobenius` 0.0 against the E4M3 decode) | `m_layer_fixture.WeightPack`, `m_layer_oracle.natural_matrices` |
| FP16 arithmetic | per 32-product step: exact double sum, then **one f16 RNE rounding** of `sum + acc`; `pC0Mode = 0` so the accumulator starts at zero; output f16 at `tok*64 + n` | `swin_layer_cs.hlsl` `Matmul` |
| Output resource | `q`, 17 489 920 B of f16 pairs; attention reads it as f16 and immediately re-encodes it with `Q8` (E4M3), so the downstream resolution is E4M3 | `swin_layer_cs.hlsl` `Attention` |
| Neighbours | `prepare`, `norm1`, `v`, `attention`, `wo`, `norm2`, `mlp`, `store`, `merge` are untouched and their shaders are unchanged (hashes in `m09-enc1q.json`) | graph diff |

[M] `h1` is therefore a *decoded* FP8 field, not codes and not a raw f16 field: the
INT8 encoder re-quantises values that were already E4M3. That is what makes the
activation error small (124 distinct values) but also why it is not zero.

## 3. Encoding and arithmetic (I2/I3)

```
activation   inv   = f32(127) / f32(max|a|)        per row, correctly rounded
             ratio = f32( f32(a) * inv )           f32 product
             code  = round_half_even(ratio), clamped to ±127
             scale = f32( f32(max|a|) * f32(1/127) )
weight       m_int8.quantize(W, 'per_output_channel'): code = clamp(rint(W/scale), ±127),
             scale[n] = f32(max_k |W[k][n]| / 127)
product      acc = Σ_k code_a[k] · code_w[n][k]        exact int32 (dot4add_i8packed)
output       y = f32( f32(acc · scale_a) · scale_w )   one f16 RNE conversion
```

Zero rows are encodable: `max|a| = 0` falls back to 1.0, so the row's codes are 0 and
its scale is 1/127 — the same rule as `m_int8.quantize_rows`. `|acc| <= 64·127·127 =
1 032 256`, so int32 is exact and `(float)acc` is lossless.

### 3.1 Three encoders that failed the bar (negative evidence)

The CPU reference and the kernel must quantise **identically** for a code-level
comparison to mean anything. Three earlier forms did not, and the mechanism each time
was measured rather than assumed:

| Form | Codes differing from the CPU reference (of 8 744 960) | Mechanism |
|---|---:|---|
| f32 `127/mx` (v_rcp_f32 + v_mul_f32) + explicit compare/select RNE | 171 092 (1.96 %) | the f32 quotient is not the correctly rounded one: `1.125 · f32(127/2.25)` is exactly 63.5 on the device and 63.499998 on the exact value |
| the same, with an explicit residual correction so `inv` was bit-identical to the CPU's | 175 474 | **the compare/select RNE was miscompiled**: for a ratio of exactly 63.5 the device produced 63 (odd side) where half-to-even requires 64. Dumping the ratio proved the arithmetic was right and the rounding was not |
| f64 quotient (`precise double`) + f32 select | 211 369 | the device's f64 divider is one ulp low for some divisors (`mx = 2.25` gives an inverse whose product is 63.49999999999999) |
| **f32 quotient + `round()` (DXIL Round = half-to-even) + f32 product** | **0** | the rule is left to the instruction that defines it |

The final form is the one in `swin_q8_quant_cs.hlsl`. Its ISA also shows the f32
division as `v_rcp_f32 + v_mul_f32`; on this operand the f32 quotient happened to be
correctly rounded for every one of the 136 640 rows (verified by dumping `inv`), and the
code comparison over 8.7 M values confirms it.

## 4. CPU oracle validation (I2)

`scripts/m09_enc1_q.py` implements the encoder in the same f32 arithmetic and the
product with an exact int64 reduction. Against the GPU, on the real captured operands:

| Check | Result |
|---|---|
| activation codes, GPU vs CPU | **8 744 960 / 8 744 960 identical**, max delta 0 |
| activation scales, GPU vs CPU | identical (f32 bit patterns) |
| weight codes / scales, graph vs CPU | 4 096 / 4 096 identical; scales identical |
| integer product: GPU codes → CPU GEMM vs GPU `q` | **bit-identical** (max abs 0.0) |
| integer product: CPU codes → CPU GEMM vs GPU `q` | **bit-identical** |
| independent scalar dot probes | 8 / 8 match the int64 matrix product |
| FP16 path: GPU `q` vs CPU k32 chain (`me.mma_chain`) | bit-identical (max abs 0.0) |
| accumulator | `|acc|max = 32 722`, int32-safe |
| encoding residual | max 0.01082683 vs half-step 0.01082677, i.e. ≤ half a step + 5·10⁻⁸ of f32 rounding |
| endpoint population | 185 780 codes at +127 (2.12 %), none at −127, 134 544 zeros |

[M] The encoder's guarantee is "nearest code, ties to even": the measured residual is
half a step up to f32 rounding, so no value is more than one code from its nearest.

## 5. ISA (I3)

[M] `Radeon GPU Analyzer 2.15`, `-s dx12 -c gfx1101`, on `build/native/swin_q8_matmul.cso`
(SHA-256 `edecf9b5c6d8951f…`, the file the graph binds and the run records):

```
v_dot4_i32_i8 v34, v35, v36, v34    // 00000000037C: CC164022 7C8A4923
v_dot4_i32_i8 v33, v35, v37, v33    // 000000000388: CC164021 7C864B23
... 16 in total
```

16 instructions for K = 64 (four codes per instruction) across 16 output columns;
**zero** `v_pk_mul_f16` / `v_mul_f16` / `v_fma_f16` / `v_mac_f16` / `v_wmma_f16`, so
nothing fell back to FP16 matrix arithmetic. The quantiser's ISA is separate
(`swin_q8_quant.cso`, SHA-256 `316fcf87d5ae3788…`) and contains the reciprocal,
the clamp and the RNE, no dot4.

## 6. Integration (I4)

One mode switch, default off:

- `scripts/m_layer_fixture.py export --q-mode int8-q` writes the same graph as `fp16`
  except that enc1's single `q` dispatch becomes two: `q-quant` (encode `h1`) and `q`
  (`swin_q8_matmul.cso`), plus four resources — `q8a` (8 744 960 B), `q8as` (546 560 B),
  `q8w` (4 096 B), `q8ws` (256 B). The `q` resource, its size, its f16 layout and every
  consumer are unchanged.
- `scripts/m_layer_suite.py build_chain --q-mode int8-q --enc1-int8 DIR` takes enc1's
  graph from the INT8 fixture and refuses to build if enc1's stages are anything other
  than `enc1.q-quant` + `enc1.q`.
- `scripts/m_b6_frame.py --q-mode int8-q --enc1-int8 DIR` runs the frame with the
  hybrid graph; the execution gate, boundary hashes and provenance are unchanged.
- The activation never leaves the GPU: `q-quant` reads `h1` and writes codes and scales
  on device. The weight encoding is precomputed from the captured matrix
  (source, codes and scales hashed in `fixture.json`).
- Dispatch counts: enc1 10 → 11, ten-layer chain 94 → 95.

## 7. Measurements

### 7.1 The projection (identical operands, identical `h1`)

| Comparison | identical f16 | rel RMS | RMS | max abs | max / peak | signed mean |
|---|---:|---:|---:|---:|---:|---:|
| INT8 `q` vs FP16 `q` | 1.97 % | **0.014956** | 0.005500 | 0.017578 | 1.23 % | −2.0e-4 |
| CPU INT8 vs FP16 `q` | 1.97 % | 0.014956 | 0.005500 | 0.017578 | 1.23 % | −2.0e-4 |

`q` peak is 1.4258; the largest INT8 deviation is 0.0176. Only 1.97 % of f16 values are
identical because a 1.5 % relative perturbation is many f16 ulps — the metric that
matters is the relative RMS, not the bit identity.

### 7.2 enc1 (hybrid vs FP16 Sette, and both against the capture)

| Field | hybrid vs FP16 | FP16 vs capture | hybrid vs capture |
|---|---:|---:|---:|
| attention output `o` | 98.00 % identical f16, rel RMS 0.009437, max 0.25 | — | — |
| residual `x1` | 40.85 % identical f16, rel RMS 0.012497, max 0.102 | — | — |
| output field `out` | 66.59 % codes, decoded rel RMS **4.95 %**, mean code delta 1.98 | 96.92 % codes, 1.24 % | 66.46 % codes, 4.94 % |
| merge field | 47.12 % codes, decoded rel RMS **6.11 %**, mean code delta 2.94 | 91.29 % codes, 1.94 % | 47.00 % codes, 6.12 % |

[M] The existing reconstruction error (FP16 vs capture, 1.2 % / 1.9 %) and the INT8
quantisation effect (4.9 % / 6.1 %) are reported separately; the hybrid's error against
the capture is the sum of the two, not a new reference disagreement.

### 7.3 The ten-layer chain

| Path | dec1 codes vs capture | decoded rel RMS | max abs | signed mean |
|---|---:|---:|---:|---:|
| FP16 Sette (unchanged) | 21.33 % | 0.0 by construction | — | — |
| hybrid (INT8 Q), vs the FP16 chain | **19.71 %** identical, mean code delta 7.90 | **0.13551** | 3.25 | −1.5e-3 |
| hybrid, vs capture | 19.73 % | — | — | — |

Both chains are deterministic over two runs and contain zero non-finite codes. The
quantisation effect at the chain's end is 13.6 % relative RMS — an order of magnitude
larger than the 1.5 % at the projection, which is the same amplification the translated
reference shows (EXP-019): ten e4m3 re-quantisation boundaries turn a small operand
difference into near-random code agreement.

### 7.4 The frame (B6, `int8-q` graph)

| Path | PSNR (peak 1) | windowed SSIM | mean abs | p99 abs | max abs | signed bias |
|---|---:|---:|---:|---:|---:|---:|
| FP16 native (regression control) | 47.051 dB | 0.999259 | 0.001086 | 0.02295 | 0.1782 | +8.8e-5 |
| **hybrid INT8 Q** | **45.422 dB** | **0.998961** | 0.001268 | 0.02759 | 0.2012 | +1.4e-4 |
| image-only control (unchanged) | 46.726 dB | 0.999208 | 0.001126 | 0.02393 | 0.1416 | +1.8e-5 |
| translated chain (unchanged) | 47.080 dB | 0.999268 | 0.001075 | 0.02246 | 0.1543 | +1.4e-4 |

[M] Replacing one projection costs **1.63 dB** and 0.0003 SSIM: still 45.4 dB against
the capture, above the image-only control's own error level, with zero NaN/Inf, zero
saturation, ten bit-identical stage boundaries across two complete runs. Frame SHA-256
`0eb0185ea7e21f20…` (FP16: `dc97c4cbe9934611…`, reference `1a7934b877995f98…`).
Exports: `frame-int8q/frame-native-{reference,reconstructed,difference}.png`.

## 8. Verdict

All six declared gates hold for this projection: the contract is established (I1), the
CPU oracle is deterministic and independently validated (I2), the GPU matches it
**bit for bit** (I3), the ISA contains native INT8 dot products and no FP16 matrix
fallback (I3), the network runs with exactly one replaced dispatch and every other
operator byte-identical (I4), and the hybrid chain and frame are deterministic, finite
and unsaturated (I5/I6).

The numerical cost is real and measured: 1.5 % relative RMS at the projection, 5–6 % at
enc1's outputs, 13.6 % at dec1, 1.63 dB at the frame. None of it is attributed to the
existing FP16 reconstruction error, which is reported alongside.

This does **not** complete M09: C05 needs representative operators and layer-level
behaviour, and the other 359 matrices are untouched. B5's `enc1.merge` exception is
unchanged and was not reopened; its gate still fails at 89.8379 % / 0.02111.

## 9. Determinism

[M] Every path was run twice: the enc1 fixture (both modes), the ten-layer chain (both
modes) and the frame (both modes, ten boundaries). All repeats are byte-identical.

## 10. Limitations

- **One projection, one layer, one content, one adapter.** No claim about other
  matrices, other layers, other content or NVIDIA hardware.
- **The encoder's f32 quotient** relies on the device's `v_rcp_f32 + v_mul_f32` agreeing
  with a correctly rounded f32 division; it did for all 136 640 rows on this operand,
  which is measured but not guaranteed by the ISA.
- **The CPU oracle is not an oracle for the encoder's correctness**, only for its
  reproducibility: both implement the same contract, and only the residual-bound test
  (≤ half a step) and the endpoint statistics check the encoding itself.
- **Weights are quantised offline** from the captured f32 matrix; a runtime path would
  have to convert the E4M3 blob itself.
- **No clipping, no asymmetric or blockwise scales, no calibration**: the encoding is
  the M07 baseline, so the measured error is a floor for this scheme, not a tuned
  result.
- **The frame is the LDR/reset variant** with captured colour and zero motion, as in
  EXP-020.
- Rate: the INT8 path adds one dispatch and four buffers; no performance claim is made
  (M14 owns that).

## 11. Reproduction

```powershell
cmake --build build/native --config Release
python scripts/m09_enc1_q.py run --out DIR --layers <ten-layer fixture dir> `
    --capture <capture-1080> --frame `
    --translated-dec1 <translated dec1 replay dir>
```

Artifacts: `DIR/m09-enc1q.json` (all measurements and hashes), `DIR/enc1-fp16`,
`DIR/enc1-int8` (fixtures with `dbg-*.bin` internals), `DIR/chain-{fp16,int8-q}`,
`DIR/frame-int8q/b6.json` and its PNGs. Everything stays under `%TEMP%`; no captured
bytes are tracked.
