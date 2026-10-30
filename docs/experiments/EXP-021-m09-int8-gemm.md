# EXP-021 — M09: one native INT8 matrix operation on gfx1101

**Milestone:** M09 · **Status:** PASS for the bounded first experiment (native INT8
execution, independent CPU reference, ISA confirmation, measured quantisation error).
No image-quality or performance claim.

## 1. Question

Can gfx1101 execute one of Preset M's matrix operations in native INT8, with a
declared quantisation contract, matching an independent CPU INT8 reference and
differing from the FP16 operator only by the quantisation that contract introduces?

## 2. Acceptance criteria (declared before the run)

1. The compiled pipeline's ISA contains a native INT8 dot-product instruction.
2. The GPU result equals an independent CPU INT8 product computed from the same codes
   and scales, to within f32 epilogue rounding (`<= 1e-5 * peak`), i.e. the integer
   reduction is exact.
3. The GPU INT8 result differs from the FP16 operator on the same operands by no more
   than the arithmetic quantisation-only bound (weight and activation encodings
   applied to the same operands), within 1% relative.
4. Two runs are byte-identical, the output is finite, and the int32 accumulator cannot
   overflow for this K.

## 3. Method

**Target matrix.** `dec0`'s expansion GEMM — the largest single matrix operation in the
capture and the one M07 identified as precision-sensitive but structurally simple:

    Y[8160 x 80] = A[8160 x 64] * W[64 x 80] + bias[80]

`A` is the captured token operand, a deterministic stride-16 sample of the 130,560-token
field (the same sample [EXP-010](EXP-010-m05-native-fp16-stage.md) used); `W` is the
captured E4M3 expansion matrix widened to f16; `bias` is the stage's f16 bias. Because
the same case already drives the FP16 stage, the FP16 comparison is a real operator
output on identical operands, not a re-derivation.

**Quantisation contract** (M07's single-accumulation baseline,
[EXP-007](EXP-007-m07-int8-feasibility.md) / [EXP-009](EXP-009-m07-complete-inventory.md)):

| Side | Encoding | Scale | Notes |
|---|---|---|---|
| Weight | signed INT8, symmetric, no zero point | one per output column | one integer reduction for the whole K |
| Activation | signed INT8, symmetric, no zero point | one per token row | `127/mx` in f32, `mx*(1/127)` in f32 |
| Rescale | f32 epilogue | — | `(float)acc * aScale * wScale + bias` |
| Accumulator | int32 | — | `|sum| <= K*127*127`; K <= 131071 |

**New pieces.** `src/native/gemm_int8_cs.hlsl` (the kernel; it builds to
`build/native/gemm_int8_cs.cso` at `cs_6_4`, because `dot4add_i8packed` is SM 6.4,
while the FP16 and reconstruction shaders stay at `cs_6_2` untouched),
`MSTAGE01` case **version 2** in `src/native/m_stage_probe.cpp` (INT8 operands,
per-row and per-column scales, f32 bias), `scripts/m09_int8_gemm.py` (quantise,
execute, compare).

```powershell
python scripts/m_stage_case.py export <capture> --matrix dec0.expansion `
    --out case-f16.bin --rows 8160 --stride 16
python scripts/m09_int8_gemm.py run --out DIR --f16-case case-f16.bin
```

## 4. Results

### 4.1 ISA

[M] `Radeon GPU Analyzer 2.15.0.30`, `-s dx12 -c gfx1101 --cs-blob gemm_int8_cs.cso`,
disassembles the compute pipeline and emits:

```
v_dot4_i32_i8 v5, v7, v6, v5        // 0000000001C8: CC164005 7C160D07
```

one instruction for four signed 8-bit products and a 32-bit accumulate — the native
INT8 arithmetic the experiment requires. [M] The dispatch of the same `.cso` through
D3D12 also succeeds (`dispatch M=8160 K=64 N=80 groups=10200 -> 2611200 bytes`).

### 4.2 Arithmetic

| Comparison | rel Frobenius | RMS | max abs | identical f16 |
|---|---:|---:|---:|---:|
| GPU INT8 vs independent CPU INT8 | **3.72e-8** | 5.5e-8 | 9.60e-7 | 99.978% |
| GPU INT8 vs FP16 stage (same operands) | 0.009915 | 0.014635 | 0.3079 | 2.6% |
| GPU INT8 vs exact float64 | 0.009914 | 0.014635 | 0.3088 | 2.7% |
| FP16 stage vs exact float64 | 0.000257 | 0.000380 | 0.00763 | 75.0% |

[M] The integer reduction is exact: the largest GPU/CPU difference is `9.60e-7`
against a `1e-5 * peak = 1.3e-4` threshold, i.e. only the f32 epilogue rounds.
`|sum|max = 1.03e6 < 2^31`, so no overflow is possible at K = 64.

[M] The INT8 deviation from the FP16 operator (0.009915) equals the
quantisation-only bound computed from the same encodings (0.009914) to four digits:
the encodings explain all of it, with no arithmetic defect added by the kernel.

### 4.3 Quantisation cost

| Quantity | Value |
|---|---|
| Weight rel Frobenius / RMS | 0.005784 / 0.000551 |
| Weight distinct codes / at endpoint | 200 / 2.60% |
| Weight storage | 5,440 B (5,120 data + 320 scale) = 1.0625x the E4M3 bytes |
| Activation distinct codes / at endpoint | 201 / 2.55% |
| Activation scale range | 0.0197 – 0.126 |
| Activation-only rel error | 0.007395 |
| Weight-only rel error | 0.006596 |
| Combined (both encodings) | 0.009914 |

The activation term is now the larger of the two, which matches M07's prediction for
this matrix.

## 5. Verdict

**PASS** on all four declared criteria. Native INT8 matrix arithmetic executes on
gfx1101, is confirmed in the ISA, reproduces an independent CPU INT8 product exactly,
and its error against the FP16 operator is fully accounted for by the quantisation
contract. Nothing here says anything about image quality or speed.

## 6. Limitations

- **One sampled matrix**, 8,160 of 130,560 rows (stride 16), K = 64, N = 80. The full
  matrix and the other 359 captured matrices are not run.
- **One scale layout.** Per-output-channel weights with a single integer reduction;
  per-K-block scales (better in M07) need a rescale per block and are not implemented.
- **No clipping.** Max-based scaling only; the 2.6% / 2.6% endpoint population is the
  whole saturation story. Asymmetric quantisation is deferred to M12.
- **The FP16 baseline is `gemm_f16_cs`**, the published f16-step chain, not the actual
  `dec0` stage output; the two agree to 97.9% of f16 values (EXP-010).
- **The kernel is not wired into the network graph.** It is a standalone operator probe
  driven by a case file; layout conversion from the captured plane-major token field
  into `A` happens outside the kernel.
- The ISA claim is offline static analysis of the same `.cso` that executes, in the
  same manner M08 used (driver analyzer report and offline RGA).

## 7. Next

The next bounded step is layout and fusion, not more precision work: run the same
contract on a matrix whose operand is already in the network's plane-major layout
(`enc1`'s Q projection), then replace one dispatch in the chain graph and measure the
stage output difference against the FP16 stage.
