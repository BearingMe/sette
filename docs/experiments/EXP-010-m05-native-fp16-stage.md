# EXP-010: Native D3D12 FP16 Stage, and the f16 Conversion Trap

**Milestone:** M05 (checkpoint B3/C18), with the first C19 groundwork
**Status:** PASS for the execution foundation; one stage measured

## Question

Can this project execute a Preset M matrix stage natively on Windows/D3D12/HLSL,
from captured weights and a captured operand, and reproduce the published
accumulator contract?  Acceptance criteria, fixed before execution:

1. A D3D12 compute path exists with explicit resource roles, shader compilation
   through DXC, deterministic buffer initialisation, readback, and error
   reporting - no general inference framework.
2. It runs on the project's target device and produces finite output.
3. Its output is compared with a reference computed independently of the shader
   (float64 mathematics and d4r's published step chain).
4. Any numerical disagreement is explained by a measured mechanism, not by a
   tolerance chosen after the fact.

## Method

New: `src/native/m_stage_probe.cpp`, `src/native/gemm_f16_cs.hlsl`,
`src/native/CMakeLists.txt`, `scripts/m_stage_case.py`.

```powershell
cmake -S src/native -B build/native -G "Visual Studio 17 2022" -A x64
cmake --build build/native --config Release
python scripts/m_stage_case.py export <capture> --matrix dec0.expansion --out case.bin --rows 8160 --stride 16
build/native/Release/m_stage_probe.exe --case case.bin --out result.bin --shader build/native/gemm_f16_cs.cso
python scripts/m_stage_case.py check --case case.bin --result result.bin
```

The stage under test is `dec0`'s expansion GEMM: `[8160 x 64] x [64 x 80]`, the
captured token operand (a deterministic stride-16 sample of the 130 560-token
field), the captured e4m3 matrix widened to f16, and the stage's f16 bias as the
initial accumulator - exactly the operands EXP-009 measured.

The shader implements the contract from M04/M06 as measured in
`dec0_m.comp`: products summed in f32 within a 32-element step, **one f16
rounding per step**, next step starting from the rounded value.

## Results

### The execution path

[M] `m_stage_probe.exe` enumerates the AMD RX 7700 XT at feature level 12_0,
compiles the shader with DXC, dispatches 10 200 workgroups of 64 threads, copies
2 611 200 bytes back through a readback heap, and exits 0.  The output is finite
with `|y|max = 13.0156`; nothing is zero-filled or skipped.

### The measured disagreement, and why

[M] First version, rounding written the obvious way
(`f16tof32(f32tof16(x))`, DXC `cs_6_0`, no 16-bit types):

| Comparison | identical f16 | max abs | rms |
|---|---:|---:|---:|
| GPU vs f16-step chain | 35.2% | 0.0156 | 0.00103 |
| GPU vs exact float64 | 27.9% | 0.0121 | 0.00090 |

The GPU was *further* from the exact product than the chained reference was,
and every sampled GPU value sat exactly one f16 step below the chain.  Three
rounding models were then computed in Python and compared with the same GPU
output:

| Model | identical f16 | rms |
|---|---:|---:|
| round toward zero | **95.75%** | 0.000085 |
| round to nearest even | 35.18% | 0.00103 |

[M] The driver's f32 -> f16 conversion truncates toward zero.  A truncating step
does not just add noise: it biases every accumulated step toward zero, which is
why the GPU drifted away from the exact product instead of around it.

[M] With the conversion replaced by explicit integer round-to-nearest-even
(`ToHalfBits` in `gemm_f16_cs.hlsl`), the same case gives:

| Comparison | identical f16 | max abs | rms | max/peak |
|---|---:|---:|---:|---:|
| GPU vs f16-step chain | **97.9%** | 0.0039 | 0.00006 | 0.00003 |
| GPU vs exact float64 | 75.0% | 0.0076 | 0.00038 | 0.00006 |
| chain vs exact float64 | 75.2% | 0.0076 | 0.00038 | 0.00006 |

[M] The GPU now agrees with the chain as closely as the chain agrees with itself
across implementations: the residual 2.1% are single-step differences where the
f32 sum of 32 products lands on the other side of a rounding boundary than the
float64 sum (`chain` and `exact` differ on 24.8% of values by construction).

### Consequences

- The native stage reproduces the published contract to within one f16 step on
  every element, with 97.9% bit-identical, from real captured weights and a real
  captured operand.
- Any future FP16 or INT8 kernel that converts f32 to f16 must use an explicit
  round-to-nearest-even path on this driver.  Relying on the intrinsic silently
  biases results by one step per accumulation, which is larger than most
  quantization errors this project is about to measure.

## Limitations

- [M] One matrix (`dec0` expansion), one operand sample (8 160 of 130 560 rows,
  stride 16), one frame, one driver.
- [U] The step sum is f32 with the compiler's own association; the contract's
  "exact sum within a step" is not reproduced bit-exactly, and the 2.1% residual
  is attributed to that difference rather than proven per element.
- [U] `dec0`'s epilogue (four surfaces, gates/mix/covariance) is not implemented;
  this experiment covers the matrix stage only.
- [U] No WMMA/tensor-core path yet: this is a scalar HLSL implementation, so no
  throughput claim is made (M14).

## Files

- `src/native/m_stage_probe.cpp`, `src/native/gemm_f16_cs.hlsl`,
  `src/native/CMakeLists.txt`
- `scripts/m_stage_case.py`
- Cases and results stay in the temporary workspace (they contain NVIDIA-derived
  weights).

## References

- [EXP-009](EXP-009-m07-complete-inventory.md) (the captured operands used here)
- [EXP-006](EXP-006-m06-operator-contract.md) (the accumulator chain contract)
- [stage graph](../findings/2026-10-29-m05-stage-graph.md) (why `dec0`'s
  epilogue cannot be validated from this capture yet)
- [EXP-001](EXP-001-reference-toolchain.md) (D3D12/DXC toolchain)
