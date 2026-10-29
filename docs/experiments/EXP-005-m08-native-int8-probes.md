# EXP-005: Native INT8 WMMA Execution and ISA Evidence on gfx1101

**Milestone:** M08
**Status:** PASS

## Question

Does native INT8 matrix hardware on the target GPU execute the operations M09
needs, with the signedness, accumulator and overflow behaviour the Preset M
numerical contract assumes? Acceptance criteria, defined before execution:

1. All four INT8 operand sign combinations produce results identical to an
   independent CPU reference.
2. The generated gfx1101 ISA contains the wave-matrix instruction, not an
   emulation, for each combination.
3. Multi-step INT32 accumulation stays exact when the exact sum fits in int32.
4. The overflow behaviour is observed, not assumed.
5. Representative network shapes (16-256 rows/columns, K 64-256) agree with the
   reference.

## Method

Two independent evidence paths, both on the RX 7700 XT (`gfx1101`):

- **Reference runtime path.** The pinned quattro harness (`ad0482a`) was built
  in a temporary copy and its `wmma-int8` and `kernels-int8` experiments run.
  These compare GPU results against quattro's CPU reference and capture the ISA
  that AMD's shader analyzer reports for the very pipeline object that ran.
- **Offline ISA path.** `RadeonGPUAnalyzer` 2.15 (`rga -s dx12 -c gfx1101`,
  DXC 1.8.2502.11, `-enable-16bit-types`, `cs_6_2`/`cs_6_4`) compiled the same
  HLSL entry points from `references/sources/quattro/src/gpu/shaders/` and
  disassembled the result.

The accumulator overflow question was answered with a **local probe added to a
disposable copy** of the quattro harness (`D4R_M08_SATURATE=1`): the tile inputs
become `A = B = 127`, the accumulator starts at `INT32_MAX - 1000`, and the exact
sum (`+258 064`) overflows int32. The probe prints the GPU result next to the
CPU wrapping result, so wrap and saturation are distinguishable in one run.
`references/sources/quattro` itself was not modified, and the probe is not part
of this repository.

## Results

### ISA: the instruction is native and sign is an operand modifier

Driver analyzer reports for the executed pipelines:

| Entry point | ISA |
|---|---|
| `main_i8` | `v_wmma_i32_16x16x16_iu8 v[0:7], neg_lo(v[0:3]), neg_lo(v[20:23]), v[8:15]` |
| `main_u8` | `v_wmma_i32_16x16x16_iu8 v[0:7], v[0:3], v[20:23], v[8:15]` |
| `main_iu8` | `v_wmma_i32_16x16x16_iu8 v[0:7], neg_lo(v[0:3]), v[20:23], v[8:15]` |
| `main_ui8` | `v_wmma_i32_16x16x16_iu8 v[0:7], v[0:3], neg_lo(v[20:23]), v[8:15]` |
| `int8_dp4a` control | `v_dot4_i32_iu8 v0, neg_lo(v22), neg_lo(v0), v19` (x4) |
| `gemm_int8_wmma_k256` | twelve+ `v_wmma_i32_16x16x16_iu8` with `neg_lo` pairs |
| FP16 reference | `v_wmma_f32_16x16x16_f16` / `v_wmma_f16_16x16x16_f16` |

[S] There is one INT8 wave-matrix opcode. Signedness is carried by the per-operand
`neg_lo` modifier: signed operands are negated, unsigned ones are not. This is
why one instruction serves all four combinations.

[M] Offline RGA produced exactly the same instruction for the same entry points,
including `neg_lo:[1,1,0]` for `i8*i8`, `neg_lo:[1,0,0]` for `i8*u8` and
`neg_lo:[0,1,0]` for `u8*i8`.

### Correctness against an independent CPU reference

| Kernel | Result | ISA | Timing |
|---|---|---|---|
| `wmma i8*i8` | 0 mismatches, idempotent | wmma x1, dot4 x0 | min 14.4 us |
| `wmma u8*u8` | 0 mismatches, idempotent | wmma x1, dot4 x0 | min 13.8 us |
| `wmma i8*u8` | 0 mismatches, idempotent | wmma x1, dot4 x0 | min 14.4 us |
| `wmma u8*i8` | 0 mismatches, idempotent | wmma x1, dot4 x0 | min 14.6 us |
| `dp4a i8*i8` (control) | 0 mismatches, idempotent | wmma x0, dot4 x4 | min 2.4 us |

[M] A single 16x16x16 tile is *faster* with DP4A (2.4 us) than with WMMA
(14.4 us): one wave-matrix instruction occupies a 32-lane wave for one tile and
pays the fragment setup, while the DP4A control uses 256 threads. This is a
per-dispatch measurement on one tile and says nothing about throughput at the
network's matrix sizes; M14 owns that comparison.

### Multi-step accumulation and representative shapes

[M] `kernels-int8` (same hardware, CPU reference per shape):

```text
gemm_raw_m16k64n16     wmma x4   | 0 mismatches
gemm_raw_m32k64n32     wmma x4   | 0 mismatches
gemm_raw_m16k256n16    wmma x16  | 0 mismatches
gemm_raw_m64k144n16    wmma x9   | 0 mismatches
gemm_raw_m64k144n32    wmma x9   | 0 mismatches
gemm_raw_m256k64n256   wmma x4   | 0 mismatches
gemm_raw_m256k256n256  wmma x16  | 0 mismatches
gemm_raw 20x24x64 (boundary)     | 0 mismatches
gemm_requant 20x24x64            | 0 mismatches vs AMD-contract reference
conv3x3p1 7x7x16->20 (boundary)  | 0 mismatches vs CPU conv
overflow guard: K=131071 allowed; K=131072 REFUSED
```

K=64/144/256 are the shapes the M network actually needs (64/96/128 channels,
K multiples of 16), and the boundary cases confirm the padded-tile handling.

### Accumulator overflow is wraparound, not saturation

[M] Saturation probe, `A = B = 127`, `initC = 2 147 482 647`, exact sum
`2 147 740 711`:

```text
gpu[0] = -2147226585   initC[0] = 2147482647   cpu_wrap[0] = -2147226585
```

[S] The GPU result equals the two's complement wrap of the exact sum
(`2 147 740 711 - 2^32 = -2 147 226 585`), for all four sign combinations. No
saturating result was produced.

Consequence for M09: the accumulator has no overflow protection. The product
bound `127 * 127 * K < 2^31` gives the `K <= 131 071` guard that quattro's
kernels enforce, and the reconstructed kernels must keep the same guarantee or
carry an explicit wider accumulator.

### Register and occupancy footprint

[M] From the driver analyzer: `vgpr 24/1536, sgpr 20/2048, lds 0 B/group` for the
tile kernel; `vgpr 26-34` for the multi-tile GEMMs. LDS is zero because the AMD
intrinsics load operands straight from memory into the fragment registers.

## Limitations and Unknowns

- [M] A HIP variant of the probe was attempted first and **failed for toolchain
  reasons, not numerically**: ROCm 7.2's Windows clang emitted a fat binary with
  no kernels for every `__global__` entry point, so `hipModuleLaunchKernel`
  reported "Cannot find Symbol". The probe source was removed rather than kept
  broken; the D3D12 path above is the project's target path anyway.
- [U] Throughput at network-sized matrices is not measured here (M14).
- [U] Only 16-deep K steps were exercised on the wave-matrix unit itself; the
  chained `K=256` cases confirm the arithmetic, not the scheduler.

## Files

- `tests/` is unaffected; no probe code was committed.
- ISA evidence regenerates with the RGA command line recorded above.

## References

- [M04 numerics](../findings/2026-10-28-preset-m-numerics.md)
- [EXP-001](EXP-001-reference-toolchain.md) (reference build and versions)
- [M05 coverage audit](../findings/2026-10-29-m05-capture-coverage.md)
