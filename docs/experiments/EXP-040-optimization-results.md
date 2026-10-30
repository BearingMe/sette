# EXP-040 — Targeted optimization results

**Milestone:** M12-D · **Status:** PASS — one optimization accepted, two deferred
on measured risk/benefit. INT8 arithmetic and frame output are unchanged.
**Depends on:** [EXP-037](EXP-037-profiling-environment.md),
[EXP-038](EXP-038-dispatch-profile.md), [EXP-039](EXP-039-bottleneck-analysis.md).

## 1. Question

Which of the measured bottlenecks can be reduced without changing any arithmetic,
and does the change produce a repeatable end-to-end GPU benefit?

## 2. Candidate selection

From the dispatch profile (old quantizer, INT8 chain 204.9 ms median):

| Candidate | Measured cost | Ceiling if removed | Risk |
|---|---:|---:|---|
| Quantization + pack adapters | 52.9 ms (28.0% of chain) | ~24% of frame | arithmetic-neutral if fused/parallelized |
| Attention `pack` alone | 18.2–19.9 ms | ~9% of frame | layout feeds three quantizers |
| `post` reconstruction | 14.8 ms (6.5% of frame) | 6.5% | shared FP16 stage, outside INT8 matrix scope |
| Fusion of quantize into GEMM load | same 52.9 ms family | ~22% of chain | new GEMM entry, full revalidation |

The quantizer itself is **not bandwidth-bound**: the measured effective bandwidth
of the large quantization dispatches is 10–22 GB/s while a small compute stage
reaches ~150 GB/s (EXP-039). That makes per-row parallelism the cheapest change.

## 3. Implemented: four lanes per token row

`src/native/swin_q8_quant_cs.hlsl` changed from one thread per token
(`[numthreads(64,1,1)]`) to four lanes per token (`[numthreads(256,1,1)]`, still
64 tokens per group, so every host dispatch dimension is unchanged):

* each lane computes `max|v|` over its quarter of the row;
* a 4-entry groupshared max combines them — `max` is exact and associative;
* each lane re-derives the same corrected f32 inverse and quantizes its quarter;
* words are written by the lane that owns them; `Scales[tok]` by lane 0;
* inactive lanes in a partial last group still reach the barrier.

All uses have `K ∈ {128, 32, 16}`, so each lane owns a whole number of 4-element
words. The change is a parallelization, not a different arithmetic contract.

## 4. Correctness gate

| Check | Result |
|---|---|
| Full INT8 frame SHA-256 | **`f76ac032…4d6c00` — identical to the accepted frame** |
| Full FP16 frame SHA-256 | `dc97c4cb…d6e95b` — unchanged |
| enc0 all-element CPU check | **13/13 PASS** (features, codes, scales, INT32, f32, f16, field, scalar probes, deterministic writes, zero debug) |
| Attention pair all-element check | **23/23 PASS** (Q/V/P codes and scales, QK, PV, outputs, bounds, zero debug) |
| M09 complete-network finalizer | **17/17 checks PASS** on the optimized build, using the new frame and the historical chain/ISA evidence |
| Regression tests | **365 passed** |
| Native matrix shaders | unchanged; `dot4` coverage and hashes untouched |

The changed shader hash is `242aaa95…d79229` (was `316fcf87…e25c52`). It is the
shared activation quantizer, not one of the five RGA-inspected matrix shaders.

## 5. Measured effect

A/B on the same build family, 1 warm-up + 4 measured repeats per side, INT8 only,
all frame hashes equal:

| Block | GPU median | min | max |
|---|---:|---:|---:|
| baseline (old quantizer) | 221.336 | 219.513 | 232.323 |
| baseline re-run | 224.401 | 220.851 | 228.679 |
| baseline (full profile) | 226.780 | 220.427 | 233.735 |
| **optimized** | **209.626** | **206.077** | **214.216** |

The optimized maximum is below the baseline minimum. Median improvement in the
A/B blocks **14.8–17.2 ms (6.6–7.6%)**; the final interleaved run (EXP-038)
measured **205.167 ms**, i.e. 7.3–9.5% below the baseline blocks. Chain-label
total 188.7 → 171.8 ms.

| Family | Baseline | Optimized | Change |
|---|---:|---:|---:|
| quantize + pack | 52.9 ms (28.0%) | 37.5 ms (21.8%) | **−29%** |
| `mlp-down-quant` | 16.66 ms | 12.73 ms | −24% |

## 6. Rejected / deferred

* **Quantize-into-GEMM fusion** — would remove the remaining 37.5 ms, but needs a
  new GEMM entry and a full CPU/ISA/frame revalidation. Not implemented; it is
  the highest-value next step.
* **Attention pack fusion** — 18.2 ms; the pack output feeds three quantizers and
  the E4M3 layout, so a fusion changes a validated boundary. Deferred.
* **`post` optimization** — 14.8 ms, shared by both modes and outside the INT8
  matrix scope. Not attempted.
* **WMMA tile / DP4A alternatives** — not applicable: the INT8 path already uses
  `dot4`; no FP16 matrix instruction remains in the full mode.

## 7. Interpretation

* [M] The quantizer was latency/occupancy-bound, not bandwidth-bound; 4-way row
  parallelism cuts its cost by ~29% with bit-identical output.
* [M] End-to-end INT8 GPU time improves ~7% without touching arithmetic.
* [S] The remaining adapters are still ~22% of the chain; fusion, not more
  parallelism, is the next lever.
* [U] Whether the gain holds on other content or after fusion.

## 8. Reproduction

```powershell
cmake --build build/native --config Release
python scripts/m12_profile.py --out "$T/m12-ab-optimized" --modes int8 --repeats 4
python scripts/m09_complete.py --phase enc0 --enc0-fixture "$F/enc0" --out "$T/enc0"
python scripts/m09_complete.py --phase pair --layers "$T/m09g-spatial-verified-20261030/layers" `
  --inputs "$T/m09h-attention-cpu-20261030/results" --out "$T/pair"
```
