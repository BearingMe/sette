# EXP-044 — MLP GELU and down-projection quantizer fusion

**Milestone:** M15-B · **Status:** PASS. The fused graph preserves the accepted
arithmetic, all 13 diagnostic tensors, the accepted frame/chain hashes, the M09
finalizer and 76/76 native INT8 coverage, and reduces the interleaved INT8 chain
interval by 13.9% and the complete frame by 11.1%. The production arena and
714.8 MB local peak are unchanged. Quality is unchanged: the 45 dB gate still
FAILs.

**Revision:** working tree based on `95d5155` (M15-A); not committed in this
session.
**Depends on:** [EXP-043](EXP-043-graph-lifetime-memory-reuse.md) (arena and
production mode), [EXP-042](EXP-042-performance-baseline.md) (M14 intervals),
[EXP-027](EXP-027-m09-enc1-int8-mlp-pair.md) (accepted MLP pair), [EXP-040](EXP-040-optimization-results.md)
(4-lane quantizer).

## 1. Question and scope

M15-A left 234 dispatches. The post-MLP GELU (`swin_mlp_gelu.cso`) and the
down-projection quantizer (`swin_q8_quant.cso`) were separate dispatches costing
27.8 ms + 12.6 ms in the interleaved control. Can they share one dispatch
without changing the E4M3 GELU values, the per-row scale, any INT8 code, or the
downstream MLP arithmetic? The scope is only those two operators. No
quantization redesign, fidelity tuning, temporal work, proxy/DLL work, new
allocator or FP16-path change is included.

## 2. Verification

| Check | Result |
|---|---|
| Tests | **395 passed** |
| Full INT8 frame hash | **`f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00`** |
| Chain tensor `out-dec1` | **`c89208ed744acd71275f349d99d758c16bd55c77a6fb5d3da88109a3573c6b0d`** |
| Diagnostic tensors | **13/13 byte-identical** to the control chain |
| Native INT8 coverage | **76/76**, 34,518,597,632 MACs (unchanged) |
| M09 finalizer | **17/17 PASS**, run on a fused diagnostic frame |
| D3D12 debug layer | zero error/corruption messages in every stage |
| Determinism | 5/5 full-frame repeats, seed 0 and seed 1 identical |
| Quality | unchanged; 45 dB gate still **FAIL** |

The finalizer's graph guard now accepts either the accepted 234-stage graph
(10 `swin_mlp_gelu` + 84 `swin_q8_quant`) or the fused 224-stage graph
(10 `swin_mlp_gelu_quant` + 74 `swin_q8_quant`), while retaining the same
no-FP16-matrix-fallback and coverage checks.

## 3. B1 — Existing operator contracts

### 3.1 MLP-up output

`swin_mlp_up_q8.cso` computes `Y = (acc * AS[tok] * WS[n]) + Bias[n]` with
`precise float`, rounds once to f16 and stores `Y` as packed f16. For the ten
layers:

| Layer | Tokens | K (input C) | N (hidden 4C) | MLP-up groups |
|---|---:|---:|---:|---:|
| enc1 | 136,640 | 64 | 256 | 34,160 |
| enc2 | 35,712 | 96 | 384 | 13,392 |
| tube#1/3/5 | 9,216 | 128 | 512 | 4,608 |
| tube#2/4/6 | 8,640 | 128 | 512 | 4,320 |
| dec2 | 32,640 | 96 | 384 | 12,240 |
| dec1 | 130,560 | 64 | 256 | 32,640 |

The shader also writes `i8_mlp-up_acc` (int32) and `i8_mlp-up_float` (f32).
Both are diagnostic-only; their writes are deliberately unchanged here.

### 3.2 GELU

`swin_mlp_gelu.cso` uses one thread per token and walks the whole hidden row:

```
z  = HalfAt(i8_z, i)                    // f16 MLP-up output
cz = clamp(z, -2, 2)
inner = Add16(0.412109375, -Mul16(0.0810546875, abs(cz)))
gl = Mul16(z, Add16(0.5, Mul16(cz, inner)))
g  = Q8(gl)                             // E4M3 RNE satfinite, stored as f16
```

`Add16`/`Mul16` are one-rounding f16 operations (exact double intermediate,
then one RNE conversion). The output is an f16 value that represents an E4M3
code, not an arbitrary f16. The quantizer therefore sees exactly the accepted
values. Any f32 reimplementation would move values across E4M3 boundaries.

### 3.3 Quantizer

`swin_q8_quant.cso` is 256 threads, 64 tokens per group, four lanes per token.
Each lane covers `K/4` contiguous elements (K is the hidden size here):

```
mx = max(abs(g)) over the row            // groupshared four-lane reduction
scale = mx * 0.0078740157f
invf  = 127.0f / mx
inv   = correctly rounded 127/mx         // f64 residual correction
code  = clamp(round(g * inv), -127, 127)
```

Codes are packed four per word in row-major `[tokens][K]`; one f32 scale per
token. The `round()` rule is nearest-even, matching the CPU `np.rint` oracle.
A zero row uses `mx = 1`, codes 0, scale `1/127`.

### 3.4 Down projection

`swin_mlp_down_q8.cso` consumes the codes and scales, performs the exact signed
`dot4` reduction over K, applies the two f32 scales, adds the f16 residual in
f32, and rounds once to f16. It does not see GELU or the quantizer internals.

### 3.5 Fusibility and the synchronization constraint

GELU itself is pointwise. The quantizer is not: every code depends on the row
maximum, which spans the whole hidden row. The accepted four-lane split keeps
that reduction inside one 64-token threadgroup. That is the safe fusion
boundary.

A deeper fusion that moves the row reduction into the MLP-up epilogue is not
safe for all layers. MLP-up maps `tile = id.x % (N/16)` and
`tok = id.x / (N/16)` over 64-thread groups. For `N = 384`, `N/16 = 24`, so one
token's 24 tiles are split across two groups (group 0 covers tiles 0-15, group
1 covers tiles 16-23). A row maximum there would need cross-threadgroup
synchronization. That alternative was rejected, not approximated.

## 4. B2 — Small oracle

`scripts/m15b_fusion.py` builds an 8x256 synthetic operand and runs two graphs:

* control: `swin_mlp_gelu` then `swin_q8_quant`;
* fused: one `swin_mlp_gelu_quant` dispatch.

Rows include zero, +2, -2, random values, and GELU/E4M3 boundary values
(`0`, `±0.5`, `±1`, `±1.5`, `±2`, `±448`, subnormal f16 values). The independent
CPU model is `model_enc3.gelu_poly` followed by `m09_enc1_q.quantize_rows_float`.

| Tensor | Control SHA-256 | Fused SHA-256 | CPU exact |
|---|---|---|---|
| GELU/E4M3 `g` | `48217c31c4d758dd9ce6ecbb3e826000ebf198aa6283a0d8af01b9e220710641` | same | yes |
| codes | `33ff5304aacba802e2cd0579d9f7bd6f3df328d7805d8b214ca8d6f3209bb9b1` | same | yes |
| scales | `83f83fc9cbb229ffa370fff581a47581b3b4ed4e0484f89cead0d94e5ad5dd17` | same | yes |

All three are byte-identical between control and fused and exactly match the
CPU oracle. The gate is repeated by `tests/test_m15b_fusion.py` when the D3D12
probe and fused CSO are built.

## 5. B3 — Alternatives

| Approach | Intermediate storage | Extra work | Synchronization | Dispatches | Numerical | Result |
|---|---|---|---|---:|---|---|
| **1. GELU + quantizer in one dispatch** | `g` written once, read once by the same thread's chunk | none beyond the existing GELU | one groupshared four-lane row reduction, one barrier | 1 instead of 2 | byte-identical | **selected** |
| 2. GELU folded into MLP-up, separate quantizer | raw `z` and/or `g`; quantizer unchanged | GELU in the MLP-up epilogue, second MLP-up shader variant | none new | 2 (up+GELU, quantizer) | expected exact | not selected: same stage count, extra variant, diagnostic `i8_z` semantics change |
| 3. Control | `z`, `g`, codes, scales | none | accepted | 2 | accepted | measured control |
| 4. MLP-up + GELU + quantizer in one dispatch | codes/scales only | GELU in epilogue | cross-threadgroup for N=384 | 1 | unsafe | **rejected** |

Approach 1 is the smallest change that removes a dispatch and a full round trip
without changing the row-reduction domain. Approach 2 would still leave two
dispatches and requires another MLP-up variant with a different output
contract; it was not implemented. Approach 4 is structurally unsafe for
`N = 384` as described in B1.

## 6. B4 — Implementation

New shader: `src/native/swin_mlp_gelu_quant_cs.hlsl`, entry `GeluQuant`,
compiled at `cs_6_4` as `swin_mlp_gelu_quant.cso`.

* Parameters: the quantizer's existing `Tokens, K, N, Unused` block.
* Bindings: SRV 0 `i8_z`; UAV 0 `i8_g`; UAV 1 codes; UAV 2 scales.
* 256 threads, `ceil(Tokens/64)` groups, four lanes per token.
* Pass 1: evaluate `GeluQ` once per element, write `g`, accumulate the lane
  maximum.
* Barrier: the accepted groupshared reduction over the four lanes.
* Pass 2: read the same thread's `g` chunk and emit codes/scales with the
  accepted `round`, clamp, scale and inverse-scale arithmetic.
* `GeluQ`, `Q8Bits`, `R16D`, `Add16` and `Mul16` are exact copies of the
  accepted shader helpers. `G` is written before the barrier and read after it;
  each thread only reads values it wrote itself. No global atomics, no new
  allocator, no cross-threadgroup synchronization.

Graph integration is in `scripts/m_layer_suite.py::fuse_mlp_stages` and is
exposed as `--fuse-mlp` on `m_b6_frame.py`, `m14_baseline.py` and
`m15_measure.py`. The rewrite replaces each `<layer>.mlp-gelu` +
`<layer>.mlp-down-quant` pair with `<layer>.mlp-fused`; the resource names,
diagnostic dumps and all other stages are unchanged. `--fuse-mlp-layers` exists
for the one-layer integration gate. The FP16 path never receives the flag.

The M15-A arena needs no new allocation: the fused graph still has 680
resources, 516 placed buffers, a 437,387,264 B heap and the same production
peak. `i8_g` remains the diagnostic tensor and is written exactly as before;
the obsolete separate `i8_z` consumer is gone, but `i8_z` itself is still
produced for the accepted diagnostic dumps. The 168 never-read `Acc`/`FloatOut`
writes remain untouched, deliberately separate from this experiment.

## 7. B5 — Incremental integration

| Step | Result |
|---|---|
| enc1 only, 233 stages | final `c89208ed…3c6b0d`; all 13 dumps byte-identical |
| all ten layers, 224 stages | final `c89208ed…3c6b0d`; 13/13 dumps byte-identical |
| seed 0 vs seed 1 | identical |
| five full-frame repeats | identical `f76ac032…4d6c00` |
| M09 finalizer | 17/17 PASS |

The one-layer step used `fuse_mlp_layers={'enc1'}` and compared every dump
against the control chain. The extension to the remaining nine layers changed
only the stage pairs; no downstream resource name or dump changed.

## 8. B6 — Measurements

All numbers are medians of five interleaved rounds, production mode with the
M15-A arena, same seed and input. Rounds rotate control/fused so clock drift is
shared. Per-dispatch timestamps overlap on this GPU; the whole-command-list
interval is authoritative and family sums are directional.

### 8.1 Chain (ten layers)

| Metric | Control | Fused | Change |
|---|---:|---:|---:|
| GPU interval | **169.466 ms** (165.975-171.258) | **145.929 ms** (142.672-152.684) | **-23.537 ms (13.9%)** |
| Dispatch sum | 168.090 ms | 143.119 ms | -24.971 ms |
| Dispatches | 234 | 224 | -10 |
| Transitions / UAV / aliasing | 1017 / 234 / 234 | 1007 / 224 / 224 | -10 each |
| GELU + quantizer | 40.355 ms | 22.267 ms (fused) | **-18.088 ms** |
| DXGI local peak | 714,829,824 B | 714,780,672 B | -49,152 B (noise) |
| Tracked total | 461,943,424 B | 461,943,424 B | 0 |
| Arena heap | 437,387,264 B | 437,387,264 B | 0 |
| Production peak transient | 376,579,856 B | 376,579,856 B | 0 |

### 8.2 Complete frame

| Metric | Control | Fused | Change |
|---|---:|---:|---:|
| GPU interval | **195.464 ms** (191.596-196.710) | **175.924 ms** (174.321-179.216) | **-19.540 ms (11.1%)** |
| Chain interval | 169.094 ms | 149.877 ms | -19.217 ms |
| Dispatches | 243 | 233 | -10 |
| GELU + quantizer | 42.124 ms | 24.737 ms (fused) | **-17.387 ms** |
| Frame hash | `f76ac032…4d6c00` | same | 0 |
| DXGI local peak | 714,829,824 B | 714,780,672 B | -49,152 B (noise) |
| Host total | 2.685 s | 2.671 s | -14 ms |
| Load/alloc | 1249.1 ms | 1247.2 ms | -1.9 ms |

Against M15-A's production frame (195.88 ms) the fused frame is 10.2% lower;
the same-session control is the stronger comparison because it shares clock and
power conditions.

### 8.3 Per-layer MLP pair (chain medians)

| Layer | GELU | Quantizer | Control sum | Fused | Saving |
|---|---:|---:|---:|---:|---:|
| enc1 | 4.975 | 3.614 | 8.589 | 5.838 | 2.751 |
| enc2 | 3.359 | 1.446 | 4.804 | 2.581 | 2.223 |
| tube#1 | 1.263 | 0.457 | 1.720 | 0.860 | 0.860 |
| tube#2 | 1.442 | 0.415 | 1.856 | 0.892 | 0.964 |
| tube#3 | 1.226 | 0.465 | 1.692 | 0.828 | 0.864 |
| tube#4 | 1.260 | 0.406 | 1.666 | 0.857 | 0.810 |
| tube#5 | 1.225 | 0.460 | 1.685 | 0.855 | 0.830 |
| tube#6 | 1.705 | 0.416 | 2.121 | 0.879 | 1.242 |
| dec2 | 2.916 | 1.430 | 4.346 | 2.557 | 1.789 |
| dec1 | 7.934 | 3.442 | 11.376 | 5.830 | 5.546 |
| **Total** | **27.304** | **12.551** | **39.855** | **21.977** | **17.878** |

The per-layer savings sum to 17.9 ms; the interval saving is 23.5 ms. The extra
is reduced launch/barrier overhead and changed overlap, not a hidden arithmetic
change. MLP-up and MLP-down dispatch times are unchanged within noise.

## 9. Rejected or deferred work

* **Row reduction in the MLP-up threadgroup** — rejected for N=384 (cross-group
  token rows).
* **GELU folded into MLP-up with a separate quantizer** — not selected: still
  two dispatches, requires a second MLP-up output contract, and would change
  the diagnostic meaning of `i8_z` or add another variant for no measured gain.
* **Removing the 168 never-read `Acc`/`FloatOut` writes** — measured in M15-A as
  the next bandwidth candidate, deliberately not bundled here.
* **Asymmetric quantization** — the accepted full-INT8 graph is symmetric; the
  opt-in asymmetric path is unchanged and was not touched.
* **FP16 path** — no shader, graph or measurement change.

## 10. Limitations and next task

The fusion is opt-in (`--fuse-mlp`) so the accepted control graph remains
available for A/B and for the M09 finalizer's 234-stage path. Per-dispatch
timestamps are not isolated; the interval is the accepted metric. Quality is
unchanged at the accepted INT8 output, so the 45 dB gate remains failed for the
reasons recorded in EXP-041. The GPU is still re-launched per experiment; M15-C
will address persistent in-process execution with changing inputs and
external-resource readiness, without changing the fused arithmetic.

## 11. Evidence

The small reports and graphs are archived outside the repository at
`%LOCALAPPDATA%\Temp\opencode\evidence-20261030-m15b`:

`oracle.json`, `m15b-measure.json`, `m15b-frame.json`,
`m15b-finalize.json`, `m15b-fused-plan.json`, the control and fused
`graph.txt`, `fused-b6.json`, per-run chain timing, the fused arena report and
the fused execution log. The multi-gigabyte frame and dump trees were removed
after the reports were written; the accepted layer fixtures remain in
`%LOCALAPPDATA%\Temp\opencode\m09h-complete-chain-20261030`.

## 12. Reproduction

```
python scripts/m15b_fusion.py --out <fresh-oracle>
python -m pytest tests/test_m15b_fusion.py -q
python scripts/m15b_measure.py \
  --control-graph <control-chain>/graph.txt \
  --fused-graph <fused-chain>/graph.txt \
  --out <fresh-chain-measure> --repeats 5
python scripts/m15b_frame.py --out <fresh-frame-measure> --repeats 5
```

Build the fused chain with `m_b6_frame.build_chain(..., fuse_mlp=True)` or the
`--fuse-mlp` flag. The fused CSO is built by the normal `cmake --build
build/native --config Release` target.
