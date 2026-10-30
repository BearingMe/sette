# EXP-029 — Complete native INT8 matrix execution

**Milestone:** M09-H · **Status:** PASS for bounded reset-frame matrix execution.
**Image regression:** FAIL, unchanged 45 dB floor. **General fidelity:** INCONCLUSIVE.
**Baseline:** `c129558` (M09-G committed after audit; no push).
Target: Windows/D3D12, RX 7700 XT / gfx1101, established SDK DXC / RGA 2.15.

## Question and acceptance criteria

Can enc0 and all twenty dynamic attention products join the cumulative native
INT8 path without changing FP16 defaults, original nonlinear arithmetic or the
76-operator MAC denominator?

Require independent CPU codes/scales, INT64 dot references, FP32 rescaling and
FP16 conversions; exact layouts/complete writes; safe GPU resource handoffs;
finite unsaturated output; deterministic execution; gfx1101 signed dot4 tied to
executed binaries; explicit FP16 fallback; full frames with unchanged quality gates.
Close enc0 first, then establish a CPU attention contract and validate one pair
before expanding. Image quality is diagnostic, not an execution acceptance test.

## H1 — Preserved baseline

[M] Before changes: **339 tests pass**. M09-G's pending work was reviewed and
committed as `c129558 feat(m09): expand cumulative native int8 matrix coverage`.
Nothing pushed. The audit rechecks its actual stage selection (54 learned neural
matrices plus dec0), accepted CPU/frame reports, generic/dec0 executed ISA hashes
and all 21 classified gaps. Its weighted total is **34,518,597,632 dense MACs**.

The [complete inventory](EXP-028-matrix-inventory.json) preserves the original
G snapshot and adds an H overlay. Dimensions, attention instance counts, padded
rows, active dec0 columns and non-neural-GEMM exclusions are unchanged. Coverage
is logical operator count and dense shape MACs, **not dispatch percentage or
measured instruction count**. Historical G evidence remains in [EXP-028](EXP-028-m09-cumulative-int8.md).

## H2 — Enc0 GPU prefix and field handoff

[S] `enc0_cs.hlsl` and the existing independent `m_enc0_model.py` establish the
reset-frame contract; no renewed FP16 reconstruction or target reverse engineering.
Input is a 960×540 RGBA16F texture, visible 960×540, padded 960×544. Each of
130,560 tokens has four pixels, 32 feature slots per pixel, for **M/K/N =
130560/128/64**. Mirror coordinates, frame-indexed dither lookup, two f16 dither
roundings and E4M3 feature encoding remain exactly the original operations.

`enc0_q8_cs.hlsl::Features` uses the original texture/dither helpers to write
a natural row-major f16 feature buffer. Values are E4M3-representable, so packing
to f16 loses no information. The existing `swin_q8_quant.cso` and
`swin_mlp_up_q8.cso` then perform the same signed symmetric row/column scheme
used by the other learned matrices. Weight-column quantization is offline and
scalar-checked; **all activation preparation and encoding remain on GPU**.

The fixed prefix in `m_image_stage.cpp` retains the real texture view and GPU
buffers through one command submission. Explicit UAV→SRV transitions connect
features, codes/scales, embedding and field epilogue; no CPU activation upload
or readback between those operations. `enc0_q8_epilogue_cs.hlsl` reads the f16
embedding and invokes the original field code through a bounded Column hook:
same ±6.28125 clamp, 64-column permutation and two-plane E4M3 output.
FP16's default shader is unchanged **byte for byte** after compilation.

The manifest's explicit `enc0_int8` mode validates input dimensions, formats,
blob sizes and reset/frame bounds. No new execution framework. B6 keeps its
existing **file boundaries between image and neural stages**; the claim is GPU
residency within each prefix and the neural chain, not one all-GPU backend or
zero-copy end-to-end application. The emitted enc0 field is correctly consumed
by the existing neural graph, proven by resource-content/graph and output hashes.

[M] All **16,711,680 feature halves**, codes/scales, 8,355,840 INT32 accumulators,
FP32 values, embedding halves and 8,355,840 field bytes match independent CPU
expectations bit for bit. All weight columns have scalar encoding checks; selected
rows/columns have independent scalar encoding/dot probes. Accumulators observed
−48,241…71,964, against the safe whole-K bound `128*127² = 2,064,512`.
Features range 0…0.8125, activation scales 0.00393700786…0.00639763754.
Two sentinel seeds reproduce all diagnostic/field bytes. Verified debug logs:
zero messages. FP16 embedding difference is **1.385387% relative RMS**, RMS
0.005341654, mean absolute 0.004276318, maximum 0.01977539. Outputs finite and
unsaturated. The final complete frame's enc0 prefix is independently rechecked.

After enc0, a complete cumulative frame was accepted for execution at **56/76
operators**, **96.9680% MAC coverage**. Its quality gate remains FAIL (table below).

## H3/H4 — CPU-first dynamic attention contract

### Observed original contract

[S] `swin_layer_cs.hlsl::Attention` uses natural padded Q/V tensors of shape
`[tokens, heads*32]`. Each head attends to 16 tokens in a **4×4 window** inside
the prepared 8×8 patch grid. The earlier Prepare stage already performs layer
shift/mirror handling. No additional shift belongs inside attention. Enc1/dec1
have two heads; enc2, six tube layers and dec2 have four (including C=96 with
128 projection channels). Q serves as K; there is no separate K projection.

Original QKᵀ uses E4M3-round-tripped Q, an exact 32-product sum plus original
score bias followed by f16 rounding. The original polynomial/packed-exponent
softmax produces f16 probabilities. PV uses those probabilities and f16 V,
an exact 16-product sum, f16 rounding then E4M3 output.

### Chosen INT8 arithmetic (not the learned-weight scale contract)

[S][M] `scripts/m09_complete.py::attention_reference` was implemented and run on
the real enc1 Q/V inputs **before the GPU replacement**. For each window/head:

* Physical packed Q is `[16,32]`; quantize each row after the original E4M3
  round trip. QKᵀ reuses that dynamic encoding as its transposed RHS.
* Encode Q, P and transposed V with the established runtime rule:
  `top=max(abs(row))`, substitute 1 for zero rows, f32 `inverse=127/top`,
  f32 `scale=top*(1/127)`, half-even rounded f32 products clipped to ±127.
  The GPU uses the existing corrected f32 inverse/product quantizer.
* QKᵀ: signed dot4 over K=32, exact INT32. Two ordered f32 multiplies by
  the query/key row scales; one f32 add of the original score bias; explicit
  RNE f16 score. **This changes the matrix precision contract intentionally**;
  it does not claim to reproduce original exact/f16 matrix arithmetic.
* Original softmax arithmetic remains unchanged: scale/clamp, f16 polynomial
  FMAs, packed-word exponent transform, f16 butterfly in order 8/2/4/1,
  f32 reciprocal rounded to f16, f16 probability multiply.
* P is `[16,16]`, quantized per probability row. V is f16 `[16,32]`, transposed
  on GPU to `[32,16]`, quantized per **window-local output column**. Its scales
  are runtime activation scales, not offline or global learned-weight scales.
* PV: signed dot4 over K=16, exact INT32. Ordered f32 P-row/V-column rescaling;
  RNE f16 output; original E4M3 conversion; scatter to the original natural
  `[tokens,heads*32]` format.

The worst-case signed accumulator magnitudes are `32*127² = 516,128` and
`16*127² = 258,064`, below both 2²⁴ (exact INT32→f32) and 2³¹. No zero points,
clipping calibration, optimizations or new normalization/residual arithmetic.

### Representative pair gate

[M] On enc1's actual cumulative INT8 inputs: 17,080 window/head instances,
Q range −1.375…1; V −3.06640625…1.947265625; P
0.000014066696…0.98291015625. QK accumulators 30,509…128,557; PV
−105,455…104,972. CPU scalar probes and integer bounds pass.

Only after that CPU gate, the GPU implementation was added. It uses a bounded
pack adapter, three invocations of the existing runtime quantizer, two new dot4
matrix shaders and an unchanged-arithmetic softmax entry point. All codes,
scales, integer dots, FP32 rescaling, f16 scores/probabilities/value conversions
and final E4M3 outputs agree **bit for bit**. Both sentinel seeds agree; zero
D3D12 messages. RGA confirms native signed dot4 before all-layer expansion.

Representative same-input errors against the independent FP16 attention model:
QK scores **0.152480% relative RMS**; complete attention output **2.048462%**.
These are expected INT8-vs-FP16 differences, not GPU implementation error.

## H5 — Systematic cumulative validation

[M] The exact same contract runs in all ten layers, without new per-layer kernels.
Window counts include padded rows, matching the existing MAC denominator:

| Layer | Window/head instances | Score relative RMS vs same-input FP16 | Attention output relative RMS |
|---|---:|---:|---:|
| enc1 | 17,080 | 0.152480% | 2.048462% |
| enc2 | 8,928 | 0.233807% | 1.821767% |
| tube#1 | 2,304 | 0.280072% | 1.669343% |
| tube#2 | 2,160 | 0.267261% | 1.593623% |
| tube#3 | 2,304 | 0.270531% | 1.414034% |
| tube#4 | 2,160 | 0.223453% | 1.486651% |
| tube#5 | 2,304 | 0.281633% | 1.417366% |
| tube#6 | 2,160 | 0.219847% | 1.464759% |
| dec2 | 8,160 | 0.251429% | 1.975805% |
| dec1 | 16,320 | 0.265904% | 1.803009% |

Across those inputs Q ranges −3.5…3.5, V −6.27734375…7.375. Every layer's
diagnostics preserve the actual runtime ranges, codes/scales and comparison
metrics. All **63,880** window/head pairs pass: **147,179,520** attention
accumulator/FP32/f16 elements, plus every packed operand, scale, probability and
final output. Original learned-weight operators and all layout/GELU handoffs
are revalidated on the new cumulative inputs: **264,440,320 elements at each of
INT32, FP32 and f16** match, in addition to 25,067,520 enc0 and 31,334,400 live dec0
accumulator/FP32/f16 elements. Counts include three distinct arithmetic boundaries;
they are not three logical operators.

Synthetic GPU cases additionally exercise two/four heads, multiple windows,
positive/negative values, entirely zero Q windows, zero V columns and signed
zeros. All CPU/GPU boundaries and both sentinel seeds match exactly.

The complete neural graph has **234 dispatches / 680 resources**; dispatches
include adapters and floating-point work and are not a coverage denominator.
All diagnostic bytes repeat across sentinel seeds. Zero D3D12 messages in
the complete chain and full-frame execution logs. Frame graph resources,
parameters, bindings and shaders match the CPU-checked graph; all exported
layer fields match and are finite with zero E4M3 saturation. Live final-frame
enc0 and dec0 arithmetic are independently checked again.

| Coverage | INT8 logical ops | Dense MACs | MAC coverage |
|---|---:|---:|---:|
| G spatial baseline | 55/76 | 32,402,440,192 | 93.8695% |
| + enc0 | 56/76 | 33,471,987,712 | 96.9680% |
| + all QKᵀ/PV | **76/76** | **34,518,597,632** | **100%** |

A strict stage audit rejects absent quantizers/products or a silent original
FP16 attention substitution. Full-frame audit additionally rejects any original
FP16 Matmul/Mlp/Merge/Attention shader in the full neural graph and requires
34 zero/bias and 20 residual learned GEMMs plus ten QKᵀ/PV pairs. Enc0/dec0
modes are explicit and separately required. Default FP16 modes remain intact.

## Native gfx1101 ISA and provenance

[M] RGA inspects the same matrix binaries used in the representative pair and
complete frame. All inspected hashes match stage/frame/current-file hashes.

| Matrix shader | Static signed dot4 | FP16 matrix instructions | SHA-256 |
|---|---:|---:|---|
| Shared zero/bias GEMM (including enc0) | 16 | 0 | `f5324bcbd9ac70e91b1cb813d2ead573445483ced6a3cf45e03df495f670e217` |
| Shared residual GEMM | 16 | 0 | `115764abfb6531673d3db2f21fff4eca005b13b6cec76e273a81fb87014d7f75` |
| dec0 GEMM | 12 | 0 | `c7fc7fa6a3aa67c540edaea7a76d0d2c3f5d1646c3b581609e8efa062e405a89` |
| Dynamic QKᵀ | 128 | 0 | `c58f77ed9db8480d9c7e11d9e609ca14050b0c07d0df5da1284fbe112a08f4c8` |
| Dynamic PV | 128 | 0 | `4741de9607e1b02bfd780615fb46eb243ba24f0b460c5b03fc9daea7426e0412` |

Opcodes are `v_dot4_i32_i8`; counts are static, not a runtime instruction trace.
The accepted ISA run is gated on the representative pair's PASS report and on
that pair's QKᵀ/PV hashes being the same binaries the final frame executed, so
the pre-expansion inspection order is evidenced rather than asserted.
Image host hash: `1419ec594e685a3637de20a421a48cffb03b464e962804f112dbd496a9a12cb5`.
Neural host remains `1e5f47adadb24a379510f5ae3e1788c2b4989e18bebbb8ced46a9d2dd165b3f2`.
Enc0 adapter/epilogue hashes and every other executed shader are retained in
reports. Original enc0 compiled hash remains
`b35c0277d2836c3599dcdca743ae08f7098f16a427dadaf71cfe965d19e76e2e`.

## H6 — Complete frames, regressions and remaining exceptions

[M] Same raw linear RGB/f16 PSNR and windowed luma SSIM as EXP-028; metrics
exclude no nonfinite samples because none occur. Alpha is byte-identical.

| Frame vs canonical | PSNR dB | Windowed SSIM | RMS | Mean abs | P99 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| FP16 control | 47.051389 | 0.999258964 | 0.00444049 | 0.00108616 | 0.0229492 | 0.178223 |
| G spatial INT8 | 44.377854 | 0.998745519 | 0.00604098 | 0.00139357 | 0.0317383 | 0.252441 |
| + enc0 INT8 | 44.211974 | 0.998589366 | 0.00615746 | 0.00143411 | 0.0324707 | 0.188477 |
| **Full matrix INT8** | **44.139286** | **0.998632607** | **0.00620920** | **0.00145075** | **0.0322266** | **0.198242** |

| Final full-INT8 direct comparison | PSNR dB | SSIM | RMS | Mean abs | P99 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| vs FP16 | 44.487529 | 0.998731370 | 0.00596518 | 0.00139603 | 0.0312500 | 0.179199 |
| vs G spatial INT8 | 43.589198 | 0.998470231 | 0.00661516 | 0.00153593 | 0.0356445 | 0.150879 |

Final-vs-canonical bias −0.00003572425; direct-vs-FP16 bias −0.00012386707.
Spatial direct-vs-FP16 luma-weighted border/interior means are
0.000937878 / 0.001136386; 16×16 block means span 0.000203312…0.004660086,
with worst blocks (y,x): (4,95), (4,103), (4,87), (4,71), (4,79). Full metrics
and canonical/FP16/previous reconstructed/difference PNGs are retained.

All **33 exported boundaries** match in fresh full-frame repeats, including
enc0's six and dec0's five internal diagnostics and all layer fields. Final
frame hash: `f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00`.
All tested layer fields, matrix outputs and frames are finite/unsaturated.

**The existing B6 45 dB image gate remains FAIL** for both new INT8 frames,
solely `native.psnr_peak1_db`. No threshold changed and no validated-quality
claim is made. Native FP16 rerun remains **47.051389 dB**, hash
`dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b`;
all its exported boundary hashes match the G FP16 control. B4 remains PASS,
99.9369%, max 7 ULP; B5 retains only documented enc1.merge (89.8379%, relative
RMS 0.02111), with expected conformance exit 1. **348 tests pass**, preserving
the 339-test baseline.

No designated matrix remains a floating-point exception in this full mode.
Explicit remaining floating-point work: quantization scale arithmetic and
correct-rounding assistance, dequantization, f16/E4M3 conversions, normalization,
original softmax, GELU, residual/bias additions, enc0 dither, dec0 nonlinear
epilogue, covariance/reconstruction filtering and blending. These are not
silently excluded neural matrices. No temporal work, calibration, performance
optimization, HIP, new backend/framework or proprietary artifact in Git.

## Failures and limitations retained

* Initial enc0 arithmetic/layout/repeat checks all passed, but the zero-message
  gate failed on a D3D12 warning about default buffers ignoring initial COPY_DEST.
  Explicit COMMON→COPY_DEST transitions fix it; verified rerun has zero messages.
* Initial attention build incorrectly selected SM 6.2 for dot4; DXC rejected it.
  Both new matrix entries now use the existing SM 6.4 build list. Two initial
  probe launches also exposed relative fixture paths; paths were resolved before
  the accepted representative run. No arithmetic gate was bypassed.
* The first representative report marked packed Q false because the CPU oracle
  returned widened storage bytes, though every code/scale/dot/output agreed.
  The expected physical buffer is now explicitly f16. A complete fresh pair
  rerun passes, rather than accepting a byte-layout mismatch.
* The first complete-chain run passed all arithmetic and repeated digests but
  failed while reporting coverage: the old inventory parser correctly rejected
  transformed graphs as baseline sources. The reporting function now follows
  the preserved canonical source-fixture pointer. Retained seed-1 digests were
  independently rechecked before accepting the recovered report; the report
  records this addendum. No shader or arithmetic result changed.

The scope remains one established reset frame/resolution/device/model. Hardware
edge tests supplement but do not prove arbitrary inputs, shapes or temporal
fidelity. End-to-end execution still has B6's file boundaries between major
stages. This closes **M09's matrix reconstruction gate**, not general quality,
temporal support, standalone deployment or M10's independent validation gate.

## Reproduction and evidence

Use fresh ignored output directories and several GiB free. In PowerShell set
`$T="$env:LOCALAPPDATA/Temp/opencode"`, `$L="$T/m09g-spatial-verified-20261030/layers"`,
`$F="$T/m-capture-run/fixtures"` and `$RGA="$T/rga-2.15/rga.exe"`.

```powershell
cmake --build build/native --config Release
python -m pytest tests -q
$env:M_STAGE_DEBUG='1'
python scripts/m09_complete.py --phase enc0 --enc0-fixture "$F/enc0" --out "$T/h-enc0"
python scripts/m_b6_frame.py --out "$T/h-enc0-frame" --enc0-mode int8-embedding `
  --cumulative-mode spatial --cumulative-layers $L `
  --fp16-frame "$T/m09g-fp16-final-20261030/downsample-native/results/out-out.bin" `
  --previous-frame "$T/m09g-spatial-frame-final-20261030/downsample-native/results/out-out.bin"
# B6 exit 1 is expected for the unchanged image gate; inspect execution separately.
python scripts/m09_complete.py --phase cpu --layers $L --frame "$T/h-enc0-frame" --out "$T/h-cpu"
python scripts/m09_complete.py --phase pair --layers $L --inputs "$T/h-cpu/results" --out "$T/h-pair"
python scripts/m09_complete.py --phase isa --frame "$T/h-enc0-frame" --inputs "$T/h-pair" --rga $RGA --out "$T/h-isa"
python scripts/m09_complete.py --phase edge --out "$T/h-edge"
python scripts/m09_complete.py --phase chain --layers $L --frame "$T/h-enc0-frame" --out "$T/h-chain"
python scripts/m_b6_frame.py --out "$T/h-frame" --enc0-mode int8-embedding --attention-mode int8 `
  --cumulative-mode spatial --cumulative-layers "$T/h-chain/layers" `
  --fp16-frame "$T/m09g-fp16-final-20261030/downsample-native/results/out-out.bin" `
  --previous-frame "$T/m09g-spatial-frame-final-20261030/downsample-native/results/out-out.bin"
python scripts/m09_complete.py --phase finalize --chain "$T/h-chain" --frame "$T/h-frame" `
  --isa "$T/h-isa" --out "$T/h-complete.json"
```

Actual accepted roots under T:

* `m09h-enc0-verified-20261030/{report.json,fixture,run0,run1}`.
* `m09h-enc0-frame-20261030/b6.json`, PNGs and raw frame.
* `m09h-attention-cpu-20261030/cpu-contract.json` and real Q/V inputs.
* `m09h-attention-pair-final-20261030/report.json`, replay graph and both runs.
* `m09h-attention-edge-20261030/report.json` and small reproducible GPU fixtures.
* `m09h-isa-final-20261030/{isa.json,*.isa,*.log}`: accepted inspection, gated on
  the pair's PASS report. (`m09h-isa-20261030` is the earlier inspection run.)
* `m09h-complete-chain-20261030/{chain,attention,learned,fields}.json`, fixtures
  and graph. Full-chain bulk diagnostic bins are pruned **only after** complete
  CPU checks and repeated digests; reports, input weights/graphs and logs remain.
* `m09h-complete-frame-20261030/b6.json`, raw frame, all output fields and PNGs.
* `m09h-complete-20261030.json`: initial CPU/graph/handoff/ISA/frame evidence ties;
  `m09h-complete-reviewed-20261030.json` is the accepted 17-check report adding
  current script/image-host/zero-debug and finite-unsaturated layer-field checks,
  without replacing the initial report.
* `m09h-fp16-20261030/b6.json` and `m09h-conformance-20261030.json`.

Initial failure roots (`m09h-enc0-20261030`, `m09h-attention-pair[-verified|-run]-20261030`)
retain their available diagnostics/fixture metadata; failures above are not
silently promoted to passing runs. All proprietary tensors remain outside Git.

## M10 preparation

M09 is complete for this established matrix inventory. M10 is **PREPARED /
PENDING**, not executed as a new milestone. Use the same complete graph and
finalizer for its complete-network resource/coverage validation; keep quality
FAIL distinct from execution PASS. Before claiming M10 completion, explicitly
resolve its persistent temporal-state checkpoint and the intended execution
scope. No temporal investigation is authorized or performed by this experiment.

## Addendum — M12 quantizer optimization (2026-10-30)

[Later milestone, recorded here because this report's `inputs.shaders` hashes
describe the pre-optimization build.] M12-D rewrote the shared activation
quantizer `swin_q8_quant_cs.hlsl` so four lanes cooperate on one token row
(`[numthreads(256,1,1)]`, same `ceil(Tokens/64)` groups). The compiled hash moved
from `316fcf87…e25c52` to `242aaa95…d79229`.

[M] The full INT8 frame remains **`f76ac032…4d6c00`**, the FP16 frame remains
`dc97c4cb…d6e95b`, the enc0 all-element CPU check passes 13/13, and the
representative attention pair passes 23/23 checks. The optimization is therefore
arithmetic-neutral for the executed graph. The complete M09 finalizer also passes
**17/17** on the optimized build, using the new frame and the historical
chain/ISA evidence. The host hashes in the tables above
(`1419ec59…`, `1e5f47ad…`) also describe the pre-instrumentation build; the M12
timed binaries are separate targets, and the accepted hosts were rebuilt from the
same sources for the optimization. See EXP-040 for the measured effect.
