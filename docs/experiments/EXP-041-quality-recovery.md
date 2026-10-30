# EXP-041 — Quality recovery: asymmetric activation encoding and selective precision

**Milestone:** M12/M13 · **Status:** PASS for the investigation. The accepted
**opt-in** full-INT8 configuration adds **asymmetric activation encoding** for
the two non-negative operands (post-GELU → MLP-down, softmax P → PV). It raises
the canonical frame from **44.1393 dB to 44.4026 dB** with **76/76 operators and
34,518,597,632 MACs still executing native INT8**, at **+2.9% GPU time**.
The unchanged 45 dB gate still **FAILS**. Mixed FP16 configurations are measured
separately; the best one on the canonical frame reaches 44.8119 dB at 56/76
operators, but is not the best across content.
**Depends on:** [EXP-034](EXP-034-quality-baselines.md),
[EXP-035](EXP-035-stage-error-attribution.md),
[EXP-036](EXP-036-precision-experiments.md),
[EXP-038](EXP-038-dispatch-profile.md), [EXP-040](EXP-040-optimization-results.md).
**Evidence:** `m13-sweep1-20261030/m13-quality.json` ...
`m13-sweep4-20261030/m13-quality.json`,
`m13-quant-probe2-20261030/quant-probe.json`,
`m13-asym-validate-20261030/validation.json`,
`m13-perf-20261030/m13-perf.json`,
`m13-sequence-20261030/m13-sequence.json`,
`m13-sequence2-20261030/m13-sequence.json`,
`m13-gate-asym/b6.json`, `m13-gate-enc0/b6.json`,
`m13-m09-finalize.json`.

## 1. Question and acceptance criteria

Can the cumulative INT8 image be improved without expanding the model, changing
the schedule, redefining the 45 dB gate, or giving up native INT8 matrix
execution? The order is fixed: reproduce the baseline, rank measured
sensitivity, test bounded quantization changes, then test selective FP16 only
where INT8 remains inadequate. Every configuration must report its own operator
and MAC coverage; a mixed configuration is never called 100% INT8.

Acceptance for this experiment: reproducible measurements on identical inputs,
an independent CPU check of any new arithmetic, preserved accepted full-INT8 and
FP16 paths, preserved regression suite, and an explicit PASS/FAIL for the
quality gate.

## 2. Baseline reproduced (Q1)

[M] The canonical frame was re-executed and re-measured from raw frames; the
accepted values reproduce exactly (EXP-034):

| Comparison | PSNR | SSIM | Notes |
|---|---:|---:|---|
| reference ↔ FP16 | 47.0514 dB | 0.999259 | image-stage ceiling 46.7260 dB |
| reference ↔ full INT8 | 44.1393 dB | 0.998633 | gate **FAIL** |
| FP16 ↔ full INT8 | 44.4875 dB | — | quantization cost 2.91 dB |

[M] Worst same-input boundaries (EXP-035): `dec0.mix` 8.28% RMS/peak,
`enc2.merge` 7.88%, `tube-6` 6.38%, `dec0.feature` 5.48%, `dec2` 5.38%. The
image stages attenuate nearly all of it: the final frame is 0.52% RMS/peak.

[M] On a real enc1 Q product, activation quantization alone contributes 0.596%
relative RMS, weight quantization 0.533%, and the accepted pair 0.807%. The
accepted per-column weight representation has 18.66% median element error but
only 0.53% output error because K averages it.

[M] Accepted GPU comparison (EXP-038): FP16 600.124 ms, full INT8 205.167 ms,
**2.93×**. This experiment's interleaved block measured 195.628 ms for the same
full-INT8 frame, so all deltas below are against **this block's** baseline, not
against the earlier absolute number.

[M] Existing precision exceptions are unchanged: `post` (Gaussian reconstruction
and covariance) and `downsample` are floating point; `enc0`/`dec0` keep their
validated INT8 image paths; attention is INT8 in the full mode.

## 3. Candidate ranking (Q2)

Selective FP16 restoration was measured one family at a time over all ten layers
(EXP-041 sweep 1). This is a direct measurement of the final frame, not an
assumption from tensor RMS:

| Restored family | Operators restored | Δ PSNR vs full INT8 | vs FP16 |
|---|---:|---:|---:|
| `expand` (dec1, dec2) | 2 | **+0.1636** | 44.4545 |
| `merge` (enc1, enc2) | 2 | **+0.1060** | 44.7582 |
| `attention` | 20 | +0.0727 | 44.0743 |
| `dec0` image stage | 1 | +0.0140 | 44.4845 |
| `wo` | 10 | −0.0393 | 44.3321 |
| `q` | 10 | −0.3361 | 44.0788 |
| `mlp` | 20 | −0.4446 | 43.9079 |
| `v` | 10 | −1.1435 | 43.4202 |
| `enc0` image stage | 1 | **+0.1925** | 44.6662 |

[M] The ranking is not the boundary ranking: restoring the numerically largest
boundary (`mlp`) makes the image **worse**, and restoring V is catastrophic. The
INT8 errors partially cancel the pre-existing reconstruction error, so
"more FP16" is not monotone in final PSNR. Per-layer restores are also
non-additive: `dec1:expand` alone is −0.20 dB and `dec2:expand` alone is
+0.10 dB, while both together are +0.16 dB.

**Selected candidates (at most three):**

1. **Asymmetric activation encoding** for post-GELU and softmax P, because the
   operator-level evidence is strong and it keeps 76/76 INT8.
2. **Selective FP16 for the image `enc0` stage**, the best single mixed restore
   (+0.19 dB for one operator), to be re-evaluated on top of (1).
3. **Per-K-block weight scaling**, measured but rejected: representation error
   falls 34× while the GEMM output improves only 12% (below).

## 4. Quantization refinement (Q3)

All probes are CPU arithmetic on real tensors; no accepted shader changed in
this step.

### 4.1 Per-K-block weight scaling

[M] 54 learned-weight operators, exact f32 matrices from the accepted fixtures:

| Weight encoding | Relative RMS (median) | Output error on enc1 Q product |
|---|---:|---:|
| per output column (accepted) | 18.66% | 0.8072% |
| per 32-element K block | **0.545%** | 0.7525% |
| per 64-element K block | 0.605% | — |
| per tensor | 1.027% | — |

[M] Per-32-block activation scales (instead of per-row) reduce the same product
to 0.7411%; both block changes together reach 0.7139%, an **11.6%** improvement.
That is not enough to justify a new accumulation contract, extra scale traffic
and a full revalidation, so block scaling is **rejected**.

### 4.2 Asymmetric activation encoding

[M] Real post-GELU (`i8_g`) and probability (`att_prob`) tensors from an executed
chain, measured against the accepted symmetric per-row scheme:

| Operand | Symmetric | Asymmetric (min/max) | Change |
|---|---:|---:|---:|
| MLP-down enc1 output | 3.194% | **1.987%** | −37.8% |
| MLP-down dec1 output | 1.749% | **1.230%** | −29.7% |
| PV enc1 output | 0.5413% | **0.2579%** | −52.3% |
| PV dec1 output | 0.6568% | **0.3158%** | −51.9% |

[M] A simple "clip at zero" unsigned encoding is **not** acceptable for GELU
(16.8% output error: GELU has a real negative tail). The implemented scheme uses
the true per-row minimum and maximum. For probabilities the two are equivalent
(0.258% vs 0.266%), but one contract covers both.

[M] These two operands were selected because they are the only large matrices
whose input distribution is strongly one-sided; Q/V/Wo/norm inputs are
zero-centred and were not changed.

## 5. Implemented asymmetric path (Q4)

Three new shaders, used only by the opt-in asymmetric configuration:

| Shader | Contract |
|---|---|
| `swin_q8_quant_asym_cs.hlsl` | 4 lanes/row; per-row min and max; `scale=(max−min)/255`; `offset=min+128·scale`; signed codes `round((v−min)/scale)−128` clamped to [−128,127] |
| `swin_mlp_down_q8_asym_cs.hlsl` | unchanged `dot4` reduction; `y=((acc·scale + offset·Σw) · wscale) + residual`; `Σw` precomputed per output column from the same weight codes |
| `swin_attention_pv_asym_cs.hlsl` | unchanged P·V `dot4` plus one `dot4` against ones for `Σv`; `y=((acc·ps + po·Σv) · vs)` |

[M] **Independent CPU validation, element by element** (enc1 and dec1):
quantizer codes/scales/offsets bit-exact; offline weight-column sums bit-exact;
int32 accumulators bit-exact; f32 epilogues bit-exact; f16 stores bit-exact;
0 non-finite and 0 saturated values. See
`m13-asym-validate-20261030/validation.json`.

[M] RGA gfx1101 inspection of the executed blobs:
`swin_mlp_down_q8_asym.cso` 16 `v_dot4_i32_i8`, `swin_attention_pv_asym.cso`
256 `v_dot4_i32_i8`, 0 FP16 matrix instructions in either; the quantizer has
neither (it is not a matrix product).

[M] Full-frame effect, canonical content:

| Configuration | PSNR vs reference | Δ vs full INT8 | SSIM | Operators INT8 | MACs INT8 |
|---|---:|---:|---:|---:|---:|
| full INT8 (accepted) | 44.1393 | — | 0.998633 | **76/76** | **34,518,597,632** |
| full INT8 + asymmetric | **44.4026** | **+0.2634** | **0.998700** | **76/76** | **34,518,597,632** |
| FP16 oracle | 47.0514 | +2.912 | 0.999259 | 0/76 | 0 |

[M] The asymmetric path is opt-in: the accepted full-INT8 graph and its matrix
shaders are unchanged, and the accepted frame still hashes to `f76ac032…`; the
host source only gained opt-in timing/selective support and the M09 finalizer
still passes 17/17, so the M09/M12 evidence remains valid.

## 6. Selective higher precision on top of the asymmetric path (Q4)

All runs use identical inputs and the same measurement code. A restore replaces
the named family with its original FP16 stage; every other matrix op remains the
validated INT8 path.

| Configuration | PSNR vs ref | Δ | vs FP16 | SSIM | INT8 ops | INT8 MACs | MAC share |
|---|---:|---:|---:|---:|---:|---:|---:|
| full INT8 + asym | 44.4026 | +0.263 | 44.5065 | 0.998700 | 76/76 | 34.519 B | 100% |
| + `enc0` FP16 | **44.7652** | **+0.626** | 44.9001 | 0.998811 | 75/76 | 33.449 B | 96.9% |
| + `attention` FP16 | **44.8119** | **+0.673** | 44.6857 | 0.998767 | 56/76 | 33.472 B | 97.0% |
| + `attention` + `enc0` FP16 | 44.7753 | +0.636 | 45.1132 | 0.998802 | 55/76 | 32.402 B | 93.9% |
| + `merge` + `enc0` FP16 | 44.1824 | +0.043 | 44.4878 | 0.998680 | 73/76 | 32.171 B | 93.2% |
| + `expand` + `enc0` FP16 | 44.7728 | +0.633 | 45.0098 | 0.998803 | 73/76 | 32.246 B | 93.4% |
| + all four spatial restores + `enc0` | 44.3140 | +0.175 | 44.6854 | 0.998712 | 71/76 | 30.967 B | 89.7% |

[M] Restoring *all* attention matrices to FP16 is the highest-quality measured
configuration, but it removes 20 of 76 operators and costs 36% GPU time. Adding
`enc0` to it **lowers** PSNR (44.775 vs 44.812), another measured non-additivity.
`enc0` + asym is the best quality-per-cost mixed point: +0.626 dB for one
restored operator and ~10% time.

[M] No mixed configuration is reported as full INT8. The `56/76` row retains
97.0% of MACs only because attention's dynamic products are small; operator
coverage is 73.7%.

## 7. Quality versus performance (Q5)

Interleaved D3D12-timestamp runs, 3 rounds per configuration, one warm-up
(`m13-perf-20261030/m13-perf.json`). Each configuration reproduced one frame hash
across all rounds.

| Configuration | GPU median (ms) | vs full INT8 | PSNR vs ref | Δ quality | INT8 ops | INT8 MACs |
|---|---:|---:|---:|---:|---:|---:|
| full INT8 | 195.628 (192.2–210.1) | 1.000× | 44.1393 | — | 76/76 | 34.519 B |
| **full INT8 + asym** | **201.241** (198.0–202.9) | **0.972×** (−2.9%) | **44.4026** | **+0.263** | **76/76** | **34.519 B** |
| `enc0` FP16 + asym | 214.555 (211.2–214.8) | 0.912× (−9.7%) | 44.7652 | +0.626 | 75/76 | 33.449 B |
| `attention` FP16 + asym | 266.434 (261.1–274.7) | 0.734× (−36.2%) | 44.8119 | +0.673 | 56/76 | 33.472 B |
| `merge` FP16 + asym | 211.978 (186.4–214.8) | 0.923× (−8.4%) | 44.3009 | +0.162 | 74/76 | 33.240 B |

[M] The asymmetric chain costs about **5.6 ms** (2.9%) because it adds a second
dot4 to PV and one epilogue multiply-add to MLP-down; the quantizer's min
reduction is already paid back by the same 4-lane structure.

**Pareto front:** `full INT8 + asym` for maximum INT8 coverage and performance;
`enc0` FP16 + asym for the best measured quality per unit of lost coverage/time.
`attention` FP16 + asym is a quality ceiling, not a recommended operating point.

## 8. Multi-frame robustness (Q6)

The canonical frame is not evidence of general quality, so the same
configurations were run on the M10 content variants (identical inputs and FP16
oracle per frame). Absolute reference PSNR exists only for canonical content;
all frames have a direct configuration-vs-FP16 measurement.

| Frame | Content | full INT8 vs FP16 | + asym vs FP16 | Δ | `enc0` FP16 + asym |
|---:|---|---:|---:|---:|---:|
| 0 | static | 44.4875 | 44.5065 | +0.019 | 44.9001 |
| 2 | pan-8 | 43.2951 | 44.0958 | +0.801 | 43.7987 |
| 3 | pan-32v | 44.3336 | 44.1228 | −0.211 | 44.7115 |
| 4 | rotate-3 | 44.7137 | 44.8446 | +0.131 | 45.3894 |
| 5 | fast-96 | 44.3932 | 44.4472 | +0.054 | 44.8473 |
| 6 | disocclude | 43.3024 | 44.0433 | +0.741 | 43.7921 |
| 7 | repeat | 42.1228 | 43.1275 | **+1.005** | 43.2663 |
| 8 | thin | 44.7241 | 44.8479 | +0.124 | 45.1577 |
| 9 | abrupt | 44.4104 | 44.1652 | −0.245 | 45.1002 |

[M] Across the nine measured variants, full INT8 spans **42.12–44.72 dB** (mean
43.98); asymmetric spans **43.13–44.85 dB** (mean **44.24**). The mean improves
by **+0.27 dB**, the worst case by **+1.00 dB**, and the spread narrows from
2.60 dB to 1.72 dB. Two variants lose 0.21 and 0.25 dB, so the change is not
uniformly better; it is better on average and materially better on the worst
content.

[M] Adding `enc0` FP16 on top of asymmetric improves **every one of the nine**
frames (mean **44.55 dB**, +0.31 over asymmetric and +0.58 over full INT8), with
the worst case at 43.27 dB and the best at 45.39 dB. This is the strongest
multi-frame evidence in the experiment, but it costs one matrix operator and
about 10% GPU time.

[M] Restoring all attention matrices, despite the best canonical number, is
**not** more robust: across the same nine frames it averages **44.34 dB** with a
minimum of **42.93 dB**, both worse than `enc0` FP16 + asym, at 36% GPU time.
It is therefore not recommended; the canonical-frame ranking alone would have
selected the wrong configuration.

## 9. Validation gates

| Check | Result |
|---|---|
| Asymmetric quantizer + MLP-down + PV, CPU vs GPU | **PASS**, bit-exact on enc1 and dec1 (codes, scales, offsets, column sums, int32, f32, f16) |
| Finite / unsaturated | 0 non-finite, 0 saturated at every checked boundary |
| RGA gfx1101 | MLP-down 16 `dot4`, PV 256 `dot4`, 0 FP16 matrix instructions |
| `full INT8 + asym` full frame, fresh repeat | **PASS** execution, deterministic, 33 boundary hashes, `25a89fc0?` |
| `enc0` FP16 + asym full frame, fresh repeat | **PASS** execution, deterministic, 27 boundary hashes (FP16 `enc0` declares fewer diagnostics), `092a77a1?` |
| Accepted full INT8 frame | **`f76ac032?4d6c00` unchanged**, execution PASS, deterministic |
| M09 complete-network finalizer | **17/17 PASS** on the current build |
| Regression suite | **377 passed** (370 before this experiment) |

[M] The deterministic frame hashes above are the evidence; the `execution PASS`
label means every declared stage ran, every output was complete, and the
fresh-repeat hashes matched. The image regression gate remains FAIL for every
INT8 configuration, which is reported separately from execution.

## 10. Gate and interpretation

* [M] **The 45 dB gate remains FAIL for every measured configuration.** Full
  INT8 + asym is 44.4026; the best mixed configuration is 44.8119. No threshold
  was changed.
* [M] **INT8 coverage is preserved** by the accepted change: 76/76 operators,
  34,518,597,632 MACs, signed dot4 in the executed blobs, no FP16 matrix
  instruction. Mixed configurations report 75/76 or 56/76 explicitly.
* [M] The asymmetric change is systematic, not a lucky restore: it reduces the
  operator error of the two non-negative operands by 30–52% and improves the
  mean and worst-case content. The remaining gap to 45 dB is not closed by
  scale granularity, clipping or these two operands.
* [M] Selective FP16 is **not monotone**: some restores improve, several degrade,
  and combinations interact. Any proposed mixed configuration must be measured
  end to end; per-stage error is not a proxy for image quality.
* [U] Whether asymmetric encoding on zero-centred operands (Q/V/Wo/up) helps;
  block-scaled weights combined with asymmetric activations; and whether a
  second precision tier (INT8 + residual) can close the remaining 0.19–0.60 dB.
* **No temporal claim.** `reset = 0` reuse remains blocked; this is static quality
  refinement only.
* **Recommended configuration.** For INT8-first use, `full INT8 + asym`:
  44.4026 dB, 76/76 operators, full MACs, +2.9% time. For quality-first use,
  `enc0` FP16 + asym: 44.7652 dB on the canonical frame and 44.55 dB mean over
  nine content variants, 75/76 operators, 96.9% MACs, +9.7% time. Do not adopt
  the attention restore: it is the canonical-frame maximum but slower and less
  robust.
* **Next independently testable task.** Measure the headroom of a second 8-bit
  residual tier (or 9-10-bit activation) for MLP-down and PV in the existing CPU
  probe before implementing anything. The block-scale experiment already showed
  that finer *weights* buy little; if a second activation tier does not project
  more than roughly 0.3 dB, the remaining gap needs a reconstruction change
  outside the 76 matrix operators rather than another quantizer variant. Keep
  the same CPU/GPU bit-exact gate and the 9-frame robustness check.

## 11. Reproduction

```powershell
cmake --build build/native --config Release

# 1. quantization probes (real weights/activations, CPU only)
python scripts/m13_quant_probe.py --out "$T/m13-quant-probe2-20261030"

# 2. build and audit the asymmetric configuration
python scripts/m13_selective.py --int8 "$T/m09h-complete-chain-20261030/layers" `
  --fp16 "$T/m-capture-run/layers" --out "$T/m13-asym" --asym

# 3. element-by-element GPU/CPU validation + RGA
python scripts/m13_validate_asym.py --layers "$T/m13-asym" --out "$T/m13-asym-validate"

# 4. quality and coverage sweep
python scripts/m13_quality.py --out "$T/m13-sweep" `
  --configs "full-int8,full-int8+asym,enc0-fp16+asym,family:attention+asym"

# 5. interleaved GPU timing
python scripts/m13_perf.py --out "$T/m13-perf" --rounds 3 `
  --configs "full-int8,full-int8+asym,enc0-fp16+asym,family:attention+asym"

# 6. multi-frame robustness
python scripts/m13_sequence.py --out "$T/m13-seq" `
  --configs "full-int8,full-int8+asym,enc0-fp16+asym,family:attention+asym" `
  --frames 0,2,3,4,5,6,7,8,9

python -m pytest tests -q        # 377 passed
```

## 12. Limitations

* One device, driver and DLL build; one scene family. The 9-frame content sweep
  is a robustness check, not a perceptual study.
* The per-frame gains are content-dependent and two variants regress slightly.
* Asymmetric arithmetic was validated bit-exactly on one encoder and one decoder
  layer (enc1, dec1); the other layers use the same shaders and shapes but were
  not each re-derived on CPU.
* Performance is timestamp-based; clock/power state is not controlled, so the
  interleaved block's 195.6 ms baseline is used for every delta.
* No hardware counters, no peak VRAM, no independent validation scene.
* The asymmetric path is an opt-in configuration; the accepted full-INT8 default
  and its hashes are preserved.
