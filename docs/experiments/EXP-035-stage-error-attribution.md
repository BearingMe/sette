# EXP-035 — Stage-level error attribution

**Milestone:** M11-B · **Status:** PASS for the attribution.
**Depends on:** [EXP-025](EXP-025-m09-dec0-int8-frame.md) (fixed-input dec0
separation), [EXP-028](EXP-028-m09-cumulative-int8.md) (cumulative ablations),
[EXP-034](EXP-034-quality-baselines.md), [EXP-030](EXP-030-temporal-state-inventory.md).
**Data:** `m10-sequence-20261030/{fp16,int8}/frame-00..11`,
`m11-quality-20261030.json`. **Command:**
`python scripts/m11_quality.py --sequence $T/m10-sequence-20261030`

## 1. Question

Where does the INT8 execution differ from the FP16 oracle, and which differences
are quantization rather than the pre-existing reconstruction error? Aggregate
PSNR cannot answer this: EXP-034 showed FP16 and INT8 both have their own gap.

## 2. Method

Twelve frames with identical inputs are executed through the full FP16 graph and
the full INT8 graph. Every exported boundary is decoded in its own storage
domain (E4M3, UNORM8, SNORM8, f16, f32, packed 11-11-10) and compared element by
element. Two normalisations are reported:

* **relative RMS** = `RMS(diff) / RMS(reference)` — sensitive to near-zero
  tensors and therefore large for E4M3 fields;
* **RMS/peak** = `RMS(diff) / max|reference|` — comparable across tensors.

The boundaries are *cumulative*: an error at `dec1` includes everything upstream.
Fixed-input measurements from EXP-025 are used to separate dec0's own
quantization from propagated error; the cumulative ablations in EXP-028 separate
families. No tensor arithmetic was changed.

## 3. Results

Median over 12 frames, ordered by RMS/peak:

| Boundary | Storage | Relative RMS | RMS/peak | Peak | Max abs | Exact |
|---|---|---:|---:|---:|---:|---:|
| dec0.mix | UNORM8 | 20.80% | **8.28%** | 1.00 | 0.702 | 31.3% |
| chain.enc2.merge | E4M3 | 15.26% | **7.88%** | 6.50 | 3.250 | 22.1% |
| chain.tube-6 | E4M3 | 26.65% | **6.38%** | 7.50 | 3.000 | 10.7% |
| dec0.feature | SNORM8 | 8.26% | **5.48%** | 1.00 | 0.567 | 17.4% |
| chain.dec2 | E4M3 | 22.26% | **5.38%** | 13.00 | 4.500 | 12.6% |
| chain.tube-1 | E4M3 | 14.81% | 4.45% | 12.50 | 3.500 | 19.3% |
| chain.dec1 | E4M3 | 17.65% | 4.16% | 14.00 | 4.750 | 15.5% |
| chain.enc1 | E4M3 | 12.18% | 4.00% | 4.50 | 1.250 | 22.5% |
| chain.tube-2 | E4M3 | 15.76% | 3.97% | 14.00 | 3.000 | 17.6% |
| chain.tube-3 | E4M3 | 16.45% | 3.94% | 14.00 | 3.500 | 16.9% |
| chain.enc2 | E4M3 | 19.96% | 3.81% | 7.50 | 2.000 | 13.5% |
| chain.tube-4 | E4M3 | 15.60% | 3.80% | 15.00 | 3.063 | 17.9% |
| chain.enc1.merge | E4M3 | 12.55% | 3.46% | 5.50 | 1.250 | 22.2% |
| chain.tube-5 | E4M3 | 13.84% | 3.42% | 16.00 | 3.000 | 20.0% |
| dec0.covariance | f16 | 27.10% | 2.30% | 58.06 | 35.008 | 36.7% |
| dec0.gate | UNORM8 | 2.34% | 1.93% | 1.00 | 0.682 | 77.5% |
| post.packed | packed | 1.31% | 0.93% | 0.75 | 0.339 | 58.9% |
| post.history | f16 | 1.30% | 0.80% | 0.75 | 0.338 | 54.6% |
| enc0.tokens | E4M3 | 2.60% | 0.73% | 1.375 | 0.125 | 67.6% |
| post.half | f32 | 1.10% | 0.68% | 0.75 | 0.245 | 80.5% |
| downsample.frame | f16 | **0.76%** | **0.52%** | 1.00 | 0.235 | 59.5% |

[M] No non-finite value at any boundary in any frame; all 21 boundaries present
in both modes for all 12 frames.

### Per-frame final error (INT8 vs FP16, same input)

| Frame | Scenario | PSNR dB | SSIM | Max abs |
|---:|---|---:|---:|---:|
| 0 | static | 44.4875 | 0.998731 | 0.1792 |
| 1 | hold | 44.4875 | 0.998731 | 0.1792 |
| 2 | pan-8 | 43.2951 | 0.998076 | 0.1516 |
| 3 | pan-32v | 44.3336 | 0.998685 | 0.1516 |
| 4 | rotate-3 | 44.7137 | 0.998628 | 0.2351 |
| 5 | fast-96 | 44.3932 | 0.998668 | 0.1870 |
| 6 | disocclude | 43.3024 | 0.998101 | 0.1785 |
| 7 | repeat | **42.1228** | 0.997668 | 0.1572 |
| 8 | thin | **44.7241** | 0.998675 | 0.1690 |
| 9 | abrupt | 44.4104 | 0.998729 | 0.1775 |
| 10 | recover | 44.4875 | 0.998731 | 0.1792 |
| 11 | static-end | 44.4875 | 0.998731 | 0.1792 |

[M] Content moves the same-input INT8 penalty from 42.12 dB (repeated texture) to
44.72 dB (thin geometry): a 2.6 dB spread with no change in the operator set.

## 4. Separating quantization from reconstruction

[M] EXP-025 measured dec0 with a **fixed input**: its own INT8-vs-FP16 relative
RMS is only 0.25% / 1.96% / 0.86% / 2.61% for gate/mix/feature/covariance, and
the INT8 expansion error is 0.53%. The same surfaces in the full INT8 chain show
2.34% / 20.80% / 8.26% / 27.10%. **Most of the dec0 boundary difference is
propagated from upstream INT8 layers, not produced by dec0's own quantization.**

[M] The same holds upward: EXP-025 measured the *FP16* chain against the capture
at 1.86% / 15.32% / 5.91% / 16.82% for the same four surfaces. The mix and
covariance boundaries are already dominated by the FP16 reconstruction gap, so
their large full-INT8 numbers must not be read as quantization cost.

[M] Family-level ablations (EXP-028) measure the end-to-end effect directly:

| Cumulative INT8 family | Frame PSNR vs canonical |
|---|---:|
| Q/V | 44.4441 dB |
| + Wo | 44.8286 dB |
| + MLP | 44.2142 dB |
| + spatial (merge/expand/dec0) | 44.3779 dB |
| full (enc0 + attention, EXP-029) | 44.1393 dB |

The MLP and spatial families are the largest single additions; adding Wo alone
*improved* the canonical PSNR, which is why dB differences between different
graphs are not additive and are reported as measurements, not as an error budget.

## 5. Interpretation

* [M] The largest *relative* deviations are in the deepest decoder fields
  (`enc2.merge`, `tube-6`, `dec2`) and in dec0's nonlinear outputs.
* [M] The image stages attenuate almost all of it: the final frame is 0.52%
  RMS/peak, 0.76% relative RMS, 44.49 dB against FP16.
* [M] dec0's own INT8 quantization is small at a fixed input; its full-run
  difference is mostly inherited. Attribution must therefore use fixed-input
  measurements, not cumulative boundaries.
* [M] Error is strongly content-dependent: 2.6 dB spread across the same operator
  set. A single-frame quality number is not a property of the quantization.
* [H] Because the accepted scale scheme is already the best tested (EXP-036), the
  remaining error is dominated by 8-bit resolution and by the FP16 baseline, not
  by a fixable scale choice.

## 6. Next

EXP-036 tests the scale/clipping hypotheses directly. M11-D uses both to bound
what can be achieved inside the 76 operators.

## 7. Limitations

* Relative RMS on E4M3/UNORM8 tensors is inflated by small values; RMS/peak is
  reported alongside and used for ranking.
* Boundaries are cumulative, so the table is not a per-operator error budget.
* No leave-one-out graph was built; family attribution relies on the published
  cumulative ablations.
* One scene family and one device.
