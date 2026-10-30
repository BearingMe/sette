# INT8 quality gate: what the evidence supports

**Date:** 2026-10-30 · **Scope:** Preset M reset frame, 960×540→1920×1080,
RX 7700 XT/gfx1101, DLL 310.7.0.
**Experiments:** [EXP-034](../experiments/EXP-034-quality-baselines.md),
[EXP-035](../experiments/EXP-035-stage-error-attribution.md),
[EXP-036](../experiments/EXP-036-precision-experiments.md).

## Summary

The unchanged 45 dB gate **FAILS** for the INT8 reconstruction and **passes** for
the FP16 oracle. The measured evidence does not support closing the gap by
adjusting activation or weight scale granularity, clipping, or attention scale
scope: the accepted contract is already the best tested option, and the
surrounding image stages impose a 46.73 dB ceiling for this frame.

## Findings

* [M] Three distinct measurements, all reproduced from raw frames:
  reference↔FP16 **47.0514 dB**, reference↔INT8 **44.1393 dB**, FP16↔INT8
  **44.4875 dB**.
* [M] The image-stage control (perfect neural `dec1`) is **46.7260 dB**, so at
  most 1.73 dB of headroom exists between a perfect network and the 45 dB floor
  for this frame. The FP16 network scores 47.05 dB; INT8 loses **2.91 dB**
  relative to it.
* [M] Same-input INT8↔FP16 error is content-dependent: 42.12–44.72 dB across
  twelve content variants with the same operator set, correlation with frame
  index 0.014.
* [M] The accepted scale scheme (per-row activation, per-output-column weight)
  beats per-tensor activation (+23%), per-tensor weight (+23%), both (+40%),
  and every tested activation clipping percentile on real `enc1` operands.
* [M] Window-local V scales beat a global per-head scale by 2.7×; per-row Q
  scales beat per-tensor by 43%. The dynamic-attention contract is
  quantitatively supported.
* [M] Decoder boundary differences are mostly inherited: dec0's own fixed-input
  INT8 error is 0.25%/1.96%/0.86%/2.61% (gate/mix/feature/covariance), while the
  full INT8 chain shows 2.34%/20.80%/8.26%/27.10% on the same surfaces. The
  difference is upstream propagation plus the pre-existing FP16 gap (EXP-025).

## Interpretation

* [M] Within the 76 designated operators, the remaining error is dominated by
  8-bit resolution itself. No tested scale or clipping change closes 2.91 dB.
* [M] The absolute target cannot be attributed to quantization alone: the FP16
  baseline and the 46.73 dB image-stage ceiling bound what any matrix change can
  buy on this frame.
* [H] Closing the absolute gate would require one of: a higher-precision or
  mixed representation for selected tensors, reconstruction changes outside the
  76 matrix operators, or a separately justified target. None is implemented.
* The 45 dB gate is **not** lowered or redefined. INT8 quality remains FAIL.

## Limitations and unknowns

* One frame family, one device, one build.
* Stage attribution is cumulative; fixed-input separations exist only where a
  prior experiment measured them (dec0, EXP-025).
* The controlled experiments use real `enc1` operands with a synthetic Q×Q-weight
  pairing and uniform P; they compare schemes, not network quality.
* No independent static validation set exists, so absolute fidelity beyond the
  canonical frame is `INCONCLUSIVE` by the B6 contract.

## References

* [EXP-029](../experiments/EXP-029-m09-complete-int8.md) — accepted INT8 frame.
* [EXP-034](../experiments/EXP-034-quality-baselines.md) — measurement contract.
* [EXP-035](../experiments/EXP-035-stage-error-attribution.md) — boundary ranking.
* [EXP-036](../experiments/EXP-036-precision-experiments.md) — controlled variants.
* `m11-quality-20261030.json`, `m11-precision-20261030/precision.json`.

## Addendum — asymmetric activation encoding (2026-10-30, EXP-041)

**Outcome:** the 45 dB gate still **FAILS**, but full INT8 quality improves from
**44.1393 dB to 44.4026 dB** (+0.263) with **76/76 operators and
34,518,597,632 MACs unchanged**, at ~2.9% higher GPU time. The change is
asymmetric per-row encoding (true min/max, offset) for the two strongly
one-sided operands: post-GELU → MLP-down and softmax P → PV. The offset is
applied in the consumer epilogue; the signed `dot4` reduction is unchanged.

* [M] Operator output error: MLP-down 3.19→1.99% (enc1), 1.75→1.23% (dec1);
  PV 0.541→0.258% (enc1), 0.657→0.316% (dec1).
* [M] Bit-exact CPU/GPU validation on enc1 and dec1; signed `dot4` verified by
  RGA; 0 FP16 matrix instructions.
* [M] Across nine content variants the mean improves 43.98→44.24 dB and the
  worst case 42.12→43.13 dB; two variants regress by 0.21/0.25 dB.
* [M] Selective FP16 was re-measured on top: `enc0` FP16 + asym reaches
  **44.7652 dB** (75/76 operators, 96.9% MACs, +9.7% time) and improves all nine
  frames; attention FP16 + asym reaches 44.8119 dB on the canonical frame but is
  worse across content (mean 44.34, min 42.93) and 36% slower, so it is rejected.
* [M] Per-K-block weight scaling was measured and rejected: representation error
  falls 34× but the GEMM output only 11.6%, for a new accumulation contract.

**Recommendation:** adopt `full INT8 + asym` as the quality-first full-INT8
configuration and `enc0` FP16 + asym as the best measured quality/cost mixed
configuration. Do not lower the 45 dB gate; the remaining 0.24–0.60 dB needs a
different representation (more bits, residual correction, or reconstruction
changes outside the 76 operators), not another scale tweak.

See [EXP-041](../experiments/EXP-041-quality-recovery.md).
