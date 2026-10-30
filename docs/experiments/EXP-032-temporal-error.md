# EXP-032 — Temporal error propagation

**Milestone:** M10-C · **Status:** PASS for the measured reset-frame sequence;
cross-frame accumulation is **UNKNOWN** because no history consumer exists.
**Depends on:** [EXP-031](EXP-031-temporal-oracle.md),
[EXP-034](EXP-034-quality-baselines.md), [EXP-035](EXP-035-stage-error-attribution.md).
**Evidence:** `m10-sequence-20261030/sequence.json`.

## 1. Question

Over a sustained sequence, does INT8 error stay bounded, accumulate, oscillate,
or depend on content — and does it persist after a reset? The answer must not
assume that a per-frame measurement describes a recurrent loop.

## 2. Method

Twelve frames with identical inputs are executed through FP16 and INT8
(EXP-031). For each frame:

* frame PSNR / SSIM / RMS / mean and maximum absolute error are measured
  **INT8 against FP16 on the same input** (quantization cost);
* the three persistent tensors the graph writes are compared element by element
  (state divergence);
* non-finite and saturated samples are counted;
* the canonical frames are additionally compared with the captured reference
  (the only absolute gate).

No threshold is invented. The project's only absolute threshold is the unchanged
45 dB B6 gate on the canonical frame; same-input FP16↔INT8 numbers are
measurements, not acceptance criteria.

## 3. Per-frame result

| Frame | Scenario | PSNR dB | SSIM | RMS | Mean abs | Max abs |
|---:|---|---:|---:|---:|---:|---:|
| 0 | static | 44.4875 | 0.998731 | 0.005965 | 0.001396 | 0.1792 |
| 1 | hold | 44.4875 | 0.998731 | 0.005965 | 0.001396 | 0.1792 |
| 2 | pan-8 | 43.2951 | 0.998076 | 0.006843 | — | 0.1516 |
| 3 | pan-32v | 44.3336 | 0.998685 | 0.006072 | — | 0.1516 |
| 4 | rotate-3 | 44.7137 | 0.998628 | 0.005812 | — | 0.2351 |
| 5 | fast-96 | 44.3932 | 0.998668 | 0.006030 | — | 0.1870 |
| 6 | disocclude | 43.3024 | 0.998101 | 0.006837 | — | 0.1785 |
| 7 | repeat | **42.1228** | 0.997668 | 0.007832 | — | 0.1572 |
| 8 | thin | **44.7241** | 0.998675 | 0.005805 | — | 0.1690 |
| 9 | abrupt | 44.4104 | 0.998729 | 0.006018 | — | 0.1775 |
| 10 | recover | 44.4875 | 0.998731 | 0.005965 | 0.001396 | 0.1792 |
| 11 | static-end | 44.4875 | 0.998731 | 0.005965 | 0.001396 | 0.1792 |

[M] 0 non-finite and 0 saturated samples in every frame.

### Absolute gate (canonical frames only)

[M] Frames 0, 1, 10 and 11 are the canonical content. Each measures
**44.1393 dB against the captured reference** (INT8) and 47.0514 dB (FP16). The
unchanged 45 dB gate **FAILS** for INT8 and passes for FP16. Frames 2–9 have no
captured reference for their content, so no absolute number is claimed for them.

## 4. Accumulation

[M] The same canonical content at frames 0, 1, 10 and 11 produces **byte-identical
frames in both modes** (`dc97c4cb…` / `f76ac032…`). The abrupt frame 9 is
followed by frame 10 that matches frame 0 exactly, so reset recovery is complete:
nothing from frames 2–9 persists.

[M] Across the 12 frames the correlation between frame index and same-input PSNR
is **0.014** — no trend. The mixed-content frames span 42.12–44.72 dB (mean
43.91 dB), a 2.60 dB spread driven by content, not by position.

[M] **There is no accumulation mechanism in the executed graph.** No stage reads
the previous frame's state (EXP-030), so each frame is a pure function of its
input. Accumulation could only be tested after the `reset = 0` branches exist.

## 5. State divergence

The persistent tensors that a future temporal loop would consume, INT8 vs FP16,
over all 12 frames:

| State tensor | Elements | Differing | Relative RMS | Max abs |
|---|---:|---:|---:|---:|
| `post` history (f16) | 18,662,400 | 8,470,169 (45.4%) | 1.263–1.661% | 0.208–0.338 |
| `post` half surface (f32) | 4,665,600 | 908,488 (19.5%) | 1.061–1.439% | 0.125–0.245 |
| `dec0` recurrent feature (SNORM8) | 2,073,600 | 1,713,614 (82.7%) | 7.226–8.845% | 0.441–0.567 |

[M] `post` is the same FP16 shader in both modes; its state differs only because
its inputs (`dec0` outputs) differ. `dec0`'s feature is the most sensitive
persistent tensor: 82.7% of SNORM8 codes differ, with a decoded maximum of 0.567
(72 codes of 127).

[M] For identical content, the state hashes are identical across positions
(frame 0 and frame 11 checkpoints share all three hashes), so state divergence is
a deterministic function of the frame, not of history.

## 6. Classification

| Failure mode | Result |
|---|---|
| Remains bounded | **Yes** for every measured frame; no non-finite/saturated output |
| Accumulates | **Not possible in this graph**; no consumer. Unknown for `reset = 0` |
| Oscillates | No oscillation over identical content (byte-identical) |
| Scene-dependent | **Yes** — 2.60 dB spread across the same operators |
| Amplifies around disocclusion/movement | Not testable as motion; the pixel-patch proxy (frame 6) is 43.30 dB, within the content spread |
| Persists after reset | **No** — exact recovery at frame 10 |

## 7. Interpretation

* [M] INT8 error is bounded per frame and content-dependent; it does not drift
  across the evaluated sequence.
* [S][M] The absence of accumulation is structural, not a property of the
  quantizer: the graph has no cross-frame consumer.
* [M] The state tensors a recurrent loop would consume already differ by
  1.1–1.5% (post) and 7–9% (dec0 feature) at their own storage precision. That
  is the divergence a real temporal path would have to tolerate; whether it
  amplifies is **unknown**.
* [U] Temporal quality of a `reset = 0` loop.

## 8. Limitations

* Twelve frames, one scene family, reset only.
* The absolute 45 dB gate is evaluated only on canonical-content frames.
* No motion-vector-driven frame exists in the reconstruction.
