# EXP-038 — Dispatch-level GPU profile

**Milestone:** M12-B · **Status:** PASS for reproducible per-dispatch timing.
**Depends on:** [EXP-037](EXP-037-profiling-environment.md).
**Data:** `m12-interleaved-20261030/m12-profile.json` (accepted comparison),
`m12-final-20261030/m12-profile.json` (sequential corroboration),
`m12-ab-*` (A/B, EXP-040).

## 1. Question

What is the GPU duration of every dispatch in the complete frame, and how do
FP16 and native INT8 compare under the same measurement method?

## 2. Method

One warm-up frame per mode, then **5 measured repeats**, each a fresh set of
processes. FP16 and INT8 are **interleaved** (FP16, INT8, FP16, …) so clock/power
drift cannot favour one mode. Every repeat's final frame must equal the accepted
hash. Timestamps surround every dispatch (EXP-037); a run with any hash mismatch
is discarded. The table reports the **median of per-run totals**, plus min/max;
per-dispatch medians are used for the stage ranking.

## 3. Whole-frame result

| | FP16 | Full INT8 |
|---|---:|---:|
| Dispatches | **98** | **243** |
| GPU median | **600.124 ms** | **205.167 ms** |
| GPU min / max | 585.385 / 610.649 | 202.876 / 209.041 |
| Frame hashes across repeats | 1 (`dc97c4cb…`) | 1 (`f76ac032…`) |

[M] **Speedup: 2.93×** (600.1 / 205.2 ms) for the same frame and inputs.
A non-interleaved profile of the same build measured 593.5 / 208.9 ms (2.84×),
so the result is stable across scheduling orders.

The 243 dispatches are the 234-dispatch neural graph plus 9 image-stage
dispatches (enc0 prefix 4, dec0 prefix 3, post 1, downsample 1).

### Per process

| Process | FP16 ms (min–max) | INT8 ms (min–max) | Speedup |
|---|---:|---:|---:|
| `enc0` | 15.010 (14.887–15.195) | 3.205 (3.189–3.310) | 4.68× |
| `chain` (neural) | 556.540 (542.704–566.313) | 183.517 (181.619–187.765) | 3.03× |
| `dec0` | 11.381 (11.038–11.780) | 1.497 (1.444–1.633) | 7.60× |
| `post` | 14.755 (14.632–14.927) | 14.802 (14.692–14.926) | 1.00× |
| `downsample` | 2.028 (1.657–3.176) | 1.716 (1.655–2.239) | 1.18× |

[M] `post` and `downsample` are precision-independent, as expected: they are the
same FP16 shaders in both modes. They are 8.0% of the optimized INT8 frame.

## 4. Slowest dispatches

FP16 (all fused stages):

| Dispatch | ms |
|---|---:|
| `enc1.mlp` | 50.61 |
| `dec1.mlp` | 48.38 |
| `enc2.mlp` | 34.93 |
| `dec2.mlp` | 29.38 |
| `enc1.attention` | 27.68 |
| `dec1.attention` | 27.51 |

INT8 (after the EXP-040 optimization):

| Dispatch | ms |
|---|---:|
| `post` main | 14.80 |
| `enc1.norm2` | 6.88 |
| `enc1.mlp-gelu` | 4.91 |
| `dec1.mlp-gelu` | 4.90 |
| `enc1.attention-pv` | 4.77 |
| `dec1.norm2` | 4.70 |

[M] The INT8 profile has no 20 ms-class neural dispatch: the largest neural
dispatch is 6.9 ms, and the largest single dispatch in the frame is now the
shared FP16 `post` stage (14.8 ms).

## 5. Variability

[M] FP16 totals span 585–611 ms (±2.2% around the median); INT8 spans 203–209 ms
(±1.5%). In the interleaved run both modes are measured under the same drift,
which is why it is the accepted comparison. Earlier sequential runs varied more
(FP16 563–648 ms).

[M] No run produced a non-finite or saturated frame, and every run reproduced its
accepted frame hash.

## 6. Interpretation

* [M] Native INT8 makes the neural chain ~3.03× faster and the decoder expansion
  ~7.6× faster; the end-to-end frame is ~2.93× faster.
* [M] The INT8 advantage is not "more dispatches": INT8 uses 2.5× as many
  dispatches (243 vs 98) and still finishes in a third of the time.
* [M] The unchanged FP16 image stages become a larger share of the INT8 frame;
  `post` alone is 7.2%.
* [U] Why FP16 varies more; clock/power state was not recorded.

## 7. Limitations

* Timestamps include `EndQuery` overhead and possible serialization; both modes
  are instrumented identically.
* Five repeats per mode; no statistical power analysis.
* One device, driver, build and resolution.
