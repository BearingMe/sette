# EXP-006: Operator-Level Probes (M06)

**Milestone:** M06
**Status:** PASS (with documented limitations)

## Question

Can Preset M's individual operations be reproduced independently, and what
numerical error does each primitive introduce relative to an oracle that is not
derived from itself? Acceptance criteria, defined before execution:

1. Each representative primitive has a reproducible test whose expectation comes
   from the specification or exact float64 mathematics.
2. Each test states the measured error, not just pass/fail.
3. A whole layer's mathematics can be driven from captured tensors and compared
   with the GPU's own output.

## Method

Two layers of evidence:

- **Primitive contract tests** — `tests/test_m06_operator_contract.py`, 12 tests
  in five groups. The oracle is the OCP e4m3 bit definition, brute-force nearest
  code search over the full f16 range, and exact float64 `erf`/`exp`/`sqrt`
  mathematics. d4r's primitives are imported read-only from the pinned
  reference, never re-implemented to match themselves.
- **Layer-level drive** — `scripts/m06_layer_model_check.py` runs d4r's
  independent numpy Preset M layer model against ten real layer launches. The
  capture stores post-launch allocations under `gpu-output/`, which this script
  maps onto the arena bytes the model expects (d4r's own script expects the
  Linux bridge's `dump/launch-*` snapshots instead).

Environment: capture `capture6` (310.7.0, preset 13, 1280x720 render extent,
shifts alternating `(2,2)`/`(0,0)`).

```powershell
python -m unittest tests.test_m06_operator_contract -v
python scripts/m06_layer_model_check.py $env:TEMP/opencode/m-capture-run/capture6
```

## Results

### e4m3 contract

| Check | Result |
|---|---|
| table matches the OCP bit definition, all 256 codes | identical, NaN included |
| every finite e4m3 value is exactly representable in f16 | yes (778 values) |
| round trip equals brute-force nearest code, ties to even, satfinite | identical over 1026 inputs |
| `600.0` and `-600.0` saturate to `±448` rather than infinity | yes |

[M] The "e4m3 → f16 is lossless" assumption in the numerics finding is confirmed
directly, not inferred.

### Accumulation contract

| Check | Measured |
|---|---|
| one 32-deep step rounds the exact sum to f16 once | max relative error `4.9e-4` ≤ `2^-11` |
| four chained steps (K=128) | max relative error ≤ `8 · 2^-11` |

[M] The rounding is real (the exact-match test fails by construction) and bounded
by one f16 ulp per step, which is what M04 records.

### RMS norm contract

[M] With the contract's `2^-13` epsilon and one f16 rsqrt:

- max relative error against exact `x/||x||` is `4.1e-4` (`< 4 · 2^-11`);
- the epsilon changes the scale by less than 1% at typical magnitudes, so it is a
  numerical floor, not a calibration term.

### GELU contract

[M] The clamped cubic is an approximation, and the tests pin its actual error:

- max absolute deviation from exact `erf`-GELU is **0.046** (worst at |z| ≈ 2);
- it is **not monotone**: the response decreases on `z ∈ [-2.0, -0.58]`;
- outside `|z| = 2` it is linear because the cubic input is clamped.

[S] This is now a known fidelity characteristic of the reference, not a defect in
the reconstruction: any reconstructed GELU must reproduce the approximation,
including its non-monotone region, to match the oracle.

### Attention softmax

The exponent trick (bit-shift of the f16 bits plus `0x7FF88000`) is validated by
properties only:

| Property | Result |
|---|---|
| rows finite, non-negative, sum to 1 (±2e-3) over σ = 0.5/2/8 score scales | pass |
| ranking preserved: larger score ⇒ larger weight | > 90% of orderings |
| extreme inputs (`±100`) stay finite and normalised | pass |

- [U] **Open:** the trick carries its own score-scale convention
  (`C_SCALE = 0.0171814`). A direct comparison against `softmax(s)` is therefore
  not an independent oracle, and the absolute convention was **not** validated
  here. Measured behaviour is `weight ∝ exp(0.65·s)` for the raw scores fed to
  `m_model.softmax_rows`, which is consistent inside d4r's model but unexplained
  by the sources read so far. Resolving it needs the PTX-level inspection of the
  attention mixer that M11 is set up to do.

### Whole-layer drive against the capture

```text
enc1       128x80x64  shift 2,2:  36.9 dB, identical 65.2%, max|d| 0.5625; merged 35.7 dB, 48.0%
enc2        64x40x96  shift 2,2:  42.6 dB, identical 61.4%, max|d| 0.5;    merged 23.1 dB, 51.2%
enc3_tube   32x20x128 shift 2,2:  39.5 dB, identical 67.8%, max|d| 1
enc3_tube   32x20x128 shift 0,0:  42.0 dB, identical 70.3%, max|d| 1
enc3_tube   32x20x128 shift 2,2:  42.9 dB, identical 70.0%, max|d| 1
enc3_tube   32x20x128 shift 0,0:  41.4 dB, identical 67.8%, max|d| 1
enc3_tube   32x20x128 shift 2,2:  42.1 dB, identical 69.0%, max|d| 1
enc3_tube   32x20x128 shift 0,0:  34.9 dB, identical 43.2%, max|d| 1.031
dec2        64x40x96  shift 0,0:  37.2 dB, identical 48.2%, max|d| 1
dec1       128x80x64  shift 0,0:  38.9 dB, identical 58.8%, max|d| 2
```

[S] The numpy model agrees with the GPU to 34.9-42.9 dB with 43-70% of values
bit-identical. The residual has a documented cause: `m_model.block()` accumulates
each matrix product exactly, while the hardware performs one f16 rounding per
k32 step (`model_enc3.mma_chain`). The model is therefore a *close* independent
reimplementation, not a bit-exact one, and the gap is attributed to rounding
order — the same conclusion EXP-003 reached from the other direction.

## Limitations and Unknowns

- [U] The bit-exact layer reproduction with `mma_chain` substituted for exact
  sums was **not** run; that requires editing the reference model, which this
  checkpoint did not do. It is the natural first step for M09/M11.
- [U] The attention scale convention above.
- Residual streams, patch-merge geometry and window shifts are exercised only
  through the whole-layer comparison, not as isolated operators.
- [M] `merged` agreement for `enc2` is much lower (23.1 dB) than its output
  (42.6 dB); the merge path is the weakest link in this comparison and is a
  candidate for a dedicated probe.

## Files

- `tests/test_m06_operator_contract.py`
- `scripts/m06_layer_model_check.py` (`--json` for machine-readable output)

## References

- [M04 numerics](../findings/2026-10-28-preset-m-numerics.md)
- [EXP-003](EXP-003-m05-layer-capture.md) (capture provenance)
