# EXP-007: INT8 Quantization Feasibility on Real Preset M Weights

**Milestone:** M07
**Status:** PASS for weight/activation characterization, with limitations

## Question

Does Preset M's information survive an INT8 weight representation, and which
scaling granularity preserves it best? Acceptance criteria, defined before
execution:

1. Convert the real captured weights to symmetric INT8 at several granularities
   and measure the error against the exact f16 tensors.
2. Report where the outliers are, not just an average.
3. Propagate the conversion through a matrix product using real captured
   activations.
4. Record activation ranges, since they set the INT8 activation budget later.

## Method

`scripts/m07_int8_weight_quant.py`:

- Weights are decoded through d4r's `plain_layer()`, which undoes NGX's blocked
  e4m3 layout into plain `W[K][N]` f16 matrices. e4m3 → f16 is lossless
  (EXP-006), so this step contributes no error.
- Each matrix is quantized with symmetric INT8 (`|W|max/127`, round to nearest)
  at four scopes: whole tensor, per output channel, per 16-row block, per
  32-row block.
- Error is relative Frobenius norm against the exact f16 matrix; the worst matrix
  and a per-channel scale-inflation ratio (`max / median` channel magnitude) show
  outlier sensitivity.
- Propagation uses the first layer's real captured input tokens for
  `enc1.q0`: `x @ W` in float64 against `x @ quantize(W)`. Accumulation is exact
  here on purpose — this isolates *weight* quantization, which is what M07 asks;
  activation quantization and k32 rounding belong to M09/M11.

```powershell
python scripts/m07_int8_weight_quant.py $env:TEMP/opencode/m-capture-run/capture6
```

## Results

### Weight error by scope (all ten layers, 312 matrices, 122 880 values)

| Scope | RMS relative error | Worst single matrix |
|---|---|---|
| per tensor | 0.0094 | 0.0148 |
| per output channel | **0.0062** | **0.0074** |
| per 16-row block | 0.0083 | 0.0121 |
| per 32-row block | 0.0088 | 0.0132 |

[M] Per-channel scaling is the best of the four on this data: 34% lower RMS
error than per-tensor. Blocking at 16/32 rows is *worse* than per-channel even
though it gives more scales, because neighbouring rows in a block have very
different column magnitudes — the block maximum is dragged up by a single row.

[M] Worst cases are the MLP up-projections (`w1`), whose matrix maximum reaches
2.0 while typical values are far smaller; `w1_5/w1_6` of `enc1` show the largest
per-tensor error (0.0135/0.0123) and the largest channel inflation (3.4x).

### Outlier structure

[M] Per-channel scale inflation (`max/median channel magnitude`) ranges from
1.29 to 3.37 across matrices. Nothing extreme (no 100x outlier channel) appears,
which is why a symmetric per-channel INT8 representation is viable at all.

### Activation ranges

| Layer | Shape | absmax | RMS | peak/RMS |
|---|---|---:|---:|---:|
| enc1 | 80x128x64 | 1.50 | 0.384 | 3.90 |
| enc2 | 40x64x96 | 5.50 | 1.512 | 3.64 |
| enc3_tube (6 launches) | 20x32x128 | 6.5-16.0 | 3.34-3.85 | 1.95-4.32 |
| dec2 | 20x32x128 | 7.00 | 1.760 | 3.98 |
| dec1 | 40x64x96 | 13.00 | 3.137 | 4.14 |

[M] Peak/RMS stays below 4.4 for the whole network, so a single per-tensor
activation scale costs about one bit of headroom rather than a catastrophic
clipping decision. The tube's range grows monotonically (6.5 → 16.0) across the
six blocks, which is expected for a residual stack.

### Propagation

[M] `enc1.q0` with the real captured tokens (16 384 tokens, K=64, N=32):

| Scope | Output relative error |
|---|---|
| per tensor | 0.0129 |
| per output channel | 0.0097 |
| per 16-row block | 0.0133 |
| per 32-row block | 0.0131 |

[S] Weight quantization alone contributes ~1% relative error to a projection
output, and per-channel scaling helps the output too. Whether ~1% per projection
is acceptable end-to-end is an M11 question; this experiment only establishes
that the representation carries the information.

## Consequences for the mixed-precision policy

- Symmetric **per-output-channel INT8 weights** for `q`, `v`, `wo`, `w1`, `w2`,
  `merge_w`, `expand_w`.
- Activations: single per-tensor scale per operand is adequate given peak/RMS
  < 4.4, but this must be re-checked per layer in M11 because the tube range
  grows across blocks.
- No operation in the layer can tolerate an unbounded accumulator: the `K <=
  131 071` guard from EXP-005 applies.
- RMS norm, softmax and e4m3 requantization boundaries stay floating point; they
  are not quantization targets.

## Limitations and Unknowns

- [U] Asymmetric (zero-point) quantization was **not** evaluated; only symmetric
  scaling was. The checkpoint list includes it, so it moves to M12 with the rest
  of the refinement work.
- [U] Clipping strategies were not swept: with a max-based scale there is no
  clipping by construction, and no optimal-clipping search was done.
- [U] Propagation is one projection, not a full block: the error does not yet
  include activation quantization, per-k32 rounding, or the residual paths.
- [M] Statistics come from one frame (preset 13, 1280x720) of one capture. Weight
  statistics are frame-independent, but activation ranges are not.

## Files

- `scripts/m07_int8_weight_quant.py` (`--json` for machine-readable output)

## References

- [EXP-006](EXP-006-m06-operator-contract.md) (e4m3 losslessness)
- [EXP-005](EXP-005-m08-native-int8-probes.md) (accumulator bound)
- [M04 numerics](../findings/2026-10-28-preset-m-numerics.md)
