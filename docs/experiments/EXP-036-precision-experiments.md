# EXP-036 — Controlled precision experiments

**Milestone:** M11-C · **Status:** PASS for the investigation.
**All variants except the accepted contract are `diagnostic_only`**; none is
implemented on the GPU, and the 76/76 native INT8 coverage is unchanged.
**Inputs:** real `enc1` Q/V operands (`m09h-attention-cpu-20261030/results`),
real `enc1` Q weight/bias (`m09h-complete-chain-20261030/layers/enc1`).
**Command:**
`python scripts/m11_precision.py --out $T/m11-precision-20261030`

## 1. Question

Which single precision choice — scale granularity, clipping, or attention scale
scope — measurably reduces the error of the accepted INT8 contract on real
operands? The accepted contract is per-row activation scales, per-output-column
weight scales, symmetric signed INT8, exact INT32 reduction, f32 rescale.

## 2. Method

Each variant changes one thing; inputs, weights, bias and the f16 store are
identical. Error is `relative RMS = RMS(INT8 − f16 reference) / RMS(f16
reference)`; the reference is the same product in f64 rounded once to f16.

* **Learned GEMM** — real `Q` `[136640, 64]` × real `Q` weight `[64, 64]` + bias.
  The *pairing* is synthetic (the real Q input is the prepared token tensor), so
  the numbers compare schemes, not network quality.
* **QKᵀ** — 4×4 token windows, 32 channels, 2 heads, 4096 windows; per-row vs
  per-tensor Q scales.
* **PV** — the same windows; V scales local to the window vs one global scale per
  head. P is held in f64 so only V quantization contributes.

## 3. Results

### Learned GEMM (relative RMS)

| Variant | Relative RMS | vs accepted |
|---|---:|---:|
| **accepted: row activation / column weight** | **0.8072%** | — |
| activation per-tensor / column weight | 0.9951% | +23% |
| row activation / weight per-tensor | 0.9959% | +23% |
| activation and weight per-tensor | 1.1322% | +40% |
| accepted + activation clip p99.99 | 0.8128% | +0.7% |
| accepted + activation clip p99.9 | 0.8985% | +11% |
| accepted + activation clip p99.5 | 2.3450% | +190% |

### Attention (mean over 4096 windows/head, 2 heads)

| Product | Accepted scope | Alternative | Accepted | Alternative |
|---|---|---:|---:|
| QKᵀ | per-row Q | per-tensor Q | **0.1126%** | 0.1606% (+43%) |
| PV | window-local V column | global V column | **0.1482%** | 0.4004% (+170%) |

Worst-window values: QK 0.2311% / 0.3446%; PV 0.2375% / 0.5726%.

## 4. Interpretation

* [M] The accepted granularity is the best measured option at both levels.
  Coarsening either activation or weight scales increases the error by ~23% per
  axis and ~40% jointly on the real Q-like operand.
* [M] Max-based scaling beats every tested clipping percentile on this data:
  clipping buys a smaller step but loses more in the clipped tail. No clipping
  change is justified by this evidence.
* [M] Window-local V scales are **2.7×** better than a global per-head scale and
  the worst window is 2.4× better. The dynamic-attention contract's choice is
  quantitatively supported, not just conservative.
* [H] The remaining ~0.8% GEMM error is dominated by 8-bit resolution itself, not
  by scale granularity: even the best scheme leaves it. Recovering the 2.91 dB
  measured in EXP-034 through these knobs alone is therefore not supported by
  the data.
* [U] Network-level effect of any variant. These are per-operator measurements;
  no variant was propagated through the ten layers, and none may be promoted
  without a full CPU/GPU/ISA validation cycle.

## 5. Next

EXP-035's stage ranking and this experiment together bound M11-D: the accepted
contract is locally well chosen, so a quality gain needs either a different
representation (more bits, mixed precision) or upstream reconstruction changes —
not another scale tweak.

## 6. Limitations

* Synthetic Q×Q-weight pairing; uniform P in the PV probe.
* One layer's operands and weights; other layers' distributions differ.
* Diagnostic CPU arithmetic only; no GPU dispatch, so no ISA or resource claim.
