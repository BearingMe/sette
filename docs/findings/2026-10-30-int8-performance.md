# Native INT8 performance on gfx1101: measured result

**Date:** 2026-10-30 · **Scope:** complete Preset M reset frame, 960×540 render
→ 1920×1080, RX 7700 XT (gfx1101), driver 32.0.32015.7029, D3D12.
**Experiments:** [EXP-037](../experiments/EXP-037-profiling-environment.md),
[EXP-038](../experiments/EXP-038-dispatch-profile.md),
[EXP-039](../experiments/EXP-039-bottleneck-analysis.md),
[EXP-040](../experiments/EXP-040-optimization-results.md).

## Summary

The native INT8 frame is **2.93× faster** than the FP16 frame on the same device,
inputs and build: **600.124 ms → 205.167 ms** median GPU time, measured with
D3D12 timestamps and interleaved repeats. The final frames are byte-identical to
the accepted M09 outputs (`f76ac032…` INT8, `dc97c4cb…` FP16), so the speedup is
not bought with a numerical change.

## Findings

* [M] **End-to-end:** FP16 600.124 ms (585.4–610.6) vs INT8 205.167 ms
  (202.9–209.0); **2.93×**; interleaved 5 repeats, 1 warm-up, all frame hashes
  equal.
* [M] **Neural chain:** 556.5 → 183.5 ms (**3.03×**).
* [M] **Front/decoder:** enc0 15.0 → 3.2 ms (4.68×); dec0 11.4 → 1.5 ms (7.60×).
* [M] **Unchanged stages:** `post` 14.8 ms in both modes; `downsample` ~1.7 ms.
  These are 8.0% of the INT8 frame.
* [M] **Dispatches:** 98 (FP16) vs 243 (INT8); the INT8 path is faster despite
  2.5× more dispatches.
* [M] **One optimization accepted:** 4-lane-per-row activation quantizer
  (`swin_q8_quant_cs.hlsl`), bit-exact, ~29% cheaper quantizer family, ~7%
  end-to-end improvement.
* [M] **Quality unchanged:** INT8 44.1393 dB vs the canonical reference,
  44.4875 dB vs FP16; the 45 dB gate still **FAILS**. Performance does not
  change the quality verdict.
* [M] **Memory:** the INT8 chain declares 680 GPU buffers / 4.39 GB total,
  largest 139.9 MB; peak resident VRAM was **not measured**.
* [H] The quantizers and adapters are latency/occupancy-bound, not
  bandwidth-bound: declared-traffic effective bandwidth is ~10–22 GB/s for the
  large quantizers and ~34 GB/s average, versus ~150 GB/s in a small compute
  stage. No hardware counter confirms this.

## Interpretation

* [M] Native `dot4` execution provides a real, reproducible end-to-end benefit
  on this device and frame, not merely instruction presence.
* [M] The benefit survives an arithmetic-neutral optimization and a
  byte-identical output requirement.
* [S] The remaining adapters (quantize/pack, ~22% of the chain) and the shared
  FP16 `post` (7.2%) are the next performance levers; fusing quantize into the
  GEMM load has the largest measured ceiling.
* Performance and quality are separate outcomes: **execution PASS, performance
  FASTER, quality FAIL**.

## Limitations

* One device, driver, DLL build, resolution and scene family.
* Timestamps add `EndQuery` overhead; both modes are instrumented equally.
* Clock/power state is uncontrolled; variability is reported, not eliminated.
* No hardware counters, so utilization claims remain inferences.
* The 2.93× is GPU execution time for the evaluated frame, not a claim about
  game latency, other content or other GPUs.

## References

* [EXP-037](../experiments/EXP-037-profiling-environment.md)
* [EXP-038](../experiments/EXP-038-dispatch-profile.md)
* [EXP-039](../experiments/EXP-039-bottleneck-analysis.md)
* [EXP-040](../experiments/EXP-040-optimization-results.md)
* `m12-interleaved-20261030/m12-profile.json`
