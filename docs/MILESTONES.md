# Milestones

## Overview

This is the authoritative progress dashboard for the DLSS 4.5 Preset M INT8 reconstruction. Milestones define outcomes, checkpoints define independently verifiable work, and validation gates determine completion. Keep status here; use OpenCode TODOs only for short-lived execution tasks.

### Status

- `NOT STARTED`: Work has not begun.
- `IN PROGRESS`: Work is underway.
- `COMPLETE`: The validation gate passed.
- `BLOCKED`: A required dependency is unavailable.

### Validation

- `PENDING`: Validation has not been performed.
- `PASS`: Acceptance criteria were satisfied with supporting evidence.
- `FAIL`: Acceptance criteria were not satisfied.
- `BLOCKED`: Validation cannot proceed because a required dependency is unavailable.
- `INCONCLUSIVE`: Evidence is insufficient to establish a result.

A milestone is `COMPLETE` only when its gate is `PASS`. Validation evidence and evidence-classification rules are defined in [`CONVENTIONS.md`](CONVENTIONS.md). A research milestone may pass with a negative result if its stated objective was to investigate that question.

## Progress

| ID | Milestone | Status | Validation |
|---|---|---|---|
| M01 | Project Initialization | IN PROGRESS | PENDING |
| M02 | Reference Acquisition | IN PROGRESS | PENDING |
| M03 | Preset M Architecture Reconstruction | NOT STARTED | PENDING |
| M04 | Numerical Contract Analysis | NOT STARTED | PENDING |
| M05 | FP16 Oracle Reproduction | NOT STARTED | PENDING |
| M06 | Operator-Level Reconstruction Probes | NOT STARTED | PENDING |
| M07 | INT8 Quantization Feasibility | NOT STARTED | PENDING |
| M08 | RDNA 3 Native INT8 Probes | NOT STARTED | PENDING |
| M09 | INT8 Kernel Reconstruction | NOT STARTED | PENDING |
| M10 | Complete Network Execution | NOT STARTED | PENDING |
| M11 | Differential Error Mapping | NOT STARTED | PENDING |
| M12 | Weight Quantization Refinement | NOT STARTED | PENDING |
| M13 | Activation and Temporal Refinement | NOT STARTED | PENDING |
| M14 | Performance Baseline | NOT STARTED | PENDING |
| M15 | RDNA 3 Optimization | NOT STARTED | PENDING |
| M16 | Final Quality Gate | NOT STARTED | PENDING |

M01 remains in progress: environment requirements and current inventory are recorded, but exact compatible toolchain versions and a reproducible reference build/run path are not established. M02 is in progress after reference acquisition was authorized; d4r, quattro, and DLSS are pinned, while source inspection and remaining acquisition are pending. No milestone after M02 has started.

---

# PHASE I: RECONNAISSANCE

## M01 — Project Initialization

**Status:** IN PROGRESS / PENDING

### Objective

Establish the exact research target, environment, scope, and references.

### Checkpoints

- [x] C01: Define project goals and exclusions in `README.md`.
- [x] C02: Identify the target GPU, OS, API, and shader toolchain; distinguish target requirements from locally observed availability.
- [ ] C03: Identify candidate Preset M model versions and artifacts.
- [x] C04: Establish the repository organization using Trace without changing generic `AGENTS.md` rules.
- [x] C05: Document initial setup requirements and current tool inventory; exact compatible versions and build instructions remain open.
- [x] C06: Record 1920×1080 output and Quality as provisional test assumptions, not yet fixed settings.
- [x] C07: Record unavailable or unverified requirements: DXC is not on PATH; D3D12 feature support and reference/model access are unverified.

### Validation Gate

- [x] Project scope and exclusions are explicit.
- [ ] Setup requirements and the reference execution path are reproducible with verified, compatible versions.
- [x] Important unknowns, provisional assumptions, and current environment limitations are recorded.

### Expected Deliverables

- `README.md`
- Updated `docs/MILESTONES.md`
- Initial `references/README.md` index

### Dependencies

- None.

---

## M02 — Reference Acquisition

### Objective

Acquire and pin the primary sources required for reconstruction.

### Checkpoints

- [ ] C01: Inspect d4r and identify relevant code paths.
- [ ] C02: Inspect quattro's validated RDNA 3 experiments.
- [x] C03: Record commit IDs and origin URLs for acquired d4r/quattro/DLSS submodules; remaining references are explicitly pending or incomplete.
- [ ] C04: Identify model and weight artifacts.
- [ ] C05: Locate conservative FP16 WMMA kernels.
- [ ] C06: Identify standalone inference entry points.
- [x] C07: Record local DLL metadata, hash, signature observation, and provenance/access limitations.
- [ ] C08: Identify source or model artifacts that remain unavailable.

### Validation Gate

All accessible references are pinned, indexed, and traceable; inaccessible or missing material is explicitly documented.

### Expected Deliverables

- Updated `references/README.md` and, where appropriate, pinned read-only sources under `references/sources/`.
- Relevant provenance/access findings under `docs/findings/`.

### Dependencies

- M01.

---

## M03 — Preset M Architecture Reconstruction

### Objective

Understand the complete neural execution graph.

### Checkpoints

- [ ] C01: Identify all major network stages.
- [ ] C02: Inventory operators and matrix multiplication sites.
- [ ] C03: Determine tensor shapes and layouts.
- [ ] C04: Identify attention, normalization, and residual operations.
- [ ] C05: Trace patch merging and reconstruction stages.
- [ ] C06: Map intermediate buffers and execution dependencies.
- [ ] C07: Identify temporal inputs, outputs, and persistent state.
- [ ] C08: Record original precision and scaling where observable.
- [ ] C09: Distinguish architectural facts from inferred behavior and gaps.

### Validation Gate

A traceable model graph and tensor contracts exist, with unresolved architectural gaps explicitly identified and claims supported by evidence.

### Expected Deliverables

- Architecture findings and graph/operator inventory in `docs/findings/`.

### Dependencies

- M02.

---

## M04 — Numerical Contract Analysis

### Objective

Understand the numerical behavior of the reference implementation.

### Checkpoints

- [ ] C01: Trace FP8 weight decoding.
- [ ] C02: Identify FP16 conversion behavior.
- [ ] C03: Identify accumulator types and rounding behavior.
- [ ] C04: Locate intermediate requantization boundaries.
- [ ] C05: Analyze normalization and softmax precision.
- [ ] C06: Determine output conversion and clipping behavior.
- [ ] C07: Identify potentially sensitive residual paths.
- [ ] C08: Compare conservative and optimized execution modes.

### Validation Gate

Important numerical operations have documented contracts supported by source observations or reproducible experiments; uncertainties are retained.

### Expected Deliverables

- Numerical-fidelity findings and reproducible probes in `docs/findings/` and `docs/experiments/`.

### Dependencies

- M02 and M03.

---

# PHASE II: ORACLE AND EXPERIMENTAL PROBING

## M05 — FP16 Oracle Reproduction

### Objective

Establish a deterministic reference execution path.

### Checkpoints

- [ ] C01: Build or execute d4r's relevant standalone engine.
- [ ] C02: Pin the model and source revision.
- [ ] C03: Establish one fixed output resolution and quality setting supported by the reference path.
- [ ] C04: Prepare deterministic inputs.
- [ ] C05: Capture final outputs.
- [ ] C06: Capture intermediate tensors where accessible.
- [ ] C07: Establish repeatability and tolerance requirements.
- [ ] C08: Identify nondeterministic behavior, if any.

### Validation Gate

Reference execution is reproducible, and captured outputs are suitable for differential testing with repeatability/tolerance criteria recorded.

### Expected Deliverables

- Oracle fixtures, execution procedures, and experiment reports under `tests/` and `docs/experiments/`.

### Dependencies

- M02, M03, and M04.

---

## M06 — Operator-Level Reconstruction Probes

### Objective

Determine how individual neural operations can be reproduced independently.

### Checkpoints

- [ ] C01: Inventory representative operator implementations.
- [ ] C02: Reconstruct CPU equivalents where feasible.
- [ ] C03: Test matrix multiplication behavior.
- [ ] C04: Test normalization.
- [ ] C05: Test attention calculations.
- [ ] C06: Test residual and activation operations.
- [ ] C07: Test tensor packing and layout conversions.
- [ ] C08: Compare against independent expectations and reference outputs.

### Validation Gate

Representative operators have reproducible tests and known numerical error characteristics derived from independent expectations or the pinned reference.

### Expected Deliverables

- Operator probes, CPU references, tests, and supporting experiment reports.

### Dependencies

- M03, M04, and M05.

---

## M07 — INT8 Quantization Feasibility

### Objective

Determine whether Preset M can retain meaningful numerical information under INT8.

### Checkpoints

- [ ] C01: Implement a CPU quantization simulator.
- [ ] C02: Investigate FP8-to-INT8 weight conversion.
- [ ] C03: Test symmetric and asymmetric representations.
- [ ] C04: Compare per-tensor, per-channel, and block scaling.
- [ ] C05: Investigate activation ranges.
- [ ] C06: Test clipping and saturation.
- [ ] C07: Measure error propagation through representative blocks.
- [ ] C08: Identify precision-sensitive operations.
- [ ] C09: Establish an initial mixed-precision policy.

### Validation Gate

Quantization behavior and limitations are characterized with reproducible numerical evidence and independent expectations.

### Expected Deliverables

- Quantization simulator, experiments, tests, and findings.

### Dependencies

- M04 and M06.

---

## M08 — RDNA 3 Native INT8 Probes

### Objective

Validate the hardware execution primitives needed by the reconstruction.

### Checkpoints

- [ ] C01: Reproduce relevant quattro INT8 experiments.
- [ ] C02: Verify native WMMA instruction generation.
- [ ] C03: Validate signedness and operand packing.
- [ ] C04: Test INT32 accumulation.
- [ ] C05: Validate CPU/GPU numerical equivalence.
- [ ] C06: Evaluate representative matrix dimensions.
- [ ] C07: Compare WMMA and DP4A implementations.
- [ ] C08: Identify layout, padding, and occupancy constraints.
- [ ] C09: Record compiler, driver, and hardware dependencies.

### Validation Gate

Native INT8 execution is demonstrated with independent numerical checks and inspected GPU ISA on the target hardware.

### Expected Deliverables

- GPU probes, validation tests, ISA evidence, and hardware findings.

### Dependencies

- M02 and M01.

---

# PHASE III: RECONSTRUCTION

## M09 — INT8 Kernel Reconstruction

### Objective

Implement the neural matrix operations required by Preset M.

### Checkpoints

- [ ] C01: Implement INT8 matrix kernels.
- [ ] C02: Support required tensor layouts.
- [ ] C03: Implement quantization/dequantization boundaries.
- [ ] C04: Support necessary floating-point operations and document each exception.
- [ ] C05: Validate representative layers.
- [ ] C06: Compare outputs against CPU INT8 references.
- [ ] C07: Inspect generated GPU ISA.
- [ ] C08: Measure numerical error before optimization.

### Validation Gate

Representative reconstructed layers execute correctly on gfx1101 against independent CPU INT8 references, and generated ISA confirms the intended native matrix arithmetic.

### Expected Deliverables

- Initial kernels and operator-level regression tests in `src/` and `tests/`.

### Dependencies

- M06, M07, and M08.

---

## M10 — Complete Network Execution

### Objective

Execute the entire reconstructed Preset M network.

### Checkpoints

- [ ] C01: Assemble the complete execution graph.
- [ ] C02: Load converted weights.
- [ ] C03: Allocate required buffers.
- [ ] C04: Implement dispatch dependencies.
- [ ] C05: Handle persistent temporal state.
- [ ] C06: Execute all required neural stages.
- [ ] C07: Produce complete output frames.
- [ ] C08: Verify finite outputs and buffer safety.
- [ ] C09: Detect omitted layers and accidental precision fallbacks.

### Validation Gate

A complete frame is produced, with designated matrix operations using native INT8 arithmetic. Image quality is not an acceptance requirement; execution coverage, buffer safety, and precision paths must be evidenced.

### Expected Deliverables

- Standalone inference implementation and end-to-end tests.

### Dependencies

- M03, M05, and M09.

---

# PHASE IV: NUMERICAL FIDELITY

## M11 — Differential Error Mapping

### Objective

Identify where INT8 reconstruction diverges from the FP16 oracle.

### Checkpoints

- [ ] C01: Compare intermediate tensors.
- [ ] C02: Measure absolute and relative errors.
- [ ] C03: Characterize distribution and saturation.
- [ ] C04: Identify first significant divergence points.
- [ ] C05: Track errors across residual connections.
- [ ] C06: Investigate temporal error accumulation.
- [ ] C07: Distinguish quantization errors from implementation defects.

### Validation Gate

Major numerical divergence sources are identified through reproducible comparisons against the established independent FP16 oracle.

### Expected Deliverables

- Error maps, differential tests, and findings.

### Dependencies

- M05 and M10.

---

## M12 — Weight Quantization Refinement

### Objective

Improve numerical fidelity through better weight representations.

### Checkpoints

- [ ] C01: Evaluate multiple scaling granularities.
- [ ] C02: Investigate outlier sensitivity.
- [ ] C03: Evaluate clipping strategies.
- [ ] C04: Test selective higher-precision weights.
- [ ] C05: Measure layer-level error changes.
- [ ] C06: Measure network-level effects.
- [ ] C07: Document storage and conversion costs.

### Validation Gate

Selected changes measurably improve fidelity on fixed validation inputs; changes and comparisons are reproducible.

### Expected Deliverables

- Quantization experiments and validated conversion procedures.

### Dependencies

- M07 and M11.

---

## M13 — Activation and Temporal Refinement

### Objective

Improve activation fidelity and temporal stability.

### Checkpoints

- [ ] C01: Analyze activation distributions.
- [ ] C02: Calibrate scaling and clipping.
- [ ] C03: Investigate quantization boundaries.
- [ ] C04: Evaluate normalization precision.
- [ ] C05: Investigate residual accumulation.
- [ ] C06: Test temporal state stability.
- [ ] C07: Compare static and moving sequences.
- [ ] C08: Identify necessary mixed-precision exceptions.

### Validation Gate

Selected changes improve measured numerical or temporal fidelity without concealing unsupported behavior or changing the oracle.

### Expected Deliverables

- Calibration procedures, temporal tests, and findings.

### Dependencies

- M11 and M12.

---

# PHASE V: PERFORMANCE

## M14 — Performance Baseline

### Objective

Establish representative end-to-end performance measurements.

### Checkpoints

- [ ] C01: Define repeatable benchmarking conditions.
- [ ] C02: Measure conservative FP16 reference latency.
- [ ] C03: Measure optimized FP16 reference latency.
- [ ] C04: Measure INT8 inference latency.
- [ ] C05: Separate kernel and dispatch overhead.
- [ ] C06: Measure memory and conversion costs.
- [ ] C07: Record VRAM consumption.
- [ ] C08: Inspect workload-dependent bottlenecks.
- [ ] C09: Control warmup, synchronization, and GPU validation overhead.

### Validation Gate

Baseline comparisons are reproducible and use equivalent workloads, stated measurement conditions, and clearly separated reference paths.

### Expected Deliverables

- Performance experiments and profiling findings.

### Dependencies

- M05 and M10.

---

## M15 — RDNA 3 Optimization

### Objective

Improve native INT8 performance without exceeding fidelity limits.

### Checkpoints

- [ ] C01: Identify dominant execution bottlenecks.
- [ ] C02: Optimize WMMA tile selection.
- [ ] C03: Compare DP4A alternatives.
- [ ] C04: Improve memory coalescing and LDS usage.
- [ ] C05: Reduce redundant conversions.
- [ ] C06: Investigate kernel fusion.
- [ ] C07: Improve buffer lifetime and reuse.
- [ ] C08: Reduce dispatch and synchronization costs.
- [ ] C09: Evaluate wave occupancy and register pressure.
- [ ] C10: Measure complete-frame improvements.

### Validation Gate

Optimizations provide reproducible end-to-end benefit without violating established correctness/fidelity limits; generated ISA and workload/measurement conditions are recorded.

### Expected Deliverables

- Optimized kernels, performance tests, and findings.

### Dependencies

- M08, M10, M11, and M14.

---

# PHASE VI: FINAL VALIDATION

## M16 — Final Quality Gate

### Objective

Determine whether the reconstructed INT8 implementation provides a meaningful alternative to FP16 on gfx1101.

### Checkpoints

- [ ] C01: Reproduce major experimental results.
- [ ] C02: Validate complete network execution.
- [ ] C03: Evaluate static image quality.
- [ ] C04: Evaluate temporal stability.
- [ ] C05: Compare numerical fidelity.
- [ ] C06: Compare complete inference latency.
- [ ] C07: Compare VRAM consumption.
- [ ] C08: Check stability across representative workloads.
- [ ] C09: Identify remaining unsupported behavior.
- [ ] C10: Review architectural and numerical claims.
- [ ] C11: Document performance regressions and limitations.

### Validation Gate

The final report accurately characterizes correctness, fidelity, performance, and reproducibility. A negative performance outcome is acceptable; a failed reconstruction must not be presented as successful.

### Expected Deliverables

- Final reproducibility report, implementation, tests, and consolidated findings.

### Dependencies

- M10 through M15.
