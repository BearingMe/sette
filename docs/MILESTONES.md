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
| M01 | Project Initialization | COMPLETE | PASS |
| M02 | Reference Acquisition | COMPLETE | PASS |
| M03 | Preset M Architecture Reconstruction | COMPLETE | PASS |
| M04 | Numerical Contract Analysis | COMPLETE | PASS |
| M05 | FP16 Oracle Reproduction | BLOCKED | BLOCKED |
| M06 | Operator-Level Reconstruction Probes | COMPLETE | PASS |
| M07 | INT8 Quantization Feasibility | COMPLETE | PASS |
| M08 | RDNA 3 Native INT8 Probes | COMPLETE | PASS |
| M09 | INT8 Kernel Reconstruction | NOT STARTED | PENDING |
| M10 | Complete Network Execution | NOT STARTED | PENDING |
| M11 | Differential Error Mapping | NOT STARTED | PENDING |
| M12 | Weight Quantization Refinement | NOT STARTED | PENDING |
| M13 | Activation and Temporal Refinement | NOT STARTED | PENDING |
| M14 | Performance Baseline | NOT STARTED | PENDING |
| M15 | RDNA 3 Optimization | NOT STARTED | PENDING |
| M16 | Final Quality Gate | NOT STARTED | PENDING |

Phase II: M06-M08 pass, M05 is capture-complete but oracle-blocked. Native INT8 wave-matrix execution is verified on gfx1101 with matching ISA ([EXP-005](experiments/EXP-005-m08-native-int8-probes.md)), Preset M's operators have contract tests with independent oracles ([EXP-006](experiments/EXP-006-m06-operator-contract.md)), and the **complete** captured weight set was quantized to INT8 at five scaling scopes ([EXP-009](experiments/EXP-009-m07-complete-inventory.md)): 360 matrices / 1 553 408 values, with per-column 16-row K blocks best (0.0048 RMS) and the activation term now co-dominant. M05's C10 is done: a rebuilt ZLUDA captures `enc0`, `dec0`, post and downsample at 1920x1080 ([EXP-008](experiments/EXP-008-m05-full-capture.md)). Its remaining blocker is an execution path for the full FP16 oracle — d4r's frame oracle is Linux/Vulkan-only and `compile_m.py` wants the `mvlo` variant the probe does not emit — so M05 stays BLOCKED at C11 while the ten-layer regression remains the layer-scope oracle ([EXP-003](experiments/EXP-003-m05-layer-capture.md)). M09 is unblocked by M06-M08 and has not started. Phase I evidence: the Windows/D3D12/DXC toolchain and pinned quattro reference ([EXP-001](experiments/EXP-001-reference-toolchain.md)); the Preset M graph and numerical contract are in [`findings/2026-10-28-preset-m-architecture.md`](findings/2026-10-28-preset-m-architecture.md) and [`findings/2026-10-28-preset-m-numerics.md`](findings/2026-10-28-preset-m-numerics.md).

---

# PHASE I: RECONNAISSANCE

## M01 — Project Initialization

**Status:** COMPLETE / PASS

### Objective

Establish the exact research target, environment, scope, and references.

### Checkpoints

- [x] C01: Define project goals and exclusions in `README.md`.
- [x] C02: Identify the target GPU, OS, API, and shader toolchain; distinguish target requirements from locally observed availability.
- [x] C03: Identify candidate Preset M model versions and artifacts: local `nvngx_dlss.dll` 310.7.0 contains `rrlite_*` M code and matches d4r's pinned artifact; no extracted weights are present in acquired sources.
- [x] C04: Establish the repository organization using Trace without changing generic `AGENTS.md` rules.
- [x] C05: Document initial setup requirements and current tool inventory, including DXC 1.8.2502.11 and the pinned reference build.
- [x] C06: Record 1920×1080 output and Quality as provisional test assumptions, not yet fixed settings.
- [x] C07: Record unavailable or unverified requirements: DXC is present in the SDK but not on `PATH`; d4r's Linux/HIP oracle path and Preset M weight compatibility are unverified.

### Validation Gate

- [x] Project scope and exclusions are explicit.
- [x] Setup requirements and a pinned reference execution path are reproducible with verified versions (`EXP-001`: CMake/VS 2022/SDK/DXC build and both WMMA probes pass). d4r's Linux/Vulkan/HIP oracle path remains unverified and is deferred to M05.
- [x] Important unknowns, provisional assumptions, and current environment limitations are recorded.

### Expected Deliverables

- `README.md`
- Updated `docs/MILESTONES.md`
- Initial `references/README.md` index

### Dependencies

- None.

---

## M02 — Reference Acquisition

**Status:** COMPLETE / PASS

### Objective

Acquire and pin the primary sources required for reconstruction.

### Checkpoints

- [x] C01: Inspect d4r and identify relevant code paths: `engine/` (M notes, runtime, model compiler), `kernels/m/` (WMMA network), `kernels/common/wmma_layout.h`, and `kernels/tools/extract_dlss_ptx.py`.
- [x] C02: Inspect quattro's validated RDNA 3 experiments (EXP-004 INT8 WMMA, EXP-005 FP16 WMMA) and reproduce its build/run path in [EXP-001](experiments/EXP-001-reference-toolchain.md).
- [x] C03: Record commit IDs and origin URLs for all four acquired submodules: d4r, quattro, NVIDIA DLSS, and NVIDIA TransformerEngine.
- [x] C04: Identify model and weight artifacts: no model or weights are present in any acquired source; d4r extracts FP8/f16 matrices locally from a user-supplied DLL; the local DLL contains M PTX and matches d4r's pinned DLL build, but its weights were not extracted and d4r's extraction path was not run.
- [x] C05: Locate conservative FP16 WMMA kernels: d4r `engine/cnn_sr/mma_f16.glsl` and `kernels/common/wmma_layout.h`; quattro `src/gpu/shaders/fp16_wmma.hlsl`.
- [x] C06: Identify standalone inference entry points: d4r `engine/replay_m.cpp`, `MModel`/`MEngine` in `engine/runtime_m.inc`, and `d4r-m`; quattro `src/main.cpp` and its experiment CLI.
- [x] C07: Record local DLL metadata, hash, signature observation, PTX inventory, and provenance/access limitations.
- [x] C08: Identify source or model artifacts that remain unavailable: NVIDIA's model definition and redistributable weights, d4r's proprietary weight packages, and quattro's uninitialized nested reference sources.

### Validation Gate

All accessible primary references are pinned, indexed, and traceable; inaccessible or missing material is explicitly documented. Evidence: [`findings/2026-10-28-reference-inventory.md`](findings/2026-10-28-reference-inventory.md), [EXP-001](experiments/EXP-001-reference-toolchain.md), and [`references/README.md`](../references/README.md).

### Expected Deliverables

- Updated `references/README.md` and, where appropriate, pinned read-only sources under `references/sources/`.
- Relevant provenance/access findings under `docs/findings/`.

### Dependencies

- M01.

---

## M03 — Preset M Architecture Reconstruction

**Status:** COMPLETE / PASS

### Objective

Understand the complete neural execution graph.

### Checkpoints

- [x] C01: Identify all major network stages: exposure, enc0 tokenization, ten Swin layers, dec0 expansion, post reconstruction, and static 3-to-2 downsample.
- [x] C02: Inventory operators and matrix multiplication sites: Q/V projections, `Q Q^T`, `P V`, output projection, per-chunk MLP, patch merge/expand, and the input/expansion GEMMs.
- [x] C03: Determine tensor shapes and layouts: token grids derive from render extent rounded to 32 pixels (`/2` tokens, `/8` tube), with 64/96/128 channels, 2/4 heads, pair-ordered memory, and FP8 plane storage.
- [x] C04: Identify attention, normalization, and residual operations: 16-token window attention with relative bias, two RMS norms per block, two f16 residual streams, and the encoder/decoder skip paths.
- [x] C05: Trace patch merging and reconstruction: `4C -> NPM` merges at half token resolution, `CIN -> 4C` expands with encoder skips, and dec0's four render-resolution surfaces feed post's Gaussian splat and temporal blend.
- [x] C06: Map intermediate buffers and execution dependencies: `tokens`, `skip1/2`, `merged1/2`, tube ping-pong buffers, `dec2Out`, `dec1Out`, and the reconstruction surfaces.
- [x] C07: Identify temporal inputs, outputs, and persistent state: current/previous motion, jitter, exposure, reset, two history images, previous feature, and the half-resolution history.
- [x] C08: Record original precision and scaling where observable: FP8 e4m3 weights and matrix operands, f16 vectors/accumulators, SNORM8 recurrent feature, and f32 reconstruction blend; details in the numerics finding.
- [x] C09: Distinguish architectural facts from inferred behavior and gaps: established source facts, DLL entry-point observations, and unverified assumptions are separated in the finding.

### Validation Gate

- [x] A traceable graph and tensor contracts exist, sourced to pinned d4r files: [`findings/2026-10-28-preset-m-architecture.md`](findings/2026-10-28-preset-m-architecture.md).
- [x] Each stage, layer, channel count, token scale, and buffer binding is traced to `runtime_m.inc`, `M_NOTES.md`, `swin_model.py`, or the stage shaders.
- [x] Unresolved gaps are explicit: no extracted weights, no local d4r execution, no local NGX launch, an unconfirmed per-frame shift schedule, and derived-only 1920x1080 Quality geometry.

### Expected Deliverables

- Architecture findings and graph/operator inventory in `docs/findings/`.

### Dependencies

- M02.

---

## M04 — Numerical Contract Analysis

**Status:** COMPLETE / PASS

### Objective

Understand the numerical behavior of the reference implementation.

### Checkpoints

- [x] C01: Trace FP8 weight decoding: e4m3 (OCP satfinite RNE), exactly representable in f16, blocked 16x16 tiles, no activation scales.
- [x] C02: Identify FP16 conversion behavior: lossless e4m3-to-f16 widening; activations re-encoded to e4m3 at every matrix-operand boundary; f16 vectors and biases.
- [x] C03: Identify accumulator types and rounding behavior: exact product sums within each 32-element step, one f16 rounding per step; f32 only in reconstruction weight sum/blend and in the relaxed build.
- [x] C04: Locate intermediate requantization boundaries: block inputs, norm outputs, Q, attention output, GELU output, block/merge outputs, expanded decoder inputs, and enc0 pixel features.
- [x] C05: Analyze normalization and softmax precision: f16 RMS sums with `2^-13`, f16 rsqrt, f16 attention scores, clamped cubic, exponent bit trick, f16 butterfly sum and reciprocal.
- [x] C06: Determine output conversion and clipping behavior: `+-2*pi` encoder clamps, e4m3 `+-448` saturation, post range compression/clamp, 11-11-10 quantization, non-finite fallback, LDR shader variant.
- [x] C07: Identify potentially sensitive residual paths: the two f16 residual streams, recurrent SNORM8 feature, reconstruction history feedback, and temporal accumulation.
- [x] C08: Compare conservative and optimized execution modes: d4r's exact, relaxed (`M_RELAXED`/`M_RESIDUAL_F32`), TEXL, and int8 `dot2` modes are recorded with their deviations and d4r's reported effects.

### Validation Gate

- [x] Numerical contracts are documented with source references and separated from inference: [`findings/2026-10-28-preset-m-numerics.md`](findings/2026-10-28-preset-m-numerics.md).
- [x] Each contract names its rounding order, dtype, constants, and the d4r file that establishes it.
- [x] Uncertainties are retained: original weight container, dither index, per-architecture NVIDIA rounding, HDR/LDR differences, and the absence of any local numerical probe.

### Expected Deliverables

- Numerical-fidelity findings in `docs/findings/`; executable probes are deferred to M06, where operator-level execution is required.

### Dependencies

- M02 and M03.

---

# PHASE II: ORACLE AND EXPERIMENTAL PROBING

## M05 — FP16 Oracle Reproduction

**Status:** BLOCKED / BLOCKED

**Blocker:** Capture coverage is complete: every stage group (`enc0`, the ten layers, `dec0`, `post_3_1`, `downsample_kernel`) dumps in one frame at 1920x1080, and the `enc0`/`dec0` weight blobs decode with d4r's layout code ([EXP-008](experiments/EXP-008-m05-full-capture.md)). The graph is confirmed from the capture itself: 14 launches, every buffer-mediated edge byte-identical ([stage graph](findings/2026-10-29-m05-stage-graph.md)). The oracle is now a reconstruction task: a D3D12/HLSL path exists and reproduces the published accumulator contract on a real stage ([EXP-010](experiments/EXP-010-m05-native-fp16-stage.md)), but `enc0`, `dec0`, `post` and `downsample` read or write textures whose contents this capture format does not record, so their stage-level validation needs a capture extension. A layer-level oracle is established: ten real-weight layers replay bit-exactly ([EXP-003](experiments/EXP-003-m05-layer-capture.md)).

### Objective

Establish a deterministic reference execution path.

### Checkpoints

- [ ] C01: Build or execute d4r's relevant standalone engine.
- [x] C02: Pin the model and source revision (d4r `05e19d4`; Windows capture tooling `7da2f9a`; local DLL 310.7.0).
- [x] C03: Establish one fixed output resolution and quality setting supported by the reference path (1920x1080 output, quality mode, preset 13; `output-0.rgba16f` captured at that geometry).
- [x] C04: Prepare deterministic inputs (captured layer inputs; fixed 256x144 probe frame).
- [x] C05: Capture final outputs (ten layer outputs, `gpu-output/`; frame `output-0.rgba16f`).
- [x] C06: Capture intermediate tensors where accessible (layer inputs, weights, and outputs).
- [x] C07: Establish repeatability and tolerance requirements (two native replays byte-identical; exact tolerance).
- [x] C08: Identify nondeterministic behavior, if any (none observed at layer scope).
- [x] C09: Capture coverage audit: map the M dispatches against the ZLUDA whitelist and classify the missing groups by weights versus tensors ([finding](../findings/2026-10-29-m05-capture-coverage.md)).
- [x] C10: Extend the Windows capture path for `enc0` and `dec0` weights first, then post/downsample tensors. Done by rebuilding ZLUDA with the four whitelist groups; every stage group now dumps in one frame at 1920x1080 and the `enc0`/`dec0` weight blobs decode with d4r's own layout code ([EXP-008](experiments/EXP-008-m05-full-capture.md)). The earlier probe-side MinHook attempt failed and is recorded in [EXP-004](experiments/EXP-004-m05-capture-hook-attempt.md).
- [ ] C11: Full-network validation: compile the model with `compile_m.py`, execute the full FP16 oracle, and compare against the captured NVIDIA reference while preserving the ten-layer regression. Two dependencies are open: `compile_m.py` names the input stage `rrlite_enc0_4x4_mvlo_{hdr,ldr}_folded` and the synthetic probe produces the `mvhi` variant, and d4r's full-frame oracle (`test_frame_m.py`, `kvk`/`evk`) is a Linux/Vulkan build, so no oracle execution path exists on Windows yet.

### Extension checkpoints: Windows-native oracle (task C16-C22)

- [x] B1 (C16): Capture completeness audit — 14 launches, 14/14 buffer edges confirmed by byte-identical tensor content, arena map established, and the image-bound stages identified as uncaptured ([stage graph](../findings/2026-10-29-m05-stage-graph.md)).
- [x] B2 (C17): Explicit variant support — `m_weights.parse_kernel()` keeps `motion`/`range`/`folded` as separate fields, tests pin the grammar, and static PTX evidence shows `mvlo`/`mvhi` share the weight-block geometry but not their code ([stage graph](../findings/2026-10-29-m05-stage-graph.md)).
- [x] B3 (C18): Windows execution foundation — D3D12 compute path in `src/native/` with DXC compilation, typed-buffer descriptor tables, upload/readback heaps, one fence, and error reporting; one real stage dispatches and reads back ([EXP-010](experiments/EXP-010-m05-native-fp16-stage.md)).
- [ ] B4 (C19): Input/output stage reconstruction — `dec0`'s matrix stage matches the published step chain (97.9% of values bit-identical, all within one f16 step), but the four-surface epilogue, and all of `enc0`/`post`/`downsample`, cannot be validated stage-by-stage because their operands and outputs are textures the capture does not contain.
- [ ] B5 (C20): Neural network integration — the ten layers are validated at layer scope ([EXP-003](experiments/EXP-003-m05-layer-capture.md)) but not yet executed inside the native path.
- [ ] B6 (C21): Full-frame oracle — not started; depends on B4/B5.
- [ ] B7 (C22): Temporal validation — not started; needs multi-frame capture (the hook overwrites per-launch dumps within a frame, and history resources are image-bound).

### Validation Gate

Reference execution is reproducible, and captured outputs are suitable for differential testing with repeatability/tolerance criteria recorded.

**Layer scope: PASS** — 10/10 captured layers replay byte-exactly and repeat byte-exactly against the captured NVIDIA outputs ([EXP-003](experiments/EXP-003-m05-layer-capture.md)).

**Full-network gate: BLOCKED** — capture coverage is complete and the `enc0`/`dec0` weights are real ([EXP-008](experiments/EXP-008-m05-full-capture.md)), and the stage graph is now confirmed edge-by-edge from tensor content ([stage graph](findings/2026-10-29-m05-stage-graph.md)). The Windows-native path has started: one real matrix stage executes on D3D12 and reproduces the published accumulator contract ([EXP-010](experiments/EXP-010-m05-native-fp16-stage.md)). What remains is the reconstruction itself (B4-B6), and a known capture gap: `enc0`, `dec0`, `post` and `downsample` bind textures/surfaces whose contents the capture format does not record, so only their effect on the final frame can be validated until the hook dumps image memory. C11 is open; see [EXP-002](experiments/EXP-002-m05-oracle-blocker.md).

### Expected Deliverables

- Oracle fixtures, execution procedures, and experiment reports under `tests/` and `docs/experiments/`.

### Dependencies

- M02, M03, and M04.

---

## M06 — Operator-Level Reconstruction Probes

**Status:** COMPLETE / PASS

### Objective

Determine how individual neural operations can be reproduced independently.

### Checkpoints

- [x] C01: Inventory representative operator implementations (`engine/model_enc3.py` primitives, `engine/m_model.py` block, `kernels/tools/swin_model.py` layout).
- [x] C02: Reconstruct CPU equivalents where feasible (d4r's numpy layer model driven against real captures by `scripts/m06_layer_model_check.py`).
- [x] C03: Test matrix multiplication behavior (one f16 rounding per k32 step; error bounded by one ulp; K=64/144/256 exact in EXP-005).
- [x] C04: Test normalization (f16 sum-of-squares + `2^-13` + f16 rsqrt; max relative error `4.1e-4`).
- [ ] C05: Test attention calculations — softmax properties verified (normalized, non-negative, rank-preserving); the exponent trick's absolute score-scale convention is **unresolved** and recorded as an open question.
- [x] C06: Test residual and activation operations (clamped-cubic GELU characterized: 0.046 max error, non-monotone on `[-2.0,-0.58]`).
- [x] C07: Test tensor packing and layout conversions (captures decode through `tokens()`/`plain_layer()` and reproduce the GPU outputs, which exercises the blocked-e4m3 and pair-order layouts).
- [x] C08: Compare against independent expectations and reference outputs (OCP e4m3 oracle + exact float64 mathematics + ten real layer launches).

### Validation Gate

Representative operators have reproducible tests and known numerical error characteristics derived from independent expectations or the pinned reference.

**PASS** — 12 contract tests (`tests/test_m06_operator_contract.py`) with specification/float64 oracles, plus a ten-layer drive of the numpy model against the capture (34.9-42.9 dB, 43-70% bit-identical). The layer residual is attributed to the model's exact accumulation versus the hardware's per-k32 f16 rounding ([EXP-006](experiments/EXP-006-m06-operator-contract.md)). C05 remains an open question, not a passing checkpoint.

### Expected Deliverables

- Operator probes, CPU references, tests, and supporting experiment reports.

### Dependencies

- M03, M04, and M05 (layer-scope oracle; the full-network oracle is not required for these probes).

---

## M07 — INT8 Quantization Feasibility

**Status:** COMPLETE / PASS

### Objective

Determine whether Preset M can retain meaningful numerical information under INT8.

### Checkpoints

- [x] C01: Implement a CPU quantization simulator (`scripts/m07_int8_weight_quant.py`).
- [x] C02: Investigate FP8-to-INT8 weight conversion (exact e4m3 -> f16 -> symmetric INT8 over the complete inventory: 360 matrices / 1 553 408 values).
- [ ] C03: Test symmetric and asymmetric representations — symmetric only; asymmetric defers to M12.
- [x] C04: Compare per-tensor, per-output-channel, per-input-channel and per-column K-block scaling (best: 16-row K block at 0.0048 RMS; per-output-channel 0.0062; per-tensor 0.0095).
- [x] C05: Investigate activation ranges (peak/RMS 1.93-4.19 across the ten layer launches; dec0's operand 7.56).
- [ ] C06: Test clipping and saturation — max-based scaling only, no clipping sweep.
- [ ] C07: Measure error propagation through representative blocks — six representative GEMMs measured, activation term separated from weight term; full block defers to M11.
- [x] C08: Identify precision-sensitive operations (dec0's expansion matrix and operand, and the MLP up-projections).
- [x] C09: Establish an initial mixed-precision policy (per-output-channel INT8 weights as the single-accumulation baseline, per-column 16-row K blocks where extra rescales fit; per-token activation scale re-checked in M11 where it is now the dominant term; floating-point RMS/softmax/requant; `K <= 131071` accumulator bound).

### Extension checkpoints (task C11-C15)

- [x] A1 (C11): Complete weight extraction — `enc0` and `dec0` decoded through checked layouts, all 360 matrices inventoried with e4m3 codes preserved ([EXP-009](experiments/EXP-009-m07-complete-inventory.md)).
- [x] A2 (C12): INT8 conversion is deterministic, half-to-even, saturating, zero-preserving, and independently verified ([`tests/test_m07_quantization_contract.py`](../tests/test_m07_quantization_contract.py)).
- [x] A3 (C13): Five scopes compared on identical data with storage and overhead accounting.
- [x] A4 (C14): INT8 GEMM error measured on captured activations for six representative matrices, with integer accumulation and separated weight/activation terms; `enc0`'s missing operand recorded rather than substituted.
- [x] A5 (C15): Every known weight matrix accounted for; remaining unknowns explicit.

### Validation Gate

Quantization behavior and limitations are characterized with reproducible numerical evidence and independent expectations.

**PASS** — real-weight and real-activation error is measured per scaling scope, with outliers and ranges reported and the surviving schemes identified ([EXP-007](experiments/EXP-007-m07-int8-feasibility.md), extended to the complete inventory in [EXP-009](experiments/EXP-009-m07-complete-inventory.md)). Asymmetric quantization, clipping sweeps and block-level propagation are explicitly deferred to M11/M12 rather than claimed. EXP-007's matrix counts and its "block16/32" scope definition were found wrong on re-run and are corrected in an addendum there.

**Weight side: closed.** 360/360 matrices extracted, decoded and characterised. Image quality remains an M11/M16 question: this milestone establishes that the information survives INT8, not that the result looks right.

### Expected Deliverables

- Quantization simulator, experiments, tests, and findings.

### Dependencies

- M04 and M06.

---

## M08 — RDNA 3 Native INT8 Probes

**Status:** COMPLETE / PASS

### Objective

Validate the hardware execution primitives needed by the reconstruction.

### Checkpoints

- [x] C01: Reproduce relevant quattro INT8 experiments (`wmma-int8`, `kernels-int8`; all four sign combinations at 0 mismatches).
- [x] C02: Verify native WMMA instruction generation (`v_wmma_i32_16x16x16_iu8` in both the driver analyzer report and offline RGA for the executed pipelines).
- [x] C03: Validate signedness and operand packing (one opcode, signedness via per-operand `neg_lo`).
- [x] C04: Test INT32 accumulation (exact for K=64/144/256; overflow wraps modulo 2^32, no saturation).
- [x] C05: Validate CPU/GPU numerical equivalence (0 mismatches, idempotent across repeated dispatches).
- [x] C06: Evaluate representative matrix dimensions (M/N 16-256, K 64-256, plus 20x24 and 7x7 boundary shapes).
- [x] C07: Compare WMMA and DP4A implementations (bit-identical results; single-tile latency measured, throughput left to M14).
- [x] C08: Identify layout, padding, and occupancy constraints (no LDS use via intrinsics, 24-34 VGPR, boundary shapes need padded tiles).
- [x] C09: Record compiler, driver, and hardware dependencies (gfx1101, driver 32.0.32015.7029, DXC 1.8.2502.11, RGA 2.15, quattro `ad0482a`).

### Validation Gate

Native INT8 execution is demonstrated with independent numerical checks and inspected GPU ISA on the target hardware.

**PASS** — all four INT8 sign combinations, multi-step INT32 accumulation and the network's matrix shapes match a CPU reference exactly on gfx1101, with the wave-matrix instruction verified in the ISA of the executed pipelines ([EXP-005](experiments/EXP-005-m08-native-int8-probes.md)).

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
