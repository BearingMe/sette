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
| M05 | FP16 Oracle Reproduction | IN PROGRESS | PENDING |
| M06 | Operator-Level Reconstruction Probes | COMPLETE | PASS |
| M07 | INT8 Quantization Feasibility | COMPLETE | PASS |
| M08 | RDNA 3 Native INT8 Probes | COMPLETE | PASS |
| M09 | INT8 Kernel Reconstruction | COMPLETE | PASS |
| M10 | Complete Network Execution | COMPLETE (bounded C05) | PASS (bounded C05) |
| M11 | Differential Error Mapping | COMPLETE | PASS |
| M12 | Weight Quantization Refinement | NOT STARTED | PENDING |
| M13 | Activation and Temporal Refinement | NOT STARTED | PENDING |
| M14 | Performance Baseline | NOT STARTED | PENDING |
| M15 | RDNA 3 Optimization | NOT STARTED | PENDING |
| M16 | Final Quality Gate | NOT STARTED | PENDING |

Phase II: M06-M08 pass. M05 is capture-complete for buffers and images and its
reconstruction executes end to end on Windows/D3D12, but the FP16 oracle is not
validated: B5 fails `enc1.merge` ([EXP-019](experiments/EXP-019-m05-native-network.md)),
B6 fidelity is INCONCLUSIVE ([EXP-020](experiments/EXP-020-m05-native-frame.md)), and the
temporal path is unimplemented ([EXP-022](experiments/EXP-022-m05-b7-readiness.md)).
M09 is COMPLETE/PASS for the bounded reset-frame matrix gate: 76/76 designated
operators, 100% of the 34,518,597,632-MAC inventory, verified signed dot4, and a
deterministic full frame ([EXP-029](experiments/EXP-029-m09-complete-int8.md)).
M10 is COMPLETE for complete-network execution and reset-frame sequences; the
persistent-state inventory is complete but cross-frame reuse is BLOCKED because
no `reset = 0` branch or labelled non-reset capture exists
([EXP-030](experiments/EXP-030-temporal-state-inventory.md),
[EXP-031](experiments/EXP-031-temporal-oracle.md),
[EXP-032](experiments/EXP-032-temporal-error.md),
[EXP-033](experiments/EXP-033-temporal-checkpoints.md)). Quality and performance
work continues in M11-M15; the unchanged 45 dB gate still FAILS for INT8.

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

**Status:** IN PROGRESS / PENDING

**Status:** Capture coverage is complete at 1920x1080 for both resource kinds - buffers
([stage graph](../findings/2026-10-29-m05-stage-graph.md)) and 33 image targets with
geometry, format, pitch and pre/post contents
([EXP-011](experiments/EXP-011-m05-image-capture.md)) - and every stage of the frame now
executes natively on Windows/D3D12: `downsample` and `post` reproduce their captured
outputs byte for byte ([EXP-012](experiments/EXP-012-m05-native-downsample.md),
[EXP-016](experiments/EXP-016-m05-post-reconstruction.md)), `enc0` reproduces its token
field byte for byte and `dec0` its four surfaces with an attributed residual
([EXP-015](experiments/EXP-015-m05-enc0-contract.md),
[EXP-014](experiments/EXP-014-m05-dec0-mapping.md)), and the ten layers run as one
GPU-resident graph ([EXP-019](experiments/EXP-019-m05-native-network.md)). The complete
1920x1080 frame has been produced ([EXP-020](experiments/EXP-020-m05-native-frame.md)).

What is **not** established is the oracle itself. The milestone's objective is a
deterministic reference, and the reference's own numerical contract is still contested:
under the rescoped translated-PTX gate B5 passes eleven of its twelve declared fields and
the chained criterion and fails `enc1.merge` (89.8379% exact, RMS 0.02111 against 90% and
0.02), and the merge was separately proven correct on its own operands, so the residual is
upstream rounding rather than a mapping defect. B6 executes correctly but its fidelity is
INCONCLUSIVE: there is one captured content, the reference uses d4r's native layer
replacements, and no independent static validation set exists. B7 (temporal) is not
started; [EXP-022](experiments/EXP-022-m05-b7-readiness.md) audits what the existing
multi-frame capture does and does not contain.

Checkpoints C01 and C11 are therefore **not** marked complete. The Windows-native path
supersedes their *purpose* (a deterministic reference execution path, validated against
the captured reference) but not their *text*: d4r's standalone engine was never executed
here (C01), and the `compile_m.py` full-FP16-oracle route is still unbuilt and, as far as
this project has established, not executable on Windows (C11). An alternative path that
reaches a similar place is not the same checkpoint.

### Objective

Establish a deterministic reference execution path.

### Checkpoints

- [ ] C01: Build or execute d4r's relevant standalone engine. **Not done, and not on
  the adopted path:** d4r's frame oracle is a Linux/Vulkan build. The reference was
  established instead by capturing d4r's launches on Windows and replaying them
  natively (B1-B6). The checkpoint's purpose is met; its stated action is not.
- [x] C02: Pin the model and source revision (d4r `05e19d4`; Windows capture tooling `7da2f9a`; local DLL 310.7.0).
- [x] C03: Establish one fixed output resolution and quality setting supported by the reference path (1920x1080 output, quality mode, preset 13; `output-0.rgba16f` captured at that geometry).
- [x] C04: Prepare deterministic inputs (captured layer inputs; fixed 256x144 probe frame).
- [x] C05: Capture final outputs (ten layer outputs, `gpu-output/`; frame `output-0.rgba16f`).
- [x] C06: Capture intermediate tensors where accessible (layer inputs, weights, and outputs).
- [x] C07: Establish repeatability and tolerance requirements (two native replays byte-identical; exact tolerance).
- [x] C08: Identify nondeterministic behavior, if any (none observed at layer scope).
- [x] C09: Capture coverage audit: map the M dispatches against the ZLUDA whitelist and classify the missing groups by weights versus tensors ([finding](../findings/2026-10-29-m05-capture-coverage.md)).
- [x] C10: Extend the Windows capture path for `enc0` and `dec0` weights first, then post/downsample tensors. Done by rebuilding ZLUDA with the four whitelist groups; every stage group now dumps in one frame at 1920x1080 and the `enc0`/`dec0` weight blobs decode with d4r's own layout code ([EXP-008](experiments/EXP-008-m05-full-capture.md)). The earlier probe-side MinHook attempt failed and is recorded in [EXP-004](experiments/EXP-004-m05-capture-hook-attempt.md).
- [ ] C11: Full-network validation: compile the model with `compile_m.py`, execute the full
  FP16 oracle, and compare against the captured NVIDIA reference while preserving the
  ten-layer regression. **Partly superseded, still genuinely open.** The comparison itself
  now exists by another route: the complete native pipeline is compared against the captured
  reference at layer scope under the translated-PTX contract (B5, FAIL on one field),
  stage by stage (B4, PASS), and at full-frame scope (B6, execution PASS / fidelity
  INCONCLUSIVE). What remains unfulfilled is the literal dependency chain: `compile_m.py`
  names the input stage as `rrlite_enc0_4x4_mvlo_{hdr,ldr}_folded` while the synthetic probe
  produces the `mvhi` variant, and d4r's full-frame oracle (`test_frame_m.py`, `kvk`/`evk`)
  is a Linux/Vulkan build, so that oracle has never been executed here. The checkpoint stays
  open because the *reference's* fidelity is what B5/B6 leave undecided.

### Extension checkpoints: Windows-native oracle (task C16-C22)

- [x] B1 (C16): Capture completeness audit — 14 launches, 14/14 buffer edges confirmed by byte-identical tensor content, arena map established, and the image-bound stages identified as uncaptured ([stage graph](../findings/2026-10-29-m05-stage-graph.md)). Extended by the image capture below: every texture and surface those stages bind is now dumped, so the audit covers both resource kinds ([EXP-011](experiments/EXP-011-m05-image-capture.md)).
- [x] B2 (C17): Explicit variant support — `m_weights.parse_kernel()` keeps `motion`/`range`/`folded` as separate fields, tests pin the grammar, and static PTX evidence shows `mvlo`/`mvhi` share the weight-block geometry but not their code ([stage graph](../findings/2026-10-29-m05-stage-graph.md)).
- [x] B3 (C18): Windows execution foundation — D3D12 compute path in `src/native/` with DXC compilation, typed-buffer descriptor tables, upload/readback heaps, one fence, and error reporting; one real stage dispatches and reads back ([EXP-010](experiments/EXP-010-m05-native-fp16-stage.md)).
- [x] B4 (C19): Input/output stage reconstruction - **done**. B4.1 (fixture
  infrastructure), B4.2 (native `downsample`) and B4.3 (native `dec0`) are done. A captured
  launch executes from its own recorded formats, geometry, pitch, parameter block and dispatch,
  a representative set of 8 captured resources round-trips through D3D12 unchanged, native
  `downsample` reproduces its captured surface **byte for byte** (16 588 800 of 16 588 800,
  two runs identical, [EXP-012](experiments/EXP-012-m05-native-downsample.md)), and native
  `dec0` reproduces all four of its surfaces on the GPU - gates 99.9991%, mix 99.9348%,
  feature 99.9787%, covariance 99.9883% of f16 values identical - with the residual attributed
  to the reference's own approximations and to E4M3 quantisation boundaries
  ([EXP-014](experiments/EXP-014-m05-dec0-mapping.md)). `dec0`'s operand and weight mapping is
  now traceable to the kernel's PTX rather than to a decode of d4r's model, and that map
  replaces [EXP-013](experiments/EXP-013-m05-dec0-epilogue.md)'s blocker. B4.4 (native `enc0`) is **done**: the resource and parameter contract, the 12 416-byte
  blob layout (dither, matrix, bias) and the output field are derived from the kernel's
  PTX, and both the CPU oracle and the native stage reproduce the captured 8 355 840-byte
  token field **byte for byte** (8 355 840 / 8 355 840, 130 560 / 130 560 tokens, two runs
  identical, and equal to `enc1`'s captured input),
  [EXP-015](experiments/EXP-015-m05-enc0-contract.md). The three errors that had to be
  corrected were the dither's add operand, eight of the 32 feature slots' channels, and the
  epilogue's channel permutation. B4.5 (native `post`) is **done**: the reconstruction reads
  five textures and writes three surfaces, and the native stage reproduces **all three byte
  for byte** - the 18 662 400-byte packed word buffer, the 37 324 800-byte RGBA16F
  reconstruction and the 18 662 400-byte half-resolution surface, two runs identical,
  [EXP-016](experiments/EXP-016-m05-post-reconstruction.md). Two of its three outputs needed a
  conversion the PTX does not show (the f16 surface truncates; the half-resolution surface is
  10-bit and holds exact `k/1023`), and the odd row of each thread's row pair keeps the even
  row's texels. B4.6 (numerical fidelity) is **done**: `dec0`'s residual is classified as
  E4M3 boundary flips - every differing covariance/gate sample matches the activation table
  exactly one code from the reference, the four surfaces' differing pixels are disjoint
  because they consume disjoint expansion columns, and no storage-conversion bias exists -
  and its propagation is measured: substituted into `post` and `downsample`, the 0.23% of
  `dec0` pixels becomes 0.064% of the final frame's samples at a median of 1 f16 ULP, while
  the same chain from captured inputs stays byte-exact. Acceptance floors for the FP16 oracle
  (per surface and end to end, plus three structural invariants) are executable in
  `scripts/m_conformance.py`, [EXP-017](experiments/EXP-017-m05-dec0-fidelity.md). B4.7 (the
  regression gate) is **done**: one command runs all four stages, requires every declared
  output at its declared size (a missing, short or undeclared comparison fails instead of
  skipping), rebuilds the `dec0 -> post -> downsample` chain, and applies the criteria -
  reproducibly, 99.9369% of the final frame at max 7 ULP,
  [EXP-018](experiments/EXP-018-m05-regression-gate.md). The gate also runs each stage twice: on its
  first run that check found `post` was non-deterministic (one block, at pixels whose
  out-of-bounds motion fetch perturbed the dispatch), which is fixed. The history branch of
  `dec0` (frames with `reset = 0`) and of `post` is not implemented.
- [ ] B5 (C20): Neural network integration - **in progress**. The ten layers execute
  natively on the GPU through one 94-dispatch D3D12 graph with GPU-resident intermediates,
  use the captured weights and inputs, and are deterministic; each layer reproduces
  93.6-98.7% of the captured e4m3 codes (enc1 96.924%, enc2 96.890%, tube#1..#6
  96.376/98.586/97.899/98.745/97.221/93.584%, dec2 94.853%, dec1 95.365%), and the
  natural-order contract is bit-identical to the kernel-order model and to the shader.
  It is **not byte-faithful**: historical alternate-arithmetic probes did not resolve
  the numerical difference; they do not rule out runtime k32 rounding behavior. The chained native
  `dec1` matches only 21.327% of the captured field, below the 99% chain floor, so the
  `--network` gate fails. The blocker, the negative results (k16/k8/f32 accumulation) and
  the toolchain findings are in [EXP-019](experiments/EXP-019-m05-native-network.md).
  **Reference-provenance correction:** the enc1-only investigation in EXP-019's
  addendum verifies the window probe and finds original interpreted PTX agrees
  with Sette in the tested blocks, while the canonical capture log selects a
  d4r-native replacement (`translated_jit=skipped`). At that checkpoint the first
  runtime-divergent operator was unknown. A subsequent enc1-only capture confirms actual
  `backend=translated` dispatch of the original, hash-matched PTX in two fresh
  runs, with byte-identical input/weights/output/merge. Against this reference,
  Sette is 96.418337% output / 89.837910% merge; see EXP-019's translated-capture
  addendum. Old conformance fixtures are preserved; no numerical fix is claimed.
  A controlled pinned-source fast/conservative enc1 replay now finds canonical
  output/merge **100% conservative-equivalent**, versus 30.537265% / 27.768650%
  for fast. Sette remains 96.924307% / 91.287531% against conservative (H3).
  Verified diagnostic blocks match through quantized norm1; the first Q/V k32
  projection step diverges. Conservative also differs from translated PTX, so an
  authoritative full-network numerical contract remains unresolved. Historical
  routing is audited for all 14 launches; numerical build modes of the other
  native modules are not certified. B4 reruns green at 99.9369%; B5 and its 99%
  chain threshold are unchanged. See EXP-003 / EXP-019 contract-validation addenda.
  **First-K32 follow-up:** raw original-PTX runtime fragments now prove identical
  operands/C for one Q mismatch and an exact-window control. Sette matches the
  translated result (`b014`); conservative returns `b013`. A replicated arithmetic
  probe localizes native deviation before FP16 conversion, already in its first
  K16 WMMA. Synthetic midpoints also distinguish translated rounding from exact
  mathematical RNE; PTX leaves accumulation rounding/order unspecified. No CPU
  or HLSL arithmetic correction is justified. Fresh ten-layer/chain measurements
  are unchanged (21.3270% chained dec1, deterministic); B4 stays green and
  204 tests pass. Original-hardware/internal-WMMA semantics remain unresolved;
  see EXP-019's one-first-K32 addendum. B5 remains incomplete.
  **Translated-PTX reference across all ten layers:** every network kernel
  (`enc1`, `enc2`, the six tube shifts, `dec2`, `dec1`) has now been executed as
  original PTX translated by ZLUDA, with the dispatch log proving
  `backend=translated`, zero native dispatches, one hash-matched module per
  kernel, and byte-identical repeat runs (details in EXP-019's ten-layer addendum).
  From identical input/weight bytes, Sette reproduces 94.64-98.80% of the
  translated e4m3 codes per layer (conservative d4r: 94.59-98.57%), with relative
  RMS 0.41-1.57%, no bias, no saturation and no non-finite values. d4r's own
  native replacement differs from the translated PTX by 1.5-5.4% per layer on
  identical operands (enc1, enc2, tube#1, dec2, dec1), so the per-layer residual
  is not Sette-specific. Run as a complete translated chain (all ten launches
  translated, skip and tube resources preserved), the original PTX reproduces
  only 21.7096% of the canonical chained dec1 field; Sette reaches 21.5728% of
  that translated chain and 21.3270% of the canonical capture. The 99% chain
  floor is therefore not a translated-reference criterion: it fails for the
  original PTX too, because the ten-stage e4m3 requantization chain amplifies
  layer-level code differences to near-random agreement (canonical-vs-translated
  is 44.63% at enc2 and 13.97% at tube#6). No threshold, fixture, gate or
  arithmetic was changed; the legacy conservative gate and the translated
  fidelity measurements are reported separately and any gate replacement needs
  an explicit, separately validated decision. B4 stays green at 99.9369%,
  208 tests pass, and 21.3270% remains the failing legacy chain result.
  **B5 rescope (authorized 2026-10-29):** acceptance moved from byte identity with
  one native trajectory to the translated-PTX contract, declared before the run
  and encoded in `m_conformance.py` (`B5`): per layer `output`/`merge` >= 90%
  exact e4m3 codes, relative RMS <= 0.02, maximum decoded deviation <= 1.5,
  |signed mean| <= 1e-3, exact saturation and zero non-finite pairs, with all
  three implementations deterministic; and chained dec1 within 2 points of the
  reference pair's own chained agreement (21.7096%), RMS within +0.02 and maximum
  within +1.0 of it. The reference must be measured with the current Sette
  binary and shaders, and a missing or stale reference fails instead of being
  skipped. The legacy native-replacement comparison still runs and is reported,
  but it is diagnostic and no longer vetoes; its 99% floor is unchanged. The
  rescoped gate was executed and **fails on exactly one field**: enc1 merge,
  89.8379% exact and RMS 0.02111 against 90% and 0.02. The other eleven fields
  (94.6417-98.7963%) and the chained criterion (21.5728% versus 21.7096%) pass.
  The residual is adjacent-code rounding spread evenly over the merge's three
  channel planes, not a mapping error. B5 therefore stays IN PROGRESS / FAIL,
  with enc1 patch merge as the single open item; no bound was loosened and the
  merge field was not exempted. 222 tests pass, including 14 new gate tests.
  **enc1 merge root cause (2026-10-29):** the merge path was isolated on identical
  operands. The store stage writes `enc8(x2b)` and the merge re-quantizes the same
  values, so the layer's output field is the merge's operand image, and every stored
  merged token reads only codes that field contains. Rebuilding the declared contract
  (natural channel order, pair permutation folded into the weights, bias as the
  initial accumulator, one f16 rounding per k32, `+-2*pi` clamp, `q8_exact`, stored
  layout) reproduces Sette's merge 100.0000%, the translated merge 99.9481% (1,627
  codes) and the canonical native merge 99.9667% (1,045 codes), all adjacent-code
  flips with no bias, saturation or non-finite values. A single-rounding arm loses
  30-35k codes, confirming the per-k32 rounding. The end-to-end 89.8379% therefore
  comes from the merge's *operands*: Sette's own output codes are 96.4183% against
  translated and the merge amplifies that (conservative shows the same amplification,
  98.4912% -> 94.3946%). There is no merge defect, so no correction was made; the
  criterion, thresholds and fixtures are untouched and `enc1.merge` still fails at
  89.8379% / RMS 0.02111. B5 stays IN PROGRESS / FAIL, now with the residual attributed
  to upstream layer rounding rather than to any named operator. 226 tests pass,
  including four new merge-contract tests.
  **B5 status is unchanged and is not reopened (2026-10-30).** `enc1.merge` remains
  89.8379% / RMS 0.02111 against 90% / 0.02; no threshold was relaxed, no field was
  exempted, no rounding was changed, and no NVIDIA-hardware parity is claimed. B6 was
  allowed to proceed independently under an explicitly experimental scope - B6's own
  execution gate does not gate on B5 and B5 does not gate on B6's image - so the full
  FP16 oracle stays qualified by this failure even though the frame is coherent.
- [ ] B6 (C21): Full-frame oracle - **EXECUTORY: execution PASS, fidelity
  INCONCLUSIVE.** The complete native pipeline (`enc0` -> ten native layers -> `dec0` ->
  `post` -> `downsample`) produces a 1920x1080 frame on gfx1101 with no captured tensor
  substituted between neural stages ([EXP-020](experiments/EXP-020-m05-native-frame.md)).
  `scripts/m_b6_frame.py` now declares its contract in code and separates three things:
  execution (all five stages, every declared output at its manifest-declared byte size,
  RGBA16F 1920x1080, zero non-finite, zero saturation, a fresh two-run byte-identical
  repeat across **all ten** stage boundaries, and a recorded hash per boundary), regression
  (declared floors of 45 dB / 0.99 windowed SSIM / mean abs 0.005 / max abs 0.5 on both the
  native frame and the image-stage control, plus a 42 dB floor on the chain's own
  contribution), and fidelity (INCONCLUSIVE without an independent validation set). It
  fails closed on missing outputs, mis-sized outputs, non-finite samples, unexpected
  saturation, a reused-but-unverified output set, and cannot report PASS for a run that
  did not re-execute.
  Measured against the canonical capture: 47.05 dB PSNR (peak 1.0), windowed SSIM
  0.999259 (the earlier 0.99794 was a labelled non-overlapping 8x8 block metric), mean
  absolute error 0.001086, 0.24% mean relative error, maximum 0.178; alpha bit-identical
  everywhere; zero NaN/Inf and zero saturation on every path. Feeding the image stages the
  captured `dec1` gives 46.73 dB and the translated-PTX chain 47.08 dB, so the chained
  neural contribution (46.59 dB against the captured-dec1 path) is the same order as the
  image-stage error and neither dominates. Error is spread (median 16x16 block 0.00077,
  worst 0.00290) with the interior slightly worse than the border, so no region collapses
  and nothing indicates a reconstruction defect.
  This is **not** a validated oracle: B5 is still IN PROGRESS / FAIL, there is one captured
  content and no independent static validation set, the reference uses d4r's native layer
  replacements, and the frame is the LDR/reset variant with captured colour and zero motion
  only. Viewable reference/reconstruction/difference RGBA PNGs are exported locally; no
  proprietary data is tracked.
- [ ] B7 (C22): Temporal validation - **NOT STARTED, prerequisites audited.** The
  multi-frame image capture does contain temporal resources, and
  [EXP-022](experiments/EXP-022-m05-b7-readiness.md) measures exactly what it contains:
  `post` history in/out at 2880x1620 (37,324,800 B, double-buffered across replays),
  `dec0`'s SNORM8 recurrent feature in/out at 960x540, and `enc0`'s previous-motion input
  with its three motion outputs (all zero). One nonzero producer->consumer handoff is
  bit-exact in **two independent stages** (`post` history out replay 3 = history in replay
  2; `dec0` feature out 3 = feature in 2), so the temporal data is real and recoverable.
  What is missing is the bookkeeping: no capture records the frame index, the `reset` flag
  or which surface the next launch binds, so no two-frame chain can be labelled with
  confidence; `post` implements only the reset path (its fixture validator rejects
  `reset != 0`), `dec0`'s recurrent branch is unimplemented, and the motion texture's
  extent (960x540 bound vs 1920x1080 sampled) is unresolved because every captured motion
  value is zero. First missing dependency: per-launch frame/reset bookkeeping. The plan is
  in EXP-022 §5; no temporal execution is claimed.

### Validation Gate

Reference execution is reproducible, and captured outputs are suitable for differential testing with repeatability/tolerance criteria recorded.

**Layer scope: PASS** — 10/10 captured layers replay byte-exactly and repeat byte-exactly against the captured NVIDIA outputs ([EXP-003](experiments/EXP-003-m05-layer-capture.md)).

**Full-network gate: PENDING / FAIL at layer scope.** Capture coverage is complete for
both resource kinds ([EXP-011](experiments/EXP-011-m05-image-capture.md)), and every stage
executes natively: `downsample` and `post` byte-exactly
([EXP-012](experiments/EXP-012-m05-native-downsample.md),
[EXP-016](experiments/EXP-016-m05-post-reconstruction.md)), `dec0` at 99.9991% / 99.9348% /
99.9787% / 99.9883% of f16 values with its residual attributed to E4M3 boundary flips
([EXP-014](experiments/EXP-014-m05-dec0-mapping.md)), `enc0` byte-exactly
([EXP-015](experiments/EXP-015-m05-enc0-contract.md)), B4 green at 99.9369% of the final
frame at max 7 ULP ([EXP-018](experiments/EXP-018-m05-regression-gate.md)). The gate is
PENDING because the network-level contract is not met: B5 fails `enc1.merge` under its
translated-PTX criteria ([EXP-019](experiments/EXP-019-m05-native-network.md)), B6 executes
but its fidelity is INCONCLUSIVE ([EXP-020](experiments/EXP-020-m05-native-frame.md)), and
B7 is not started ([EXP-022](experiments/EXP-022-m05-b7-readiness.md)). C11 remains open:
the literal `compile_m.py` full-FP16-oracle route has never been executed here
([EXP-002](experiments/EXP-002-m05-oracle-blocker.md)).

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

**Status:** COMPLETE / PASS for the established Preset M reset-frame matrix
scope — **76/76 logical operators, 100% of 34,518,597,632 dense shape MACs**
execute native INT8 ([EXP-029](experiments/EXP-029-m09-complete-int8.md)).
Enc0 and all 20 dynamic QKᵀ/PV products now join EXP-028's learned-weight graph.
Independent CPU arithmetic, gfx1101 ISA, resource handoffs and determinism pass.
This is an execution result, **not validated image quality or M10 completion**:
the final 44.1393 dB frame retains the unchanged B6 image-regression FAIL.
FP16 remains byte-identical; B4/B5 unchanged. Temporal behavior is not evaluated.

### Objective

Implement the neural matrix operations required by Preset M.

### Checkpoints

- [x] C01: Implement INT8 matrix kernels. Bounded kernels include the
  standalone GEMM of [EXP-021](experiments/EXP-021-m09-int8-gemm.md) and the in-graph
  Q projection of [EXP-023](experiments/EXP-023-m09-int8-enc1-q.md) (encode + matmul
  entry points, `dot4add_i8packed`, `cs_6_4`). The isolated dec0 expansion also executes
  over all 130,560 rows ([EXP-024](experiments/EXP-024-m09-dec0-int8-expansion.md));
  its original nonlinear/surface stage now consumes that INT8 expansion through
  an optional GPU-resident prefix (EXP-025); the FP16 default remains unchanged.
  Enc1's full 64×256 up and 256×64 down projections now execute with native
  INT8; original GELU/E4M3 and the norm2 residual remain floating-point. Down
  explicitly uses whole-K INT32, f32 rescale/residual-add, and f16 output (EXP-027).
  EXP-028 reuses these kernels cumulatively for all Q/V/Wo and MLP pairs across
  ten layers plus both merges/decoder expansions and dec0. No new matrix kernel;
  enc0 and dynamic attention were still unconverted at that checkpoint.
  EXP-029 closes enc0 with the existing GEMM and implements both dynamic dot4
  products in all ten layers. No designated matrix remains FP16 in the full mode.
- [x] C02: Support required tensor layouts. Row-major f16 activations and
  per-column K-major f32 weights, both quantised to INT8 on the same word layout the
  graph uses. A second path now reads plane-major E4M3 directly on GPU and consumes
  offline-quantized blocked dec0 weights, with all physical offsets independently
  checked ([EXP-024](experiments/EXP-024-m09-dec0-int8-expansion.md)). Other required
  layouts were initially open; offline weight preparation is permitted.
  EXP-026 additionally validates enc1's eight blocked W1 chunks/bias vectors and
  supplies a natural-order row-major f16 pre-GELU boundary without host activations.
  EXP-027 validates all eight W2 chunks and the 256-channel GPU GELU/quantizer
  boundary, preserving the original row-major x2b output contract.
  EXP-028 adds and independently validates GPU decoder plane unpack, merge Q8
  patch gather and original-format merge store. C=64/96/128 and K through 512
  execute correctly. EXP-029 closes the reset-frame texture/dither adapter,
  field handoff and both dynamic window/head operand layouts.
- [x] C03: Implement quantization/dequantization boundaries. Symmetric
  INT8 with one scale per activation row and one per output column, zero-row handling,
  clamping and the f16 output boundary are implemented and measured in the graph;
  EXP-029 adds runtime Q-row/P-row/window-local V-column scales and explicit
  score-bias/f16/output boundaries. Per-K-block scales, calibration, clipping
  and asymmetric encodings are not required for this execution contract.
- [x] C04: Support necessary floating-point operations and document each exception.
  The f32 epilogue (two multiplies) and the single f16 RNE conversion are
  documented; attention's own E4M3 re-quantisation of the Q output is recorded as the
  next boundary in the initial probe. Dec0's E4M3 round-trip and floating-point
  nonlinear/surface epilogue are preserved as explicit exceptions (EXP-025).
  MLP up adds bias in f32 then writes f16 RNE; original GELU/E4M3 and norm2
  residual remain floating-point. INT8 down replaces the original eight f16 k32
  accumulator roundings with whole-K integer reduction and one f32 residual add
  plus f16 RNE store (EXP-027); it is not bit-equivalent FP16 accumulation.
  EXP-028 explicitly retains dynamic attention matrices as unresolved INT8
  contracts, not justified permanent FP16 exceptions. Nonmatrix softmax,
  normalization, residuals and reconstruction filtering remain floating-point.
  EXP-029 resolves both dynamic matrix contracts, retaining the original
  floating-point softmax, normalization, GELU, residual and image epilogues.
  Quantization/dequantization scale arithmetic remains explicitly FP32/FP64-assisted.
- [x] C05: Validate representative layers. Enc1 has validated Q-only and
  MLP-pair INT8 modes. Dec0 passes both captured/native-token stage
  comparisons and a dec0-only hybrid frame; EXP-024's deferred C5/C6 boundary is
  resolved in EXP-025. Enc1's MLP up is now CPU/GPU bit-exact in the actual graph,
  with verified original-remainder handoff and full-frame propagation (EXP-026).
  The **enc1 MLP pair is now complete**: both matrices pass CPU/GPU, unchanged
  GELU/FP16-split checks, native ISA and actual frame handoffs (EXP-027).
  EXP-028 now validates every learned projection/MLP/merge/expand in the ten-layer
  cumulative chain, plus live-frame dec0. Dynamic QKᵀ/PV and enc0 still prevent
  a complete INT8-network/representative-full-layer claim at that checkpoint.
  EXP-029 now validates all matrices, adapters and nonlinear handoffs in the
  cumulative ten-layer chain using its live INT8 enc0 input, plus live dec0.
  Synthetic two-/four-head zero/sign/window tests also pass. This does not
  establish temporal or out-of-scope shape fidelity.
- [x] C06: Compare outputs against CPU INT8 references — the in-graph Q projection is
  **bit-identical** to the independent CPU INT8 reference (codes, scales and output),
  with 8/8 independent scalar dot probes agreeing
  ([EXP-023](experiments/EXP-023-m09-int8-enc1-q.md)). Dec0's 10,444,800 integer,
  f32 and f16 outputs also agree exactly, with 48 scalar dot checks (EXP-024).
  MLP up's 34,979,840 accumulators/f32/f16 outputs agree bit for bit, with 48
  scalar dot probes, all-column scalar weight checks and synthetic GPU cases (EXP-026).
  Down's 8,744,960 accumulators/f32/f16 outputs also agree exactly, with 48
  scalar probes and nonuniform K=256 GPU edge cases (EXP-027).
  EXP-028's final neural batch compares 264,440,320 accumulator/f32/f16 elements
  bit for bit, all codes/scales and eight scalar probes per operator, plus
  10,444,800 live dec0 elements. Both sentinel seeds and all layout gates pass.
  EXP-029 revalidates those learned outputs on new inputs, plus 25,067,520 enc0
  and 147,179,520 dynamic-attention accumulator/f32/f16 elements, codes/scales,
  softmax probabilities and final E4M3 conversions. All are bit-exact.
- [x] C07: Inspect generated GPU ISA — 16 `v_dot4_i32_i8` for K = 64 and zero FP16
  matrix instructions in the executed `swin_q8_matmul.cso`
  ([EXP-023](experiments/EXP-023-m09-int8-enc1-q.md) §5); `v_dot4_i32_i8` also
  confirmed for the standalone GEMM ([EXP-021](experiments/EXP-021-m09-int8-gemm.md)).
  EXP-024 confirms native dot4 and no FP16 matrix fallback in the hashed dec0 shader.
  EXP-026 confirms 16 static dot4 and zero FP16 matrix instructions in the
  executed MLP-up shader, with inspected/executed hashes matched.
  EXP-027 confirms the same native signed dot4/no-FP16-matrix property in the
  executed down shader; K=256 loop iterations are not static instruction counts.
  EXP-028 confirms native signed dot4/no FP16 matrix fallback for both reused
  kernels across every cumulative batch, with actual stage/frame hashes checked.
  EXP-029 additionally inspects both new dynamic shaders (128 static dot4 each)
  and matches all five matrix shader hashes to actual complete-frame execution.
- [x] C08: Measure numerical error before optimization — Q 1.50% relative RMS, enc1
  out 4.95% / merge 6.11% decoded, chained dec1 13.55%, frame 45.42 dB against 47.05 dB
  for the FP16 path ([EXP-023](experiments/EXP-023-m09-int8-enc1-q.md) §7).
  Dec0 expansion: 0.5259% relative RMS vs native FP16, zero CPU/GPU implementation
  error (EXP-024). Dec0-only hybrid: 47.0249 dB vs canonical, direct hybrid-vs-FP16
  62.9160 dB / RMS 0.000714826, 15 deterministic boundaries (EXP-025).
  MLP up: 1.0002% relative RMS, enc1 out/merge 4.6447%/6.1756%, chained dec1
  13.5094% vs FP16; MLP-only frame 45.9208 dB vs canonical, direct FP16 comparison
  45.4375 dB, 18 deterministic boundaries. All three earlier frame hashes remain
  unchanged; 314 tests pass (EXP-026).
  Complete pair: same-input down error 2.7956% relative RMS; pair enc1 out/merge
  5.5981%/7.0224%, chained dec1 13.7555% vs FP16; frame 45.6815 dB vs canonical,
  45.7754 dB directly vs FP16. All 23 boundaries deterministic; 321 tests pass,
  all four earlier control hashes unchanged (EXP-027).
  Cumulative Q/V → +Wo → +MLP → +spatial coverage is 20/30/50/55 of 76 operators,
  16.2928/24.4392/84.7438/93.8695% of MACs. Canonical frames measure
  44.4441/44.8286/44.2142/44.3779 dB; final direct-vs-FP16 44.9235 dB.
  All execution/arithmetic gates pass; the unchanged 45 dB B6 image-regression
  floor fails for every cumulative frame and remains explicitly reported.
  339 tests pass; B4/B5 and FP16 output are unchanged (EXP-028).
  EXP-029: full native INT8 frame 44.1393 dB / SSIM 0.998632607; direct-vs-FP16
  44.4875 dB / RMS 0.00596518. All 33 boundaries deterministic; 348 tests pass.
  The unchanged 45 dB image floor remains FAIL, with no nonfinite/saturated output.

### Validation Gate

Representative reconstructed layers execute correctly on gfx1101 against independent
CPU INT8 references, and generated ISA confirms the intended native matrix arithmetic.

**PASS for the bounded reset-frame scope.** All 76 designated matrix operators
execute natively with independent CPU arithmetic checks, verified signed dot4,
correct graph/field handoffs and deterministic finite output. The complete
[inventory](experiments/EXP-028-matrix-inventory.json) retains its historical G
rows and adds the H status overlay. EXP-029 records the new arithmetic contracts,
executed hashes, cumulative frame and limitations. No designated matrix fallback
remains in the full mode; original FP16 modes remain available explicitly.
The decoder expansion and plane-major operand pass isolated C1–C4 (EXP-024) and
now feed dec0's typed image outputs and a full hybrid frame (EXP-025). No M10
completion is inferred from static matrix coverage alone. Temporal
fidelity and the existing B5 failure are unchanged.

### Full INT8 coverage priority

The user has prioritized full INT8 coverage after completing the enc1 MLP pair.
Grow one cumulative real-network path, not more detached projection modes.
Final reset-frame coverage after EXP-029:

| Matrix family | Established native INT8 coverage | Remaining coverage |
|---|---|---|
| MLP up/down | All ten pairs, C=64/96/128, cumulative | No learned MLP matrix gap in the reset graph |
| Q / V / Wo | All ten layers, cumulative | No learned projection gap in the reset graph |
| Attention score/value products | All ten QKᵀ/PV pairs, runtime-quantized, cumulative | No designated dynamic matrix gap in the reset graph |
| Patch merge/expand | Both merges, both decoder expansions and dec0, cumulative | No learned spatial matrix gap in the reset graph |
| Front-end matrix work | Enc0 GPU texture/dither encoder, existing INT8 GEMM and field epilogue | No designated front-end matrix gap in the reset graph |

**Next:** M10 complete-network execution validation, using the same full-INT8
path and EXP-029's CPU/ISA/handoff evidence. Matrix coverage is 100%; there is no
remaining designated FP16 matrix exception. Do not infer temporal completeness,
quality validation or performance from this static result.
Each added family still needs independent CPU arithmetic, native ISA tied to
executed shaders, resource/handoff and determinism gates. Count actual executed
coverage, not available kernels. Keep nonlinear/normalization/residual exceptions
explicit. Quality and optimization follow execution completeness; do not divert
into calibration, FP16 rounding or new backends. Temporal/M10 gates remain separate
and must not be marked passed from static matrix coverage alone.

### Expected Deliverables

- Initial kernels and operator-level regression tests in `src/` and `tests/`.

### Dependencies

- M06, M07, and M08.

---

## M10 — Complete Network Execution

**Status:** PASS for complete-network execution and the bounded reset-frame
sequence gate; **BLOCKED** for cross-frame persistent-state reuse. M09's reset
matrix gate passes, and EXP-029 provides the 76-operator graph. M10-A..E now add
a complete temporal-state inventory (EXP-030), a deterministic 12-frame FP16/INT8
sequence oracle whose frame 0 is the accepted M09 frame (EXP-031), per-frame error
and state-divergence measurements (EXP-032), and checkpoint/replay (EXP-033).
No `reset = 0` branch exists and no labelled non-reset capture exists, so history
reuse is not implemented or claimed (EXP-022, EXP-030 §5). Existing B4/B5 and
image thresholds remain unchanged.

**Task mapping.** The current mission's temporal plan ("M10-A..E", "M11-A..D",
"M12-A..E") maps onto this roadmap as: temporal M10 → this milestone's C05 plus
M13 C06–C07; quality M11 → M11 (differential mapping) plus M12 (quantization
refinement); performance M12 → M14 (baseline) plus M15 (optimization). Repository
numbering remains authoritative; the mission labels are recorded here only to
avoid two progress dashboards.

### Objective

Execute the entire reconstructed Preset M network.

### Checkpoints

- [x] C01: Assemble the complete execution graph.
- [x] C02: Load converted weights.
- [x] C03: Allocate required buffers.
- [x] C04: Implement dispatch dependencies.
- [x] C05: Handle persistent temporal state — **bounded**. EXP-030 inventories
  every original and reconstructed state element with producer, consumer,
  format, initialisation and reset behaviour; EXP-031/032/033 execute a 12-frame
  sequence and measure the state the graph still writes. The executed graph has
  **no cross-frame consumer**: `post`/`dec0` do not bind history/feature inputs,
  `enc0` declares no motion outputs, and `reset = 0` is rejected or unimplemented
  in all three stages. Cross-frame reuse is therefore BLOCKED, not passed.
- [x] C06: Execute all required neural stages.
- [x] C07: Produce complete output frames.
- [x] C08: Verify finite outputs and buffer safety.
- [x] C09: Detect omitted layers and accidental precision fallbacks.

### Validation Gate

A complete frame is produced, with designated matrix operations using native INT8 arithmetic. Image quality is not an acceptance requirement; execution coverage, buffer safety, and precision paths must be evidenced.

**PASS for complete-network execution and reset-frame sequences.** [M] Twelve
frames with different inputs executed in FP16 and full INT8: 33 boundary hashes
per frame, zero non-finite/saturated samples, and exact replay of frames 0/6/11
(frame, boundaries and three persistent tensors). Frame 0 reproduces the accepted
M09 hashes (`dc97c4cb…`, `f76ac032…`). Same-input INT8↔FP16 error spans
42.12–44.72 dB with no trend (index correlation 0.014); canonical frames remain
44.1393 dB against the reference and still FAIL the unchanged 45 dB gate. The
persistent tensors a future loop would consume differ by 1.06–1.66% RMS (post) and
7.23–8.85% (dec0 feature). **No temporal reuse is claimed**: the state inventory
shows writers without readers, and `reset = 0` remains unimplemented. See
EXP-030 (inventory), EXP-031 (oracle), EXP-032 (error), EXP-033 (checkpoints).

### Expected Deliverables

- Standalone inference implementation and end-to-end tests.

### Dependencies

- M03, M05, and M09.

---

# PHASE IV: NUMERICAL FIDELITY

## M11 — Differential Error Mapping

**Status:** COMPLETE / PASS for the differential investigation. The unchanged
45 dB image gate remains **FAIL** for INT8 and is not redefined. See
[EXP-034](experiments/EXP-034-quality-baselines.md) (baselines),
[EXP-035](experiments/EXP-035-stage-error-attribution.md) (stage attribution),
[EXP-036](experiments/EXP-036-precision-experiments.md) (controlled variants) and
the [quality-gate finding](findings/2026-10-30-int8-quality-gate.md).

### Objective

Identify where INT8 reconstruction diverges from the FP16 oracle.

### Checkpoints

- [x] C01: Compare intermediate tensors — 21 boundaries decoded in their own
  storage domains across 12 frames (E4M3, UNORM8, SNORM8, f16, f32, packed).
- [x] C02: Measure absolute and relative errors — relative RMS, RMS/peak,
  absolute RMS, percentiles and maximum per boundary.
- [x] C03: Characterize distribution and saturation — exact-match fractions,
  finite/saturation counts, content spread; no non-finite or saturated sample.
- [x] C04: Identify first significant divergence points — enc0 tokens are the
  earliest boundary (0.73% RMS/peak); the largest deviations are dec0 mix
  (8.28%), enc2 merge (7.88%), tube-6 (6.38%) and the dec0 feature (5.48%).
- [x] C05: Track errors across residual connections — every residual-bearing
  layer field is compared at its boundary (`enc1/enc2/tube-1..6/dec1/dec2` and
  the two merges); no separate residual ablation was run.
- [x] C06: Investigate temporal error accumulation — **bounded**: 12-frame
  sequences show no trend (index correlation 0.014) and exact reset recovery;
  accumulation through real history reuse is BLOCKED (EXP-030/032).
- [x] C07: Distinguish quantization errors from implementation defects — M09
  established bit-exact CPU/GPU arithmetic; EXP-035 uses fixed-input
  measurements (EXP-025) to separate propagation from dec0's own quantization.

### Validation Gate

Major numerical divergence sources are identified through reproducible comparisons against the established independent FP16 oracle.

**PASS.** [M] The three measurements are reproduced independently
(47.0514 / 44.1393 / 44.4875 dB). The image-stage ceiling is 46.7260 dB and the
FP16 margin above the 45 dB floor is 2.05 dB; INT8 loses 2.91 dB. The accepted
scale scheme is the best of the tested variants, and window-local V scales are
2.7× better than a global scale. Closing the absolute gate through scale or
clipping changes is not supported by the evidence; the gate stays **FAIL**.

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
