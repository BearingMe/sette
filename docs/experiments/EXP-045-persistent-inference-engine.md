# EXP-045 — Persistent in-process inference engine and external GPU resources

**Milestone:** M15-C · **Status:** PASS for persistent multi-frame execution with
live GPU resources. Twelve changing frames execute in one process with one
device, one set of pipelines, one descriptor/weight set and one arena; every
frame is byte-identical to its accepted standalone reset-frame output, frame 0
is byte-identical to the accepted M09 frame, and a replay of frame 0 recovers
exactly. The engine also records onto a caller-owned command list and consumes
caller-owned input/output textures. GPU time is essentially unchanged: the gain
is the removal of per-frame host, process and allocation work, not new GPU
arithmetic. Quality is unchanged (the 45 dB gate still FAILs).

**Revision:** working tree based on the M15-B commit (`perf(m15): fuse mlp gelu
with the down-projection quantizer`); M15-C is not committed in this session.
**Depends on:** [EXP-043](EXP-043-graph-lifetime-memory-reuse.md) (arena,
production mode), [EXP-044](EXP-044-mlp-gelu-quantizer-fusion.md) (fused MLP),
[EXP-031](EXP-031-temporal-oracle.md) (twelve-frame content sequence),
[EXP-030](EXP-030-temporal-state-inventory.md) (reset-only state contract).

## 1. Question and scope

The accepted reconstruction is a set of fixture runners: every stage is a
process, every tensor handoff is a file, and every run rebuilds the device,
pipelines, descriptors, weights and (for the chain) the arena. Can the complete
frame become a persistent in-process engine that:

* initializes once and executes changing frames without rebuilding anything;
* accepts live `ID3D12Resource` textures and a caller-owned output texture;
* records onto a caller-owned command list without closing, executing or waiting
  on it;
* preserves every accepted hash, the M09 finalizer and determinism?

The scope is the existing five-stage frame (enc0 INT8, ten-layer INT8 chain,
dec0 INT8, post, downsample). No quantization, fidelity, temporal, proxy or DLL
work is included. The accepted per-stage probes remain the independent oracle.

## 2. Verification

| Check | Result |
|---|---|
| Tests | **399 passed** (4 new M15-C tests) |
| Twelve changing frames | **12/12 byte-identical** to standalone reset-frame references |
| Frame 0 | **`f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00`** |
| Chain tensor `out-dec1` | **`c89208ed744acd71275f349d99d758c16bd55c77a6fb5d3da88109a3573c6b0d`** |
| Frame-0 chain diagnostics | **12/12** match the accepted M09 `chain.json` hashes |
| Frame-0 enc0 tokens | `e7d8f97c2a92a97c3bb14841565d94156ce30817656a43f08d94da98d339bae4` (accepted) |
| Recovery | replay of frame 0 after frame 11 is byte-identical |
| Caller-owned list | one full frame, exact hash, zero debug messages |
| D3D12 debug layer | zero error/corruption/warning messages |
| Native INT8 coverage | **76/76**, 34,518,597,632 MACs (unchanged) |
| M09 finalizer | **17/17 PASS** on the accepted path; engine frame/chain are byte-identical |
| Quality | unchanged; 45 dB gate still **FAIL** |

The finalizer ran on a fresh M15-B diagnostic frame (fused graph, enc0/dec0
INT8) and returned 17/17. The engine's frame-0 output and `out-dec1` are
byte-identical to that frame and to the accepted M09 tensors, so the finalizer's
numerical gates transfer. The finalizer does not inspect the engine binary; that
equivalence is established by the hashes above, not assumed.

## 3. C1 — Idempotent lifecycle

`src/native/m_inference_engine.h` defines the lifecycle:

```
Initialize(Config)                  device, manifest, arena, weights,
                                    pipelines, root signatures, descriptor
                                    heaps, parameter buffers, readback targets
SetExternal(External)               rebind one external resource (descriptor
                                    view update only)
ExecuteFrame(FrameRequest, list)    record one frame; submit+wait only when the
                                    engine owns the queue and no list is passed
Destroy()                           drain, then release everything
```

The configuration is a composed manifest, a shader directory, arena on/off,
production/validation mode and a sentinel seed. `Initialize` performs one
allocation pass and one upload pass; `ExecuteFrame` allocates nothing. A second
`Initialize` on the same object is rejected by the `initialized_` guard.

The engine never recompiles a shader, recreates a pipeline, reloads a weight or
rebuilds the arena per frame. The only per-frame CPU work is resetting the
allocator/list (engine-owned mode), setting pipeline/root/descriptor state,
recording transitions and dispatches, and — when the caller supplies a different
input texture — rewriting that texture's SRV descriptor.

## 4. C2 — External resource contract

### 4.1 Inputs and outputs

| Item | Contract |
|---|---|
| Frame colour input | `ID3D12Resource`, `TEXTURE2D`, `R16G16B16A16_FLOAT`, 960×540, one mip, one array slice, `ALLOW_UNORDERED_ACCESS` not required |
| Static auxiliary surfaces | post colour (`RGBA16F` 960×540), post motion (`RG16F` 960×540), downsample colour (`RGBA16F` 960×540); loaded once from the accepted fixtures |
| Output | caller-owned `ID3D12Resource`, `TEXTURE2D`, `R16G16B16A16_FLOAT`, 1920×1080, one mip, `ALLOW_UNORDERED_ACCESS` required |
| Frame parameters | the accepted reset frame only (`reset = 1`); no temporal parameters exist |

The accepted reconstruction binds the external frame colour only to the encoder.
Post and downsample consume the captured canonical colour from their own
fixtures; changing that would change the accepted output. This is a property of
the reconstruction, not of the engine, and is recorded as a limitation for M16.

### 4.2 States and synchronization

* The caller declares the state each external resource is in at the start of the
  frame (`FrameRequest::External::state`); the engine transitions from it.
* The engine leaves the output in `D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE`
  by default (`SetOutputState` changes it). Inputs are left in the state of their
  last use (normally `NON_PIXEL_SHADER_RESOURCE`).
* The caller must serialize frames: a descriptor view is rewritten when the
  external resource changes, and that is only safe after the previous frame's
  GPU work using it has completed.
* A borrowed command list must be open, compute-type and exclusively owned for
  the duration of `ExecuteFrame`. The engine never calls `Close`, `ExecuteCommandLists`
  or a fence wait on a borrowed list.
* The engine owns the device/queue/fence when constructed with `Initialize(Config)`;
  `Initialize(device, queue, config)` is available for an embedder that already
  owns a compute queue, in which case the engine only records on borrowed lists.

### 4.3 Explicit rejection

Unsupported inputs fail rather than degrade:

* a texture with the wrong dimension, width, height or format;
* a buffer shorter than the manifest extent;
* an unbound external resource;
* a caller list with validation dumps (readback requires engine-owned submission);
* a manifest with unknown keys, duplicate resources, out-of-order binding
  registers, mismatched buffer/texture binding kinds, or a copy between
  differently sized surfaces.

Formats beyond the accepted set (`R16_FLOAT`, `R16G16_FLOAT`,
`R16G16B16A16_FLOAT`, `R32_FLOAT`, `R32G32_FLOAT`, `R32G32B32A32_FLOAT`,
`R8G8B8A8_UNORM`, `R8G8B8A8_UINT`, `R32_UINT`) are rejected by the manifest
parser.

## 5. C3 — Twelve-frame persistent execution

`scripts/m15c_sequence.py` generates the twelve M10 content variants (static,
hold, pan-8, pan-32v, rotate-3, fast-96, disocclude, repeat, thin, abrupt,
recover, static-end), uploads each as one GPU texture at setup, and executes all
twelve in one `m_inference_stage` process. Every frame uses `reset = 1`; no
temporal history is read or written.

| Frame | Scenario | Engine hash = reference |
|---:|---|---|
| 0 | static | yes (`f76ac032…4d6c00`) |
| 1 | hold | yes |
| 2 | pan-8 | yes |
| 3 | pan-32v | yes |
| 4 | rotate-3 | yes |
| 5 | fast-96 | yes |
| 6 | disocclude | yes |
| 7 | repeat | yes |
| 8 | thin | yes |
| 9 | abrupt | yes |
| 10 | recover | yes (`f76ac032…`) |
| 11 | static-end | yes (`f76ac032…`) |
| replay 0 | recovery | yes (`f76ac032…`) |

The references are produced independently by the accepted `m_b6_frame.py` path
(one process per stage, files between stages, fused MLP, production mode,
arena). The engine matches every one.

## 6. C4 — GPU-resident input and output

The composed manifest (`scripts/m15c_compose.py`) flattens the accepted frame
into **235 operations** (233 compute dispatches and 2 byte copies) and **715
resources**:

* enc0 INT8 prefix: features, quantizer, up projection, epilogue;
* the ten-layer chain, with the fused GELU/quantizer;
* dec0 INT8 prefix: quantizer, matmul, epilogue;
* post and downsample.

Handoffs are wired by resource name, not files:

```
enc0_tokens  = enc0 epilogue -> chain enc1_in
dec0_tokens  = chain dec1_out -> dec0 quantizer
post_packed  = post -> downsample
dec0_cov     = dec0 epilogue -> post (same format, aliased)
dec0_gate/mix-> post_gate/mix (byte copy; R32_UINT -> R8G8B8A8_UNORM)
```

D3D12 cannot create an `R8G8B8A8_UNORM` SRV on an `R32_UINT` resource, so the two
gate/mix handoffs use an explicit GPU byte copy through a persistent staging
buffer. This is a GPU copy, not a CPU/file round trip. The cov handoff is a plain
alias because both stages use `RGBA16F`.

In the steady-state production path there are no disk reads, no CPU staging and
no readbacks. Readback exists only in validation mode, where the requested dump
resources are copied to reusable readback buffers after the frame.

## 7. C5 — Correctness and lifecycle

* **Byte-identical frames:** 12/12 versus standalone reset references; frame 0
  and the chain tensor equal the accepted M09 hashes.
* **Changing inputs:** three distinct non-canonical variants plus the recovery
  frames produce the expected distinct hashes; no frame inherits a prior frame's
  tensor (the harness reuses the arena and all transients are rewritten).
* **Recovery:** frame 0 replayed after frame 11 is byte-identical.
* **No stale descriptors:** changing the input texture rewrites the SRV on the
  affected stages before recording; the output descriptor is stable.
* **No use-after-free or aliasing corruption:** the engine drains on `Destroy`,
  the arena uses the accepted aliasing-barrier schedule, and the debug layer
  reports zero errors, corruptions and warnings.
* **Caller-owned list:** one full frame recorded on a harness-owned list, then
  submitted and read back by the harness; exact `f76ac032…` output.
* **Finalizer:** 17/17 on the accepted path; the engine's frame and chain are the
  same bytes.
* **Regression suite:** 399 tests pass, including the existing M15-B, M09, M14
  and image-stage contracts.

## 8. C6 — Measurements

All numbers are from `m15c-seq-4` on the RX 7700 XT. The engine owns the device
and queue; GPU intervals are D3D12 timestamps around the whole command list.
Production mode has no readbacks.

### 8.1 One-time initialization

| Phase | ms |
|---|---:|
| Parse + lifetime analysis + arena plan | 37.94 |
| Resource creation (committed + 531 placed) | 97.37 |
| Pipelines, root signatures, descriptor heaps, params | 235.48 |
| Upload wait (weights, static textures, arena sentinel) | 0.01 |
| **Total** | **1173.19** |

Allocation inventory (one-time): 173 committed resources, 531 placed resources,
233 descriptor heaps, 1,176 descriptors, 59,648 B of parameter uploads.

### 8.2 Steady state (twelve production frames)

| Metric | Value |
|---|---:|
| GPU interval median | **172.215 ms** (169.507–175.586) |
| CPU record median | **1.100 ms** |
| Submit + fence wait median | 173.150 ms |
| Compute dispatches per frame | 233 |
| Byte copies per frame | 2 |
| Transitions / UAV / aliasing per frame | 1,079 / 233 / 235 |
| New GPU allocations per frame | **0** |
| Readbacks per frame | **0** |

The submit/wait figure is the fence wait, which contains the GPU interval; the
difference from the GPU interval (≈0.94 ms) is submission and fence overhead.

### 8.3 Interleaved comparison with M15-B

Five rounds, alternating the accepted `m_b6_frame` fused production frame and
the persistent engine on the canonical input.

| Path | GPU interval median |
|---|---:|
| M15-B (five processes, files) | **172.060 ms** |
| Persistent engine (one process) | **171.437 ms** |
| Saving | 0.62 ms (0.36%) |

An earlier session measured 166.354 vs 164.502 ms (1.85 ms, 1.1%). The
same-session interleaved block is authoritative; the GPU-side saving is small and
within clock/power variance. The architectural saving is the removal of four
extra process launches, four file handoff trees, and all per-frame resource
creation — not a change in GPU arithmetic.

### 8.4 Memory

| Metric | Persistent engine | M15-B reference |
|---|---:|---:|
| Arena heap | 450,625,536 B | 437,387,264 B |
| Committed default resources | 92,897,408 B | ~24.4 MB persistent |
| Placed logical bytes | 4,622,911,344 B | 4.39 GB declared |
| Peak live bytes | 394,792,976 B | 385,209,616 B |
| **Peak transient bytes** | **376,579,856 B** | **376,579,856 B** |
| DXGI local, after execute | 833.9 MB (one frame) / 885.8 MB (12-frame validation) | 714.8 MB (chain process) |

The production transient peak is identical: the chain's arena is unchanged. The
committed total and DXGI usage are higher because the four image stages that M14
measured as separate processes now coexist with the chain in one process. The
M15-B 714.8 MB number is the maximum across five processes, not their sum; it is
not directly comparable to a single-process full-frame number. The merged
pipeline's extra committed cost is the post history/half surfaces, the dec0
textures, the static post/downsample colour/motion surfaces and the copy staging.

## 9. Rejected or deferred work

* **Texture-aware arena/aliasing plan.** The 450.6 MB arena holds buffers only;
  the image-stage textures are committed. Placing textures into a shared heap
  with aliasing would reduce the merged peak but is a new allocator design, not
  part of this milestone.
* **Live colour into post/downsample.** The accepted reconstruction feeds those
  stages the captured canonical colour. Feeding the changing frame colour is a
  fidelity change, not an engine change.
* **Temporal history.** Every frame is `reset = 1`; no cross-frame state is
  consumed. `reset = 0` remains unimplemented (EXP-030).
* **DLL/proxy integration.** The external contract is tested with harness-owned
  resources, not a DXGI hook.
* **Changing arithmetic for speed.** The engine reuses the accepted shaders and
  their hashes; no shader was modified.

## 10. Limitations

* The GPU interval is essentially unchanged; the persistent engine should not be
  described as a GPU-speed optimization.
* Peak local VRAM is higher than the M15-B chain process because the full frame
  is now resident in one process. The transient peak is unchanged.
* The external colour is bound only to the encoder, matching the accepted frame;
  post/downsample colour and motion are static reconstruction surfaces.
* The M09 finalizer validates the accepted path; the engine is bridged by
  byte-identical tensors, not by a finalizer run on the engine binary.
* Frames must be serialized; concurrent recording on the shared descriptor heaps
  is not supported.
* Only the accepted 960×540 → 1920×1080 reset frame is supported. Other
  dimensions, formats and reset values are rejected.

## 11. M16 readiness recommendations

1. **External-resource conformance:** negative tests for format, dimension,
   state and ownership, plus a caller-owned list recorded from a real swapchain
   texture rather than a harness texture.
2. **Queue/fence ownership:** test the `Initialize(device, queue, config)` path
   with a caller queue and an externally signaled fence, including cancellation.
3. **Texture-aware memory plan:** decide whether the merged pipeline peak needs a
   placed-texture heap before claiming a VRAM budget.
4. **Content source:** keep the twelve-frame suite as the reset baseline; do not
   claim temporal quality without a labelled non-reset capture.
5. **Quality gate:** the 45 dB verdict is unchanged and still fails; M16 must not
   present the persistent engine as a fidelity improvement.

## 12. Reproduction

```
python scripts/m15c_compose.py --out <fresh-compose>
build/native/Release/m_inference_stage.exe \
  --manifest <fresh-compose>/engine.txt --shaders build/native \
  --input enc0_in_colour=<compose>/work/enc0/in-colour.bin \
  --output frame_out --frames 1 --mode validation \
  --dump-dir <fresh-dumps> --dump-frame 0 --report <report.json>

python scripts/m15c_sequence.py --out <fresh-sequence> --repeats 5
python -m pytest tests/test_m15c_engine.py -q
```

The sequence driver regenerates the twelve references with `m_b6_frame.py`,
validates the engine against them, runs the recovery and caller-list checks, and
performs the interleaved M15-B block. `--reuse` reuses a compose tree and its
`references.json` for a repeat measurement.

## 13. Evidence

Archived outside the repository at
`%LOCALAPPDATA%\Temp\opencode\evidence-20261031-m15c`:

`engine.txt` (composed manifest), `sequence.json` (12-frame hashes, recovery,
caller-list, benchmark, interleaved block), `engine-validation.json`,
`steady/report.json`, `persistent/run-*/report.json`, `m15b/run-*/` timing and
VRAM JSON, `finalizer.json` (17/17), and the finalizer frame/chain hashes. The
multi-gigabyte dump and reference trees were removed after the reports were
written.
