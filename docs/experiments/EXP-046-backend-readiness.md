# EXP-046 — Backend readiness: caller-owned GPU resources and borrowed command lists

**Milestone:** M16-A through M16-G (explicitly scoped backend readiness).
**Status:** **PASS for the scoped backend-readiness objective.** The original M16
final-quality gate is **not passed**: the 45 dB image floor is still FAIL, the
quality verdict is unchanged, and no temporal (reset = 0) path exists. This
report validates Sette as a reusable reset-frame D3D12 inference backend, not
as a completed DLSS replacement.

**Revision:** working tree based on `72ee1f2` (M15-C persistent engine).
**Depends on:** [EXP-043](EXP-043-graph-lifetime-memory-reuse.md) (arena,
production mode), [EXP-044](EXP-044-mlp-gelu-quantizer-fusion.md) (fused MLP),
[EXP-045](EXP-045-persistent-inference-engine.md) (persistent engine, external
resources, twelve-frame reset sequence).

## 1. Question and scope

M15-C proved that the accepted frame can execute as a persistent engine with
twelve changing encoder textures. It left three readiness questions:

1. Are **all** frame-varying image inputs caller-owned, or do post and
   downsample still read captured fixtures?
2. Does the engine reject unsupported external resources and malformed
   manifests explicitly, without corrupting a live engine?
3. Can an embedder own the device, queue, command list, fences and resources,
   and record a frame without Sette submitting, waiting or closing anything?

The scope is the supported reset frame (`reset = 1`, 960×540 render →
1920×1080 output). No fidelity tuning, temporal reconstruction, DXGI proxy
investigation or DLL work is included. The accepted per-stage probes remain the
independent oracle.

## 2. Verification summary

| Check | Result |
|---|---|
| Tests | **405 passed** (10 M15-C/M16 engine tests) |
| Canonical twelve-frame regression | **12/12** byte-identical to the M15-C historical hashes |
| Live twelve-frame inputs | **12/12** byte-identical to fresh independent references |
| Frame 0 / accepted frame | `f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00` |
| Chain tensor | `c89208ed744acd71275f349d99d758c16bd55c77a6fb5d3da88109a3573c6b0d` |
| Exact recovery | frame 0 replayed after frame 11 matches in both sequences |
| Caller-owned list | exact frame, engine never submits or waits |
| External device + queue | exact frame on caller device/queue |
| Cancellation | recorded frame discarded without submission; clean exit |
| External negative suite | **16/16** expected outcomes, zero debug errors |
| D3D12 debug layer | zero errors, corruptions and warnings |
| Native INT8 coverage | **76/76**, 34,518,597,632 MACs |
| M09 finalizer | **17/17 PASS**, 76/76 coverage |
| Image quality | 44.1393 dB PSNR, 0.99863 windowed SSIM; **45 dB gate FAIL** |

## 3. Public engine interface

`src/native/m_inference_engine.h` exposes:

```
Initialize(const Config&)                 engine-owned device and queue
Initialize(ID3D12Device*, ID3D12CommandQueue*, const Config&)
                                          caller-owned device and queue
SetExternal(const External&)              bind/validate one external resource
ExecuteFrame(const FrameRequest&, list)   record one frame; submit+wait only
                                          when the engine owns the queue and
                                          no list is passed
SetOutputState(state)                     state left on external outputs
Destroy()                                 drain (engine-owned queue) and release
```

`Config` selects the composed manifest, shader directory, production/validation
mode, arena on/off, debug layer and sentinel seed. `Initialize` parses the
manifest, plans the arena, creates resources, builds root signatures,
pipelines and descriptor heaps, uploads weights and static surfaces, and
creates validation readback targets. It performs no work per frame after that.

`FrameRequest` carries one entry per external input and output, each with the
resource pointer and the state the caller guarantees at frame start. A frame is
validated completely before the allocator or command list is touched, so a
rejected frame leaves the engine and a borrowed list reusable.

### 3.1 Lifecycle invariants

* One device, one queue (owned or borrowed), one pipeline per stage, one
  descriptor heap per stage, one weight set and one arena survive all frames.
* `ExecuteFrame` creates no GPU resource. Its only operations are allocator/list
  reset (engine-owned mode), descriptor rebinds when a caller resource changes,
  barriers, sets, dispatches, the two dec0→post byte copies, and optional
  validation readbacks.
* The engine never closes, executes or waits on a borrowed command list.
  `FrameStats::engineSubmitted` and `engineWaited` are false for that path.
* An external device/queue engine uses the caller queue for the one-time upload
  during `Initialize`; it does not drain on `Destroy`. The caller must keep the
  queue alive and idle at destruction.
* Frames must be serialized. Descriptor heaps are shared across frames.

## 4. External resource contract

### 4.1 Supported inputs and output

| Resource | Role | Format | Size | Requirements |
|---|---|---|---|---|
| `enc0_in_colour` | encoder render colour | `R16G16B16A16_FLOAT` | 960×540 | `TEXTURE2D`, one mip, one array slice, one sample |
| `post_colour` | post render colour | `R16G16B16A16_FLOAT` | 960×540 | same |
| `post_motion` | post motion | `R16G16_FLOAT` | 960×540 | same; bound but unread on reset |
| `down_colour` | downsample colour | `R16G16B16A16_FLOAT` | 960×540 | same; only alpha is read |
| `frame_out` | output | `R16G16B16A16_FLOAT` | 1920×1080 | same plus `ALLOW_UNORDERED_ACCESS` |

The stages keep distinct declarations: the engine does not assume that the
encoder colour, post colour and downsample colour have identical formats,
dimensions or interpretations. In the accepted reconstruction the three colour
files happen to be byte-identical, but the manifest and validation treat them
as separate resources.

### 4.2 Immutable model resources

Weights, INT8 scale/bias tables, the shape constant, the encoder/decoder
prefix buffers and the chain's learned tensors are loaded once from the
accepted fixtures. They are not frame inputs. Post history (`5760×1620
R32_UINT`) and post half (`1440×810 RGBA32F`) are produced inside the frame and
are not consumed by the reset path.

### 4.3 States, ownership and synchronization

* The caller declares each external resource's state before the frame. The
  engine transitions from that state; its own tracking is reset to the declared
  state each frame.
* Inputs are left in `NON_PIXEL_SHADER_RESOURCE` after their last use. Outputs
  are left in the same state by default (`SetOutputState` changes it).
* The caller owns every external resource and must keep it alive until the GPU
  work that reads or writes it has completed. The engine never releases caller
  resources.
* The caller must serialize frames and must not record concurrently on the
  shared descriptor heaps.
* A borrowed list must be open, compute-type and exclusively owned for the
  duration of `ExecuteFrame`.

### 4.4 Explicit rejection

The engine rejects, rather than degrading:

* null resources;
* resources created on another device (checked with `ID3D12Resource::GetDevice`);
* textures that are not `TEXTURE2D`, or have the wrong width, height or format;
* textures with more than one mip, array slice or sample;
* outputs without `ALLOW_UNORDERED_ACCESS`;
* buffers shorter than the manifest extent, or of the wrong dimension;
* an external resource bound in the wrong direction, or as both input and
  output;
* a frame that omits any external resource the manifest binds;
* validation dumps on a caller-owned list;
* malformed manifests: unknown keys, duplicate resources, binding registers out
  of order, kind mismatches, or a copy between differently sized surfaces.

## 5. M16-B — Complete live resource wiring

Every image-stage binding was inventoried from the accepted fixtures and the
composed manifest.

| Stage | Binding | Classification | M16 disposition |
|---|---|---|---|
| enc0 | weights (buffer) | immutable model | loaded once |
| enc0 | colour (RGBA16F 960×540) | frame input | external (already) |
| enc0 | motion | **not declared** | no enc0 motion input exists |
| dec0 | weights, tokens (buffers) | model + intra-frame | internal |
| dec0 | gate/mix/feat/cov (textures) | intra-frame producers | internal, dumped |
| post | colour (RGBA16F 960×540) | frame input | **external (new)** |
| post | gate/mix/cov | dec0 intra-frame | internal |
| post | motion (RG16F 960×540) | frame input, unread on reset | **external (new)** |
| post | packed/history/half | outputs | internal producers |
| downsample | packed (buffer) | post intra-frame | internal |
| downsample | colour (RGBA16F 960×540) | frame input (alpha only) | **external (new)** |
| downsample | out (RGBA16F 1920×1080) | frame output | external |

The composed manifest now declares **five external resources** (four inputs and
one output) and **715 resources total**. `scripts/m15c_compose.py` no longer
copies the post/downsample colour or motion fixtures into the manifest. The
canonical fixture regression is preserved by the harness supplying the
canonical files as caller textures, not by the engine reading them from disk.

Motion deserves a separate note: `post_cs.hlsl` explicitly does not fetch the
motion vector on `reset != 0`, because the history sample is the colour texel
and the captured texture is smaller than the address the display-motion path
would form. The engine still validates and binds motion as a caller resource;
the twelve-frame live run supplies different motion bytes per frame and matches
references produced with the canonical motion, which proves the unread
property rather than assuming it.

## 6. M16-C — External-resource and negative tests

`m_inference_stage --external-selftest DIR` runs sixteen cases against the live
engine and writes `external-selftest.json`. All sixteen matched the expected
outcome and the debug layer stayed at zero errors.

| Case | Expected | Observed |
|---|---|---|
| valid input bind | accept | accept |
| null input | reject | reject |
| wrong width | reject | reject |
| wrong height | reject | reject |
| wrong format | reject | reject |
| array size 2 | reject | reject |
| mip levels 2 | reject | reject |
| sample count 2 | reject | reject |
| output without UAV flag | reject | reject |
| output wrong format | reject | reject |
| missing required resource | reject | reject |
| output also bound as input | reject | reject |
| resource created on another device | reject | reject |
| malformed manifest | reject | reject |
| valid frame after all rejections | accept | accept |
| zero debug errors | 0 | 0 |

The device-mismatch case uses a WARP device: on this driver a second
`D3D12CreateDevice` call returned the same runtime object, so WARP is the
reliable foreign-device source. The engine rejects the resource by comparing
`GetDevice` with its own device before creating any view.

The malformed-manifest case initializes a separate engine from an invalid
manifest, catches the parse failure, lets that engine destruct, and then runs a
complete frame on the original engine. This demonstrates cleanup after failed
initialization without leaking or corrupting the live engine.

## 7. M16-D — Real multi-frame execution

Two twelve-frame sequences run in one process each:

* **canonical** — the frame variant feeds `enc0_in_colour`; post, motion and
  downsample use the canonical fixture files. This is the historical M15-C
  workload and must reproduce its hashes exactly.
* **live** — the frame variant feeds `enc0_in_colour`, `post_colour` and
  `down_colour`; a distinct synthetic RG16F buffer feeds `post_motion`. Fresh
  references are produced by `m_b6_frame.py` with `--colour` and the new
  `--image-colour`, so the independent per-stage path sees the same post and
  downsample colour.

### 7.1 Canonical regression

| Frame | Scenario | Engine hash | Match |
|---:|---|---|---|
| 0 | static | `f76ac032…4d6c00` | yes |
| 1 | hold | `f76ac032…4d6c00` | yes |
| 2 | pan-8 | `f55f889a…8133a` | yes |
| 3 | pan-32v | `ee55a8e1…5cf39` | yes |
| 4 | rotate-3 | `58355869…82e55f7` | yes |
| 5 | fast-96 | `44b35acd…b78b` | yes |
| 6 | disocclude | `5c4685a8…70ff1` | yes |
| 7 | repeat | `67af7cbb…64a8a5` | yes |
| 8 | thin | `633b99a0…fb4a8` | yes |
| 9 | abrupt | `1dc97896…12fd9` | yes |
| 10 | recover | `f76ac032…4d6c00` | yes |
| 11 | static-end | `f76ac032…4d6c00` | yes |

### 7.2 Live inputs

| Frame | Scenario | Live engine hash (all 64 hex digits) |
|---:|---|---|
| 0 | static | `f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00` |
| 1 | hold | `f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00` |
| 2 | pan-8 | `3bb7042ebfb662cb30af279435cdeef03f370e8ca9bf1fec801246acd26e474c` |
| 3 | pan-32v | `25528e43a64426c2303bf7a3396b32f8a9d74fdbcc36d935b998239af49826fb` |
| 4 | rotate-3 | `0481a32005a9fa36e50e64a0434109d900b73a6a9c108b80bc04f1b5245b530f` |
| 5 | fast-96 | `0289e853d336a5b7a1101e3c69e8a70c3b16357da3c414a54ff7f1aac9aa573e` |
| 6 | disocclude | `1a18e00a26f1a6fa88b24bfe1144a6e01162c2d201a0f89b54d601f0f1601296` |
| 7 | repeat | `483b48ae6204ce50d7049cb53c90460e171ccf9d2bf91117b38ceaf158d3cbb9` |
| 8 | thin | `92f632133b322a9d7f2d3ddb11c4022d9e87ba69f87dfb414f55b77afb35faec` |
| 9 | abrupt | `060c5ddb0186e087cada2c6936e1d0cbf95e242fc631378ea07cc5ddc1e1cb9c` |
| 10 | recover | `f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00` |
| 11 | static-end | `f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00` |

All twelve match their independent references. Frames 10 and 11 return to the
static input and reproduce frame 0 exactly; the explicit recovery execution
also matches. The live hashes differ from the historical canonical hashes for
frames 2–9 because post and downsample now consume the changing colour, which
is exactly the intended new input combination; no historical hash is demanded
for it.

Targeted tests additionally prove:

* changing only `post_colour` changes the output;
* changing only the alpha of `down_colour` changes the output;
* changing only `post_motion` leaves the output unchanged (reset path is unread);
* the manifest contains no fixture-backed post/downsample colour or motion.

## 8. M16-E — Borrowed command list, device and cancellation

| Check | Result |
|---|---|
| Caller-owned list records the whole frame | 233 dispatches, 2 copies, 6 descriptor rebinds |
| Engine submits the borrowed list | **no** (`engineSubmitted = false`) |
| Engine waits on the borrowed list | **no** (`engineWaited = false`) |
| Caller submits and waits | harness owns the queue, fence and event |
| Output read back after caller submission | `f76ac032…4d6c00` |
| Twelve-frame borrowed sequence | all 12 recorded, none submitted by the engine |
| Caller device and queue via `Initialize(device, queue, config)` | exact frame |
| Cancellation: record then discard without submission | clean exit, zero debug errors |
| Caller resources outlive the engine | engine destroyed before harness textures |
| Output state after frame | `NON_PIXEL_SHADER_RESOURCE` by default |

The engine's one-time `Initialize` uploads weights and static surfaces through
the caller queue and waits on its own fence. After initialization, a borrowed
list is recorded only; the caller controls submission and synchronization.

## 9. M16-F — Performance and memory

All numbers are from `m16-work/run-1` and the interleaved A/B block on the RX
7700 XT. The engine owns the device and queue for the steady measurement.

### 9.1 Initialization

| Phase | ms |
|---|---:|
| Parse, lifetime analysis, arena plan | 35.37 |
| Resource creation (170 committed + 531 placed) | 94.57 |
| Pipelines, root signatures, descriptor heaps, params | 260.99 |
| Upload wait | 0.01 |
| **Total** | **1125.72** |

### 9.2 Steady state (twelve production frames)

| Metric | Value |
|---|---:|
| GPU interval median | **174.114 ms** (170.494–183.725) |
| CPU record median | **1.342 ms** |
| Submit + fence wait median | 174.224 ms |
| Dispatches / byte copies per frame | 233 / 2 |
| Transitions / UAV / aliasing per frame | 1,060 / 233 / 235 |
| External descriptor rebinds | 6 on frame 0, 5 on later frames |
| Production readbacks | **0** |
| Resource creation in `ExecuteFrame` | **0** |

The run-to-run GPU spread is large enough that the absolute median is not a
reliable cross-session number. A same-session interleaved A/B against the
committed M15-C engine on the identical twelve-frame workload is authoritative:

| Path | GPU interval median (5 rounds) |
|---|---:|
| M15-C engine (`72ee1f2`) | **173.518 ms** |
| M16 engine | **172.969 ms** |
| Difference | 0.549 ms (**0.32%**) |

The M16 change is resource wiring, not arithmetic: the GPU work is unchanged.
The CPU record time rose from 1.06–1.19 ms (M15-C) to 1.36–1.97 ms (M16) in the
A/B block, and measured 1.34 ms in the readiness run. The harness supplies a
distinct texture per input per frame, so the engine rewrites five input
descriptors (six descriptors on the first frame). Reusing a texture across
frames drops the rebinds to zero (tested).

### 9.3 Memory

| Metric | M16 one-frame process | M15-C one-frame process |
|---|---:|---:|
| Arena heap | 450,625,536 B | 450,625,536 B |
| Committed default resources | 82,529,408 B | 92,897,408 B |
| External caller bytes (one frame) | 31,104,000 B | 20,736,000 B |
| Placed logical bytes | 4,622,911,344 B | 4,622,911,344 B |
| Peak live bytes | 384,424,976 B | 394,792,976 B |
| **Peak transient bytes** | **376,579,856 B** | **376,579,856 B** |
| DXGI local after execute | **833,900,544 B** | **833,900,544 B** |

The three formerly committed static colour/motion textures moved to caller
ownership without changing the one-frame local footprint: committed bytes fell
by exactly the external bytes that were added. The twelve-frame process samples
at ~1,018 MB because the harness preloads twelve input sets (12 × 31.1 MB); that
is a test-harness working set, not the engine's per-frame footprint.

## 10. M16-G — Preserved results

| Item | Result |
|---|---|
| Test suite | **405 passed** |
| Native INT8 operators | **76/76**, 100% of 34,518,597,632 MACs |
| M09 finalizer | **17/17 PASS** on the accepted path |
| Engine frame / chain vs finalizer frame / chain | byte-identical |
| Frame determinism | repeat and recovery hashes identical, 33 boundary hashes stable |
| Enc0/dec0 INT8 | explicit `int8-embedding` / `int8-expansion`; no FP16 matrix fallback |
| Layer handoffs | finalizer `layer_handoffs`, `enc0_handoff`, `repeat` pass |
| Image quality | 44.139286 dB PSNR, 0.998633 windowed SSIM, mean abs 0.001451 |
| 45 dB gate | **FAIL, unchanged** |

The finalizer still runs on the accepted `m_b6_frame` path, not on the engine
binary. The engine's canonical frame and chain tensors are byte-identical to
that path, so its numerical gates transfer by equality rather than assumption.

## 11. Unsupported behavior and blockers

* **`reset = 0` / temporal history.** Every frame is a reset frame. The post
  history, half-resolution and recurrent-feature producers are written but
  never consumed across frames.
* **Resolution and format.** Only 960×540 → 1920×1080 with the listed formats.
  Other sizes, aspect ratios, HDR, display-motion configurations and integration
  paths are unvalidated and rejected or unsupported.
* **Motion.** The reset post shader does not read motion; motion is validated
  and bound for contract completeness only.
* **Concurrent recording.** Descriptor heaps are shared; frames must be
  serialized.
* **Independent scene validation.** The live references use the same accepted
  per-stage binaries with different inputs, not a new captured game scene.
* **DLL/proxy.** No DXGI hook, swapchain integration or DLL has been built.
* **Finalizer scope.** The M09 finalizer validates the accepted path; engine
  equivalence is established by byte-identical tensors.
* **Quality.** The 45 dB gate remains FAIL. Backend readiness is not a fidelity
  improvement.

## 12. Backend-readiness verdict

**PASS, explicitly scoped.** For the supported reset frame:

* every frame-varying image input is a caller-created `ID3D12Resource`
  (`enc0_in_colour`, `post_colour`, `post_motion`, `down_colour`);
* the output is a caller-owned UAV texture;
* twelve changing frames execute correctly in one process, including changing
  post and downsample colour, with independent references;
* the engine records safely onto a caller-owned command list and on a
  caller-owned device/queue without closing, executing or waiting on borrowed
  work;
* unsupported resources and malformed manifests are rejected explicitly, and
  the engine remains usable after rejection;
* all historical hashes, tests, coverage and finalizer gates are preserved.

This does **not** satisfy the original M16 final-quality gate. The quality
verdict is unchanged and temporal execution remains unimplemented.

## 13. Requirements for the future DLL adapter

1. **Swapchain resources.** Accept the real back-buffer format and size (not
   only `RGBA16F`), or document a conversion pass. The current contract requires
   exact format and dimensions.
2. **State ownership.** Integrate with the application's resource-state
   tracker; declare the true incoming state per resource and respect the
   `NON_PIXEL_SHADER_RESOURCE` output state (or set a different one).
3. **Queue and fence.** Decide whether the adapter records onto the game's
   compute/graphics queue or its own; propagate a fence the caller can wait on.
   The engine must never submit a borrowed list.
4. **Descriptor concurrency.** The current engine uses one descriptor heap per
   stage and requires serialized frames. A DLL adapter needs either a frame
   ring of descriptor heaps or an explicit serialization contract with the
   application.
5. **Lifetime.** The adapter must keep caller textures alive until the GPU
   work completes and must destroy the engine before releasing descriptors and
   resources.
6. **Error handling.** Map the engine's explicit rejection messages to the
   adapter's fallback path (for example, skip DLSS/INT8 and keep the game's
   native output) rather than silently degrading.
7. **Memory budget.** Reserve the 450.6 MB arena plus the immutable committed
   set and one external frame set (31.1 MB); do not assume the test harness's
   twelve-frame preload.
8. **Resolution/aspect policy.** Define supported render/output pairs and
   rejection behavior before exposing the engine to arbitrary game settings.
9. **Temporal contract.** Do not expose history/recurrent inputs until a
   labelled `reset = 0` capture and the missing branches are implemented.
10. **Validation hook.** Run the external selftest and canonical regression in
    the adapter's CI before enabling the path.

## 14. Reproduction

```
python scripts/m15c_compose.py --out <compose>
python scripts/m16_readiness.py --out <run> --manifest <compose>/engine.txt
python -m pytest tests -q

build/native/Release/m_inference_stage.exe \
  --manifest <compose>/engine.txt --shaders build/native \
  --frames-file <run>/frames-live.json --output frame_out \
  --mode validation --dump-dir <dumps> --dump-frame 0 --recover --debug
```

The readiness driver generates the twelve live references with
`m_b6_frame.py --colour X --image-colour X`, validates the canonical and live
engine sequences, exercises the caller-owned list, external device, cancellation
and sixteen negative cases, then measures the production workload. Canonical
hashes are reused from the M15-C evidence when present; otherwise the driver
regenerates them.

## 15. Evidence

Archived outside the repository at
`%LOCALAPPDATA%\Temp\opencode\evidence-20261031-m16`:

`readiness.json` (all checks, hashes, benchmark, one-frame memory), `sequence`
reports, `finalizer.json` (17/17), `external-selftest.json` (16/16),
`ab-interleaved.json` (five-round M15-C/M16 comparison), and the canonical/live
reference manifests. The multi-gigabyte compose and reference trees were
removed after the reports were written.
