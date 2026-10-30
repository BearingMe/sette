# EXP-030 — Persistent temporal-state inventory

**Milestone:** M10-A · **Status:** PASS for the inventory; cross-frame reuse is
**BLOCKED** by missing implementation and missing oracle (see §5).
**Scope:** the established Preset M reset frame, 960×540 render → 1920×1080,
`rrlite_*_mvhi_ldr_folded`, DLL 310.7.0, RX 7700 XT / gfx1101, D3D12.
**Depends on:** [EXP-011](EXP-011-m05-image-capture.md) (captured temporal edges),
[EXP-014](EXP-014-m05-dec0-mapping.md), [EXP-015](EXP-015-m05-enc0-contract.md),
[EXP-016](EXP-016-m05-post-reconstruction.md), [EXP-022](EXP-022-m05-b7-readiness.md)
(B7 readiness). **Supersedes nothing.**

## 1. Question

Which state survives between frames, who produces and consumes it, how is it
initialised and invalidated, and what does the reconstruction actually execute?
The answer must separate the *original network's* temporal graph from the
*reconstructed* one; M09 covered the second only for a single reset frame.

## 2. Method

Two independent sources, kept separate:

1. **Original network.** The captured bindings and parameter blocks of the four
   image stages (EXP-011/EXP-014/EXP-015/EXP-016) and the 24-frame edge audit in
   EXP-011. Roles come from parameter offsets, not from shader guesswork.
2. **Reconstructed graph.** The stage manifests under
   `%LOCALAPPDATA%/Temp/opencode/m-capture-run/fixtures`, the shader declarations,
   the fixture validators (`scripts/m_stage_fixture.py`), and the host's resource
   lifecycle in `src/native/m_image_stage.cpp` / `m_network_stage.cpp`.

The M10 sequence driver (`scripts/m10_temporal.py`) then executes frames and
captures the tensors the graph still writes, so the inventory's "produced" claims
are measured rather than inferred.

## 3. Original network: persistent state

Every row is `[M]` from captured bindings/content except where marked. "Frame N"
is the producing frame; all consumers are in frame N+1.

| # | State | Producer → consumer | Format / geometry | Bytes | Init | Reset behaviour | Evidence |
|---:|---|---|---|---:|---|---|---|
| 1 | Reconstruction history (ping-pong pair) | `post` `+96` → next frame `post` `+56` | 2880×1620, RGBA16F bytes (2×`R32_UINT`), 8 B/texel | 37,324,800 | zero on frame 1 | `reset != 0` forces the sample to the colour texel; the history fetch is skipped | EXP-011 §4, EXP-016 §2, EXP-022 |
| 2 | Half-resolution surface | `post` `+112` → next frame `enc0` `+152` | 1440×810, packed 10:10:10:2 in `R32G32B32A32_FLOAT`, 16 B/texel | 18,662,400 | zero on frame 1 | history slot zeroed by `reset` | EXP-011 §3–4 |
| 3 | Recurrent feature | `dec0` `+128` → next frame `enc0` `+160` and `dec0` `+64` | 960×540, SNORM8×4 | 2,073,600 | zero on frame 1 | `reset` zeroes the history term of `(1-g)·h + g·tanh(v)` | EXP-011 §4, EXP-014 |
| 4 | Motion A/B/C | `enc0` `+280/288/296` → next frame `enc0` `+96` | 960×540, RG16F, 3 surfaces | 3 × 2,073,600 | zero on frame 1 | not established | EXP-011 §3–4, EXP-022 |
| 5 | Previous motion input | external / frame N−1 → `enc0` `+96` | 960×540, RG16F | 2,073,600 | zero | zero on the captured frame | EXP-011 |
| 6 | Frame index | host → `enc0` `+312` | `uint` | 4 | 0 | indexes the dither table; not reset-dependent | EXP-015, `m_enc0_model.py` |
| 7 | Reset flag | host → `enc0` `+316`, `dec0` `+120`, `post` `+168` | `uint`/`int` | 4 | 1 | the switch that invalidates rows 1–5 | EXP-014/015/016 |
| 8 | Jitter | host → `enc0` `+168` | `float2` | 8 | captured constant | per-frame camera offset; not advanced on the captured frame | EXP-015 |
| 9 | Exposure / 1×1 textures | host → `enc0` `+112/+120` | 1×1 f16 ×2 (unidentified) | 4 each | unknown | unknown | EXP-011 §3, EXP-022 |
| 10 | Motion scale/offset, display extent, window maxima, history size/inverse | host → params | scalars | — | captured constants | constant | EXP-015/016 |

[M] Rows 1–4 are the four captured cross-frame edges; EXP-011 verified each
hand-off byte for byte over 24 frame pairs (row 4 only as all-zero content).

[H] Replay ordering. The three captured replays are not in recorded temporal
order, and no per-launch `reset`/frame index was captured (EXP-022). Row 1 is the
only edge whose producer and consumer were observed nonzero; its link is
`out(3) = in(2)`, consistent with a `3 → 2 → 1` order.

## 4. Reconstructed graph: what actually executes

| # | State | Native producer | Native consumer | Native format | FP16 vs INT8 |
|---:|---|---|---|---|---|
| 1 | `post` history | **yes** — `out-history.bin` on every frame | **none** — `post/stage.txt` declares five SRVs and no history input | 5760×1620 `R32_UINT`, 37,324,800 B | identical shader; values differ only because `dec0` inputs differ |
| 2 | `post` half surface | **yes** — `out-half.bin` | **none** — no fixture binds it | 1440×810 `R32G32B32A32_FLOAT`, 18,662,400 B | same |
| 3 | `dec0` feature | **yes** — `out-feat.bin`, written by the epilogue | **none** — `dec0/stage.txt` binds weights and tokens only | 960×540 `R32_UINT` (SNORM8 bytes), 2,073,600 B | written by both; INT8 values come from the INT8 expansion |
| 4 | `enc0` motion A/B/C | **no** — neither fixture declares them; INT8 `enc0` emits tokens and diagnostics only | **none** | — | absent in both; the original writes zeros anyway |
| 5 | `enc0` previous motion, half history, feature | **no** — not bound by either `enc0` fixture | — | — | absent |
| 6 | Frame index | **constant 0** in `params.bin` | dither table | `uint` | same in both |
| 7 | Reset | **constant 1**, validated | all three stages | `uint` | same in both |
| 8 | Jitter | constant from the capture | `enc0` | `float2` | same in both |

[S] The host is a one-shot stage runner. `m_image_stage.cpp` allocates every
resource inside `Runner::Run`, uploads inputs, dispatches once, copies every UAV
to a readback buffer, waits on one fence (`+492–494`), writes files and exits.
`m_network_stage.cpp` does the same for the token graph. No resource, allocator,
descriptor heap or fence survives a process, so there is no in-process
ping-pong and no GPU-side state serialization.

[S] Output resources are default-heap and are either left in `COMMON` or seeded
with a sentinel pattern when `--seed` is given (`+347–376`); every declared
output is then copied out completely. This is a complete-write check, not a
state restore.

[S] `m_stage_fixture.py` rejects `reset != 0` for `enc0` and `post`, and
`m_image_stage.cpp` rejects it for `dec0` INT8 (`+305–306`). The fixture
validators are the executable form of the reset-only contract.

## 5. Gate: lifecycle completeness

| Class | Elements |
|---|---|
| Lifecycle documented (producer, consumer, format, init, reset) | 1, 2, 3, 5, 6, 7, 8, 10 |
| Lifecycle documented, semantics not established | 4 (written values unobserved; unknown whether the kernel writes) |
| Explicitly unresolved | 9 (two 1×1 textures), original frame ordering (EXP-022), `reset = 0` arithmetic for rows 1–4 |

**First missing dependency for real temporal validation:** an *executable*
consumer. Rows 1–3 are written by the reconstruction and never bound; row 4 is
not produced. Binding them requires the `reset = 0` branches, which are absent
from all three shaders (`post_cs.hlsl` lines 311–344 selects the colour fallback
unconditionally; `dec0_cs.hlsl` lines 323–324 drop the history term;
`enc0_cs.hlsl` implements the reset path only). Implementing them needs a
labelled non-reset frame and a nonzero-motion capture, neither of which exists.

**Consequence:** repeated frames can be executed and compared, but they are
reset frames. No cross-frame history hand-off can currently change any output.
This is a bounded-execution finding, not temporal correctness.

## 6. Interpretation

* [S] The original network has four persistent cross-frame tensors plus per-frame
  bookkeeping (frame index, reset, jitter, exposure).
* [S] The reconstruction produces three of those tensors and consumes none; the
  fourth and all half/feature/motion *inputs* are not bound at all.
* [S] State initialisation and invalidation are hard-wired constants
  (`frame = 0`, `reset = 1`), not a runtime sequence.
* [M] The three produced tensors are deterministic per frame and differ between
  FP16 and INT8; EXP-032 quantifies that divergence. They are the state a future
  temporal loop would consume.
* [U] Whether `reset = 0` would reproduce the original arithmetic, and what the
  original motion/frame-index/exposure semantics are.

## 7. Next

1. Run the frame-level sequence oracle (EXP-031) over the reset contract.
2. Measure per-frame and per-state divergence (EXP-032).
3. Define checkpoint/replay over inputs, boundaries and state hashes (EXP-033).
4. If cross-frame reuse is ever authorised, the first required artefact is a
   labelled two-frame capture (`reset = 1` then `0`, non-zero history and
   motion), followed by `dec0`'s recurrent branch, then `post`'s Catmull-Rom
   path, then `enc0` motion. Do not implement those from inference alone.

## 8. Limitations

* One DLL build, one preset, one resolution, one device, one synthetic scene.
* Original-state claims come from captured bindings and content, not from a
  runtime trace of a live sequence.
* The two 1×1 exposure textures and the motion write path remain unidentified.
* No claim is made about HDR, `mvlo`, dynamic resolution, or real game frames.
