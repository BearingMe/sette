# EXP-031 — Deterministic multi-frame oracle

**Milestone:** M10-B · **Status:** PASS for the bounded reset-frame sequence.
Cross-frame history reuse remains **BLOCKED** (see §6 and EXP-030).
**Depends on:** [EXP-029](EXP-029-m09-complete-int8.md),
[EXP-030](EXP-030-temporal-state-inventory.md).
**Evidence:** `m10-sequence-20261030/sequence.json`,
`m10-sequence-20261030.log`.

## 1. Question

Can reproducible multi-frame inputs be executed through the complete FP16 and
INT8 graphs with identical initial conditions, so that frame ordering, input
identity, state initialisation, reset behaviour and determinism are evidenced
rather than assumed?

## 2. Method

`scripts/m10_temporal.py` drives the unmodified approved harness
(`scripts/m_b6_frame.py`) once per frame and mode. `m_b6_frame.py` gained one
backward-compatible option, `--colour`, which clones the `enc0` fixture and
replaces its 960×540 RGBA16F input; the shared fixture is never modified and the
default path is unchanged (359 tests pass, M09 frame hashes unchanged).

* **Sequence:** 12 frames, each a deterministic content variant of the same
  canonical colour: static, identical hold, ±translation, 3° rotation, fast
  translation, a 96×96 reveal patch, a tiled texture, 1-px diagonals, an inverted
  abrupt change, recovery to canonical, and a final hold.
* **Identity:** both modes receive the exact same colour file (SHA-256 recorded
  per frame); all weights, layers, shaders and host binaries are hashed.
* **Reset:** every frame is a `reset = 1` frame, the only implemented contract.
* **State:** the three persistent tensors the graph still writes are captured
  and hashed after every frame; full bytes are kept for frames 0, 5 and 11.
* **Replay:** frames 0, 6 and 11 are re-executed from their recorded inputs and
  must reproduce all 33 boundary hashes, the frame hash and all three state
  hashes.

```powershell
python scripts/m10_temporal.py --out "$T/m10-sequence-20261030" --frames 12 `
  --replay 0,6,11 --checkpoint 0,5,11
```

## 3. Results

[M] **Frame 0 is the accepted M09 frame, byte for byte.** FP16
`dc97c4cb…d6e95b`, INT8 `f76ac032…4d6c00` — the same hashes accepted in
EXP-029 — even though this run used a cloned colour fixture and a different
output directory.

[M] All 12 frames executed in both modes: every declared stage present, 33
boundary hashes per frame, every output the declared size, zero non-finite or
saturated samples. The machine checks were `first_frame_matches_accepted_m09`,
`all_frames_executed`, `frame_finite`, `state_finite`, `replay_reproduces`,
`checkpoints_kept`: **all true**, overall `PASS`.

[M] Replay of frames 0, 6 and 11 reproduced, for both modes:

| Frame | Frame hash | 33 boundaries | 3 state tensors |
|---:|---|---|---|
| 0 | identical | identical | identical |
| 6 | identical | identical | identical |
| 11 | identical | identical | identical |

[M] **Reset recovery is exact.** Frames 0, 1, 10 and 11 all contain the canonical
colour (10 after an abrupt inverted frame, 11 after recovery). Their FP16 and
INT8 frame hashes are identical across all four, so a preceding frame cannot
leave a trace once the reset contract is used.

## 4. Inputs and provenance

[M] The colour variants are generated deterministically from
`fixtures/enc0/in-colour.bin` (SHA-256 recorded per frame), with alpha preserved
and RGB clipped to [0, 1]; no random state is used. The transformations change
*pixels*, not the motion vectors: the reset branch ignores motion, so these are
**content variants, not validated camera motion**. That distinction is the
subject of §6.

[M] The layer fixtures, weights and graph are the accepted
`m09h-complete-chain-20261030/layers`; the runtime modes are the accepted
`int8-embedding` + `int8` attention + `spatial` cumulative graph. The FP16 mode
is the unchanged default graph.

## 5. Gate

| Criterion | Result |
|---|---|
| Repeatable, comparable execution | PASS — replay reproduces exactly |
| Documented fixture provenance | PASS — per-frame colour SHA, layer/weight/host hashes in `sequence.json` |
| Frame ordering | PASS — each frame is keyed by index and scenario |
| Input identity | PASS — one colour file per frame, shared by both modes |
| State initialisation | PASS for the reset contract — state is produced on frame 0 and is deterministic |
| Reset behaviour | PASS — exact recovery after an abrupt frame |
| Output finiteness | PASS — 0 non-finite, 0 saturated |
| Determinism | PASS — 12/12 frames, 3/3 replays, 33/33 boundaries |

## 6. What this oracle does **not** cover

**BLOCKED:** a frame that consumes the previous frame's history. No stage binds
history/feature/motion inputs (EXP-030 §4), the `reset = 0` branches are absent
from `post`, `dec0` and `enc0`, and the B7 capture has no labelled non-reset
frame (EXP-022). Consequently:

* camera translation/rotation here means different pixels, not motion-vector
  reprojection;
* disocclusion is a pixel patch, not a history-validity test;
* state reuse is file-level capture, not GPU-side consumption;
* no claim is made about temporal behaviour outside the evaluated fixtures.

The sequence is therefore an oracle for **frame independence under reset**, which
is a prerequisite for temporal work but not a substitute for it.

## 7. Interpretation

* [M] Complete FP16 and INT8 frames reproduce the accepted M09 outputs from
  independent invocations, so the frame pipeline is deterministic and portable
  across output directories.
* [M] The reset contract isolates frames exactly: identical inputs give identical
  hashes regardless of sequence position or preceding content.
* [S] The reconstructed host has no in-process frame loop; persistence is
  file-level (EXP-030). That is sufficient for this gate and insufficient for
  history reuse.

## 8. Limitations

* One scene family, one resolution, one device, one DLL build.
* Motion, exposure, jitter and frame-index advancement are not exercised.
* PSNR against the canonical reference is meaningful only for the canonical
  frames; other frames are compared FP16↔INT8 only (EXP-032).
