# EXP-033 — Temporal checkpoints and replay

**Milestone:** M10-D · **Status:** PASS for input/state capture and replay of the
reset contract. Restoring state into a consumer is **not possible** (no consumer).
**Depends on:** [EXP-030](EXP-030-temporal-state-inventory.md),
[EXP-031](EXP-031-temporal-oracle.md).
**Evidence:** `m10-sequence-20261030/sequence.json`,
`m10-sequence-20261030/state/{fp16,int8}/frame-{00,05,11}/`.

## 1. Question

Can a frame in a sequence be captured, restored and replayed deterministically,
and can its persistent tensors be validated across repeated runs — using the
minimum mechanism over existing infrastructure?

## 2. Mechanism

No new execution framework. A checkpoint is a JSON record plus the existing
output files:

* **Inputs:** frame index, scenario, absolute colour path and its SHA-256.
* **Configuration:** mode, cumulative layer directory, fixture directory.
* **Execution identity:** every declared boundary hash (33), the final frame
  hash, and all three persistent-state hashes.
* **State bytes:** `post` history, `post` half surface and `dec0` feature are
  copied to `state/<mode>/frame-NN/` for frames 0, 5 and 11. Other frames keep
  hashes only.

Replay is `m_b6_frame.py` re-invoked with the recorded colour file; it is the
same code path as a fresh run, not a special "replay mode". A replay check
requires the frame hash, every boundary hash and every state hash to match.

## 3. Results

[M] Three checkpoints per mode were kept. SHA-256 (first 10 hex):

| Frame | Mode | history | half | feature |
|---:|---|---|---|---|
| 0 | fp16 | `fc55d28985` | `a72f161cde` | `e7f74daaa5` |
| 0 | int8 | `9ec2e4c05f` | `12b3f1c789` | `045b5d9aaa` |
| 5 | fp16 | `a942b8cb88` | `2ae24a4022` | `8e3996f889` |
| 5 | int8 | `2e6d2f2090` | `788bed0527` | `6cf1d64311` |
| 11 | fp16 | `fc55d28985` | `a72f161cde` | `e7f74daaa5` |
| 11 | int8 | `9ec2e4c05f` | `12b3f1c789` | `045b5d9aaa` |

[M] Replay of frames 0, 6 and 11 in both modes reproduced **the frame hash, all
33 boundary hashes and all three state hashes** with zero differing boundaries.
Every checkpoint is therefore a valid replay target.

[M] Frame 0 and frame 11 checkpoints are identical in both modes because the
content is identical; frame 5 (fast translation) differs. This is the expected
behaviour of a deterministic, reset-isolated producer.

[M] An independent reset is exact: replaying frame 11 without frame 10 in its
process reproduces the in-sequence result byte for byte. The host executes one
frame per process, so there is no hidden in-process state to restore.

## 4. What "restore" can and cannot mean here

[S] The three captured tensors are *produced* by `post` and `dec0` and never
*bound* by any stage (EXP-030 §4). A checkpoint can restore the bytes, but no
shader reads them, so a restored state cannot change an output. A test that
claimed "state restore validated" from this would be vacuous.

[U] Restoring state into a `reset = 0` consumer. That requires implementing the
missing branches and a labelled non-reset capture before a checkpoint can be
meaningful; see EXP-030 §5 and EXP-022.

## 5. Gate

| Criterion | Result |
|---|---|
| Captured | PASS — inputs, configuration, 33 boundaries, frame, 3 state tensors |
| Restored | PASS for bytes; **not observable by any shader** |
| Replayed | PASS — 3/3 frames × 2 modes |
| Compared | PASS — bit-exact frame/boundary/state hashes |
| Reset independently | PASS — frame 11 reproduces without frame 10 |
| Validated across repeated runs | PASS — deterministic over fresh processes |

Determinism limit: equality is byte-exact for this device, build and fixtures.
It is not a claim of cross-device or cross-driver reproducibility.

## 6. Reproduction

```powershell
python scripts/m10_temporal.py --out "$T/m10-sequence-20261030" --frames 12 `
  --replay 0,6,11 --checkpoint 0,5,11
```

The driver writes `sequence.json` after every frame and after every replay, so a
partial run retains its checkpoints. `--replay` and `--checkpoint` accept any
in-range indices; replaying a frame does not require its predecessors.

## 7. Interpretation

* [M] Checkpoint/replay over the existing harness is sufficient for
  deterministic reproduction of any evaluated frame.
* [S] It captures the producer side of the temporal state completely and the
  consumer side not at all, because the consumer does not exist.
* [U] Whether a restored recurrent state would reproduce the original network's
  frame 2.

## 8. Limitations

* Three of twelve frames keep state bytes; the rest keep hashes only.
* No GPU-side serialization: checkpoints are files, and the graph has no
  persistent GPU resource across frames.
* One device and build.
