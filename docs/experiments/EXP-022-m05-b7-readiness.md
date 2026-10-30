# EXP-022 — B7 readiness: what the multi-frame capture does and does not contain

**Milestone:** M05 (B7 / C22) · **Status:** INCONCLUSIVE for temporal validation;
the audit itself is complete and its claims are measured. B7 stays **NOT STARTED**.

## 1. Question

Temporal validation needs two consecutive frames with a real history handoff. Before
building any of it: which temporal resources does the existing capture contain, at
what geometry and format, and is a producer→consumer handoff directly observable?

## 2. Method

`scripts/m_b7_readiness.py` audits a capture directory of `replay-NNNNNN-<kernel>`
folders. For each launch the hook recorded, per bound image, its parameter offset, kind
(texture/surface), geometry, format code, pitch and byte count, plus the bytes behind
it. The script resolves the temporal roles from the offsets already established by the
stage contracts — `post` `+56` history in / `+96` history out / `+112` half-resolution
out ([EXP-016](EXP-016-m05-post-reconstruction.md)), `dec0` `+56`/`+64` recurrent
feature in / `+128` out ([EXP-014](EXP-014-m05-dec0-mapping.md)), `enc0` `+96` previous
motion in / `+280`/`+288`/`+296` motion out ([EXP-015](EXP-015-m05-enc0-contract.md)) —
and then hashes and byte-tests every replay, and every producer/consumer pair.

```powershell
python scripts/m_b7_readiness.py <capture-img> --json b7-readiness.json
```

## 3. Results

[M] The image capture holds three replays of every stage except the tube (18 replays):
`enc0`, `dec0`, `post_3_1`, `downsample`, `dec1`, `dec2`, `enc1`, `enc2`.

| Stage | Role | Geometry | Bytes | Distinct contents | Zero on replay |
|---|---|---|---:|---:|---|
| `post` | history in (`+56`) | 2880x1620, 8 B/texel | 37,324,800 | 3 | 1 |
| `post` | history out (`+96`) | 2880x1620, 8 B/texel | 37,324,800 | 2 | 1, 2 |
| `post` | half out (`+112`) | 1440x810, 16 B/texel | 18,662,400 | 3 | 1 |
| `dec0` | feature in, f16 view (`+56`) | 960x540, 4 B/texel | 2,073,600 | 1 | 1, 2, 3 |
| `dec0` | feature in, SNORM8 (`+64`) | 960x540, 4 B/texel | 2,073,600 | 3 | 1 |
| `dec0` | feature out, SNORM8 (`+128`) | 960x540, 4 B/texel | 2,073,600 | 2 | 1, 2 |
| `enc0` | previous motion in (`+96`) | 960x540, 4 B/texel | 2,073,600 | 1 | 1, 2, 3 |
| `enc0` | motion A/B/C out (`+280/288/296`) | 960x540, 4 B/texel | 2,073,600 | 1 | 1, 2, 3 |

[M] **History is double-buffered and the buffers rotate.** The bound image addresses
differ per replay (`post` history in: `0x2708000000`, `0x2708002000`, `0x2708000000`;
history out: `0x2700d40000`, `0x2700d44000`, `0x2700d40000`), so a replay's output and a
replay's input are never the same surface within one replay.

[M] **One nonzero handoff is bit-exact, in two independent stages:**

| Handoff | Byte sizes | Equal pairs (producer replay → consumer replay) |
|---|---|---|
| `post`.history out → in | equal (37,324,800) | `3 -> 2` nonzero; `1 -> 1`, `2 -> 1` equal only because both are zero |
| `dec0`.feature out → in | equal (2,073,600) | `3 -> 2` nonzero; `1 -> 1`, `2 -> 1` equal only because both are zero |

[M] Nothing else links: for `post`, `history out` of replays 1 and 2 is all zero while
`history in` of replays 2 and 3 is not, so the `1 -> 2` and `2 -> 3` links fail; for
`dec0` the same pattern holds. `enc0`'s motion chain is uninformative: the previous-motion
input and all three motion outputs are bit-identical zeros on all three replays.

[H] **Replay order.** The only consistent reading of the equal pairs is that replay
numbering is *not* temporal order: taking replays 3 → 2 → 1, the nonzero handoff
`out(3) = in(2)` and the zero handoff `out(2) = in(1)` both hold. The colour texture is
bit-identical across all three replays, so the scene does not change and the three
replays differ only in history state. Which replay is which game frame, and which
replays were invoked with `reset = 0`, is **not recorded anywhere in the capture**.

[U] **Motion geometry.** Every motion image is 960x540 RG16F, and every motion value is
zero, on every replay. `post` samples motion at up to (1919, 1079)
(`trunc(pos/recon * display)`) from a 960x540 texture ([EXP-016](EXP-016-m05-post-reconstruction.md)
§6). Zero motion hides whether the texture is a half-resolution motion field with an
implicit 2x scale, or a full-resolution field captured at the wrong extent.

## 4. What the audit establishes about each dependency

| Prerequisite for B7 | State |
|---|---|
| History resources identified with geometry/format | **known** for `post` (2880x1620, 8 B/texel, 37,324,800 B) and `dec0` (960x540, 4 B/texel) |
| Consecutive-frame inputs recoverable | **partly** — one nonzero handoff per stage is captured bit-exactly; the rest of the links fall on zero content |
| Replay ↔ frame ordering | **missing** — no frame index or reset flag is recorded per launch |
| Which data are overwritten between launches | **known** that the buffers rotate per replay; which surface the next launch will bind is not recorded |
| `reset = 0` branches | **not implemented** in any native stage: `post` implements the reset path only and its fixture validator rejects `reset != 0`; the `dec0` recurrent-feature branch is unimplemented; `enc0`'s motion outputs are written as zeros |
| Motion texture geometry and sampling bounds | **unresolved** (see [U] above) |
| State that must persist between frames | `post` history (37,324,800 B), `dec0` SNORM8 recurrent feature (2,073,600 B, and its f16 view), `enc0` previous motion (2,073,600 B) — plus which surface of each pair is bound as input next |
| Output surfaces that feed the next frame | `post` `+96` → next frame's `+56`; `dec0` `+128` → next frame's `+64`; `enc0` motion out → next frame's `+96` (unverified, all zeros) |

**First missing dependency:** the per-launch temporal bookkeeping. The capture records
pointer values and bytes, but not the frame index, the `reset` flag, or the identity of
the history surface as seen by the *next* launch. Without that, no two-frame chain can
be assembled with confidence, and the one link that is bit-exact cannot be labelled as
"frame k → frame k+1".

## 5. Minimal B7 plan (not started)

1. **Capture a labelled two-frame chain.** Extend the existing hook to record, per
   launch, the frame index and the `reset` flag from the launch's parameter block, and
   the surface identity each temporal binding resolves to. Two consecutive frames with
   `reset = 1` then `reset = 0`, same content, is enough.
2. **Persistence harness.** A single process that holds the three persistent buffers
   between two native pipeline runs and swaps them per frame, with the swap recorded, so
   the handoff is exercised by construction rather than inferred.
3. **Implement the missing branches** in this order: `dec0` recurrent feature (SNORM8
   in/out, simplest contract), then `post` history and Catmull-Rom reprojection, then
   `enc0` previous motion.
4. **Resolve the motion geometry** against a non-zero motion capture before any branch
   that samples motion.
5. **First falsifiable test:** feed frame 2's captured history to the native `post`
   implementation with `reset = 0` and require the reconstruction surface to match the
   capture byte-for-byte or with a stated, attributed residual.

Do not treat three captured frame edges as validated temporal execution: the frame
above shows how much of the temporal path they leave unbound.

## 6. Limitations

- The audit reads only what the capture recorded; a role absent from the manifest is
  reported absent rather than inferred from a shader.
- "Distinct contents" counts hashes within the replays present; three replays cannot
  distinguish rotation period 2 from a longer ring.
- The role names come from the established stage contracts, not from a fresh PTX read;
  a wrong role mapping would change which offsets are hashed (the content claims
  themselves are independent of the naming).
- Local artifacts stay under `%TEMP%`; no capture bytes are tracked.
