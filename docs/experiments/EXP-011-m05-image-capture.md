# EXP-011: Image-Backed Capture for `enc0`, `dec0`, `post` and `downsample` (M05 B1-B4)

**Milestone:** M05 (checkpoints B1/C16, B3/C18 groundwork, B4/C19 prerequisite)
**Status:** PASS for the capture extension and for image-level resource validation
**Supersedes:** the "texture/surface contents are not captured" limitation recorded in
[EXP-008](EXP-008-m05-full-capture.md), [EXP-010](EXP-010-m05-native-fp16-stage.md) and
[the stage graph](../findings/2026-10-29-m05-stage-graph.md)

## Question

`enc0`, `dec0`, `post_3_1` and `downsample` bind textures and surfaces whose contents the
buffer-only replay hook could not read, so those four stages had no captured operand or
result. Acceptance criteria, fixed before execution:

1. Every image-backed resource those stages bind is resolved to its backing storage, with
   dimensions, format, channels, row pitch and byte extent recorded.
2. The readback reproduces known contents byte for byte, including a nontrivial pitch, a
   3D array, and an alias where one array is bound through two handles.
3. Capturing does not modify the resource: a kernel that writes nothing must leave both the
   pre- and post-launch image identical.
4. The extension leaves the existing buffer capture unchanged.
5. At least one captured image is validated against an oracle that does not come from the
   capture path itself.

## Method

### Where the pixels are

[S] A ZLUDA texture object is HIP's 272-byte device object and a surface object is a
96-byte device block ZLUDA builds in `zluda/src/impl/surf.rs`. The launch argument holds
that device handle, not the image, and HIP's array is behind it. Three facts made a
readback possible without a new GPU path:

- `cuArrayCreate_v2` / `cuArray3DCreate_v2` see every array's geometry at creation, so the
  geometry can be recorded then instead of guessed from a descriptor;
- `cuTexObjectCreate` / `cuSurfObjectCreate` already store the `HIP_RESOURCE_DESC` they were
  given (`tex.rs`, `surf.rs`), which maps a handle back to its array or pitched allocation;
- `hipMemcpyParam2D` / `hipDrvMemcpy3D` copy an array to host memory without touching it.

### The patch

[`patches/zluda/0018-m05-image-capture.patch`](../../patches/zluda/0018-m05-image-capture.patch)
adds `zluda/src/impl/image_capture.rs`, an array registry in `array.rs`, a
`resource_desc()` accessor in `tex.rs` and `surf.rs`, and the image half of the replay hook.
It applies after d4r's audited ZLUDA series (see [patches/README.md](../../patches/README.md)).

The hook keeps the old `alloc`/`pointer` manifest lines unchanged and adds:

```text
image <id> <tex|surf> <handle> <array|0> <pitched|0> <w> <h> <depth> <format> <channels> <original> <row_bytes> <bytes> <pitch>
image-bind <id> <arg offset> <tex|surf>
image-unknown <arg offset> <handle> <reason>
```

One target per backing array: the same array bound as a texture and as a surface is one
`image` with two `image-bind` lines, which is what makes aliasing visible. A binding that
cannot be read is named, never dropped. `D4R_CUDA_IMAGE_DUMP=0` restores the buffer-only
capture, and nothing runs when `D4R_CUDA_REPLAY_DUMP_DIR` is unset.

### Environment

Windows 11, RX 7700 XT (gfx1101), AMD driver 32.0.32015.7029, ROCm/HIP 7.2.60201,
DXC 1.8.2502.11, ZLUDA `ee2f25a` + d4r patches + this patch (`nvcuda.dll` sha256
`F2880A56…DDD98C`), `nvngx_dlss.dll` 310.7.0, preset 13 (`rrlite_…_mvhi_ldr_folded`),
1920x1080 output with a 960x540 render.

## Results

### 1. Synthetic round-trip (acceptance criteria 2 and 3)

[M] `scripts/m_image_capture_test.py` — two separate processes (the hook reads its output
directory once per process), 3 targets, 0 problems, image contents byte-identical between
runs:

| Case | Shape | Storage | Result |
|---|---|---|---|
| 2D RGBA16F | 37x13, 296-byte rows | `0x10` x4 | pre == upload (padded 328-byte host pitch), post == pre |
| 2D UNORM_INT8X4, bound as texture **and** surface | 5x7, 20-byte rows | backing `0x01` x4, original `194` | one target with two bindings, pre == upload, post == pre |
| 3D HALF | 3x4x2 | `0x10` x1 | pre == upload, post == pre |

The manifest recorded the driver's own geometry and the backing format (`0x01` for the
D3D11-style `194`), not the requested one — the mapping is visible in the capture instead
of being applied silently.

[U] A hand-written surface store (`sust.2d.b32` and `sust.2d.v4.b32`) fails to compile in
this ZLUDA build (`cuModuleGetFunction` → 500 for both, while the same module with `ret`
loads), so the synthetic test does not prove that a *written* image changes. `dec0`'s four
NVIDIA-written surfaces do (see 3), which is better evidence anyway.

[U] `PITCH2D` and `LINEAR` resources have a readback path in the patch, but this driver
rejects every `PITCH2D` texture descriptor tried (7 shapes, `cuTexObjectCreate` → 1,
including a `cuMemAllocPitch` allocation with the pitch it returned), so that path is
unexercised. No captured NGX binding uses it: all 46 bindings in this frame are arrays.

### 2. Frame capture (acceptance criteria 1 and 4)

[M] One 1920x1080 frame, 14 launches, 9 stage groups, 33 image targets. `enc0` 11,
`dec0` 6, `post_3_1` 13, `downsample` 3. No `image-unknown` line in any stage.

The probe was allowed 25 frames (24 readbacks); the capture filled the volume (23.2 GB) and
the probe's last frame failed to save (`FAIL D3D12_DLSS Saving DLSS output failed`,
exit 4). Frames 1-3 are kept (3.7 GB) and the 25-frame temporal checks below ran before
pruning. **The capture size is now a real constraint**: one frame is ~0.95 GB.

Buffer capture is unchanged: [M] `enc0`'s 12 416-byte weight blob and its 8 355 840-byte
token output are byte-identical to the pre-extension 1080p capture, and all 14 buffer edges
still verify.

### 3. What each stage binds (acceptance criteria 1 and 5)

[M] Formats are the driver's backing storage codes; `orig` is CUDA's requested enum.
Changed bytes are pre-vs-post within the launch.

| Stage | Slot | Kind | Geometry | Format | Role in this frame |
|---:|---|---|---|---|---|
| enc0 | 80 | tex | 960x540x1 | `0x10`x4 | colour, unchanged (probe pattern, see 5) |
| enc0 | 96 | tex | 960x540x1 | `0x10`x2 | display motion, all zero |
| enc0 | 112, 120 | tex | 1x1 | `0x10`x1 | two 1x1 f16 textures, both 0.0 (unidentified) |
| enc0 | 128 | tex | 960x540x1 | `0x10`x2 | motion candidates, all zero |
| enc0 | 136 | tex | 960x540x1 | `0x20`x1 | depth, 0.5 everywhere |
| enc0 | 152 | tex | 1440x810x1 | `0x20`x4, orig 80 | half-resolution history, packed 10:10:10:2 |
| enc0 | 160 | tex | 960x540x1 | `0x08`x4, orig 200 | recurrent feature, SNORM8 |
| enc0 | 280-296 | surf | 960x540x1 | `0x10`x2 | three motion surfaces, zero before and after, **unchanged** |
| dec0 | 56 | tex | 960x540x1 | `0x10`x2 | motion |
| dec0 | 64 | tex | 960x540x1 | `0x08`x4 | previous feature |
| dec0 | 128 | surf | 960x540x1 | `0x08`x4, orig 200 | feature out, 2 064 707 B changed |
| dec0 | 136 | surf | 960x540x1 | `0x01`x4, orig 194 | gates, 1 555 200 B |
| dec0 | 144 | surf | 960x540x1 | `0x01`x4, orig 194 | mix, 1 527 684 B |
| dec0 | 152 | surf | 960x540x1 | `0x10`x4 | covariance, 3 106 845 B |
| post_3_1 | 96 | surf | 2880x1620x1 | `0x10`x4 | history out, 21 805 383 B |
| post_3_1 | 112 | surf | 1440x810x1 | `0x20`x4, orig 80 | half-res out, 13 416 997 B |
| downsample | 32 | surf | 1920x1080x1 | `0x10`x4 | final image, 13 538 212 B |

[M] Roles are content-based, so a rewrite with identical bytes cannot be told from no
write. `enc0`'s three `0x10`x2 surfaces are zero before **and** after the dispatch: in the
`mvhi` (display-motion) variant they do not receive motion candidates from this kernel, or
they are rewritten with identical zeros. A nonzero-motion capture would settle it. This
corrects the earlier statement that `enc0`'s three motion outputs were merely *uncaptured*.

### 4. Edges inside the frame and across frames

[M] Image edges use the same rule as buffer edges: identical bytes, producer before
consumer, and now a real allocation identity (the array) rather than a pooled address.

| Edge | Evidence |
|---|---|
| `dec0` gates → `post` gateTex | identical, 2 073 600 B |
| `dec0` mix → `post` mixTex | identical, 2 073 600 B |
| `dec0` covariance → `post` covTex | identical, 4 147 200 B |
| `dec0` feature (frame N) → `enc0` feature (frame N+1) | identical, 24/24 frame pairs |
| `post` history (frame N) → `post` history input (frame N+1) | identical, 24/24 (ping-pong between two 2880x1620 arrays) |
| `post` half-res (frame N) → `enc0` half-res (frame N+1) | identical, 24/24 |
| `enc0` motion A (frame N) → `enc0` motion input (frame N+1) | identical, 24/24 |

[M] `enc0`'s inputs with no in-frame producer are external or temporal: colour, depth,
display motion, the two 1x1 textures, and the previous frame's feature and history. The
audit reports them as `EXTERNAL` instead of inventing a producer.

### 5. Independent oracles (acceptance criterion 5)

[M] `scripts/m_image_validate.py` compares against evidence that does not come from the
capture path:

| Check | Result |
|---|---|
| `enc0` colour equals the probe's checkerboard over all 518 400 pixels, 4 channels | 0 differing per channel |
| `enc0` depth is the constant 0.5 the probe uploads | true |
| `enc0` motion is zero | true |
| frame-1 recurrent feature and half-resolution history start empty | 0 non-zero elements |
| `downsample` output equals the probe's D3D12 readback `output-0.rgba16f` | **0 of 16 588 800 bytes differ** |
| `dec0` surfaces match `dec0_m.comp`'s declared formats and geometry | 4/4 |

The `downsample` result is the strongest: a CUDA array copied out by HIP equals, byte for
byte, the frame the probe read back from the D3D12 output texture through a different API.
The captured final image *is* the network's output frame, so post/downsample can now be
validated against a real reference frame rather than against `post`'s internal buffer.

## Interpretation

- [M] The image-capture blocker is closed. All four image-bound stages have captured
  operands and results with machine-checkable metadata, and two independent oracles agree
  with the capture.
- [M] The `enc0`/`dec0`/`post`/`downsample` bindings are now known by slot, kind, geometry,
  format and content, which is what a stage reconstruction needs (C1-C3).
- [M] Persistent temporal state is real and exact: feature, history and motion buffers are
  handed from one frame to the next byte for byte, with the 2880x1620 history ping-ponging
  between two arrays.
- [U] Whether `enc0` writes its three motion surfaces cannot be decided from zero-motion
  input.
- [U] The two 1x1 `0x10`x1 textures bound by `enc0` are unidentified; the probe's 1x1
  exposure texture is `R32F`, so they are not that texture.

## Limitations

- [M] One DLL version, one adapter, one preset, one geometry, 25 frames of a static
  synthetic scene with zero motion. Nothing here is evidence about `mvlo`, HDR, dynamic
  resolution, another GPU or a real game frame.
- [M] The 25-frame capture exhausted the volume; later frames may be absent, and the
  probe's own final readback is truncated. The kept frames are complete.
- [U] `PITCH2D`/`LINEAR` image readback is implemented but could not be exercised: this
  driver rejects `PITCH2D` texture objects.
- [U] The pre-image used to verify `patches/zluda/0018-m05-image-capture.patch` was
  reconstructed from the audited files, not from a fresh clone, because the workspace holds
  d4r's ZLUDA delta as uncommitted modifications. The patch applies cleanly to that
  reconstruction and reproduces the built sources (line endings aside).
- [M] Capture overhead is large: 0.95 GB and roughly 8 s of extra work per 1080p frame, so
  temporal runs must bound their frame count.

## Next

- C1 `enc0`, C2 `dec0`'s epilogue and C3 `post`/`downsample` can now be reconstructed and
  compared stage by stage against these captures.
- A nonzero-motion frame (or an `mvlo` frame) would settle the `enc0` motion-surface
  question and exercise the variant selection.

## Files

- `patches/zluda/0018-m05-image-capture.patch`, `patches/README.md`
- `scripts/m_image_capture_test.py` (synthetic round-trip)
- `scripts/m_image_validate.py` (B1/B2/B3 checks against independent oracles)
- `scripts/m_capture_audit.py` (image inventory, roles, in-frame edges)
- `tests/test_m_capture_image_audit.py` (metadata, association, integrity)
- Captures stay in the temporary workspace: they contain NVIDIA-derived weights and frames.
