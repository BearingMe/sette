# Project-owned patches

Changes this project needs in a pinned reference, kept as patches so the reference
submodules stay read-only and every delta is reviewable.

## `zluda/0018-m05-image-capture.patch`

Adds texture/surface capture to ZLUDA's diagnostic replay hook (`M05`, image
resources). It applies **after** d4r's audited ZLUDA delta:

1. ZLUDA at `ee2f25a` (`v7-preview.11`);
2. `references/sources/d4r/patches/zluda/0002`…`0014` plus the K output-capture
   delta (`tools/windows/zluda-k-output-capture.patch`), as described in
   [EXP-008](../docs/experiments/EXP-008-m05-full-capture.md);
3. this patch.

```powershell
git -C <zluda> apply --whitespace=nowarn <repo>/patches/zluda/0018-m05-image-capture.patch
```

What it changes:

| File | Change |
|---|---|
| `zluda/src/impl/image_capture.rs` | new: resolve a texture/surface handle to its array or pitched storage and copy it to host |
| `zluda/src/impl/array.rs` | register every live array's geometry at creation; unregister on destroy |
| `zluda/src/impl/tex.rs`, `surf.rs` | expose the resource descriptor an object was created from |
| `zluda/src/impl/mod.rs` | declare the new module |
| `zluda/src/impl/d4r_replay.rs` | dump the image behind each bound handle, pre and post launch, and name bindings that cannot be read |

Behaviour is unchanged unless `D4R_CUDA_REPLAY_DUMP_DIR` is set. `D4R_CUDA_IMAGE_DUMP=0`
keeps the pre-extension, buffer-only capture format byte for byte.

### Verification performed

- The patch applies cleanly (`git apply` exit 0) to a reconstruction of the
  pre-patch tree, and the result is byte-identical to the sources that were
  built and run, apart from line endings.
- The pre-image used for that check was reconstructed from the audited files
  rather than from a fresh clone of ZLUDA plus d4r's series, because the
  workspace holds the d4r delta as uncommitted modifications. The patch content
  itself is a real diff of those files; see the limitation note in
  [EXP-011](../docs/experiments/EXP-011-m05-image-capture.md).
- Built library: `nvcuda.dll` sha256
  `F2880A56B152C3B906A72B5C2D0DD21E4B2D00B43715A2F992E03605D5DDD98C`
  (ZLUDA `ee2f25a`, Rust 1.98.1, LLVM `22.0.0git` at `ff4dc1f` + the gfx1154
  backport, `x86_64-pc-windows-gnu`, release).
