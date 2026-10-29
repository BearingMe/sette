# M05 Capture Coverage Audit

## Summary

The Windows capture path is complete for the ten `rrlite_*_4x4` Swin layers and
incomplete for the remaining Preset M stages. This audit maps the 35 `rrlite_*`
entry points in the local 310.7.0 DLL against the ZLUDA replay whitelist and
classifies each missing group by whether a complete model needs its **weights**
or only its **tensors**.

## Capture whitelist

[S] The Windows ZLUDA replay hook (`zluda/src/impl/d4r_replay.rs`, tag
`v7-preview.11`) whitelists only:

- the eleven `dltss_pwin_*` K layers;
- the five M layer kernels `rrlite_enc1_4x4`, `rrlite_enc2_4x4`,
  `rrlite_dec1_4x4`, `rrlite_dec2_4x4`, `rrlite_enc3_tube_4x4`;
- one audited K output variant.

[M] A preset-13 run with `D4R_CUDA_REPLAY_DUMP_FILTER=rrlite_` produced exactly
the ten M layer directories (enc1, enc2, six tubes, dec2, dec1), confirming the
whitelist in practice.

## M entry points and coverage

[S] The local DLL contains these 35 `rrlite_*` entries; the parameter block size
is the `.param .align 8 .b8 ...[N]` size in the extracted PTX.

| Group | Entries | Param bytes | Captured? | Model need |
|---|---|---:|---|---|
| Layers | `rrlite_enc1_4x4`, `rrlite_enc2_4x4`, `rrlite_enc3_tube_4x4`, `rrlite_dec2_4x4`, `rrlite_dec1_4x4` | 56, 56, 88, 56, 56 | yes | weights + tensors |
| Input (M) | `rrlite_enc0_4x4_{mvlo,mvhi}_{hdr,ldr}_folded` | 360 | no | **weights** + tensors |
| Expansion (M) | `rrlite_dec0_4x4_folded` | 168 | no | **weights** + tensors |
| Reconstruction 3_1 | `rrlite_post_3_1_{mvlo,mvhi}_{hdr,ldr}_folded` | 248 | no | tensors only |
| Reconstruction 3_2 | `rrlite_post_3_2_{mvlo,mvhi}_{hdr,ldr}_folded` | 248 | no | tensors only |
| Downsample | `rrlite_downsample_kernel_{static,dynamic}_{hdr,ldr}` | 96 | no | tensors only |
| Preset L inputs | `rrlite_enc0_4x4_{mvlo,mvhi}_{hdr,ldr}` (unfolded) | 360 | no | out of M scope |
| Preset L expansion | `rrlite_dec0_4x4` (unfolded) | 168 | no | out of M scope |
| Preset L reconstruction | `rrlite_post_3_{1,2}_{mvlo,mvhi}_{hdr,ldr}` (unfolded) | 248 | no | out of M scope |

The hook dumps the layer entries under the exact names above. The input,
expansion, reconstruction, and downsample groups are absent from the whitelist,
so the capture contains no `enc0`, `dec0`, `post`, or `downsample` directory.

## Weight versus tensor requirements

- **enc0 (folded, one HDR/LDR and mvlo/mvhi variant per frame):** `compile_m.py`
  reads its weight blob at argument offset 0 and extracts the embedding
  `[128][64]`, the 64-value f16 bias, and the 4096-byte dither table. A complete
  model needs this weight allocation captured. Inputs/outputs are useful for
  stage validation but not required to build the model.
- **dec0 (folded):** `compile_m.py` reads its weight blob at offset 0 and
  extracts the `[64][128]` matrix and 128-value f16 bias, using 20 columns per
  sub-position. A complete model needs this weight allocation captured.
- **post_3_1 / post_3_2 (folded):** no learnable weights are extracted by
  `compile_m.py`; the stage uses runtime constants and dec0/enc0 tensors. Its
  capture is needed only to validate reconstruction intermediates and the final
  blend, not to build the model.
- **downsample (static):** no weight buffer is extracted; d4r's `down_m.comp`
  uses compile-time filter taps. Tensor capture is needed only for stage
  validation.
- **Layers:** weights and tensors already captured; ten-layer regression remains
  the acceptance baseline.

## Consequence

A complete Preset M model is reachable by adding **two weight captures**
(`enc0` folded and `dec0` folded) to the existing ten-layer capture. Post and
downsample tensor capture is required for full intermediate validation but not
for `compile_m.py` to produce a model. Exposure needs no weights.

## C02 extension notes

The minimal ZLUDA whitelist additions, using the PTX parameter sizes above:

| Kernel name pattern | Bytes |
|---|---:|
| `rrlite_enc0_4x4_*_folded` | 360 |
| `rrlite_dec0_4x4_folded` | 168 |
| `rrlite_post_3_1_*_folded` | 248 |
| `rrlite_post_3_2_*_folded` | 248 |
| `rrlite_downsample_kernel_*` | 96 |

The existing hook writes the same fixture format (`manifest.txt`, `args.bin`,
`alloc-N.bin`) that `compile_m.py` and `d4r_native_replay_probe` consume.
Rebuilding ZLUDA requires the pinned patched LLVM tree; the local workspace has
ROCm 7.2 clang but no LLVM development build, so that rebuild is the main C02
cost. A probe-side `cuLaunchKernel` hook using `cuFuncGetParamInfo` is a lighter
fallback if the LLVM rebuild is not practical.

## Limitations and Unknowns

- [U] The HDR/LDR and mvlo/mvhi variant the project's 1920x1080 Quality path will
  use is not fixed; all folded variants share the 360/248-byte layouts.
- [U] Whether the post/down kernels write intermediate tensors that are
  reproducible at the same precision as the layer outputs is untested.
- [S] Post and downsample have no model weights in `compile_m.py`; their d4r
  shaders carry the constants and filter taps.

## References

- `engine/compile_m.py` (enc0/dec0 extraction, layer packing).
- `engine/M_NOTES.md` (stage descriptions and variant names).
- `tools/windows/zluda-k-output-capture.patch` (whitelist and sizes).
- [EXP-003](2026-10-29-m05-layer-capture.md) (ten-layer capture and exact replay).
- Local DLL entry extraction: `kernels/tools/extract_dlss_ptx.py` over
  `references/binaries/nvngx_dlss.dll`.
