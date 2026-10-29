# Preset M Architecture Reconstruction

## Summary

Preset M is a ten-layer Swin-style token network between a pixel-tokenization stage and a pixel reconstruction stage. d4r reconstructs it from captured NGX launches and NVIDIA PTX, and every stage has a matching shader and captured-launch test. This finding consolidates the stage graph, tensor contracts, buffer dependencies, and temporal state at d4r `05e19d4`. The numeric contract is in [the numerics finding](2026-10-28-preset-m-numerics.md); artifact provenance is in [the reference inventory](2026-10-28-reference-inventory.md).

The graph is **derived from one reverse-engineering implementation**, not from NVIDIA source. d4r reports byte-identical tokens and layer outputs against captured launches, and a coupled engine byte-identical to its own native HIP network [S]; this repository has not executed that path [U].

## Pipeline

| # | Stage | d4r implementation | Function |
|---|---|---|---|
| 0 | Exposure | `exposure_k.comp`, `cuda_dldn_engine_*` | One luma mean and exposure scalar per frame |
| 1 | Input / tokenization | `enc0_m.comp`, `rrlite_enc0_4x4_*_folded` | 8x8-pixel blocks to 4x4 FP8 tokens (64 channels) plus motion and history features |
| 2 | Network | `layer_m.comp`, `rrlite_{enc1,enc2,enc3_tube,dec2,dec1}_4x4` | Two encoder blocks, six tube blocks, and two decoder blocks with patch merges/expands |
| 3 | Expansion | `dec0_m.comp`, `rrlite_dec0_4x4_folded` | Token to per-pixel mix, gates, recurrent feature, and covariance surfaces |
| 4 | Reconstruction | `post_m.comp`, `rrlite_post_3_*` | Anisotropic splat, temporal blend, history and packed output |
| 5 | Downsample | `down_m.comp`, `rrlite_downsample_kernel_*` | Static 3-to-2 filter to display resolution |

Entry points for both HDR and LDR variants exist in the local 310.7.0 DLL (`rrlite_enc0_4x4_{mvlo,mvhi}_{hdr,ldr}[_folded]`), so the shader pair is selected by NGX's HDR flag; d4r records that both read the same weights and constants [S].

## Execution graph

```
colour, depth, motion, jitter, exposure, reset
  |
  +-> exposure ------------------------------+
  |                                          |
  +-> enc0 -> tokens(64) --------------------+--> enc1 -> skip1(64) + merged1(96)
                                                   |
                             merged1(96) --> enc2 --+--> skip2(96) + merged2(128)
                                                   |
                             merged2(128) --> tube x6 (ping-pong 128)
                                                   |
                             tube out(128) --> dec2 (+skip2 96) -> 96
                                                   |
                             96 --> dec1 (+skip1 64) -> 64
                                                   |
  history, prev motion, prev feature --> dec0 <--+
                                                   |
                              mix, gates, cov, feat, motions
                                                   |
  history, motion, exposure --> post -> next history, packed RGB10, half-res history
                                                   |
                                            down -> output
```

Buffer allocation confirms the ping-pong and skip wiring: `MEngine::Impl` binds layer inputs `{tokens, merged1, merged2, A, B, A, B, A, B, dec2Out}` and outputs `{skip1, skip2, A, B, A, B, A, B, dec2Out, dec1Out}` (`references/sources/d4r/engine/runtime_m.inc:398-431`). Decoder skips are `skip2` (dec2) and `skip1` (dec1).

## Stage contracts

### Exposure

- Input: render-resolution colour. Output: one f16 exposure per frame, plus a two-word state so the next frame keeps the previous value [S] (`runtime_m.inc:390-391, 487-489`).
- `sum` is the mean of `min(max(.25R + .5G + .25B, 0), 10000)`; exposure is `0.23547077 / max(sum, 0.005)` with an f64 division, stored as f16; the 0.2195 parameter is unused [S] (`M_NOTES.md:122-124`).
- Preset M measures exposure every frame regardless of the NGX AutoExposure flag, exposure texture, pre-exposure, and scale [S] (`engine/README.md:921-923`).

### Enc0 (input and tokenization)

- One workgroup covers 8x8 render pixels (32 lanes x two pixels), which is 4x4 tokens of 2x2 pixels; pixels outside the render rect are mirrored (`|v|`, then `2n - v - 2`) and flagged so they write no surfaces [S] (`M_NOTES.md:89-96`).
- Per pixel, 32 FP8 feature slots carry range-compressed colour, two motion components, four history samples along motion candidates, and a motion-change scalar; the slots form two 16-byte planes, and the token's K index is `32*q + 16*plane + byte` with `q = 2*(py&1) + (px&1)` [S] (`M_NOTES.md:100-116`).
- Embedding GEMM: K=128 (four pixels x 32 features) to N=64, f16 bias, clamp to `+-2*pi`, output FP8 [S] (`M_NOTES.md:116-119`).
- Side outputs: three render-resolution 2-channel f16 motion surfaces (`mvA`, `mvB`, `mvC`) and the previous-output half-resolution feature sampled at the candidate position [S] (`M_NOTES.md:10-12, 96-112`).

### Swin network

Ten launches, in order, with per-layer constants in `runtime_m.inc:5-12`:

| Layer | C | Heads | Token scale | Base shift (x,y) | Merge / expand |
|---|---:|---:|---:|---|---|
| enc1 | 64 | 2 | 1 | 2, 2 | merge to 96 |
| enc2 | 96 | 4 | 2 | 2, 2 | merge to 128 |
| tube 1 | 128 | 4 | 4 | 2, 0 | none |
| tube 2 | 128 | 4 | 4 | 0, 0 | none |
| tube 3 | 128 | 4 | 4 | 2, 0 | none |
| tube 4 | 128 | 4 | 4 | 0, 0 | none |
| tube 5 | 128 | 4 | 4 | 2, 0 | none |
| tube 6 | 128 | 4 | 4 | 0, 0 | none |
| dec2 | 96 | 4 | 2 | 0, 0 | expand from 128 |
| dec1 | 64 | 2 | 1 | 0, 0 | expand from 96 |

- `MScale` divides the token grid; layer windows are 4x4 tokens (16 tokens), four windows per 8x8-token block [S] (`runtime_m.inc:8-12`, `kernels/m/swin_block.h:1`).
- The runtime alternates the horizontal shift by 2 on odd frames (`sx = MShift[l] ^ ((frame & 1) * 2)`), while `M_NOTES.md:15` reports shifts as fixed in the captures. The alternation is d4r runtime behavior, not a verified NGX rule [S]/[U] (`runtime_m.inc:650-653, 724`).
- Block internals [S] (`kernels/tools/swin_model.py:101-125`, `kernels/m/swin_block.h`):
  1. RMS norm over natural-order channels, f16 sum of squares, `+2^-13`, f16 reciprocal square root, scale by gamma, FP8 requantization (h1).
  2. Per head: Q and V projections `C -> 32`; Q is reused as both query and key (`S = Q Q^T`), with a learned 16x16 relative-position bias. There is no separate key projection, and each 32-channel head attends only within its 16-token window.
  3. Softmax over the 16 window tokens, approximate exponential, f16 weights; `P V` keeps V in f16.
  4. Attention outputs for all heads are concatenated (`32*heads -> C`) by an output projection, added to the block input `x0` and a bias (first residual).
  5. Second RMS norm (h2), then an MLP of `C/8` independent 32-channel hidden chunks: `C -> 32`, bias, GELU, FP8, `32 -> C`, accumulated into a second residual stream.
  6. FP8 block output; encoders then run a patch merge `4C -> NPM` over each 2x2 token group at half the token resolution.
- Decoders first expand the low-resolution input `CIN -> 4C` (four sub-positions), select the sub-position per token with FP8 rounding, add the encoder skip in f16, and then run the same block [S] (`swin_model.py:149-168`).
- Channel order is permuted in memory ("pair" order) for tensor-core fragments: `gp(c) = 32*(c//32) + 2*((c%32)//4) + (c%2) + 16*((c%4)//2)`; the native kernels and d4r's model convert at block boundaries [S] (`swin_model.py:8-16`, `m_model.py:25-27`).

### Dec0 (expansion)

- Grid is one 8x8-pixel block per 4x4 low-resolution tokens; each token's 64 channels are projected to four sub-positions and, in NGX's kernel, 32 values per sub-position. d4r's package retains 20 columns per sub-position, the range its stage consumes [S] (`M_NOTES.md:16-19`, `compile_m.py:103-106`).
- Render pixel `(x,y)` takes columns `32q + j` of token `(x/2, y/2)`, `q = 2*(y&1) + (x&1)`, each rounded to FP8; only `j = 0..16` are used [S] (`M_NOTES.md:33-36`).
- Four render-resolution surfaces [S] (`M_NOTES.md:37-47`):
  - `S128` (SNORM8x4): recurrent feature `out_i = (1-g_i)*hist_i + g_i*tanh(v[2+i])`, gates `g_i = .5*tanh(.5*v[6+i]) + .5`, f16 arithmetic, history reprojected by motion and reset on request/off-screen.
  - `S136` (UNORM8x4): `sigmoid(v0)`, `sigmoid(v1)`, `sigmoid(v10)`, constant 0.
  - `S144` (UNORM8x4): three-way softmax `(v11, v12, v13)`, alpha 0.
  - `S152` (RGBA16F): gate pair `a,b` from `sigmoid(clamp(v14/15, +-20))`, `t = v16*0.0786`, `rho = tanh(t)` with a piecewise polynomial, output `(a^2, b^2, a*b*rho, 0)`.
- In the captured path, `S136` supplies the reconstruction gates (`g`), `S144` the three-way mix (`m`), and `S152` the anisotropic covariance; `S128` is the recurrent feature sampled by enc0 and dec0 [S] (`M_NOTES.md:19, 76`, `engine/post_m.comp` bindings 2-4).

### Reconstruction (post)

- Output grid is the reconstruction image, `1.5 x` the **output** size, not `3 x` render. `rrlite_post_3_1` (tile 10x16) runs when `reconstruction/render >= 3`; `rrlite_post_3_2` (tile 15x27, same body, sized for the 1.5x end) runs below that, including Quality [S] (`M_NOTES.md:127-131`, `engine/post_m.comp:7-11`).
- Per high-resolution pixel: a 3x3 render-texel anisotropic Gaussian splat weighted by dec0's covariance; then three motion candidates (A/B/C from enc0) are paired with dec0's mix values, the two largest weights are kept, and each samples history with a five-tap Catmull-Rom cross [S] (`M_NOTES.md:49-79`).
- Blend: `hist' = b1*h1 + b2*h2`; range expand `(10^(clamp(v,1e-4,3)/.36)-1)/100`; gate blend `a0 = clamp(gain*W*g.x, 0, 1)`, `a1/a2` from `g.y/g.z`; non-finite output falls back to the current colour [S] (`M_NOTES.md:80-83`).
- Outputs: next-frame history (RGBA16F), a packed 11-11-10 copy (scale `.4081632`, rounded), and a half-resolution mean of the blended result [S] (`M_NOTES.md:84-86`).
- Display-resolution motion (`_mvhi_`) is an alternative enc0/post pair in the same model package, not a separate network [S] (`compile_m.py:18`).

### Downsample

- Static 3-to-2 filter from reconstruction resolution to output; weights/associations are d4r's `down_m.comp`, byte-identical to the capture in d4r's test [S] (`M_NOTES.md:26`, `engine/README.md:874`).
- Runtime dispatch is 8x64 output pixels per workgroup [S] (`runtime_m.inc:706-707`).

## Tensor geometry

For the documented capture (640x360 render to 1280x720 output) [S]:

| Quantity | Value |
|---|---|
| Render pixel grid (32-aligned) | 640 x 384 |
| Token grid | 320 x 192 |
| Tube token grid | 80 x 48 |
| Reconstruction | 1920 x 1080 |
| Reconstruction / render | 3.0 (post_3_1) |

For the project's provisional 1920x1080 output at Quality (1.5x, 1280x720 render), derived only from the formulas above [H]:

| Quantity | Value |
|---|---|
| Render pixel grid (32-aligned) | 1280 x 736 |
| Token grid | 640 x 368 |
| Tube token grid | 160 x 92 |
| Reconstruction | 2880 x 1620 |
| Reconstruction / render | 2.25 (post_3_2, 15x15 tile) |

These derived dimensions have not been executed or compared with an NGX launch.

## Temporal state

Persistent per-feature state [S] (`runtime_m.inc:401-408, 471-483`):

- Two history images at reconstruction resolution (RGBA16F), ping-ponged; the half-resolution history is RGBA32F-backed (or native RGB10) and is sampled by enc0.
- Two previous motion surfaces plus the previous half-resolution feature surface, sampled by enc0/dec0.
- Previous exposure, previous jitter, and reset flags; a reset or extent change sets the stages' reset flag, which makes history fall back to the current colour instead of previous state.
- Reconstruction mixes two history candidates per pixel and feeds the blended result back as the next history; errors therefore propagate temporally, and d4r fixed a frame-ten feedback divergence caused by a single f16 rounding order [S] (`engine/README.md:903-916`).

## Facts and inferences

| Claim | Class | Basis |
|---|---|---|
| Stage order, ten layers, channel counts, heads, scales, shifts | [S] | `runtime_m.inc:5-12`, `M_NOTES.md` |
| Block internals (RMS, Q/Q attention, MLP chunks, merge/expand) | [S] | `swin_model.py`, `swin_block.h`, `m_model.py` |
| Byte-identical layer reproduction on captured launches | [S] | d4r's reported tests, `engine/README.md:888-893` |
| Local DLL carries matching M entry points | [R] | 35 `rrlite_*` entries, [reference inventory](2026-10-28-reference-inventory.md) |
| Local DLL weights equal d4r's extracted weights | [U] | Extraction was not run |
| NGX alternates window shifts per frame | [U] | d4r's runtime does; `M_NOTES.md` reports fixed shifts from captures |
| 1920x1080 Quality geometry | [H] | Arithmetic from documented formulas, not executed |

## Limitations and Unknowns

- [U] No model or weights were extracted from the local 310.7.0 DLL; the graph is d4r's reconstruction, not a verified property of this DLL's weight set.
- [U] d4r's standalone engine is Linux/Vulkan/HIP and was not built or executed here, so none of d4r's reproduction claims were independently confirmed in this repository.
- [U] The exact NGX per-frame shift schedule, HDR/LDR shader differences, and the `rrlite_post_3_2` tile are known from d4r source and DLL entry names but not from a local launch.
- [U] Whether the 20-column subset in d4r's dec0 package is sufficient for every mode is asserted by d4r's code, not independently checked.
- [U] The local DLL also contains preset L entry points; this finding covers only the M (`_folded`) path.
- [H] The 1280x720-to-1920x1080 Quality geometry above assumes the same 32-pixel alignment and 1.5x reconstruction ratio as the captured modes.

## References

- d4r `05e19d4`: `engine/M_NOTES.md`, `engine/README.md` (Preset M sections), `engine/runtime_m.inc`, `engine/m_model.py`, `engine/compile_m.py`, `engine/post_m.comp`, `kernels/m/swin_block.h`, `kernels/m/swin_common.h`, `kernels/tools/swin_model.py`, `kernels/tools/model_enc3.py`.
- NVIDIA DLSS 310.7.0 entry-point inventory: [reference inventory](2026-10-28-reference-inventory.md), `references/binaries/README.md`.
- d4r's owned evidence is its own tests and captured-launch comparisons; it is not reproduced here.
