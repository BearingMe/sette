# Preset M Numerical Contract

## Summary

Preset M's network is an FP8 (`e4m3`) weight and activation graph with **binary16 intermediate arithmetic**: products are summed exactly within each 32-element MMA step and rounded to f16 once per step. Normalization, attention, GELU, residuals, and the token stages have their own fixed f16 rounding orders, and the reconstruction stage is f16-accumulated with f32 accumulation for its weight sum and blend. This finding records the contract as reconstructed by d4r `05e19d4`; it is a source-level contract, not a locally executed one. Architecture and tensor shapes are in [the architecture finding](2026-10-28-preset-m-architecture.md).

d4r states that reproducing these roundings makes all ten layers and both residual streams byte-identical to captured NVIDIA launches [S]. That claim was not rerun here [U].

## Weight and value representations

- [S] Weights are FP8 `e4m3` (OCP, `satfinite`, round-to-nearest-even). d4r reads the DLL bytes as e4m3 and every e4m3 value is exactly representable in f16, so widening is lossless (`kernels/tools/model_enc3.py:21-51`, `engine/m_model.py:1-8`).
- [S] Blocked layout: a logical `W[K][N]` is stored as `[K/16][N/16][n16][k16]` 256-byte tiles; f16 vectors (RMS gamma, biases) and the enc0 dither table are separate 16-byte-aligned data (`engine/test_m.py:pack8`, `engine/compile_m.py:98-116`).
- [S] Activations that feed a matrix product as the left operand are re-encoded to e4m3 with RNE and saturation at `+-448`; NaNs stay NaN rather than becoming a saturation code (`kernels/m/swin_common.h:136-290`).
- [S] There is no per-tensor or per-channel activation scale in the graph: the FP8 conversion is absolute. This differs from the INT8 path d4r uses on RDNA2, which introduces per-column scales (see below).
- [S] Vectors and biases are f16; attention probabilities and V are f16; the recurrent feature surface is SNORM8.

## Matrix multiplication

- [S] NVIDIA semantics as reproduced by d4r: for each `k32` step, the 32 products are summed exactly and added to the accumulator, then rounded to f16 **once** (`kernels/m/swin_common.h:592-605`, `kernels/tools/model_enc3.py:90-98`). A `K`-deep GEMM therefore performs `K/32` f16 roundings.
- [S] GEMM depths: encoder/decoder blocks `K = C` (2, 3, or 4 steps); MLP first layer `K = C`, second layer `K = 32` (one step); patch merge `K = 4C`; patch expand `K = CIN`; attention `Q Q^T` `K = 32` and `P V` `K = 16` (one rounding each).
- [S] On gfx11 d4r does not use FP8 tensor cores (none exist): it widens e4m3 values to f16 and uses f16 WMMA, emulating the `k32` rounding. Native FP8 WMMA is gated to gfx12 (`kernels/m/swin_common.h:1-8, 27-28, 46-49`). The emulation matches NVIDIA's codes on captured launches, so on RDNA 3 the FP8 graph is numerically an f16 graph with exact FP8 operand values.
- [S] The alternative `dot2` path (RDNA2) widens weights to f16 and uses `v_dot2`/`v_dot4`; its int8 matrices carry per-column `max|W|/127` scales (RNE) and int32 accumulation (`engine/runtime_m.inc:69-78`).

## Normalization

- [S] RMS norm is computed in f16: sum of squares in a kernel-specific tree order, `+2^-13` epsilon, f16 reciprocal square root, multiply by the f16 gamma vector, then e4m3 requantization (`kernels/tools/swin_model.py:30-52`, `m_model.py:86-89`).
- [S] The summation order matters: d4r had to match the native kernels' order to reach byte-identical layers (`engine/README.md:888-893`). The two norms use different trees (`sumsq_tokens` versus `tree_norm2`).
- [H] A correctly rounded rsqrt is not equivalent to NVIDIA's; d4r's ZLUDA-reproduction path uses the hardware approximations `v_rsq_f32`/`v_rcp_f32` (`kernels/m/swin_common.h:297-300`).

## Attention and softmax

Inside each 32-channel head, over the 16 tokens of a window [S] (`kernels/tools/swin_model.py:60-66`, `kernels/tools/model_enc3.py:54-80`):

1. `S = f16(Q_q @ Q_q^T + bias)`, one f16 rounding; `Q_q` is the FP8-rounded Q, and the bias is a learned 16x16 f16 matrix.
2. `t = clamp(f16(S * C_SCALE), +-0.55615234375)` with `C_SCALE = f16(0.0972222164273262 / 5.656854152679443)`.
3. Cubic approximation `poly = f16(t * f16(t * (-t) + 0.92730712890625) + 1.375)`.
4. Exponential by bit manipulation, not a library `exp`: the two f16 columns of a pair are packed into one 32-bit word, shifted left by 5, and added to `0x7FF88000`, then the halves are unpacked.
5. Row sum by an f16 butterfly in the order `c ^ 8`, `c ^ 2`, `c ^ 4`, `c ^ 1`; reciprocal in f16; normalized weights in f16.
6. `O = f16(P @ V)`; V stays f16. `O` is then e4m3-requantized before the output projection.

The bit-level exponential and the exact butterfly order are part of the contract; a mathematically equivalent softmax will not reproduce the codes.

## Activation and residuals

- [S] GELU is a polynomial in f16: `cz = clamp(z, -2, 2)`, `inner = 0.41216981 - 0.08108133*|cz|`, `y = z * (0.5 + cz*inner)`; the result is e4m3-requantized before the second MLP matrix (`kernels/tools/model_enc3.py:83-87`).
- [S] First residual: `x1 = f16(O @ Wo + x0 + bo)`, where `x0` is the f16 block input and `bo` the f16 output-projection bias.
- [S] Second residual: `x2 = f16(x1 + b2)`, then each of the `C/8` MLP chunks adds `q8(gelu(...)) @ w2` in f16. In the exact build the residual stream is rounded at every chunk.
- [S] Encoders clamp the patch-merge result to `+-2*pi` as f16 before e4m3 encoding; decoders add the f16 encoder skip after FP8-requantizing their expanded input (`kernels/m/swin_common.h:318-319`, `swin_block.h:171-224`).
- [S] Block outputs are e4m3; patch-merge outputs are e4m3; the network therefore hands FP8 codes to the expansion stage.

## Input stage quantization

- [S] enc0 builds 32 feature slots per pixel from range-compressed colour, candidate history colours, motion change, and the recurrent feature. Values are clamped to 1024 and encoded by **dithered truncation**: a 16-bit dither value is added to the value's f16 mantissa inside its exponent bucket, the bucket base is subtracted back, and the low 7 bits are cleared to produce an e4m3 code (`M_NOTES.md:113-115`).
- [S] The embedding is a `K = 128 -> N = 64` GEMM with f16 bias; its result is clamped to `+-2*pi` and stored as FP8 (`M_NOTES.md:116-119`).
- [H] d4r marks the dither index formula as "to be confirmed"; the exact index derivation is therefore not established.

## Expansion and reconstruction numerics

- [S] dec0 computes the four sub-position projections with f16 accumulation, requantizes the selected columns to e4m3, and keeps the recurrent blend, gates, three-way softmax, and covariance in f16 (`M_NOTES.md:33-47`).
- [S] The recurrent feature is `out = (1-g)*hist + g*tanh(v)`, with `g = .5*tanh(.5*v) + .5`; history is a linear sample at the reprojected position, zero on reset or off-screen. This is the network's only explicit long-term state besides the reconstruction history.
- [S] post accumulates the 3x3 Gaussian splat in f16 FMA chains, but accumulates the weight sum `W` in f32 and divides the f16 accumulators by the unrounded sum. The blend, the range expand, and the gate mix are f32 (`M_NOTES.md:59-83`).
- [S] Range compression is `log10(max(100*min(e*c, 32752) + 1, 1e-4)) * 0.36` with exposure `e` (0 treated as 1); expansion is `(10^(clamp(v, 1e-4, 3)/0.36) - 1)/100`. The post gain is `2.75301`, the packed-output scale `0.4081632` (reciprocal 2.45) (`M_NOTES.md:53, 80-86, 132`).

## Output conversion and clipping

- [S] The reconstruction writes: an RGBA16F history at reconstruction resolution; a packed 11-11-10 buffer with `trunc(v*2047 + 0.5)` per component and a 10-bit blue channel; and a half-resolution mean stored in RGBA32F or native RGB10.
- [S] A non-finite blended output is replaced by the current colour before storage; reconstruction clamps the expansion input to `[1e-4, 3]`.
- [S] The downsample applies the fixed 3-to-2 filter and an output-format conversion (`RGBA16F` or `RGB10A2`); the LDR package selects different shaders but reuses the same weights and constants (`engine/compile_m.py:77-78`).

## Exposure

- [S] Mean luma `min(max(.25R + .5G + .25B, 0), 10000)` over the exposure grid; `f64` division `0.23547077 / max(mean, 0.005)`; stored f16 (`M_NOTES.md:122-124`). The exposure scalar consumed by enc0/post falls back to 1 when zero (`M_NOTES.md:53, 87`).

## Alternative arithmetic modes (d4r)

These are d4r build options, not Preset M behavior; they matter because a reconstruction may choose one as its reference or as an optimization target [S] (`engine/compile_m.py:33-48`, `engine/README.md:1091-1175`):

| Mode | Accumulator | Activation | Residual | Relation to NVIDIA codes |
|---|---|---|---|---|
| `M_EXACT=1` | f16, rounded per k32 | f16, kernel order | f16 per chunk | Intended to reproduce NVIDIA (d4r: byte-identical) |
| Default relaxed (`M_RELAXED`, `WPW=2`, `M_RESIDUAL_F32`) | f32 through a GEMM chain | f32, converted to FP8 at the boundary | f32 across MLP chunks, one f16 rounding | About 1 dB from NVIDIA in d4r's comparison |
| `TEXL=1` | ZLUDA's lowering of NVIDIA PTX | ZLUDA order | ZLUDA order | Matches d4r's CUDA capture path |
| `dot2` (RDNA2) | int32 for int8 matrices, f16 elsewhere | f16, converted | f16 | d4r's approximate RDNA2 path |

d4r reports, against an RTX PRO 4500 capture: the then-default exact build `38.4 / 38.2 / 37.9 dB` and the relaxed build `37.6 / 37.2 / 37.2 dB`, with the network at 1.42 ms versus 1.13 ms (1280x720 to 4K). Those are d4r's measurements on its hardware, not this project's [S]. `M_RESIDUAL_F32` is reported 43-46 dB from the previous relaxed build on smooth frames but 17-18 dB on noise, which suggests that residual rounding matters most on high-frequency content [S]/[H] (`engine/README.md:1168-1175`).

## Consequences for the INT8 reconstruction

- [H] Preset M has no activation scales to preserve: INT8 activations will need scales that the original graph does not have. Per-channel weight scales and per-tensor/per-token activation scales are reconstruction choices, not recovered parameters.
- [H] The most sensitive sites are the attention exponential (clamped cubic plus bit trick), the two RMS norms (epsilon and summation order), the f16-per-k32 accumulator boundary, and the two residual streams; d4r reports that a single FP8 rounding flip changes a value by 6-12% and compounds through ten layers (`engine/README.md:885-887`).
- [H] The recurrent dec0 feature and the reconstruction history create temporal error accumulation; a per-frame comparison alone will not characterize them.
- [S] d4r's own int8 path is a useful precedent: symmetric int8 with per-column `max|W|/127` scales, int32 accumulation, and f32 dequantization, applied only to matrices where its error was acceptable (`runtime_m.inc:69-78`).
- [U] No evidence establishes that an INT8 activation graph can retain Preset M's FP8 behavior; this is M07's question, not an established result.

## Limitations and Unknowns

- [U] The contract is d4r's reconstruction of NVIDIA PTX and captured launches at 310.7.0; the local DLL's weights were not extracted, and no local launch was traced. The local DLL matching d4r's pinned build makes the code path plausible, not verified.
- [U] The original container of the weights is unknown: d4r reads e4m3 bytes from the DLL allocation, but whether the DLL stores e4m3 directly or converts at load was not established.
- [U] The dither index formula, the exact shift alternation, and HDR/LDR differences are d4r source behavior with open questions in d4r's own notes.
- [U] Whether NVIDIA's hardware rounds to f16 per k32 on every architecture is d4r's reproduction result; NVIDIA documentation was not consulted for this claim.
- [U] No numerical probe was run in this repository; every `[S]` claim is source observation, and every d4r-reported `[M]` result remains d4r's.

## References

- d4r `05e19d4`: `engine/M_NOTES.md`, `engine/README.md`, `engine/m_model.py`, `engine/compile_m.py`, `engine/runtime_m.inc`, `engine/post_m.comp`, `kernels/m/swin_common.h`, `kernels/m/swin_block.h`, `kernels/tools/swin_model.py`, `kernels/tools/model_enc3.py`, `engine/test_m.py`.
- Architecture and stage graph: [2026-10-28-preset-m-architecture.md](2026-10-28-preset-m-architecture.md).
- Artifact and reference provenance: [2026-10-28-reference-inventory.md](2026-10-28-reference-inventory.md).
