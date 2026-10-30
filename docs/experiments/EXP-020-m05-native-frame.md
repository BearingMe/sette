# EXP-020 — M05 B6: complete native frame, execution gate and static-frame contract

**Milestone:** M05 (B6 / C21) · **Status:** EXECUTORY — execution gate PASS, fidelity
INCONCLUSIVE (no independent static validation set). Not an FP16 oracle result.

B5 remains IN PROGRESS / FAIL (`enc1.merge`, thresholds untouched, see [EXP-019](EXP-019-m05-native-network.md) §B5 addenda).

## 1. Question

Two separate questions, kept separate here because the first version of this report
mixed them:

1. **Execution.** Does the complete native pipeline execute correctly — every declared
   stage, at the declared size and format, finite, unsaturated, deterministic, with no
   captured neural tensor substituted?
2. **Fidelity.** How close is the resulting image to an independently identified
   reference, and is the evidence strong enough to call it validated?

## 2. Method

`scripts/m_b6_frame.py` connects the real chain and substitutes no neural tensor:

1. `enc0` runs natively from its fixture and writes the token field.
2. The ten-layer chain (`enc1`, `enc2`, `tube#1..#6`, `dec2`, `dec1`) is built as one
   GPU graph whose first resource — `enc1_in` — is rewired to `enc0`'s output.
   Intermediates stay GPU-resident (94 dispatches, 146 resources); `out-dec1.bin` is the
   only readback.
3. `dec0` runs from that `dec1` field, `post` consumes `dec0`'s gate/mix/covariance
   surfaces, `downsample` consumes `post`'s packed buffer.

The only captured inputs are the ones the network does not produce: **colour and
display motion** for `post` (the LDR/reset variant `post` binds nothing else) and the
captured weights. The reset frame binds no history texture, so no history is
substituted — see [EXP-016](EXP-016-m05-post-reconstruction.md) §6.

Three paths are measured against the canonical captured frame: `native` (this
reconstruction), `image` (image stages fed the captured `dec1` field, isolating the
image stages) and `translated` (image stages fed the translated-PTX ten-layer chain).

Every run executes the whole pipeline twice and compares **all ten declared stage
outputs** by hash, not just the three that were checked before.

Declared before the run and encoded in the script:

| Contract | Contents |
|---|---|
| `EXECUTION` | all five stages present; every declared output present at the byte size its own `stage.txt` declares; 1920x1080x4 f16 frame; zero non-finite `dec1` codes; zero non-finite frame values; zero samples at the f16 limit; a fresh two-run byte-identical repeat across every boundary; at least 7 boundary hashes recorded |
| `REGRESSION` | native **and** image control >= 45.0 dB PSNR (peak 1), >= 0.99 windowed SSIM, mean abs <= 0.005, max abs <= 0.5, finite, unsaturated; chain contribution (image-vs-native) >= 42.0 dB |
| `fidelity` | **INCONCLUSIVE** unless measured on an independent validation set with predeclared floors. The regression bounds are not a fidelity claim and are labelled as such |

The regression bounds are anchored on the image-stage control, which contains every
stage except the neural chain and still measures 46.73 dB: a 45 dB floor cannot be met
by a reconstruction defect in the chain, so it detects collapse without pretending to
measure fidelity.

## 3. Provenance

[M] Two complete runs are byte-identical at **every** declared boundary — `enc0`
tokens, chained `dec1`, all four `dec0` surfaces, all three `post` surfaces and the
final frame (`deterministic: true`, `fresh: true`, 10 boundary hashes compared).

| Boundary | SHA-256 (prefix) |
|---|---|
| `enc0/out-tokens` | `aaf0801b8d4ece12` |
| `chain/out-dec1` | `c8d0c93cb202c25b` |
| `dec0/out-gate` | `9a6d1f246444987d` |
| `dec0/out-mix` | `c0c94a2141954ea2` |
| `dec0/out-feat` | `e7f74daaa5836771` |
| `dec0/out-cov` | `23ccb3fe21084c65` |
| `post/out-packed` | `5bd4273b7d2b6220` |
| `post/out-history` | `fc55d289859ad059` |
| `post/out-half` | `a72f161cdefd9fb0` |
| `downsample/out-out` | `dc97c4cbe9934611` |

Frame hashes: native `dc97c4cb…`, image `0714d03c…`, translated `f43fac93…`,
reference `1a7934b8…`. `m_image_stage.exe` `bf832f22…`, `m_network_stage.exe`
`85c062a9…`; all shader hashes and every fixture/layer input hash are recorded in
`b6.json` (`inputs`), and a `--reuse` run is refused unless they all still match.

Command (fresh directory; `--reuse` re-measures without re-executing and cannot pass):

```powershell
python scripts/m_b6_frame.py --out NEW_DIR `
  --translated-dec1-capture "$env:LOCALAPPDATA\Temp\opencode\chain-translated-20261029\capture\replay-000001-rrlite_dec1_4x4"
```

[M] The native frame is byte-identical (`dc97c4cb…`) to the frame the earlier,
unhardened runner produced, so the hardening changed the reporting, not the arithmetic.
The gate was then re-run once more with the final build (16 shaders, including the new
INT8 kernel of [EXP-021](EXP-021-m09-int8-gemm.md)) and reproduced every hash above,
including the frame, with `execution: PASS` and ten identical boundaries.

## 4. Results (1920x1080, RGB, reference-relative)

| Path | PSNR (peak 1) | PSNR (peak ref 0.7515) | SSIM (windowed) | SSIM (legacy block8) | mean abs | p50 abs | p99 abs | max abs | mean rel | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | **47.05 dB** | 44.57 dB | **0.999259** | 0.997940 | 0.001086 | 0.000244 | 0.02295 | 0.1782 | 0.002398 | +8.8e-5 |
| image | 46.73 dB | 44.24 dB | 0.999208 | 0.997912 | 0.001126 | 0.000244 | 0.02393 | 0.1416 | 0.002483 | +1.8e-5 |
| translated | 47.08 dB | 44.60 dB | 0.999268 | 0.998051 | 0.001075 | 0.000244 | 0.02246 | 0.1543 | 0.002374 | +1.4e-4 |

Two SSIM values are reported because they are different estimators. The primary one is
the standard windowed definition — 11x11 Gaussian window, sigma 1.5, K1 = 0.01,
K2 = 0.03, L = 1.0, Gaussian-weighted moments (no N/(N-1) correction), BT.709 luma of
the RGB channels. The second is the earlier non-overlapping 8x8 block-luminance
metric, kept only as a labelled diagnostic; it is not standard SSIM and must not be
quoted as such.

[M] **Non-finite and saturation:** zero NaN/Inf values in the native `dec1` field
(zero E4M3 NaN codes), in the native frame and in the reference; zero samples at
+-65504 in any frame. Any of these would now fail execution, not merely be counted.

[M] **Alpha:** all 2,073,600 alpha samples are bit-identical and the maximum alpha
difference is 0.0 on every path. Alpha is reported separately and is never mixed into
the RGB statistics.

[M] **Spatial distribution** (16x16 blocks of mean absolute luma error): median
0.00077, worst 0.00290, best 0.00018. The worst blocks are near image borders —
native `(9,3)`, `(58,12)`, `(34,116)`, `(26,116)`, `(42,116)`; image `(5,3)`, `(64,7)`,
`(42,108)`, `(26,108)`, `(18,108)` — but the border/interior split (32-pixel margin)
shows the **interior mean is slightly higher** than the border mean on every path
(native 0.000872 interior vs 0.000696 border, image 0.000911 vs 0.000697, translated
0.000865 vs 0.000752). The error is therefore spread over the frame with a few
locally worse blocks at the edges, not concentrated at the border.

[M] **Chain contribution:** `native` versus `image` (captured `dec1` replaced by the
reconstructed chain) is 46.59 dB, windowed SSIM 0.999179, mean abs 0.001134, max
0.1875; `native` versus `translated` is 46.86 dB. The neural chain's own contribution
and the image-stage error are the same order of magnitude and neither dominates.

## 5. Exports

Per path: `frame-<path>-reference.png`, `-reconstructed.png`, `-difference.png`, all
RGBA (alpha exported, not dropped), plus `frame-reference-*.png`. The 2.2-gamma display
mapping is for viewing only; every metric uses raw f16 values. Difference images are
scaled by the 99.9th percentile of the RGB difference (recorded per path).

## 6. Verdict

**Execution: PASS.** Every declared stage ran, wrote every declared output at its
declared size, stayed finite and unsaturated, and repeated byte for byte across the ten
recorded boundaries. The pipeline is a correct execution of the reconstruction as it
stands.

**Fidelity: INCONCLUSIVE.** One canonical frame is a regression control, not a
validation set. Its numbers (47.05 dB, windowed SSIM 0.999259, mean absolute error
0.001086, 0.24% mean relative error) show a coherent image with no collapsed region,
and the paired controls bound how much of that error belongs to the neural chain, but
they cannot establish general fidelity: the only captured content available is this one
frame (the 24 frame-pair edge checks reuse the same content), and the reference itself
is a d4r capture that uses native layer replacements, not the original PTX.

This does **not** validate the FP16 oracle and does not change B5. The chained `dec1`
is still 21.5728% against the translated chain, while its frame is 47.08 dB: image
fidelity is far less sensitive than code-level chain comparison, which is exactly why
the two must not be conflated.

## 7. Corrections to the first version of this report

The first version (commit `c207d4b`) contained five defects that this revision fixes.
They are listed rather than silently edited, because the measurements themselves did
not change:

1. **`status: pass` was written without evaluating any criterion.** The runner now
   separates `execution` (PASS/FAIL/NOT-EXECUTED), `fidelity` (PASS/FAIL/INCONCLUSIVE)
   and an overall `status` (PASS/FAIL/EXPLORATORY), and cannot write PASS for a run
   that did not re-execute.
2. **The repeat check compared three outputs** (`enc0`, `dec1`, frame). It now compares
   all ten declared stage outputs.
3. **Non-finite samples were replaced with zero** before computing error statistics,
   which would hide exactly the samples worth failing on. They are now excluded from
   the statistics, counted, and fail the execution gate.
4. **`--reuse` accepted any existing output.** It now refuses unless the recorded
   input manifest (probe, network probe, every shader, every fixture and layer file)
   and every recorded boundary hash still match.
5. **"mean relative error 0.00240, below 0.1%" was wrong**: 0.00240 is 0.24%. The
   earlier SSIM was also a non-overlapping 8x8 block metric (0.99794), not standard
   SSIM; the windowed value is 0.999259. Both are now labelled explicitly.

Two further statements in the first version were not supported: the border-bias
reading (contradicted by the border/interior split above) and "captured
history/motion inputs" (the reset frame's `post` binds no history texture, and its
display-motion texture is all zeros; only colour and zero motion are captured).

## 8. Limitations

- One static frame, one content, one adapter (gfx1101), one DLL build (310.7.0).
  No evidence about other content or about temporal behaviour.
- The reference is the canonical d4r capture: native image stages plus translated
  `enc0`/`dec0`/`post`/`downsample` and d4r's own native layer replacements. The
  `translated` path here is a fully translated neural chain with the same native image
  stages, not a fully translated frame.
- The frame is the LDR/TEXL folded variant with `reset = 1`. `post`'s Catmull-Rom
  history path and the recurrent-feature path are not exercised (see
  [EXP-022](EXP-022-m05-b7-readiness.md)).
- Local artifacts (frames, PNGs, stage outputs) stay under `%TEMP%`; nothing
  proprietary is tracked.
