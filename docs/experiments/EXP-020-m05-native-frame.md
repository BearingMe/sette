# EXP-020 — M05 B6 (EXPLORATORY): first complete native frame

**Milestone:** M05 (B6 / C21) · **Status:** EXPLORATORY — complete frame produced and
measured; B6 has no validation criteria yet and is not claimed as passed or as a
validated FP16 oracle.

## 1. Question

With the ten neural layers reconstructed and connected to the image stages, does the
existing numerical difference produce a coherent, usable frame, or does it reveal a
substantial reconstruction defect? B5 remains IN PROGRESS / FAIL: its `enc1.merge`
criterion is unchanged and no threshold is touched here.

## 2. Method

`scripts/m_b6_frame.py` connects the real chain and substitutes no neural tensor:

1. `enc0` runs natively from its captured fixture and writes the token field.
2. The ten-layer chain (`enc1`, `enc2`, `tube#1..#6`, `dec2`, `dec1`) is built as one
   GPU graph whose first resource — `enc1_in` — is rewired to `enc0`'s output.
   Intermediates stay GPU-resident; `out-dec1.bin` is the only readback.
3. `dec0` runs from that `dec1` field, then `post` consumes `dec0`'s gate/mix/cov
   surfaces, then `downsample` consumes `post`'s packed buffer.

The only captured inputs left are the stage inputs the neural network does not
produce (colour, motion, history). Every stage runs through the B4 fixture executor,
which rejects a missing or undeclared output.

Three paths are measured against the canonical captured frame:

| Path | `dec1` input to the image stages |
|---|---|
| `native` | this reconstruction's ten-layer chain |
| `image` | captured `dec1` (isolates the image stages) |
| `translated` | the translated-PTX ten-layer chain |

The whole pipeline is executed twice; all stage outputs are compared by hash.

## 3. Provenance

[M] Two complete runs are byte-identical at every boundary: `enc0` tokens, chained
`dec1`, and the final frame (`deterministic: true`, repeat frame hash recorded in
`b6.json`). Every stage returned success and wrote its declared outputs. Frame hashes:
native `dc97c4cbe9934611…`, image `0714d03c7caeedcf…`, translated `f43fac93b10fcefb…`,
reference `1a7934b877995f98…` (full digests in `b6.json`).

Command (fresh directory; `--reuse` re-measures existing stage outputs):

```powershell
python scripts/m_b6_frame.py --out NEW_DIR `
  --translated-dec1-capture "$env:LOCALAPPDATA\Temp\opencode\chain-translated-20261029\capture\replay-000001-rrlite_dec1_4x4"
```

## 4. Results (1920x1080, RGB, reference-relative)

| Path | PSNR (peak 1) | PSNR (peak ref) | SSIM (luma) | mean abs | p50 abs | p99 abs | max abs | mean rel | signed mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | **47.05 dB** | 44.57 dB | **0.99794** | 0.001086 | 0.000244 | 0.0229 | 0.178 | 0.00240 | +8.8e-5 |
| image | 46.73 dB | 44.24 dB | 0.99791 | 0.001126 | 0.000244 | 0.0239 | 0.142 | 0.00248 | +1.8e-5 |
| translated | 47.08 dB | 44.60 dB | 0.99805 | 0.001075 | 0.000244 | 0.0225 | 0.154 | 0.00237 | +1.4e-4 |

[M] Non-finite and saturation: zero NaN/Inf pairs on every path, zero non-finite
values in any frame, and zero samples at `+-65504`. No path produced a saturated or
non-finite image.

[M] Spatial distribution (16x16 blocks of mean absolute luma error): the median block
is 0.00077 and the worst block is 0.0029 (native), so the error is spread rather than
concentrated. The worst blocks cluster near image borders — native `(9,3)`, `(58,12)`,
`(34,116)`, `(26,116)`, `(42,116)`; image `(5,3)`, `(64,7)`, `(42,108)`, `(26,108)`,
`(18,108)` — consistent with the boundary/padding effects seen at layer scope.

[M] The neural chain's own contribution to the frame (`native` versus `image`, i.e.
captured `dec1` replaced by the reconstructed chain): **46.59 dB**, SSIM 0.99770, mean
absolute 0.001134, maximum 0.1875. `native` versus `translated`: 46.86 dB, SSIM
0.99783. So the chained neural difference and the image-stage difference are the same
order of magnitude, and neither dominates the frame error.

## 5. Exports

Viewable PNGs in the local experiment directory (`--out`), for each path:
`frame-<path>-reference.png`, `frame-<path>-reconstructed.png`,
`frame-<path>-difference.png` (99.9th-percentile scaled), plus `frame-reference-*.png`.
The 2.2-gamma display mapping is only for viewing; all metrics use the raw f16 values.

## 6. Verdict

The existing differences produce a coherent, usable image: ~47 dB PSNR, 0.998 SSIM,
sub-0.1% mean relative error, no NaN/Inf, no saturation, deterministic across two
complete runs. No sub-1% region collapses; the residual is spread over the frame with
a mild border bias. Nothing here indicates a substantial reconstruction defect.

This does **not** validate the FP16 oracle and does not change B5: the chained `dec1`
is still 21.5728% against the translated chain, and that chain's frame is 47.08 dB —
i.e. the frame is far less sensitive than the code-level chain comparison. B6 remains
EXPLORATORY with no acceptance criteria; establishing them (PSNR/SSIM floors, a
temporal criterion, a provenance requirement for the translated reference) is separate
work.

## 7. Limitations

- The frame is static (frame 1 of one captured pair); history and motion inputs are
  captured, not produced. It is not evidence about temporal stability.
- `post` and `downsample` also consume captured colour/motion; the comparison is a
  reconstruction-stage comparison, not a full DLSS invocation.
- The canonical reference itself was produced with translated `enc0`, `dec0`, `post`
  and `downsample`, and with d4r's native replacements for the ten layers; the
  `translated` path here is translated neural layers with the same native image
  stages, not a fully translated frame.
- PSNR values above are linear-light f16 with peak 1.0 (and the reference peak as a
  second reading); no display encoding is applied in the metrics.
- Local artifacts (frames, PNGs, stage outputs) stay under `%TEMP%`; no proprietary
  data is added to Git.
