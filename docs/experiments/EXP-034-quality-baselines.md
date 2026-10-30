# EXP-034 — Reconstruction quality baselines

**Milestone:** M11-A · **Status:** PASS for measurement reproduction and separation.
The 45 dB gate is still **FAIL**; this experiment does not change it.
**Depends on:** [EXP-016](EXP-016-m05-post-reconstruction.md),
[EXP-020](EXP-020-m05-native-frame.md), [EXP-029](EXP-029-m09-complete-int8.md),
[EXP-030](EXP-030-temporal-state-inventory.md).
**Frames:** `m10-sequence-20261030/{fp16,int8}/frame-00`, identical inputs.

## 1. Question

Are the reported numbers reproducible, and are the three comparisons
(reference↔FP16, reference↔INT8, FP16↔INT8) kept distinct? A quality argument
that mixes them cannot say whether a gap is a reconstruction limit or a
quantization cost.

## 2. Method

Inspected the measurement path, then recomputed the numbers from raw frames with
an independent PSNR implementation (no shared helper):

* `scripts/m_b6_frame.py::frame` reads the raw RGBA16F frame, 1920×1080×4.
* `metrics()` computes RGB-only error over finite pairs: `PSNR = 10·log10(1/MSE)`
  with a **fixed peak of 1.0**; alpha is excluded from every RGB statistic and
  counted separately. A second PSNR (`psnr_peak_reference_db`) uses the
  reference's own maximum and is reported for context, never for the gate.
* `ssim_windowed_luma()` uses an 11×11 Gaussian window (σ 1.5, reflect padding,
  Gaussian moments without N−1 correction), K1 0.01, K2 0.03, L = 1.0, on BT.709
  luma. `ssim_block8_luma_legacy` is retained as a diagnostic only.
* The reference is `fixtures/downsample/ref-out.bin`, the canonical d4r capture
  of the same frame, sha256 `1a7934b8…eab91e2`; it is never derived from either
  reconstruction.
* Frames are linear RGB f16. The PNG export applies a 2.2 display gamma for
  viewing only; no metric uses it. No exposure, normalization or border crop is
  applied to the gated PSNR; the 32-pixel border split is diagnostic.

## 3. Results

### Independent recomputation

[M] Direct `numpy` PSNR over the three frame pairs, with no call into the B6
helper, reproduces the accepted values exactly:

| Comparison | PSNR (dB) | MSE | finite | alpha identical |
|---|---:|---:|---|---:|
| reference ↔ FP16 | **47.05138949457013** | 1.9717917743104474e-05 | yes | 2,073,600 / 2,073,600 |
| reference ↔ full INT8 | **44.13928631943929** | 3.8554170892340105e-05 | yes | 2,073,600 / 2,073,600 |
| FP16 ↔ full INT8 | **44.48752927480509** | 3.55833696743757e-05 | yes | 2,073,600 / 2,073,600 |

[M] Frame hashes: FP16 `dc97c4cb…d6e95b`, INT8 `f76ac032…4d6c00`, reference
`1a7934b8…eab91e2`. Alpha is exactly 1.0 across the reference and both outputs,
so the alpha-identical count is not hiding a composite difference.

### The three measurements are not interchangeable

[M] The same frame, three questions:

| Measurement | Value | What it means |
|---|---:|---|
| reference ↔ FP16 oracle | 47.0514 dB | reconstruction + oracle error; **the FP16 path passes 45 dB with 2.05 dB margin** |
| reference ↔ INT8 | 44.1393 dB | reconstruction + INT8 error; **0.86 dB below the gate** |
| FP16 ↔ INT8 | 44.4875 dB | quantization cost alone, with reconstruction error removed |

[M] The INT8 path therefore adds **2.91 dB** of relative error on top of the
FP16 path, while the FP16 path itself is only 2.05 dB above the floor. Neither
measurement can be substituted for the other.

[M] The image-stage control (native `dec0`/`post`/`downsample` fed the captured
`dec1` field, i.e. a *perfect* neural chain) measures **46.7260 dB** for the same
frame (sha `0714d03c…`). That is 1.73 dB above the gate and identical for both
precision modes, because those stages are not INT8. The absolute gate is thus
not unboundedly achievable by quantizing the network alone: the surrounding
reconstruction already consumes most of the margin.

## 4. Interpretation

* [M] The reported values are reproducible to the last printed digit from raw
  frames, and the gating metric is defined before the comparison.
* [S] PSNR is a peak-1.0 linear-RGB measure; SSIM is windowed BT.709 luma. Neither
  is a perceptual or temporal metric, and neither is an oracle for image quality
  in a different content class.
* [M] FP16 passes the unchanged gate; INT8 fails it; the quantization cost is
  2.91 dB. A quality investigation must explain *both* the 2.05 dB FP16 margin
  and the 2.91 dB INT8 penalty.
* [U] No independent static validation set exists (different content captured
  through the same reference path); the canonical frame is a regression control,
  so B6 fidelity remains `INCONCLUSIVE` by contract. Absolute numbers are
  evidence only for this frame.

## 5. Next

EXP-035 attributes the 2.91 dB to stages. The image-stage control bounds what
quantization can recover: improvements inside the 76 matrix operators cannot
raise the ceiling above ~46.73 dB for this frame, and the 45 dB floor leaves
1.73 dB of headroom for everything the network does.

## 6. Limitations

* One frame, one scene, one device, one DLL build.
* SSIM was inspected and property-tested (`tests/test_b6_frame.py`), not
  independently reimplemented; PSNR was.
* No temporal quality claim: EXP-032 covers sequences.
