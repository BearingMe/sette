# EXP-025 — dec0-only INT8 stage and full-frame integration

**Milestone:** M09-D · **Status:** PASS for D1–D6; M09 remains IN PROGRESS.
**Foundation revision:** `378365b` (`41e460c` plus the committed M09-C work).
**Environment:** RX 7700 XT / gfx1101, Windows, D3D12, Windows SDK DXC, RGA 2.15.

## Question and acceptance criteria

Can the validated INT8 expansion feed the original dec0 epilogue without a CPU
round-trip, and produce a deterministic dec0-only hybrid frame? Before the frame:
require CPU/GPU integer and conversion agreement, correct four-surface output,
complete writes, finite values, and an independently checked epilogue handoff.
Then require two full frame executions, boundary determinism, native INT8 ISA,
and unchanged FP16 / enc1-Q controls. Quality is measured, not required to equal
FP16. No B4/B5 threshold is changed.

## D1 — Preserved foundation

[M] Reviewed and committed the seven M09-C source/test/documentation files as
`378365b feat(m09): validate dec0 int8 expansion from physical layouts`.
Before integration: 293 tests passed, native build passed, all three experiment
shader hashes matched the PASS report of
[EXP-024](EXP-024-m09-dec0-int8-expansion.md), and the working tree was clean.
No captured data or generated shader was staged. Nothing was pushed.

## D2/D3 — Integration architecture

[S] `m_image_stage.cpp` retains its version-2 image fixture and default one-dispatch
path. The experimental manifest adds one dec0-specific directive:

```text
shader dec0_q8_epilogue.cso
dec0_int8 int8-wc.bin int8-ws.bin int8-bias.bin
```

`m_stage_fixture.prepare_dec0_mode` prepares **only weights** offline from the
original 8,448-byte blob using the EXP-024 mapping/quantization. It does not open
the activation file. FP16 preparation is a byte-preserving no-op.

Within the existing image executor, one command list executes:

| Dispatch | Input / output | Groups |
|---|---|---|
| Existing `dec0_q8_quant.cso` | Original plane-major E4M3 token SRV → packed INT8 activation + row scale | 2040×1×1 |
| Existing `dec0_q8_matmul.cso` | Activation + offline INT8 weight/scale/bias → f16 expansion; integer/f32 diagnostics | 2040×40×1 |
| `dec0_q8_epilogue.cso` | Expansion SRV t2 → original gate, mix, feature, covariance textures | 2040×1×1 |

The prefix is fixed to this dec0 case, not a second graph language or scheduler.
All resources, descriptor heaps, signatures, PSOs, uploads and diagnostics live
until the submission fence completes. There is no readback, Python quantization,
or reupload between these three dispatches. Diagnostics are copied only after
the epilogue has consumed the expansion.

* Activation codes: 8,355,840 B; row scales: 522,240 B.
* f16 expansion: 20,889,600 B, row-major `[130560][80]`.
* Integer and f32 diagnostics: 41,779,200 B each; 16 B unused descriptor padding.
* Offline weight codes/scales/bias: 5,120 / 320 / 320 B.
* Existing weights t0, tokens t1, 32-byte parameter block and four output
  descriptors/formats remain unchanged. The internal expansion is the only
  added epilogue input (t2).
* Gate/mix/feature remain 960×540 R32_UINT views (2,073,600 B each); covariance
  remains 960×540 RGBA16F (4,147,200 B).

The wrapper maps physical column `32*s+i` to logical `20*s+i` at
`(480*ty+tx)*80+20*s+i`. It supplies only the expansion value to the original
`dec0_cs.hlsl` main via `DEC0_LOAD_COLUMN`. All E4M3 round-trips, sigmoid,
softmax, feature and covariance operations, and surface stores are the original
source. **Every pre-existing shader hash, including FP16 dec0, remains identical
to EXP-024.** The nonlinear epilogue remains floating-point by design.

### Synchronization and safety

[S][M] New buffers start COMMON and explicitly transition to UAV. Quantized
activations/scales transition UAV→NON_PIXEL_SHADER_RESOURCE before the GEMM;
expansion transitions likewise before the epilogue. These transition barriers
provide the write→read synchronization; no redundant same-state UAV dependency
is relied upon. Final readback sources transition to COPY_SOURCE, and the CPU
maps readback only after the fence.

The existing image runner's output readback transition incorrectly named
COPY_DEST on the source; this was corrected to COPY_SOURCE without changing
arithmetic. Initial experimental buffer creation in UAV state produced six
debug warnings about ignored initial states. Explicit COMMON→UAV transitions
removed them. **Final stage/frame executions have zero D3D12 debug messages**, not
just zero errors. The runner fails on debug-layer ERROR/CORRUPTION when
`M_STAGE_DEBUG=1`; that setting requires the debug layer rather than silently
skipping validation.

[D] Resource-state and readback conventions were checked against Microsoft's
[resource barrier documentation](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
and [readback documentation](https://learn.microsoft.com/en-us/windows/win32/direct3d12/readback-data-using-heaps).

## D4 — Stage validation, before frame acceptance

[M] `scripts/m09_dec0_integration.py` executes both modes twice with different
position-dependent sentinels. Every declared surface has the required size and
is byte-identical between seeds. All GPU INT8 activation codes/scales, 10,444,800
integer accumulators, f32 epilogue values, and f16 expansion values remain
**bit-identical** to the independent CPU INT8 reference.

Two validation-only paths isolate integration defects:

1. Feed a CPU INT8 expansion directly to the image epilogue. All four surfaces
   are byte-identical to the integrated GPU prefix→epilogue path.
2. Feed the established CPU k32 FP16 expansion to the same epilogue. All four
   surfaces are byte-identical to the original native FP16 stage.

CPU uploads in these two paths are explicit validation oracles, **not inference**.
Both tests pass on the captured token field and on the native FP16 ten-layer
chain's actual output. A final audit ties every checked diagnostic/surface byte
to the buffers and surfaces used by the full hybrid frame.

### Primary stage comparison: native-chain tokens

This is the input used by the full-frame experiment, identical in both modes.
Integer surface error is in byte-code units (feature uses signed SNORM codes);
covariance error is in decoded floating-point units. Alpha is included in these
four-channel stage statistics and is unchanged.

| Surface | Identical samples | Relative RMS vs FP16 | Absolute RMS | Max abs | Signed bias |
|---|---:|---:|---:|---:|---:|
| Gate | 97.6324% | 0.2564% | 0.539922 | 6 codes | +0.046362 |
| Mix | 69.1153% | 2.0459% | 2.076557 | 30 codes | -0.000099 |
| Feature | 80.7977% | 0.8668% | 0.730555 | 7 codes | +0.024216 |
| Covariance | 85.1467% | 2.5274% | 0.123209 | 2.71875 | -0.008848 |

Expansion error on native-chain tokens is 0.526296% relative RMS, absolute RMS
0.01995598, max 0.279296875. This differs from the captured-token measurement
(0.525882%, max 0.21875) because the input field differs, not because the integer
implementation changed. GPU/CPU implementation error is zero in both cases.

### Separate existing reconstruction error from INT8 effects

| Native-chain surface | FP16 vs captured surface, rel RMS | INT8 vs captured surface, rel RMS |
|---|---:|---:|
| Gate | 1.8632% | 1.8697% |
| Mix | 15.3159% | 15.3888% |
| Feature | 5.9074% | 5.9109% |
| Covariance | 16.8196% | 16.8913% |

The large pre-existing difference here includes the FP16 network's reconstruction
error. It must not be attributed to dec0 INT8. With the **captured input field**
instead, INT8-vs-FP16 stage relative RMS is 0.2539% / 1.9624% / 0.8633% / 2.6107%
for gate/mix/feature/covariance, respectively. Captured-input FP16 agreements
remain 99.9991% / 99.9348% / 99.9787% / 99.9883% numerically. Full per-channel
errors, bias and captured-reference comparisons for both input fields are in
their `stage.json` files. No non-finite or saturated f16 surface samples occur.

## D5 — Full frame

[M] `m_b6_frame.py --dec0-mode int8-expansion` runs:

```text
enc0 → ten FP16 layers (94 dispatches) → dec0 INT8 expansion + original epilogue
     → original post → original downsample → RGBA16F frame
```

FP16 remains the default. Combining `--q-mode int8-q` with dec0 INT8 is rejected.
The image-only and translated diagnostic control paths remain FP16. The B6
harness retains its existing file boundaries **between** major stages; only the
new internal dec0 sequence is claimed GPU-resident end to end.

Fresh complete runs of FP16, enc1-Q-only and dec0-only were executed twice.
Metrics use raw finite RGB f16 pairs, peak 1; SSIM is the existing 11×11 Gaussian
window on BT.709 luma. Alpha is reported separately and is identical throughout.

| Against canonical reference | PSNR dB | Windowed SSIM | RMS | Mean abs | P99 abs | Max abs | Signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 47.051389 | 0.999258964 | 0.00444049 | 0.00108616 | 0.0229492 | 0.178223 | +0.00008814 |
| Enc1-Q-only INT8 | 45.421826 | 0.998961386 | 0.00535684 | 0.00126789 | 0.0275879 | 0.201172 | +0.00013652 |
| **dec0-only INT8** | **47.024874** | **0.999255693** | **0.00445406** | **0.00109134** | **0.0229492** | **0.179688** | **+0.00007136** |

Direct comparisons isolate the quantization-induced image difference:

| dec0 hybrid against | PSNR dB | Windowed SSIM | RMS | Mean abs | P99 abs | Max abs | Signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FP16 frame** | **62.915993** | **0.999931287** | **0.000714826** | **0.000278997** | **0.00292969** | **0.0434570** | **-0.00001678** |
| Enc1-Q-only frame | 45.888244 | 0.999052210 | 0.00507677 | 0.00120373 | 0.0263672 | 0.136963 | -0.00006516 |

The canonical PSNR difference is **-0.026516 dB**, not the 62.916 dB direct
comparison. These are distinct measurements. No perceptual-equivalence claim is
made from either.

### Spatial distribution and determinism

[M] Direct hybrid-vs-FP16 weighted absolute RGB error: 32-pixel border mean
0.00023938, interior mean 0.00025795. The 16×16 block mean spans
0.00008957–0.00056553; worst block coordinates (y,x) are
(52,7), (28,7), (44,7), (20,7), (36,7). Raw-RGB mean/p99/max and this weighted
spatial statistic have different definitions; all are retained in `b6.json`.

All **15 boundaries** (10 original plus five dec0 internal diagnostics) match
across fresh runs. In particular the FP16 neural chain's output hash is unchanged.
There are zero non-finite frame/dec1 values and zero f16-saturated frame samples.
The frame's canonical B6 regression bounds still pass, while static fidelity
remains **INCONCLUSIVE** under the unchanged B6 definition.

## ISA and provenance

[M] The actual integrated prefix uses exactly the EXP-024 quantizer/GEMM binaries.
RGA DX12 gfx1101 inspection of that same hashed GEMM still shows 12 static
`v_dot4_i32_i8` instructions and zero FP16 matrix multiply/FMA/MAC/WMMA
instructions. This is static lowering evidence for the executed DXIL, not a
runtime instruction trace. `integration.json` checks the stage, frame manifest,
current shader file, and inspected file hashes agree.

| Artifact | SHA-256 |
|---|---|
| FP16 dec0 (unchanged) | `949b579dd419615be8687163c7e5c545b5577087749fd560af9c22ba56f8fb6f` |
| New epilogue wrapper | `17bcc062527a3dc4783c140c6bde67b878f4c3fedad202016a33633c21b6d3a3` |
| Existing GPU quantizer | `36a9e300c98e9fc78e3b77c49536b54398c79d6427eb25df21bdab1d2ec58d15` |
| Existing INT8 GEMM | `c7fc7fa6a3aa67c540edaea7a76d0d2c3f5d1646c3b581609e8efa062e405a89` |
| Final image executor | `a491595e030bf9649aeeda77f13a536c33d7c010faa8e40520f32b231ded2789` |
| dec0-only hybrid frame | `d7ef968356d95369a276edf3d4af3baefb1d46c7a3f98b292e3483637bb7075a` |
| FP16 frame (unchanged) | `dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b` |
| Enc1-Q-only frame (unchanged) | `0eb0185ea7e21f20617b67476519310ae2dfe8b402188f3207fb197b35299419` |

## D6 — Regressions and status

[M] **303 tests pass**: all 293 existing tests plus 10 mode/fixture tests.
Native build and `git diff --check` pass. Fresh conformance: B4 PASS,
99.9369%, max 7 ULP; B5 still FAIL only `enc1.merge`, 89.8379% / 0.02111
(expected exit 1). FP16 and enc1-Q frame hashes above remain byte-identical.

This closes EXP-024's optional stage/frame integration boundary for this case.
It does **not** close M09's representative-operator gate: attention, MLP and
other matrix operations have not been converted. M09 remains **IN PROGRESS**.

## Reproduction and evidence

All paths containing captured bytes must be outside Git. Example PowerShell
(use fresh output directories; `F`, `LAYERS`, `ENC1Q` denote established fixtures):

```powershell
cmake --build build/native --config Release
python -m pytest tests -q
$env:M_STAGE_DEBUG = '1'
python scripts/m09_dec0_integration.py --fixture "$F/dec0" --out "$T/captured-stage"
python scripts/m_b6_frame.py --fixtures $F --layers $LAYERS --out "$T/fp16"
python scripts/m_b6_frame.py --fixtures $F --layers $LAYERS --out "$T/enc1q" `
    --q-mode int8-q --enc1-int8 $ENC1Q
python scripts/m_b6_frame.py --fixtures $F --layers $LAYERS --out "$T/dec0" `
    --dec0-mode int8-expansion `
    --fp16-frame "$T/fp16/downsample-native/results/out-out.bin" `
    --enc1q-frame "$T/enc1q/downsample-native/results/out-out.bin"
python scripts/m09_dec0_integration.py --fixture "$T/fp16/dec0-native" `
    --out "$T/native-stage" --frame-run "$T/dec0" `
    --fp16-run "$T/fp16" --enc1q-run "$T/enc1q" --rga $RGA
```

Final local evidence under `%LOCALAPPDATA%\Temp\opencode\`:

* `m09d-final-captured-stage-20261030/stage.json`: captured-input stage gate.
* `m09d-final-native-stage-20261030/`: native-input `stage.json`, final
  `integration.json` (all 24 checks PASS), executed-GEMM ISA and both oracles.
* `m09d-final-{fp16,enc1q,dec0}-20261030/b6.json`: fresh full-frame controls and hybrid.
* `m09d-final-conformance-20261030.json`: unchanged B4/B5 results.
* Hybrid PNGs: `m09d-final-dec0-20261030/frame-native-{reference,reconstructed,difference}.png`;
  direct difference PNGs: `hybrid-vs-fp16-{reference,reconstructed,difference}.png`
  and `hybrid-vs-enc1q-{reference,reconstructed,difference}.png` in that directory.

PNGs use the existing gamma display mapping; all measurements use raw f16.
Pre-final directories without `final` in their names are exploratory results,
not the provenance for the final executor hash.

## Limitations and next step

One fixed-resolution reset frame, one content set and one adapter/driver.
No temporal reconstruction, combined-INT8 mode, calibration, optimization, or
performance claim. Weight quantization is offline; diagnostic readbacks are
intentionally retained. Debug-layer/sentinel validation is bounded evidence,
not a proof for arbitrary workloads.

**Execution:** PASS. **Static fidelity:** measured, INCONCLUSIVE as a general
fidelity claim. **Temporal fidelity:** not evaluated. **FP16 oracle:** B5 still
FAIL. **INT8 readiness:** two operators now integrated independently, not a full
INT8 network.

Next independently testable step, subject to authorization: choose one MLP
projection and establish its own operand/boundary contract and isolated CPU/GPU
comparison before any additional integration. Do not combine the two existing
experimental modes or infer milestone completion from this frame.
