# EXP-017: dec0's Numerical Fidelity, Its Propagation, and the FP16 Acceptance Criteria (M05 B4.6)

**Milestone:** M05 (checkpoint B4.6)
**Status:** **PASS** — the residual is classified, its propagation is measured, and
the tolerances for the FP16 oracle are set and executable
(`scripts/m_conformance.py`).  No attempt was made to force `dec0` to byte
identity; §6 records why the evidence says that would be the wrong investment.
**Depends on:** [EXP-012](EXP-012-m05-native-downsample.md),
[EXP-014](EXP-014-m05-dec0-mapping.md) (`dec0`'s mapping),
[EXP-016](EXP-016-m05-post-reconstruction.md) (the downstream stage),
[EXP-009](EXP-009-m07-complete-inventory.md).
**Capture:** `replay-000001-*` of `capture-img`, frame 1, `reset = 1`, zero motion.

## 0. The question

`dec0` is the only reconstructed stage that is not byte-identical: its four
surfaces agree at 99.91%–99.998% and its residuals were never explained.  Before
integrating the ten-layer network (B5) the project needs to know whether that
residual is **(a)** noise at the storage precision, **(b)** a structural error
that B5 will inherit and compound, or **(c)** something in between.  This report
answers that with measurements, then states what B5 must hold itself to.

All percentages below are **per sample over the channels the surface uses**, not
per byte: `gate`/`mix` are 960x540x3 = 1 555 200 samples, `feat` is 960x540x4 =
2 073 600, `cov` is 960x540x3 = 1 555 200.  The fixture's own `check` reports
per-byte figures over 4 channels, so its percentages are slightly different for
the same bytes.

## 1. What the residual is

| surface | encoding | identical | differing | max delta | differing pixels |
|---|---|---|---|---|---|
| gate | UNORM8 x3 | 99.9988% | 19 | 3 codes | 19 |
| mix | UNORM8 x3 | 99.9130% | 1353 | 8 codes | 592 |
| feat | SNORM8 x4 | 99.9787% | 442 | 3 codes | 442 |
| cov | f16 x3 | 99.9814% | 290 | 25% relative | 156 |

Across all four surfaces, **1209 of 518 400 pixels** differ (0.233%).  Two
structural facts shaped the whole investigation:

**[M] The four surfaces' differing pixels are completely disjoint.**  `gate` and
`mix` share 0 pixels, `mix` and `feat` 0, `cov` and `mix` 0 - every pair.  A
difference in something upstream of the epilogue would make them coincide.  They
do not, because the epilogue consumes *disjoint* expansion columns: `gate` uses
v0/v1/v10, `feat` v2-v9, `mix` v11-v13, `cov` v14/v16.  A difference in column k
can only reach the surface fed by k.

**[M] Every difference is a difference of one E4M3 code in the expansion value,
not a difference in the arithmetic that follows it.**  `cov.xx = (sigma *
sigmoid(v))^2` is a monotonic function of a single E4M3 code, so its reachable
values form a 254-entry table.  Matching each differing sample against that table
(built from the format definition and the shader's own sigmoid, not from
`dec0_cs.hlsl`):

- 87 of 87 differing `xx`/`yy` samples land on table entries **one index apart**;
- the match residual is at most 8.1e-4 relative, which is the f16 store's own
  granularity - the values are *on* the table, so the activation is not the
  source;
- `gate`'s 19 samples land on their table 1 (7 samples) or 2 (12) entries away,
  with a residual below half a UNORM8 step; the distance-2 cases are the flat
  part of the sigmoid, where two codes map to indistinguishable 8-bit values;
- `mix`'s deltas are the softmax signature: 576 of 592 differing pixels have one
  channel up and the others down, and the three deltas sum to 0/-1/+1 (probability
  is conserved), which is what a one-code change in one of v11-v13 produces.

So the chain is: **the f16 accumulator differs from NGX's by one ULP for a few
percent of the 8.29M expansion values; the ones that sit on an E4M3 boundary flip
their code, and each surface's activation amplifies the flip by its own slope** -
1-3 codes for the sigmoid/tanh surfaces, up to 8 for the softmax, up to 25%
relative for the square in the covariance.

The "few percent" is an inference, and it is worth stating its arithmetic: an f16
ULP is 4.9e-4 relative, an E4M3 step is 6-12% relative, so a value that differs by
one ULP flips its code about 0.5% of the time.  The measured flip rate is 1209
pixels over 8.29M values (0.015%), which implies **~3% of expansion values differ
by one ULP** and the other 97% are bit-identical.  The alternative reading - that
only 0.015% differ - would require a sub-ULP difference, which two f16
accumulators cannot have.

## 2. Error classes

| class | where it appears | measured contribution |
|---|---|---|
| **FP16 rounding** | the 32-product accumulator, one rounding per k32 step | the origin of everything above: ~3% of expansion values differ by one ULP |
| **Approximate instructions** | `ex2.approx`/`rcp.approx` in the epilogue (sigmoid, softmax, correlation) | **not measurable as an independent term**: each surface's values sit on the activation table to within the store's granularity, so the approximations are absorbed by the following quantiser |
| **Quantisation boundaries** | the E4M3 round-trip of the expansion value; UNORM8/SNORM8/f16 at the output | the amplifier: one E4M3 step becomes 1-8 output codes |
| **Storage conversions** | the f16 store of `cov`, the byte packing of the 8-bit surfaces | none detectable: `cov` compared against the capture is consistent with the truncating store (99.98% identical); no sign bias or systematic offset appears |

The last row is the useful negative result: there is **no storage-conversion bug
hiding in the residual**.  If the store convention were wrong (round vs truncate)
the identical fraction would be near 50%, not 99.98%.

## 3. Propagation: does it matter downstream?

The chain was run twice with the *real* stages - once with the captured `dec0`
surfaces and once with the native ones - and both were compared to the captured
references:

| step | input | result |
|---|---|---|
| `post` | captured gate/mix/cov | 100.0000% of the packed buffer (0 of 4 665 600 words) |
| `post` | **native** gate/mix/cov | 99.9002% (4654 words differ, r/g/b deltas -4..+11, median 0) |
| `downsample` | captured packed buffer | **100.0000%** of the final 1920x1080 frame |
| `downsample` | native packed buffer | 99.9364% of the frame (5278 of 8 294 400 f16 samples) |

The last row is the answer: **`dec0`'s 0.23% of pixels becomes 0.064% of the final
frame's samples, with a median difference of 1 f16 ULP and a maximum of 7** (a
full f16 code at values around 0.25-1.0).  The signed mean over the differing
samples is -0.205 ULP with a 3018/2260 split - a sub-ULP asymmetry, not an
offset: the whole-frame mean is -1.3e-4 ULP.

The control rows matter as much as the measurement: with exact inputs the same
stages reproduce the capture **byte for byte**, so the 0.064% is entirely
`dec0`'s residual and nothing else in the chain.

## 4. Acceptance criteria for the FP16 oracle

`scripts/m_conformance.py` is the executable form.  Two kinds of criterion:

**Identity floors** (measured minus margin, on this frame):

| scope | floor | also required |
|---|---|---|
| a stage that claims exactness (`downsample`, `enc0`, `post`) | 1.0 | - |
| `dec0` UNORM8 surfaces | 0.9990 | max code delta ≤ 8 |
| `dec0` SNORM8 surface | 0.9995 | max code delta ≤ 3 |
| `dec0` f16 surface | 0.9980 | every difference one E4M3 code from the table |
| the final frame (`dec0` native, rest of the chain native) | 0.9990 | max 8 ULP, \|mean\| ≤ 1 ULP |

**Structural invariants**, which are what distinguish (a) from (b):

1. every differing covariance/gate sample matches the activation table **one
   code** from the reference (a larger distance means a real value error);
2. no pixel differs on two surfaces (an upstream error makes them coincide);
3. the end-to-end residual is zero-mean to within half an output ULP (a bias
   means a systematic error, not boundary noise).

A stage that fails an identity floor but passes the invariants is *classified*
rather than accepted blindly; a stage that breaks an invariant fails regardless
of its identity fraction.  The floors are for this frame: a new frame
re-measures, it does not inherit.

## 5. What this means for B5

- The matrix contract is right to within one f16 ULP on roughly 97% of values and
  exactly on the rest.  B5 runs the same `mma.m16n8k32...e4m3.e4m3.f16` chain 360
  times per frame, so a contract error would appear as a *coherent* residual (many
  values, biased), which invariant 3 detects.  It should not be pre-emptively
  "fixed" now.
- The residual does not compound through the chain: it arrives at the frame at
  0.064% of samples, 1 ULP median, and the stages downstream reproduce their
  references exactly when given exact inputs.
- The one thing B5 must not do is inherit `dec0`'s *outputs* as if they were
  reference data: the fixture compares against the capture, and the tolerance
  table above is what a native `dec0` is allowed to differ by.

## 6. What was deliberately not done

No change was made to `dec0`'s accumulation contract to chase byte identity.  The
evidence says the residual is *at* the f16 rounding boundary of a value that is
stored as an f16 anyway, not a systematic error: a f16-ULP-level contract
difference is expected to flip roughly 0.5% of the values it touches, and the
measured flip rate is 0.015% - consistent with ~3% of values differing by one ULP
and 97% bit-identical, with no coherent direction.  Forcing the remainder would
mean reproducing NVIDIA's internal `mma` accumulation order - a hardware-semantics
reverse-engineering project with no effect on the frame beyond 0.064% of samples at
1 ULP.  If B5's ten layers show a coherent residual, that question changes and this
decision should be revisited.

## 7. Limitations

- One frame, one adapter, `reset = 1`, zero motion.  The propagation measurement
  covers `dec0 -> post -> downsample` only; the ten layers are not in the chain
  yet.
- The `mix` mechanism is supported by the probability-conservation signature but
  not proven at the strength of the `cov`/`gate` table match, because the softmax
  is a function of three codes rather than one.
- The "~3% of expansion values differ by one ULP" figure is inferred from the
  flip rate (0.233% of pixels over 16 used columns per pixel) and the ratio of the
  f16 ULP to the E4M3 step, not read from the hardware.
- The floors are set with margin on a synthetic frame whose colours are flat
  blocks; a natural frame with more values near quantisation boundaries may sit
  lower, and must be re-measured rather than assumed.

## 8. Files

- `scripts/m_conformance.py` - the criteria, the E4M3 table and the classifier
- `tests/test_conformance.py` - the table, the classifier, the invariants and the
  verdict logic, from synthetic surfaces
- Reproduce: run the four fixtures, then `python scripts/m_conformance.py`
