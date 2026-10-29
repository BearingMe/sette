# EXP-018: The Stage Regression Gate (M05 B4.7)

**Milestone:** M05 (checkpoint B4.7)
**Status:** **PASS** — one command runs every reconstructed stage, compares every
declared output, rebuilds the propagation chain, and applies the EXP-017 criteria.
**Depends on:** [EXP-012](EXP-012-m05-native-downsample.md),
[EXP-014](EXP-014-m05-dec0-mapping.md),
[EXP-015](EXP-015-m05-enc0-contract.md),
[EXP-016](EXP-016-m05-post-reconstruction.md),
[EXP-017](EXP-017-m05-dec0-fidelity.md) (the criteria it enforces).

## 1. What it is

`scripts/m_conformance.py` is the gate.  It replaces the ad-hoc sequence of
`m_stage_fixture.py run <stage>` per stage plus a separate comparison pass.

```
python scripts/m_conformance.py            # run every stage, then gate
python scripts/m_conformance.py --no-run   # gate existing results (still requires them)
```

It runs `downsample`, `enc0`, `dec0` and `post` from their captured fixtures, then
rebuilds `post-from-native` (post fed by the native `dec0` surfaces) and
`downsample-from-native` (downsample fed by that post), and compares each against
the captured reference.

## 2. The failure modes it closes

The gate's design goal is that **a comparison it did not make cannot pass**.  Each
stage declares its outputs up front, and each of these is a failure rather than a
note:

| condition | before B4.7 | now |
|---|---|---|
| results missing | printed `no results (run the fixture first)` and continued to "all criteria met" | `missing result` |
| reference missing | stage silently skipped | `missing reference` |
| result shorter than the reference | compare would fail on shapes, or pass on the prefix | explicit size check |
| result shorter than the fixture's own declaration | invisible | `the fixture declares N B` |
| output not declared by the fixture | invisible | `does not declare this output` |
| empty comparison | `identical` of an empty array | explicit check |
| stale results from an earlier run | possible with `check` | `--run` clears and regenerates |

The comparison for the f16 surface is by **bits**, not by value: 47 of `dec0`'s
covariance differences are `-0.0` against `+0.0`, which a float comparison hides
while the bytes differ.

`tests/test_conformance.py` builds a synthetic fixture tree and asserts that each
row of that table produces a problem and an empty result set - the gate's own
failure behaviour is tested, not just its success path.

## 3. Reproducibility

The gate runs every stage twice and requires the two runs to be byte-identical.
That check exists because it caught a real defect on its first run.

**It found `post` was not deterministic.**  Until then `post` had been measured as
byte-exact over two runs (EXP-016 §5); a 16-run sample showed 4 of 16 producing a
different result - always in one block, always at pixels whose *display-motion*
fetch was out of bounds, with the red channel saturating to its maximum.  The
shader fetched `Motion` at `trunc(pos/recon*display)`, i.e. up to (1919, 1079),
while the bound texture is 960x540, and that out-of-bounds read perturbed the
dispatch.  The fetch is dead on the `reset = 1` path (the history sample cannot
depend on the motion vector), so it was removed; the shader now runs 16 of 16
identical and exact.  The load must return with the Catmull-Rom path when the
history branch is implemented.

This is also the answer to a question EXP-016 left open: the motion texture's
sampling window (1919x1079) exceeds the captured texture (960x540), so the OOB
read is structural to the capture, not a fixture mistake.

Two consecutive full gate runs produced identical results for all nine surfaces
and the frame (compared as JSON).  The stages are deterministic as well: three
runs of `dec0` and sixteen of `post` are byte-identical.

The gate's frame figure supersedes the ad-hoc measurement first recorded in
EXP-017 §3 (99.9364%, 5278 samples): the gate rebuilds the chain from scratch on
every run and reproducibly reports **99.9369%, 5237 of 8 294 400 samples, max 7 ULP,
mean -0.193 ULP**.  The ad-hoc harness is not retained.

## 4. Current status

| scope | identical | residual | verdict |
|---|---|---|---|
| downsample | 100.0000% | 0 | pass |
| enc0 | 100.0000% | 0 | pass |
| dec0 gate | 99.9988% | 19 samples, max 3 codes | pass |
| dec0 mix | 99.9130% | 1353 samples, max 8 codes | pass |
| dec0 feat | 99.9787% | 442 samples, max 3 codes | pass |
| dec0 cov | 99.9814% | 290 samples, max 25% rel | pass |
| post (packed / history / half) | 100.0000% | 0 / 0 / 0 | pass |
| final frame (native dec0 chain) | 99.9369% | 5237 samples, max 7 ULP | pass |

Structural invariants: `dec0`'s four surfaces remain disjoint, and every differing
covariance value matches the E4M3 activation table exactly one code from the
reference.

## 5. Limitations

- The gate needs the capture-derived fixtures (they are built by `export` and are
  not in the repository, as the NVIDIA-derived data must stay local).
- It covers the four image stages.  The ten-layer network is not in the chain yet;
  when it lands, B5 adds its stages to `STAGES` and its edge to `CHAIN` and the same
  gate applies unchanged.
- The thresholds are the canonical frame's.  A new frame re-measures (EXP-017 §4).

## 6. Files

- `scripts/m_conformance.py` - the gate
- `tests/test_conformance.py` - the table, the classifier, the invariants, and the
  completeness failure modes
