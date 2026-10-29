# EXP-019: The Native Ten-Layer Network (M05 B5)

**Milestone:** M05 (checkpoint B5/C20)
**Status:** **IN PROGRESS** — all ten layers execute natively and are reproducible,
but the native chain does not reproduce the captured dec1 field; the per-layer
residual compounds.  The chain gate therefore fails (EXP-019 §7).
**Depends on:** [EXP-003](EXP-003-m05-layer-capture.md),
[EXP-006](EXP-006-m06-operator-contract.md),
[EXP-014](EXP-014-m05-dec0-mapping.md),
[EXP-015](EXP-015-m05-enc0-contract.md),
[EXP-017](EXP-017-m05-dec0-fidelity.md),
[EXP-018](EXP-018-m05-regression-gate.md).

## 1. Question

Can the ten neural layers between `enc0` and `dec0` be executed by Sette's own
D3D12/HLSL path, consuming each other's native outputs, and reproduce the
captured `dec1` field?  The layer-scope reference already replays byte-exactly
([EXP-003](EXP-003-m05-layer-capture.md)); this experiment asks whether the
*reconstruction* does.

## 2. Acceptance criteria (declared before the run)

1. Every layer output is compared at its declared size, with missing/short/
   undeclared outputs a failure (the EXP-018 rule).
2. **Layer floor 90%** of e4m3 codes.  This is a regression bound, not the
   claim; the measured minima are 91.1% (`enc2` merge) and 93.6% (tube#6).
3. **Chain floor 99%** of the 8 355 840 captured `dec1` bytes.  A chain below
   this is not a faithful reconstruction.
4. Two executions must be byte-identical.

## 3. Execution architecture

One dispatch per operator, all buffers GPU-resident, one command list:

| Dispatch | Input | Output |
|---|---|---|
| `expand` (decoders) | low-resolution e4m3 field | f16 `[low tokens][4C]` |
| `prepare` | e4m3 field (encoder) or expand + skip | f16 `x0 [padded tokens][C]` |
| `norm1` | `x0`, `g1`, `bo` | `h1` (q8 values), `res = x0+bo` |
| `q`, `v` | `h1`, weights | f16 projections `[tokens][32H]` |
| `attention` | `Q`, `V`, relative bias | q8 attention output `O` |
| `wo` | `O`, `Wo`, `res` | `x1 = f16(O Wo + res)` |
| `norm2` | `x1`, `g2`, `b2` | `h2`, `x2 = x1+b2` |
| `mlp` | `h2`, `w1/b1/w2`, `x2` | `x2` after all `C/8` chunks |
| `store` | `x2` | e4m3 field |
| `merge` (encoders) | `x2`, merge weights | e4m3 merged field |

`m_network_stage.cpp` runs a graph manifest: named DEFAULT-heap buffers, a
fixed sequence of dispatches, explicit resource-state transitions and UAV
barriers between stages, and a readback only of the fields the manifest asks
for.  A resource bound as an SRV is never also written as a UAV in one
dispatch, so the state of each resource is unambiguous.  A resource without an
init file is filled with a position-dependent sentinel; two runs with different
seeds distinguish bytes a stage wrote from bytes it left alone
(`m_layer_fixture.py check`).

The ten layers are 94 dispatches and 146 resources.  Mode A runs each layer
against its captured input; mode B concatenates the same dispatches and rewires
each consumer's input to the producer's resource.  The chain never round-trips
through the host.

## 4. Numerical contract

The block runs in **natural channel order**; the kernel's pair-order and
fragment permutations are folded into the weight axes, so no channel
permutation survives into the shader.  `natural_matrices()` builds each matrix
from the captured weight blob with the kernel's own `woff_table` and the
permutations derived in §5.  The following are the contract:

- **k32 step**: the 32 products are accumulated in f32 (the width of the
  captured MMA's accumulator) and added to the f16 accumulator with one
  round-to-nearest-even; `R16D` rounds an exact double to f16 in one step and is
  used wherever the exact result is available.
- **f16 operations**: each rounds once.  `Add16`, `Mul16`, `Fma16` compute the
  exact result in f64 and round once.  (A TwoSum-based f32 version was measured
  and rejected: it rounded one polynomial case the wrong way.)
- **RMS**: `sumsq_tokens` for norm1 and `tree_norm2` for norm2, with the
  kernel's own trees, `+2^-13`, f16 add, f32 sqrt and f32 reciprocal, one f16
  rounding.
- **Attention**: one k32 score step with the bias as the accumulator, the
  clamped cubic, the bit-level exponent (pair packing, `<<5`, `+0x7FF88000`),
  the `c^8,c^2,c^4,c^1` butterfly, f16 reciprocal, and a K=16 `P V`.
- **GELU**: the captured cubic `z*(0.5 + cz*(0.412109375 - 0.0810546875*|cz|))`,
  with the f16 constants (the literals `0.41216981`/`0.08108133` round to
  exactly these), then q8.
- **Patch merge**: `q8(x2)` values into a `4C` row, K=`4C`, the `+-2*pi` clamp,
  then e4m3.

### 4.1 Negative results (enc1, identity of the layer output)

The contract was chosen by measurement, not by preference:

| accumulation | enc1 identity |
|---|---:|
| exact sum per k32, accumulator inside (chosen) | **96.924%** |
| one f16 rounding per k16 | 72.537% |
| one f16 rounding per k8 | 61.188% |
| f32 through the whole GEMM, one rounding at the end | 78.936% |
| accumulator added after the k32 chain | 81.079% |

Two independent f32 reduction trees (sequential / pairwise / 4-way) produced
*no* change in any f16 result on this data, so the residual is not an
accumulation-order effect.

## 5. The natural-order form, proven before it was compiled

`KernelBlock` runs the kernel-order model (`swin_model.swin_block`: pair-order
fragments); `NaturalGridBlock` runs the natural-order form intended for HLSL.
They are **bit-identical on all ten captured layers** (`m_layer_oracle.py
--cross-check`).  The permutations that make that true:

| matrix | natural form |
|---|---|
| Q, V per head | columns `gp` (fragment column -> natural channel) |
| Wo | raw `woff_table` image, columns `gp` (pair-order output) |
| w1 | columns `gp`; `b1` permuted by `gp` |
| w2 | raw image, columns `gp` |
| g1/bo/g2/b2 | `plain_layer`'s already-naturalised vectors |
| merge/expand | `plain_layer`'s natural-order images |

The model and the native shader then agree **bit-for-bit**: every per-layer
number in §6 is identical to the CPU oracle's, and the dumped intermediates
(`x0, h1, q, v, o, x1, h2, x2, x2b`) match the oracle exactly on the layers
inspected.

## 6. Per-layer measurements

Native layer against the captured field (mode A, captured input).  All outputs
are e4m3 fields; `merge` is the encoder's patch-merge output.

| Layer | Output | identical | differing | decoded rel RMS |
|---|---|---:|---:|---:|
| enc1 | out | 96.924% | 257 000 | 0.41% |
| enc1 | merge | 91.288% | 273 000 | — |
| enc2 | out | 96.890% | 97 437 (of 3 133 440) | 0.19% |
| enc2 | merge | 91.139% | 92 551 | — |
| tube#1 | out | 96.376% | 37 847 | 0.35% |
| tube#2 | out | 98.586% | 14 768 | — |
| tube#3 | out | 97.899% | 21 947 | — |
| tube#4 | out | 98.745% | 13 112 | — |
| tube#5 | out | 97.221% | 29 029 | — |
| tube#6 | out | 93.584% | 67 014 | — |
| dec2 | out | 94.853% | 161 281 | — |
| dec1 | out | 95.365% | 387 282 | 0.36% |

Every layer is deterministic (two executions byte-identical).  The raw maximum
code delta is large only because e4m3 code distance is not value distance (a
sign flip is 128 codes); the decoded relative RMS is the meaningful figure,
0.2-0.4%.

`dec1` is the field `dec0` consumes, so the physical contract is satisfied: the
native `dec1` output is the same 480x272x64 two-plane field the existing `dec0`
stage reads.

## 7. The chain (mode B)

Concatenated and rewired, the ten layers produce a native `dec1` field:

| | identical | differing | deterministic |
|---|---:|---:|---|
| chain `dec1` | **21.327%** | 6 573 786 / 8 355 840 | yes |

Propagation, native field against its captured counterpart at each boundary:

| Stage | identical |
|---|---:|
| enc1 out / merge | 96.924% / 91.288% |
| enc2 out / merge | 41.131% / 44.752% |
| tube#1 | 34.765% |
| tube#2 | 28.424% |
| tube#3 | 25.209% |
| tube#4 | 25.643% |
| tube#5 | 26.786% |
| tube#6 | 13.785% |
| dec2 | 18.535% |
| dec1 | 21.327% |

**This is the blocker.**  The per-layer residual is a real 0.2-0.4% numerical
difference, not one-code boundary noise: a layer fed a field with 8.7% wrong
codes (enc1's merge) falls to 41%, and the error does not recover.  d4r's
warning that a single FP8 rounding flip changes a value by 6-12% and compounds
through the layers is exactly what is observed.  A faithful chain therefore
requires byte identity at layer scope; 95% is not enough.

## 8. What is established, and what is not

**Established by measurement**

- The ten layers execute natively on the RX 7700 XT through D3D12/HLSL, with
  captured weights and captured inputs, and are deterministic.
- The native shader and the CPU natural-order contract are bit-identical; the
  natural-order contract and the kernel-order model are bit-identical.
- The kernel's k32 contract (one f16 rounding per 32 products, accumulator
  inside the step) is the closest of five measured accumulation contracts, by a
  wide margin (96.9% vs 61-81%).
- The chain runs as one 94-dispatch D3D12 graph with GPU-resident intermediates.

**Not established**

- **Byte identity to the capture.**  The earliest unresolved divergence is the
  per-token f16 rounding sequence inside the block: the residual is uniform
  across channels, unbiased (signed mean ~1e-4), and not explained by the
  accumulation contract, the RMS square-root variant, or the q8 round trip
  (each tested).  It is therefore a remaining operation-level difference
  between d4r's numpy reconstruction and the captured ZLUDA execution, and it
  must be localized against `swin_block.h`/the original PTX before B5 can pass.
- **A faithful chain.**  Because the residual compounds, the chain is 21.3%,
  far below the 99% floor.

B5 is therefore **incomplete**, and the gate says so: `m_conformance.py
--network` exits 1 on `network: the native chain did not meet its floor`.

## 9. Toolchain findings (durable)

- `-enable-16bit-types` on DXC 1.8.2502.11 miscompiles 16-bit integer
  truncation: `(uint)(uint16_t)asuint(x)`, `asuint(x) & 0xFFFF`, `(x<<16)>>16`
  all evaluate to 0.  The shader therefore uses the project's explicit
  32-bit integer f16 conversion and no 16-bit types.
- A read-modify-write of a `RWBuffer<uint>` element inside one dispatch (two
  half stores into one word) does not land on this driver.  The shader writes
  both halves in one store.
- Resources left in `COMMON` while written through a UAV are not written; the
  graph runner transitions every binding explicitly.  A dispatch also requires
  `SetPipelineState` (reset alone does not bind one).
- A descriptor heap must place UAV descriptors after the SRV range; writing
  them at the SRV slots silently drops every write.

## 10. Reproduction

```powershell
# one layer, independently
python scripts/m_layer_fixture.py export CAPTURE --layer tube#2 --out DIR
python scripts/m_layer_fixture.py run DIR
python scripts/m_layer_fixture.py check DIR

# all ten + the chain
python scripts/m_layer_suite.py export CAPTURE --out LAYERS
python scripts/m_layer_suite.py check LAYERS
python scripts/m_layer_suite.py chain LAYERS      # exits 1 below 99%

# the consolidated gate (B4 stays green; --network adds the B5 verdict)
python scripts/m_conformance.py --network
```

Fixtures and weight blobs are capture-derived and stay in the ignored local
workspace.  The pinned d4r sources are read, never modified.

## 11. Files

- `src/native/swin_layer_cs.hlsl` — the block's seven entry points
- `src/native/m_network_stage.cpp` — the graph runner
- `scripts/m_layer_oracle.py` — the natural-order contract and the capture oracle
- `scripts/m_layer_fixture.py` — one-layer export/run/check
- `scripts/m_layer_suite.py` — the ten layers, mode A and mode B
- `scripts/m_conformance.py` — `--network` adds the B5 verdict
- `tests/test_network_contract.py` — the capture-free contract tests
