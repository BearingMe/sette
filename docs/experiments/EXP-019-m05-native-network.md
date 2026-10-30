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

## Addendum: enc1 divergence probe and reference provenance (2026-10-29)

**Outcome: INCONCLUSIVE for the first divergent runtime operation; B5 remains
incomplete.** No production arithmetic, weights, thresholds, or backend changed.
The earlier description of the residual as an established per-token rounding
error is too strong: that remains a hypothesis.

### Verified mapping and reconstruction boundaries

[M] `scripts/m_enc1_probe.py` selects complete interior 4x4 windows and verifies
their input bytes by explicit allocation offsets, decoded field indexing, and
`swin_model.load_tokens`. It also verifies fixture geometry and the entire packed
weight buffer against the capture-derived matrices. Output comparison uses raw
codes in the two physical channel planes, not decoded values or padded-grid
reshaping. The previous ad-hoc perturbation probe is not used.

| Selected window | Block (bx,by), window | Field coordinates | Mismatching codes |
|---|---|---|---:|
| Worst interior mismatch | (4,1), 3 | x=34..37, y=10..13 | 398 / 1024 |
| Exact control | (2,2), 2 | x=14..17, y=18..21 | 0 / 1024 |

[M] For both selected blocks, CPU and native dumps agree at **all nine** exposed
boundaries: `x0,h1,q,v,o,x1,h2,x2,x2b`. The native enc1 output and merge are
byte-identical across two executions with different scratch sentinel seeds.
This establishes reconstruction consistency, **not original-kernel fidelity**.

[M] Each token in both windows was independently perturbed by one f16 ULP in
norm1 reciprocal, norm2 reciprocal, attention scale, and softmax reciprocal.
There are 128 experiments per window. The largest improvement is token 61's
softmax reciprocal +1 ULP: window mismatches 398 -> 357, with no newly wrong
codes in that window; the token itself has one remaining wrong code. This is
sensitivity evidence only. Other perturbations can regress exact control codes.
No perturbation was promoted into the implementation.

### Original PTX, interpreted rather than executed

[R] Symbol search identifies `rrlite_enc1_4x4` in local
`m-ptx-m05/dlss-0156-00.ptx` (not an assumed module index), SHA-256
`f51bdc9e7b0e136865cc90ebdf1702614877a6cef65c387127038ac3e42e5a04`.
The DLL hash is `be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e`.
Relevant bounded slices include:

- norm1 f16-to-f32, `rsqrt.approx.ftz.f32`, RNE conversion back to f16:
  lines 1204-1213;
- attention scale f32 constants, `div.approx.ftz.f32`, RNE f16 conversion:
  3197-3203;
- clamped polynomial and two **fused** f16x2 instructions: 3207-3267;
- MLP clamp/multiply/subtract/multiply/add/multiply, with separate f16x2
  roundings: 8032-8094.

[M] `scripts/m_enc1_ptx_probe.py` runs the pinned `ptxsim.py` directly on this
PTX and the captured input allocation. For each selected block all 4096 ordinary
output bytes are confirmed written. The interpreted PTX output is byte-identical
to Sette's CPU model: **0 differences** in either block. Against the capture,
the blocks differ by 464 and 37 bytes respectively (the control *window* is
exact, not its whole block). The interpreter stops on its unsupported divergent
backward branch at line 20505 during merge stores; **merge execution is partial
and is not validated by this probe**. Approximate instructions are replaced by
correctly-rounded operations in this interpreter. This is not NVIDIA runtime
evidence and cannot rule out every hardware/lowering difference.

[M] A D3D12-only diagnostic (`m_reciprocal_probe.py`,
`reciprocal_probe_cs.hlsl`) evaluates `rsqrt` and `rcp` for all widened binary16
arguments. For the 31,743 positive finite nonzero inputs, respectively 3,223 and
2,746 results differ from correctly-rounded f32, but **zero differ after f16
rounding**. This does not explain the residual, and does not prove what another
compiler or runtime executed. Final-f32, sequential-f32 and pairwise-f32 P/V
diagnostics also change no q8 attention codes in either selected block.

### Critical correction: the canonical capture selected native replacements

[M] The retained capture log
`%TEMP%/opencode/m-capture-run/probe-1080b.log:510-516` identifies
`d4r-runtime/d4r/native/gfx1101/rrlite_enc1_4x4.hsaco` and explicitly reports
`module_direct kernel="rrlite_enc1_4x4" translated_jit=skipped`.
The same log identifies native replacements for enc2 and the tube kernel.
Thus the canonical enc1 reference is **d4r-native execution selected by ZLUDA**,
not independent execution of the original NVIDIA PTX through its translator.

[S] The local Windows source checkout is `7da2f9ac399404c23b2afe2dc26f245fca16fffb`;
`external/ZLUDA/zluda/src/impl/function.rs:215-242` adjusts native launch geometry,
launches the selected function and only then calls `replay.finish`. This explains
why the recorded original launch dimensions alone do not establish the backend.
The pinned mathematical sources remain `05e19d402781083118d106f2a21d0cf2bc43f744`.

[R] REA rejected the replacement's GPU ELF as unsupported architecture; static
inspection used installed ROCm `llvm-objdump -d --mcpu=gfx1101` instead. The
currently installed module hashes to
`aa5e403e0abdcc8299f265122fe37787da27e4217161bc39e147878ef8f6c6c1`.
Its ISA has packed f16 polynomial FMAs (e.g. addresses 0x5994..0x5a00).
The historical capture log binds a module **path**, not its historical hash;
the current module must not be assumed byte-identical to the capture-time file.
No HIP compilation, execution, or backend experiment was introduced here.

This corrects the independence claim, not the measured byte comparisons in §6-7.
Exact replay of a native replacement against output produced by that replacement
does not establish original NVIDIA fidelity. The original PTX and Sette agree
under the interpreter's semantics in the tested blocks; the first difference
inside the actual capture-producing replacement remains **unknown**.

### Reproduction, validation, and missing evidence

All diagnostic data remains local under `%TEMP%/opencode/enc1-divergence-worst`,
`enc1-divergence-control`, and `enc1-reciprocal`; no tensors, ISA, PTX, or modules
are added to Git. With `CAPTURE`, `ENC1_FIXTURE`, `PTX` and `OUT` set to local paths:

```powershell
python scripts/m_enc1_probe.py CAPTURE ENC1_FIXTURE --out OUT --native --perturb
python scripts/m_enc1_ptx_probe.py PTX CAPTURE --block 4 1 --out OUT
python scripts/m_enc1_ptx_probe.py PTX CAPTURE --block 2 2 --out CONTROL_OUT
python scripts/m_reciprocal_probe.py --out RECIPROCAL_OUT
python -m unittest discover -s tests -p "test_*.py"
python scripts/m_conformance.py
```

[M] **180 tests pass**, including three new coordinate regressions. B4 passes
again, final frame 99.9369%. Native enc1 was rerun twice; no fidelity correction
was made. The ten-layer suite/chain were not rerun because this investigation
remains enc1-only and no shared fix exists. Their prior failed measurements and
the 99% chain floor remain unchanged.

Next independently verifiable action: obtain an enc1-only runtime reference that
logs **translated** execution with the native replacement disabled (preserving
this capture), or instrument the exact capture-producing replacement and bind
its module hash to the trace. The required intermediate evidence is `h1`, Q/V,
softmax probabilities/O, `x1`, `h2`, and MLP boundaries for the verified windows.
Without either reference, naming the first runtime-divergent operation or changing
Sette's arithmetic to fit these bytes would be guessing. Byte-exact enc1 and merge
have **not** been achieved; B5 must not be closed.

## Addendum: original enc1 translated execution captured (2026-10-29)

**Capture validation: PASS. Numerical B5 status: unchanged/incomplete.**

[M] `scripts/m_capture_translated_enc1.py` creates a fresh private native-module
directory, removes only `rrlite_enc1_4x4` from its allowlist, and omits that code
object. All other native modules are copied unchanged. Neither the installed
runtime nor the old captures/fixtures are modified. A new process uses that
directory with verbose dispatch logging and an enc1-only capture filter.

Two fresh runs succeeded (`returncode=0`, `PASS D3D12_DLSS`, Preset 13, gfx1101):

- `%TEMP%/opencode/enc1-translated-20261029/`
- `%TEMP%/opencode/enc1-translated-20261029-repeat/`

[M] The first `probe.log:370-372` records the allowlist miss and translated
fallback; **line 459 confirms the actual dispatch**:

```text
[d4r-launch] kernel="rrlite_enc1_4x4" backend=translated grid=61,35,1 block=32,1,2
```

The enc1 capture begins at line 458 and completes at line 460. There is exactly
one translated enc1 dispatch, zero native enc1 dispatches, and no enc1
`module_direct` shortcut. Line 461 confirms enc2 remains `backend=native`.
Thus this executes NVIDIA's original PTX **through ZLUDA on AMD**; it is not a
claim about NVIDIA hardware execution. Other network layers remain replacements.

[M] The dumped module is `ptx/module-246ed4dd4573ee6d-0000.ptx`, whose SHA-256 is
`f51bdc9e7b0e136865cc90ebdf1702614877a6cef65c387127038ac3e42e5a04` in both runs:
exactly the original enc1 PTX identified in the preceding investigation. The
script records executable/DLL hashes, command and environment in `run.json`.
Translation settings include WMMA/FP8 rewriting enabled, native FP8 disabled,
and `D4R_ZLUDA_WMMA_F16_REFERENCE=1`; these are explicit properties of this
ZLUDA reference, not proof of equivalence to NVIDIA tensor-core hardware.

[M] Both runs have byte-identical enc1 input, weights, output and merge fields.
The input and weights also match canonical `capture-1080b` exactly, so the
new comparison is not confounded by a changed upstream tensor.

| Field | Bytes | Native Sette vs translated capture | Old native-replacement reference vs translated capture |
|---|---:|---:|---:|
| enc1 output | 8,355,840 | 96.418337% (299,278 differ) | 98.491175% (126,075 differ) |
| enc1 merge | 3,133,440 | 89.837910% (318,423 differ) | 94.394595% (175,642 differ) |

The translated output-field SHA-256 is
`1d99c4d685458fe50e28b243b5058409ee0a458895e8edeb5e84b2382d29f728`;
merge is `dc8cbbf4d2c161bfa71bd6082e187788762768346a3042277c23c1f42ac33ecc`.
These are hashes of the actual field ranges, not the entire arena. Local
`comparison.json` and `repeat.json` retain the comparisons and repeat checks.

Reproduce into a **new, nonexistent** output directory:

```powershell
python scripts/m_capture_translated_enc1.py --out NEW_LOCAL_DIRECTORY
```

The capture is under `capture/replay-000001-rrlite_enc1_4x4/`; its `args.bin`,
manifest, input allocations and `gpu-output` allocations are preserved locally.
No proprietary payload, PTX, code object or frame is tracked. The old reference
was deliberately not replaced in the conformance fixtures. The next numerical
probe can now use this independently routed enc1 output; original runtime
intermediates and the first divergent operation are still unestablished.

## Addendum: oracle provenance and controlled contract validation (2026-10-29)

**Bounded enc1 comparison/localization: PASS. Authoritative original numerical
contract: INCONCLUSIVE. B5: incomplete.** No production arithmetic was changed.

### Question and controls

Does the canonical reference use incompatible fast arithmetic, accounting for
Sette's discrepancy? Build pinned conservative enc1 first, replay the existing
input, then independently identify fast, conservative, Sette, and translated-PTX
paths. Keep the 99% chain gate and all old fixtures unchanged.

[S] Pinned source is clean at `05e19d402781083118d106f2a21d0cf2bc43f744`.
`rrlite_enc1_4x4.hip:4-5` defines the fast macros; `swin_common.h:40-57`
restores quantization and per-k32 half accumulation under `D4R_ACCURACY`.
The default conservative `k32` at `swin_common.h:625-633` actually executes
two FP16-operand/FP32-accumulator AMD WMMAs with C in the first, then converts
to FP16. Its source label "exact" does not establish hardware exact-sum behavior
or NVIDIA equivalence. Sette's CPU/HLSL contract uses an exact k32 product sum
and accumulator with one RNE half rounding. These must be measured, not equated.

[M] `scripts/m_enc1_contract.py` builds **separate** fast/conservative modules,
checks effective preprocessed macros (accuracy must undefine both fast macros),
and runs the explicit module path with `d4r_native_replay_probe.exe`. The probe
loads that module directly with `hipModuleLoad`, resolves the intended kernel,
runs its prep and dispatch, synchronizes and reads back the captured allocations.
This bypasses ZLUDA override/cache selection entirely. Each log proves
`grid=61,35,1 block=32,1,4` and successful module load/prep/dispatch. Each native
mode runs in two fresh processes; Sette runs twice with different sentinel seeds.

The installed ROCm 7.2 compiler initially failed on the newer
`__builtin_elementwise_{minimum,maximum}num` spellings in the pinned merge clamp.
No pinned file was changed. Both builds use explicit spelling aliases to the
older `__builtin_elementwise_min/max`; a separate retained LLVM-IR probe verifies
`llvm.minnum.v2f16` / `llvm.maxnum.v2f16`, without fast math. This is a documented
toolchain compatibility adaptation, not a universal NaN/signed-zero conformance
claim. All compared output fields are finite. An earlier device-function IR
probe was optimized away; it was corrected with `used` before relying on it.

### Execution identities

Local final experiment: `%TEMP%/opencode/enc1-contract-final-20261029/`.
`report.json` owns full commands, compiler identity, source hashes, binary hashes,
effective macros, replay logs, capture/fixture hashes, Sette executable/shaders,
field hashes and comparisons. The builds use gfx1101, C++17, `-O3`, device-only
HIP, `--no-gpu-bundle-output`, and the pinned gfx1101
`-DSWIN_BUFFER_BLOAD_FAST`; only conservative adds `-DD4R_ACCURACY`.

| Explicitly executed binary | SHA-256 |
|---|---|
| Pinned conservative `rrlite_enc1_4x4.hsaco` | `dcf5b5727ee474d6da18e1ffc5e304a50503f2bcb8bcb7cbcfdd468220b7dad3` |
| Pinned fast `rrlite_enc1_4x4.hsaco` | `113bf4bc6efad9aa644f370ae9f33ba8ac5c1a345d038fd02cc87f000c82852d` |
| Sette `m_network_stage.exe` | `85c062a9efbe0377d895542cbc5db032b0c9783b8afe2f9bed9d453cbdf5b4ab` |

[M] Input field SHA-256 is
`aaf0801b8d4ece12f4c2575471ab05e13137757748557e13961e2544eabf54a2`;
weight allocation SHA-256 is
`8a36288ea3fbfa19a47723d4c549eaa440a733291ad4df0588df2467f8fdee97`.
Both native paths consume exactly the same allocation files. The script also
verifies Sette's `in.bin` equals this field and its packed weights equal the
captured weights under the existing independently checked mapping. Translated
PTX input and weight bytes are identical as well.

| Execution | Output-field SHA-256 | Merge-field SHA-256 |
|---|---|---|
| Conservative (also canonical capture) | `5cbd013febb7ecbbcfc3f722f130567988cc9ad0ec02777e9e4f6991d602044a` | `46d9964621b3b5b87feea80b58fcb53ebc7571429fcf7438a04541171c5b721f` |
| Fast | `00a0ddcea727a7a59ec58c59efde1390a02101e7732ae364cfef3d6eefb7a6be` | `43c90b0da8506fef553075bbb3881cab1c3111d43a5c85008b29e6012ed7ceb4` |
| Sette | `ec2073d3f6a524facf0c25b4ac80be7e936c90156ea6e17522137236f595f114` | `2f4f018e62ed7a7984f578d8b35f8076f2eb0ed41f5fe5b6405b768763b5433f` |
| Translated PTX | `1d99c4d685458fe50e28b243b5058409ee0a458895e8edeb5e84b2382d29f728` | `dc8cbbf4d2c161bfa71bd6082e187788762768346a3042277c23c1f42ac33ecc` |

### Pairwise results

[M] Fields contain 8,355,840 output codes and 3,133,440 merge codes. Relative
RMS is `sqrt(mean((decoded_right-decoded_left)^2)/mean(decoded_left^2))` over
finite pairs. Maximum is decoded absolute deviation, not unsigned code distance;
this avoids treating sign-bit changes near zero as large numerical errors.

| Left vs right | Field | Exact bytes | Differing codes | Relative RMS | Max decoded abs |
|---|---|---:|---:|---:|---:|
| Canonical vs fast | output | 30.537265% | 5,804,195 | 0.08738245 | 0.625 |
| Canonical vs fast | merge | 27.768650% | 2,263,326 | 0.09530241 | 0.625 |
| Canonical vs conservative | output | 100% | 0 | 0 | 0 |
| Canonical vs conservative | merge | 100% | 0 | 0 | 0 |
| Conservative vs Sette | output | 96.924307% | 257,000 | 0.01240001 | 0.5 |
| Conservative vs Sette | merge | 91.287531% | 273,000 | 0.01942310 | 0.5 |
| Fast vs Sette | output | 30.528840% | 5,804,899 | 0.08742528 | 0.625 |
| Fast vs Sette | merge | 27.761406% | 2,263,553 | 0.09541363 | 0.625 |
| Conservative vs translated PTX | output | 98.491175% | 126,075 | 0.00832049 | 0.5 |
| Conservative vs translated PTX | merge | 94.394595% | 175,642 | 0.01461920 | 0.5 |
| Fast vs translated PTX | output | 30.599808% | 5,798,969 | 0.08738918 | 0.625 |
| Fast vs translated PTX | merge | 27.710695% | 2,265,142 | 0.09508899 | 0.625 |
| Sette vs translated PTX | output | 96.418337% | 299,278 | 0.01335316 | 0.5 |
| Sette vs translated PTX | merge | 89.837910% | 318,423 | 0.02110843 | 0.5 |

Fast, conservative and Sette outputs/merge are deterministic across two runs.
The conservative run additionally reproduces the **entire** historical output
allocations, not just the field ranges. The translated control's two fresh runs
are byte-identical as recorded in the preceding addendum. Its enc1 dispatch is
rechecked by the contract script against the actual log, rejecting a native
dispatch, `module_direct`, or duplicate translated dispatch. This is original
hash-matched PTX translated by the recorded ZLUDA configuration on AMD, not an
unmodified NVIDIA hardware oracle. No original intermediate trace is inferred
from the interpreter or from matching terminal output hashes.

### First divergent operation localized

[M] `scripts/m_enc1_native_trace.py` instruments only a disposable copy of the
pinned headers. The existing native replay probe downloads diagnostic globals.
**Every instrumented output allocation must hash-identically to the
uninstrumented conservative run**, or the trace is rejected. Natural-order
coordinates and weight operands are checked against the verified probe and
loader, rather than assumed from diagnostic buffer indices.

For block `(4,1)` (all four windows, 4096 values per boundary):

| Boundary | Conservative vs CPU differing FP16 bit patterns |
|---|---:|
| Input `x0` | 0 |
| Norm1 sum / reciprocal | 0 / 0 (64 values each) |
| Quantized norm1 `h1` | 0 |
| Q projection, **first k32 step** | **51** |
| V projection, **first k32 step** | **165** |
| Complete Q / V projections | 109 / 176 |
| Quantized attention output | 12 |

All 64x64 dumped Q and V weight operands exactly match the CPU's mapped
matrices. The first-step independent expected sum uses these E4M3 operands in
FP64, then rounds once to FP16; this does not derive an oracle from AMD WMMA.
For example, Q token 2/channel 40 has exact sum `-0.12738037109375`, RNE half
`-0.12744140625` (`0xb014`), but conservative gives `-0.1273193359375`
(`0xb013`). This is a rounding-boundary discrepancy, not an observed missing
requantization or different input/weight mapping. It does **not** establish
which lower-level WMMA/conversion behavior caused the discrepancy.

[M] Control block `(2,2)` also has exact input, sum, reciprocal and h1; its
first-step Q/V differ at 99/49 values and completed Q/V at 135/81 values, while
quantized attention output is exact. Thus exact final control-window output
does not imply exact intermediate arithmetic. These GPU boundaries are compared
with the same CPU contract whose nine exposed boundaries previously matched
Sette HLSL in these blocks; CPU/HLSL agreement is not original-GPU fidelity.

The accepted traces are retained under
`%TEMP%/opencode/enc1-contract-audited-trace-20261029/` and
`enc1-contract-control-trace-20261029/`. A subsequent exploratory pre-conversion
FP32 dump under `enc1-contract-trace-f32-20261029/` failed an internal consistency
check (1856 values per projection did not round back to the adjacent half dump).
That diagnostic is **not validated and is not evidence about WMMA precision**;
raw files are preserved, and its added instrumentation was removed from the
reproducible trace script. No arithmetic correction was based on it.

### Classification, remaining limitations, and recovery

**H3 is supported:** Sette differs from independently identified conservative
d4r, beginning in the first Q/V k32 projection in the tested blocks. H1/H2/H4
are not supported by these measurements. **H5 still applies to choosing a
fully authoritative original numerical oracle:** conservative d4r also differs
from translated PTX, and no NVIDIA hardware evidence or translated runtime
intermediates establish which complete numerical contract is correct.

The provenance audit of all fourteen historical launches is in EXP-003 and local
`provenance.json`. Native routing is proven for all ten neural launches, but the
historical fast/conservative build flags of the other four unique modules remain
unrecorded; they are explicitly unknown, not inferred from pinned defaults.
Only enc1 has the new controlled mode comparison. This audit limitation prevents
calling the whole historical chain an authoritative conservative reference.

No enc2/tube/decoder recovery or conservative-chain replacement was started:
enc1 must first be validated against the independently translated execution.
Next isolate the implicated k32 slice with a validated operand/result probe;
then, if justified, derive an operator regression before changing CPU or HLSL.
RMS normalization, Q/V and attention have partial boundary evidence here;
output projection, GELU/MLP, residuals, final encoding and merge are downstream
terminal comparisons, not newly validated original intermediates.

[M] B4 reran successfully: final-frame identity remains **99.9369%**, maximum
7 ULP, all existing criteria unchanged. **190 repository tests passed.** A fresh
hardened translated capture at `enc1-translated-contract-20261029/` also passes
the execution-mode checks and reproduces the prior translated input/output/merge
bytes exactly (`fresh-translated-check.json` in the final experiment directory).
The capture runner now removes inherited `D4R_*` / `ZLUDA_*` settings before
applying its explicitly recorded configuration. The repository test suite is run as
`python -m pytest tests -q`; an initial unscoped pytest invocation also collected
reference command-line experiments and failed collection (nine argument errors),
so it is not reported as a passing run. Execution-mode tests reject absent or
contradictory dispatch evidence, wrong effective macros, missing load/dispatch,
unbound capture completions and ambiguous instrumentation anchors.

### Reproduction

Use fresh, nonexistent local output directories. Commands assume the installed
ROCm 7.2 and existing Windows replay probe; full compiler/replay commands are
emitted in `report.json`, not hidden behind an accuracy environment variable.

```powershell
$T = "$env:LOCALAPPDATA\Temp\opencode"
python scripts/m_enc1_contract.py --out "$T\enc1-contract-NEW"
python scripts/m_enc1_native_trace.py --experiment "$T\enc1-contract-NEW" --out "$T\enc1-trace-NEW" --block 4,1
python scripts/m_enc1_native_trace.py --experiment "$T\enc1-contract-NEW" --out "$T\enc1-control-NEW" --block 2,2
python scripts/m_capture_provenance.py --log "$T\m-capture-run\probe-1080b.log" --capture "$T\m-capture-run\capture-1080b" --json "$T\enc1-contract-NEW\provenance.json"
python scripts/m_capture_translated_enc1.py --out "$T\enc1-translated-NEW"
python -m pytest tests -q
python scripts/m_conformance.py --json "$T\enc1-contract-NEW\b4.json"
```

All binaries, generated diagnostic headers, PTX, weights, inputs and outputs stay
local/untracked. The pinned submodule and production shaders remain unchanged.

## Addendum: one first-K32 Q operation, runtime operands and arithmetic (2026-10-29)

**Selected discrepancy characterized; no numerical correction justified. B5
remains IN PROGRESS / FAIL at its existing chain gate.** This addendum supersedes
the earlier lack of translated runtime intermediates **for the two selected Q
operations only**. It does not resolve the whole enc1 discrepancy.

### Selection and direct operand proof

[M] The mismatching window is block `(4,1)`, window 0, token 2, natural Q channel
40, field coordinate `(32,6)`. It has **2** final output-code differences against
the conservative capture. The exact control is block `(2,2)`, **window 2**, token
34, channel 40, field `(16,18)`, with **0** final output differences. Window 0 of
the control block is *not* exact (32 differences): an initial probe incorrectly
used token 2 there, failed the explicit selection check, and was not accepted as
the final control. The final script verifies both window classifications.

[R/M] `scripts/m_k32_ptx_trace.py` replays the **existing** translated enc1 input
allocations, not a newly generated frame. It adds diagnostic stores following
the original first Q K32 instructions in these windows; their calculations and
register operands are retained. The logged dispatch is `backend=translated`,
`grid=61,35,1 block=32,1,2`, with no native dispatch or `module_direct`.
**Both complete ordinary output allocations hash-identically to the existing
uninstrumented translated capture.** Output invariance does not prove identical
register allocation or scheduling; it is the required experimental control,
not an assertion that instrumentation leaves every compiler decision unchanged.

Original PTX: local `module-246ed4dd4573ee6d-0000.ptx`, SHA-256
`f51bdc9e7b0e136865cc90ebdf1702614877a6cef65c387127038ac3e42e5a04`,
PTX 8.7 / `sm_89`. Both sites use
`mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`:

| Window | Original line | A registers | B registers | C registers | D registers |
|---|---:|---|---|---|---|
| Mismatch | 2365 | `%r9747..%r9750` | `%r396,%r397` | `%r1975,%r1975` | `%r404,%r405` |
| Exact control | 2421 | `%r9763..%r9766` | `%r396,%r397` | `%r1975,%r1975` | `%r484,%r485` |

[M] For **each** selected operation, decoding the actual runtime fragment words
proves that all 32 activation E4M3 codes, all 32 weight E4M3 codes, and the zero
FP16 accumulator equal the verified conservative diagnostic operands. No operand
or normalization mismatch was found. The original PTX output is read from the
instruction's raw FP16 D registers, **not inferred from final FP8 codes**.

Both operations use PTX wave 1 (head 1), MMA coordinate `(row=2,col=4)`, D lane
10 / low half of its first D word. The natural weight column is 40. In d4r's
pair-order WMMA representation, this is column-lane 4, accumulator element 1;
the waves are 1 for the mismatch and 3 for the control (which carries window 2).
The runtime input/weight proof, all 32 raw E4M3 and FP16 operand patterns,
decoded values, accumulator and coordinate metadata are local in
`%TEMP%/opencode/k32-contract-evidence-20261029/report.json`. Captured operands are
deliberately not copied into tracked fixtures.

### Independent operator probe and raw results

[M] `scripts/m_k32_probe.py` runs replicated matrix tiles, so every physical
row/column computes the **same** selected dot product. It checks every lane and
result element, compares with the original native boundary, verifies every
dumped FP32 value rounds to the adjacent FP16 result, and rejects unequal tiles.
A separate result-only build omitting FP32 diagnostic stores gives identical
ordinary arithmetic outputs. The HIP module is an arithmetic diagnostic, **not
a new Sette backend**. CPU, D3D12/HLSL, and translated PTX receive losslessly
repacked identical operands. The mathematical oracle sums integer E4M3 products
on the `2^-18` lattice, independently of BLAS and GPU reduction behavior.

The d4r K16 slot orders, from pinned `swin_common.h:343-350`, are:

```text
first:   0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23
second:  8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31
```

Pinned conservative `k32` converts C to FP32, executes two chained
FP16-operand / FP32-accumulator WMMAs, then converts once to FP16. There is no
intermediate FP16 rounding between its two K16s. Sette's production CPU helper
and the diagnostic using the production HLSL `R16D` helper both agree with the
independent integer oracle for all tested cases, including two successive K32s.

| Selected mismatch, C = `0x0000` | Exact mathematical value / bits | Native WMMA probe value / bits |
|---|---|---|
| First K16 + C | `-0.03833770751953125`, FP32 `0xbd1d0800` | `-0.038337692618370056`, FP32 `0xbd1d07fc` |
| Second K16 product sum alone | `-0.08904266357421875` | Not a separate native accumulator operation |
| Chained K32 before FP16 conversion | `-0.12738037109375`, FP32 `0xbe027000` | `-0.1273803412914276`, FP32 `0xbe026ffe` |
| Final FP16 | **`0xb014`** | **`0xb013`** |

Thus the first observed incompatible arithmetic is already the **first K16
WMMA accumulation**, before the final FP16 conversion. The two-FP32-ULP K32
deviation moves an exact FP16 midpoint to one side. The final native conversion
correctly rounds its own FP32 input; an arbitrary FP16 ULP fix would target the
wrong operation. These raw FP32 observations are from the validated replicated
microprobe, **not claimed as a successful in-place dump of the original native
kernel's hidden registers**. Its selected final result matches that kernel.

| Execution | Mismatch first K32 FP16 | Exact-window control FP16 |
|---|---:|---:|
| Integer mathematical oracle | `b014` | `b170` |
| Sette production CPU helper | `b014` | `b170` |
| Sette HLSL helper diagnostic | `b014` | `b170` |
| Conservative d4r boundary and isolated native WMMA | `b013` | `b170` |
| Original enc1 translated runtime D register | **`b014`** | **`b170`** |
| Isolated original PTX instruction through the same runtime | `b014` | `b170` |

[M] A K-grouping control using contiguous `0..15`, then `16..31`, returns
`b014` for the mismatch, but is **not** a proposed correction. Uniform `2^-8`
operand scaling / `2^-16` C scaling / inverse result scaling with the original
d4r slot order still returns `b013`; scaling alone does not explain this case.
The probe also retains a fast FP32-through-chain control: for the contiguous
grouping the selected second operation is `b414` with per-K32 FP16 rounding,
but `b413` when the first FP32 result is retained. This is not a replacement
for the separately identified full fast d4r kernel, whose prior negative-control
measurements remain unchanged.

### PTX contract and limits of equivalence

[D] NVIDIA's [PTX ISA, `mma` precision and rounding](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
(section 9.7.16.5.14, consulted 2026-10-29) states for E4M3/E5M2 operations that
intermediate accumulation uses at least single precision, but **accumulation
order, rounding, and handling of subnormal inputs are unspecified**. It does
not require exact mathematical summation followed by one RNE half conversion.
The locally retained page is `%TEMP%/opencode/ptx-isa-20261029.html`, SHA-256
`a5dd908b2c7dbdb9c6815424b3a355d3ef18a09b7851594367559a8a174b2c9e`.
The Context7 lookup returned CUTLASS rather than this instruction specification;
the precision claim above is from the actual NVIDIA PTX specification.

[M] Thirteen microprobe cases cover captured operands, zero/nonzero C, signed
products, cancellation, E4M3 subnormals, maximum finite operands, positive and
negative FP16 midpoints, and successive K32s. Native, HLSL and translated
operator outputs are deterministic across two runs; HLSL also uses two sentinel
seeds. Maximum finite products overflow to half infinity in this non-saturating
instruction; this is not FP8 encoding's satfinite policy.

The nonproprietary negative midpoint is `-(1*1 + (1/32)*(1/64))`, with all other
products zero and C zero. Its exact FP32 result is `0xbf801000`, exact RNE half
`0xbc00`; native WMMA gives FP32 `0xbf801001` and half `0xbc01`.
The translated instruction also returns `bc01`, while CPU/HLSL return `bc00`.
Changing the unused activation zeros from negative zero to positive zero does
not change this outcome. Since the only nonzero products and their sum are
exactly representable in FP32, this case cannot be explained solely by choosing
a different tree of **correctly rounded IEEE FP32 additions**. It exposes
non-exact WMMA internal arithmetic/rounding, not a prescribed exact-dot contract.
The unspecified PTX rounding permits a difference from mathematical RNE; this
experiment is not an exhaustive PTX-conformance certification.

[R] Actual native and cache-extracted translated code objects are disassembled
in `native-isa.txt` and `translated-isa.txt`. The native probe's initial K32
uses `v_wmma_f32_16x16x16_f16` at `0x1ae8` and `0x1b08`; the translated
two-instruction synthetic kernel contains four such WMMAs (two per PTX MMA).
Scaling/conversion instructions are retained as well. ROCm 7.2's disassembler
marks some translated conversion-register decodes invalid, so this is bounded
instruction evidence, not a complete verified ISA reconstruction. Internal
WMMA reduction/rounding and the original translated instruction's hidden FP32
partials are still unobserved; they cannot be deduced uniquely from its D bits.

### Provenance, failures, and verdict

The final local artifacts are:

- `enc1-k32-original-final-20261029/`: instrumented original PTX, raw runtime
  fragments, output-allocation hashes, explicit dispatch log and API calls.
- `k32-contract-evidence-20261029/`: raw operand/result reports, native/HLSL/PTX
  microprobes, repeated executions, result-only control, actual code objects,
  compiler commands, runtime identity and disassemblies.
- `k32-verified-20261029/b4-b5.json`: fresh full regression-gate measurements.

| Executed artifact | SHA-256 |
|---|---|
| Native arithmetic microprobe | `5912a8a9aa890bf4ace4857fd1b4322aadc37dda00d12cc59d67cf4a085b2d27` |
| HLSL diagnostic | `a4b70d6dfeed595ee0fe1eac31fb0d71fb3652014c458400a7451806f3c0f0d9` |
| Translated operator code object (actual cache bytes) | `8890c624d1e76675e6daff902be921e42089329c7004e8c586d7a8429732eb62` |
| Actual loaded `nvcuda.dll` | `f2880a56b152c3b906a72b5c2d0dd21e4b2d00b43715a2f992e03605d5ddd98c` |

The runtime cache identifies gfx1101 and ZLUDA base
`ee2f25a180099fa42f36b2346732e1f2470a03ad` with additional delta identifiers
and the recorded WMMA settings. The installed `build-info.json` lists an older,
different DLL hash (`71b45190...`); it is not used as proof of the executed
binary. Pinned d4r source describes its conservative algorithm and intended
translation, but does not by itself establish this locally modified translator's
complete behavior. All settings and actual binary/cache hashes are retained.

Three new in-place native FP32-dump attempts (`enc1-k32-raw-*`) preserved ordinary
outputs but **failed internal FP32-to-adjacent-FP16 consistency** (192/256 results
in the first attempt). Unrolling/literal-index stores did not validate them.
They remain rejected diagnostics, not evidence of a compiler bug or WMMA
precision; their generated sources and dumps remain local. The standalone
probe avoids that failure with replicated tiles and independently checks every
result. The first original-PTX replay's PowerShell stderr redirection reported
a shell error despite its passing worker report; the final launcher captures
stdout/stderr and verifies process status and dispatch explicitly.

**Verdict:** A (Sette wrong) is **not supported for this selected operation**:
Sette matches the original translated runtime result. B is supported only in
the narrow sense that d4r's conservative "exact" label is **not mathematical
exact-sum/RNE behavior**; this does not establish a PTX-spec violation. C
(distinct permitted rounding) is supported by the independent synthetic probe.
D is excluded for the selected operands/C. E (compiler bug) is not established;
different lowerings exist, but no faulty lowering has been proven. F remains
for the full original-hardware contract and a bit-exact model of internal WMMA
rounding. No NVIDIA hardware claim is made.

**Correction decision:** retain the existing CPU and HLSL arithmetic. Neither
copying native WMMA's `b013`, nor switching all K groupings, follows from the
original PTX instruction's contract. Only the misleading shader comment equating
the PTX accumulator declaration with an exact FP32 contract was corrected.
No fitted constants, ULP nudges, quantization changes or performance work were
introduced. The remaining smallest question is the precise internal WMMA
rounding model and its relationship to the translated runtime on actual
divergent operands—not another layout/normalization investigation.

[M] Fresh independent replays of all ten neural layers and the full chain
reproduce the prior measurements. With **no arithmetic change**, before/after
enc1 is **96.924307% output / 91.287531% merge** against conservative, and
chained dec1 is **21.3270%**, deterministic. The original instrumented translated
replay is byte-identical, so the prior enc1 comparison against translated remains
**96.418337% / 89.837910%**; no improved full-network fidelity is claimed.
`m_conformance.py --network` returns the expected failure at the unchanged **99%**
chain floor. B4 remains green at **99.9369%**, maximum 7 ULP. **204 tests pass**,
including all 190 existing tests and 14 new nonproprietary regressions.

### Reproduction (new local output directories)

```powershell
$T = "$env:LOCALAPPDATA\Temp\opencode"
python scripts/m_k32_ptx_trace.py --ptx "$T\enc1-translated-contract-20261029\ptx\module-246ed4dd4573ee6d-0000.ptx" --capture "$T\enc1-translated-contract-20261029\capture\replay-000001-rrlite_enc1_4x4" --out "$T\enc1-k32-original-NEW"
python scripts/m_k32_probe.py --out "$T\k32-NEW" --experiment "$T\enc1-contract-final-20261029" --trace "$T\enc1-contract-audited-trace-20261029" --control "$T\enc1-contract-control-trace-20261029" --original-trace "$T\enc1-k32-original-NEW"
python -m pytest tests -q
python scripts/m_conformance.py --network --json "$T\k32-NEW\b4-b5.json"
```

The last command is expected to fail B5, not B4. No proprietary binaries,
weights, PTX dumps or captured tensors are added to Git; the pinned submodule
remains unchanged. No downstream matrix investigation or B6/B7 work was started.

## Addendum: translated-PTX reference for all ten layers and the complete chain (2026-10-29)

**Mode A (independent layers) and mode B (complete chain) both executed. The chain
gate cannot be satisfied by the original PTX itself. B5 remains IN PROGRESS / FAIL at
its unchanged 99% chain floor. No arithmetic, threshold or fixture was changed.**

### 1. Execution provenance

`scripts/m_capture_translated.py` (renamed from `m_capture_translated_enc1.py`; the
enc1 behaviour is unchanged) writes a private native allowlist, copies every other
native module unchanged, removes the named `D4R_*`/`ZLUDA_*` inherited settings, and
records the executable, DLL, command and complete environment in `run.json`. The
translation settings are the enc1 reference's: `D4R_ZLUDA_WMMA=1`,
`D4R_ZLUDA_WMMA_FP8=1`, `D4R_ZLUDA_WMMA_FP8_NATIVE=0`,
`D4R_ZLUDA_WMMA_F16_REFERENCE=1`, gfx1101, and a per-run ZLUDA cache directory.

[M] Every capture in this addendum has `backend=translated` for the selected kernel,
zero `backend=native` dispatches of it, no `module_direct` shortcut, exactly one
translated dispatch per expected launch, and one dumped PTX module whose identity hash
equals the manifest entry for that kernel.

| Layer | Kernel | Dispatch (grid, block) | PTX module SHA-256 | Module identity |
|---|---|---|---|---|
| enc1 | `rrlite_enc1_4x4` | 61,35,1; 32,1,2 | `f51bdc9e7b0e136865cc90ebdf1702614877a6cef65c387127038ac3e42e5a04` | 246ed4dd4573ee6d |
| enc2 | `rrlite_enc2_4x4` | 31,18,1; 32,1,4 | `670204a0fc1d6789b3d1c65a57c90643bde7e44d37003a35e254c226c9e7ae92` | b4758c1bc25cb96a |
| tube#1..#6 | `rrlite_enc3_tube_4x4` | 16,9,1 and 15,9,1; 32,1,4 | `222e99a1fb9f1ac69acf926a5821419cc8dbd2fa9e0e12ffab6f56b337c36e41` | c7b41e54ab68d071 |
| dec2 | `rrlite_dec2_4x4` | 30,17,1; 32,1,4 | `8b86902e88367fab3c7da4fe762ec58d18bc24a459610e3ed212656bd0eb2f53` | b686d5b2f6d57a00 |
| dec1 | `rrlite_dec1_4x4` | 60,34,1; 32,1,2 | `741500aa00bf59b5d707f94fe1ac53fdfc450cbf97cc47c8f6d724dd74e7e2f9` | 482e5a3398d76ec1 |

[M] Two fresh enc2 runs are byte-identical across every captured file (PTX hash
included), and the earlier enc1 pair is identical as well. Translated runs are also
consistent across processes: the chained enc2 input equals the chained enc1 merge
byte for byte, and the chained enc1 output and merge equal the single-layer translated
enc1 capture. This is ZLUDA translation executed on AMD; it is not NVIDIA hardware
behaviour and no NVIDIA-hardware parity is claimed.

### 2. Mode A: each layer from identical input bytes

[M] Each translated capture is exported into a Sette fixture (`m_layer_fixture.py
export`, one replay directory accepted directly), so Sette, the pinned conservative
and fast modules, and the translated PTX consume the same input and weight bytes; the
contract script asserts this before comparing. Fields are e4m3 code stencils; RMS,
maximum and bias are decoded values.

| Layer | Sette vs translated | conservative vs translated | conservative vs Sette |
|---|---|---|---|
| enc1 output | 96.4183% / rms 0.01335 / max 0.50 / bias -0.000031 | 98.4912% / 0.00832 / 0.50 / +0.000016 | 96.9243% / 0.01240 / 0.50 / +0.000047 |
| enc1 merge | 89.8379% / 0.02111 / 0.50 / +0.000304 | 94.3946% / 0.01462 / 0.50 / -0.000012 | 91.2875% / 0.01942 / 0.50 / -0.000316 |
| enc2 output | 96.2217% / 0.00888 / 0.25 / +0.000006 | 96.9929% / 0.00954 / 0.50 / -0.000073 | 96.8904% / 0.00972 / 0.50 / -0.000079 |
| enc2 merge | 90.0806% / 0.01566 / 0.50 / +0.000166 | 92.2784% / 0.01498 / 0.50 / -0.000111 | 91.1390% / 0.01616 / 0.50 / -0.000277 |
| tube#1 | 97.5460% / 0.00879 / 1.00 / +0.000496 | 98.1885% / 0.00615 / 0.50 / +0.000027 | 96.3765% / 0.01028 / 1.00 / -0.000469 |
| tube#2 | 95.9546% / 0.01177 / 1.00 / -0.000601 | 97.1611% / 0.00949 / 1.00 / -0.000355 | 97.1804% / 0.01001 / 1.00 / +0.000246 |
| tube#3 | 98.7963% / 0.00410 / 0.50 / +0.000165 | 98.5659% / 0.00642 / 0.50 / -0.000044 | 97.7421% / 0.00731 / 0.50 / -0.000209 |
| tube#4 | 96.7105% / 0.00944 / 1.00 / +0.000085 | 97.6565% / 0.00737 / 1.00 / +0.000126 | 97.2874% / 0.00847 / 1.00 / +0.000041 |
| tube#5 | 95.0695% / 0.01182 / 1.00 / +0.000477 | 95.7411% / 0.01121 / 1.00 / +0.000349 | 96.6670% / 0.00972 / 1.00 / -0.000129 |
| tube#6 | 95.7403% / 0.01556 / 0.75 / -0.000069 | 95.1269% / 0.01682 / 0.59 / -0.000173 | 94.5373% / 0.01844 / 0.75 / -0.000104 |
| dec2 | 94.6417% / 0.01565 / 1.00 / -0.000034 | 94.5945% / 0.01588 / 1.00 / -0.000318 | 94.8529% / 0.01565 / 1.00 / -0.000284 |
| dec1 | 96.0181% / 0.01399 / 1.06 / +0.000063 | 95.5735% / 0.01509 / 1.06 / +0.000090 | 95.3655% / 0.01547 / 1.00 / +0.000027 |

[M] In every local comparison there are zero non-finite code pairs, zero saturated
codes (`0x7e`/`0xfe`), and the signed bias stays below `6.1e-4` in all sixteen fields;
Sette, fast and conservative are deterministic across repeats. Sette is not
distinguishable from d4r's own native replacement by these measures: it is closer on
tube#3, tube#6, dec2 and dec1 and farther on the rest, always inside the same band.

[M] Sette versus translated, distribution of the differing codes (natural channel
order; field layouts decoded, not compared as raw offsets):

| Layer | Field | worst channel (id, exact) | channels < 99% | token median / min | worst token (y,x) | worst 8x8 sub-position |
|---|---|---:|---:|---|---|---:|
| enc2 | output | 6, 92.98% | 94 | 100% / 39.58% | (1, 238) | 78.16% |
| enc2 | merge | 29, 74.42% | 122 | 96.88% / 63.28% | (7, 7) | 64.84% |
| tube#1 | output | 17, 91.21% | 105 | 100% / 53.91% | (8, 7) | 60.32% |
| tube#2 | output | 96, 89.62% | 118 | 100% / 59.38% | (6, 4) | 61.04% |
| tube#3 | output | 43, 95.75% | 65 | 100% / 55.47% | (2, 4) | 76.39% |
| tube#4 | output | 106, 88.28% | 101 | 100% / 52.34% | (63, 3) | 74.56% |
| tube#5 | output | 9, 87.78% | 119 | 100% / 49.22% | (6, 117) | 75.05% |
| tube#6 | output | 44, 91.52% | 128 | 100% / 26.56% | (61, 113) | 62.71% |
| dec2 | output | 5, 91.00% | 96 | 100% / 25.00% | (38, 12) | 80.44% |
| dec1 | output | 50, 92.92% | 64 | 100% / 20.31% | (226, 55) | 90.58% |

[H] The residual is neither a single-channel defect nor a global offset: every channel
contributes, at least half of the tokens are fully exact, and the differing codes
cluster in a minority of tokens whose worst 8x8 sub-position still matches at 60-91%.
Some worst tokens lie beside a padded/shifted edge (enc2 x=238 of 240 with the grid
padded to 248; tube#5 x=117 of 120 with the grid padded to 128), which is consistent
with a boundary operand effect, but that clustering is an observation here, not an
established cause.

Translation output-field SHA-256 (written regions only):
enc1 `1d99c4d685458fe50e28b243b5058409ee0a458895e8edeb5e84b2382d29f728`,
enc2 `9ed07168859727dd2d63413a0ff616f63d05e32a17be0c0fae61c4fe64a79158`,
tube#1 `0419846eede7b3f1`, tube#2 `389cc88b5d9027af`, tube#3 `df7bda3f9a92ae0e`,
tube#4 `fdebea21d28996fc`, tube#5 `383ff8de36cdc98a`, tube#6 `4b922f22d1ce0a69`
(16-hex prefixes of the full digest in `summary.json`),
dec2 `b583cdcf9eda65ab`, dec1 `38782c8dac8eac4b`.

### 3. Unconfounded native-versus-translated controls

[M] For enc1, enc2, tube#1, dec2 and dec1 the translated capture consumed exactly the
canonical capture's input and weight bytes (all other modules stayed native in those
runs). With identical operands, the original PTX translated by ZLUDA and d4r's own
native replacement still differ:

| Layer | canonical native vs translated | translated-vs-native differs |
|---|---|---|
| enc1 | 98.4912% output / 94.3946% merge | 126,075 / 175,642 codes |
| enc2 | 96.9929% / 92.2784% | 94,226 / 80,651 codes |
| tube#1 | 98.1885% | 18,921 codes |
| dec2 | 94.5945% | 169,378 codes |
| dec1 | 95.5731% | 369,903 codes |

[M] The historical enc2, tube#1 and dec2 native output fields equal a fresh pinned
conservative build of the same kernel byte for byte (0 differing codes), so those
launches are confirmed conservative. enc1 was already identical this way. dec1's
historical field differs from the fresh build in 33 of 8,355,840 codes, so it is
conservative-equivalent but not byte-identical; that 33-code gap is the only
difference between the "canonical" and "conservative" columns for dec1 (95.5731%
versus 95.5735%).

### 4. Mode B: the complete translated chain

[M] A second capture removes all five network kernels from the native allowlist, so
all ten neural launches dispatch `backend=translated` (enc1, enc2, six tube shifts,
dec2, dec1), with the dec1 launch dumped. The pipeline is otherwise the real one:
skip tensors, tube ping-pong, alternating shifts and padded geometry are the runtime's
own resources, no intermediate is substituted.

| Chain comparison (dec1 field, 8,355,840 codes) | exact | relative RMS | max | bias |
|---|---:|---:|---:|---:|
| translated chain vs canonical capture | 21.7096% | 0.12270 | 2.50 | -0.000125 |
| Sette chain vs translated chain | 21.5728% | 0.12312 | 2.50 | +0.000281 |
| Sette chain vs canonical capture | 21.3270% | 0.12555 | 3.25 | -0.000407 |

[M] Chained agreement per boundary, all three implementations started from the same
enc1 input bytes (translated chain dec1 SHA-256
`dac25760612b5fe3bb59526b2a5b0d0fcd56d0a0eb1c94f29b2130e7cf427a4b`, canonical
`5c22225b0039d80aef53c655d4965c111f65c3729362060a1bdf44e5b2522a78`):

| Boundary | canonical vs translated | canonical vs Sette | translated vs Sette |
|---|---:|---:|---:|
| enc1 | 98.4912% | 96.9243% | 96.4183% |
| enc2 | 44.6289% | 41.1305% | 40.3931% |
| tube#1 | 36.6020% | 34.7653% | 35.2237% |
| tube#2 | 28.4725% | 28.4242% | 28.9720% |
| tube#3 | 25.9093% | 25.2088% | 24.9759% |
| tube#4 | 25.7634% | 25.6434% | 26.1515% |
| tube#5 | 26.6632% | 26.7856% | 26.9205% |
| tube#6 | 13.9660% | 13.7847% | 14.0543% |
| dec2 | 19.1203% | 18.5347% | 18.4737% |
| dec1 | 21.7096% | 21.3270% | 21.5728% |

[M] No chained boundary in any pairing produced a non-finite or saturated code, and
the largest chained decoded bias is `9.4e-3` (dec2, canonical vs translated); the
divergence is spread rather than carried by one sign. Chained agreement therefore
degrades by code-level disagreement, not by systematic saturation or offset.

### 5. Classification

[M] The local residual is a boundary-requantization residual: ~1-5% of e4m3 codes,
RMS 0.4-1.7%, no bias, no saturation, no non-finite values, maximum decoded deviation
of about one or two quantization steps. It appears between d4r's own native kernel and
the original PTX at the same magnitude (section 3), so it is **not** attributable to
Sette's token/channel mapping, norms or weight operands, which the earlier traces
already proved identical.

[M] Chained agreement collapses in every pairing, including the two original
implementations. enc2 already drops to ~40% because the enc1 merge (89.8% Sette vs
translated) is re-quantized; by tube#6 all pairings sit near 14% and dec1 near 21%.
The chained figure therefore measures sensitivity of the e4m3 requantization chain to
individual code changes, not a defect that is specific to Sette.

Classification of the observed differences: no reconstruction defect is demonstrated
by these measurements; the enc1 K32 probe classified its single operation as permitted
rounding (see the preceding addendum); ZLUDA lowering and AMD instruction-level
differences remain possible contributors at operator scope and cannot be separated
from each other or from NVIDIA-hardware rounding without NVIDIA hardware evidence;
exact internal WMMA rounding stays unresolved.

Limits of this classification: an operator-level divergence was localized only for
enc1 (the first Q/V K32 step). For the other nine layers this addendum establishes
scope and magnitude, not the first divergent instruction; doing that would repeat the
K32 probe per layer. Because the native-versus-translated gap is the same size as the
Sette-versus-translated gap, that work would at most identify which operator first
crosses a code boundary, not establish which implementation is correct. The
quantization-boundary flips themselves cannot be attributed to Sette without NVIDIA
hardware evidence, and no such evidence exists locally.

### 6. Gate implications (no threshold changed)

- Legacy conservative-d4r gate: chained dec1 21.3270% (FAIL, as before).
- Translated-PTX layer fidelity: 94.64-98.80% exact codes for Sette, 94.59-98.57% for
  conservative d4r; every layer is deterministic and within one or two code steps.
- Translated-PTX chain fidelity: 21.5728%, and the *translated reference itself*
  reaches only 21.7096% against the canonical capture. A 99% chain floor is therefore
  not a translated-reference fidelity criterion; it cannot be met by the original PTX
  either, because the chain re-quantizes ten times and the two original paths already
  disagree at layer scope.
- The 99% floor, the legacy fixtures and the fast/conservative separation are
  unchanged. Replacing or retiring the chain gate, or introducing a translated-reference
  acceptance criterion, needs an explicit decision and its own validation; it was not
  done here. B5 stays IN PROGRESS and `m_conformance.py --network` keeps failing.

### 7. Reproduction

Fresh, nonexistent local directories. Each capture writes proprietary PTX, weights and
tensors only under `%TEMP%`.

```powershell
$T = "$env:LOCALAPPDATA\Temp\opencode"
python scripts/m_capture_translated.py --kernel rrlite_enc2_4x4 --out "$T\enc2-translated-NEW"
python scripts/m_capture_translated.py --kernel rrlite_enc3_tube_4x4 --limit 6 --out "$T\tube-translated-NEW"
python scripts/m_capture_translated.py --kernel rrlite_dec2_4x4 --out "$T\dec2-translated-NEW"
python scripts/m_capture_translated.py --kernel rrlite_dec1_4x4 --out "$T\dec1-translated-NEW"
python scripts/m_translated_layer_suite.py --out "$T\translated-layers-NEW" --enc2 "$T\enc2-translated-NEW" --tube "$T\tube-translated-NEW" --dec2 "$T\dec2-translated-NEW" --dec1 "$T\dec1-translated-NEW"
$ex = @('--exclude','rrlite_enc1_4x4','--exclude','rrlite_enc2_4x4','--exclude','rrlite_enc3_tube_4x4','--exclude','rrlite_dec2_4x4','--exclude','rrlite_dec1_4x4')
python scripts/m_capture_translated.py --kernel rrlite_dec1_4x4 @ex --out "$T\chain-translated-NEW"
python scripts/m_enc1_contract.py --out "$T\enc1-contract-NEW"
python scripts/m_layer_suite.py chain "$T\m-capture-run\layers" --json "$T\chain-translated-NEW\sette-chain.json"
python -m pytest tests -q
python scripts/m_conformance.py --json "$T\chain-translated-NEW\b4.json"
```

Local experiment directories for this addendum: `enc1-translated-20261029`,
`enc2-translated-20261029` (+`-repeat`), `tube-translated-20261029-b`,
`dec2-translated-20261029`, `dec1-translated-20261029`,
`translated-layers-20261029-b` (`summary.json` holds every comparison and hash),
`chain-translated-*-20261029` (chained boundaries in `chained-boundaries.json`),
`enc1-contract-translated-20261029`. No proprietary payload, PTX, code object, weight
or capture is added to Git, and the pinned d4r submodule is unchanged.
