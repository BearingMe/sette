# EXP-028 — Cumulative native INT8 learned-weight coverage

**Milestone:** M09-G · **Status:** PASS for four bounded cumulative batches;
**full INT8 coverage remains incomplete**. M09 IN PROGRESS / PENDING.
**Foundation:** `6e4e6ea` commits M09-F; 321 tests passed before proceeding.
No push. Windows/D3D12, RX 7700 XT / gfx1101, SDK DXC, RGA 2.15.

## G1 — Complete matrix inventory and denominator

[S] The machine-readable [complete inventory](EXP-028-matrix-inventory.json)
records every logical matrix operator's layer/dispatch, M/K/N, batching,
operand/weight layout, original precision, FP16 implementation, INT8 kernel or
gap, required conversion and final status. Per-family contract records apply to
each operator row. It contains shapes/status metadata, **no proprietary tensors**.

Sources: the canonical ten `m_layer_fixture.py` graphs/parameter blocks and
`swin_layer_cs.hlsl::{Matmul,Attention,Mlp,Merge}`, `enc0_cs.hlsl::Column`,
`dec0_cs.hlsl::Column`; weight naturalization is the existing `WeightPack` /
`m_layer_oracle.natural_matrices` contract. No operator reverse engineering was
restarted. The initial FP16 inventory was generated before conversion.

**76 logical layer/family operators:** 56 learned-weight, 20 dynamic attention.
These are not the 360 physical weight matrices/tiles and not dispatch counts.
Q and V aggregate their learned head columns; fused MLP counts as up and down;
each attention dispatch contains both QKᵀ and PV, batched over windows/heads.
The implementation uses Q as K too; there is no omitted separate K projection.

Workload weights are **dense shape MACs**: `M*K*N*instances`, including padded
rows the source computes before cropping. They are not measured hardware
instruction counts; compiler zero folding/sparsity and nonmatrix work are not
modeled. One MAC is not two FLOPs. Total: **34,518,597,632 MACs**.
Only dec0's 80 actually consumed columns count, not all 128 physical columns.

| Family, all layers | Logical operators | Dense MACs |
|---|---:|---:|
| Q | 10 | 2,812,018,688 |
| V | 10 | 2,812,018,688 |
| Output projections | 10 | 2,812,018,688 |
| MLP up | 10 | 10,408,165,376 |
| MLP down | 10 | 10,408,165,376 |
| Encoder merges | 2 | 1,278,345,216 |
| Decoder expansions | 2 | 1,203,240,960 |
| dec0 expansion | 1 | 668,467,200 |
| enc0 embedding | 1 | 1,069,547,520 |
| Dynamic QKᵀ | 10 | 523,304,960 |
| Dynamic PV | 10 | 523,304,960 |

The inventory also explicitly excludes **non-neural-GEMM** arithmetic: post's
data-dependent 3×3 Gaussian filtering/covariance quadratic forms and
downsample's 5×5 reconstruction stencil. They remain justified floating-point
exceptions, not silently omitted neural matrix operators. This is the current
reset-frame graph, not an inventory of unimplemented temporal branches.

## G2/G3 — Bounded generalization and actual batches

[S][M] `scripts/m09_cumulative.py` transforms the established fixed ten-layer
fixtures by family. Existing matrix kernels were already parameterized by
M/K/N: reuse the original GPU row quantizer, `swin_mlp_up_q8.cso` for zero/bias
GEMMs, and `swin_mlp_down_q8.cso` for residual GEMMs. No new matrix kernel,
executor, scheduler or generic inference framework.

The symmetric signed INT8 scheme is unchanged from EXP-026/027: correctly
rounded f32 activation inverse/product, half-even codes clamped ±127,
per-row scales; offline per-output-column weight scales; exact whole-K INT32;
two f32 rescale multiplies; f32 bias/residual add; one explicit f16 RNE store.
All K are multiples of four, all N multiples of 16, largest K=512, and
`512*127²=8,258,048 < 2²⁴ < 2³¹`. This is not the old FP16 k32-rounding model.

Only three **layout adapter** entry points were missing:

* Decoder low field: GPU plane-major E4M3 → natural row-major f16.
  Byte input `(k//32)*M*32+row*32+k%32`; output half `row*K+k`.
* Merge gather: Q8 of each original 2×2 padded token patch, natural row-major
  `[M,4C]`, q order top-left/right then bottom-left/right.
* Merge store: existing clamp/half/E4M3 encoding, valid-region crop and
  plane-major output mapping, reading the INT8 GEMM's f16 result.

They call the existing layer helpers. GELU/E4M3, attention, normalization,
residual producers, patch preparation, image surfaces and downstream formats
stay explicit. The new unpack stage and its diagnostics are correctly rewired
to each decoder's **live native producer**, not its captured low input.

Original FP16 graphs/shaders remain available as the fallback for every family;
the default and independent Q/MLP/dec0 modes are unchanged. Cumulative modes
are explicit and separate. An audit requires the exact selected operator count
and INT8 matrix shader identities; missing families or FP16 matrix substitutions
fail rather than silently decreasing coverage.

| Accepted cumulative configuration | INT8 ops / 76 | Op coverage | INT8 MACs | MAC coverage | Neural dispatches |
|---|---:|---:|---:|---:|---:|
| All Q/V | 20 | 26.3158% | 5,624,037,376 | 16.2928% | 114 |
| + all output projections | 30 | 39.4737% | 8,436,056,064 | 24.4392% | 124 |
| + all MLP pairs | 50 | 65.7895% | 29,252,386,816 | 84.7438% | 164 |
| + merges, decoder expansions, dec0 | **55** | **72.3684%** | **32,402,440,192** | **93.8695%** | **174** |

The last configuration converts **55/56 learned-weight operators**, but not
enc0 and not dynamic attention. Dispatch counts include adapters/nonmatrix work
and are deliberately not used as coverage percentages.

## Independent validation, resource safety and provenance

[M] Each batch first executes the **complete GPU neural chain** with diagnostic
dumps. All selected operators are checked against independent CPU INT64 dots,
f32 conversion/epilogue and f16 outputs using the actual GPU-produced operands,
not captured substitutes. Every weight column is also independently scalar
encoded offline; each operator has eight scalar row/dot/output probes.

The final neural batch compares **264,440,320** accumulator, f32 and f16 output
elements bit for bit, plus all activation codes/scales and weight encodings.
The live-frame dec0 prefix additionally matches all **10,444,800** integer/f32/f16
elements and activation codes/scales to its independent CPU oracle. All agree.

CPU references independently check every GELU output, decoder unpack, merge
gather and merge field epilogue. Re-execution with different sentinel seeds
reproduces every diagnostic byte, proving complete writes under that test.
All diagnostic runs report zero D3D12 debug messages.

Only after those gates pass does B6 execute a fresh two-run frame. Its resources,
bindings, parameters and shader identities must match the validated chain;
the sole input filename difference is required to contain byte-identical native
enc0 output. Every exported layer field matches the checked chain. Repeated
frame boundaries: 22 for the first three configurations, **27** with dec0's
five internal diagnostics. No CPU activations are fed between neural operators.
B6's existing file boundaries between major neural/image stages remain unchanged.

Bulk diagnostic files are deliberately disposable: after CPU checks/digests,
they are pruned before the next seed/run to bound disk use. Reports retain
hashes, graphs, weight fixtures, debug logs, metrics and raw reconstructed frames.
Small field-only graph reruns reproduce the recorded CPU-checked field hashes
and measure every layer against the FP16 fields (`fields.json`). No proprietary
capture, weight, tensor, PNG or generated shader is added to Git.

### Errors against the same-input FP16 algorithm

These are CPU k32 FP16-oracle comparisons on the **same cumulative operands**,
not a claim that the whole hybrid agrees with the original FP16 network.

| Final batch family | Relative RMS range across layers |
|---|---:|
| Q | 1.0047–1.4956% |
| V | 0.9218–1.7504% |
| Output | 0.4013–0.9954% |
| MLP up | 0.9439–1.0910% |
| MLP down + residual | 0.6074–2.8185% |
| Merge | 0.9738–1.0647% |
| Decoder expand | 0.6678–0.9174% |

Chained dec1 decoded relative RMS **against the native FP16 chain**, by batch:
15.4697%, 15.7815%, 16.3726%, **16.2352%**. Every measured layer field is finite
and has zero E4M3 saturation. Operator f16 outputs and final frames are also
finite and unsaturated. INT8 scale endpoints are recorded separately from
floating-point saturation.

### Native ISA / executed hashes

[M] Every batch inspects the exact executed generic matrix binaries through
RGA DX12 gfx1101. Each has **16 static `v_dot4_i32_i8`** instructions and zero
FP16 matrix mul/FMA/MAC/WMMA instructions. Dynamic K loops mean these are not
dynamic instruction counts. Stage/frame/current-file hashes are checked at
execution; no runtime instruction trace or performance claim is made.
The final dec0 shader is separately reinspected against its executed frame/CPU
hashes: 12 static signed dot4, zero FP16 matrix instructions (`dec0-isa.json`).

| Binary | SHA-256 |
|---|---|
| Shared quantizer | `316fcf87d5ae37885730849f741451f469842ca40360eab6aa23187f24e25c52` |
| Zero/bias INT8 GEMM | `f5324bcbd9ac70e91b1cb813d2ead573445483ced6a3cf45e03df495f670e217` |
| Residual INT8 GEMM | `115764abfb6531673d3db2f21fff4eca005b13b6cec76e273a81fb87014d7f75` |
| Existing dec0 INT8 GEMM | `c7fc7fa6a3aa67c540edaea7a76d0d2c3f5d1646c3b581609e8efa062e405a89` |
| Final decoder unpack | `c177764ca474b5ce83bc470f4069caae7c853b7f4785489dff1389eb7e333423` |
| Merge gather | `336229b1e84071ac87514a985487482f06fa2044fa3413462db1dcbbd9071fbe` |
| Merge field store | `f099a52d3f49811f526e722bad5327053c3c1b36653a6f9cb8c8d7e3fdbccd92` |
| Final network host | `1e5f47adadb24a379510f5ae3e1788c2b4989e18bebbb8ced46a9d2dd165b3f2` |

Existing shader binaries remain unchanged. Q/V and output batches used the
pre-write-guard host; later batches added a host-only incomplete-write check.
Field-only reruns with the final host reproduce all earlier accepted field hashes.
Each batch retains its own actually executed host/shader provenance, rather than
relabeling earlier runs with a later executable hash.

## G5 — Cumulative frames and diagnostic quality

[M] Existing linear RGB raw-f16 PSNR (peak 1), 11×11 windowed luma SSIM;
alpha unchanged. PNGs use the existing display mapping, not metric inputs.

| Configuration vs canonical | PSNR dB | SSIM | RMS | Mean abs | P99 | Max abs | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 control | 47.051389 | 0.999258964 | 0.00444049 | 0.00108616 | 0.0229492 | 0.178223 | +0.00008814 |
| Q/V | 44.444061 | 0.998770718 | 0.00599511 | 0.00136946 | 0.0307617 | 0.198242 | +0.00021198 |
| + output | 44.828645 | 0.998844702 | 0.00573545 | 0.00132424 | 0.0292969 | 0.214844 | +0.00012696 |
| + MLP pairs | 44.214184 | 0.998626027 | 0.00615589 | 0.00142322 | 0.0322266 | 0.217773 | +0.00006295 |
| **+ spatial/dec0** | **44.377854** | **0.998745519** | **0.00604098** | **0.00139357** | **0.0317383** | **0.252441** | **+0.00007109** |

| Direct against FP16 | PSNR dB | SSIM | RMS | Mean abs | P99 | Max abs | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q/V | 44.723225 | 0.998783085 | 0.00580549 | 0.00136207 | 0.0307617 | 0.173828 | +0.00012384 |
| + output | 45.010825 | 0.998851376 | 0.00561641 | 0.00130875 | 0.0290527 | 0.144043 | +0.00003882 |
| + MLP pairs | 44.506578 | 0.998676871 | 0.00595211 | 0.00139242 | 0.0322266 | 0.145020 | -0.00002519 |
| **+ spatial/dec0** | **44.923517** | **0.998827520** | **0.00567315** | **0.00133444** | **0.0300293** | **0.152832** | **-0.00001705** |

Direct previous→current configuration comparisons:

| Change | PSNR dB | SSIM | RMS | Mean abs | P99 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| Q/V → +output | 45.281097 | 0.998928803 | 0.00544434 | 0.00125845 | 0.0280762 | 0.123779 |
| +output → +MLPs | 45.211256 | 0.998914465 | 0.00548829 | 0.00127890 | 0.0280762 | 0.161377 |
| +MLPs → +spatial | 46.228121 | 0.999127021 | 0.00488196 | 0.00117088 | 0.0253906 | 0.134766 |

All canonical, FP16 and previous-configuration comparisons retain full spatial
statistics and reconstructed/difference PNGs. Final direct-vs-FP16 weighted
spatial error: 32-pixel border/interior means 0.000940157 / 0.001057044;
16×16 block means 0.000188312–0.003176903; worst (y,x) blocks
(63,12), (12,7), (4,51), (4,35), (4,43). Weighted spatial statistics differ
from raw RGB absolute-error means. No subjective visual-equivalence claim.

**Quality gates were not relaxed:** all four cumulative frames fail the existing
B6 native PSNR regression floor of 45 dB; their B6 reports remain FAIL for that
reason. Execution, arithmetic, safety and determinism pass. M09-G accepts these
bounded execution results because quality is explicitly diagnostic here, not
because the B6 regression failure was hidden. FP16 B6 still passes unchanged.

Final frame SHA-256:
`ecb78e43c2af36f88d0f71d59d9828c3af746a7ffcc1f380648efe67b1e24fcb`.

## G4 — Explicit remaining blockers

* **enc0 embedding: INT8 feasible but not implemented.** Its proven reset-frame
  contract is known. The smallest layout gap is a GPU texture/dither feature
  adapter and image-executor/field-epilogue handoff; current GEMMs consume buffers,
  not that texture-derived vector. It remains FP16 and contributes 3.0985% of
  dense MACs. This is not a permanent FP16 exception.
* **QKᵀ/PV: unsupported or unresolved INT8 contracts**, ten of each, jointly
  3.0320% of dense MACs. The RHS is dynamic per window/head. Offline weight
  quantization is invalid here. Missing pieces are runtime RHS layout/scales,
  QK score-bias precision and PV probability/value encoding/rescaling, while
  preserving the original softmax boundaries. All remain explicit FP16 matrices.
* Normalization, GELU, residuals, softmax/nonlinear epilogues and image filtering
  remain justified floating-point operations; they are not claimed as converted.

Thus **55/76 operators, not full INT8 coverage**. M09's full gate remains open.
Next implementation priority is enc0's bounded adapter to close learned-weight
coverage, then the smallest GPU dynamic-RHS/scale contract and integration for
QKᵀ/PV in this **same cumulative path**, not a new detached experiment track.

## Regressions, failures and retention

[M] **339 tests pass**, retaining all 321. Native build and whitespace checks
pass. Final FP16 frame is byte-identical (`dc97c4cb…`), 47.051389 dB. B4 remains
PASS, 99.9369%, max 7 ULP. B5 retains only enc1.merge, 89.8379% / relative RMS
0.02111 (expected conformance exit 1). A Sette-only D3D12 refresh reproduces all
22 archived independent/chained fields after the host write guard was added;
no HIP, reference rebuild, rounding investigation or acceptance change.

Two meaningful failures are retained:

1. Initial MLP diagnostics exhausted disk; its last residual dump was empty.
   No MLP batch/frame was accepted. Added explicit write/flush completion checks
   in the network runner and diagnostic byte-size checks. Removed redundant
   completed-control copies, preserving raw frames/reports and authoritative
   stage evidence; a clean rerun passes.
2. Initial spatial unpack differed in **241 negative-zero sign bits**, with
   zero numerical differences and exact INT8 expansion arithmetic. The layout
   gate stopped advancement before any spatial frame. DXC's floating E4M3
   helper canonicalized zero; the adapter now propagates its source sign in
   integer half bits. An independent nonuniform GPU test including 0x80 and
   the complete rerun both pass bit for bit. No rounding investigation or
   quantization tolerance relaxation was used.

Failed bulk tensor bytes are pruned after recording diagnoses; their JSON,
graphs, fixture metadata and logs remain. Explicitly named redundant M09-F
control-frame copies were also pruned to release space; their raw final frames
and reports remain. Original captures and verified operator-stage evidence were
not deleted. This storage policy is not a performance optimization claim.

## Reproduction and evidence

Use fresh output directories; T is `%LOCALAPPDATA%\Temp\opencode`, L the canonical
`m-capture-run/layers`, RGA the established executable. Enable the debug layer:

```powershell
cmake --build build/native --config Release
python -m pytest tests -q
$env:M_STAGE_DEBUG = '1'
python scripts/m_b6_frame.py --out "$T/fp16"
$B = "$T/fp16/downsample-native/results/out-out.bin"
python scripts/m09_cumulative.py --layers $L --out "$T/inventory"
python scripts/m09_cumulative.py --layers $L --out "$T/qv" --batch qv --baseline $B --rga $RGA --prune-bulk
python scripts/m09_cumulative.py --layers $L --out "$T/output" --batch output --baseline $B --rga $RGA `
    --previous "$T/qv/frame/downsample-native/results/out-out.bin" --prune-bulk
python scripts/m09_cumulative.py --layers $L --out "$T/mlp" --batch mlp --baseline $B --rga $RGA `
    --previous "$T/output/frame/downsample-native/results/out-out.bin" --prune-bulk
python scripts/m09_cumulative.py --layers $L --out "$T/spatial" --batch spatial --baseline $B --rga $RGA `
    --previous "$T/mlp/frame/downsample-native/results/out-out.bin" --prune-bulk
```

Bulk diagnostic dumps require several GiB free; keep a safety margin. Final
local roots are `m09g-{qv,output,mlp-resumed,spatial-verified}-20261030`:

* `cpu.json`: full per-operator/adapter checks, scalar probes, metrics and hashes.
* `batch.json`: executed coverage, graph/frame handoffs, ISA, image-regression
  failures and canonical/direct metrics; `fields.json`: hash-tied layer errors.
* `frame/b6.json`, logs, raw final frame, and PNG prefixes `frame-native`,
  `hybrid-vs-fp16`, `hybrid-vs-previous`.
* `layers/` and `validated-chain/`: replayable graph/weight fixtures; bulk raw
  diagnostic tensors deliberately not retained after checks.

Other evidence: `m09g-inventory-20261030/inventory.json`,
`m09g-unpack-sign-20261030`, `m09g-fp16-final-20261030`,
`m09g-translated-refresh-20261030`, `m09g-conformance-20261030.json`.
The final post-guard full-frame rerun is
`m09g-spatial-frame-final-20261030`: all 27 boundaries and every layer handoff
match the CPU-checked batch. Its current script/host/shader hashes and execution
are tied by nine PASS checks in `m09g-spatial-verified-20261030/final-frame.json`.
The separate `dec0-isa.json` ties dec0's native lowering to its executed hashes.
Failure roots: `m09g-mlp-20261030/failure.json` and
`m09g-spatial-20261030/{cpu,layout-failure}.json`.

One reset frame/content/resolution/adapter; general static fidelity remains
INCONCLUSIVE and temporal fidelity untested. No calibration, performance
optimization, new backend or generalized framework. The next goal remains
**complete native INT8 matrix coverage**, not prettier images or peak speed.
