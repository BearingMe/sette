# EXP-026 — One native INT8 MLP projection inside enc1

**Milestone:** M09-E · **Status:** PASS for E1–E6; M09 remains IN PROGRESS.
**Baseline:** `f6d7c31`, 303 tests, clean working tree. M09-D was already committed
and pushed before this request; no pending foundation changes required a commit.
**Environment:** Windows/D3D12, RX 7700 XT / gfx1101, SDK DXC, RGA 2.15.
No references, proprietary data, generated shaders, or acceptance thresholds changed.

## Question and gates

Replace exactly enc1's **complete up projection W1+b1**, not the down projection
or entire MLP. Require a physical-layout contract, independent scalar integer
checks, all-element CPU/GPU agreement, preserved nonlinear/down/residual behavior,
native signed dot4 ISA tied to executed shader hashes, and a deterministic real
network/frame. Image error is diagnostic, not an arithmetic correctness gate.

## E2 — Source-grounded operator contract

[S] Primary implementation: `src/native/swin_layer_cs.hlsl`, `Norm`, `Mlp`,
`GeluQ`, `Store`, `Merge`; `scripts/m_layer_fixture.py`, `WeightPack` and
`params_block`; `scripts/m_layer_oracle.py`, `natural_matrices` and `NaturalBlock`.
Independent pinned source: d4r `05e19d402781083118d106f2a21d0cf2bc43f744`,
`kernels/tools/swin_model.py`, `gp`, `woff_table`, `layout`, `swin_block`.
This extends the established FP16 reconstruction contract; it does not claim
exact original NVIDIA accumulation behavior or reopen B5 rounding.

| Boundary | Contract |
|---|---|
| Geometry | enc1 C=64, 2 heads; 480×272 valid field; shift (2,2); padded 488×280 = **136640 rows** |
| Input | `h2`, norm2's GPU output, f16 `[M,64]`, each value E4M3-representable |
| Up projection | **M=136640, K=64, N=256**, eight 32-column chunks, one logical W1 projection |
| Bias | 256 original f16 values widened losslessly to f32, one per output column |
| Up output | f16 `[M,256]`, **before GELU**, no E4M3 conversion at this boundary |
| GELU | Original f16-rounded cubic `GeluQ`, followed by original E4M3 round-trip |
| Down projection | `[M,256] × [256,64]`, still the original eight sequential k32 FP16 steps |
| Residual | `x2=Add16(x1,b2)` from norm2; each down chunk adds to this f16 accumulator, with f16 RNE after that chunk; no extra down bias |
| External outputs | Original `x2b` f16 resource; Store and Merge keep their original E4M3 layouts, clamping and sizes |

### Physical addressing

Let `s=n//32`, `j=n%32`, and
`gp(j)=32*(j//32)+2*((j%32)//4)+(j%2)+16*((j%4)//2)`.
All offsets below are **bytes**, except explicitly marked f32 element indices.

* Activation: `h2[row,k]` is at `2*(64*row+k)`; row maps to padded
  `(py,px)=(row//488,row%488)`.
* Captured W1, natural output j: put `p=gp(j)`, then
  `B[s]+512*(k//32)+1024*(p//16)+64*(p%8)+16*((k%16)//4)`
  `+8*((p//8)%2)+4*((k%32)//16)+(k%4)`.
* Captured up bias: `B[s]+2048+2*gp(j)`.
* Raw chunk bases B: **13696, 17984, 22144, 26304, 30464, 34624, 38784, 42944**.
  The extra first-chunk stride accommodates the residual-bias vector.
* Existing exported W1 f32 element index:
  `pOffW1+s*2048+k*32+j`; b1: `pOffB1+32*s+j`.
  Concatenating these chunks across N yields natural `[64,256]`.
* Original exported W2 element index:
  `pOffW2+s*2048+j*64+c`; the raw W2 base is `B[s]+2048+64`, using
  `woff_table(base,0,512,32,64)` and the natural output permutation gp(c).
* INT8 activation code byte: `64*row+k`; weight code byte: `64*n+k`.
  Both pack four signed bytes in little-endian R32_UINT words. Scale byte offsets
  are `4*row` and `4*n`; the bias byte offset is `4*n`.
* New f16 up boundary: `2*(256*row+n)`. Existing final enc1 output plane byte:
  `(((c//32)*272+y)*480+x)*32+c%32`.

[M] All 16,384 W1 elements and 256 biases match independent scalar addressing of
the real capture, not just a shape/size check. Each chunk's 2,048 offsets is
unique. Nonuniform synthetic address identities match the pinned kernel table.
The capture is `m-capture-run/capture-c10/replay-000001-rrlite_enc1_4x4`;
`h2` itself is a native intermediate reconstructed from that capture's inputs,
**not** a separately captured internal tensor.

### Precision that remains unchanged

The original up path starts each chunk with b1 and performs two exact k32 sums,
each followed by one f16 RNE. Its GELU computes, with the existing f16 rounding:
`cz=clamp(z,-2,2)`, `inner=0.412109375-0.0810546875*abs(cz)`,
`g=Q8(z*(0.5+cz*inner))`. Down accumulation remains in its original chunk order.
Store encodes E4M3; Merge re-quantizes its own operands and clamps to ±2π.
These are explicit floating-point exceptions, not additional INT8 conversions.

## E3/E4 — Independent arithmetic and native GPU implementation

[S][M] Reuse EXP-023's GPU f16-row quantizer unchanged:
correctly rounded f32 `inv=127/max_abs`, f32 product, ties-to-even `Round`, clamp
±127; scale `f32(max_abs*f32(1/127))`. Zero rows substitute max_abs=1.
Weights reuse `m_int8.quantize(W,'per_output_channel')`: f32 `max_abs/127`
scale, f64 ratio against that stored scale, ties-to-even clamp ±127; zero
columns have scale 1. No calibration, clipping, or alternative scale scheme.

The INT8 up kernel uses the established packed `dot4add_i8packed` reduction:

```text
acc[row,n] = sum_k int32(codeA[row,k])*int32(codeW[k,n])
y = f32(f32(f32(acc)*scaleA[row])*scaleW[n])
y = f32(y+bias[n])
up[row,n] = f16_RNE(y)
```

This intentionally replaces the original up projection's two f16 k32 rounding
points with one exact whole-K integer reduction and the established f32 rescale,
bias and f16 store. It does not approximate the original accumulation as exact.
The bound is `64*127²=1,032,256 < 2³¹`; integer→f32 conversion is exact.

`scripts/m09_mlp_up.py` independently checks all activation codes/scales,
weight codes/scales, **34,979,840 accumulators**, f32 values and f16 outputs.
All are **bit-identical**. The largest measured accumulator is 80,927.
All 256 weight columns also pass a separate scalar encoder; 48 scalar row/dot
and output probes agree. CPU tests cover ties, signed values, zero rows/columns,
endpoints and accumulator bounds. A nonuniform 8-row GPU fixture covers those
same cases; every diagnostic agrees with its CPU oracle under two sentinel seeds.

Activation INT8 endpoints are expected scale extrema, not f16 saturation:
8,048 codes at +127, 136,404 at -127, 86,500 zero codes; all 8,744,960 codes stay
in range. Row scales span 0.0157480314–0.0216535442, with no zero scales.

### Integration, unchanged default and handoff verification

[S][M] `m09_mlp_up.prepare` clones an established FP16 enc1 fixture; it quantizes
only offline weights, never opens the activation file. Experimental mode:

```text
norm2 → existing GPU row quantizer → INT8 W1+b1 → original GELU/W2/residual
      → original Store/Merge → remaining nine original FP16 layers
```

The epilogue wrapper defines `MLP_LOAD_UP` to read B3/t3. With the macro absent,
the default shader is unchanged; **every pre-existing shader hash remains
identical**, including `swin_mlp.cso`. The original executor handles resources,
SRVs/UAVs, transition barriers and lifetime; no new scheduler/graph language.
Full chain: 96 dispatches / 154 resources versus FP16's 94 / 146.

* Quantizer groups: 2135×1×1; INT8 up groups: 34160×1×1; original MLP remainder:
  2135×1×1. No dispatch exceeds the existing dimension limit.
* Codes/scales: 8,744,960 / 546,560 B; up f16: 69,959,680 B.
  Integer and f32 diagnostics: 139,919,360 B each, retained for validation.
* Activation encoding, dot product, GELU and down projection remain GPU-resident
  in the same real neural graph. Readback happens after consumers execute.
* `--mlp-mode int8-up --enc1-mlp DIR` is explicit and experimental. FP16 is still
  the default. Combining it with enc1-Q or dec0 INT8 is rejected.

A validation-only FP16 projection probe agrees with the independent CPU k32
chain for every element of every chunk. Feeding that FP16 boundary through the
same wrapper reproduces **x2b, enc1 out and enc1 merge byte-for-byte** versus
the original fused MLP. This checks the split without INT8 quantization error.
All eight diagnostic resources and both enc1 surfaces used by the final frame
match the CPU-checked stage run. Every non-MLP stage's parameters, shader and
bindings match the FP16 chain.

### Failures preserved and resource safety

The first stage run had CPU/GPU agreement but failed split equivalence: t3 was
declared before t0/t1/t2, while this bounded runner creates descriptors in
declaration order. A subsequent fixture correction initially added the register
to the validation probe instead of the remainder stage and failed PSO creation.
Both failures are retained under the pre-verified temporary roots; **no frame
was attempted until split equivalence passed**. The corrected generator and
tests enforce declaration order. The runner now rejects out-of-order register
declarations instead of silently misbinding them.

Debug message collection exposed existing COPY_DEST buffer-creation warnings.
Initialization now explicitly performs COMMON→COPY_DEST→COMMON. Final stage,
synthetic and full hybrid network runs report **zero D3D12 debug messages**.
Debug mode requires the layer, and ERROR/CORRUPTION aborts execution.
This uses the same information-queue validation as the image runner;
[Microsoft's SDK-layer interfaces](https://learn.microsoft.com/en-us/windows/win32/direct3d12/direct3d-12-sdklayers-interfaces)
were checked. No GPU-based instrumentation/bounds proof is claimed.

## E5 — Layer and chained-network effects

[M] Bias below is **hybrid minus FP16**. Field metrics decode E4M3 rather than
treating byte-code distance as numerical error.

| Boundary | Relative RMS vs FP16 | Max absolute error | Signed bias |
|---|---:|---:|---:|
| Up projection, pre-GELU | **1.00019%** | 0.14355469 | -0.000899925 |
| x2b before Store | 3.43056% | 0.34765625 | -0.002484371 |
| Final enc1 output | **4.64469%** | 0.5 | -0.002467315 |
| enc1 merge | **6.17562%** | 0.75 | +0.000646134 |
| Chained dec1 output | **13.50944%** | 3.0625 | -0.000271588 |

Separate existing FP16 reconstruction error from the induced change:

| Boundary vs canonical captured field | FP16 relative RMS | Hybrid relative RMS |
|---|---:|---:|
| enc1 output | 1.24000% | 4.69170% |
| enc1 merge | 1.94231% | 6.19297% |
| Chained dec1 | 12.55491% | 13.73727% |

All projection/x2b, enc1/merge, dec1 and frame outputs are finite and unsaturated.
Different sentinel seeds produce identical complete stage outputs. All 18 B6
boundaries repeat exactly; the audit additionally compares every exported neural
chain field and diagnostic between the two executions. The integration audit
passes **71 checks**; the stage validation passes **41 checks**.

## E6 — Full MLP-only hybrid frame

[M] Raw RGB f16, peak 1; existing 11×11 Gaussian windowed BT.709-luma SSIM.
Alpha is identical. Fresh double runs of every control preserve its prior hash.

| Against canonical reference | PSNR dB | Windowed SSIM |
|---|---:|---:|
| FP16 | 47.051389 | 0.999258964 |
| enc1-Q-only INT8 | 45.421826 | 0.998961386 |
| dec0-only INT8 | 47.024874 | 0.999255693 |
| **enc1-MLP-up-only INT8** | **45.920786** | **0.999078068** |

| MLP hybrid against | RMS | Mean abs | P99 abs | Max abs | Signed bias |
|---|---:|---:|---:|---:|---:|
| Canonical | 0.00505779 | 0.00118701 | 0.0253906 | 0.241699 | +0.00004117 |
| FP16 | 0.00534717 | 0.00123989 | 0.0270996 | 0.150879 | -0.00004697 |
| enc1-Q-only | 0.00522633 | 0.00122290 | 0.0266113 | 0.171387 | -0.00009535 |
| dec0-only | 0.00536660 | 0.00124439 | 0.0273438 | 0.144043 | -0.00003019 |

Direct hybrid-vs-FP16: **45.437526 dB / SSIM 0.998978101**.
Against enc1-Q-only: 45.636057 dB / 0.999021819;
against dec0-only: 45.406020 dB / 0.998972931.
The canonical PSNR change from FP16 is -1.130604 dB; it is not the direct error.

Spatial diagnostic (the runner's weighted absolute RGB error): hybrid-vs-FP16
32-pixel border/interior means 0.000783514 / 0.001003872. 16×16 block means range
0.000201690–0.003347397; worst (y,x): (31,40), (31,16), (31,48), (31,56), (31,24).
These weighted spatial values differ in definition from raw RGB mean absolute
error. Both canonical and all direct comparisons retain complete spatial metrics.
PNG exports use the existing display mapping; **metrics never use PNG pixels**.

## ISA, hashes and regressions

[M] RGA DX12 gfx1101 disassembly of the same executed up-projection shader:
**16 static `v_dot4_i32_i8`**, zero FP16 matrix mul/FMA/MAC/WMMA instructions.
This is ISA-lowering evidence tied to executed DXIL, not a runtime instruction trace.

| Artifact | SHA-256 |
|---|---|
| INT8 up shader | `f5324bcbd9ac70e91b1cb813d2ead573445483ced6a3cf45e03df495f670e217` |
| Original-remainder wrapper | `4cafedaf158b7316d2ff2bb2b0b5b18bf4b4b8ef9540071a4ad7cc2ecf3dce87` |
| FP16 up probe, validation only | `54c2d126399e26b9ccf014f4bb046850228ce62a4f3cd4e389bafbb0fdaef3dc` |
| Final network executor | `d52f4821f129ab6410cf9428b059ad71da91d4123003fafa06d7b46228272299` |
| MLP-only frame | `00a438dc2d8c74be5ccf514c5846a8484ce91cfbae5b722409d45bd6c663949c` |

Existing FP16 / enc1-Q / dec0 frame hashes remain those of
[EXP-025](EXP-025-m09-dec0-int8-frame.md). **314 tests pass**, preserving all
303 and adding 11 arithmetic/layout/mode/fixture/provenance regression cases.
Native build and diff whitespace validation pass.

B4 remains PASS, 99.9369%, max 7 ULP. B5 retains only the established enc1.merge
failure, 89.8379% agreement / relative RMS 0.02111 (conformance exit 1 expected).
The first conformance run correctly rejected the old host-executable provenance.
`m_translated_layer_suite.py --refresh-sette-from` now re-executes **only Sette
D3D12**, with two sentinel seeds, requiring all 22 independent-layer/chained
fields to match archived bytes before retaining their reference-pair metrics.
Historical summaries are not overwritten; a new summary links the old hash,
new execution commands, current executable/shaders and byte-equivalence checks.
No HIP execution, reference rebuild, rounding investigation or gate change.

## Reproduction and local evidence

PowerShell, using fresh directories. T is `%LOCALAPPDATA%\Temp\opencode`,
L is the established `m-capture-run/layers`, C is the enc1 replay above:

```powershell
cmake --build build/native --config Release
python -m pytest tests -q
$env:M_STAGE_DEBUG = '1'
python scripts/m09_mlp_up.py --source "$L/enc1" --capture $C --out "$T/stage"
# Execute controls with the existing B6 runner before the new frame.
python scripts/m_b6_frame.py --out "$T/fp16"
python scripts/m_b6_frame.py --out "$T/enc1q" --q-mode int8-q --enc1-int8 $Q
python scripts/m_b6_frame.py --out "$T/dec0" --dec0-mode int8-expansion
python scripts/m_b6_frame.py --out "$T/mlp" --mlp-mode int8-up --enc1-mlp "$T/stage/int8-up" `
    --fp16-frame "$T/fp16/downsample-native/results/out-out.bin" `
    --enc1q-frame "$T/enc1q/downsample-native/results/out-out.bin" `
    --dec0-frame "$T/dec0/downsample-native/results/out-out.bin"
python scripts/m09_mlp_up.py --out "$T/stage" --frame-run "$T/mlp" `
    --fp16-run "$T/fp16" --enc1q-run "$T/enc1q" --dec0-run "$T/dec0" --rga $RGA
python scripts/m_translated_layer_suite.py --refresh-sette-from $ARCHIVED_SUITE `
    --refresh-chain-baseline $ARCHIVED_FP16_CHAIN --out "$T/refreshed-suite"
python scripts/m_conformance.py --network --translated-network "$T/refreshed-suite" `
    --json "$T/conformance.json" # Expected exit 1: existing B5 numerical failure.
```

Authoritative local evidence under T:

* `m09e-verified-stage-20261030/validation.json`: physical, synthetic, independent
  CPU, split-equivalence and complete-write checks; `integration.json`: actual
  frame handoffs, stage isolation, control hashes, debug logs and executed ISA.
* `m09e-verified-{fp16,enc1q,dec0,mlp}-20261030/b6.json`: fresh full-frame runs.
* MLP root: `frame-native-{reference,reconstructed,difference}.png`, and
  `hybrid-vs-{fp16,enc1q,dec0}-{reference,reconstructed,difference}.png`.
* `m09e-translated-refresh-20261030/{refresh,summary}.json` and
  `m09e-conformance-20261030.json`: final unchanged B4/B5 evidence.
* Earlier `m09e-stage-*`, `m09e-frame-*`, `m09e-final-*` and the initial
  `m09e-verified-conformance-*` retain failed/provisional results; only the
  **verified** stage/frame roots and refreshed conformance above describe the
  final executable.

All these data/shader/image outputs stay outside Git.

## Limits and next action

One resolution/reset frame/content set and one adapter. No temporal work,
performance claim, calibration, or complete-MLP/network INT8 claim. Diagnostic
buffers/readbacks remain intentionally large. The down projection, attention,
normalization, GELU and residuals are explicit unchanged FP16/floating-point
operations. Major-stage file boundaries in B6 remain as documented in EXP-025.
General static fidelity stays INCONCLUSIVE; B5 stays FAIL by design.

**Next priority is systematic INT8 coverage toward M10, not another detached
probe.** Under the next authorized scope, complete the MLP pair and grow the
existing graph's matrix-family coverage, using per-operator CPU/ISA/handoff
checks as integration gates. Track remaining FP16 matrix exceptions and their
coverage in M09 rather than claiming completeness from three individual paths.
Do not enable combined modes or advance M10 without the corresponding scope
and validation. M09 remains **IN PROGRESS / PENDING**.
