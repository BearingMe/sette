# EXP-042 — Complete performance baseline: GPU interval, host phases and VRAM

**Milestone:** M14 · **Status:** PASS for the achievable baseline. **C03
(optimized FP16 comparator) is BLOCKED**: no comparable optimized FP16 backend
exists in the pinned references or this D3D12 harness. The conservative FP16 and
full-INT8 comparisons are reproducible; the 45 dB quality gate is unchanged and
still FAIL.
**Revision:** the commit containing this report (M14 instrumentation; parent `dc92493`).
**Depends on:** [EXP-037](EXP-037-profiling-environment.md),
[EXP-038](EXP-038-dispatch-profile.md), [EXP-039](EXP-039-bottleneck-analysis.md),
[EXP-040](EXP-040-optimization-results.md), [EXP-041](EXP-041-quality-recovery.md).

## 1. Question and scope

What is the complete frame's GPU execution interval, how much host work surrounds
it, how much VRAM does it actually use, and which measured costs should drive
M15? Existing evidence is re-used where it is still valid; the new measurements
close the missing VRAM, host-phase and whole-command-list timing gaps. No
quantization, temporal or proxy-integration work is performed.

## 2. Verification

| Check | Result |
|---|---|
| Repository revision | `dc92493` + M14 instrumentation |
| Test suite | **383 passed** (377 before this experiment) |
| FP16 frame hash | `dc97c4cb…d6e95b`, 3/3 runs |
| Full INT8 frame hash | `f76ac032…4d6c00`, 3/3 runs |
| Native INT8 coverage | **76/76**, 34,518,597,632 MACs (unchanged) |
| M09 finalizer | 17/17 PASS on the accepted path (EXP-041) |
| Quality verdict | 45 dB gate **FAIL** (44.1393 full INT8; 44.4026 + asym) |

## 3. Timing methodology

The `*_timed` probes now record three scopes in one run:

* **per-dispatch** `BeginQuery`/`EndQuery` around every dispatch;
* **whole command list** one timestamp pair from before the first dispatch to
  after the last dispatch (`global.ms`);
* **host phases** `steady_clock` points around startup, parse/load+allocation,
  command-list recording, `ExecuteCommandLists`+fence wait, and readback/write.

The GPU timestamp frequency is **100,000,000 Hz** in every process. FP16 and INT8
are interleaved (FP16, INT8, FP16, …), with one warm-up frame per mode and three
measured repeats. Each run's final frame must equal the accepted hash; otherwise
the run is discarded. The host executes one command list and waits on one fence;
there is no mid-frame wait.

**Why two GPU totals.** Per-dispatch intervals can overlap: in one repeat the FP16
dispatch sum was **567.6 ms** while the whole-list interval was **525.8 ms**
(negative "gap"), because adjacent dispatches pipelined. Summing per-dispatch
deltas can therefore overcount. The **whole-command-list interval** is the
primary GPU execution measure; the dispatch sum is retained as a diagnostic and
for compatibility with EXP-038. Neither is game-frame latency: both exclude
process startup, resource creation/upload, Python/fixture I/O, and readback
writes.

## 4. Results

### 4.1 Whole-frame GPU interval

| Mode | Dispatches | GPU interval median | Min–max | Dispatch sum | Gap (interval − sum) |
|---|---:|---:|---:|---:|---:|
| FP16 | 98 | **565.472 ms** | 564.35–566.22 | 564.283 | +0.077 |
| Full INT8 | 243 | **207.463 ms** | 204.29–209.02 | 202.429 | +6.592 |
| Speedup | — | **2.726×** | — | 2.788× | — |

[M] The M14 interval-based speedup is **2.73×**; the dispatch-sum ratio is
**2.79×**. The earlier EXP-038 result (600.124 / 205.167, **2.93×**) was a
dispatch-sum comparison in a different clock window. The current interleaved
measurement is the authoritative M14 baseline; the difference is within the
observed FP16 run-to-run range (525–600 ms) and does not change the conclusion
that native INT8 is substantially faster.

### 4.2 Per-process GPU and host phases (medians, first-repeat process set)

INT8:

| Process | Dispatches | GPU interval | Dispatch sum | Host total | Load/alloc | Readback/write |
|---|---:|---:|---:|---:|---:|---:|
| `enc0` | 4 | 7.34 ms | 3.17 ms | 364.7 ms | 150.2 ms | 65.4 ms |
| `chain` | 234 | **182.34 ms** | 181.10 ms | **5869.8 ms** | **5534.9 ms** | 18.0 ms |
| `dec0` | 3 | 1.33 ms | 1.32 ms | 251.2 ms | 35.9 ms | 60.9 ms |
| `post` | 1 | 14.79 ms | 14.79 ms | 296.7 ms | 24.2 ms | 52.7 ms |
| `downsample` | 1 | 1.68 ms | 1.68 ms | 253.6 ms | 12.6 ms | 9.8 ms |
| **Total** | **243** | **207.46 ms** | **202.43 ms** | **7032.8 ms** | — | — |

FP16:

| Process | Dispatches | GPU interval | Dispatch sum | Host total | Load/alloc | Readback/write |
|---|---:|---:|---:|---:|---:|---:|
| `enc0` | 1 | 14.66 ms | 14.66 ms | 167.9 ms | 28.8 ms | 5.7 ms |
| `chain` | 94 | **523.47 ms** | 522.11 ms | 1860.4 ms | 1202.5 ms | 17.6 ms |
| `dec0` | 1 | 10.95 ms | 10.94 ms | 200.1 ms | 31.4 ms | 9.2 ms |
| `post` | 1 | 14.79 ms | 14.78 ms | 297.5 ms | 24.5 ms | 54.1 ms |
| `downsample` | 1 | 1.66 ms | 1.66 ms | 255.9 ms | 12.6 ms | 10.3 ms |
| **Total** | **98** | **565.47 ms** | **564.28 ms** | **2777.5 ms** | — | — |

[M] The GPU frame is **207 ms** but the sum of the five probe process lifetimes
is **7.03 s** (FP16: 2.78 s). The chain alone spends **5.53 s** creating and
uploading 680 resources; process startup and readback/write add the rest. The B6
orchestrator's wall time is **18.0 s INT8 / 13.0 s FP16** median, adding Python
graph construction, fixture cloning and file copies. None of this is game-frame
latency; it is the integration gap.

[M] `post` and `downsample` are precision-independent (14.79 ms and ~1.7 ms in
both modes). `post` is now **7.1%** of the INT8 GPU interval.

### 4.3 Scope separation

For INT8:
* GPU whole-list interval: **207.46 ms**
* Sum of per-process `ExecuteCommandLists`+fence windows: **229.79 ms**
* Difference: **22.3 ms**, which includes the 151.7 MB of readback copies
  (41.8 MB in the chain) and submission/queue/fence latency; it is not pure CPU
  overhead.
* Host preparation: **7.03 s** sum of probe lifetimes (B6 wall 18.0 s),
  dominated by resource creation/upload.
* Dispatch gaps: 6.59 ms within the measured list; no evidence of a mid-frame
  synchronization point.

[M] Only **41.8 MB** of the chain's outputs are read back in the B6 frame. A
diagnostic run that dumps every intermediate allocates and copies **4.30 GB** of
readbacks, adding **~232 ms** of GPU copy time outside the dispatch interval. The
M12 profiling chain did not do that; the difference explains why its window and
dispatch totals were closer.

## 5. VRAM

### 5.1 Adapter and budget

| Item | Value |
|---|---:|
| Adapter | AMD Radeon RX 7700 XT, vendor 0x1002, device 0x747E |
| Dedicated video memory | 12,667,666,432 B (12.67 GB) |
| DXGI local budget (process) | **11,862,360,064 B (11.86 GB)** |
| Observed local peak, full INT8 | **4,692,504,576 B (4.693 GB)** |
| Observed local peak, FP16 | 1,019,101,184 B (1.019 GB) |

### 5.2 Tracked allocations

| Scope | FP16 | Full INT8 |
|---|---:|---:|
| Allocations tracked (all processes) | 192 | 752 |
| Total allocated, all heaps | 1,031.1 MB | **5,066.2 MB** |
| Default heap | 852.3 MB | 4,522.4 MB |
| Upload heap | 27.0 MB | 182.1 MB |
| Readback heap | 151.7 MB | 151.7 MB |
| Mixed `enc0`/`dec0` prefix buffers (heap not separately recorded) | 0.0 MB | 210.0 MB |
| Transient staging peak (one resource) | 17.5 MB | 139.9 MB |
| Conservative peak bound (total + transient) | 1,061.1 MB | **5,218.6 MB** |
| Largest single allocation | 37.3 MB (`post` history) | **139.9 MB** (`enc1` MLP-up accumulator) |

[M] The INT8 chain alone declares **680 resources / 4,389,619,168 B**, exactly
the 4.39 GB reported earlier. The extra tracked bytes are the image stages,
readbacks and staging. The `enc0`/`dec0` prefix buffers are counted in the total
but not split by heap, so `default + upload + readback + mixed = total`.
Readbacks are non-local system memory; the DXGI local usage is the VRAM measure.

[M] The chain's observed local usage rises from **4,473.6 MB after load** to
**4,692.5 MB after execution**; the ~219 MB difference is driver/runtime
allocation, not graph default buffers. The tracked default sum plus a ~300 MB
driver baseline explains the observed local usage; do not equate declared bytes
with residency.

### 5.3 Lifetime and aliasing bound

Using the production chain graph (13 output dumps, no diagnostic dumps):

| Measure | Bytes |
|---|---:|
| Declared resources | 4,389,619,168 |
| Peak live with perfect aliasing | **385,209,616 (385.2 MB)** |
| Potential saving | **4,004,409,552 (4.00 GB, 91.2%)** |
| Longest-lived chain resources | per-layer MLP temporaries, one layer at a time |

The evidence graph that dumps every intermediate peaks at **4,312,663,760 B**
because its 159 diagnostic resources are live to the final dump; those
diagnostics total **2,581,770,240 B** (2.58 GB). The host allocates all 680
resources up front and never reuses them, so it realizes the full 4.39 GB
instead of the 385 MB live bound.

[M] Per-layer temporaries are the dominant allocations: `enc1` MLP-up
accumulator and float output are 139.9 MB each; `dec1` equivalents are 133.7 MB.
They are live only around their producing layer in the production graph, so an
arena could reuse them across layers.

[M] The B6 chain still **writes** those diagnostic accumulators even though it
does not dump them, so the 2.58 GB is also write traffic. Removing the writes
requires shader variants or a diagnostic flag; that is M15 work, not a measured
win yet.

## 6. Optimized FP16 comparator (C03)

**BLOCKED. No comparable optimized FP16 implementation is available.**

* [S] d4r (`05e19d4`) has a native Preset M path, but it is Linux/Vulkan with
  ROCm/HIP; it is not built on this Windows/D3D12 workstation. Its README calls
  the optimized paths secondary performance references. No D3D12 FP16 backend
  exists in the pinned tree.
* [S] quattro (`ad0482a`) contains validated FP16 WMMA **microbenchmarks**
  (`v_wmma_f32_16x16x16_f16`, `v_wmma_f16_16x16x16_f16`), not the Preset M
  network. The reference inventory explicitly notes that its driver-generated
  operand-gather prologue dominates and is not evidence about optimized
  throughput.
* [S] The DLSS SDK reference is headers only; TransformerEngine is CUDA FP8
  recipe context; neither contains a runnable Preset M FP16 backend.

The conservative FP16 reconstruction (565.472 ms interval) is therefore the only
FP16 comparator measured here. Building a new optimized FP16 backend solely to
close C03 is explicitly out of scope.

## 7. M14 checkpoint verdicts

| Checkpoint | Verdict | Evidence |
|---|---|---|
| C01 repeatable conditions | **PASS** | Interleaved modes, warm-up, 3 repeats, frame-hash gate, 100 MHz timestamps, whole-list interval |
| C02 conservative FP16 latency | **PASS** | 565.472 ms interval (525–600 ms across sessions; EXP-038 600.124) |
| C03 optimized FP16 latency | **BLOCKED** | No comparable optimized FP16 backend; §6 |
| C04 INT8 latency | **PASS** | 207.463 ms interval, 202.429 ms dispatch sum |
| C05 kernel vs dispatch/host overhead | **PASS (bounded)** | Per-dispatch, whole-list, submit/fence, record and host phases separated; no mid-frame sync |
| C06 memory/conversion costs | **PASS (bounded)** | Tracked 5.066 GB all heaps; 4.693 GB local peak; 4.30 GB diagnostic readback in a full-dump run; no DRAM counters |
| C07 VRAM | **PASS** | 4.693 GB local peak, 11.86 GB budget, largest 139.9 MB, 385 MB alias bound |
| C08 workload-dependent bottlenecks | **PASS (bounded)** | Family profile unchanged; 9-frame content sweep in EXP-041; no new scene family |
| C09 warm-up/sync/validation overhead | **PASS** | One warm-up, one fence per process, timestamps inside the submission, readbacks after the interval |

**Gate:** PASS for the bounded baseline. C03 remains BLOCKED and is not presented
as measured. The 45 dB quality verdict and native INT8 coverage are unchanged.

## 8. M15 candidates (ranked, not implemented)

| Rank | Candidate | Measured cost | Expected benefit | Risk | Decision |
|---:|---|---:|---|---|---|
| 1 | **MLP-up epilogue GELU + MLP-down quantization fusion** | GELU 24.44 ms + MLP-down quant 12.73 ms = **37.2 ms** (20.4% of the 182.34 ms INT8 interval); 70 MB intermediate | Remove one pass and one 70 MB write/read; up to ~20–30 ms realistic | Medium-high: GELU is pointwise and bit-exact feasible; the row-max quantizer needs a cross-tile reduction; full revalidation | **Select** |
| 2 | **Production arena + diagnostic elimination** | Declared 4.39 GB vs **0.385 GB** alias bound; 2.58 GB diagnostic writes; 5.53 s chain load/alloc | 91% lower chain footprint; lower host preparation; GPU-time effect to be measured | Medium: graph/allocator/barrier rewrite, shader variants, no arithmetic change | **Select** |
| 3 | Quantize-into-GEMM for Q/V/Wo/merge/expand | remaining quantizer family ~24.8 ms | ~12–25 ms | Medium-high: more validated GEMM boundaries | Next, not selected |
| 4 | Attention pack + P-quant fusion | pack 18.2 ms | ~9–18 ms | Medium: layout/handoff change | Not selected |
| 5 | `post` reconstruction optimization | 14.79 ms (7.1% of INT8 frame) | Up to ~14 ms, benefits both modes | Medium: shared FP16 stage, outside INT8 matrix scope | Not selected |
| 6 | Dispatch-dependency reduction | 6.59 ms gap + 21.5 ms in sub-0.5 ms dispatches | Bounded by the above fusions | Low alone, low payoff | Fold into 1/3 |

**Selected M15 task (bounded):**

1. **MLP chain fusion.** In the validated MLP-up shader, apply the original GELU
   to the f16-rounded accumulator and write the post-GELU f16 field; then
   quantize that field for MLP-down, or in a fused epilogue. The GELU half is a
   pointwise epilogue; the quantizer half needs a per-token reduction across the
   4C output tiles, so the combined pass carries more risk than GELU alone.
   Success criterion: bit-exact or documented-equivalent output against the
   current `mlp-gelu` + `mlp-down-quant` path, 76/76 INT8, CPU/GPU differential,
   and a measured GPU interval reduction. Reject if any arithmetic drift is
   unexplained.
2. **Production arena.** Alias non-overlapping per-layer buffers and skip
   diagnostic `Acc`/`FloatOut` writes in a production shader variant. Success
   criterion: the accepted frame hash/content unchanged, peak live footprint
   reduced toward the 385 MB bound, and no new barrier/synchronization defect.

No implementation is started in M14.

## 9. Limitations

* One device, driver, DLL build, resolution and scene family.
* The GPU interval is one timestamp pair around one command list; it excludes
  queue submission before the first dispatch and process preparation. It is not
  a game-frame latency.
* DXGI `CurrentUsage` is process usage but includes driver allocations; it is
  not equal to the tracked graph sum. Peak is sampled at defined points; a
  transient between samples could be higher.
* Readbacks are non-local system memory; the tracked all-heap total is not VRAM.
* No hardware counters, clock/power control or presentmon; utilization claims
  remain timing-based inferences.
* The 385 MB alias bound is an interval-graph lower bound, not a measured arena
  implementation; alignment, descriptor heaps and driver overhead will raise it.
* The optimized FP16 comparator is missing; C03 is BLOCKED, not complete.

## 10. Reproduction

```powershell
cmake --build build/native --config Release

python scripts/m14_baseline.py --out "$T/m14-baseline" --repeats 3

python scripts/m14_memory.py --build-from-layers `
  "$T/m09h-complete-chain-20261030/layers" --out "$T/m14-memory-production.json"

python -m pytest tests -q        # 383 passed
```

Evidence: `m14-baseline-20261030/m14-baseline.json`,
`m14-memory-production.json`, `m14-memory-chain.json`, per-process
`timing/*.json` and `vram/*.json` under the baseline directory.
