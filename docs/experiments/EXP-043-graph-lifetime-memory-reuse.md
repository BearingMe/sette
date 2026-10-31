# EXP-043 — Production graph, buffer lifetimes and GPU memory reuse

**Milestone:** M15-A · **Status:** PASS for the production graph and arena;
numerical outputs are byte-identical and the production/diagnostic split is
implemented. Full quality is unchanged: the 45 dB gate still FAILs, and no
quantization, temporal or proxy work is included.

**Revision:** the commit containing this report; parent `afdca86` (M14).
**Depends on:** [EXP-042](EXP-042-performance-baseline.md) (M14 baseline),
[EXP-040](EXP-040-optimization-results.md) (accepted quantizer),
[EXP-041](EXP-041-quality-recovery.md) (quality configurations).

## 1. Question and scope

The M14 baseline showed a 4.39 GB declared chain, a 385.2 MB perfect-aliasing
lower bound and a 5.5 s resource-creation phase. Can the same graph execute with
explicit production/diagnostic modes and a bounded, hazard-free GPU arena, and
does reuse change the GPU interval or any accepted output? No numerical
algorithm, quantization scheme or fidelity threshold is changed.

## 2. Verification

| Check | Result |
|---|---|
| Tests | 391 passed |
| FP16 frame hash | `dc97c4cb…d6e95b` (committed and production+arena) |
| Full INT8 frame hash | `f76ac032…4d6c00` (committed and production+arena) |
| Chain tensor `out-dec1` | `c89208ed…3c6b0d` (all configurations) |
| Native INT8 coverage | **76/76**, 34,518,597,632 MACs (unchanged) |
| M09 finalizer | **17/17 PASS** with the diagnostic arena |
| Diagnostic tensors | 13/13 byte-identical to committed |
| D3D12 debug layer | zero errors, corruptions and warnings |
| Quality | 45 dB gate **FAIL**, unchanged (44.4875) |

## 3. A1 — Lifetime inventory

`m_network_stage` now derives first/last use from the graph and classifies every
resource before allocating it. `scripts/m15_arena.py` reproduces the analysis and
emits a machine-readable inventory (`m15a-inventory-final.json`):

| Class | Resources | Bytes (aligned) | Rule |
|---|---:|---:|---|
| Persistent | 153 | 24,379,392 | has an init file and is bound/dumped; uploaded once |
| Transient | 516 | — | allocated at first use, freed after last use, may alias |
| Dedicated | 0 | 0 | read-before-write, or SRV+UAV in one stage |
| Dead | 11 | 0 | never bound or dumped (per-layer input placeholders) |

The accepted chain has **234 stages, 680 resources and 13 dumps**. Declared bytes
remain **4,389,619,168**. Every transient resource is written before it is read;
no resource is bound as both SRV and UAV in the same stage, so the conservative
alias gate admits all 516. The 11 placeholders (`dec1_low`, `dec1_skip`,
`dec2_low`, `dec2_skip`, `enc2_in`, six `tube_*_in`) are rewired away by
`CHAIN_EDGES`; the host no longer uploads them (25.1 MB of init data skipped).

The inventory records `first_write`, `last_read`, `first`, `last`, `init`,
`srv`/`uav`, `both_states_one_stage`, `reads_before_write`, `dumped`,
`lifetime`, `offset` and `aligned_bytes` for every resource. A dump is a use after
the final dispatch, so dumped tensors are live to frame end.

**Never-read tensors.** 180 resources (2,606,837,920 B) have no SRV binding; one
of them is the final output. The other 179 (2,598,482,080 B) are never read: 11
are unbound placeholders and 168 are write-only temporaries produced by required
stages (2,573,414,560 B). Those 168 cannot be dropped without a shader variant:
they are `Acc`/`FloatOut`/scratch UAVs of the validated matrix stages, not
separate dispatches. Their memory is already aliased away; their write bandwidth
remains the next optimization target.

## 4. A2 — Production and diagnostic graphs

`--mode diagnostic` preserves the accepted behaviour: every declared dump is
read back. `--mode production` keeps only dumps whose resource is a UAV of the
final stage (`dec1.store`), one per resource, preferring the last declared file.
For the accepted chain that is **13 → 1 dump** (`out-dec1.bin`); 12 intermediate
readbacks are removed.

Production also shortens lifetimes: intermediate encoder/decoder outputs are no
longer live to frame end. No computational stage is omitted, because every stage
still writes at least one tensor that is consumed later or is the final output.
The 12 removed dumps are validation outputs, not network state.

| Mode | Dumps | Chain readback |
|---|---:|---:|
| Diagnostic | 13 | 41,779,200 B |
| Production | 1 | 8,355,840 B |

Boundary hashes (33 in the diagnostic B6 frame, 21 in production) remain available and are
used by the M09 finalizer. Production mode is for execution, not inspection.

## 5. A3 — Arena implementation

The arena is a static, graph-driven plan, not a general allocator:

* one `D3D12_HEAP_TYPE_DEFAULT` heap with `ALLOW_ONLY_BUFFERS`, size
  **437,387,264 B (437.4 MB)**;
* 516 placed buffers, each 64 KiB aligned (the D3D12 placement requirement),
  sized to its logical tensor and created with `ALLOW_UNORDERED_ACCESS`;
* best-fit interval allocation: allocate at `first`, free after `last`, never
  overlap nested lifetimes (10 bump regions; the diagnostic peak covers 88.0%
  of the heap, production 86.1%);
* persistent weights/inputs are separate committed buffers and never alias;
* dead resources are not created at all;
* a full-heap canary resource receives a seed-dependent sentinel once, so a
  read-before-write is visible as a seed-dependent output;
* a global `D3D12_RESOURCE_BARRIER_TYPE_ALIASING` is emitted before every
  dispatch (234 per frame), alongside the existing per-stage UAV barrier;
* resource state is tracked per placed object, so transitions remain per logical
  tensor even when two tensors share memory.

Conservative rejections are implemented and tested: a resource whose first use is
an SRV, or that is bound as SRV and UAV in one stage, becomes dedicated. The
accepted graph has none, but the rule prevents unsafe aliasing for future graphs.

The **385.2 MB** figure from M14 is still only a lower bound. The measured arena
is **437,387,264 B (437.4 MB)**, with **384,890,896 B (384.9 MB)** peak transient live bytes
(**88.0%** utilization). Alignment and best-fit fragmentation account
for the difference.

`--frames N` executes the same initialized engine N times, resetting the
allocator/list per frame while keeping device, pipelines, weights and arena
resident. Dumps are written to `frame-0..N-1` for comparison.

## 6. A4 — Numerical preservation

| Comparison | Result |
|---|---|
| Diagnostic arena vs committed, 13 dumps | **byte-identical** |
| Production final tensor vs committed | **byte-identical** |
| Full B6 frame, INT8 | `f76ac032…4d6c00` |
| Full B6 frame, FP16 with production+arena | `dc97c4cb…d6e95b` |
| Seed 0 vs seed 1 (canary) | identical |
| Three frames in one process | identical (`frame-0 == frame-1 == frame-2`) |
| M09 finalizer | 17/17 PASS |
| D3D12 debug layer | zero error/corruption messages |

The finalizer was run on a diagnostic arena frame (the only mode that preserves
all 13 tensors) with the accepted chain and ISA evidence. It re-checked enc0,
dec0, layer handoffs, native ISA (`dot4 > 0`, `fp16_matrix == 0`), script/host
provenance, complete writes and finite/unsaturated fields. Quality is unchanged:
the canonical full-INT8 comparison remains 44.14 dB and this frame's reference
comparison reports 44.4875 dB; the 45 dB gate still fails.

## 7. A5 — Measurements

Conditions: RX 7700 XT, Windows, D3D12 compute, same accepted chain graph and
token input, seed 0, `M_STAGE_DEBUG=1`, one warm-up per configuration and three
interleaved rounds. Every run reproduced the accepted `out-dec1` hash. Reported
scopes are kept separate.

| Chain configuration | GPU interval (ms) | Dispatch sum (ms) | DXGI local peak | Tracked total | Load/alloc | Host total |
|---|---:|---:|---:|---:|---:|---:|
| Committed, no arena | 187.59 | 187.18 | 4,667.3 MB | 4,406.3 MB | 6,023.09 ms | 6,397.88 ms |
| Arena, diagnostic | 165.66 | 165.05 | 714.8 MB | 495.4 MB | 989.29 ms | 1,338.36 ms |
| Arena, production | 166.12 | 164.35 | 714.8 MB | 461.9 MB | 987.88 ms | 1,306.21 ms |

| Change | Saving |
|---|---:|
| Arena GPU interval vs committed | 21.93 ms (11.7%) |
| Production GPU interval vs committed | 21.47 ms (11.4%) |
| Local VRAM peak | 6.53x lower (3.95 GB) |
| Chain load/allocation | 5,035.20 ms lower (83.6%) |
| Chain host total | 5,091.66 ms lower (79.6%) |
| Production readback time | 15.66 ms lower |
| Production readback bytes | 33,423,360 B removed |

The largest remaining transient is `enc1_i8_mlp-up_acc` (139,919,360 B); it is
aliased with later layers' temporaries rather than allocated per layer. Tracked
totals are requested sizes; the persistent plan bytes above are 64 KiB aligned.
DXGI local usage is about 245 MB above the tracked total, consistent with the
driver/runtime overhead seen in M14.

The arena also changes the measured barrier profile: the first frame records
1028 transitions, 234 UAV barriers and 234 aliasing barriers. In a three-frame
run the last frame records 707 transitions because resource states persist
across frames; UAV and aliasing counts are unchanged. Production drops 11
cold-frame transitions by removing the 12 intermediate dump paths. Dispatch
count is unchanged at 234, so the GPU interval change is not a workload change.

Full-frame A/B in the same session (M14 harness, interleaved FP16/INT8, 3
repeats). INT8 uses the arena; FP16 stays on committed buffers because its
chain is already under 1 GB and the arena has no measured benefit there:

| Mode | Committed | Production | Change |
|---|---:|---:|---:|
| FP16 GPU interval | 602.05 ms | 609.08 ms | -1.2% |
| INT8 GPU interval | 213.72 ms | 195.88 ms | 8.4% |
| INT8 local VRAM peak | 4,667.3 MB | 714.8 MB | 84.7% lower |
| FP16 local VRAM peak | 993.9 MB | 993.9 MB | unchanged |
| INT8 probe host total | 7,830.18 ms | 2,548.81 ms | 67.4% lower |

The INT8 result is consistent with the chain-level comparison. The FP16
difference is within the observed session drift (the FP16 GPU path and its
committed buffers are unchanged by production mode) and is not claimed as a
gain or a regression. For reference, forcing the arena on the FP16 chain in an
earlier run lowered its local peak from 993.9 MB to 376.3 MB but did not lower
the interval; the production policy therefore leaves FP16 committed. The
chain-level interleaved numbers are the primary
evidence because they isolate the changed component; full-frame numbers include
the unchanged image stages and the B6 file handoffs.

**Interpretation.** The GPU interval improves by roughly 8-12% depending on the
baseline: 11.7% against this session's committed block, 9.0-9.5% against the
M14 chain median, and 8.4% for the complete frame, even though the dispatch list
and arithmetic are unchanged. The plausible cause is cache/TLB locality: 4.39 GB
of distinct committed buffers become a 437 MB working set.
This is a timing inference, not a hardware-counter measurement; no clock/power
control or utilization counter is available.

## 8. A6 — Reusable execution boundaries

| Object | Lifetime | Evidence |
|---|---|---|
| Device, queue, allocator, fence, event | process | created once in `Runner` |
| Root signature, PSO, descriptor heap per stage | process | 234 stages prepared once |
| Weights and persistent inputs | process | 24.4 MB, uploaded once |
| Transient arena | process | one heap, allocated once |
| Command allocator/list | frame | reset per frame, one fence |
| Input bindings | frame | persistent tensor content; no per-frame upload |
| Output resources | frame | a fresh readback buffer per dump; production uses one |
| Command recording | frame | 234 dispatches, 234 UAV + 234 aliasing barriers |
| Readback | frame | after the GPU interval; production final tensor only |

This is the boundary the standalone engine can expose to a later in-process
adapter: initialize once, then call `ExecuteFrame`. The DLL adapter, proxy
contract and cross-frame temporal state are explicitly out of scope.

## 9. Limitations and risks

* One device, driver, DLL build, resolution and scene. The 8-9% GPU saving is
  device/clock dependent and inferred from timing, not counters.
* The arena is static for the accepted graph. A new graph must be re-analyzed;
  the conservative rules are tested but the planner is not a general allocator.
* 64 KiB placement and best-fit fragmentation cost 13.6% over the
  385.2 MB live lower bound; the lower bound is never reported as usage.
* The canary fill uploads the whole heap once (437,387,264 B (437.4 MB)), trading init bandwidth
  for stale-data detection. Persistent data is still resident.
* 2.6 GB of never-read tensors are still written by required stages. Eliminating
  those writes needs a production shader variant, not an allocator change.
* Production mode intentionally removes intermediate readbacks; inspection and
  the M09 boundary checks require diagnostic mode.
* DXGI `CurrentUsage` is process-wide and includes driver allocations; the peak
  is sampled at defined points, not continuously.
* No quantization, fidelity, temporal, DXGI proxy or DLL work was performed.

## 10. Reproduction

```powershell
cmake --build build/native --config Release

# Inventory + native plan agreement
python scripts/m15_arena.py `
  --graph "$T/m15a-measure/chain/graph.txt" `
  --out "$T/inventory.json" --check-report "$T/arena.json"

# Interleaved committed / arena / production comparison
python scripts/m15_measure.py `
  --build-from-layers "$T/m09h-complete-chain-20261030/layers" `
  --enc1-input "$T/m15a-final-diag/enc0-native/out-tokens.bin" `
  --out "$T/m15a-measure" --repeats 3

# Full frame, production graph
python scripts/m_b6_frame.py --out "$T/frame-prod" `
  --enc0-mode int8-embedding --attention-mode int8 `
  --cumulative-mode spatial `
  --cumulative-layers "$T/m09h-complete-chain-20261030/layers" `
  --network-mode production --network-arena

python -m pytest tests -q        # 391 passed
```

Evidence: `m15a-measure/m15a-measure.json`, `m15a-full-base/m14-baseline.json`,
`m15a-full-prod/m14-baseline.json`, `m15a-finalize.json`,
`m15a-inventory-final.json`, and the per-run `timing.json`, `vram.json` and
`arena.json` files.

## 11. Recommended M15-B

**MLP-up epilogue GELU + MLP-down quantization fusion**, as selected in EXP-042.
Measured cost is GELU 24.44 ms + MLP-down quantization 12.73 ms (37.2 ms, 20.4%
of the 182.34 ms chain). The GELU half is pointwise and can be bit-exact; the
row-max quantizer needs a cross-tile reduction. Success requires the accepted
hashes, 76/76 coverage and a measured interval reduction with the arena enabled.

A second, separately validated candidate is a production shader variant that
stops writing the 168 never-read `Acc`/`FloatOut` tensors (2.57 GB). It changes
no arithmetic but requires new shader entry points and a full revalidation; it
should not be bundled with the MLP fusion.
