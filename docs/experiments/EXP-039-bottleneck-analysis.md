# EXP-039 — GPU bottleneck analysis

**Milestone:** M12-C · **Status:** PASS for the timing-based analysis.
**Hardware counters are unavailable**; everything marked `[H]` is an inference
from timing and declared traffic, not a measured utilization counter.
**Depends on:** [EXP-037](EXP-037-profiling-environment.md),
[EXP-038](EXP-038-dispatch-profile.md). **Data:** `m12-analysis.json`.

## 1. Question

Where does the frame's GPU time go, and is the dominant limit compute, memory,
dispatch overhead, or synchronization? The answer must not be inferred from the
presence of `dot4` instructions.

## 2. Method

Per-dispatch GPU timestamps (EXP-038) joined with the graph's declared resource
sizes. For each dispatch:

* measured time from the timestamp delta;
* declared traffic = sum of bound SRV resource sizes + bound UAV resource sizes
  (**upper bound**: a stage may touch less than the whole resource);
* effective bandwidth = traffic / time.

Family totals are sums of per-dispatch medians. They are not a hardware-counter
trace.

## 3. Results

### Chain time by family

FP16 (sum of label medians 543.6 ms):

| Family | ms | Share |
|---|---:|---:|
| `mlp` (fused FP16) | 286.8 | **52.8%** |
| `attention` (fused FP16) | 116.8 | **21.5%** |
| V projection | 24.7 | 4.5% |
| Q projection | 24.7 | 4.5% |
| Wo projection | 24.3 | 4.5% |
| merge | 19.7 | 3.6% |
| norm2 | 18.8 | 3.5% |
| expand | 13.7 | 2.5% |

INT8 (sum of label medians 188.7 ms baseline, 171.8 ms optimized):

| Family | baseline ms | share | effective GB/s |
|---|---:|---:|---:|
| mlp-gelu | 24.4 | 12.9% | 22.7 |
| mlp-up | 21.8 | 11.5% | 57.6 |
| attention-pv | 22.3 | 11.8% | 22.2 |
| attention-pack | 19.9 | 10.6% | 14.4 |
| norm2 | 17.6 | 9.3% | 9.9 |
| mlp-down-quant | 16.7 | 8.8% | 28.9 → 28.9 (optimized 12.7) |
| norm1 | 8.8 | 4.7% | 22.2 |
| wo | 6.2 | 3.3% | 64.3 |

Quantization + pack adapters: **52.9 ms / 28.0%** before the optimization,
**37.5 ms / 21.8%** after (EXP-040).

### Traffic and effective bandwidth

[M] Declared chain resources: **680 buffers**, 4.39 GB total, largest 139.9 MB.
[M] Sum of declared per-dispatch traffic (upper bound): **6.34 GB per frame**.
At 188.7 ms that is an average of ~33.6 GB/s; even at the optimized 171.8 ms it
is ~37 GB/s.

[M] Individual large stages:

| Stage | ms | Effective GB/s |
|---|---:|---:|
| `mlp-up` (GEMM) | 21.8 | 57.6 |
| `wo` (GEMM) | 6.2 | 64.3 |
| `mlp-down-quant` | 16.7 | 28.9 |
| `attention-pv` | 22.3 | 22.2 |
| `attention-pack` | 19.9 | 14.4 |
| `norm2` | 17.6 | 9.9 |
| `prepare` (small, FP16) | 0.03 | ~150 |
| `attention-p-quant` (tiny) | 0.007 | noise |

### Dispatch overhead

[M] FP16: 98 dispatches over 615.6 ms → 6.3 ms average. INT8: 243 dispatches over
226.8 ms → 0.93 ms average. 60 INT8 dispatches are below 0.1 ms and total
2.5 ms; 137 are below 0.5 ms and total 21.6 ms. The timestamp deltas cannot
separate launch/barrier overhead from the dispatch's own work, so no
per-dispatch overhead constant is claimed.

### Synchronization

[S] One fence wait per process; five processes per frame. Timestamps measure
inside each submission, so host waits and file I/O are excluded from every
number above.

## 4. Interpretation

* [M] FP16 spends **74%** of its chain time in the fused MLP and attention
  stages. Those are exactly the stages the INT8 path replaces.
* [M] After replacement, the INT8 chain is spread across MLP, attention, norms
  and quantization adapters; no single family exceeds ~14%.
* [M] The average effective bandwidth (~34 GB/s from an over-counted traffic
  bound) is far below a bandwidth-saturated kernel. [H] The dominant INT8 limits
  are therefore instruction/latency/occupancy and the serialized quantize →
  GEMM → layout structure, not DRAM bandwidth. This is an inference; no counter
  confirms it.
* [M] The quantization and pack adapters are a first-order cost (28%, then 22%),
  not incidental glue.
* [M] `post` (14.8 ms) and `downsample` (1.7 ms) are unchanged by the precision
  mode and become 7.9% of the optimized INT8 frame.
* [U] Register pressure, occupancy, cache hit rates, and the exact split between
  compute and memory stalls.

## 5. Next

EXP-040 tests the one change these data select: more parallelism inside the
quantizer. Fusion of quantize into the GEMM load is the next candidate.

## 6. Limitations

* Traffic is an upper bound from declared resource sizes.
* Family shares use per-label medians; their sum differs from the median of
  per-run totals by ~4–10%.
* No hardware counters, no clock/power control.
