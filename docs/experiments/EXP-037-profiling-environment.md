# EXP-037 — GPU profiling environment

**Milestone:** M12-A · **Status:** PASS for the environment and method.
Hardware performance counters are **unavailable** on this path; all numbers are
GPU timestamp measurements (see §5).
**Build:** `m_image_stage_timed.exe` / `m_network_stage_timed.exe`
(`-DM_TIMING`); the accepted hosts are unchanged.

## 1. Question

Can the executed frame be timed on the GPU with a documented, reproducible
method, without confusing CPU dispatch time with GPU execution time?

## 2. Environment

| Item | Value | Source |
|---|---|---|
| GPU | AMD Radeon RX 7700 XT (gfx1101), 12 GB class | [M] adapter enumeration |
| Driver | `32.0.32015.7029`, 2026-10-17 | [M] `Win32_VideoController` |
| OS | Windows 11 Pro 10.0.26200, x64 | [M] `Win32_OperatingSystem` |
| CPU | AMD Ryzen 7 5700X3D, 8C/16T | [M] `Win32_Processor` |
| API | D3D12 compute, feature level 12_0 | [S] `m_*_stage.cpp` |
| Shader compiler | DXC 1.8.2502.11; SM 6.2 / SM 6.4 (`dot4`) | [M] `dxc --version`, CMake |
| C++ compiler | MSVC 14.44.35229.0 | [M] `cl.exe` product version |
| Build | Release; no debug layer for timing runs | [S] `CMakeLists.txt` |
| Display | 2560×1080 desktop (work is offscreen 1920×1080) | [M] |
| Workload | full B6 frame: enc0 → ten layers → dec0 → post → downsample | [S] |
| Render / output | 960×540 / 1920×1080, RGBA16F | [S] |
| GPU clock / power state | **not measured** (`[U]`); no vendor clock/power tool used | — |
| Hardware counters | **not available** — no RGP / rocprof / PresentMon on this machine | [M] `Get-Command` |

[M] The timestamp frequency reported by `ID3D12CommandQueue::GetTimestampFrequency`
is **100,000,000 Hz** (10 ns ticks) in every process, so all processes share one
clock domain and their deltas are comparable.

## 3. Instrumentation

`src/native/m_timing.h` implements a small D3D12 timer:

* one `D3D12_QUERY_HEAP_TYPE_TIMESTAMP` heap per process (capacity 32 for the
  image stages, `stages + 1` for the graph);
* `EndQuery(TIMESTAMP)` immediately before and after **every** `Dispatch`;
* one `ResolveQueryData` into a READBACK buffer before the command list closes;
* JSON output `{frequency, dispatches:[{index,label,begin,end,ticks,ms}]}`.

The source is compiled only into the `*_timed` targets. The accepted
`m_image_stage.exe` and `m_network_stage.exe` are built from the same sources
without `M_TIMING`, and the timed binaries reproduce the accepted frame hashes
(§4), so instrumentation does not change arithmetic.

## 4. Methodology

1. One unmeasured warm-up frame per mode.
2. Five measured repeats per mode, each a fresh process set.
3. Every repeat's final frame SHA-256 must equal the accepted M09 hash for that
   mode (`dc97c4cb…` FP16, `f76ac032…` INT8); a mismatch aborts the run.
4. Per-dispatch time is the timestamp delta in the same command list.
5. Process totals are the sum of their dispatch deltas; frame total is the sum
   of the five process totals.
6. Medians, p95, min and max are reported over repeats.

Each executable submits one command list and waits on one fence; the timestamps
measure GPU execution inside that submission, not Python or file I/O. Wall time
is recorded only as context.

```powershell
python scripts/m12_profile.py --out "$T/m12-profile-20261030" --repeats 5
```

## 5. Limits of the method

* Timestamps add an `EndQuery` per dispatch; they can slightly serialize or
  perturb the measured work. Both modes are measured with the same overhead, and
  the reported comparison is between those instrumented runs.
* No hardware counters: instruction utilization, cache behaviour, memory
  bandwidth and occupancy cannot be measured here. Any such statement in
  EXP-039 is explicitly inferential.
* GPU clock and power state are not controlled or recorded; run-to-run variance
  is reported instead of being assumed away.
* One device, one driver, one build, one resolution.
* Five repeats bound the variability but are not a statistical power study.

## 6. Interpretation

* [M] A GPU-side timing path exists, is reproducible, and does not alter the
  frame.
* [M] The measurement is per dispatch and per process, with a common 100 MHz
  clock, so the FP16 and INT8 totals are directly comparable under the same
  instrumented method.
* [U] Absolute hardware utilization and the effect of clock/power state.
