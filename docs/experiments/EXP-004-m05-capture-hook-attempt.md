# EXP-004: Probe-Side Capture Hook Attempt

**Milestone:** M05 (C10)
**Status:** FAIL (negative result; C10 remains open)

## Question

Can the missing `enc0`/`dec0`/post/downsample launches be captured with a
MinHook DLL loaded into the existing D3D12 probe, without rebuilding ZLUDA?

## Method

- Built `capture_hook.dll` with MinHook 1.3.4 and llvm-mingw.
- Patched the temporary probe build to load it through `D4R_CAPTURE_HOOK`.
- Hooked the ZLUDA exports `cuLaunchKernel`, `cuLaunchKernelEx`,
  `cuGraphAddKernelNode_v2` (fallback `cuGraphAddKernelNode`), and
  `cuGraphExecKernelNodeSetParams`.
- Logged every call (name, grid/block, parameter count/size, pointer fields) to
  `D4R_CAPTURE_LOG`; ran preset 13 with the proven probe configuration.

## Results

- [M] The hook DLL loaded and installed all hooks successfully:
  `launch=... ex=... graphAdd=... graphSet=...`.
- [M] The probe still completed preset 13 (`PASS D3D12_DLSS`) and the existing
  ZLUDA replay hook still produced the ten layer captures.
- [M] The hook observed **zero** launch or graph-node calls, including the ten
  layer kernels that are known to execute and are captured by ZLUDA.
- [S] ZLUDA's own replay hook therefore intercepts an internal path (the graph
  execution path inside `nvcuda.dll`), not the exported driver entry points that
  an external DLL can hook.

## Interpretation

- [M] A probe-side export hook cannot extend capture coverage. Extending the
  ZLUDA replay hook itself is required, confirming the preference in the plan.
- [S] The required ZLUDA whitelist additions and their PTX parameter sizes are
  known: `enc0` folded 360 bytes, `dec0` folded 168, post 248, downsample 96
  ([coverage audit](../findings/2026-10-29-m05-capture-coverage.md)).
- [U] The ZLUDA rebuild was not attempted: the workspace has ROCm 7.2 clang but
  no patched LLVM development build, which `build-zluda-llvm.ps1` produces from
  the pinned LLVM source. This is the remaining C10 cost.

## Next

1. Clone ZLUDA `v7-preview.11` and apply d4r's `patches/zluda` (including
   `d4r_replay.rs`).
2. Add the four missing `parameter_size` groups with the sizes above.
3. Build the pinned patched LLVM, then ZLUDA (`build-zluda-llvm.ps1`,
   `build-zluda-windows.ps1`).
4. Re-run the probe with the replacement `nvcuda.dll` and confirm `enc0`/`dec0`
   replay directories appear; add post/downsample coverage next.
5. Keep the existing ten-layer captures and `replay_m_layers.py` as the
   regression baseline.

## Files

- Prototype hook and patched probe existed only in the local temporary build;
  they are not committed because they do not capture anything.
