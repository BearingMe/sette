# References

This directory contains external material used to investigate, reconstruct, and validate the target system. References should be **traceable, versioned, and reproducible** whenever practical.

## Structure

- `sources/`: External source repositories, preferably pinned to specific commits.
- `README.md`: Index of references, their purpose, origin, and version.

Do not create additional reference indexes without a concrete need.

## Reference Index

The primary M02 references are pinned: d4r, quattro, NVIDIA DLSS, TransformerEngine, and the local DLL. All four source checkouts were inspected on 2026-10-28; see [`docs/findings/2026-10-28-reference-inventory.md`](../docs/findings/2026-10-28-reference-inventory.md). The DLSS and TransformerEngine checkouts are shallow and sparse, and the DLSS checkout's parent submodule configuration ignores worktree dirt, so inspect it directly before relying on a clean status. References are evidence candidates, not automatically established oracles. Do not silently track upstream `HEAD`.

| Reference | Purpose | Origin | Version / Commit / Hash | Local Path / Status |
|---|---|---|---|---|
| d4r | Primary implementation oracle candidate: Preset M stage algorithms, FP8/f16 WMMA kernels, weight-extraction tools, and standalone execution. Optimized paths are secondary performance references. | <https://github.com/countervolts/d4r> | `05e19d402781083118d106f2a21d0cf2bc43f744` (`v0.1.2-200-g05e19d4`) | `sources/d4r` submodule; inspected. M weights are extracted locally from a user DLL and are not in the repository. Linux/Vulkan/HIP execution path, not yet built here. |
| quattro | gfx1101 D3D12 INT8 WMMA/DP4A and FP16 WMMA, instruction lowering, quantization, ISA inspection, and validated numerical experiments. | <https://github.com/BearingMe/quattro> | `ad0482a952b1af80b8be2dcaa6a49e098aa57fc1` (main at acquisition) | `sources/quattro` submodule; inspected. Validated experiments use the same GPU model and driver as this workstation. Nested `.references/sources/*` submodules are uninitialized. FSR 4.1 model only. |
| NVIDIA DLSS | Official product/API material. Availability must not be interpreted as access to Preset M implementation, model, or weights. | <https://github.com/NVIDIA/DLSS> | `9981c6350dc58c0ce21f024fd65e31e851c31689` (`v310.9.16`) | `sources/dlss` submodule; shallow sparse `include/` only. NGX headers and license; no model, weights, or inference runner. |
| NVIDIA TransformerEngine | NVIDIA FP8/FP4 numerical-recipe context only; relevance to Preset M must be established from separate evidence. | <https://github.com/NVIDIA/TransformerEngine> | `dfc4158d1597f72972fe2863adac436fa3fe1d4d` (main at acquisition, 2026-10-28) | `sources/transformerengine` submodule; shallow sparse `transformer_engine/common/{recipe,cast,include}`. CUDA-only; no DLSS or AMD material. |
| Local NVIDIA DLL | Local binary and Preset M code evidence; not an extracted or validated weight set. | Original source URL and acquisition provenance unknown; embedded metadata identifies NVIDIA. | SHA-256 `3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983`; raw file/product version `310,9,1,0`. | [`binaries/README.md`](binaries/README.md); DLL is local-only and gitignored. Contains `rrlite_*` PTX; weight compatibility with d4r tooling unknown. |
| AMD GPUOpen | Official AMD documentation discovery point for RDNA 3, D3D12, WMMA, and shader-intrinsic specifications. | <https://gpuopen.com/> | Specific documents/revisions not yet selected. | Not acquired. The RDNA 3 ISA reference is cited by quattro EXP-004 but no local copy is held. |
| DirectX Shader Compiler | Compiler behavior and shader-model support for the target. | Windows SDK 10.0.26100.0 | DXC `1.8.2502.11` at `Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe` | Present locally; not on `PATH`. Newer standalone DXC for SM 6.9 is not installed. |

## Rules

1. **Prefer primary sources.** Official documentation, original implementations, binaries, and directly observed behavior take priority over secondary descriptions.
2. **Inspect actual implementations.** Do not rely exclusively on README files or project descriptions.
3. **Pin versions.** Record commits, releases, build identifiers, or hashes whenever possible.
4. **Preserve provenance.** Identify where material originated and how it was obtained.
5. **Respect compatibility.** Record relevant platform, hardware, driver, API, and version constraints.
6. **Avoid modifying references.** Keep upstream material intact. Place experimental changes in project code or an explicitly identified fork.
7. **Avoid unnecessary duplication.** Link to existing material rather than copying large excerpts.
8. **Record limitations.** Identify missing, inaccessible, incomplete, or potentially incompatible references.

## Source Repositories

Prefer Git submodules for external source repositories when appropriate. Submodules must reference known commits rather than relying on the latest upstream state. Keep them read-only and unmodified.

Clone the project with its references:

```bash
git clone --recurse-submodules <repository-url>
```

Initialize references in an existing checkout:

```bash
git submodule update --init --recursive
```

All four source repositories are pinned as read-only submodules and were inspected for M02. The DLSS and TransformerEngine checkouts are shallow and sparse; their omitted subsets are intentional. quattro's nested reference submodules are uninitialized. Do not modify pinned sources; place experiments in project code.

## Binary References

For binary artifacts, record:

- Original filename and source.
- Version or build identifier.
- SHA-256 hash when available.
- Target platform and architecture.
- Relevant symbols, offsets, or signatures.

Do not commit large or proprietary binaries without a justified need and appropriate rights. The local DLL's metadata, hash, and Preset M PTX inventory are recorded in [`binaries/README.md`](binaries/README.md); its acquisition source and redistribution rights remain unknown. No extracted or validated Preset M weight set exists in the repository.

## External Documentation

For documentation that cannot reasonably be stored locally, record:

- Source URL.
- Relevant document version or revision.
- Access date for mutable resources.
- Relevant sections or identifiers.

Preserve snapshots only when necessary and permitted. AMD GPUOpen and DXC documentation remain candidates; quattro's vendored AMD headers and its recorded experiments are the current pinned intrinsic evidence.

## Validation

**A reference is evidence, not automatically an oracle.** Distinguish documented or intended behavior, behavior observed in reference source code, behavior experimentally observed on the target, and hypotheses inferred from incomplete evidence.

Use the evidence classifications defined in [`docs/CONVENTIONS.md`](../docs/CONVENTIONS.md). Validation must remain independent of the implementation being tested.

## Maintenance

Update this index when references are added, replaced, or removed. When upgrading a reference, record the new revision and assess whether existing findings or experiments need revalidation. Do not automatically update references to their latest versions.
