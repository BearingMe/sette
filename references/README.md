# References

This directory contains external material used to investigate, reconstruct, and validate the target system. References should be **traceable, versioned, and reproducible** whenever practical.

## Structure

- `sources/`: External source repositories, preferably pinned to specific commits.
- `README.md`: Index of references, their purpose, origin, and version.

Do not create additional reference indexes without a concrete need.

## Reference Index

The entries below distinguish acquired source repositories from pending sources and a local binary. d4r, quattro, and NVIDIA DLSS are pinned submodules, but their source has not yet been inspected. The DLSS checkout is shallow and sparse. TransformerEngine is not yet acquired. The local `nvngx_dlss.dll` is documented separately, but its acquisition provenance and relationship to Preset M are unknown. References are evidence candidates, not automatically established oracles. Continue M02 by inspecting sources and documenting gaps; do not silently track upstream `HEAD`.

| Reference | Purpose | Origin | Version / Commit / Hash | Local Path / Status |
|---|---|---|---|---|
| d4r | Primary implementation oracle candidate: Preset M reconstruction, conservative FP16 WMMA path, weights, layouts, operators, intermediate validation, and standalone/temporal execution. Optimized paths are secondary performance references. | <https://github.com/countervolts/d4r> | `05e19d402781083118d106f2a21d0cf2bc43f744` (`v0.1.2-200-g05e19d4`) | `sources/d4r` submodule; pinned, not yet inspected. |
| quattro | RDNA 3 research candidate: gfx1101 INT8 WMMA/DP4A, instruction lowering, quantization, compilation, ISA inspection, validation, and measurement. | <https://github.com/BearingMe/quattro> | `ad0482a952b1af80b8be2dcaa6a49e098aa57fc1` (main at acquisition) | `sources/quattro` submodule; pinned, not yet inspected. |
| NVIDIA DLSS | Official project material for product/API context. Its availability must not be interpreted as access to Preset M implementation, model, or weights. | <https://github.com/NVIDIA/DLSS> | `9981c6350dc58c0ce21f024fd65e31e851c31689` (`v310.9.16`) | `sources/dlss` submodule; shallow sparse checkout, source inspection pending. |
| NVIDIA TransformerEngine | Candidate reference for NVIDIA Transformer Engine numerical/FP8 context only; relevance to Preset M must be established from source. | <https://github.com/NVIDIA/TransformerEngine> | Unpinned; acquisition pending. | Not acquired. |
| Local NVIDIA DLL | Local binary metadata and identity; not established as the Preset M model or weights. | Original source URL and acquisition provenance unknown; embedded metadata identifies NVIDIA. | SHA-256 `3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983`; raw file/product version `310,9,1,0`. | [`binaries/README.md`](binaries/README.md); DLL is local-only and gitignored. |
| AMD GPUOpen | Official AMD documentation discovery point for relevant D3D12, RDNA 3, WMMA, and shader-intrinsic specifications. Specific documents and revisions remain to be identified. | <https://gpuopen.com/> | Specific pages/revisions pending M02. | Not acquired; applicable documentation not yet selected. |
| DirectX Shader Compiler | Candidate primary source for DXC versions, HLSL support, and compiler behavior relevant to the target. | <https://github.com/microsoft/DirectXShaderCompiler> | Unpinned; compatible release/commit pending M02. | Not acquired; DXC is not found on the current PATH. |

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

d4r and quattro are pinned as read-only submodules. Their source paths and relevant code have not yet been inspected; acquisition of the other listed repositories remains pending or incomplete.

## Binary References

For binary artifacts, record:

- Original filename and source.
- Version or build identifier.
- SHA-256 hash when available.
- Target platform and architecture.
- Relevant symbols, offsets, or signatures.

Do not commit large or proprietary binaries without a justified need and appropriate rights. The local DLL's metadata and hash are recorded in [`binaries/README.md`](binaries/README.md); its acquisition source and redistribution rights remain unknown. No Preset M model/weight artifact has been identified.

## External Documentation

For documentation that cannot reasonably be stored locally, record:

- Source URL.
- Relevant document version or revision.
- Access date for mutable resources.
- Relevant sections or identifiers.

Preserve snapshots only when necessary and permitted. Specific AMD GPUOpen and shader-intrinsic documents have not yet been selected.

## Validation

**A reference is evidence, not automatically an oracle.** Distinguish documented or intended behavior, behavior observed in reference source code, behavior experimentally observed on the target, and hypotheses inferred from incomplete evidence.

Use the evidence classifications defined in [`docs/CONVENTIONS.md`](../docs/CONVENTIONS.md). Validation must remain independent of the implementation being tested.

## Maintenance

Update this index when references are added, replaced, or removed. When upgrading a reference, record the new revision and assess whether existing findings or experiments need revalidation. Do not automatically update references to their latest versions.
