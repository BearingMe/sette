# Local DLSS DLL

This directory holds a local-only binary reference. `nvngx_dlss.dll` is excluded from Git by the root `.gitignore`; do not commit or redistribute it unless its provenance, license, and redistribution rights are established.

## Observed artifact

| Field | Observation |
|---|---|
| Filename | `nvngx_dlss.dll` |
| Size | 58,956,912 bytes |
| SHA-256 | `3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983` |
| Format / architecture | PE32+ / x86-64, machine `0x8664` |
| File description | `NVIDIA DLSS - DVS PRODUCTION` |
| Product | `NVIDIA Deep Learning SuperSampling` |
| File version (raw version-resource value) | `310,9,1,0` |
| Product version (raw version-resource value) | `310,9,1,0` |
| Original filename (version resource) | `CL 38868064` |
| Internal name | `DLSS` |
| Company / copyright | `NVIDIA` / `Copyright 2018 - 2024 NVIDIA CORP` |
| Authenticode | PowerShell reported `Valid` on 2026-10-28; signer `NVIDIA Corporation`, thumbprint `7B7B0B6697AFB438CF6F65A155F00E86676FB186` |

## Evidence and limitations

- `[M]` File size and SHA-256 were measured locally on 2026-10-28 with `Get-Item` and `Get-FileHash -Algorithm SHA256`.
- `[R]` Static PE inspection on 2026-10-28 identified PE32+, x86-64, and one `RT_VERSION` resource (type ID 16, resource ID 1, language ID 1033). The resource inventory was complete; the PE parser did not decode the version-resource payload.
- `[M]` Windows `FileVersionInfo` supplied the raw version strings above. `Get-AuthenticodeSignature` reported a valid signature from NVIDIA Corporation on the same workstation/date. These are local observations, not proof of the download source or a particular release.
- Reproduce the hash, version metadata, and signature checks in PowerShell:

  ```powershell
  Get-FileHash references/binaries/nvngx_dlss.dll -Algorithm SHA256
  [System.Diagnostics.FileVersionInfo]::GetVersionInfo((Resolve-Path 'references/binaries/nvngx_dlss.dll')) |
    Format-List FileDescription, CompanyName, ProductName, ProductVersion, FileVersion, OriginalFilename, InternalName, LegalCopyright
  Get-AuthenticodeSignature references/binaries/nvngx_dlss.dll
  ```

- `[R]` Extracting embedded CUDA fatbins with d4r's `kernels/tools/extract_dlss_ptx.py` produced 169 PTX modules on 2026-10-28. Three modules contain 35 unique Preset M (`rrlite_*`) entry points, covering the folded/unfolded input, expansion, network, reconstruction, and downsample stages. This establishes that the DLL contains Preset M code, not that this checkout has compatible weights or that the code executes here.
- `[S]` d4r's pinned release tooling targets a different Super Resolution DLL: 310.7.0, SHA-256 `be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e`. Compatibility of this 310.9.1 DLL with d4r's extractor and model package format is unknown.
- Reproduce the PTX inventory; the extracted proprietary files stay in the temporary directory and are not tracked:

  ```powershell
  $out = Join-Path $env:TEMP 'sette-dlss-ptx'
  python references/sources/d4r/kernels/tools/extract_dlss_ptx.py references/binaries/nvngx_dlss.dll $out
  Select-String -Path (Join-Path $out '*.ptx') -Pattern '\.entry\s+(rrlite_[A-Za-z0-9_]+)' -AllMatches |
    ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
  ```

- **Unknown:** Original download/source URL, acquisition provenance, exact package/release, intended target application, and license terms. The file-version values are DLL metadata. Weight location, layout compatibility, and runtime behavior were not verified.
