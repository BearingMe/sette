
# References

This directory contains external material used to investigate, reconstruct, and validate the target system.

References should be **traceable, versioned, and reproducible** whenever practical.

## Structure

- `sources/`: External source repositories, preferably pinned to specific commits.
- `README.md`: Index of references, their purpose, origin, and version.

Do not create additional reference indexes without a concrete need.

## Reference Index

Add entries as references are introduced.

| Reference | Purpose | Origin | Version / Commit / Hash | Local Path |
|---|---|---|---|---|
| | | | | |

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

Prefer Git submodules for external source repositories when appropriate.

Clone the project with its references:

```bash
git clone --recurse-submodules <repository-url>
```

Initialize references in an existing checkout:

```bash
git submodule update --init --recursive
```

Submodules should reference known commits rather than relying on the latest upstream state.

## Binary References

For binary artifacts, record:

- Original filename and source.
- Version or build identifier.
- SHA-256 hash when available.
- Target platform and architecture.
- Relevant symbols, offsets, or signatures.

Do not commit large or proprietary binaries without a justified need and appropriate rights.

## External Documentation

For documentation that cannot reasonably be stored locally, record:

- Source URL.
- Relevant document version or revision.
- Access date for mutable resources.
- Relevant sections or identifiers.

Preserve snapshots only when necessary and permitted.

## Validation

**A reference is evidence, not automatically an oracle.**

Distinguish:

- Documented or intended behavior.
- Behavior observed in reference source code.
- Behavior experimentally observed on the target.
- Hypotheses inferred from incomplete evidence.

Use the evidence classifications defined in `docs/CONVENTIONS.md`.

Validation must remain independent of the implementation being tested.

## Maintenance

Update this index when references are added, replaced, or removed.

When upgrading a reference, record the new revision and assess whether existing findings or experiments need revalidation.

Do not automatically update references to their latest versions.
