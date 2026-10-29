# Milestones

## Overview

This document is the single source of truth for project progress.

- Milestones define **what** must be accomplished.
- Checkpoints define **what can be independently verified**.
- Validation gates determine whether a milestone is complete.
- OpenCode TODOs track individual execution tasks.

Do not duplicate milestone status in other documents.

## Progress

| ID | Milestone | Status | Validation |
|---|---|---|---|
| M01 | Project initialization | NOT STARTED | PENDING |
| M02 | Reference collection | NOT STARTED | PENDING |
| M03 | Reconnaissance | NOT STARTED | PENDING |
| M04 | Experimental probing | NOT STARTED | PENDING |
| M05 | Reconstruction | NOT STARTED | PENDING |
| M06 | Differential validation | NOT STARTED | PENDING |
| M07 | Integration | NOT STARTED | PENDING |
| M08 | Final quality gate | NOT STARTED | PENDING |

### Status

- `NOT STARTED`: Work has not begun.
- `IN PROGRESS`: Work is underway.
- `COMPLETE`: Validation gate passed.
- `BLOCKED`: A required dependency is unavailable.

### Validation

- `PENDING`: Validation has not been performed.
- `PASS`: Acceptance criteria satisfied with evidence.
- `FAIL`: Acceptance criteria not satisfied.
- `BLOCKED`: Validation cannot proceed.
- `INCONCLUSIVE`: Evidence is insufficient.

A milestone may be marked `COMPLETE` only when its
validation gate is `PASS`.

A failed experiment can be a successful research result.
Its milestone passes when the objective was to investigate
that behavior and the findings satisfy the acceptance criteria.

---

## M01 — Project Initialization

### Objective

Establish the project's scope, constraints, and reproducible
working environment.

### Checkpoints

- [ ] C01: Define the research objective.
- [ ] C02: Identify the target system and versions.
- [ ] C03: Document environmental requirements.
- [ ] C04: Establish the initial project structure.

### Validation Gate

- [ ] Scope and exclusions are explicit.
- [ ] Environment requirements are documented.
- [ ] Initial setup is reproducible.

### Deliverables

- `README.md`
- `AGENTS.md`

---

## M02 — Reference Collection

### Objective

Acquire and organize authoritative reference material.

### Checkpoints

- [ ] C01: Identify primary sources.
- [ ] C02: Acquire relevant references.
- [ ] C03: Record versions, commits, and provenance.
- [ ] C04: Identify missing or inaccessible sources.

### Validation Gate

- [ ] References are accessible or limitations documented.
- [ ] External source revisions are identifiable.
- [ ] Reference purposes are documented.
- [ ] Known gaps are recorded.

### Deliverables

- `references/README.md`
- `references/sources/`

---

## M03 — Reconnaissance

### Objective

Understand the target system sufficiently to identify its
components, behavior, dependencies, and unknowns.

### Checkpoints

- [ ] C01: Inventory relevant components.
- [ ] C02: Identify system boundaries and dependencies.
- [ ] C03: Trace important execution paths.
- [ ] C04: Classify findings and unresolved questions.

### Validation Gate

- [ ] Important claims have traceable evidence.
- [ ] Observations and hypotheses are distinguished.
- [ ] Major unknowns are explicitly recorded.
- [ ] Findings identify useful targets for probing.

### Deliverables

- `docs/findings/`

---

## M04 — Experimental Probing

### Objective

Resolve important unknowns through reproducible experiments.

### Checkpoints

- [ ] C01: Identify testable hypotheses.
- [ ] C02: Define experiment inputs and expected observations.
- [ ] C03: Execute experiments.
- [ ] C04: Record results, failures, and limitations.
- [ ] C05: Consolidate supported conclusions.

### Validation Gate

- [ ] Experiments have reproducible procedures.
- [ ] Actual results are recorded.
- [ ] Failed experiments are preserved.
- [ ] Conclusions are supported by evidence.
- [ ] Remaining unknowns are documented.

### Deliverables

- `docs/experiments/`
- `docs/findings/`
- `scripts/` or `tests/` when reusable code is needed

---

## M05 — Reconstruction

### Objective

Implement the smallest meaningful reconstruction of the
target behavior.

### Checkpoints

- [ ] C01: Select a bounded reconstruction target.
- [ ] C02: Establish independently derived expectations.
- [ ] C03: Implement the required behavior.
- [ ] C04: Execute initial validation.
- [ ] C05: Document known deviations.

### Validation Gate

- [ ] Implementation executes successfully.
- [ ] Required behavior is demonstrated.
- [ ] Tests do not derive expected results from the
      implementation under test.
- [ ] Unsupported behavior is not presented as faithful.
- [ ] Known limitations are documented.

### Deliverables

- `src/`
- `tests/`

---

## M06 — Differential Validation

### Objective

Compare the reconstruction against an independent oracle
or authoritative behavioral evidence.

### Checkpoints

- [ ] C01: Identify the comparison oracle.
- [ ] C02: Establish representative test cases.
- [ ] C03: Compare outputs or observable behavior.
- [ ] C04: Investigate discrepancies.
- [ ] C05: Record fidelity limitations.

### Validation Gate

- [ ] Comparisons use independent expectations.
- [ ] Test inputs and environments are recorded.
- [ ] Differences are measured or characterized.
- [ ] Fidelity claims match the available evidence.

### Deliverables

- `tests/`
- `docs/experiments/`
- `docs/findings/`

---

## M07 — Integration

### Objective

Demonstrate the reconstructed behavior in its intended
execution environment.

### Checkpoints

- [ ] C01: Identify integration requirements.
- [ ] C02: Implement the minimum integration path.
- [ ] C03: Execute an end-to-end scenario.
- [ ] C04: Record runtime behavior and limitations.

### Validation Gate

- [ ] Integration executes in the intended environment.
- [ ] At least one representative end-to-end scenario passes.
- [ ] Runtime evidence is recorded.
- [ ] Unsupported integration claims are excluded.

### Deliverables

- `src/`
- `tests/`
- `docs/experiments/`

---

## M08 — Final Quality Gate

### Objective

Evaluate the reconstruction's correctness, reproducibility,
and documented limitations.

### Checkpoints

- [ ] C01: Review milestone evidence.
- [ ] C02: Execute applicable regression tests.
- [ ] C03: Check documentation against implementation.
- [ ] C04: Identify unresolved defects and limitations.
- [ ] C05: Consolidate final findings.

### Validation Gate

- [ ] Required tests pass.
- [ ] Important claims have supporting evidence.
- [ ] Known failures and limitations remain visible.
- [ ] Documentation reflects actual implementation status.
- [ ] Final deliverables are reproducible.

### Deliverables

- `docs/findings/`
- Final implementation and validation artifacts
