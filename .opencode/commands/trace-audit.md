
---
description: Independently audit repository consistency, agent executability, reproducibility, and unnecessary complexity
agent: plan
subtask: true
---

# Independent Repository Audit

Perform an adversarial, evidence-based audit of the current repository.

Additional focus, if provided: $ARGUMENTS

## Objective

Determine whether the repository is internally consistent, practical, reproducible, and sufficiently minimal for its intended purpose.

This is an audit, not an implementation task.

**Do not modify files.**

## 1. Reconnaissance

Inspect the actual repository before making judgments.

Read relevant files, including when present:

- README.md
- AGENTS.md
- docs/MILESTONES.md
- docs/CONVENTIONS.md
- references/README.md
- .gitignore
- Project-specific documentation
- Relevant source code, tests, and scripts

Inspect the repository structure and Git state.

Follow references between documents.

Do not assume that files, workflows, or conventions exist.

Adapt the audit to the actual project.

## 2. Audit Categories

### A. Consistency

Identify:

- Contradictory instructions.
- Duplicated responsibilities.
- Conflicting definitions.
- Ambiguous terminology.
- Inconsistent completion criteria.
- References to nonexistent files or workflows.
- Documentation that contradicts implementation.

### B. Agent Executability

Determine whether an autonomous coding agent can follow the instructions without guessing.

Evaluate:

- Session initialization.
- Task and milestone selection.
- Evidence collection.
- Experiment execution.
- Implementation and reconstruction.
- Validation.
- Documentation updates.
- Filesystem restrictions.
- Session recovery.

Identify instructions that are impossible, ambiguous, or unnecessarily restrictive.

### C. Minimalism

Identify:

- Redundant rules.
- Excessive documentation requirements.
- Unnecessary approval gates.
- Repeated information.
- Premature abstractions.
- Processes that add overhead without improving correctness.

Prefer removing complexity over introducing new mechanisms.

### D. Reproducibility

Evaluate:

- Source provenance and version pinning.
- Experiment reproducibility.
- Environment documentation.
- Binary provenance, when applicable.
- Independent validation.
- Negative result preservation.
- Regression testing.
- Session recovery.

Distinguish documented procedures from demonstrated reproducibility.

### E. Filesystem Discipline

Check whether the repository prevents:

- Arbitrary root-level files.
- Duplicate documentation.
- Disposable scripts becoming permanent clutter.
- Generated artifacts entering Git.
- Accidental reference modifications.
- Uncontrolled directory proliferation.

Ensure restrictions still permit legitimate project-specific architecture.

### F. Practical Scenarios

Simulate relevant workflows against the repository's actual instructions.

For reverse-engineering projects, consider:

1. Investigating an undocumented binary.
2. Reconstructing behavior from reference source code.
3. Running experiments requiring specialized hardware.
4. Investigating a failed hypothesis.
5. Resuming work after context loss.
6. Correcting a previously accepted finding.

For other projects, adapt scenarios to their actual purpose.

These are reasoning exercises, not executed tests.

Identify where instructions succeed, fail, or require undocumented decisions.

## 3. Severity

Classify findings:

- CRITICAL: Prevents reliable operation or permits fundamentally invalid results.
- HIGH: Significant contradiction, reproducibility failure, or workflow defect.
- MEDIUM: Ambiguity, avoidable overhead, or maintainability problem.
- LOW: Minor clarity or consistency improvement.

Only report actionable findings.

Every finding must have a plausible failure scenario.

Do not inflate severity.

## 4. Output

### Verdict

Provide:

- Overall assessment.
- Readiness: READY / READY WITH FIXES / NOT READY.
- Three most important findings.

### Findings

For each finding, provide:

- ID.
- Severity.
- File and line references.
- Exact problem.
- Concrete failure scenario.
- Smallest reasonable fix.

Distinguish confirmed defects from recommendations.

### Cross-Document Contradictions

List confirmed contradictions separately.

If none exist, explicitly say so.

### Redundancy and Bureaucracy

Identify rules or sections that should be simplified or removed.

### Scenario Results

Report PASS / PARTIAL / FAIL for each applicable scenario.

Explain the judgment briefly.

Do not present simulated scenarios as executed validation.

### Minimal Patch Plan

Propose the smallest set of changes necessary.

Prioritize:

1. Correctness.
2. Reproducibility.
3. Simplicity.
4. Maintainability.

Do not propose new documents, directories, frameworks, or governance mechanisms unless existing structures cannot solve the problem.

## 5. Constraints

- READ ONLY.
- No commits.
- No file modifications.
- No invented test results.
- No generic best-practice checklists.
- No speculative findings without plausible failure scenarios.
- Every finding must reference actual repository evidence.
- Distinguish confirmed defects from recommendations.
- Do not praise the project without evidence.
- Prefer deleting or simplifying rules over adding new ones.
- Do not repeat the same finding under multiple categories.

Be skeptical, precise, and economical.
