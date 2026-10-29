
# AGENTS.md

## Purpose

This repository uses **Trace**, a lightweight reverse-engineering harness.

The objective is to investigate systems, preserve reliable knowledge,
reconstruct behavior when appropriate, and validate claims against
independent evidence.

Prioritize correctness, reproducibility, and simplicity.

Do not confuse activity with progress.

## 1. Authority

The repository is the authoritative source of project state.

| Source | Responsibility |
|---|---|
| `AGENTS.md` | Agent behavior, workflow, and restrictions |
| `README.md` | Project scope, setup, and entry points |
| `docs/MILESTONES.md` | Progress, checkpoints, and validation gates |
| `docs/CONVENTIONS.md` | Evidence, experiments, findings, and validation conventions |
| `docs/experiments/` | Experimental procedures and observations |
| `docs/findings/` | Consolidated technical knowledge |
| `references/` | External reference material |
| `src/` | Implementation |
| `tests/` | Automated tests and regression checks |
| `scripts/` | Reusable research and development utilities |

Git records implementation history.

OpenCode TODOs track active execution tasks.

External memory systems may assist retrieval, but must never
override repository evidence.

Do not maintain competing sources of truth.

## 2. Filesystem Discipline

Use the established project structure.

### Rules

1. Do not create arbitrary files or directories at the repository root.
2. Do not introduce competing directories for existing responsibilities.
3. Do not create placeholder files, abstractions, or directories speculatively.
4. Keep implementation, tests, scripts, documentation, and references separate.
5. Keep generated artifacts out of tracked source directories.
6. Prefer existing project organization over introducing new structural layers.
7. Do not reorganize unrelated files while completing a milestone.
8. New top-level directories require an explicit technical justification.

### Temporary Artifacts

Disposable scripts, logs, dumps, traces, intermediate binaries,
and generated outputs belong in ignored temporary or build locations.

Do not commit disposable artifacts.

Promote useful scripts into:

- `scripts/` for reusable investigation utilities.
- `tests/` for automated validation and regression.

Retain small test fixtures when required for reproducibility.

### Project-Specific Architecture

Project-specific structure may be defined in the README or
dedicated architecture documentation when necessary.

Follow established language and ecosystem conventions.

Do not impose a universal implementation architecture.

## 3. Starting a Session

Before making changes:

1. Read `README.md` to understand the project objective.
2. Read `AGENTS.md` and relevant conventions.
3. Inspect the active milestone and its validation gate.
4. Run `git status`.
5. Inspect relevant implementation, tests, findings, and references.
6. Retrieve external memory only when it can answer a specific question.

For subsequent sessions, use targeted reading rather than
reloading the entire repository.

Identify:

- The current objective.
- What is already established.
- What remains unknown.
- What evidence is required.
- The next independently verifiable action.

Do not restart completed investigations without a reason.

## 4. Milestone Execution

`docs/MILESTONES.md` owns project progress.

Milestones define outcomes, not lists of arbitrary coding tasks.

Checkpoints must be independently verifiable.

### Workflow

1. Select the active milestone.
2. Identify its acceptance criteria and dependencies.
3. Inspect existing evidence and relevant references.
4. Plan the smallest useful investigation or implementation step.
5. Execute the work.
6. Validate the result.
7. Record meaningful evidence.
8. Update milestone progress when justified.

Use OpenCode TODOs for short-lived execution tasks.

Do not duplicate TODO lists in documentation.

Do not expand scope simply because related work is available.

Do not advance to a new milestone without user authorization,
unless the user explicitly delegates milestone progression.

### Blockers

When blocked:

- Identify the missing dependency or evidence.
- Record what was attempted.
- Preserve useful partial results.
- State what is required to resume.

Do not invent substitute evidence to bypass a blocker.

## 5. Research and References

Prefer direct inspection of the target system.

Consult existing local references before repeating external research.

When investigating source code:

- Inspect actual implementations, not just README descriptions.
- Record relevant versions, commits, symbols, and file paths.
- Distinguish observed behavior from intended behavior.
- Avoid assuming the current upstream version matches the target.

When investigating binaries:

- Record relevant build identifiers and hashes.
- Preserve meaningful offsets, symbols, and signatures.
- Distinguish static analysis from observed runtime behavior.

Use the evidence classifications defined in
`docs/CONVENTIONS.md`.

Never present a hypothesis as an established fact.

Do not fabricate citations, measurements, test results,
source observations, or runtime behavior.

## 6. Experiments

Experiments must answer a concrete question.

Before execution, identify:

- The question or hypothesis.
- Relevant inputs and environment.
- The expected observation.
- The validation method or oracle.

After execution:

- Record actual observations.
- Preserve meaningful failures and negative results.
- Identify supported conclusions.
- Record important remaining uncertainties.

Use `docs/experiments/` for durable experiment reports.

Use `docs/findings/` for consolidated knowledge.

Do not create a new document for every trivial probe.

Do not silently overwrite historical experimental results.
Append corrections or addenda when necessary.

### Execution

Prefer existing commands and test infrastructure.

Use a documented, reproducible execution path whenever possible.

Do not create a new runner, framework, or script architecture
for every experiment.

One-off exploratory commands may remain temporary.

Repeatable probes and regression checks should be durable.

## 7. Reconstruction

Reconstruct only behavior supported by evidence or
explicitly identified assumptions.

Prefer the smallest implementation that can be validated.

Do not invent behavior to fill unknown gaps.

When approximations are necessary:

- State what is approximated.
- Explain why.
- Identify expected deviations.
- Avoid claiming exact fidelity.

Use established project conventions.

Prefer simple code over unnecessary abstraction.

Avoid speculative extensibility and premature optimization.

Optimize only against a defined objective and measurable baseline.

## 8. Validation

Validation must match the claim.

| Claim | Required Evidence |
|---|---|
| Documentation or source fact | Traceable primary-source evidence |
| Binary behavior | Reproducible analysis or runtime observation |
| Algorithm correctness | Independent expected results or oracle comparison |
| Performance improvement | Controlled measurements against a baseline |
| Integration correctness | Execution in the intended environment |
| Reconstruction fidelity | Comparison against independent reference behavior |

### Rules

1. Never derive the oracle solely from the implementation under test.
2. Compilation is not proof of runtime correctness.
3. Passing unit tests is not proof of end-to-end integration.
4. Synthetic benchmarks are not proof of production performance.
5. Tests must exercise the behavior being claimed.
6. Preserve relevant failures and discrepancies.
7. Never report tests as executed unless they actually ran.
8. Do not mark a validation gate as passed without supporting evidence.

If required hardware, software, or external access is unavailable,
report the limitation and mark validation accordingly.

A negative finding may satisfy a research objective.

It does not automatically satisfy an implementation objective.

## 9. Definition of Done

A milestone is complete only when:

- Its objective has been achieved.
- Required checkpoints are satisfied.
- Its validation gate passes.
- Supporting evidence is recorded and traceable.
- Relevant tests or experiments are reproducible.
- Known limitations are documented.
- Milestone status reflects the actual result.

Do not require unrelated cleanup or speculative improvements.

Do not mark partially validated behavior as complete.

### Independent Review

Independent or adversarial review is recommended for:

- High-risk architectural decisions.
- Security-sensitive reconstruction.
- Major fidelity claims.
- Irreversible changes.
- Final release or quality gates.

Review is risk-based, not mandatory for every milestone.

Review findings must be actionable and supported by evidence.

Do not introduce repeated review cycles without new findings.

## 10. Documentation and Memory

Documentation should preserve knowledge, not narrate activity.

### Rules

- Keep documents concise and navigable.
- Link to existing evidence instead of duplicating it.
- Update existing documents before creating new ones.
- Avoid session summaries unless recovery genuinely requires them.
- Do not create documentation solely to demonstrate work.
- Do not maintain duplicate progress dashboards.

### External Memory

AgentMemory may store concise durable facts and pointers.

MemPalace may index repository knowledge for retrieval.

Neither is required for project correctness or session recovery.

Do not write memory entries merely to demonstrate memory usage.

Do not copy entire repository documents into memory.

Prefer storing references to authoritative files.

When memory conflicts with repository evidence,
the repository wins.

### Session Recovery

Prefer recovering state from:

1. `docs/MILESTONES.md`
2. Git status and history
3. Relevant experiments and findings
4. Existing implementation and tests

Create a standalone session checkpoint only when important
unfinished context cannot be recovered efficiently from these sources.

Do not checkpoint every routine action.

## 11. Change Discipline

Keep changes focused on the active objective.

Before finishing:

1. Review the diff.
2. Run relevant validation.
3. Remove accidental generated artifacts.
4. Check for unrelated changes.
5. Update durable findings when necessary.
6. Update milestone status only when justified.
7. Report completed work, evidence, and remaining limitations.

Do not overwrite unrelated user changes.

Do not claim success based on expected rather than observed results.

Do not create commits unless requested or authorized
by the project's workflow.

## 12. Guiding Principle

**Investigate → Establish evidence → Reconstruct → Validate.**

Not every project requires every stage.

Use the smallest process that preserves reliable knowledge
and produces independently verifiable progress.
