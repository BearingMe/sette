
# Research Conventions

## 1. Principles

- Evidence over assumptions.
- Primary sources over secondary interpretations.
- Reproducible observations over plausible explanations.
- Preserve failures, contradictions, and unknowns.
- Never claim validation that was not performed.
- Keep documentation proportional to the investigation.

Research must distinguish **what is known**, **how it is known**, and **what remains uncertain**.

## 2. Evidence Classification

Tag non-trivial technical claims with their evidence class.

| Tag | Classification | Meaning |
|---|---|---|
| `[D]` | Documentation | Supported by official or authoritative documentation. |
| `[S]` | Source observation | Observed directly in source code, with file and revision identified. |
| `[R]` | Reverse engineering | Derived from binary inspection, disassembly, traces, or other reverse-engineering analysis. |
| `[M]` | Measurement | Observed through an executed experiment or test. |
| `[H]` | Hypothesis | Plausible explanation not yet verified. |
| `[U]` | Unknown | Unresolved question or missing evidence. |

### Rules

1. Evidence classes describe **provenance**, not confidence.
2. No class automatically overrides another.
3. Claims must identify relevant versions, environments, and limitations.
4. Hypotheses must not be presented as established facts.
5. Contradictory evidence must be preserved and investigated.
6. Reclassify claims when new evidence becomes available.

### Example

[D] The documented instruction operates on 16×16 matrices.
Source: Official ISA documentation, section X.

[S] The reference implementation uses a specific tensor layout.
Source: `references/sources/example/src/kernel.cpp`, commit `abc123`.

[M] The reconstructed kernel produced zero mismatches against
the CPU reference for the recorded test vectors.
Source: `docs/experiments/EXP-001-kernel.md`.

[H] A different layout may improve memory throughput.

[U] Performance under sustained multi-dispatch workloads.

## 3. Source Provenance

Every important claim must be traceable to its supporting evidence.

Include relevant information:

- Source name, file, URL, or artifact path.
- Version, commit, build, or binary hash.
- Relevant line, symbol, address, or offset when available.
- Hardware, operating system, driver, or runtime when relevant.
- Access date for mutable external resources.

Prefer pinned references under `references/`.

Do not repeatedly copy large source excerpts into research documents.

Reference the original material instead.

## 4. Experiments

Location:

`docs/experiments/EXP-NNN-slug.md`

Experiments answer specific questions through observation.

Use the following template:

### EXP-NNN: Question or Experiment Name

**Milestone:** MXX
**Status:** PENDING | PASS | FAIL | BLOCKED | INCONCLUSIVE (see definitions in §6)

Experiment status records whether its predefined acceptance criteria were met,
not whether the hypothesis was supported. A `PASS` may mean the experiment
disproved the hypothesis. Record the hypothesis outcome separately under
Interpretation.

#### Question

What specific behavior or hypothesis is being investigated?

#### Method

- Environment and relevant versions.
- Inputs and preconditions.
- Commands or reproduction steps.
- Expected observations.
- Comparison oracle, if applicable.

#### Results

Record what actually happened.

Include measurements, outputs, errors, and links to
supporting artifacts.

Do not invent results or substitute expected behavior
for observed behavior.

#### Interpretation

Explain what the results support, contradict, or leave unresolved.

Classify important conclusions using the evidence tags.

#### Next

Record follow-up investigations only when necessary.

### Experiment Rules

1. Record meaningful failures and negative results.
2. Never silently rewrite historical observations. Append corrections or addenda.
3. An experiment may successfully answer a question with a negative result.
4. Successful compilation alone does not prove runtime correctness.
5. A plausible output is not sufficient evidence of fidelity.
6. Use reusable scripts when experiments must be repeated.
7. Disposable exploration does not require permanent scripts or reports unless it produces durable findings.
8. Keep raw outputs in the project's ignored artifact directory unless small fixtures are required for reproducibility.

## 5. Findings

Location:

`docs/findings/YYYY-MM-DD-topic.md`

Findings consolidate durable knowledge from documentation,
source inspection, reverse engineering, and experiments.

Use the following template:

### Topic

#### Summary

Briefly describe what was established.

#### Findings

List supported technical conclusions with evidence tags
and source references.

#### Limitations and Unknowns

Record unresolved questions, version constraints,
contradictions, and known gaps.

#### References

Link to supporting sources and experiments.

### Finding Rules

1. Findings summarize evidence; they do not replace it.
2. Do not duplicate complete experiment reports.
3. Preserve relevant version and environment constraints.
4. Update findings when new evidence changes a conclusion.
5. Clearly mark superseded or contradicted findings.
6. Avoid creating separate findings for trivial observations.

## 6. Validation

Validation outcomes:

| Status | Meaning |
|---|---|
| `PASS` | Predefined acceptance criteria were met with sufficient evidence. |
| `FAIL` | Acceptance criteria were not met, including because of procedural or execution failures. |
| `BLOCKED` | A required dependency prevents execution. |
| `INCONCLUSIVE` | Execution occurred, but evidence cannot establish a conclusion. |
| `PENDING` | The experiment or validation has not been executed. |

### Rules

- Define acceptance criteria before claiming completion.
- Validate the claim being made, not merely the execution of a tool.
- Derive expected results independently of the implementation under test.
- Prefer comparisons against authoritative sources or independent oracles.
- Record relevant discrepancies and limitations.
- Never turn `BLOCKED`, `FAIL`, or `INCONCLUSIVE` into `PASS` without new evidence.

Research findings and implementation fidelity are separate concerns.

A successful investigation can establish that reconstruction
is currently impossible or that an approach does not work.

## 7. Documentation Discipline

- `docs/MILESTONES.md` owns progress and validation gates.
- `docs/experiments/` owns experimental observations.
- `docs/findings/` owns consolidated technical knowledge.
- `references/` owns external research material.
- `scripts/` owns reusable research utilities.
- `tests/` owns automated regression and validation checks.
- `AGENTS.md` owns agent behavior and operational rules.

Do not duplicate authoritative information across files.

Prefer links and references over copied content.

Create documentation when it preserves useful knowledge,
supports reproducibility, or enables future work.

**Do not create documentation merely to demonstrate progress.**
