# Trace

A lightweight, evidence-driven harness for reverse-engineering projects.

Trace provides a reusable structure for investigating systems, documenting discoveries, reconstructing behavior, and validating results against independent evidence.

**The goal is simple: preserve knowledge, make progress measurable, and minimize administrative overhead.**

## Philosophy

- **Evidence first.** Distinguish observations, hypotheses, and unknowns.
- **Reproducibility.** Important experiments and conclusions must be verifiable.
- **Incremental progress.** Work through independently testable milestones.
- **Minimal bureaucracy.** Document meaningful knowledge, not routine activity.
- **Independent validation.** Never use an implementation as its own oracle.
- **Repository ownership.** Project state and evidence live in Git, not external memory systems.

Trace is methodology and structure, not a framework or runtime.

## Repository Structure

```text
trace/
├── AGENTS.md
├── README.md
├── docs/
│   ├── MILESTONES.md
│   ├── CONVENTIONS.md
│   ├── experiments/
│   └── findings/
├── references/
│   ├── README.md
│   └── sources/
├── scripts/
├── src/
└── tests/
```

| Location | Purpose |
|---|---|
| `AGENTS.md` | Agent behavior, workflow, and filesystem restrictions |
| `docs/MILESTONES.md` | Progress dashboard, checkpoints, and validation gates |
| `docs/CONVENTIONS.md` | Research conventions and evidence classification |
| `docs/experiments/` | Reproducible investigations and observations |
| `docs/findings/` | Consolidated technical knowledge |
| `references/` | External reference material and provenance |
| `scripts/` | Reusable investigation utilities |
| `src/` | Reconstruction and implementation |
| `tests/` | Automated validation and regression |

Implementation-specific architecture is determined by each project.

## Workflow

1. **Define:** Establish the target, scope, constraints, and success criteria.
2. **Investigate:** Inspect primary sources, existing implementations, and observable behavior.
3. **Experiment:** Resolve important unknowns through reproducible probes.
4. **Reconstruct:** Implement the smallest behavior supported by evidence.
5. **Validate:** Compare results against independent expectations or reference behavior.
6. **Consolidate:** Preserve durable findings and update milestone progress.

These stages are not mandatory phases. A research-only project may never require reconstruction or integration.

### Milestones

Each milestone contains:

- An objective describing the desired outcome.
- Independently verifiable checkpoints.
- A validation gate defining completion.
- Deliverables identifying the expected artifacts.

`docs/MILESTONES.md` is the authoritative progress dashboard.

## Evidence

Evidence tags distinguish provenance, not a universal ranking of reliability.
See [Research Conventions](docs/CONVENTIONS.md#2-evidence-classification) for
the authoritative classifications and usage rules.

## Using Trace

1. Create a repository from this template.
2. Replace this README with the project's actual objectives, scope, environment, and execution commands.
3. Adapt `docs/MILESTONES.md` to the investigation. The example milestones are not mandatory.
4. Populate `references/` with relevant, preferably pinned sources.
5. Begin the first milestone and record evidence as it becomes useful.

Keep the generic operational rules in `AGENTS.md`. Add project-specific restrictions only when justified.

### Agent Workflow

Agents should read `AGENTS.md`, inspect the active milestone, and consult relevant references before working.

OpenCode TODOs may manage active execution tasks. AgentMemory and MemPalace may assist retrieval, but are optional.

**The repository must remain independently recoverable without external memory.**

## Non-Goals

Trace is not intended to provide:

- A universal application architecture.
- A mandatory programming language or build system.
- Automatic reverse engineering.
- A separate document for every experiment or session.
- Mandatory review cycles for every milestone.
- A replacement for testing, independent oracles, or runtime validation.

## Status

**Template development.** Trace is being refined through practical use on reverse-engineering projects.

Its structure and conventions should evolve based on demonstrated needs, not speculative complexity.
