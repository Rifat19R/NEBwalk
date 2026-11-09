# NEBwalk — repository instructions for coding agents

These instructions apply to the whole repository. Treat NEBwalk as scientific
research software: numerical correctness, evidence, reproducibility, and API
compatibility take precedence over code churn or apparent simplification.

## Mission and boundaries

- Maintain a transparent, ASE-compatible NEB/CI-NEB library and its MLIP-to-DFT
  workflows. Do not turn it into a general-purpose agent platform.
- Preserve the existing `src/NEBwalk/` import namespace, `NEBwalk` CLI and
  package name. Legacy `nebwalk.*` schema identifiers are intentionally retained
  for compatibility with frozen artifacts.
- Prefer the smallest, reviewable change that addresses a demonstrated issue.
  Do not restructure modules or add dependencies as a side effect of a fix.
- Claude Code may implement changes; the maintainer decides scientific intent,
  accepts evidence, and authorizes commits, merges, and releases. An AI review
  is advisory, not a substitute for tests or domain validation.

## Source of truth: read only what is relevant to the task

- Architecture and dependency direction: `docs/reference/architecture.md`.
- Contribution, quality, and evidence expectations: `CONTRIBUTING.md`.
- Scientific change classification and acceptance: `harness/SCIENTIFIC_CHANGE_POLICY.md`.
- ASE comparison scope, methods and limitations:
  `benchmarks/ase_comparison/README.md`.
- Scientific API and compatibility gates: `tests/test_architecture.py`,
  `tests/test_public_api.py` and `tests/test_ase_comparison_benchmark.py`.
- Public behavior: `README.md`, `docs/`, `pyproject.toml` and `CHANGELOG.md`.

Do not treat a previous conversation, unpublished work, or an old benchmark
summary as stronger evidence than the repository's audited artifacts.

## Architecture and compatibility invariants

- The NEB core must not import handoff or generic ML-lifecycle modules; generic
  lifecycle modules must not import the core or handoff. Check the architecture
  document before adding imports. `test_architecture.py` enforces this boundary.
- Preserve public v0.11.0 import paths, `__all__` exports, and lazy optional
  ML framework imports. Do not silently change deprecation policy.
- Each NEB image needs an independent ASE calculator instance. Do not enable
  concurrent thread evaluation for CUDA-backed calculators.
- Do not unify deliberately different QE settings hashes, force-disagreement
  definitions, peak-selection policies, or geometry-recovery acceptance rules.
  They have distinct scientific and dataset-compatibility semantics.

## Scientific integrity — non-negotiable

- Never rewrite a reference barrier, frozen dataset, recorded model metric,
  provenance hash or regression threshold merely to make a test pass.
- A method or tolerance change requires a physical/numerical rationale,
  a failing-before/fixed-after test where feasible, and independent comparison
  or clearly justified benchmark evidence. State units and convergence criteria.
- In the single-shot MLIP-image labeling path, do not associate DFT energy or
  forces from a displaced geometry with the original MLIP geometry. Campaign
  labeling may accept recovered geometry only under its documented tolerance
  and must record the actual geometry and provenance.
- Sparse QE single-point validation is not full DFT NEB. Cross-model or
  committee disagreement is an uncertainty proxy, not calibrated uncertainty.
- The audited research claim is 13 artifact-backed validation cases. The two
  ASE comparison systems are a specifically bounded consistency check, not a
  universal accuracy or performance guarantee.
- Preserve exact versions, calculator settings, pseudopotential identity,
  structures, seed, units, and environment metadata when making scientific
  claims. Report unavailable checks as *not run*, never as *passed*.

## Workflow for every nontrivial change

1. Inspect `git status --short`, current branch, diff and relevant files.
   Never overwrite existing local changes without asking the maintainer.
2. State the problem, affected files, scientific risk, minimal plan, and
   acceptance criteria before modifying code. Keep unrelated work out of scope.
3. Add or update a focused test for changed behavior. For a reported bug,
   establish the failure before modifying the implementation when feasible.
4. Implement the smallest fix; avoid automatically editing curated evidence,
   manuscript claims, version numbers or release metadata.
5. Run `python scripts/verify_change.py --quick` during development and
   `python scripts/verify_change.py --full` before proposing merge when the
   required development dependencies are available. Add task-specific QE/MACE
   checks only in a suitable, explicitly authorized environment.
6. Report changed files, exact commands, pass/fail/skip outcomes, scientific
   residual risks, and any unverified assumptions. Do not conceal failures.

Do not commit, push, merge, publish, delete branches, force-reset, or launch
expensive QE/MACE/HPC calculations without explicit maintainer authorization.
No automatic threshold relaxation, reference-data regeneration, or bulk
refactoring. If a request conflicts with scientific integrity, explain and
escalate rather than manufacturing a passing result.
