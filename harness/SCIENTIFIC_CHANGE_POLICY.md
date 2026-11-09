# NEBwalk scientific change and verification policy

Status: development guidance (not a scientific validation result).
Owner: NEBwalk maintainer. Applies to human and AI-assisted changes.

## Purpose

Use existing tests and CI as the enforcement mechanism. This file classifies
change risk, specifies necessary evidence, and prevents automated edits from
quietly changing published or frozen scientific meaning. It does **not** change
NEBwalk numerical behavior, scientific thresholds, or public API.

Read alongside `CONTRIBUTING.md`, `docs/reference/architecture.md`,
`benchmarks/ase_comparison/README.md`, and the code/tests being modified.

## Change classes and minimum acceptance

| Class | Examples | Minimum evidence before merge |
| --- | --- | --- |
| A: non-scientific | Typos, formatting, docs-only clarification | Relevant link/content review; normal CI for PR. No new numerical claims. |
| B: software behavior | CLI, API, packaging, import boundaries, restart and reporting | Focused before/after test; public API and architecture tests; full local gate and CI. |
| C: scientific behavior | NEB forces, tangents, MIC, optimization, interpolation, barriers, uncertainty, selected-image/DFT geometry, recovery | Class B plus explicit physical rationale, numerical regression(s), independent comparison where feasible, units/tolerances and provenance review. |
| D: external-calculator/data lifecycle | QE settings, pseudopotentials, MACE fine-tuning, campaign artifacts, training datasets | Class C where relevant plus calculator-specific reproducibility evidence; state exact external checks *not run* if unavailable. |
| E: evidence or release | Frozen results, benchmark tables, schema/hash meaning, releases, manuscript-facing claims | Explicit maintainer approval, provenance audit and reviewable before/after evidence; never silently regenerate baselines. |

A change may belong to multiple classes. Apply the highest required level.
Passing the fast gate is never sufficient proof for Class C/D/E.

## Existing verification contracts

The repository already provides:

1. `tests/test_architecture.py` — module dependency boundaries.
2. `tests/test_public_api.py` — v0.11.0 public symbols, `__all__`, optional
   import laziness (`import NEBwalk` must not eagerly load torch/MACE).
3. `tests/test_forces.py`, `tests/test_neb.py`, and related tests — targeted
   numerical behavior, including reference force projection.
4. `tests/test_ase_comparison_benchmark.py` — convergence and tolerance-based
   NEBwalk/ASE comparison for analytical Morse H3 and Al adatom/Al(100) EMT.
5. `.github/workflows/ci.yml` — tests/coverage on Python 3.9–3.12, Ruff,
   mypy, package build, wheel smoke test and strict MkDocs build.
6. `.github/workflows/mace-smoke.yml` — optional MACE environment/CLI smoke,
   **not** proof that a real MLIP/QE campaign is physically valid.

The convenience runner `scripts/verify_change.py` has two modes:

- `--quick`: Ruff lint/format checks plus targeted architecture, public API,
  NEB/force, recovery, labeling and ASE comparison regression tests.
- `--full`: Ruff, mypy, complete pytest coverage run, source/wheel build,
  Twine validation and strict MkDocs build. Build artifacts use a temporary
  directory. The existing CI remains authoritative for its exact matrix,
  wheel-installation smoke test, and branch status.

Setup once (from an activated Python environment):

```bash
python -m pip install -e ".[dev,docs]"
python scripts/verify_change.py --quick
python scripts/verify_change.py --full
```

Neither mode automatically installs QE, MACE, licensed pseudopotentials,
models, or machine-specific executables. An inability to run external tests
must be reported, never replaced with fabricated results.

## Scientific invariants to check in review

- **Consistent paths:** For comparisons, document endpoints, atom mapping,
  interpolation, MIC/PBC, image count, spring constants, climbing-image
  settings, force threshold, optimizer settings, calculator and constraints.
  The existing ASE benchmark intentionally uses `climb=False`, common
  starting coordinates, and different native FIRE defaults. It supports
  implementation *consistency*, not a performance claim.
- **Geometry-label integrity:** A QE label must be associated with the exact
  coordinates it evaluates. Single-shot labeling rejects geometry-changing
  recovery; campaign labeling may preserve accepted recovered geometry under
  documented tolerances and separately hashes it. Review the entire retry
  chain, including geometry then parameter-only retry.
- **Provenance and compatibility:** Do not merge the two distinct QE settings
  hashes, disagreement metric definitions, peak-selection policies, or
  recovered-geometry policies described in `docs/reference/architecture.md`.
  Keep public v0.11.0 imports and legacy `nebwalk.*` schema identifiers.
- **Calculator boundaries:** Fresh calculator per image; no unsafe threaded
  CUDA evaluations. MACE/QE are optional; import should remain lightweight.
- **Scientific claims:** Model disagreement is an uncertainty proxy; sparse
  QE labeling/validation is not DFT NEB. Do not inflate the 13 audited cases
  or extrapolate two ASE examples to all materials or calculators.

## Tolerance and evidence policy

- Keep numerical tolerances specific to the tested quantity and problem.
  The current ASE-comparison tests use their own documented, physically
  motivated thresholds. Do not apply a single global barrier tolerance.
- If a legitimate algorithm fix changes a value, first demonstrate *why*,
  compare with independent or analytical evidence where possible, and record
  old/new values with units and machine-readable artifacts. Maintainer
  approval is required before changing any curated reference result.
- Preserve seeds, calculator/software versions, relevant QE settings and
  pseudopotential checksums or identifiers, starting geometries, units,
  convergence parameters and selection rules when reporting results.
- Do not commit raw trajectories, proprietary structures, pseudopotentials,
  model weights, temporary QE directories, or unreviewed computed outputs.
- A green CI run establishes tested software behavior under its environment,
  not broad physical accuracy across untested chemical systems.

## Required handoff report from a coding agent

Before asking the maintainer to merge, supply:

1. **Problem and risk:** concise issue, change class and affected surfaces.
2. **Files and diff:** exact modified paths, no unrelated changes.
3. **Verification:** commands, environment, passed/failed/skipped tests;
   identify missing MACE/QE/HPC validation explicitly.
4. **Scientific evidence:** what physical quantities/claims may change,
   supporting independent comparison, provenance, and numerical differences.
5. **Open questions:** remaining uncertainty and decisions requiring approval.

The agent must not claim that a patch is published, merged, released, or
scientifically validated unless that action/evidence has actually occurred.
