# Changelog

All notable changes to **NEBwalk** are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Audited corrections made after the v0.11.0 release. The `v0.11.0` tag and
PyPI release are unchanged, and the package still reports version 0.11.0
until the next release.

### Added
- Architecture and ownership reference (`docs/reference/architecture.md`). It
  separates the NEB core, the NEB-specific MLIP-to-DFT handoff layer, and the
  transitional generic ML-lifecycle layer, and records the compatibility
  policy for any future relocation and the parallel policies kept on purpose.
- Tests that pin the v0.11.0 public API, check that `import NEBwalk` does not
  load PyTorch or MACE, and enforce the module dependency direction.
- API reference pages for `NEBwalk.selection`, `NEBwalk.finetune`, and
  `NEBwalk.validate`.

### Changed
- `paper/paper.md` now matches the audited evidence. The validation section
  uses the 13 re-audited benchmark cases and the NEBwalk-versus-ASE
  consistency benchmark, replacing the older 20-case table (W/Mo
  comparators, error percentages, and four entries that were not
  re-audited). The fixed test count is gone. The MLIP-to-DFT and
  disagreement claims are bounded (sparse single points are not a full DFT
  NEB, and disagreement is an uncertainty proxy). The unpublished MXene study
  is no longer presented as a demonstration.

### Fixed
- `label_selected_images()` no longer pairs a DFT label with the wrong
  geometry. When a QE geometry-instability retry displaced atoms, the energy
  and forces belonged to the displaced structure but were reported, compared
  against the MLIP, and exported for the undisplaced MLIP image. Such images
  are now reported as failed and listed in
  `metadata["geometry_changed_during_recovery_indices"]`. Parameter-only
  recoveries (for example SCF mixing changes) are unaffected.
- `nebwalk_version` in MLIP-selection and DFT-labeling metadata now comes
  from `NEBwalk.__version__`, as in reproducibility bundles. It previously
  came from installed distribution metadata with hard-coded `0.7.1`/`0.10.0`
  fallbacks, which could record the wrong version (for example under a stale
  editable install).
- `run_with_recovery()` now validates and syncs the recovered geometry
  whenever any retry in the chain displaced atoms. Previously, a
  geometry-instability retry followed by a parameter-only retry (for example
  an SCF mixing change) skipped both steps, so the result of the displaced
  structure was returned for the caller's undisplaced atoms. This affected NEB
  image recovery, `label_selected_images()` (which could then accept the
  label), and `QEReferenceLabeler` (which could record
  `geometry_changed_during_recovery` as false).
- Recovery geometry synchronization now compares actual atomic positions
  instead of trusting the failure type, so any retry that moves atoms
  (including one that mutates the input in place) is validated and synced.
  The synced positions are written without applying constraints, so atoms
  under `FixAtoms` that a retry displaced keep the positions the calculation
  used, and a sync that does not reproduce them exactly raises an error.
- Retry parameters proposed by a recovery strategy are now verified after
  they are applied. If the calculator does not hold the requested values,
  recovery stops with `RecoveryExhausted` instead of re-running the
  calculation. Previously such a retry could silently repeat the identical
  calculation under the original settings.
- Labeling retries (`single_point_with_recovery()` and
  `QEReferenceLabeler`) now fail closed on any retry parameter other than
  `mixing_beta` that would differ from the configured settings (for example
  the recovery-raised `degauss`), because the label records the configured
  QE settings hash. NEB image recovery keeps its existing retry parameters.
- Retry parameters are now applied to ASE Espresso calculators after their
  first run. ASE replaces `parameters["input_data"]` with a `Namelist`
  (a `UserDict`, not a `dict`) when it writes `espresso.pwi`; the parameter
  helpers accepted only `dict`, so no retry parameter was ever written to a
  later pw.x input. They now accept any `Mapping`/`MutableMapping`.
- QE convergence retries now follow the documented base-relative sequence
  `mixing_beta = base * 0.5**attempt` (base 0.3: 0.15, 0.075, 0.0375).
  `run_with_recovery()` previously passed each retry's already-reduced value
  back as the base, compounding the reduction (0.15, 0.0375, 0.0046875).
  `QERecoveryStrategy.propose_retry()` takes an optional `base_params`,
  which `run_with_recovery()` supplies to strategies that accept it. The
  final-attempt `degauss` increase is unchanged (and still refused by the
  labeling paths).
- Reference labels (`single_point_with_recovery()`, `label_selected_images()`
  and `QEReferenceLabeler` `REF_forces`) now store the raw calculator forces.
  They were read through `Atoms.get_forces()`, which zeroes atoms under
  `FixAtoms`, so constrained atoms were stored with zero DFT force although
  pw.x reports nonzero forces for them. Constraints remain on the stored
  structures as geometry metadata. `max_force_disagreement_eV_A` keeps its
  existing constraint-applied convention.
- `QEReferenceLabeler` QE version detection accepts `pw.x -version` output
  that identifies `Program PWSCF` with a version, regardless of exit status.
  QE 7.4.1 prints the banner and then exits with status 1, which previously
  made environment validation fail. Output without the banner still fails.
- NEB image recovery parameters are now attempt-local. After each image
  evaluation (recovered or exhausted), the calculator's retry-managed
  parameters (`mixing_beta`, `conv_thr`, `degauss`) are restored to their
  pre-recovery values, and a parameter a retry added is removed again.
  Previously a successful retry left, for example, `mixing_beta=0.15` on the
  calculator, so every later NEB step ran with the reduced value and used it
  as the base of its own retry sequence (0.15 -> 0.075 -> ...). Within one
  recovery chain the sequence is unchanged (`base * 0.5`, `0.25`, `0.125`).
  The cached result of the recovered evaluation is kept, so post-run readers
  do not trigger a new calculation.
- Documentation: `DFTLabel.max_force_disagreement_eV_A` is now documented
  (docstring, label README disclosure, architecture reference) as an
  operational constrained-force disagreement metric, computed after the
  image's ASE constraints are applied to both force sets. It is not a raw
  DFT-force disagreement. The stored label forces are raw.

## [0.11.0] — 2026-10-05

### Added
- Reproducible NEBwalk-versus-ASE implementation-consistency benchmark for
  analytical Morse-H3 and periodic Al-adatom/Al(100) paths, including
  provenance-locked JSON/CSV results and automated regression tests.
- Canonical extended-XYZ reference datasets with strict energy/force and
  provenance validation, deterministic structure and dataset checksums,
  conflicting-label detection, atomic manifests, conflict-aware merge and
  deduplication, and leakage-free path-grouped train/validation/test splits.
- Backend-neutral fine-tuning contracts, typed naive/LoRA/multihead-replay
  configuration, safe official MACE CLI command generation and execution,
  environment/capability checks, complete training provenance, explicit
  evaluation failures, energy/force/pathway metrics, and a checksummed local
  model registry that selects active models using validation metrics.
- Generic ASE committee calculator/evaluator with independent member
  calculators, partial-failure handling, mean predictions, energy/force and
  relative-energy disagreement proxies, per-image diagnostics, and barrier
  spread; plus mandatory-and-ranked candidate selection with peak awareness,
  failure/high-force inclusion, path diversity, and explicit fallback records.
- Resumable `QEReferenceLabeler` built on the existing QE factory and recovery
  machinery, with isolated raw work directories, canonical dataset keys,
  settings hashes, recovery/geometry-change provenance, atomic manifests,
  partial-failure retention, and checksum-verified reuse of completed labels.
- Atomic, lock-protected `ActiveLearningCampaign` state machine covering
  bootstrap labeling, leakage-free dataset versioning, committee retraining,
  validation-based model activation, iterative NEB selection, explicit
  stopping decisions, failure tracebacks and stage resume, immutable iteration
  artifacts, and separate sparse/full-QE final validation semantics.
- `NEBwalk` CLI commands for dataset validation/splitting, MACE environment
  checks/fine-tuning/evaluation, and campaign initialization, execution,
  status, resume, and final validation; deterministic JSON, Markdown, and CSV
  reports; a dependency-free campaign dry run; and editable campaign templates.
- Production quality gates for Ruff, mypy, branch coverage, package validation,
  and strict documentation builds across Python 3.9-3.12.
- MkDocs user guide and generated API reference, contribution and conduct
  policies, correct security policy, citation metadata, issue forms, PR
  checklist, CODEOWNERS, and Dependabot configuration.
- Early validation for invalid `NEBRunConfig` values.
- Independent force-regression tests against ASE's improved-tangent NEB for
  standard and climbing-image paths.
- Five organic Egret-1t/MACE-OFF23 disagreement-selection example scripts:
  ethane, propane, methanol, ethanol, and dimethyl ether.
- Organic disagreement result summaries with literature-scale comparison notes
  and explicit caveats that these runs are exploratory checks, not DFT
  validation or calibrated uncertainty quantification.
- Pt vacancy migration added to `vacancy_benchmark_suite.py` (EMT), kept as a
  documented calculator-limitation case rather than a NEBwalk bug.

### Changed
- Re-audited the historical README benchmark table: quantitative claims are
  restricted to 13 cases supported by surviving artefacts; seven historical
  entries are explicitly excluded from the current validation count, and the
  38-system development scaffold is no longer described as an executed public
  benchmark suite.
- **Breaking:** package/import renamed from `nebwalk` to `NEBwalk`
  (`import NEBwalk`, `src/NEBwalk/`, CLI command `NEBwalk`, PyPI project
  `NEBwalk`, GitHub repo `Rifat19R/NEBwalk`). Existing `pip install nebwalk`
  users must update `import nebwalk` to `import NEBwalk`; the PyPI
  distribution name is unchanged after PEP 503 normalization, so
  `pip install nebwalk`/`pip install NEBwalk` both still resolve to the same
  project. Internal data-format identifiers (the `"schema": "nebwalk.*.v1"`
  strings written into dataset manifests, campaign state, and reproducibility
  bundles) are intentionally left as `nebwalk.*` so already-frozen artifacts
  from prior releases keep loading without a schema-version bump.
- `ag/au/cu/ni/pd/pt_vacancy_emt.py` collapsed into thin wrappers around
  `vacancy_benchmark_suite.main()`; removes ~600 lines of duplicated
  build/relax/NEB/report boilerplate across the six scripts.
- `ethane_egret_maceoff23_disagreement.py` and `maceoff23_sanity_check.py` now
  share their `ethane()` geometry builder and calculator factories from
  `organic_disagreement_common.py` instead of each carrying its own copy.
  This also aligns `maceoff23_sanity_check.py`'s MACE-OFF23 precision from
  `float32` to `float64`, matching the project-wide convention used by
  `vacancy_benchmark_suite.py` and the other MACE-MP-0 examples.
- `al_vacancy_qe.py` QE params (`ecutwfc`, `ecutrho`, `conv_thr`) brought in
  line with `vacancy_benchmark_suite.qe_params_for()` — the two Al/QE entry
  points share output files and were silently drifting apart.
- Mg (HCP, basal-plane) vacancy reference barrier corrected from 0.60 eV to
  0.52 eV in `mg_vacancy_macemp.py`, matching the canonical value in
  `vacancy_benchmark_suite.SYSTEMS["mg"]`.
- `mg_vacancy_mgo_macemp.py` output filenames now consistently prefixed
  `mg_vacancy_mgo_macemp_*` instead of `mg_vacancy_mgo_*`.
- README's MACE-OFF23 code sample corrected to `default_dtype="float64"`
  (was stale `float32`, contradicting the actual `organic_disagreement_common.py`
  implementation and the MACE-MP-0 example immediately below it); added the
  missing `default_dtype="float32"` to the sample's Egret-1t factory to match
  actual code.
- README's `examples/organic_disagreement_results_v0.10.0.md` reference
  replaced by `examples/organic_disagreement_literature_comparison_v0.10.0.md`,
  which now carries both the disagreement diagnostics and the literature
  comparison in one file.

### Removed
- `examples/organic_disagreement_results_v0.10.0.md`, superseded by
  `examples/organic_disagreement_literature_comparison_v0.10.0.md`.

### Notes
- Acetaldehyde was tested but intentionally excluded from the pushed example
  set because its barrier was a clear outlier and default force-disagreement
  selection was endpoint-adjacent.

## [0.10.0] — 2026-06-18

### Added
- Cross-model disagreement (uncertainty proxy) image selection strategy:
  `uncertainty_disagreement`.
- `NEBwalk.uncertainty` module with `DisagreementResult` and
  `compute_cross_model_disagreement()`.
- Secondary-calculator support in `MLIPActiveNEBConfig` for comparing
  converged primary MLIP images against a second organic-domain MLIP.
- Optional `energy_disagreement` and `force_disagreement` fields on exported
  `SelectedImage` metadata.
- MACE-OFF23 sanity-check script and ethane Egret-1t/MACE-OFF23 end-to-end
  disagreement-selection example.
- Tests covering relative-energy disagreement, calculator isolation,
  per-image failures, reference-image failure, fallback behavior, and JSON
  export metadata.

### Changed
- `select_images()` dispatch now supports `uncertainty_disagreement` while
  preserving existing `peak_plus_neighbors` behavior.
- `run_mlip_assisted_neb()` now distinguishes missing secondary-calculator
  configuration errors from per-run fallback when too few valid disagreement
  results are available.
- Exported selected-image README files disclose when cross-model disagreement
  was used as an uncertainty proxy.
- Test suite expanded: 144 -> 156 tests.

### Notes
- This release closes the previously documented uncertainty-guided selection
  gap only through cross-model disagreement between Egret-1t and MACE-OFF23.
  It is not calibrated uncertainty quantification and no committee/ensemble
  was trained.
- The disagreement proxy is only meaningful where both calculators are within
  their validated chemical domain. In v0.10.0 this is documented for organic
  systems, with ethane torsion as the exercised example.
- Does not implement automatic MLIP retraining, automatic QE/DFT refinement,
  adaptive image insertion/removal, or inorganic/vacancy disagreement pairing.

## [0.9.0] - 2026-06-17

### Added
- Automatic failed-image recovery for Quantum ESPRESSO image calculations.
- Generic recovery layer: FailureType, RecoveryAttempt,
  NoOpRecoveryStrategy, RecoveryExhausted, and run_with_recovery.
- QE-specific recovery strategy: classify convergence failures, geometry
  instabilities, process failures, and unknown failures from QE output.
- Retry policy for QE convergence failures: reduce mixing_beta on retries
  and increase degauss on the final retry.
- Retry policy for QE geometry instabilities: deterministic bounded
  displacement with seed control and post-recovery geometry validation.
- Optional recovery-log serialization in reproducibility bundles.
- Recovery test suite covering retry taxonomy, QE strategy behavior,
  recovered-geometry synchronization, and stale-result clearing.

### Changed
- QE calculator factories now attach QERecoveryStrategy by default while
  preserving default no-op recovery behavior for non-QE calculators.
- NEB optimization threads recovery strategy and recovery_log through
  per-image evaluation.
- Public API exports recovery primitives and QERecoveryStrategy.

---

## [0.8.0] — 2026-06-13

### Added
- `NEBwalk.reproduce` module: `save_bundle()` and `ReproBundle`.
  Exports input structures, NEBRunConfig, results, energy profile, trajectory,
  SHA-256 manifest, software environment, and a human-editable rerun template
  into a self-contained directory and optional `.tar.gz`.
- `run_neb_calculation()` accepts `reproduce_dir` and `calc_params` keyword
  arguments for one-line reproducibility capture.
- 16 new tests in `tests/test_reproduce.py`.
- README: Reproducibility section with usage examples.

### Notes
- No new runtime dependencies. Reproducibility uses stdlib only.
- Calculator factory code is intentionally not serialized. Provide `calc_params`
  as a plain dict to document your calculator configuration.

---

## [0.7.1] — 2026-06-12

### Fixed
- Improved-tangent bisection branch now weights unit vectors by energy
  differences per Henkelman-Jonsson Eq. 10, avoiding displacement-length bias
  at extremum images.
- Variable spring constants now reference the global path minimum, preserving
  spring ordering when intermediates lie below both endpoints.
- CUDA-backed calculators now emit a warning when used with thread-parallel
  image evaluation because CUDA force evaluation from multiple Python threads
  can silently corrupt results.
- Exported selected-images README now reports the actual selection strategy.
- Example benchmark runner now defaults to the full suite, avoids duplicate
  safe-mode runs, documents all QE pseudopotentials, uses `np=4` with Open MPI
  oversubscription for WSL, and cleans stale QE workdirs before QE reruns.
- QE example runner now preserves a user-provided `ESPRESSO_COMMAND` and falls
  back to an absolute `pw.x` path, avoiding PATH-dependent failures in ASE
  subprocesses.

### Docs
- Tangent fix may shift converged barriers slightly for paths with uneven image
  spacing. Benchmark table values will be re-validated in a follow-up run.

---

## [0.7.0] — 2026-06-12

### Added
- `NEBwalk.active` module with `run_mlip_assisted_neb()` high-level workflow.
- `MLIPActiveNEBConfig`, `MLIPActiveNEBResult`, and `SelectedImage` dataclasses.
- `NEBwalk.selection` module with `peak_plus_neighbors` image selection strategy.
- Selected-image export to `.xyz`, `.traj`, and `.json` formats.
- Examples for MLIP-assisted NEB using EMT and MACE template.
- Tests for selection logic and active workflow exports.

### Changed
- Test suite expanded: 94 → 111 tests.

### Notes
- Introduces an active-learning-ready MLIP-assisted NEB workflow.
- Does not yet implement: uncertainty-guided selection, automatic MLIP
  retraining, adaptive image insertion/removal, or QE failed-image recovery.

---

## [0.6.0] — 2026-06-10

### Added
- **Approximate geodesic-style interpolation** (`geodesic_interpolate`) — IDPP
  interpolation with overlap repulsion, not the mass-weighted internal-coordinate
  geodesic method; a heuristic alternative to plain IDPP for large conformational
  changes.
- **Quantum ESPRESSO interface** (`NEBwalk.qe`): `QEParams`, `make_qe_factory`,
  `validate_qe_setup` — generate and validate QE PWSCF input files for
  DFT-level NEB workflows.
- `.gitattributes` for consistent cross-platform line endings.
- `docs/` placeholder for future Sphinx documentation.
- New validated examples (14 systems total):
  - Cu adatom diffusion on Cu(100) / EMT (4.6% vs DFT-PBE)
  - Ni adatom diffusion on Ni(100) / EMT (12% vs DFT-PBE)
  - Li vacancy migration in Li₂O / MACE-MP-0 (1.4% vs DFT-GGA, ref. 0.28 eV)
  - Mg vacancy migration in MgO / MACE-MP-0 (2.5% vs DFT-PBE)
  - H diffusion on Cu(111) / MACE-MP-0 (~16%, documented surface limitation)

### Changed
- Test suite expanded: 68 → 94 tests (QE interface coverage added).
- Repo root cleaned: model files, logs, outputs added to `.gitignore`.

---

## [0.5.0] — 2026-06-08

### Added
- Improved tangent estimate (Henkelman & Jónsson, J. Chem. Phys. 113, 9978, 2000).
- IDPP interpolation (Smidstrup et al., J. Chem. Phys. 141, 214106, 2014).
- MIC-aware periodic boundary conditions.
- Variable spring constants (Lindh et al., Chem. Phys. Lett. 241, 423, 1995).
- FIRE optimizer (Bitzek et al., PRL 97, 170201, 2006).
- High-level API: `NEBRunConfig`, `run_neb_calculation`, `NEBRunResult`.
- Thread-based parallel image evaluation.

### Fixed
- Convergence criterion changed to per-atom force magnitude (norm-based).
  Confirmed by ethane barrier shift: 0.108 → 0.113 eV.

### Removed
- L-BFGS-B optimizer (inappropriate for non-conservative NEB force field).

---

## [0.4.0] — 2026-06-01

### Added
- Core NEB: spring forces, perpendicular force projection, upwind tangent.
- Climbing Image NEB (CI-NEB).
- Linear interpolation (MIC-aware).
- ASE compatibility, energy profile plot, CSV, `.traj` output.
- GitHub Actions CI, MIT license.
