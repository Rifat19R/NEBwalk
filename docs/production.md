# Production use

Pin nebwalk, ASE, NumPy, SciPy, calculator, model, and pseudopotential versions.
Store a reproducibility bundle for every reported calculation. Run expensive
jobs in isolated work directories and preserve raw calculator logs outside the
Git repository.

Treat `converged=False`, non-finite energies or forces, discontinuous profiles,
atom overlap, endpoint drift, and recovery exhaustion as hard failures. Recovery
attempts must be reviewed; successful SCF recovery does not independently prove
scientific validity.

Thread parallelism is supported for suitable CPU calculators. CUDA calculators
must use serial image evaluation unless process isolation is managed externally.
Do not share mutable calculator instances between images.

Security reports follow `SECURITY.md`; ordinary scientific or convergence
problems belong in a reproducible bug report.
