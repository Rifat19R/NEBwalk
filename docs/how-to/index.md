# How-to guides

Task-oriented guides for getting specific things done with nebwalk. Each
guide assumes you have already completed [Getting started](../getting-started/index.md)
and are working toward a concrete goal rather than learning the library.

- [Closed-loop active learning](active_learning.md) — run the audited
  campaign workflow that persists configuration, relaxed endpoints, and
  iteration state.
- [MACE fine-tuning](mlip_finetuning.md) — fine-tune an MLIP against
  DFT-labeled data with a tested MACE integration range.
- [Quantum ESPRESSO reference labeling](qe_labeling.md) — generate
  independent fixed-geometry single-point references with `QEReferenceLabeler`.
- [Campaign recovery and audit trail](campaign_recovery.md) — recover from
  failed runs using the atomically replaced, lock-guarded campaign state.
- [Production use](production.md) — pin dependencies, store reproducibility
  bundles, and isolate work directories for reported calculations.
