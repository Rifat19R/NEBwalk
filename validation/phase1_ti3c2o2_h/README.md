# Phase 1 Ti3C2O2/H flagship validation

This directory freezes the physical system and QE/PBE CI-NEB protocol without
altering the existing `MXenes/` results. Existing runs are evidence inputs only;
their summaries state `converged: false` and they are not reference barriers.

Run the read-only audit first:

```bash
python validation/phase1_ti3c2o2_h/audit_protocol.py
```

The baseline and four one-factor convergence variants must each use a new,
previously nonexistent output directory. Do not mix their pathways with
fine-tuning data. A domain expert must review adsorption-site interpretation,
slab/vacuum suitability, and every converged trajectory before acceptance.

`design_phase1_dataset.py` creates deterministic, path-grouped O-termination
training, validation, strict-holdout, and flagship candidates under the ignored
`generated/` directory. It rejects exact duplicates and
translation/permutation-invariant species-distance duplicates across all roles.
Run `diagnose_e0_rank.py` after generating the design. Phase 1 has one fixed
composition (`C8H1O8Ti12`), so its composition matrix has rank one for four
elemental unknowns. Fine-tuning must use `E0s="foundation"`; independently
fitted elemental E0 values and cross-termination absolute-energy claims are not
allowed.

Use `label_phase1_group.py GROUP --sanity-only` before labeling. Without the
sanity flag it labels exactly one frozen group with the baseline QE/PBE
protocol and stores resumable raw calculations under ignored `generated/`
paths. The strict holdout and flagship groups are never training or
model-selection inputs.
