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
