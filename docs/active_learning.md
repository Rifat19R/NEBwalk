# Closed-loop active learning

The campaign workflow follows a fixed, auditable order:

1. persist configuration and relaxed endpoints;
2. obtain or validate bootstrap reference labels;
3. create a versioned, path-grouped dataset split;
4. train independent committee members and evaluate validation data;
5. register the best validation model;
6. run MLIP NEB paths and committee diagnostics;
7. select mandatory peak/failure/high-force images, then uncertainty-ranked
   and diverse candidates;
8. label selected structures with QE and merge only new valid references;
9. evaluate objective stopping criteria and repeat when required;
10. perform separately recorded final sparse QE validation, with full QE NEB
    only when explicitly configured.

Copy and edit `examples/active_learning_templates/mace_qe_campaign.json`, then:

```bash
nebwalk campaign init campaign.json
nebwalk campaign run campaign.json
nebwalk campaign status campaign-output
nebwalk campaign validate-final campaign.json
```

For a no-MACE, no-QE workflow check:

```bash
python examples/active_learning_dry_run.py
```

The dry run exercises state transitions, labels, dataset versioning, committee
training artifacts, selection, resume-compatible manifests, stopping, and
reports with deterministic fakes. It is a software sanity test, not scientific
validation.

Committee disagreement is an uncertainty proxy and is not calibrated. Missing
configured metrics never satisfy a stopping threshold. Sparse selected-image
QE checks do not constitute a full DFT NEB, and reports state that limitation.
Use stable path identifiers, keep chemically related images in the same split,
and independently validate the final model and barrier for the target domain.
