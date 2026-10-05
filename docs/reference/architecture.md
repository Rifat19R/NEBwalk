# Architecture and ownership boundaries

NEBwalk's job is NEB and CI-NEB physics and the transition-state workflow
around it. Since v0.11.0 the package also contains general ML-lifecycle code
(datasets, fine-tuning, committees, model registry, campaign state). That
code is expected to move to a separate ML-lifecycle package later. This page
records which layer each module belongs to, which way dependencies may point,
and how a move will stay compatible with the released API.

## Layers

| Layer | Modules | Owner |
|---|---|---|
| NEB core | `neb`, `forces`, `optimize`, `interpolate`, `engine`, `output`, `recovery`, `qe`, `reproduce` | NEBwalk |
| MLIP→DFT handoff (NEB-specific) | `active`, `selection`, `uncertainty`, `label`, `labeling`, `finetune`, `candidate_selection`, `validate`, `campaign`, `dryrun`, `cli` | NEBwalk |
| Generic ML lifecycle (transitional) | `datasets`, `campaign_state`, `reporting`, `mlip.base`, `mlip.mace_backend`, `mlip.metrics.compute_model_metrics`, `mlip.registry`, `mlip.committee` | planned to move |

The handoff layer stays in NEBwalk because it encodes path-specific policy:
peak and neighbor selection, relative-energy zeroing along a path, DFT labels
at the exact MLIP image geometry, QE recovery provenance, and
sparse-versus-full-QE validation semantics. `mlip.metrics.compute_pathway_metrics`
and the barrier spread in `CommitteePathDiagnostics` are NEB-specific, even
though they sit in modules that are otherwise generic. Parts of `finetune`
(isolated-atom E0 references, the `mace_run_train` command template) and of
`validate` (`mace_loader_read_test`) are generic and are candidates to move
with the lifecycle layer. Their NEB-label export glue stays here.

## Dependency direction

- The NEB core imports nothing from the other two layers.
  `tests/test_architecture.py` enforces this.
- The handoff layer may use the core and the lifecycle layer.
- Lifecycle modules import nothing from the NEB core or the handoff layer.
  `tests/test_architecture.py` enforces this as well.

The handoff layer reaches lifecycle functionality through explicit
interfaces. A future external implementation can satisfy these without any
change to NEB code:

| Interface | Defined in | Used by |
|---|---|---|
| `ReferenceLabeler` (protocol) | `labeling` | `campaign` |
| `FineTuningBackend` (protocol) | `mlip.base` | `campaign` |
| `CampaignPathRunner` (protocol) | `campaign` | `campaign` |
| calculator factories (`Callable[[], Calculator]`) | everywhere | core and handoff |

## Compatibility policy for relocation

The public API is what each module's `__all__` lists, or its public top-level
names where there is no `__all__`. `tests/test_public_api.py` pins the
v0.11.0 API. If generic code moves out of NEBwalk:

1. Its v0.11.0 import path keeps working through a re-export that emits a
   `DeprecationWarning` naming the new location.
2. The shim stays for at least one minor release and is removed only in a
   major release, with a CHANGELOG entry.
3. NEBwalk depends on the new package only after that package is released.
   Optional imports stay lazy, so `import NEBwalk` still does not import
   PyTorch or MACE.

NEBwalk does not relocate anything in 0.11.x.

## Parallel policies kept deliberately

The following implementations look alike but intentionally differ. Merging
any of them would change released numbers or dataset hashes, so each is
documented rather than unified.

- **DFT settings hash.** `validate.qe_settings_hash()` hashes a
  `QESettingsRecord`, which is used by the single-shot export and the
  published `datasets/` files. `QEReferenceLabeler.settings_hash()` hashes the
  full labeler payload, including the command and QE version, and is used by
  campaigns. The same physical settings produce different hashes under the
  two schemes, so `datasets.write_dataset()` will not merge datasets from the
  two pipelines.
- **Force disagreement.** `DisagreementResult.force_disagreement` is the
  largest atom-wise vector norm of the force difference.
  `DFTLabel.max_force_disagreement_eV_A` is the largest absolute Cartesian
  component. Committee diagnostics report standard deviations
  (`max_atom_force_std`, `rms_force_disagreement`). These metrics cannot be
  compared directly.
- **Peak selection.** `selection.select_peak_plus_neighbors()` excludes the
  endpoints by default. `candidate_selection.select_active_learning_candidates()`
  takes the argmax over the whole path and adds the endpoints during
  bootstrap.
- **Recovered geometries.** `label.label_selected_images()` rejects a label
  if recovery moved any atom, because that label is compared with the MLIP
  image. `labeling.QEReferenceLabeler` accepts a recovered geometry within the
  strategy's minimum-image tolerance and records it under its own
  `structure_hash`, with `recovered_geometry_accepted=True`.

All of these share one implementation of single-point evaluation with QE
recovery (`recovery.run_with_recovery`). The single-shot labeler and the
isolated-atom reference also share `label.single_point_with_recovery`.

## Scientific limits that apply in every layer

- Cross-model or committee disagreement is an uncertainty proxy. It is not
  calibrated uncertainty.
- Sparse QE single points on an MLIP path are not a full DFT NEB.
- Running images in parallel on a GPU is limited by the calculator. That is
  not a defect in the NEB algorithm.
