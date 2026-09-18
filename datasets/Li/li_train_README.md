# NEBwalk MACE training-set export

4 configuration(s), path_id='li_vacancy', elements: Li.
sha256: 4f543b865a0d50fe50b60e44113e7740b2d3d82e6e7160a27f0eb06c2065852d

This file was exported by NEBwalk.finetune.export_mace_training_set()
via NEBwalk.datasets.write_dataset() -- it passed that module's real
geometry/duplicate/provenance validation, not just a bespoke check
local to this module. It is correctly *formatted* for MACE, but a
handful of configs from a single NEB path is NOT enough data to
responsibly fine-tune a foundation model -- it validates that the
labeling-to-training pipeline works, not that the resulting checkpoint
would be trustworthy. Real fine-tuning needs many more labeled
configurations across multiple paths/systems/compositions before
training on this data for anything beyond a pipeline smoke test.

Energies/forces are stored as REF_energy/REF_forces (atoms.info /
atoms.arrays), not attached via an ASE calculator with the plain
energy/forces keys -- MACE's own loader warns that the plain keys are
no longer safe to share with ASE (since ASE 3.23) and recommends the
REF_ prefix, which also matches mace.data.utils.DefaultKeys.

This file does NOT contain an IsolatedAtom entry: an isolated atom in a
vacuum cell is evaluated at different QE settings (gamma-only, no
smearing) than the bulk path, so it has a different dft_settings_hash
and NEBwalk.datasets.write_dataset() correctly refuses to mix that into
one dataset file. Its energy reference is instead saved alongside this
file (see isolated_atom_reference.json / save_isolated_atom_reference())
and should be passed to mace_run_train explicitly via --E0s, not fit
from this file with --E0s=average and not taken from the foundation
model's own built-in E0s -- different codes and pseudopotential
families (e.g. VASP/PAW vs. Quantum ESPRESSO/PSL) have unrelated
absolute energy zeros even under the same nominal functional, so
reusing the foundation model's E0s here would reintroduce the same
reference-energy mismatch NEBwalk.label guards against one stage
earlier.
