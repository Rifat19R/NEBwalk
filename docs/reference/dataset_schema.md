# Reference dataset schema

nebwalk stores reference data as ASE extended XYZ. Each frame must contain a
finite scalar `REF_energy` in `Atoms.info` and an `(n_atoms, 3)` finite
`REF_forces` array. `REF_stress` is optional.

Required provenance fields are `config_type`, `campaign_id`, `iteration`,
`path_id`, `image_index`, `selection_reason`, `calculator_name`,
`dft_settings_hash`, and `structure_hash`. The structure hash covers symbols,
positions, cell, and periodic boundary flags; it is not a hash of the labels.

Validation rejects missing or non-finite labels, malformed forces, stale
structure hashes, dangerously short distances, duplicate structures,
conflicting labels, incompatible settings, and inconsistent atom order within
a path. Mixed DFT settings are rejected by default.

```bash
nebwalk dataset validate references.extxyz
nebwalk dataset summarize references.extxyz
nebwalk dataset split references.extxyz split --group-key path_id --seed 17
```

Splits are group-aware: every `path_id` belongs to exactly one of train,
validation, or test. Files and manifests are written atomically and include
SHA-256 checksums. Never construct random frame-level splits for NEB images
from the same path, because that leaks nearly identical geometries.

Dataset metadata records provenance; it does not establish scientific accuracy.
Review the underlying calculator inputs, pseudopotentials, convergence tests,
and raw outputs before using labels as reference data.
