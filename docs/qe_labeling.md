# Quantum ESPRESSO reference labeling

`QEReferenceLabeler` performs independent fixed-geometry single-point
calculations through ASE. Supply an explicit executable, pseudopotential
directory, element-to-UPF mapping, and converged `QEParams`.

Each input receives an isolated raw directory. Successful structures are saved
with canonical reference keys; failures remain explicit in the manifest and do
not erase successful labels. The settings hash covers QE parameters,
pseudopotential names, executable, and resolved pseudo directory.

Interrupted batches can resume. Existing results are reused only when the
manifest, extxyz count, structure identities, and settings hash agree. Recovery
attempts and any recovery-induced geometry change are recorded. A changed
geometry receives a new structure hash.

Before production labeling:

1. converge cutoffs, k-points, smearing, spin, and cell treatment;
2. verify every UPF file and its provenance;
3. run a small isolated smoke calculation with the exact command;
4. inspect forces and raw output, not only the return code;
5. retain raw QE directories outside version control.

Automated tests use injected fakes and do not prove that a local QE or MPI
installation is operational. QE labels are evidence only within the documented
numerical settings and convergence study.
