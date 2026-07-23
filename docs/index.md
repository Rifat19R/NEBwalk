# nebwalk

nebwalk is a compact, transparent implementation of NEB and CI-NEB workflows
for ASE calculators. It is intended for inspectable research workflows—not as
a substitute for calculator validation, convergence studies, or transition-state
verification.

## Capabilities

- MIC-aware linear, IDPP, and overlap-regularized interpolation
- Improved tangents, climbing images, variable springs, and FIRE optimization
- Classical, MLIP, and DFT calculators through ASE
- QE setup validation and failed-image recovery
- Reproducibility bundles and MLIP/DFT disagreement diagnostics

Start with [Getting started](getting-started.md), then read the
[scientific guidance](scientific-guidance.md) before interpreting barriers.
