# Contributing to nebwalk

Thank you for improving nebwalk. Focused bug fixes, tests, documentation, and
reproducible scientific benchmarks are welcome.

## Development workflow

1. Create a branch from `main` and install `pip install -e ".[dev,docs]"`.
2. Add tests that fail before the change and pass afterward.
3. Run the complete local gate:

   ```bash
   ruff check .
   ruff format --check .
   mypy nebwalk
   pytest --cov=nebwalk
   python -m build
   twine check dist/*
   mkdocs build --strict
   ```

4. Update `CHANGELOG.md` for user-visible changes and open a pull request.

Keep public APIs typed and documented. Avoid broad exception handling unless a
calculator boundary requires failure isolation. New dependencies need a clear
benefit and must remain optional unless essential to the core algorithm.

## Scientific changes

Numerical or methodological claims must include units, calculator and model
versions, structures, convergence parameters, random seeds where relevant,
machine-readable results, and primary references. Compare new NEB behavior with
an analytical result or an independent established implementation when possible.
Do not describe cross-model disagreement as calibrated uncertainty.

Generated trajectories, model weights, QE scratch directories, and licensed
pseudopotentials must not be committed. Curated benchmark results should be
small, reviewable, and accompanied by exact reproduction commands.

## Compatibility

nebwalk follows semantic versioning. Removing or changing public names requires
a deprecation period except when repairing unsafe or scientifically incorrect
behavior. Python and ASE support changes must be documented in the changelog.
