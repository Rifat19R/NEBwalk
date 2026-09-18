# Getting started

Install the stable release:

```bash
python -m pip install NEBwalk
```

```python
from ase.calculators.emt import EMT
from NEBwalk import NEBRunConfig, run_neb_calculation

config = NEBRunConfig(
    n_images=7,
    interpolation="idpp",
    climb=True,
    climb_delay=50,
    fmax=0.05,
)
result = run_neb_calculation(
    initial,
    final,
    calculator_factory=EMT,
    config=config,
    reproduce_dir="run-repro",
    calc_params={"calculator": "EMT"},
)
if not result.converged:
    raise RuntimeError("NEB did not converge")
print(result.barrier)
```

Every image receives a fresh calculator instance. Use `n_workers=1` for CUDA
calculators. Endpoint structures should be independently relaxed with compatible
cell, atom ordering, constraints, and calculator settings.
