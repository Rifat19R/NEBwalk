# NEBwalk vs. ASE NEB: implementation-consistency check

This benchmark answers one narrow, reviewer-relevant question: **with
matched physical and path inputs, does NEBwalk's NEB force projection and FIRE
optimizer converge to the same minimum-energy path as ASE's own `ase.mep.NEB`
+ `ase.optimize.FIRE`?**

It is **not** a performance benchmark. No speed or efficiency claim is made
anywhere in this directory or in the manuscript text that cites it.

## What is held identical between the two runs

- Endpoints (built once, shared)
- Number of images (5 intermediate images, 7 total)
- Calculator (same calculator class/factory, fresh instance per image)
- Starting path geometry: intermediate images are interpolated **once**
  with NEBwalk's own MIC-aware `linear_interpolate`, then deep-copied so
  both tools start from byte-identical coordinates. This removes
  interpolator differences as a confound.
- Spring constant (`k = 0.10` eV/A^2, constant — NEBwalk's variable
  `k_min`/`k_max` spring scheme has no direct ASE equivalent, so constant
  `k` is used on both sides for this check)
- Force convergence threshold (`fmax = 0.05` eV/A) and step budget
  (`max_steps = 500`)
- Climbing image **disabled** on both sides (`climb=False`). CI-NEB is out
  of scope for this check: NEBwalk gates climbing with an integer
  `climb_delay` step count, ASE's `climb` flag uses its own internal
  heuristic, and the two timings are not directly comparable.
- Tangent formula: ASE's NEB `method` is pinned explicitly to
  `"improvedtangent"` (Henkelman & Jonsson 2000), the same formula NEBwalk
  implements. This is pinned rather than left at ASE's installed-version
  default because ASE has changed that default between versions and emits
  a `UserWarning` if it is left unset.

## What is deliberately *not* held identical

Each tool runs with **its own FIRE optimizer's own default
hyperparameters** — NEBwalk's `FIREOptimizer` defaults to `dt=0.01,
dt_max=0.10` (`src/NEBwalk/optimize.py`); ASE's `FIRE` defaults to
`dt=0.1, dtmax=1.0`. Neither was retuned to match the other. This means the
two runs take a different number of optimizer steps to converge, and that
difference is expected — it reflects the two tools' own default time-step
scale, not a bug. Reporting a "faster" step count for either side would be
comparing default hyperparameters, not the underlying force/optimizer
algorithm, so we do not draw a speed conclusion from it.

## Systems

**A. Morse H3 collinear exchange** — analytical calculator (sum of two
Morse pairs), 3 atoms, no periodicity. Endpoints and calculator are
identical to `examples/morse_h3.py`.

**B. Al adatom hollow-to-hollow hop on Al(100), EMT** — periodic, 25
atoms, 2x2x3 slab with the bottom two layers fixed. Endpoints and calculator
match `examples/al_diffusion_emt.py`. This provides a lightweight periodic
surface-diffusion test using ASE's built-in EMT calculator.

Both systems run in well under a second on CPU (see `results/*.json` for
measured wall time) and use only built-in ASE calculators (EMT + a small
pure-Python analytical calculator) — no MACE/Egret/QE dependency — so both
are cheap enough to run in CI.

## Metrics reported

For each system: convergence status, number of optimizer steps, forward
and reverse barrier, full per-image energy array, and wall time, for both
tools; plus a direct comparison: barrier difference, max per-image energy
difference, and max per-atom geometry difference between the two converged
paths (minimum-image-convention-aware for the periodic system, via
`ase.geometry.find_mic`, matching NEBwalk's own documented approach to
periodic displacements).

## Results (last run, see `results/*.json` for the authoritative numbers)

| System | NEBwalk barrier (eV) | ASE barrier (eV) | Barrier diff | Max geometry diff | NEBwalk steps | ASE steps |
|---|---|---|---|---|---|---|
| Morse H3 collinear | 0.200251 | 0.200251 | 0.0000 meV | 0.0000 mÅ | see JSON | see JSON |
| Al100 adatom, EMT | 0.237198 | 0.237233 | 0.035 meV | 2.07 mÅ (0.00207 Å) | 53 | 26 |

Both systems converged under both tools. The barrier difference (<0.04 meV
in the worst case here) and the geometry difference (~0.002 A, two orders
of magnitude below a typical bond length) are consistent with two
independently-implemented optimizers converging to the same physical
minimum-energy path under a `fmax = 0.05 eV/A` threshold, not with a
disagreement in the underlying NEB force construction.

**We read this as:** NEBwalk reproduces ASE NEB barriers and converged
geometry within numerical tolerance on these two systems, while adding the
workflow-level provenance and MLIP-to-DFT handoff functionality described
in the manuscript. We do not extrapolate this two-system result to a
general accuracy or performance claim about either tool.

## Reproducing

```bash
pip install -e ".[dev]"   # needs ase and NEBwalk; no MACE/QE/Egret required
python benchmarks/ase_comparison/run_comparison.py
```

Writes `results/morse_h3_collinear_comparison.json`,
`results/al100_adatom_emt_comparison.json`, and `results/summary.csv`.
Machine environment (Python version, platform, processor string) is
recorded in each JSON file's `environment` key.

## Tests

`tests/test_ase_comparison_benchmark.py` exercises both the analytical
Morse H3 comparison and the periodic Al-adatom/Al(100) EMT comparison. The
tests assert convergence and tolerance-based agreement in barrier energies
and path geometry, so both implementation-consistency checks are part of the
normal `pytest` suite rather than standalone examples.
