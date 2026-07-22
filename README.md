# nebwalk

[![CI](https://github.com/Rifat19R/nebwalk/actions/workflows/ci.yml/badge.svg)](https://github.com/Rifat19R/nebwalk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nebwalk.svg)](https://pypi.org/project/nebwalk/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/nebwalk/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**nebwalk** is a lightweight, transparent, ASE-compatible Nudged Elastic Band
(NEB/CI-NEB) library for minimum-energy paths, transition-state barriers, and
diffusion mechanisms in atomistic simulations.

It is designed for fast prototyping with classical and machine-learned
calculators, while remaining compatible with DFT backends through ASE.

```bash
pip install nebwalk
```

Current source version: **v0.10.0**.

---

## Why nebwalk?

`nebwalk` focuses on one task: making NEB workflows simple, inspectable, and
calculator-agnostic.

It is useful when you want to:

- build and optimize minimum-energy paths;
- run standard NEB or climbing-image NEB;
- switch between EMT, MACE-MP-0, Egret-1t, Quantum ESPRESSO, VASP, or any ASE-compatible calculator;
- prototype surface diffusion, vacancy migration, adsorbate hopping, and molecular conformational changes;
- export profiles as plots, CSV files, and ASE trajectories;
- keep the NEB implementation transparent enough to inspect, test, and modify.

This is not a black-box workflow manager. It is a compact research-code layer
around ASE calculators and atomic structures.

---

## Features

- Standard NEB and **climbing-image NEB (CI-NEB)**.
- Improved tangent estimate following Henkelman and Jónsson.
- Perpendicular potential-force projection and spring-force projection.
- **FIRE optimizer**, suitable for the non-conservative NEB force field.
- Linear, IDPP, and regularized geodesic-style interpolation.
- Minimum-image-convention-aware interpolation and NEB forces for periodic systems.
- Variable spring constants to concentrate images near high-energy regions.
- High-level `run_neb_calculation()` API with calculator factories.
- Thread-based parallel image evaluation for CPU calculators.
- **GPU note:** thread-parallel image evaluation (`n_workers > 1`) is only safe
  with CPU calculators. With CUDA-backed calculators (MACE-MP-0 or Egret-1t on
  GPU), use `n_workers=1`. GPU parallelism across images requires per-image
  CUDA streams or process isolation, which is out of scope for nebwalk.
- Restart from ASE `.traj` files with fresh calculator instances.
- Energy-profile plotting, CSV export, and `.traj` output.
- Quantum ESPRESSO helper layer through ASE calculator construction.
- Automatic failed-image recovery for Quantum ESPRESSO image calculations:
  convergence failures and geometry instabilities can be retried with logged
  recovery attempts; non-retryable process failures stop immediately.

### MLIP-assisted workflow (v0.7.0+)

In addition to the v0.6.x NEB/CI-NEB engine, v0.7.0 adds:

- MLIP-assisted NEB workflow through `run_mlip_assisted_neb()`.
- Selection step of an active-learning workflow.
- `peak_plus_neighbors` selection of the barrier-sensitive image and neighboring images.
- `uncertainty_disagreement` selection by cross-model disagreement
  (uncertainty proxy) between Egret-1t and MACE-OFF23 for exercised organic
  systems.
- Selected-image export as `.xyz`, `.traj`, and `.json`.
- Clean handoff from MLIP/MACE NEB to DFT/QE refinement.

---

## Installation

### Stable installation from PyPI

```bash
pip install nebwalk
```

Optional MACE support:

```bash
pip install "nebwalk[mace]"
```

Check the installed version:

```bash
python -c "import nebwalk; print(nebwalk.__version__ if hasattr(nebwalk, '__version__') else 'installed')"
```

### Latest source from GitHub

```bash
pip install git+https://github.com/Rifat19R/nebwalk.git
```

### Development installation

```bash
git clone https://github.com/Rifat19R/nebwalk.git
cd nebwalk
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Quick start

```python
from ase.calculators.emt import EMT
from nebwalk import NEBRunConfig, run_neb_calculation

initial = ...  # relaxed ase.Atoms endpoint
final = ...    # relaxed ase.Atoms endpoint

config = NEBRunConfig(
    n_images=7,
    interpolation="idpp",
    k=0.1,
    k_min=0.033,
    climb=True,
    climb_delay=50,
    fmax=0.05,
    max_steps=500,
)

result = run_neb_calculation(
    initial=initial,
    final=final,
    calculator_factory=lambda: EMT(),
    config=config,
)

print(f"Converged       : {result.converged}")
print(f"Barrier         : {result.barrier:.3f} eV")
print(f"Reverse barrier : {result.reverse_barrier:.3f} eV")
print(f"Reaction energy : {result.reaction_energy:.3f} eV")

result.neb.plot("profile.png")
result.neb.save_csv("profile.csv")
result.neb.save_trajectory("path.traj")
```

The calculator factory must return a fresh calculator instance. Do not share one
ASE calculator object across all images.

---

## MLIP-assisted NEB workflow

`nebwalk` can now run a fast calculator/MLIP NEB first, identify the
barrier-sensitive images, and export those images for higher-level DFT/QE
refinement.

This is the selection step of an active-learning workflow. It does not retrain
the MLIP, automatically label data, or adaptively insert/remove images.

```python
from ase.calculators.emt import EMT
from nebwalk import NEBRunConfig
from nebwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb

result = run_mlip_assisted_neb(
    initial=initial,
    final=final,
    mlip_calculator_factory=lambda: EMT(),  # replace with MACE/Egret/custom MLIP
    neb_config=NEBRunConfig(n_images=7, interpolation="idpp", climb=True),
    active_config=MLIPActiveNEBConfig(
        selection_strategy="peak_plus_neighbors",
        n_select=3,
        output_dir="selected_for_qe",
    ),
)

print(result.mlip_barrier)
print(result.selected_indices)
```

The selected images are intended for DFT/QE refinement, single-point validation,
or later labeling in an active-learning workflow.

### Cross-model disagreement selection (v0.10.0)

`uncertainty_disagreement` ranks images by cross-model disagreement
(uncertainty proxy) between Egret-1t as the primary production calculator and
MACE-OFF23 as the secondary disagreement calculator. This v0.10.0 path is
organic-domain-only and is currently exercised in nebwalk on ethane torsion.
Do not use this Egret-1t disagreement workflow with MACE-MP-0 or with
inorganic/vacancy systems.

```python
from mace.calculators import MACECalculator, mace_off
from nebwalk import NEBRunConfig
from nebwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb

def make_egret():
    return MACECalculator(
        model_paths="EGRET_1T.model", device="cpu", default_dtype="float32"
    )

def make_mace_off23():
    return mace_off(model="medium", device="cpu", default_dtype="float64")

result = run_mlip_assisted_neb(
    initial=ethane(60.0),
    final=ethane(180.0),
    mlip_calculator_factory=make_egret,
    neb_config=NEBRunConfig(n_images=7, interpolation="idpp", climb=True),
    active_config=MLIPActiveNEBConfig(
        selection_strategy="uncertainty_disagreement",
        secondary_calculator_factory=make_mace_off23,
        n_select=3,
        output_dir="ethane_disagreement_selected",
    ),
)
```

Selection used cross-model disagreement between two independently-trained MLIPs
as an uncertainty proxy, not a calibrated uncertainty quantification method (no
committee/ensemble was trained). This proxy is only meaningful where both
calculators are within their validated chemical domain; consult nebwalk's
documented domain-failure list before trusting results outside that domain.

---

## Example with MACE-MP-0

```python
from mace.calculators import mace_mp
from nebwalk import NEBRunConfig, run_neb_calculation

def make_calc():
    return mace_mp(
        model="medium",
        dispersion=False,
        default_dtype="float64",
        device="cpu",  # use "cuda" if available
    )

config = NEBRunConfig(
    n_images=5,
    interpolation="idpp",
    k=0.5,
    climb=True,
    climb_delay=50,
    fmax=0.03,
    max_steps=1000,
)

result = run_neb_calculation(initial, final, make_calc, config)
```

MACE-MP-0 is a PBE-level foundation model. It can be excellent for rapid
prototyping, but absolute barriers should be validated against DFT or experiment
for the specific chemistry.

---

## What materials can nebwalk handle?

`nebwalk` operates on ASE `Atoms` objects, so the practical materials space is
defined by the calculator you attach.

Currently demonstrated or directly supported workflow classes include:

| Class | Examples |
|---|---|
| Molecular paths | H3 Morse benchmark, ethane torsion |
| Surface diffusion | Al/Cu/Ni/Pd/Ru slab adsorbate hopping, H on Cu(111) |
| Bulk vacancy migration | fcc metals, hcp Mg, simple oxides |
| Ionic migration | MgO, Li2O-style vacancy/interstitial paths |
| 2D/slab systems | MXenes, MAX phases, graphene-derived slabs, catalytic surfaces |
| DFT-backed systems | Quantum ESPRESSO via ASE calculator helpers |

The code is calculator-agnostic. The scientific reliability of any result still
depends on the chosen calculator, pseudopotentials, k-points, slab thickness,
coverage, spin state, and convergence settings.

---

## Benchmark spotlight: H diffusion on Cu(111)

A reproducible benchmark is included in:

```text
benchmarks/h_cu111_mace_mp/
```

The measured result uses MACE-MP-0 with a 4×4×4 Cu(111) slab, one H adatom
(1/16 ML), 15 Å vacuum, bottom two Cu layers fixed, IDPP interpolation, and
CI-NEB.

### Elementary fcc → hcp hop

| Quantity | Result |
|---|---:|
| Calculator | MACE-MP-0 small |
| Path | fcc hollow → bridge-like TS → hcp hollow |
| Internal images | 5 |
| Converged | True |
| Wall time | 2.67 min |
| Forward barrier | 125.92 meV |
| Reverse barrier | 136.67 meV |
| Reaction energy | −10.75 meV |
| TS image | 3 |

The profile is a clean single-saddle elementary hop:

```text
image 00:   +0.000 meV   fcc initial
image 01:  +44.614 meV
image 02: +111.234 meV
image 03: +125.917 meV   transition state
image 04: +104.770 meV
image 05:  +33.294 meV
image 06:  -10.749 meV   hcp final
```

### fcc → fcc full hop

A longer fcc-to-fcc hop also converged. It shows the expected two-saddle
sequence:

```text
fcc → bridge TS → hcp hollow → bridge TS → fcc
```

| Quantity | Result |
|---|---:|
| Calculator | MACE-MP-0 small |
| Internal images | 7 |
| Converged | True |
| Wall time | 6.42 min |
| Forward barrier | 125.56 meV |
| Reverse barrier | 125.41 meV |
| Reaction energy | +0.146 meV |

This confirms endpoint equivalence and path symmetry.

### Interpretation

The Cu(111) benchmark validates the NEB workflow: endpoint handling,
minimum-image interpolation, CI-NEB optimization, image-energy reporting, and
profile export. The absolute MACE barrier should not be overinterpreted as an
experimental activation energy. Experiments on H/Cu(111) report macroscopic
diffusion behavior affected by thermal activation, quantum effects, coverage,
and surface morphology. MACE-MP-0 is trained to reproduce PBE-level energetics,
so a lower static flat-terrace barrier is expected.

---

## Validated results

> EMT is included for fast local testing only — no GPU, no download required.
> For quantitative results use MACE-MP-0 or DFT (QE). Errors attributed to
> the calculator are documented explicitly; convergence of the NEB path itself
> is verified in every case by checking endpoint ΔE ≈ 0 for symmetric hops.

20 systems validated across 4 calculators.

| System | Calculator | Purpose | Barrier | Reference | Error |
|---|---|---|---|---|---|
| Morse H3 collinear | Morse (analytical) | algorithm test | 0.200 eV | 0.193 eV (exact) | 4% |
| Al adatom / Al(100) | EMT | speed test | 0.237 eV | ~0.40 eV (DFT-PBE) | finite-size slab |
| Cu adatom / Cu(100) | EMT | speed test | 0.418 eV | ~0.44 eV (DFT-PBE) | 4.6% |
| Ni adatom / Ni(100) | EMT | speed test | 0.555 eV | ~0.63 eV (DFT-GGA) | 12%§ |
| Ethane C–C torsion | Egret-1t | ML organic | 0.113 eV | 0.126 eV (exp.) | 10% |
| Al vacancy / FCC Al | MACE-MP-0 | benchmark | 0.508 eV | 0.61 eV (DFT-PBE) | 17%† |
| Mg vacancy / HCP Mg | MACE-MP-0 | benchmark | 0.508 eV | ~0.52 eV (DFT-PBE) | 2% |
| Li vacancy / Li₂O | MACE-MP-0 | benchmark | 0.284 eV | ~0.28 eV (DFT-GGA) | 1.4% |
| Mg vacancy / MgO | MACE-MP-0 | benchmark | 2.254 eV | ~2.20 eV (DFT-PBE) | 2.5% |
| H diffusion / Cu(111) | MACE-MP-0 | benchmark | 0.126 eV | ~0.15 eV (DFT-PBE) | ~16%¶ |
| Cu vacancy / FCC Cu | EMT | speed test | 0.755 eV | ~0.70 eV (DFT-PBE) | 7.9% |
| Ni vacancy / FCC Ni | EMT | speed test | 1.095 eV | ~1.04 eV (DFT-PBE) | 5.3% |
| Pd vacancy / FCC Pd | EMT | speed test | 0.839 eV | ~0.91 eV (DFT-PBE) | 7.8% |
| Ag vacancy / FCC Ag | EMT | speed test | 0.682 eV | ~0.66 eV (DFT-PBE) | 3.3% |
| Pt vacancy / FCC Pt | EMT | speed test | 0.971 eV | ~1.49 eV (DFT-PBE) | 34.8%‡ |
| Au vacancy / FCC Au | EMT | speed test | 0.590 eV | ~0.68 eV (DFT-PBE) | 13.3% |
| Au vacancy / FCC Au | MACE-MP-0 | benchmark | 0.725 eV | ~0.68 eV (DFT-PBE) | 6.7% |
| Al vacancy / FCC Al | QE/PBE     | benchmark | 0.564 eV | ~0.61 eV (DFT-PBE) | 7.6% |
| W vacancy / BCC W   | QE/PBE     | benchmark | 1.561 eV | ~1.66 eV (DFT-PBE) | 5.9% |
| Mo vacancy / BCC Mo | QE/PBE     | benchmark | 1.281 eV | ~1.35 eV (DFT-PBE) | 5.1% |

† MACE-MP-0 systematically underestimates vacancy migration barriers by
10–20%. Known model limitation, not a nebwalk bug.

‡ EMT does not capture relativistic effects in Pt. NEB converged cleanly
in 60 steps — error is from the calculator.

§ Ni adatom on Ni(100): EMT underestimates barriers due to d-band character
at the Ni(100) saddle-point geometry. Profile shape and convergence correct.

¶ H/Cu(111): surface diffusion benchmark. Error reflects MACE-MP-0 surface
accuracy limitations, not a library bug.

---

## Other examples

```bash
python examples/morse_h3.py
python examples/mlip_assisted_neb_emt.py
python examples/al_diffusion_emt.py
python examples/ethane_egret.py
python examples/ethane_egret_maceoff23_disagreement.py
python examples/propane_egret_maceoff23_disagreement.py
python examples/methanol_egret_maceoff23_disagreement.py
python examples/ethanol_egret_maceoff23_disagreement.py
python examples/dimethyl_ether_egret_maceoff23_disagreement.py
python examples/al_vacancy_macemp.py
python examples/mg_vacancy_macemp.py
python examples/li2o_vacancy_macemp.py
python examples/cu_adatom_cu100_emt.py
python examples/ni_adatom_ni100_emt.py
python examples/mg_vacancy_mgo_macemp.py
python examples/al_diffusion_qe.py
python examples/al_vacancy_qe.py
```

A larger benchmark suite (38 systems, EMT/MACE/Egret) is available in `examples/v8/`.

Organic Egret-1t/MACE-OFF23 disagreement-selection runs, per-image diagnostics,
and literature-scale torsional-barrier comparisons are in
`examples/organic_disagreement_literature_comparison_v0.10.0.md`. These are
exploratory organic-domain checks, not DFT validation or calibrated uncertainty
quantification. Acetaldehyde was tested but excluded from the pushed example set
because its barrier was a clear outlier and default force-disagreement selection
was endpoint-adjacent.

For a clean Al vacancy QE rerun, remove stale partial QE outputs through the
script's explicit clean flag:

```bash
export ESPRESSO_PSEUDO=$HOME/pseudo
export AL_PSEUDO=Al.pbe-n-kjpaw_psl.1.0.0.UPF
export ESPRESSO_COMMAND="/absolute/path/to/pw.x"
export NEBWALK_QE_CLEAN=1
python examples/al_vacancy_qe.py
tail -f al_vacancy_qe.log
```

Some calculators are intentionally optional. Egret-1t model files and MACE model
weights are not distributed with this repository.

---

## Quantum ESPRESSO Interface

nebwalk includes a built-in interface to [Quantum ESPRESSO](https://www.quantum-espresso.org/)
via `nebwalk.qe`. Use this for production NEB calculations where universal MLIPs are out of
distribution — surface reactions, MXene catalysis, MAX phase defects.

```python
from nebwalk import run_neb_calculation, NEBRunConfig
from nebwalk.qe import QEParams, make_qe_factory, validate_qe_setup

params = QEParams(
    ecutwfc                = 60.0,       # plane-wave cutoff (Ry)
    ecutrho                = 480.0,      # 8× ecutwfc for PAW
    kpts                   = (4, 4, 1), # k-point grid for metal slab
    occupations            = "smearing",
    smearing               = "marzari-vanderbilt",
    degauss                = 0.02,
    nspin                  = 2,          # required for Fe, Co, Ni, Mn
    starting_magnetization = {1: 0.5},   # species 1 = 50% spin-up
)

# Validates pw.x binary and all UPF files exist before starting
validate_qe_setup(
    pseudo_dir       = "/path/to/pseudo",
    pseudopotentials = {"Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
                        "N":  "N.pbe-n-radius_5.UPF"},
    command          = "/absolute/path/to/pw.x",
)

# Each image gets an independent subdirectory — prevents wavefunction conflicts
factory = make_qe_factory(
    params           = params,
    pseudo_dir       = "/path/to/pseudo",
    pseudopotentials = {"Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
                        "N":  "N.pbe-n-radius_5.UPF"},
    base_dir         = "neb_qe_workdir",
    command          = "/absolute/path/to/pw.x",   # or "mpirun -np 4 /absolute/path/to/pw.x"
)

result = run_neb_calculation(
    initial            = initial,
    final              = final,
    calculator_factory = factory,
    config             = NEBRunConfig(n_images=7, climb=True, fmax=0.05),
)
print(f"Barrier: {result.barrier:.3f} eV")
```

**`QEParams` key parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `ecutwfc` | 40.0 | Plane-wave cutoff (Ry). Check your pseudopotential's recommended value. |
| `ecutrho` | 320.0 | Charge density cutoff (Ry). Use 8× ecutwfc for USPP/PAW, 4× for NC. |
| `kpts` | (1,1,1) | k-point grid. (1,1,1) for molecules; (4,4,1) for metal slabs. |
| `nspin` | 1 | Set to 2 for magnetic systems (Fe, Co, Ni, Mn). Required — not optional. |
| `starting_magnetization` | None | Dict of {species_index: value}. Required when nspin=2. |
| `conv_thr` | 1e-8 | SCF convergence threshold (Ry). Tight convergence reduces NEB force noise. |

**Requirements:** Quantum ESPRESSO ≥ 6.8, a working `pw.x` executable, and UPF
pseudopotential files. Prefer an absolute `pw.x` path in scripted runs so ASE
subprocesses do not depend on an interactive-shell `PATH`.
Recommended set: [SSSP Efficiency](https://www.materialscloud.org/discover/sssp) (PBE).

See `examples/template_qe_neb.py` for a complete annotated template and
`examples/al_vacancy_qe.py` for a concrete Al vacancy QE/PBE benchmark script.

---

## Reproducibility

nebwalk can export a self-contained bundle for every NEB run. The bundle
contains input structures, configuration, results, energy profile, trajectory,
software environment, SHA-256 checksums, and a rerun template.

**Via `run_neb_calculation`:**

```python
from nebwalk import run_neb_calculation, NEBRunConfig

result = run_neb_calculation(
    initial, final, calculator_factory,
    config=NEBRunConfig(n_images=7, climb=True),
    reproduce_dir="my_run_repro",
    calc_params={"type": "MACE-MP-0", "model": "small", "device": "cpu"},
)
```

**Directly:**

```python
from nebwalk import save_bundle

bundle = save_bundle(
    result, initial, final, config,
    calc_params={"type": "EMT"},
    compress=True,
)
print(bundle.tarball)   # Path to .tar.gz
```

**Verify a stored run:**

```bash
cd my_run_repro
# Edit rerun_template.py to restore your calculator
python rerun_template.py
```

The rerun script prints the original and recomputed barriers and flags
differences larger than 5 meV.

---

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

The test suite covers interpolation, NEB force projection, tangent
construction, minimum-image convention handling, variable springs, parallel
image evaluation, restart helpers, calculator-factory workflows, and
reproducibility bundles, active-workflow helpers, and QE failed-image recovery.

---

## Roadmap

Short-term priorities:

1. Keep PyPI, GitHub tags, and source metadata synchronized for v0.9.x releases.
2. Add H/Cu(111) benchmark scripts, static output CSVs, and profile plots.
3. Expand DFT-backed Quantum ESPRESSO benchmarks with small inputs and clear cost warnings.
4. Add post-NEB transition-state refinement using a dimer method.
5. Build a documentation site with theory, API usage, calculator setup, and benchmarks.

Future active-learning roadmap:

- DFT/MLIP hybrid barrier correction
- adaptive image insertion/removal
- benchmark-grade reproducibility archives

Long-term priorities:

- dimer/RFO transition-state refinement;
- imaginary-mode validation utilities;
- better benchmark provenance files;
- larger measured MACE-medium/large CPU/GPU parallel-scaling benchmarks;
- more surface diffusion and ionic migration examples.

---

## Release checklist for maintainers

Use this when publishing a new release:

```bash
python -m pip install --upgrade build twine
rm -rf dist/ build/ *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
VERSION=v0.10.0
git tag -a "$VERSION" -m "nebwalk $VERSION"
git push origin main "$VERSION"
```

After release:

```bash
pip install --upgrade nebwalk
python -c "import nebwalk; print(nebwalk.__version__ if hasattr(nebwalk, '__version__') else 'installed')"
```

---

## References

1. H. Jónsson, G. Mills, K. W. Jacobsen, *Nudged Elastic Band Method for Finding Minimum Energy Paths of Transitions*, World Scientific, 1998.
2. G. Henkelman and H. Jónsson, *Improved tangent estimate in the nudged elastic band method for finding minimum energy paths and saddle points*, J. Chem. Phys. 113, 9978 (2000).
3. G. Henkelman, B. P. Uberuaga, and H. Jónsson, *A climbing image nudged elastic band method for finding saddle points and minimum energy paths*, J. Chem. Phys. 113, 9901 (2000).
4. E. Bitzek, P. Koskinen, F. Gähler, M. Moseler, and P. Gumbsch, *Structural Relaxation Made Simple*, Phys. Rev. Lett. 97, 170201 (2006).
5. S. Smidstrup, A. Pedersen, K. Stokbro, and H. Jónsson, *Improved initial guess for minimum energy path calculations*, J. Chem. Phys. 141, 214106 (2014).
6. Batatia et al., A foundation model for atomistic materials chemistry, arXiv:2401.00096.
7. H/Cu(111) diffusion literature should be treated as macroscopic diffusion reference, not a one-to-one static NEB barrier target.

---
## Acknowledgments

The development of **nebwalk** benefited from selective AI-assisted support using **Claude** and **OpenAI Codex/ChatGPT** for code review, debugging guidance, documentation refinement, and release-workflow cleanup.

The scientific direction, algorithmic design, implementation decisions, validation strategy, benchmark interpretation, and release responsibility remain fully maintained by **Md. Rifat Khandaker**. AI tools were used only as auxiliary development aids to improve clarity, review consistency, and workflow efficiency.
---

## License

MIT License. See [LICENSE](LICENSE).

## Features in v0.10.0

In addition to the v0.9.0 QE recovery workflow, v0.10.0 adds:

- `uncertainty_disagreement` image selection by cross-model disagreement
  (uncertainty proxy).
- `nebwalk.uncertainty` with `DisagreementResult` and
  `compute_cross_model_disagreement()`.
- Optional secondary calculator factory in `MLIPActiveNEBConfig`.
- Exported `energy_disagreement` and `force_disagreement` metadata for selected
  images.
- Ethane Egret-1t/MACE-OFF23 disagreement example and MACE-OFF23 sanity-check
  script.

## Features in v0.9.0

In addition to the v0.8.0 reproducibility workflow, v0.9.0 adds:

- Automatic failed-image recovery for Quantum ESPRESSO image calculations.
- Generic recovery primitives in `nebwalk.recovery`.
- QE-specific `QERecoveryStrategy` with retryable convergence and geometry
  failure handling.
- Recovery-attempt logging through `neb.recovery_log`.
- Optional recovery-log serialization in reproducibility bundles.

## Features in v0.8.0

In addition to the v0.7.x engine and MLIP workflow, v0.8.0 adds:

- Reproducibility bundles via `nebwalk.reproduce.save_bundle()`.
- Self-contained output directory: input structures, NEBRunConfig, results,
  trajectory, pip environment capture, SHA-256 checksums, and a rerun script.
- Optional `.tar.gz` archive of the full bundle.
- `run_neb_calculation()` accepts `reproduce_dir` and `calc_params` keyword
  arguments for one-line bundle creation.
