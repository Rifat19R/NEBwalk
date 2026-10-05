---
title: 'NEBwalk: a calculator-agnostic nudged elastic band library with MLIP-assisted barrier pre-screening'
tags:
  - Python
  - computational chemistry
  - nudged elastic band
  - transition state theory
  - machine learning interatomic potentials
  - MACE
  - Quantum ESPRESSO
  - materials simulation
  - active learning
authors:
  - name: Md. Rifat Khandaker
    orcid: 0000-0003-0520-4654
    affiliation: 1
affiliations:
  - name: Department of Chemical Engineering, Dhaka University of Engineering and Technology, Gazipur 1707, Bangladesh
    index: 1
date: 13 June 2026
bibliography: paper.bib
---

# Summary

NEBwalk is a Python library for nudged elastic band (NEB) calculations. The
user provides two relaxed structures, a reactant and a product. NEBwalk builds
an interpolated path between them, optimizes that path, and reports the
minimum-energy path along with the forward and reverse activation barriers.

The library implements the improved tangent estimate [@henkelman2000a] and the
climbing-image variant [@henkelman2000b], the FIRE optimizer [@bitzek2006], and
IDPP path interpolation [@smidstrup2014]. Spring constants can be held constant
or varied along the path so that images cluster near the saddle point. For
periodic systems, all displacements between images use the minimum-image
convention through ASE [@larsen2017].

The same code runs with four kinds of calculator and no changes beyond the
calculator factory: EMT for fast local tests, the MACE-MP-0 [@batatia2024] and
Egret-1t machine-learned potentials for cheap screening, and Quantum ESPRESSO
[@giannozzi2009] for DFT. Each QE image runs in its own working directory so
that wavefunction files from different images do not overwrite each other.

NEBwalk also includes a two-stage workflow for combining machine-learned
potentials with DFT. It runs the NEB once with a fast potential, picks out a
few images near the barrier (or where two potentials disagree most), and
writes them to disk ready for DFT single-point checks or for an
active-learning dataset. A separate module
saves a full record of any run, including structures, settings, results,
trajectory, file checksums, and a script to repeat the calculation.

# Statement of need

Transition states show up in a lot of materials work: vacancies moving through
a crystal, atoms hopping across a surface, molecules changing shape, and
elementary steps in catalysis. The NEB method [@jonsson1998] is the usual way
to find them, but most NEB code is locked to one engine. The neb.x tool in
Quantum ESPRESSO only drives pw.x. The NEB inside VASP only works with VASP.
Getting a machine-learned potential into either of these takes real effort.

ASE [@larsen2017] ships its own NEB, which is more flexible, and leaves the
choice of optimizer to the user. Quasi-Newton optimizers such as L-BFGS are a
common choice but are not ideal here, because the NEB force is not the
gradient of a single energy function and quasi-Newton methods assume that it
is [@bitzek2006]. NEBwalk uses FIRE everywhere instead, which does not make
that assumption.

The other gap is more recent. Universal machine-learned potentials like
MACE-MP-0 [@batatia2024] are now fast enough that running a full NEB on a
50-atom cell takes minutes on a laptop CPU, where the same calculation in DFT
takes hours. The catch is that the barrier from a universal potential carries a
systematic error against DFT, and how large that error is depends on the
chemistry. The common fix, rerunning the whole NEB in DFT, is expensive and
usually overkill. What is actually needed is a way to use the cheap result to
decide which images deserve a DFT calculation.

NEBwalk does this. It runs the cheap NEB, selects a small number of images
(by default the highest-energy image and its neighbours; optionally the images
where two potentials disagree most), and exports them in a form ready for DFT
or for active learning. This lets a user check a machine-learned path against
DFT where it matters for the barrier without writing their own glue code.
Sparse DFT single points on a machine-learned path are a check of that path,
not a full DFT NEB, and disagreement between potentials is an uncertainty
proxy, not calibrated uncertainty.

The library is small on purpose. The force routine maps line for line onto the
equations in @henkelman2000a, so it is also useful for teaching the method or
for testing changes to it.

# State of the field

The NEB method is widely available, but most implementations are tied either to
a particular electronic-structure engine or to a particular transition-state
workflow. Plane-wave DFT packages such as Quantum ESPRESSO [@giannozzi2009]
provide production NEB drivers, but these drivers are naturally coupled to that
engine's input format, parallel model, and restart files. They are the right
choice for a fully DFT calculation, but they are less convenient when the first
path search should be done with a machine-learned potential and only selected
images should later be checked with DFT.

ASE [@larsen2017] provides a general Python NEB interface and is the closest
software foundation to NEBwalk. NEBwalk builds on the same calculator-agnostic
idea, but narrows the workflow around reproducible barrier screening: FIRE is
the default optimizer, IDPP and minimum-image interpolation are part of the
standard path setup, every QE image gets an isolated working directory, and the
active-learning helper exports the highest-value images for DFT refinement.
This makes the common MLIP-to-DFT screening pattern a first-class workflow
rather than a set of scripts around ASE.

Other transition-state tools focus on different parts of the problem. Sella is
designed for direct saddle-point optimization once a reasonable transition-state
guess is available. Engine-specific NEB drivers focus on robust DFT
execution. NEBwalk occupies the space between them: it is a lightweight Python
library for building and optimizing full paths with any ASE-compatible
calculator, then preserving the path, metadata, and selected images needed for
DFT validation or active learning.

# Implementation

NEBwalk is pure Python and depends on ASE [@larsen2017], NumPy, SciPy, and
Matplotlib. MACE and Egret are optional: importing NEBwalk does not import
PyTorch or MACE, and they are loaded only by the user's calculator factory or
the optional MACE fine-tuning backend. Quantum ESPRESSO support goes through ASE's
Espresso interface and needs a working pw.x and UPF pseudopotential files.

The force routine computes the improved tangent [@henkelman2000a], projects out
the component of the potential force along the tangent, and adds the spring
force along it. The climbing image [@henkelman2000b] turns on at the
highest-energy image after a set number of steps. FIRE [@bitzek2006] carries a
single velocity array across all the moving images. For periodic cells, every
displacement between neighboring images is wrapped with ASE's `find_mic` so
that an atom crossing a cell boundary does not create a false jump in the path.

The two-stage workflow lives in `NEBwalk.active`. The function
`run_mlip_assisted_neb()` takes any ASE calculator factory, runs the NEB, and
hands the energies to the selection module. The default rule,
`peak_plus_neighbors`, keeps the highest-energy image and the images on either
side of it. The selected images are written out as `.xyz`, `.traj`, and `.json`
with their metadata. The selection rule is a named component, so a different
rule can be added without touching the core code.

The reproducibility module is `NEBwalk.reproduce`, with one main function,
`save_bundle()`. It writes the two endpoints in extended XYZ, the full
configuration as JSON, the results and the per-step convergence history, the
whole path trajectory, the output of `pip freeze`, SHA-256 checksums for every
file, and a rerun script. The calculator is deliberately not saved as code. A
Python calculator factory can be a lambda or a closure, and those cannot be
serialized in a reliable way. Instead the user passes a plain dictionary
describing the calculator, which is stored as JSON, and the rerun script leaves
a clearly marked spot for the user to put the factory back. This is more honest
than trying to capture arbitrary source code and more likely to actually work
on another machine.

The automated test suite covers interpolation, the force projection, the
tangent construction, minimum-image handling, variable springs, the two-stage
workflow, the reproducibility bundles, and the QE interface and its failure
recovery. Independent regression tests compare NEBwalk's improved-tangent and
climbing-image forces with ASE's NEB implementation. The package is on PyPI and
is tested on every push through GitHub Actions.

# Software design

NEBwalk is organized around small modules with one responsibility each. The core
NEB force projection and optimizer live separately from interpolation, image
selection, QE calculator setup, and reproducibility output. This keeps the
scientific algorithm easy to inspect while allowing optional workflows, such as
QE-backed calculations or active-learning image export, to sit at the edge of
the package.

The main user-facing object is an ASE `Atoms` path with an attached calculator
factory. Users can provide any calculator that returns energies and forces, and
NEBwalk does not need to know whether that calculator is EMT, a machine-learned
potential, or a DFT wrapper. File-writing code is similarly separated from the
optimization loop, so validation examples can write plots, CSV files,
trajectories, and reproducibility bundles without changing the NEB force
routine. The optional active-learning and fine-tuning modules form outer
layers that the NEB core does not import, a rule the test suite enforces.

# Validation

Two kinds of evidence are kept separate.

**Implementation consistency.** A reproducible benchmark in
`benchmarks/ase_comparison/` runs NEBwalk and ASE's NEB on matched inputs. For
an analytical Morse-H3 collinear path the two give identical barriers. For a
periodic Al adatom on Al(100) with EMT, the barriers differ by about 0.035 meV
and the converged paths by at most about 0.0021 Å. Both implementations keep
their own default FIRE time steps, so step counts are not compared.

**Calculator benchmarks.** The quantitative benchmark claims were re-audited
against the surviving output files in September 2026. Thirteen cases were
reconfirmed (below). Seven older entries could not be re-audited from complete
artefacts and are not counted as validated results.

| System | Calculator | NEBwalk barrier | External comparator | Interpretation |
|---|---|---:|---:|---|
| Morse H3 collinear | analytical Morse | 0.200 eV | 0.193 eV analytical scale | algorithm/path check |
| Al adatom / Al(100) | EMT | 0.237 eV | — | periodic workflow check |
| Al vacancy / FCC Al | MACE-MP-0 | 0.508 eV | 0.58 eV | calculator benchmark |
| H diffusion / Cu(111) | MACE-MP-0 | 0.126 eV | 0.13–0.15 eV | surface-diffusion scale check |
| Cu vacancy / FCC Cu | EMT | 0.755 eV | 0.72 eV | calculator/workflow check |
| Ni vacancy / FCC Ni | EMT | 1.095 eV | 1.08 eV | calculator/workflow check |
| Pd vacancy / FCC Pd | EMT | 0.839 eV | 0.95 eV | calculator/workflow check |
| Ag vacancy / FCC Ag | EMT | 0.682 eV | 0.70 eV | calculator/workflow check |
| Pt vacancy / FCC Pt | EMT | 0.971 eV | 1.24 eV | known calculator limitation |
| Au vacancy / FCC Au | EMT | 0.590 eV | 0.53 eV | calculator/workflow check |
| Au vacancy / FCC Au | MACE-MP-0 | 0.725 eV | 0.53 eV | calculator limitation |
| W vacancy / BCC W | QE/PBE | 1.561 eV | 1.78 eV | DFT-backed benchmark |
| Mo vacancy / BCC Mo | QE/PBE | 1.281 eV | 1.28 eV | DFT-backed benchmark |

The EMT cases are inexpensive workflow and implementation checks. The
MACE-MP-0 and QE/PBE values show how those calculators behave on these systems.
Agreement or disagreement with the literature reflects the calculator and
setup, and is not a measure of the accuracy of NEBwalk itself.

# Research impact statement

NEBwalk is intended for researchers who need activation barriers but cannot run
every candidate path directly in DFT. It supports a practical screening pattern:
run many paths cheaply with a machine-learned potential, identify the important
images, and reserve DFT for the small number of structures that control the
barrier. This is useful for vacancy migration, surface diffusion, catalytic
elementary steps, and other materials problems where the same transition search
must be repeated across many sites, compositions, or adsorbates.
NEBwalk is being used in an ongoing study of hydrogen migration on Ti₃C₂O₂,
Ti₃C₂(OH)₂, and Ti₃C₂F₂ MXene surfaces with QE/PBE. That study is not yet
published and is not part of the validation reported here.

The package also has educational value. The implementation keeps the tangent
construction, force projection, climbing-image modification, and FIRE update in
plain Python. This makes the method easier to audit than a monolithic engine
driver and gives students or method developers a compact reference
implementation that still works with real calculators.

# AI usage disclosure

Claude and OpenAI Codex were used during development for code review, debugging,
documentation editing, and assistance with example scripts. They were not used
as scientific authorities. All equations, validation choices, benchmark
interpretation, and final claims were checked by the author against the code,
the generated outputs, and the cited literature.

# Acknowledgements

I thank Prof. Dr. Mohammad Asaduzzaman Chowdhury (DUET) for supervision and
support. I used Claude (Anthropic) and OpenAI Codex during development for code
review, debugging, and help with the documentation. The scientific choices, the
algorithm and validation work, and responsibility for the code are mine.

# References
