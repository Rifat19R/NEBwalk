"""Implementation-consistency check: NEBwalk's NEB/FIRE vs. ASE's own NEB/FIRE.

Purpose
-------
This is NOT a performance benchmark and makes no speed claim. It checks
whether NEBwalk's NEB force projection and path optimizer converge to the
same minimum-energy path as ASE's own `ase.mep.NEB` + `ase.optimize.FIRE`
when both start from an *identical* set of interpolated images, use the
same calculator, the same spring constant, the same convergence threshold,
and climbing images disabled on both sides (climb=False). Climbing-image
timing differs by construction between the two implementations (NEBwalk
gates it with an integer `climb_delay` step count; ASE's `climb` flag
follows its own internal heuristic), so CI-NEB equivalence is out of scope
for this specific check.

Two systems are used:
  A. Morse H3 collinear exchange -- analytical calculator, 3 atoms, no PBC.
     Endpoints/calculator match `examples/morse_h3.py` exactly.
  B. Al adatom hollow-to-hollow hop on Al(100), EMT -- periodic, 25 atoms.
     Endpoints/calculator match `examples/al_diffusion_emt.py`. This provides
     a lightweight periodic surface-diffusion comparison using ASE's built-in
     EMT calculator. Constant k is used here (no variable spring constant),
     since ASE's NEB does not expose the same
     k_min/k_max scheme as NEBwalk's `variable_spring_constants` -- this
     is a deliberate simplification to keep the two runs comparable, not
     an attempt to hide a difference.

Design
------
Intermediate images are interpolated exactly once, with NEBwalk's own
`linear_interpolate` (MIC-aware). Two independent deep copies of that same
image list are then built: one is handed to NEBwalk's `NEB`, the other to
`ase.mep.NEB`. This removes "did the two interpolators produce different
starting geometries" as a possible source of disagreement, isolating the
comparison to the force projection + optimizer step.

Known, disclosed hyperparameter difference
-------------------------------------------
NEBwalk's `FIREOptimizer` defaults to dt=0.01, dt_max=0.10
(`src/NEBwalk/optimize.py`). ASE's `ase.optimize.FIRE` defaults to dt=0.1,
dtmax=1.0. Both sides are run with their own tool's default FIRE
hyperparameters (neither is retuned to match the other), so a difference in
the number of optimizer steps to converge is expected and is not itself
evidence of a bug; it reflects a real difference in the two FIRE
implementations' default time-step scale. This is reported, not hidden.

Usage
-----
    python benchmarks/ase_comparison/run_comparison.py

Writes one JSON file per system and one summary CSV into
`benchmarks/ase_comparison/results/`. Requires `ase` and `NEBwalk`
(no MACE/QE/Egret dependency; both systems use built-in ASE calculators).
"""

from __future__ import annotations

import copy
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ase
import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.mep import NEB as ASE_NEB
from ase.optimize import BFGS
from ase.optimize import FIRE as ASE_FIRE

import NEBwalk
from NEBwalk import NEB as NEBWALK_NEB
from NEBwalk import linear_interpolate

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Shared convergence settings applied identically to both tools.
N_IMAGES = 5
K_SPRING = 0.10
FMAX = 0.05
MAX_STEPS = 500


@dataclass
class RunMetrics:
    tool: str
    system: str
    converged: bool
    n_steps: int
    barrier_eV: float
    reverse_barrier_eV: float
    reaction_energy_eV: float
    energies_eV: list[float]
    wall_time_s: float
    optimizer_defaults: dict[str, float]


# ---------------------------------------------------------------------------
# System A: Morse H3 collinear exchange (matches examples/morse_h3.py)
# ---------------------------------------------------------------------------

D, A_PARAM, R0 = 4.52, 1.93, 0.74


def _morse_pair(r: float) -> tuple[float, float]:
    u = np.exp(-A_PARAM * (r - R0))
    E = D * (1.0 - u) ** 2 - D
    dEdr = 2.0 * D * A_PARAM * u * (1.0 - u)
    return E, dEdr


class MorsePairH3:
    """SEAM calculator for collinear H3: E = V(r_HA-HB) + V(r_HB-HC)."""

    def get_potential_energy(
        self, atoms: Atoms, force_consistent: bool = False
    ) -> float:
        r_ab = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
        r_bc = float(np.linalg.norm(atoms.positions[2] - atoms.positions[1]))
        e_ab, _ = _morse_pair(r_ab)
        e_bc, _ = _morse_pair(r_bc)
        return e_ab + e_bc

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        pos = atoms.positions
        r_vec_ab = pos[1] - pos[0]
        r_vec_bc = pos[2] - pos[1]
        r_ab = float(np.linalg.norm(r_vec_ab))
        r_bc = float(np.linalg.norm(r_vec_bc))
        _, dEdr_ab = _morse_pair(r_ab)
        _, dEdr_bc = _morse_pair(r_bc)
        r_hat_ab = r_vec_ab / r_ab
        r_hat_bc = r_vec_bc / r_bc
        f = np.zeros((3, 3))
        f[0] += dEdr_ab * r_hat_ab
        f[1] -= dEdr_ab * r_hat_ab
        f[1] += dEdr_bc * r_hat_bc
        f[2] -= dEdr_bc * r_hat_bc
        return f


def build_morse_h3() -> tuple[Atoms, Atoms, list[int]]:
    """Return (initial, final, fixed_indices_for_intermediate_images)."""
    start = Atoms("H3", positions=[[0.00, 0, 0], [0.74, 0, 0], [2.50, 0, 0]])
    end = Atoms("H3", positions=[[0.00, 0, 0], [1.76, 0, 0], [2.50, 0, 0]])
    return start, end, [0, 2]


# ---------------------------------------------------------------------------
# System B: Al adatom hollow-to-hollow hop on Al(100), EMT
# (matches examples/al_diffusion_emt.py)
# ---------------------------------------------------------------------------


def _make_slab() -> Atoms:
    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    add_adsorbate(slab, "Al", height=1.7, position="hollow")
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
    slab.calc = EMT()
    return slab


def build_al100_adatom() -> tuple[Atoms, Atoms, list[int]]:
    initial = _make_slab()
    BFGS(initial, logfile=None).run(fmax=0.01)
    final = initial.copy()
    final.calc = EMT()
    adatom_idx = len(final) - 1
    final.positions[adatom_idx] += np.array([4.05 / 2, 0.0, 0.0])
    BFGS(final, logfile=None).run(fmax=0.01)
    fixed_indices = [atom.index for atom in initial if atom.tag > 1]
    return initial, final, fixed_indices


# ---------------------------------------------------------------------------
# Shared run/compare machinery
# ---------------------------------------------------------------------------


def build_shared_images(
    initial: Atoms, final: Atoms, fixed_indices: list[int], constrain_endpoints: bool
) -> list[Atoms]:
    """Interpolate once with NEBwalk's own MIC-aware linear interpolator."""
    images = linear_interpolate(initial, final, n_images=N_IMAGES)
    lo = 0 if constrain_endpoints else 1
    hi = len(images) if constrain_endpoints else len(images) - 1
    for img in images[lo:hi]:
        if fixed_indices:
            img.set_constraint(FixAtoms(indices=fixed_indices))
    return images


def run_nebwalk_side(
    images: list[Atoms], calc_factory, system: str
) -> tuple[RunMetrics, list[Atoms]]:
    imgs = copy.deepcopy(images)
    for img in imgs:
        img.calc = calc_factory()
    neb = NEBWALK_NEB(imgs, k=K_SPRING, climb=False)
    t0 = time.perf_counter()
    converged = neb.optimize(fmax=FMAX, max_steps=MAX_STEPS, verbose=False)
    wall = time.perf_counter() - t0
    metrics = RunMetrics(
        tool="NEBwalk",
        system=system,
        converged=bool(converged),
        n_steps=len(neb.history),
        barrier_eV=neb.get_barrier(),
        reverse_barrier_eV=neb.get_reverse_barrier(),
        reaction_energy_eV=neb.get_reaction_energy(),
        energies_eV=neb.get_energies(),
        wall_time_s=wall,
        optimizer_defaults={"dt": 0.01, "dt_max": 0.10},
    )
    return metrics, neb.images


def run_ase_side(
    images: list[Atoms], calc_factory, system: str
) -> tuple[RunMetrics, list[Atoms]]:
    imgs = copy.deepcopy(images)
    for img in imgs:
        img.calc = calc_factory()
    # method="improvedtangent" is pinned explicitly (not left to ASE's
    # installed-version default) because it is the same tangent formula
    # NEBwalk implements (Henkelman & Jonsson 2000); ASE has changed its
    # own default for this parameter between versions (a UserWarning is
    # raised by ASE if it is left unset), so pinning it keeps this
    # comparison meaningful and reproducible across ASE versions.
    neb = ASE_NEB(imgs, k=K_SPRING, climb=False, method="improvedtangent")
    opt = ASE_FIRE(neb, logfile=None)
    t0 = time.perf_counter()
    converged = opt.run(fmax=FMAX, steps=MAX_STEPS)
    wall = time.perf_counter() - t0
    energies = [float(img.get_potential_energy()) for img in imgs]
    barrier = max(energies) - energies[0]
    reverse = max(energies) - energies[-1]
    reaction = energies[-1] - energies[0]
    metrics = RunMetrics(
        tool="ase",
        system=system,
        converged=bool(converged),
        n_steps=int(opt.nsteps),
        barrier_eV=barrier,
        reverse_barrier_eV=reverse,
        reaction_energy_eV=reaction,
        energies_eV=energies,
        wall_time_s=wall,
        optimizer_defaults={"dt": 0.1, "dt_max": 1.0},
    )
    return metrics, imgs


def max_geometry_difference(images_a: list[Atoms], images_b: list[Atoms]) -> float:
    """Max per-atom displacement (MIC-aware) between two converged paths."""
    max_disp = 0.0
    for img_a, img_b in zip(images_a, images_b):
        dr = img_a.positions - img_b.positions
        if np.any(img_a.pbc):
            dr, _ = find_mic(dr, img_a.cell, img_a.pbc)
        disp = np.sqrt((dr**2).sum(axis=1)).max()
        max_disp = max(max_disp, float(disp))
    return max_disp


def compare_system(
    system: str,
    initial: Atoms,
    final: Atoms,
    fixed_indices: list[int],
    calc_factory,
    constrain_endpoints: bool,
) -> dict[str, Any]:
    shared_images = build_shared_images(
        initial, final, fixed_indices, constrain_endpoints
    )
    nw_metrics, nw_images = run_nebwalk_side(shared_images, calc_factory, system)
    ase_metrics, ase_images = run_ase_side(shared_images, calc_factory, system)

    max_disp = max_geometry_difference(nw_images, ase_images)
    energies_diff = [
        abs(a - b) for a, b in zip(nw_metrics.energies_eV, ase_metrics.energies_eV)
    ]

    return {
        "system": system,
        "shared_settings": {
            "n_images": N_IMAGES,
            "k_spring": K_SPRING,
            "fmax": FMAX,
            "max_steps": MAX_STEPS,
            "interpolation": "linear (MIC-aware, NEBwalk.linear_interpolate), "
            "identical starting images fed to both optimizers",
            "climb": False,
        },
        "NEBwalk": asdict(nw_metrics),
        "ase": asdict(ase_metrics),
        "comparison": {
            "barrier_diff_eV": abs(nw_metrics.barrier_eV - ase_metrics.barrier_eV),
            "reverse_barrier_diff_eV": abs(
                nw_metrics.reverse_barrier_eV - ase_metrics.reverse_barrier_eV
            ),
            "max_per_image_energy_diff_eV": max(energies_diff),
            "max_geometry_diff_angstrom": max_disp,
            "both_converged": nw_metrics.converged and ase_metrics.converged,
            "n_steps_nebwalk": nw_metrics.n_steps,
            "n_steps_ase": ase_metrics.n_steps,
            "note": "Step-count difference is expected and is not evidence of "
            "a bug; see module docstring on differing FIRE dt/dt_max defaults.",
        },
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy_version": np.__version__,
        "ase_version": ase.__version__,
        "nebwalk_version": NEBwalk.__version__,
        "git_commit": _git_commit(),
    }

    results = []

    m_init, m_final, m_fixed = build_morse_h3()
    r_morse = compare_system(
        "morse_h3_collinear", m_init, m_final, m_fixed, MorsePairH3,
        constrain_endpoints=False,
    )
    results.append(r_morse)

    a_init, a_final, a_fixed = build_al100_adatom()
    r_al = compare_system(
        "al100_adatom_emt", a_init, a_final, a_fixed, EMT,
        constrain_endpoints=True,
    )
    results.append(r_al)

    for r in results:
        out_path = RESULTS_DIR / f"{r['system']}_comparison.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"environment": env, **r}, f, indent=2)
        print(f"Wrote {out_path}")

    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system", "nebwalk_barrier_eV", "ase_barrier_eV", "barrier_diff_eV",
            "max_per_image_energy_diff_eV", "max_geometry_diff_angstrom",
            "nebwalk_converged", "ase_converged", "n_steps_nebwalk", "n_steps_ase",
        ])
        for r in results:
            c = r["comparison"]
            writer.writerow([
                r["system"], r["NEBwalk"]["barrier_eV"], r["ase"]["barrier_eV"],
                c["barrier_diff_eV"], c["max_per_image_energy_diff_eV"],
                c["max_geometry_diff_angstrom"], r["NEBwalk"]["converged"],
                r["ase"]["converged"], c["n_steps_nebwalk"], c["n_steps_ase"],
            ])
    print(f"Wrote {summary_path}")

    print("\n=== Summary ===")
    for r in results:
        c = r["comparison"]
        print(
            f"{r['system']}: barrier diff = {c['barrier_diff_eV']*1000:.4f} meV, "
            f"max geometry diff = {c['max_geometry_diff_angstrom']*1000:.4f} mÅ, "
            f"both converged = {c['both_converged']}"
        )


if __name__ == "__main__":
    main()
