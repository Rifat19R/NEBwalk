"""Shared vacancy benchmark helpers for nebwalk examples.

This module keeps the material definitions, literature reference values, and
calculator setup in one place so the per-material example files stay small.
It is an examples-only helper, not part of the public nebwalk API.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from ase.optimize import BFGS

from nebwalk import NEB, QEParams, idpp_interpolate, linear_interpolate
from nebwalk.qe import make_qe_factory, validate_qe_setup

Backend = Literal["emt", "mace", "qe"]


@dataclass(frozen=True)
class VacancySystem:
    symbol: str
    crystal: Literal["fcc", "bcc", "hcp", "diamond"]
    lattice_a: float
    repeat: tuple[int, int, int]
    reference_barrier: float
    reference_label: str
    description: str
    lattice_c: float | None = None
    magnetic: bool = False


SYSTEMS: dict[str, VacancySystem] = {
    "al": VacancySystem(
        symbol="Al",
        crystal="fcc",
        lattice_a=4.05,
        repeat=(2, 2, 2),
        reference_barrier=0.61,
        reference_label="~0.61 eV DFT-PBE, Mantina et al.",
        description="FCC Al vacancy migration",
    ),
    "cu": VacancySystem(
        symbol="Cu",
        crystal="fcc",
        lattice_a=3.615,
        repeat=(2, 2, 2),
        reference_barrier=0.70,
        reference_label="~0.70 eV DFT-PBE",
        description="FCC Cu vacancy migration",
    ),
    "ag": VacancySystem(
        symbol="Ag",
        crystal="fcc",
        lattice_a=4.085,
        repeat=(2, 2, 2),
        reference_barrier=0.66,
        reference_label="~0.66 eV DFT-PBE",
        description="FCC Ag vacancy migration",
    ),
    "ni": VacancySystem(
        symbol="Ni",
        crystal="fcc",
        lattice_a=3.524,
        repeat=(2, 2, 2),
        reference_barrier=1.04,
        reference_label="~1.04 eV DFT-PBE",
        description="FCC Ni vacancy migration",
        magnetic=True,
    ),
    "pd": VacancySystem(
        symbol="Pd",
        crystal="fcc",
        lattice_a=3.890,
        repeat=(2, 2, 2),
        reference_barrier=0.91,
        reference_label="~0.91 eV DFT-PBE",
        description="FCC Pd vacancy migration",
    ),
    "au": VacancySystem(
        symbol="Au",
        crystal="fcc",
        lattice_a=4.08,
        repeat=(2, 2, 2),
        reference_barrier=0.68,
        reference_label="~0.68 eV DFT-PBE",
        description="FCC Au vacancy migration",
    ),
    "pt": VacancySystem(
        symbol="Pt",
        crystal="fcc",
        lattice_a=3.924,
        repeat=(2, 2, 2),
        reference_barrier=1.49,
        reference_label="~1.49 eV DFT-PBE",
        description="FCC Pt vacancy migration",
    ),
    "w": VacancySystem(
        symbol="W",
        crystal="bcc",
        lattice_a=3.165,
        repeat=(2, 2, 2),
        reference_barrier=1.66,
        reference_label="~1.66 eV DFT-PBE",
        description="BCC W vacancy migration along <111>",
    ),
    "mo": VacancySystem(
        symbol="Mo",
        crystal="bcc",
        lattice_a=3.147,
        repeat=(2, 2, 2),
        reference_barrier=1.35,
        reference_label="~1.35 eV DFT-PBE",
        description="BCC Mo vacancy migration along <111>",
    ),
    "si": VacancySystem(
        symbol="Si",
        crystal="diamond",
        lattice_a=5.431,
        repeat=(2, 2, 2),
        reference_barrier=0.45,
        reference_label="~0.45 eV DFT-PBE, neutral vacancy order-of-magnitude",
        description="Diamond Si vacancy migration",
    ),
    "mg": VacancySystem(
        symbol="Mg",
        crystal="hcp",
        lattice_a=3.209,
        lattice_c=5.211,
        repeat=(2, 2, 2),
        reference_barrier=0.52,
        reference_label="~0.52 eV DFT-PBE, HCP Mg vacancy",
        description="HCP Mg vacancy migration",
    ),
    "li": VacancySystem(
        symbol="Li",
        crystal="bcc",
        lattice_a=3.51,
        repeat=(2, 2, 2),
        reference_barrier=0.05,
        reference_label="~0.05 eV DFT-PBE, BCC alkali-metal vacancy scale",
        description="BCC Li vacancy migration",
    ),
    "fe": VacancySystem(
        symbol="Fe",
        crystal="bcc",
        lattice_a=2.866,
        repeat=(2, 2, 2),
        reference_barrier=0.67,
        reference_label="~0.67 eV DFT-PBE, ferromagnetic BCC Fe vacancy",
        description="BCC Fe vacancy migration",
        magnetic=True,
    ),
}

QE_MATERIALS = {"al", "cu", "ag", "w", "mo", "si", "mg", "li", "fe", "ni"}
EMT_MATERIALS = {"al", "cu", "ag", "ni", "pd", "au", "pt"}
MACE_MATERIALS = {"al", "cu", "ag", "ni", "pd", "au", "w", "mo", "si"}
MACE_MODEL_BY_MATERIAL = {
    "al": "medium",
    "ag": "medium",
    "ni": "large",
    "si": "medium",
}
QE_PSEUDO_ENV = {
    key: f"{value.symbol.upper()}_PSEUDO"
    for key, value in SYSTEMS.items()
    if key in QE_MATERIALS
}


def build_bulk(system: VacancySystem) -> Atoms:
    if system.crystal == "diamond":
        atoms = bulk(system.symbol, "diamond", a=system.lattice_a, cubic=True)
    elif system.crystal == "hcp":
        atoms = bulk(
            system.symbol,
            "hcp",
            a=system.lattice_a,
            c=system.lattice_c,
            orthorhombic=True,
        )
    else:
        atoms = bulk(system.symbol, system.crystal, a=system.lattice_a, cubic=True)
    atoms = atoms.repeat(system.repeat)
    atoms.pbc = True
    return atoms


def make_vacancy_endpoints(system: VacancySystem) -> tuple[Atoms, Atoms, int, float]:
    """Create symmetry-equivalent vacancy endpoints with stable atom ordering."""
    full = build_bulk(system)
    vacancy_pos = full.positions[0].copy()
    base = full.copy()
    del base[0]

    distances = np.linalg.norm(base.positions - vacancy_pos, axis=1)
    nn_index = int(np.argmin(distances))
    nn_distance = float(distances[nn_index])

    initial = base.copy()
    final = base.copy()
    final.positions[nn_index] = vacancy_pos
    return initial, final, nn_index, nn_distance


def make_vacancy_endpoints_variant(
    system: VacancySystem,
    *,
    neighbor_shell: int = 1,
    repeat: tuple[int, int, int] | None = None,
) -> tuple[Atoms, Atoms, int, float]:
    """Like make_vacancy_endpoints, but for a genuinely different migration
    path: a farther neighbor shell (neighbor_shell=2 -> second-nearest, a
    real, higher-barrier alternate hop, not a symmetry-equivalent repeat of
    the nearest-neighbor one) and/or a different supercell size (finite-size
    correction). neighbor_shell=1 with the default repeat reproduces
    make_vacancy_endpoints's own choice exactly. See RESEARCH_PLAN.md
    section 5b for why this exists (single-hop, single-supercell vacancy
    data has no genuine within-element path diversity to draw on).
    """
    working_system = system if repeat is None else replace(system, repeat=repeat)
    full = build_bulk(working_system)
    vacancy_pos = full.positions[0].copy()
    base = full.copy()
    del base[0]

    distances = np.linalg.norm(base.positions - vacancy_pos, axis=1)
    # Group into shells by rounded distance (fp-noise tolerant); pick the
    # neighbor_shell-th distinct distance, 1-indexed (1 == nearest).
    shells = sorted({round(float(d), 6) for d in distances})
    if neighbor_shell > len(shells):
        raise ValueError(
            f"neighbor_shell={neighbor_shell} exceeds {len(shells)} distinct "
            f"shells available for this supercell size"
        )
    target = shells[neighbor_shell - 1]
    candidates = [i for i, d in enumerate(distances) if round(float(d), 6) == target]
    nn_index = candidates[0]
    nn_distance = float(distances[nn_index])

    initial = base.copy()
    final = base.copy()
    final.positions[nn_index] = vacancy_pos
    return initial, final, nn_index, nn_distance


def make_emt_calc() -> EMT:
    return EMT()


def mace_model_for(material: str) -> str:
    return os.environ.get(
        "NEBWALK_MACE_MODEL",
        MACE_MODEL_BY_MATERIAL.get(material, "small"),
    )


def make_mace_calc(model: str = "small"):
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise RuntimeError(
            "MACE is not installed. Install optional support with: "
            'pip install "nebwalk[mace]"'
        ) from exc

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return mace_mp(
                model=model,
                dispersion=False,
                default_dtype=os.environ.get("NEBWALK_MACE_DTYPE", "float64"),
                device=os.environ.get("NEBWALK_MACE_DEVICE", "cpu"),
            )


def relax_endpoint(atoms: Atoms, calc_factory, fmax: float, steps: int) -> float:
    atoms.calc = calc_factory()
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy())


def run_ase_neb(material: str, backend: Backend) -> None:
    system = SYSTEMS[material]
    name = f"{material}_vacancy_{backend}"
    mace_model = mace_model_for(material) if backend == "mace" else None
    calc_factory = (
        make_emt_calc
        if backend == "emt"
        else lambda: make_mace_calc(model=mace_model or "small")
    )

    if backend == "emt" and material not in EMT_MATERIALS:
        raise RuntimeError(f"EMT benchmark is not configured for {system.symbol}.")
    if backend == "mace" and material not in MACE_MATERIALS:
        raise RuntimeError(f"MACE benchmark is not configured for {system.symbol}.")

    print_header(system, backend)
    if backend == "mace":
        print(f"MACE model     : {mace_model}")
    initial, final, nn_index, nn_distance = make_vacancy_endpoints(system)
    print(f"Migrating atom : index {nn_index}")
    print(f"NN distance    : {nn_distance:.4f} Angstrom")

    relax_fmax = 0.02 if backend == "emt" else 0.03
    neb_fmax = 0.05 if backend == "emt" else 0.06
    max_steps = 300 if backend == "emt" else 180
    n_images = 7 if backend == "emt" else 5

    e_initial = relax_endpoint(initial, calc_factory, relax_fmax, steps=300)
    e_final = relax_endpoint(final, calc_factory, relax_fmax, steps=300)
    print(f"Endpoint dE    : {e_final - e_initial:.6f} eV")

    images = linear_interpolate(initial, final, n_images=n_images)
    for image in images:
        image.calc = calc_factory()

    neb = NEB(images, k=0.1, climb=True, climb_delay=30)
    converged = neb.optimize(fmax=neb_fmax, max_steps=max_steps, verbose=False)

    barrier = neb.get_barrier()
    error = abs(barrier - system.reference_barrier) / system.reference_barrier * 100.0

    print(f"\nConverged      : {converged}")
    print(f"Forward barrier: {barrier:.4f} eV")
    print(f"Reference      : {system.reference_label}")
    print(f"Error          : {error:.1f}%")

    neb.plot(f"{name}_profile.png", title=f"{system.description} [{backend.upper()}]")
    neb.save_csv(f"{name}_profile.csv")
    neb.save_trajectory(f"{name}.traj")


def qe_params_for(system: VacancySystem) -> QEParams:
    is_metal = system.symbol not in {"Si"}
    extra_system: dict[str, object] = {"input_dft": "PBE"}
    nspin = 2 if system.magnetic else 1
    starting_magnetization = {system.symbol: 0.3} if system.magnetic else None

    if is_metal:
        return QEParams(
            ecutwfc=float(os.environ.get("NEBWALK_QE_ECUTWFC", "50")),
            ecutrho=float(os.environ.get("NEBWALK_QE_ECUTRHO", "400")),
            kpts=(2, 2, 2),
            occupations="smearing",
            smearing="marzari-vanderbilt",
            degauss=0.02,
            conv_thr=1.0e-7,
            mixing_beta=0.3,
            nspin=nspin,
            starting_magnetization=starting_magnetization,
            extra_system=extra_system,
        )
    return QEParams(
        ecutwfc=float(os.environ.get("NEBWALK_QE_ECUTWFC", "50")),
        ecutrho=float(os.environ.get("NEBWALK_QE_ECUTRHO", "400")),
        kpts=(2, 2, 2),
        occupations="fixed",
        conv_thr=1.0e-8,
        mixing_beta=0.4,
        nspin=nspin,
        starting_magnetization=starting_magnetization,
        extra_system=extra_system,
    )


def qe_pseudopotentials(system: VacancySystem) -> dict[str, str]:
    env_key = f"{system.symbol.upper()}_PSEUDO"
    pseudo = os.environ.get(env_key)
    if not pseudo:
        raise RuntimeError(
            f"Set {env_key}=<UPF filename> for the QE {system.symbol} benchmark."
        )
    return {system.symbol: pseudo}


def print_qe_environment_hint() -> None:
    print("QE environment example:")
    print("  export NEBWALK_RUN_QE=1")
    print("  export ESPRESSO_PSEUDO=/path/to/pseudos")
    print('  export ESPRESSO_COMMAND="mpirun --oversubscribe -np 4 pw.x"')
    for material in sorted(QE_PSEUDO_ENV):
        env_key = QE_PSEUDO_ENV[material]
        symbol = SYSTEMS[material].symbol
        print(f"  export {env_key}={symbol}.pbe-<family>.UPF")


def run_qe_setup_or_neb(material: str) -> None:
    if material not in QE_MATERIALS:
        raise RuntimeError(f"QE benchmark is not configured for {material}.")

    system = SYSTEMS[material]
    name = f"{material}_vacancy_qe"
    print_header(system, "qe")
    initial, final, nn_index, nn_distance = make_vacancy_endpoints(system)
    print(f"Migrating atom : index {nn_index}")
    print(f"NN distance    : {nn_distance:.4f} Angstrom")

    setup_dir = Path(f"{name}_inputs")
    setup_dir.mkdir(exist_ok=True)
    write(setup_dir / "initial_unrelaxed.vasp", initial)
    write(setup_dir / "final_unrelaxed.vasp", final)

    print(f"Wrote QE endpoint starter structures to {setup_dir}/")
    if os.environ.get("NEBWALK_RUN_QE") != "1":
        print("QE run skipped because NEBWALK_RUN_QE is not 1.")
        print_qe_environment_hint()
        return

    pseudo_dir = Path(os.environ.get("ESPRESSO_PSEUDO", Path.home() / "pseudo"))
    qe_command = os.environ.get("ESPRESSO_COMMAND", "pw.x")
    pseudos = qe_pseudopotentials(system)
    validate_qe_setup(pseudo_dir, pseudos, qe_command)
    print(f"QE command     : {qe_command}")
    print(f"Pseudo dir     : {pseudo_dir}")
    print(f"Pseudopotential: {system.symbol} -> {pseudos[system.symbol]}")

    params = qe_params_for(system)
    workdir = Path(f"{name}_workdir")
    if os.environ.get("NEBWALK_QE_CLEAN") == "1" and workdir.exists():
        shutil.rmtree(workdir)

    initial_factory = make_qe_factory(
        params=params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudos,
        base_dir=workdir / "initial_relax",
        command=qe_command,
    )
    final_factory = make_qe_factory(
        params=params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudos,
        base_dir=workdir / "final_relax",
        command=qe_command,
    )
    e_initial = relax_endpoint(initial, initial_factory, fmax=0.05, steps=120)
    e_final = relax_endpoint(final, final_factory, fmax=0.05, steps=120)
    print(f"Endpoint dE    : {e_final - e_initial:.6f} eV")

    images = idpp_interpolate(initial, final, n_images=5)
    neb_factory = make_qe_factory(
        params=params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudos,
        base_dir=workdir / "neb",
        command=qe_command,
    )
    for image in images:
        image.calc = neb_factory()

    neb = NEB(images, k=0.1, k_min=0.033, climb=True, climb_delay=50, n_workers=1)
    converged = neb.optimize(fmax=0.10, max_steps=220, verbose=True)
    barrier = neb.get_barrier()
    error = abs(barrier - system.reference_barrier) / system.reference_barrier * 100.0

    print(f"\nConverged      : {converged}")
    print(f"Forward barrier: {barrier:.4f} eV")
    print(f"Reference      : {system.reference_label}")
    print(f"Error          : {error:.1f}%")

    neb.plot(f"{name}_profile.png", title=f"{system.description} [QE/PBE]")
    neb.save_csv(f"{name}_profile.csv")
    neb.save_trajectory(f"{name}.traj")


def print_header(system: VacancySystem, backend: Backend) -> None:
    print("=" * 72)
    print(f"{system.description}")
    print(f"Calculator      : {backend.upper()}")
    crystal_line = f"{system.crystal}, a={system.lattice_a:.4f} Angstrom"
    if system.lattice_c is not None:
        crystal_line += f", c={system.lattice_c:.4f} Angstrom"
    print(f"Crystal         : {crystal_line}")
    print(f"Supercell       : {system.repeat[0]}x{system.repeat[1]}x{system.repeat[2]}")
    print(f"Reference       : {system.reference_label}")
    print("=" * 72)


def main(material: str | None = None, backend: Backend | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "material",
        nargs="?",
        default=material,
        choices=sorted(SYSTEMS),
    )
    parser.add_argument(
        "backend",
        nargs="?",
        default=backend,
        choices=["emt", "mace", "qe"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.material is None or args.backend is None:
        parser.error("material and backend are required")

    if args.dry_run or os.environ.get("NEBWALK_DRY_RUN") == "1":
        print_header(SYSTEMS[args.material], args.backend)
        initial, final, nn_index, nn_distance = make_vacancy_endpoints(
            SYSTEMS[args.material]
        )
        print(f"Initial atoms   : {len(initial)}")
        print(f"Final atoms     : {len(final)}")
        print(f"Migrating atom  : index {nn_index}")
        print(f"NN distance     : {nn_distance:.4f} Angstrom")
        print("Dry run         : no calculator invoked")
        return

    logging.disable(logging.INFO)
    warnings.filterwarnings("ignore")
    if args.backend == "qe":
        run_qe_setup_or_neb(args.material)
    else:
        run_ase_neb(args.material, args.backend)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
