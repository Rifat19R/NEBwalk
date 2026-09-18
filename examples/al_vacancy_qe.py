#!/usr/bin/env python3
"""Al vacancy migration with NEBwalk and Quantum ESPRESSO.

System:   FCC Al, 2x2x2 cubic supercell, one vacancy
Method:   QE/PBE with PAW pseudopotential
Path:     nearest-neighbor atom fills the vacancy
Output:   relaxed endpoints, NEB trajectory, CSV, and energy-profile plot

Example:
    export ESPRESSO_PSEUDO=$HOME/pseudo
    export AL_PSEUDO=Al.pbe-n-kjpaw_psl.1.0.0.UPF
    export ESPRESSO_COMMAND="mpirun -np 8 pw.x"
    export NEBWALK_QE_CLEAN=1  # optional: remove previous QE outputs first
    python examples/al_vacancy_qe.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.optimize import BFGS

from NEBwalk import NEB, idpp_interpolate
from NEBwalk.qe import QEParams, make_qe_factory, validate_qe_setup

PSEUDO_DIR = Path(os.environ["ESPRESSO_PSEUDO"]).expanduser()
PSEUDO = {"Al": os.environ.get("AL_PSEUDO", "Al.pbe-n-kjpaw_psl.1.0.0.UPF")}
QE_COMMAND = os.environ.get("ESPRESSO_COMMAND", "pw.x")
WORKDIR = Path("al_vacancy_qe_workdir")
LOG_FILE = Path("al_vacancy_qe.log")
CLEAN_RUN = os.environ.get("NEBWALK_QE_CLEAN", "0") == "1"

PARAMS = QEParams(
    # Matches vacancy_benchmark_suite.qe_params_for() for metals — keep the two
    # Al/QE entry points numerically consistent since they share output files.
    ecutwfc=50.0,
    ecutrho=400.0,
    kpts=(2, 2, 2),
    koffset=(0, 0, 0),
    occupations="smearing",
    smearing="marzari-vanderbilt",
    degauss=0.02,
    conv_thr=1.0e-7,
    mixing_beta=0.3,
    extra_system={"input_dft": "PBE"},
)

N_IMAGES = 5
FMAX_RELAX = 0.05
FMAX_NEB = 0.10
K_SPRING = 0.10
K_MIN = 0.033
CLIMB = True
CLIMB_DELAY = 50
MAX_STEPS = 300


class Tee:
    """Write Python output to terminal and a live log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def clean_previous_outputs() -> None:
    """Remove previous QE outputs when explicitly requested by environment."""
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    for path in (
        Path("al_vacancy_qe_profile.png"),
        Path("al_vacancy_qe_profile.csv"),
        Path("al_vacancy_qe_path.traj"),
    ):
        if path.exists():
            path.unlink()


def make_al_vacancy_endpoints() -> tuple[Atoms, Atoms]:
    """Build symmetry-equivalent vacancy endpoints with stable atom ordering."""
    al = bulk("Al", "fcc", a=4.05, cubic=True).repeat(2)
    vac_pos = al.positions[0].copy()
    hop_pos = al.positions[1].copy()
    hop_distance = np.linalg.norm(hop_pos - vac_pos)

    start = al.copy()
    del start[0]

    end = start.copy()
    end.positions[0] = vac_pos

    print(f"    Supercell atoms before vacancy : {len(al)}")
    print(f"    NEB atoms after vacancy        : {len(start)}")
    print(f"    Vacancy site                   : {vac_pos}")
    print(f"    Migrating atom start           : {hop_pos}")
    print(f"    Hop distance                   : {hop_distance:.3f} Angstrom")
    return start, end


def attach_qe_calculator(atoms: Atoms, subdir: str) -> None:
    """Attach one fresh QE calculator to one Atoms object."""
    factory = make_qe_factory(
        params=PARAMS,
        pseudo_dir=PSEUDO_DIR,
        pseudopotentials=PSEUDO,
        base_dir=WORKDIR / subdir,
        command=QE_COMMAND,
    )
    atoms.calc = factory()


def relax_endpoint(atoms: Atoms, name: str) -> float:
    """Relax one endpoint and return its final potential energy."""
    attach_qe_calculator(atoms, f"{name}_relax")
    optimizer = BFGS(
        atoms,
        trajectory=str(WORKDIR / f"{name}_relax.traj"),
        logfile=str(WORKDIR / f"{name}_relax.log"),
    )
    optimizer.run(fmax=FMAX_RELAX)
    write(WORKDIR / f"{name}_relaxed.vasp", atoms)
    return float(atoms.get_potential_energy())


def attach_neb_calculators(images: list[Atoms]) -> None:
    """Attach independent QE calculators to every NEB image, endpoints included."""
    factory = make_qe_factory(
        params=PARAMS,
        pseudo_dir=PSEUDO_DIR,
        pseudopotentials=PSEUDO,
        base_dir=WORKDIR / "neb",
        command=QE_COMMAND,
    )
    for image in images:
        image.calc = factory()


def main() -> None:
    """Run endpoint relaxation and CI-NEB for Al vacancy migration."""
    if CLEAN_RUN:
        clean_previous_outputs()

    print("=" * 72)
    print("NEBwalk + Quantum ESPRESSO: Al vacancy migration")
    print("=" * 72)
    print(f"Live log: {LOG_FILE.resolve()}")
    if CLEAN_RUN:
        print("Clean run: removed previous QE workdir/profile outputs")

    print("\n[1] Validating QE setup")
    validate_qe_setup(PSEUDO_DIR, PSEUDO, QE_COMMAND)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    print(f"    command    : {QE_COMMAND}")
    print(f"    pseudo_dir : {PSEUDO_DIR}")
    print(f"    pseudo     : {PSEUDO['Al']}")
    print(f"    workdir    : {WORKDIR.resolve()}")

    print("\n[2] Building endpoint structures")
    start, end = make_al_vacancy_endpoints()

    print(f"\n[3] Relaxing endpoints to {FMAX_RELAX:.3f} eV/Angstrom")
    e_start = relax_endpoint(start, "start")
    print(f"    E(start) = {e_start:.8f} eV")
    e_end = relax_endpoint(end, "end")
    print(f"    E(end)   = {e_end:.8f} eV")
    print(f"    Delta E  = {e_end - e_start:.6f} eV")

    print(f"\n[4] Building IDPP path with {N_IMAGES} intermediate images")
    images = idpp_interpolate(start, end, n_images=N_IMAGES)
    attach_neb_calculators(images)
    print(f"    Total images with independent QE calculators: {len(images)}")

    print("\n[5] Running CI-NEB")
    print(f"    k={K_SPRING:.3f}, k_min={K_MIN:.3f}, fmax={FMAX_NEB:.3f}")
    print(f"    climb={CLIMB}, climb_delay={CLIMB_DELAY}, max_steps={MAX_STEPS}")
    neb = NEB(
        images,
        k=K_SPRING,
        k_min=K_MIN,
        climb=CLIMB,
        climb_delay=CLIMB_DELAY,
        n_workers=1,
    )
    converged = neb.optimize(fmax=FMAX_NEB, max_steps=MAX_STEPS, verbose=True)

    barrier = neb.get_barrier()
    reverse = neb.get_reverse_barrier()
    reaction = neb.get_reaction_energy()

    status = "converged" if converged else "not converged"
    print("\n" + "=" * 72)
    print(f"  Status                  : {status}")
    print(f"  Forward barrier QE/PBE  : {barrier:.6f} eV")
    print(f"  Reverse barrier QE/PBE  : {reverse:.6f} eV")
    print(f"  Reaction energy         : {reaction:.6f} eV")
    print("=" * 72)

    neb.plot("al_vacancy_qe_profile.png", title="Al vacancy migration - QE/PBE")
    neb.save_csv("al_vacancy_qe_profile.csv")
    neb.save_trajectory("al_vacancy_qe_path.traj")

    print("\nOutput files")
    print("  al_vacancy_qe_profile.png")
    print("  al_vacancy_qe_profile.csv")
    print("  al_vacancy_qe_path.traj")
    print(f"  {WORKDIR / 'start_relaxed.vasp'}")
    print(f"  {WORKDIR / 'end_relaxed.vasp'}")


if __name__ == "__main__":
    with LOG_FILE.open("w", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            main()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
