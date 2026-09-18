"""Template for running NEBwalk with Quantum ESPRESSO via ASE.

This example is intentionally explicit. It expects relaxed endpoint structures
and local pseudopotentials, then delegates all repeated NEB setup to NEBwalk's
shared engine and QE calculator factory.

Before running, configure the paths and command below for a native Linux QE
installation. Production QE calculations are not recommended from WSL.
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read

from NEBwalk import (
    NEBRunConfig,
    QEParams,
    make_qe_factory,
    run_neb_calculation,
    validate_qe_setup,
)

PSEUDO_DIR = Path("/path/to/pseudopotentials")
PSEUDOPOTENTIALS = {
    "Al": "Al.pbe-n-rrkjus_psl.1.0.0.UPF",
}
QE_COMMAND = "mpirun -np 8 pw.x"

INITIAL_STRUCTURE = Path("initial.vasp")
FINAL_STRUCTURE = Path("final.vasp")
QE_WORKDIR = Path("qe_neb_workdir")


def main() -> None:
    """Run a QE-backed NEB calculation from two relaxed endpoint structures."""
    validate_qe_setup(
        pseudo_dir=PSEUDO_DIR,
        pseudopotentials=PSEUDOPOTENTIALS,
        command=QE_COMMAND,
    )

    initial = read(INITIAL_STRUCTURE)
    final = read(FINAL_STRUCTURE)

    params = QEParams(
        ecutwfc=40.0,
        ecutrho=320.0,
        kpts=(3, 3, 1),
        occupations="smearing",
        smearing="marzari-vanderbilt",
        degauss=0.02,
        conv_thr=1.0e-8,
        mixing_beta=0.3,
        extra_system={"input_dft": "PBE"},
    )
    calculator_factory = make_qe_factory(
        params=params,
        pseudo_dir=PSEUDO_DIR,
        pseudopotentials=PSEUDOPOTENTIALS,
        base_dir=QE_WORKDIR,
        command=QE_COMMAND,
    )

    config = NEBRunConfig(
        n_images=7,
        interpolation="idpp",
        k=0.10,
        k_min=0.03,
        climb=True,
        climb_delay=50,
        n_workers=1,
        fmax=0.05,
        max_steps=300,
        verbose=True,
    )
    result = run_neb_calculation(
        initial=initial,
        final=final,
        calculator_factory=calculator_factory,
        config=config,
    )

    result.neb.save_csv("qe_neb_profile.csv")
    result.neb.plot("qe_neb_profile.png", title="QE/PBE CI-NEB")
    result.neb.save_trajectory("qe_neb.traj")

    print(f"Converged       : {result.converged}")
    print(f"Forward barrier : {result.barrier:.6f} eV")
    print(f"Reverse barrier : {result.reverse_barrier:.6f} eV")
    print(f"Reaction energy : {result.reaction_energy:.6f} eV")


if __name__ == "__main__":
    main()
