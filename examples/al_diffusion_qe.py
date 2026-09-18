"""Al adatom diffusion on Al(100) with Quantum ESPRESSO via ASE.

This is a production-template example, not a CI test. It requires native Linux,
`pw.x`, and a validated PBE Al pseudopotential. WSL2 is intentionally blocked
because QE file I/O and performance measurements are unreliable there.

Run only after configuring:
    export NEBWALK_RUN_QE=1
    export ESPRESSO_COMMAND="mpirun -np 8 pw.x"
    export ESPRESSO_PSEUDO="/path/to/pseudos"
    export AL_PSEUDO="Al.pbe-n-rrkjus_psl.1.0.0.UPF"

Then:
    python examples/al_diffusion_qe.py
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc100
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from NEBwalk import NEB, linear_interpolate

N_IMAGES = 5
K_MAX = 0.10
K_MIN = 0.033


def require_native_linux() -> None:
    """Refuse accidental WSL/non-Linux QE production runs."""
    if os.environ.get("NEBWALK_RUN_QE") != "1":
        raise RuntimeError("Set NEBWALK_RUN_QE=1 to run the QE example.")
    if platform.system() != "Linux" or "microsoft" in platform.release().lower():
        raise RuntimeError("QE example requires native Linux, not WSL2.")


def make_qe_calculator(label: str) -> Espresso:
    """Create one fresh Quantum ESPRESSO calculator instance."""
    command = os.environ.get("ESPRESSO_COMMAND", "pw.x")
    pseudo_dir = Path(os.environ["ESPRESSO_PSEUDO"]).expanduser()
    al_pseudo = os.environ.get("AL_PSEUDO", "Al.pbe-n-rrkjus_psl.1.0.0.UPF")

    profile = EspressoProfile(command=command, pseudo_dir=str(pseudo_dir))
    input_data = {
        "control": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
            "tstress": False,
        },
        "system": {
            "ecutwfc": 40.0,
            "ecutrho": 320.0,
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": 0.02,
            "input_dft": "PBE",
        },
        "electrons": {
            "conv_thr": 1.0e-7,
            "mixing_beta": 0.3,
        },
    }
    return Espresso(
        profile=profile,
        pseudopotentials={"Al": al_pseudo},
        input_data=input_data,
        kpts=(3, 3, 1),
        directory=f"qe_{label}",
    )


def make_slab() -> Atoms:
    """Build Al(100) 2x2x3 slab with one Al adatom at a hollow site."""
    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    add_adsorbate(slab, "Al", height=1.7, position="hollow")
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
    return slab


def main() -> None:
    """Run QE-backed CI-NEB for Al adatom diffusion."""
    require_native_linux()

    initial = make_slab()
    initial.calc = make_qe_calculator("initial")
    BFGS(initial, logfile="qe_initial_relax.log").run(fmax=0.03)

    final = initial.copy()
    adatom_idx = len(final) - 1
    final.positions[adatom_idx] += np.array([4.05 / 2, 0.0, 0.0])
    final.calc = make_qe_calculator("final")
    BFGS(final, logfile="qe_final_relax.log").run(fmax=0.03)

    images = linear_interpolate(initial, final, n_images=N_IMAGES)
    for i, image in enumerate(images):
        image.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in image]))
        image.calc = make_qe_calculator(f"image_{i:02d}")

    neb = NEB(images, k=K_MAX, k_min=K_MIN, climb=True, climb_delay=50)
    converged = neb.optimize(fmax=0.05, max_steps=300)

    print(f"Converged       : {converged}")
    print(f"Forward barrier : {neb.get_barrier():.4f} eV")
    neb.plot("al_diffusion_qe_profile.png", title="Al diffusion [QE/PBE, CI-NEB]")
    neb.save_csv("al_diffusion_qe_profile.csv")
    neb.save_trajectory("al_diffusion_qe.traj")


if __name__ == "__main__":
    main()
