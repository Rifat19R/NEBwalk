"""Au vacancy migration in FCC Au with MACE-MP-0.

Reference context:
    FCC Au vacancy migration barriers are commonly reported near 0.68 eV from
    DFT-PBE literature.  This MACE-MP-0 example is a higher-fidelity companion
    to au_vacancy_emt.py and is expected to be closer to DFT than EMT.

Run:
    python examples/au_vacancy_macemp.py
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings

import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from mace.calculators import mace_mp

from NEBwalk import NEB, linear_interpolate

logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

NAME = "au_vacancy_macemp"
REFERENCE_BARRIER_EV = 0.68
MODEL = "small"
DEVICE = "cpu"
DTYPE = "float64"


def make_calc():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return mace_mp(
                model=MODEL,
                dispersion=False,
                default_dtype=DTYPE,
                device=DEVICE,
            )


def make_relaxed_endpoints():
    full = bulk("Au", "fcc", a=4.08, cubic=True).repeat((2, 2, 2))
    vacancy_pos = full.positions[0].copy()

    base = full.copy()
    del base[0]

    distances = np.linalg.norm(base.positions - vacancy_pos, axis=1)
    nn_index = int(np.argmin(distances))
    nn_distance = float(distances[nn_index])

    initial = base.copy()
    initial.calc = make_calc()

    final = base.copy()
    final.positions[nn_index] = vacancy_pos
    final.calc = make_calc()

    BFGS(initial, logfile=None).run(fmax=0.03, steps=100)
    BFGS(final, logfile=None).run(fmax=0.03, steps=100)

    print(f"Migrating atom : index {nn_index}")
    print(f"NN distance    : {nn_distance:.4f} Angstrom")
    endpoint_delta = final.get_potential_energy() - initial.get_potential_energy()
    print(f"Endpoint dE    : {endpoint_delta:.6f} eV")
    return initial, final


def main() -> None:
    initial, final = make_relaxed_endpoints()

    images = linear_interpolate(initial, final, n_images=5)
    for image in images:
        image.calc = make_calc()

    neb = NEB(images, k=0.1, climb=True, climb_delay=20)
    converged = neb.optimize(fmax=0.06, max_steps=180, verbose=False)

    barrier = neb.get_barrier()
    error = abs(barrier - REFERENCE_BARRIER_EV) / REFERENCE_BARRIER_EV * 100.0

    print(f"\nConverged      : {converged}")
    print(f"Forward barrier: {barrier:.4f} eV")
    print(f"Reference (DFT): ~{REFERENCE_BARRIER_EV:.2f} eV")
    print(f"Error          : {error:.1f}%")

    neb.plot(
        f"{NAME}_profile.png",
        title="Au vacancy migration in FCC Au [MACE-MP-0, CI-NEB]",
    )
    neb.save_csv(f"{NAME}_profile.csv")
    neb.save_trajectory(f"{NAME}.traj")


if __name__ == "__main__":
    main()
