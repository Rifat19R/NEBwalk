"""v0.7.0 active workflow validation: Al vacancy migration with EMT.

Reference context:
    FCC Al vacancy migration is commonly reported near 0.61 eV with DFT-PBE
    (Mantina et al., Phys. Rev. Lett. 100, 215901, 2008).  This script uses
    EMT only as a fast dependency-light validation calculator, so it asserts a
    broad sanity range of 0.3-0.8 eV rather than a publication-quality value.

Run:
    python examples/active_al_vacancy_emt.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from NEBwalk import NEBRunConfig
from NEBwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb
from NEBwalk.selection import peak_plus_neighbors

NAME = "active_al_vacancy_emt"
REFERENCE_BARRIER_EV = 0.61


def make_calc() -> EMT:
    return EMT()


def make_relaxed_endpoints():
    """Build a 31-atom FCC Al vacancy hop with consistent atom ordering."""
    full = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
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

    BFGS(initial, logfile=None).run(fmax=0.02)
    BFGS(final, logfile=None).run(fmax=0.02)

    print(f"Migrating atom : index {nn_index}")
    print(f"NN distance    : {nn_distance:.4f} Angstrom")
    endpoint_delta = final.get_potential_energy() - initial.get_potential_energy()
    print(f"Endpoint dE    : {endpoint_delta:.6f} eV")
    return initial, final


def run_active_case(n_select: int):
    output_dir = Path(f"{NAME}_selected_n{n_select}")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    initial, final = make_relaxed_endpoints()
    result = run_mlip_assisted_neb(
        initial=initial,
        final=final,
        mlip_calculator_factory=make_calc,
        neb_config=NEBRunConfig(
            n_images=5,
            interpolation="linear",
            climb=True,
            climb_delay=20,
            fmax=0.05,
            max_steps=300,
            verbose=False,
        ),
        active_config=MLIPActiveNEBConfig(
            selection_strategy="peak_plus_neighbors",
            n_select=n_select,
            include_endpoints=False,
            export_selected=True,
            output_dir=output_dir,
        ),
    )

    energies = result.neb_result.neb.get_energies()
    alias_indices = peak_plus_neighbors(energies, n_select=n_select)
    assert tuple(alias_indices) == result.selected_indices
    assert len(result.selected_indices) == n_select
    assert 0.3 <= result.barrier <= 0.8

    return result


def validate_exports(result) -> None:
    output_dir = result.output_dir
    assert output_dir is not None

    json_path = output_dir / "selected_images.json"
    traj_path = output_dir / "selected_images.traj"
    xyz_paths = sorted(output_dir.glob("selected_image_*.xyz"))

    assert json_path.exists()
    assert traj_path.exists()
    assert len(xyz_paths) == len(result.selected_indices)

    xyz_atoms = [read(path) for path in xyz_paths]
    traj = Trajectory(traj_path)
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    assert xyz_atoms
    assert len(traj) == len(result.selected_indices)
    assert payload["schema"] == "nebwalk.selected_images.v1"
    assert payload["selected_indices"] == list(result.selected_indices)
    assert payload["selected_images"]


def main() -> None:
    result_one = run_active_case(n_select=1)
    result_three = run_active_case(n_select=3)
    validate_exports(result_three)

    result_three.neb_result.neb.plot(
        f"{NAME}_profile.png",
        title="Al vacancy migration [EMT, MLIP-assisted NEB validation]",
    )
    result_three.neb_result.neb.save_csv(f"{NAME}_profile.csv")
    result_three.neb_result.neb.save_trajectory(f"{NAME}.traj")

    print(f"Converged n_select=1 : {result_one.neb_result.converged}")
    print(f"Converged n_select=3 : {result_three.neb_result.converged}")
    print(f"Forward barrier      : {result_three.barrier:.4f} eV")
    print(f"Reference barrier    : ~{REFERENCE_BARRIER_EV:.2f} eV DFT-PBE")
    print(f"Selected n=1         : {result_one.selected_indices}")
    print(f"Selected n=3         : {result_three.selected_indices}")
    print(f"Export directory     : {result_three.output_dir}")


if __name__ == "__main__":
    main()
