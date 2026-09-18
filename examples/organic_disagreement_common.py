"""Shared helpers for organic Egret-1t/MACE-OFF23 disagreement examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from mace.calculators import MACECalculator, mace_off

from NEBwalk import NEBRunConfig
from NEBwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb
from NEBwalk.uncertainty import compute_cross_model_disagreement

EGRET_MODEL = Path("EGRET_1T.model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CELL = np.diag([20.0, 20.0, 20.0])

# Ethane geometry, staggered <-> eclipsed methyl rotation. Built by hand rather
# than via isolated_molecule()/rotate_group() because we need the rotation
# angle as a free parameter, not just a start/end pair for a fixed conformer.
_ETHANE_CC = 0.770
_ETHANE_R_CH = 1.090
_ETHANE_DZ = abs(_ETHANE_R_CH * np.cos(np.radians(111.2)))
_ETHANE_RL = _ETHANE_R_CH * np.sin(np.radians(111.2))


def ethane(phi_c2_deg: float) -> Atoms:
    """Return isolated ethane with one methyl rotated by phi_c2_deg."""
    phi = np.radians(phi_c2_deg)
    positions = [[0.0, 0.0, -_ETHANE_CC], [0.0, 0.0, _ETHANE_CC]]
    for idx in range(3):
        angle = idx * (2 * np.pi / 3)
        positions.append(
            [
                _ETHANE_RL * np.cos(angle),
                _ETHANE_RL * np.sin(angle),
                -_ETHANE_CC - _ETHANE_DZ,
            ]
        )
    for idx in range(3):
        angle = phi + idx * (2 * np.pi / 3)
        positions.append(
            [
                _ETHANE_RL * np.cos(angle),
                _ETHANE_RL * np.sin(angle),
                _ETHANE_CC + _ETHANE_DZ,
            ]
        )

    atoms = Atoms("C2H6", positions=positions)
    atoms.set_cell(CELL)
    atoms.pbc = False
    atoms.center()
    return atoms


def make_egret():
    """Fresh Egret-1t primary calculator."""
    kwargs = {"device": DEVICE, "default_dtype": "float32"}
    try:
        return MACECalculator(model_paths=str(EGRET_MODEL), **kwargs)
    except TypeError:
        return MACECalculator(model_path=str(EGRET_MODEL), **kwargs)


def make_mace_off23():
    """Fresh MACE-OFF23 secondary calculator."""
    return mace_off(model="medium", device=DEVICE, default_dtype="float64")


def isolated_molecule(name: str) -> Atoms:
    """Return a centered isolated molecule from ASE's G2 collection."""
    atoms = molecule(name)
    atoms.set_cell(CELL)
    atoms.pbc = False
    atoms.center()
    return atoms


def rotate_group(
    atoms: Atoms,
    axis_start: int,
    axis_end: int,
    indices: list[int],
    angle_deg: float,
) -> Atoms:
    """Rotate selected atoms around the axis_start -> axis_end bond axis."""
    rotated = atoms.copy()
    origin = rotated.positions[axis_start].copy()
    axis = rotated.positions[axis_end] - origin
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    for idx in indices:
        relative = rotated.positions[idx] - origin
        rotated.positions[idx] = origin + rotation @ relative
    rotated.center()
    return rotated


def torsion_pair(
    molecule_name: str,
    axis_start: int,
    axis_end: int,
    rotating_indices: list[int],
    final_angle_deg: float = 120.0,
) -> tuple[Atoms, Atoms]:
    """Build initial/final torsion endpoints from a G2 molecule."""
    initial = isolated_molecule(molecule_name)
    final = rotate_group(
        initial,
        axis_start=axis_start,
        axis_end=axis_end,
        indices=rotating_indices,
        angle_deg=final_angle_deg,
    )
    return initial, final


def print_disagreement_table(result: Any) -> None:
    """Print per-image disagreement diagnostics for the converged NEB path."""
    disagreements = compute_cross_model_disagreement(
        result.neb_result.neb.images,
        make_mace_off23,
    )
    print("\nPer-image cross-model disagreement")
    print("image  valid  dE_rel(eV)    dF_max(eV/A)")
    for item in disagreements:
        energy = (
            f"{item.energy_disagreement:+.8f}"
            if item.energy_disagreement is not None
            else "None"
        )
        force = (
            f"{item.force_disagreement:.8f}"
            if item.force_disagreement is not None
            else "None"
        )
        print(f"{item.index:02d}     {str(item.valid):5s}  {energy:>11s}  {force:>12s}")


def run_organic_disagreement_example(
    title: str,
    initial: Atoms,
    final: Atoms,
    output_dir: str,
) -> Any:
    """Run the organic disagreement-selection example and print diagnostics."""
    neb_cfg = NEBRunConfig(
        n_images=7,
        interpolation="idpp",
        k=0.10,
        k_min=0.033,
        climb=True,
        climb_delay=60,
        n_workers=1,
        fmax=0.05,
        max_steps=400,
    )
    result = run_mlip_assisted_neb(
        initial=initial,
        final=final,
        mlip_calculator_factory=make_egret,
        neb_config=neb_cfg,
        active_config=MLIPActiveNEBConfig(
            selection_strategy="uncertainty_disagreement",
            secondary_calculator_factory=make_mace_off23,
            n_select=3,
            output_dir=output_dir,
        ),
    )

    converged = result.neb_result.converged
    steps_used = len(result.neb_result.neb.history)
    print(f"System           : {title}")
    print(f"Device           : {DEVICE}")
    print(f"Converged        : {converged} ({steps_used}/{neb_cfg.max_steps} steps)")
    if not converged:
        print(
            "WARNING: NEB did not converge within max_steps. Barrier and "
            "disagreement numbers below are from an unconverged path and "
            "should not be trusted until this is resolved (raise max_steps, "
            "loosen fmax, or retune k/k_min for this system)."
        )
    print(f"MLIP barrier     : {result.mlip_barrier:.6f} eV")
    print(f"Selected indices : {result.selected_indices}")
    print(f"Output directory : {result.output_dir}")
    print_disagreement_table(result)
    return result


if __name__ == "__main__":
    print(
        "organic_disagreement_common.py is a shared helper module. "
        "Run one of the molecule scripts, for example:\n"
        "  python examples/propane_egret_maceoff23_disagreement.py"
    )
    raise SystemExit(0)
