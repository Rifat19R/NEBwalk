"""MLIP-assisted NEB example using EMT as a fast calculator stand-in.

This demonstrates the v0.7.0 MLIP-assisted workflow without requiring MACE.
In production, replace the EMT factory with a MACE/Egret/custom MLIP
calculator factory.
"""

from __future__ import annotations

from pathlib import Path

from ase.build import bulk
from ase.calculators.emt import EMT

from NEBwalk import NEBRunConfig
from NEBwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb


def make_initial_final():
    """Create a small illustrative Al endpoint pair."""
    initial = bulk("Al", "fcc", a=4.05).repeat((2, 2, 2))
    final = initial.copy()
    final.positions[0] += [0.3, 0.0, 0.0]
    return initial, final


def make_calc():
    return EMT()


def main() -> None:
    initial, final = make_initial_final()

    result = run_mlip_assisted_neb(
        initial=initial,
        final=final,
        mlip_calculator_factory=make_calc,
        neb_config=NEBRunConfig(
            n_images=3,
            interpolation="linear",
            climb=False,
            fmax=0.2,
            max_steps=20,
            verbose=False,
        ),
        active_config=MLIPActiveNEBConfig(
            selection_strategy="peak_plus_neighbors",
            n_select=3,
            include_endpoints=False,
            export_selected=True,
            output_dir=Path("mlip_assisted_emt_selected"),
        ),
    )

    print(f"Converged: {result.neb_result.converged}")
    print(f"MLIP-assisted barrier: {result.mlip_barrier:.6f} eV")
    print(f"Selected indices: {result.selected_indices}")
    print(f"Output directory: {result.output_dir}")


if __name__ == "__main__":
    main()
