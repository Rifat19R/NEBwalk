"""MLIP-assisted NEB template using MACE.

Requires:
    pip install "NEBwalk[mace]"

This is a template. Users must provide chemically meaningful relaxed endpoints.
"""

from __future__ import annotations

from pathlib import Path

from NEBwalk import NEBRunConfig
from NEBwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb


def make_mace_calc():
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError(
            "MACE is not installed. Install optional support with: "
            'pip install "NEBwalk[mace]"'
        ) from exc

    return mace_mp(
        model="small",
        dispersion=False,
        default_dtype="float64",
        device="cpu",
    )


def make_initial_final():
    raise NotImplementedError(
        "Provide relaxed initial and final ase.Atoms endpoints before running "
        "this MACE template."
    )


def main() -> None:
    initial, final = make_initial_final()

    result = run_mlip_assisted_neb(
        initial=initial,
        final=final,
        mlip_calculator_factory=make_mace_calc,
        neb_config=NEBRunConfig(
            n_images=7,
            interpolation="idpp",
            climb=True,
            fmax=0.05,
            max_steps=500,
        ),
        active_config=MLIPActiveNEBConfig(
            selection_strategy="peak_plus_neighbors",
            n_select=3,
            output_dir=Path("mace_selected_for_qe"),
        ),
    )

    print(result.selected_indices)


if __name__ == "__main__":
    main()
