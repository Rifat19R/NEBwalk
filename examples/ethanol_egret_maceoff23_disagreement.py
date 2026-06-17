"""Ethanol C-C torsion with Egret-1t/MACE-OFF23 disagreement.

Run:
    python examples/ethanol_egret_maceoff23_disagreement.py
"""

from __future__ import annotations

from organic_disagreement_common import (
    run_organic_disagreement_example,
    torsion_pair,
)


def main() -> None:
    initial, final = torsion_pair(
        molecule_name="CH3CH2OH",
        axis_start=0,
        axis_end=1,
        rotating_indices=[2, 3, 4, 5],
    )
    run_organic_disagreement_example(
        title="Ethanol C-C torsion",
        initial=initial,
        final=final,
        output_dir="ethanol_egret_maceoff23_disagreement_selected",
    )


if __name__ == "__main__":
    main()
