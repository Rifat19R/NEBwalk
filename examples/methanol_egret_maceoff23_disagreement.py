"""Methanol hydroxyl torsion with Egret-1t/MACE-OFF23 disagreement.

Run:
    python examples/methanol_egret_maceoff23_disagreement.py
"""

from __future__ import annotations

from organic_disagreement_common import (
    run_organic_disagreement_example,
    torsion_pair,
)


def main() -> None:
    initial, final = torsion_pair(
        molecule_name="CH3OH",
        axis_start=0,
        axis_end=1,
        rotating_indices=[3],
    )
    run_organic_disagreement_example(
        title="Methanol hydroxyl torsion",
        initial=initial,
        final=final,
        output_dir="methanol_egret_maceoff23_disagreement_selected",
    )


if __name__ == "__main__":
    main()
