"""Dimethyl ether methyl torsion with Egret-1t/MACE-OFF23 disagreement.

Run:
    python examples/dimethyl_ether_egret_maceoff23_disagreement.py
"""

from __future__ import annotations

from organic_disagreement_common import (
    run_organic_disagreement_example,
    torsion_pair,
)


def main() -> None:
    initial, final = torsion_pair(
        molecule_name="CH3OCH3",
        axis_start=0,
        axis_end=1,
        rotating_indices=[3, 4, 5],
    )
    run_organic_disagreement_example(
        title="Dimethyl ether methyl torsion",
        initial=initial,
        final=final,
        output_dir="dimethyl_ether_egret_maceoff23_disagreement_selected",
    )


if __name__ == "__main__":
    main()
