"""Propane terminal-methyl torsion with Egret-1t/MACE-OFF23 disagreement.

Run:
    python examples/propane_egret_maceoff23_disagreement.py
"""

from __future__ import annotations

from organic_disagreement_common import (
    run_organic_disagreement_example,
    torsion_pair,
)


def main() -> None:
    initial, final = torsion_pair(
        molecule_name="C3H8",
        axis_start=0,
        axis_end=1,
        rotating_indices=[5, 7, 8],
    )
    run_organic_disagreement_example(
        title="Propane terminal methyl torsion",
        initial=initial,
        final=final,
        output_dir="examples/organic_disagreement/propane_egret_maceoff23_disagreement_selected",
    )


if __name__ == "__main__":
    main()
