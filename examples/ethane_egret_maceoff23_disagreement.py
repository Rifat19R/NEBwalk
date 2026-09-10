"""Ethane methyl torsion with Egret-1t/MACE-OFF23 disagreement.

Run:
    python examples/ethane_egret_maceoff23_disagreement.py
"""

from __future__ import annotations

from organic_disagreement_common import ethane, run_organic_disagreement_example


def main() -> None:
    run_organic_disagreement_example(
        title="Ethane methyl torsion",
        initial=ethane(60.0),
        final=ethane(180.0),
        output_dir="examples/organic_disagreement/ethane_egret_maceoff23_disagreement_selected",
    )


if __name__ == "__main__":
    main()
