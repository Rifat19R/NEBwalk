"""v0.7.0 selection logic validation.

This script uses a synthetic 7-image energy profile with a clear saddle-region
peak at image 3.  It validates the documented peak-plus-neighbors behavior
without requiring ASE calculators or NEB optimization.

Run:
    python examples/active_selection_logic.py
"""

from __future__ import annotations

from NEBwalk.selection import peak_plus_neighbors


def main() -> None:
    energies = [0.0, 0.1, 0.3, 0.6, 0.4, 0.2, 0.0]
    selected = peak_plus_neighbors(energies, n_select=3, include_endpoints=False)

    assert 3 in selected
    assert 2 in selected or 4 in selected
    assert selected == [2, 3, 4]

    selected_one = peak_plus_neighbors(
        energies,
        n_select=1,
        include_endpoints=False,
    )
    assert selected_one == [3]

    print(f"Energies         : {energies}")
    print(f"Selected n=3     : {selected}")
    print(f"Selected n=1     : {selected_one}")
    print("Selection logic  : OK")


if __name__ == "__main__":
    main()
