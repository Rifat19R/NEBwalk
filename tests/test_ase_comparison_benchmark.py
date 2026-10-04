"""Implementation-consistency check against ASE's own NEB/FIRE.

Exercises both benchmark systems from benchmarks/ase_comparison/
run_comparison.py: the analytical Morse H3 path and the periodic
Al-adatom/Al(100) EMT path. Both are lightweight enough to run in the normal
test suite without MACE, Egret, or Quantum ESPRESSO dependencies.

See benchmarks/ase_comparison/README.md for what is and is not held
identical between the two tools, and why a step-count difference is
expected and not evidence of a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ase.calculators.emt import EMT

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "ase_comparison"
sys.path.insert(0, str(BENCHMARK_DIR))

from run_comparison import (  # noqa: E402
    MorsePairH3,
    build_al100_adatom,
    build_morse_h3,
    compare_system,
)


def test_nebwalk_matches_ase_on_morse_h3():
    initial, final, fixed = build_morse_h3()
    result = compare_system(
        "morse_h3_collinear",
        initial,
        final,
        fixed,
        MorsePairH3,
        constrain_endpoints=False,
    )

    assert result["NEBwalk"]["converged"] is True
    assert result["ase"]["converged"] is True

    comparison = result["comparison"]
    # Loose, physically-motivated tolerances, not tuned to the exact
    # observed values: two independently implemented FIRE optimizers with
    # different default dt should agree well inside 1 meV / 10 mÅ on a
    # 3-atom analytical system with a unique, smooth minimum-energy path.
    assert comparison["barrier_diff_eV"] < 1e-3
    assert comparison["max_per_image_energy_diff_eV"] < 1e-3
    assert comparison["max_geometry_diff_angstrom"] < 1e-2


def test_nebwalk_barrier_matches_known_analytical_value():
    initial, final, fixed = build_morse_h3()
    result = compare_system(
        "morse_h3_collinear",
        initial,
        final,
        fixed,
        MorsePairH3,
        constrain_endpoints=False,
    )
    # examples/morse_h3.py documents the analytical TS barrier as ~0.19 eV
    # at r_AB = r_BC = 1.25 A for this Morse parameterization.
    assert abs(result["NEBwalk"]["barrier_eV"] - 0.193) < 0.02
    assert abs(result["ase"]["barrier_eV"] - 0.193) < 0.02


def test_nebwalk_matches_ase_on_periodic_al100_emt():
    initial, final, fixed = build_al100_adatom()
    result = compare_system(
        "al100_adatom_emt",
        initial,
        final,
        fixed,
        EMT,
        constrain_endpoints=True,
    )

    assert result["NEBwalk"]["converged"] is True
    assert result["ase"]["converged"] is True

    comparison = result["comparison"]

    # Deliberately looser than the observed values. These are regression
    # tolerances for implementation consistency, not fitted acceptance limits.
    assert comparison["barrier_diff_eV"] < 5e-4       # 0.5 meV
    assert comparison["max_per_image_energy_diff_eV"] < 1e-3
    assert comparison["max_geometry_diff_angstrom"] < 5e-3  # 5 mÅ
