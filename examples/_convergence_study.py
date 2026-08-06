"""RESEARCH_PLAN.md Phase 2: ecutwfc/k-point convergence study for Al and Fe.

Tests convergence of the barrier-relevant quantity (E_peak - E_reference) at
several cutoffs/k-grids, on the ALREADY-relaxed reference (index 0) and
DFT-peak (index 3) geometries from the existing labeled dataset -- not by
re-relaxing the NEB at every setting, which would be far more expensive and
is not what convergence testing needs: convergence w.r.t. plane-wave cutoff
and k-point sampling is a property of the electronic-structure calculation at
FIXED geometry, not of the geometry itself. Re-relaxation convergence is a
separate, already-implicit check (the geometries were relaxed under
MACE-MP-0, not QE, so this only tests the QE single-point's own numerical
convergence, matching how these labels are actually used).

Al is the simple nonmagnetic fcc prototype; Fe is the harder magnetic bcc
prototype (nspin=2, tot_magnetization-constrained). Two elements are enough
to establish that the production settings are convergence-justified rather
than merely "the SSSP defaults," without re-running this for every element
in the deepened dataset.

Run:
    python examples/_convergence_study.py al
    python examples/_convergence_study.py fe
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from _pilot_vacancy_dft_labeling import (
    PSEUDOPOTENTIAL_BY_MATERIAL,
    QE_COMMAND,
    qe_params_for_material,
)
from ase.io import read
from vacancy_benchmark_suite import SYSTEMS

from nebwalk.qe import make_qe_factory

# (ecutwfc, ecutrho) pairs; ecutrho held at production's ratio to ecutwfc.
CUTOFF_SWEEP = [(40.0, 320.0), (50.0, 400.0), (60.0, 480.0), (70.0, 560.0)]
KPTS_SWEEP = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]

MATERIALS = ["al", "fe"]


def _single_point(atoms, params, pseudo_dir, pseudopotentials, base_dir) -> float:
    factory = make_qe_factory(
        params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudopotentials,
        base_dir=base_dir,
        command=QE_COMMAND,
    )
    evaluated = atoms.copy()
    evaluated.calc = factory()
    return float(evaluated.get_potential_energy())


def main(material: str) -> None:
    if material not in MATERIALS:
        raise SystemExit(f"Convergence study only covers {MATERIALS}, got {material!r}")

    system = SYSTEMS[material]
    pseudo_dir, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
    pseudopotentials = {system.symbol: pseudo_file}
    base_params = qe_params_for_material(material, system)

    labels_dir = Path(f"_pilot_{material}_vacancy_dft_labels")
    reference = read(labels_dir / "dft_label_00.xyz")
    peak = read(labels_dir / "dft_label_03.xyz")

    results: list[dict] = []

    print("=" * 72)
    print(f"Convergence study: {system.symbol}")
    print("=" * 72)
    print("-- ecutwfc/ecutrho sweep (kpts fixed at production value) --")
    for ecutwfc, ecutrho in CUTOFF_SWEEP:
        params = dataclasses.replace(base_params, ecutwfc=ecutwfc, ecutrho=ecutrho)
        tag = f"_convergence_{material}_ecut{int(ecutwfc)}"
        e_ref = _single_point(
            reference, params, pseudo_dir, pseudopotentials, f"{tag}_ref"
        )
        e_peak = _single_point(
            peak, params, pseudo_dir, pseudopotentials, f"{tag}_peak"
        )
        delta = e_peak - e_ref
        print(
            f"ecutwfc={ecutwfc:5.1f} Ry  ecutrho={ecutrho:6.1f} Ry  "
            f"E_peak-E_ref = {delta:.6f} eV"
        )
        results.append(
            {
                "sweep": "ecutwfc",
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
                "kpts": list(base_params.kpts),
                "barrier_proxy_eV": delta,
            }
        )

    print("\n-- k-point sweep (ecutwfc/ecutrho fixed at production value) --")
    for kpts in KPTS_SWEEP:
        params = dataclasses.replace(base_params, kpts=kpts)
        tag = f"_convergence_{material}_kpts{kpts[0]}{kpts[1]}{kpts[2]}"
        e_ref = _single_point(
            reference, params, pseudo_dir, pseudopotentials, f"{tag}_ref"
        )
        e_peak = _single_point(
            peak, params, pseudo_dir, pseudopotentials, f"{tag}_peak"
        )
        delta = e_peak - e_ref
        print(f"kpts={kpts}  E_peak-E_ref = {delta:.6f} eV")
        results.append(
            {
                "sweep": "kpts",
                "ecutwfc": base_params.ecutwfc,
                "ecutrho": base_params.ecutrho,
                "kpts": list(kpts),
                "barrier_proxy_eV": delta,
            }
        )

    out_dir = Path(f"datasets/{system.symbol}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nebwalk.convergence_study.v1",
        "material": material,
        "symbol": system.symbol,
        "production_ecutwfc": base_params.ecutwfc,
        "production_ecutrho": base_params.ecutrho,
        "production_kpts": list(base_params.kpts),
        "note": (
            "barrier_proxy_eV = E(DFT-peak image) - E(reference image) at "
            "fixed MACE-MP-0-relaxed geometry (indices 3 and 0 of the "
            "5-image path); tests QE numerical convergence, not geometric "
            "relaxation convergence."
        ),
        "results": results,
    }
    convergence_path = out_dir / "convergence_study.json"
    convergence_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {convergence_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python _convergence_study.py <al|fe>")
    main(sys.argv[1].lower())
