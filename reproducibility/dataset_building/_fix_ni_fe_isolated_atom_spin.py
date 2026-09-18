"""Recompute isolated-atom E0 references for Ni and Fe with the robust
tot_magnetization-constrained spin treatment (see ATOMIC_UNPAIRED_ELECTRONS
in _pilot_vacancy_dft_labeling.py) and replace their existing entries in the
master training set. Bulk path data for these materials is unaffected and
is left untouched -- only the IsolatedAtom config per element is replaced.

Ni's existing isolated-atom entry happened to converge to a plausible
nonzero moment under the old seed-only approach, but Fe's collapsed to
exactly 0 Bohr mag/cell and failed to converge in 300 iterations -- both
get redone here for one consistent methodology across the dataset.

Run:
    python reproducibility/dataset_building/_fix_ni_fe_isolated_atom_spin.py
"""

from __future__ import annotations

import dataclasses

from _pilot_vacancy_dft_labeling import (
    ELECTRON_MAXSTEP_OVERRIDE,
    ISOLATED_ATOM_CELL_ANGSTROM,
    PSEUDOPOTENTIAL_BY_MATERIAL,
    QE_COMMAND,
    isolated_atom_qe_params,
)
from ase import Atoms
from ase.io import read, write
from vacancy_benchmark_suite import SYSTEMS, qe_params_for

from NEBwalk.finetune import compute_isolated_atom_reference
from NEBwalk.qe import make_qe_factory

MASTER_TRAIN_FILE = "_vacancy_training_set/vacancy_train.extxyz"
MATERIALS_TO_FIX = ["ni", "fe"]


def main() -> None:
    configs = read(MASTER_TRAIN_FILE, index=":")
    symbols_to_fix = {SYSTEMS[m].symbol for m in MATERIALS_TO_FIX}

    def is_old_isolated_atom(c) -> bool:
        return (
            c.info.get("config_type") == "IsolatedAtom"
            and len(c) == 1
            and c.get_chemical_symbols()[0] in symbols_to_fix
        )

    kept = [c for c in configs if not is_old_isolated_atom(c)]
    print(f"Kept {len(kept)}/{len(configs)} (dropped old Ni/Fe isolated atoms)")

    new_isolated = []
    for material in MATERIALS_TO_FIX:
        system = SYSTEMS[material]
        pseudo_dir, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
        pseudopotentials = {system.symbol: pseudo_file}
        base_params = qe_params_for(system)
        max_step = ELECTRON_MAXSTEP_OVERRIDE.get(material)
        if max_step is not None:
            base_params = dataclasses.replace(
                base_params,
                extra_electrons={
                    **base_params.extra_electrons,
                    "electron_maxstep": max_step,
                },
            )
        isolated_params = isolated_atom_qe_params(
            material, system, dataclasses.replace(base_params, kpts=(1, 1, 1))
        )
        print(
            f"{material}: nspin={isolated_params.nspin} "
            f"starting_magnetization={isolated_params.starting_magnetization} "
            f"tot_magnetization={isolated_params.extra_system.get('tot_magnetization')}"
        )

        factory = make_qe_factory(
            isolated_params,
            pseudo_dir=pseudo_dir,
            pseudopotentials=pseudopotentials,
            base_dir=f"_fix_{material}_isolated_atom_qe_workdir",
            command=QE_COMMAND,
        )
        c = ISOLATED_ATOM_CELL_ANGSTROM
        atom = Atoms(
            system.symbol,
            positions=[[0.0, 0.0, 0.0]],
            cell=[c, c, c],
            pbc=True,
        )
        ref = compute_isolated_atom_reference(atom, factory)
        print(f"  Isolated {system.symbol} (QE/PBE, corrected): {ref.energy_eV:.6f} eV")
        new_isolated.append(ref.atoms)

    write(MASTER_TRAIN_FILE, kept + new_isolated, format="extxyz")
    print(f"\nWrote {len(kept) + len(new_isolated)} configs to {MASTER_TRAIN_FILE}")


if __name__ == "__main__":
    main()
