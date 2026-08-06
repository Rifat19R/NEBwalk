"""Recompute isolated-atom E0 references for Al, Cu, Ag with the correct
atomic ground-state spin (nspin=2, see ATOMIC_UNPAIRED_ELECTRONS in
_pilot_vacancy_dft_labeling.py) and replace their old nspin=1 entries in the
master training set. Bulk path data for these materials is unaffected and
is left untouched -- only the IsolatedAtom config per element is replaced.

Run:
    python examples/_fix_isolated_atom_spin.py
"""

from __future__ import annotations

import dataclasses

from _pilot_vacancy_dft_labeling import (
    ISOLATED_ATOM_CELL_ANGSTROM,
    PSEUDOPOTENTIAL_BY_MATERIAL,
    QE_COMMAND,
    isolated_atom_qe_params,
)
from ase import Atoms
from ase.io import read, write
from vacancy_benchmark_suite import SYSTEMS, qe_params_for

from nebwalk.finetune import compute_isolated_atom_reference
from nebwalk.qe import make_qe_factory

MASTER_TRAIN_FILE = "_vacancy_training_set/vacancy_train.extxyz"
MATERIALS_TO_FIX = ["al", "cu", "ag"]


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
    print(f"Kept {len(kept)}/{len(configs)} (dropped old nspin=1 isolated atoms)")

    new_isolated = []
    for material in MATERIALS_TO_FIX:
        system = SYSTEMS[material]
        pseudo_dir, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
        pseudopotentials = {system.symbol: pseudo_file}
        base_params = qe_params_for(system)
        if "oncv" in pseudo_file.lower():
            base_params = dataclasses.replace(base_params, ecutwfc=60.0, ecutrho=240.0)
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
        print(f"  Isolated {system.symbol} (QE/PBE, nspin=2): {ref.energy_eV:.6f} eV")
        new_isolated.append(ref.atoms)

    write(MASTER_TRAIN_FILE, kept + new_isolated, format="extxyz")
    print(f"\nWrote {len(kept) + len(new_isolated)} configs to {MASTER_TRAIN_FILE}")


if __name__ == "__main__":
    main()
