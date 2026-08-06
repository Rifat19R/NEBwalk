"""Final restructuring: rebuild datasets/<Symbol>/ through the canonical
nebwalk.datasets writer/validator instead of a bespoke extxyz writer.

Per material this writes:
  - <material>_train.extxyz (+ .manifest.json) via nebwalk.datasets.write_dataset(),
    config_type=Default, one dft_settings_hash (the bulk vacancy-path QE setup).
  - isolated_atom_reference.json: the isolated-atom E0 reference, kept OUT of
    the training file because it was computed at different QE settings
    (gamma-only, no smearing) and therefore has a different dft_settings_hash
    -- nebwalk.datasets.write_dataset() correctly refuses to mix that into
    one dataset file. See nebwalk.finetune.TRAINING_SET_DISCLOSURE.
  - validation_summary.json / VALIDATION_SUMMARY.md, now including a real
    nebwalk.datasets.load_dataset() pass, not just this module's own checks.
  - finetune_command.sh: a ready-to-review mace_run_train command with E0s
    passed explicitly (not --E0s=average).

No raw per-material data is merged; datasets/manifest.json only indexes.
Source data (_pilot_<material>_vacancy_dft_labels/, _vacancy_training_set/)
is read-only here and untouched. New materials run through
_pilot_vacancy_dft_labeling.py write straight into datasets/<Symbol>/
already; this script's own per-material export is what makes it safe to
rerun (idempotent) to refresh the validation checkpoint and manifest.

Run:
    python examples/_finalize_datasets.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from _pilot_vacancy_dft_labeling import (
    PSEUDOPOTENTIAL_BY_MATERIAL,
    isolated_atom_qe_params,
    qe_params_for_material,
)
from ase.io import read
from vacancy_benchmark_suite import SYSTEMS

from nebwalk.finetune import (
    IsolatedAtomReference,
    export_mace_training_set,
    generate_finetune_command,
    save_isolated_atom_reference,
)
from nebwalk.label import DFTLabel
from nebwalk.validate import (
    build_material_validation_summary,
    qe_settings_hash,
    qe_settings_record,
    write_manifest,
)

MASTER_TRAIN_FILE = "_vacancy_training_set/vacancy_train.extxyz"
DATASETS_DIR = Path("datasets")

# Isolated-atom QE workdir per material: _fix_* where the spin bug required
# a redo (Al/Cu/Ag/Ni/Fe), _pilot_* everywhere else (correct on first try).
ISOLATED_ATOM_WORKDIR_PREFIX: dict[str, str] = {
    "al": "_fix",
    "cu": "_fix",
    "ag": "_fix",
    "ni": "_fix",
    "fe": "_fix",
    "pd": "_pilot",
    "au": "_pilot",
    "pt": "_pilot",
    "w": "_pilot",
    "mo": "_pilot",
    "si": "_pilot",
    "mg": "_pilot",
    "li": "_pilot",
}

MATERIALS = list(PSEUDOPOTENTIAL_BY_MATERIAL)


def _load_labels_and_images(material: str) -> tuple[list[DFTLabel], list, dict]:
    labels_dir = Path(f"_pilot_{material}_vacancy_dft_labels")
    payload = json.loads((labels_dir / "dft_labels.json").read_text())
    labels = [
        DFTLabel(
            index=item["index"],
            energy_eV=item["energy_eV"],
            forces_eV_A=np.array(item["forces_eV_A"]),
            mlip_energy_eV=item["mlip_energy_eV"],
            dft_relative_energy_eV=item["dft_relative_energy_eV"],
            mlip_relative_energy_eV=item["mlip_relative_energy_eV"],
            relative_energy_disagreement_eV=item["relative_energy_disagreement_eV"],
            max_force_disagreement_eV_A=item["max_force_disagreement_eV_A"],
            is_reference=item["is_reference"],
        )
        for item in payload["labels"]
    ]
    max_index = max(label.index for label in labels)
    images: list = [None] * (max_index + 1)
    for label in labels:
        images[label.index] = read(labels_dir / f"dft_label_{label.index:02d}.xyz")
    return labels, images, payload["metadata"]


def main() -> None:
    all_configs = read(MASTER_TRAIN_FILE, index=":")
    by_symbol_isolated = {}
    for config in all_configs:
        symbol = config.get_chemical_symbols()[0]
        if config.info.get("config_type") == "IsolatedAtom" and len(config) == 1:
            by_symbol_isolated[symbol] = config

    summaries = []
    dataset_paths: dict[str, str] = {}

    for material in MATERIALS:
        system = SYSTEMS[material]
        symbol = system.symbol
        print(f"=== {material} ({symbol}) ===")

        labels, images, metadata = _load_labels_and_images(material)
        reference_label = next(label for label in labels if label.is_reference)

        pseudo_dir, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
        bulk_params = qe_params_for_material(material, system)
        bulk_hash = qe_settings_hash(
            qe_settings_record(pseudo_dir, pseudo_file, bulk_params)
        )

        material_dir = DATASETS_DIR / symbol
        artifact = export_mace_training_set(
            images=images,
            labels=labels,
            output_path=material_dir / f"{material}_train.extxyz",
            path_id=f"{material}_vacancy",
            dft_settings_hash=bulk_hash,
            campaign_id="vacancy_migration_single_shot",
            calculator_name="quantum_espresso",
        )
        dataset_paths[material] = str(artifact.path)

        isolated_params = isolated_atom_qe_params(
            material, system, dataclasses.replace(bulk_params, kpts=(1, 1, 1))
        )
        isolated_hash = qe_settings_hash(
            qe_settings_record(pseudo_dir, pseudo_file, isolated_params)
        )
        iso_config = by_symbol_isolated[symbol]
        isolated_ref = IsolatedAtomReference(
            symbol=symbol,
            energy_eV=float(iso_config.info["REF_energy"]),
            forces_eV_A=np.asarray(iso_config.arrays["REF_forces"]),
            atoms=iso_config,
        )
        save_isolated_atom_reference(
            isolated_ref,
            material_dir / "isolated_atom_reference.json",
            dft_settings_hash=isolated_hash,
        )

        (material_dir / "finetune_command.sh").write_text(
            "#!/usr/bin/env bash\n"
            "# Reviewable starting point -- see\n"
            "# nebwalk.finetune.generate_finetune_command/TRAINING_SET_DISCLOSURE\n"
            "# before running.\n"
            + generate_finetune_command(
                train_file=artifact.path,
                isolated_atom_references=[isolated_ref],
                name=f"{material}_vacancy_finetune",
            ),
            encoding="utf-8",
        )

        workdir_prefix = ISOLATED_ATOM_WORKDIR_PREFIX.get(material, "_pilot")
        qe_workdirs = [
            f"_pilot_{material}_vacancy_dft_labels_qe_workdir",
            f"{workdir_prefix}_{material}_isolated_atom_qe_workdir",
        ]

        summary = build_material_validation_summary(
            material=material,
            symbol=symbol,
            labels=labels,
            failed_indices=metadata["failed_indices"],
            reference_label=reference_label,
            n_atoms=len(images[0]),
            isolated_atom_reference=isolated_ref,
            pseudo_dir=pseudo_dir,
            pseudo_file=pseudo_file,
            params=bulk_params,
            training_set_path=artifact.path,
            output_dir=material_dir,
            qe_workdirs=qe_workdirs,
        )
        summaries.append(summary)
        status = "PASSED" if summary.passed else "FAILED"
        print(
            f"  {status}: cohesive={summary.cohesive_energy_eV_per_atom:.4f} eV/atom, "
            f"scf_clean={summary.scf_converged_cleanly}, "
            f"finite_ok={not summary.finite_check_problems}, "
            f"mace_loader_ok={summary.mace_loader_ok}, "
            f"nebwalk_dataset_ok={summary.nebwalk_dataset_ok}"
        )

    manifest_path = write_manifest(
        summaries, dataset_paths, DATASETS_DIR / "manifest.json"
    )
    n_passed = sum(1 for s in summaries if s.passed)
    print(f"\nManifest: {manifest_path}")
    print(f"{n_passed}/{len(summaries)} materials passed all checks")


if __name__ == "__main__":
    main()
