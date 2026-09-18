"""RESEARCH_PLAN.md Phase 2: rebuild datasets/<Symbol>/ for the 5-element
publication subset (Al, Cu, Si, Mg, Fe) with full 5/5-image path coverage.

Merges each element's original labels (indices 0, 2, 3, 4, from
_pilot_<material>_vacancy_dft_labels/) with the newly-labeled index 1 (from
_pilot_<material>_vacancy_dft_labels_index1/, written by
_label_missing_index1.py) into one complete 5-image DFTLabel set, then
re-exports through the same reconciled NEBwalk.finetune/NEBwalk.datasets
pipeline used for the original 13-material dataset (real checksums, real
NEBwalk.datasets.validate_dataset() pass). This supersedes the 4-image
datasets/<Symbol>/ entries for these 5 elements only -- the other 8
elements' entries in datasets/manifest.json are untouched.

Run (after all 5 elements have their index-1 label):
    python reproducibility/dataset_building/_rebuild_publication_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from _pilot_vacancy_dft_labeling import (
    PSEUDOPOTENTIAL_BY_MATERIAL,
    qe_params_for_material,
)
from ase.io import read
from vacancy_benchmark_suite import SYSTEMS

from NEBwalk.finetune import export_mace_training_set, load_isolated_atom_reference
from NEBwalk.label import DFTLabel
from NEBwalk.validate import (
    build_material_validation_summary,
    qe_settings_hash,
    qe_settings_record,
    write_manifest,
)

DATASETS_DIR = Path("datasets")
MATERIALS = ["al", "cu", "si", "mg", "fe"]


def _load_labels(material: str) -> tuple[list[DFTLabel], list, dict]:
    """Merge the original (0,2,3,4) and index-1-only (0,1) label sets."""
    original_dir = Path(f"_pilot_{material}_vacancy_dft_labels")
    index1_dir = Path(f"_pilot_{material}_vacancy_dft_labels_index1")

    original = json.loads((original_dir / "dft_labels.json").read_text())
    index1 = json.loads((index1_dir / "dft_labels.json").read_text())
    index1_by_index = {item["index"]: item for item in index1["labels"]}

    by_index: dict[int, dict] = {item["index"]: item for item in original["labels"]}
    if 1 not in by_index:
        by_index[1] = index1_by_index[1]

    if sorted(by_index) != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"{material}: expected full path indices [0..4], got {sorted(by_index)}"
        )

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
        for item in (by_index[i] for i in range(5))
    ]

    images = [None] * 5
    for index in (0, 2, 3, 4):
        images[index] = read(original_dir / f"dft_label_{index:02d}.xyz")
    images[1] = read(index1_dir / "dft_label_01.xyz")

    failed_indices = sorted(
        set(original["metadata"]["failed_indices"])
        | set(index1["metadata"]["failed_indices"])
    )
    return labels, images, {"failed_indices": failed_indices}


def main() -> None:
    manifest_path = DATASETS_DIR / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )

    summaries = []
    dataset_paths: dict[str, str] = {}

    for material in MATERIALS:
        system = SYSTEMS[material]
        symbol = system.symbol
        print(f"=== {material} ({symbol}) : rebuilding full 5-image path ===")

        labels, images, metadata = _load_labels(material)
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
            # Index 1 was never chosen by peak_plus_neighbors -- it was
            # added afterward to complete full-path DFT coverage (see
            # RESEARCH_PLAN.md Phase 2). Recording this accurately matters:
            # Phase 3's active-vs-random comparison depends on knowing
            # exactly which images the selection strategy actually picked.
            selection_reason_overrides={1: "full_path_completion"},
        )
        dataset_paths[material] = str(artifact.path)

        # Isolated-atom reference and its QE settings are unchanged by this
        # rebuild (no new isolated-atom calculation needed) -- reload the
        # existing validated one.
        isolated_ref = load_isolated_atom_reference(
            material_dir / "isolated_atom_reference.json"
        )

        qe_workdirs = [
            f"_pilot_{material}_vacancy_dft_labels_qe_workdir",
            f"_pilot_{material}_vacancy_dft_labels_index1_qe_workdir",
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
            f"  {status}: n_labeled={summary.n_labeled}, "
            f"cohesive={summary.cohesive_energy_eV_per_atom:.4f} eV/atom, "
            f"scf_clean={summary.scf_converged_cleanly}, "
            f"nebwalk_dataset_ok={summary.nebwalk_dataset_ok}"
        )

    # Merge into the existing 13-material manifest: replace these 5 entries,
    # keep the other 8 untouched.
    if existing_manifest is not None:
        other_entries = [
            e for e in existing_manifest["materials"] if e["material"] not in MATERIALS
        ]
        other_paths = {e["material"]: e["dataset_path"] for e in other_entries}
        all_dataset_paths = {**other_paths, **dataset_paths}

        # Reconstruct summaries for the untouched 8 from the existing manifest
        # entries (write_manifest only needs material/symbol/n_labeled/
        # n_failed/passed, already present there).
        class _FrozenSummary:
            def __init__(self, entry):
                self.material = entry["material"]
                self.symbol = entry["symbol"]
                self.n_labeled = entry["n_labeled"]
                self.n_failed = entry["n_failed"]
                self.passed = entry["passed"]

        all_summaries = [_FrozenSummary(e) for e in other_entries] + summaries
    else:
        all_dataset_paths = dataset_paths
        all_summaries = summaries

    manifest_path = write_manifest(all_summaries, all_dataset_paths, manifest_path)
    n_passed = sum(1 for s in all_summaries if s.passed)
    print(f"\nManifest: {manifest_path}")
    print(f"{n_passed}/{len(all_summaries)} materials passed all checks")
    print(
        f"Publication subset (full 5/5-image path): "
        f"{sum(1 for s in summaries if s.passed)}/{len(summaries)} passed"
    )


if __name__ == "__main__":
    main()
