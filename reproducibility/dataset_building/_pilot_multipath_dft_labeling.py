"""RESEARCH_PLAN.md section 5b: extra migration paths for within-system
(Al: 2nd-nearest-neighbor hop, 3x3x3 supercell) and finite-size-correction
(Cu/Fe/Mg/Si: 3x3x3 supercell) diversity, per Dr. Ali's feedback.

Runs the full Stage 0-3 pipeline for ONE additional path (MACE-MP-0 relax +
NEB, then QE/PBE labels ALL 5 images -- not just peak_plus_neighbors -- since
the point here is maximum information per path, not minimum DFT cost), and
exports it as its own path_id under datasets/<Symbol>/, distinct from the
element's original 1st-nearest-neighbor 2x2x2-supercell path so per-path_id
provenance stays honest (NEBwalk.datasets requires consistent atom
count/cell/pbc within one path_id; a 3x3x3-supercell path has a different
atom count than the 2x2x2 one, so it MUST be a separate path_id, not merged
into the original file).

Run:
    python reproducibility/dataset_building/_pilot_multipath_dft_labeling.py al 2nn
    python reproducibility/dataset_building/_pilot_multipath_dft_labeling.py al 3x3x3
    python reproducibility/dataset_building/_pilot_multipath_dft_labeling.py cu 3x3x3
    python reproducibility/dataset_building/_pilot_multipath_dft_labeling.py fe 3x3x3
    python reproducibility/dataset_building/_pilot_multipath_dft_labeling.py mg 3x3x3
    python reproducibility/dataset_building/_pilot_multipath_dft_labeling.py si 3x3x3
"""

from __future__ import annotations

import dataclasses
import sys

from _pilot_vacancy_dft_labeling import (
    PSEUDOPOTENTIAL_BY_MATERIAL,
    QE_COMMAND,
    qe_params_for_material,
)
from vacancy_benchmark_suite import (
    SYSTEMS,
    mace_model_for,
    make_mace_calc,
    make_vacancy_endpoints_variant,
    relax_endpoint,
)

from NEBwalk import MLIPActiveNEBConfig, NEBRunConfig, run_mlip_assisted_neb
from NEBwalk.active import SelectedImage
from NEBwalk.label import label_selected_images
from NEBwalk.qe import make_qe_factory

VARIANTS = {
    "2nn": {"neighbor_shell": 2, "repeat": None, "path_suffix": "2nn_hop"},
    "3x3x3": {
        "neighbor_shell": 1,
        "repeat": (3, 3, 3),
        "path_suffix": "3x3x3_supercell",
    },
}
MATERIALS = list(PSEUDOPOTENTIAL_BY_MATERIAL)


def main(material: str, variant: str) -> None:
    if material not in MATERIALS:
        raise SystemExit(f"Unknown material {material!r}. Expected one of {MATERIALS}")
    if variant not in VARIANTS:
        raise SystemExit(
            f"Unknown variant {variant!r}. Expected one of {list(VARIANTS)}"
        )

    spec = VARIANTS[variant]
    system = SYSTEMS[material]
    pseudo_dir, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
    pseudopotentials = {system.symbol: pseudo_file}
    params = qe_params_for_material(material, system)
    mace_model = mace_model_for(material)
    run_tag = f"{material}_{variant}"

    print("=" * 72)
    print(f"Extra path: {system.symbol} / {spec['path_suffix']}")
    print("=" * 72)

    def mace_factory():
        return make_mace_calc(model=mace_model)

    initial, final, nn_index, nn_distance = make_vacancy_endpoints_variant(
        system, neighbor_shell=spec["neighbor_shell"], repeat=spec["repeat"]
    )
    print(f"Migrating atom : index {nn_index}")
    print(f"Hop distance   : {nn_distance:.4f} Angstrom")
    print(f"Atoms in cell  : {len(initial)}")

    relax_endpoint(initial, mace_factory, fmax=0.03, steps=300)
    relax_endpoint(final, mace_factory, fmax=0.03, steps=300)

    result = run_mlip_assisted_neb(
        initial=initial,
        final=final,
        mlip_calculator_factory=mace_factory,
        neb_config=NEBRunConfig(
            n_images=5,
            interpolation="idpp",
            k=0.1,
            k_min=0.033,
            climb=True,
            climb_delay=30,
            fmax=0.05,
            max_steps=300,
        ),
        active_config=MLIPActiveNEBConfig(
            selection_strategy="peak_plus_neighbors",
            n_select=3,
            output_dir=f"_pilot_{run_tag}_vacancy_selected",
        ),
    )
    print(f"MLIP barrier   : {result.mlip_barrier:.6f} eV")
    print(f"Selected       : {result.selected_indices}")

    images = result.neb_result.neb.images
    ref_energy = float(images[0].get_potential_energy())
    all_non_reference = tuple(
        SelectedImage(
            index=i,
            energy=float(images[i].get_potential_energy()),
            relative_energy=float(images[i].get_potential_energy()) - ref_energy,
        )
        for i in (1, 2, 3, 4)
    )
    result_full_path = dataclasses.replace(
        result, selected_images=all_non_reference, selected_indices=(1, 2, 3, 4)
    )

    dft_factory = make_qe_factory(
        params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudopotentials,
        base_dir=f"_pilot_{run_tag}_vacancy_dft_labels_qe_workdir",
        command=QE_COMMAND,
    )
    label_result = label_selected_images(
        result_full_path,
        dft_calculator_factory=dft_factory,
        output_dir=f"_pilot_{run_tag}_vacancy_dft_labels",
    )
    print(f"\nRequested: {label_result.metadata['requested_count']}")
    print(f"Labeled  : {label_result.metadata['labeled_count']}")
    print(f"Failed   : {label_result.failed_indices}")
    for label in label_result.labels:
        tag = " (reference)" if label.is_reference else ""
        print(
            f"  image {label.index:02d}{tag}  "
            f"DFT_rel={label.dft_relative_energy_eV:+.6f} eV  "
            f"MLIP_rel={label.mlip_relative_energy_eV:+.6f} eV  "
            f"E_disagreement={label.relative_energy_disagreement_eV:+.6f} eV"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python _pilot_multipath_dft_labeling.py <material> <2nn|3x3x3>"
        )
    main(sys.argv[1].lower(), sys.argv[2].lower())
