"""RESEARCH_PLAN.md Phase 2: complete full-path DFT coverage for the 5
publication-subset elements (Al, Cu, Si, Mg, Fe).

The existing dataset has DFT labels at path indices [0, 2, 3, 4] of each
element's 5-image vacancy-migration path (reference + peak_plus_neighbors
selection) -- index 1 was never selected and its MLIP-relaxed geometry was
never persisted to disk. This reruns Stage 0+1 (MACE-MP-0 relax + NEB,
deterministic given fixed structures/settings) to regenerate the full path in
memory, then DFT-labels index 1 specifically by constructing a
MLIPActiveNEBResult with selected_images overridden to just that index,
reusing nebwalk.label.label_selected_images() end to end (recovery, output
export, disclosure README) rather than hand-rolling a parallel QE call.
Output goes to a separate directory per material
(_pilot_<material>_vacancy_dft_labels_index1/) so the existing, already-
validated 4-label directory is never touched; a later script merges the two.

Run:
    python examples/_label_missing_index1.py <material>
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
    make_vacancy_endpoints,
    relax_endpoint,
)

from nebwalk import MLIPActiveNEBConfig, NEBRunConfig, run_mlip_assisted_neb
from nebwalk.active import SelectedImage
from nebwalk.label import label_selected_images
from nebwalk.qe import make_qe_factory

MATERIALS = ["al", "cu", "si", "mg", "fe"]


def main(material: str) -> None:
    if material not in MATERIALS:
        raise SystemExit(f"Unknown material {material!r}. Expected one of {MATERIALS}")

    system = SYSTEMS[material]
    pseudo_dir, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
    pseudopotentials = {system.symbol: pseudo_file}
    params = qe_params_for_material(material, system)
    mace_model = mace_model_for(material)

    print("=" * 72)
    print(f"Completing full-path DFT coverage: {system.symbol} (index 1 only)")
    print("=" * 72)

    def mace_factory():
        return make_mace_calc(model=mace_model)

    initial, final, nn_index, nn_distance = make_vacancy_endpoints(system)
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
            output_dir=f"_pilot_{material}_vacancy_selected",
        ),
    )

    images = result.neb_result.neb.images
    image1_energy = float(images[1].get_potential_energy())
    ref_energy = float(images[0].get_potential_energy())
    index1_only = (
        SelectedImage(
            index=1, energy=image1_energy, relative_energy=image1_energy - ref_energy
        ),
    )
    result_index1 = dataclasses.replace(
        result, selected_images=index1_only, selected_indices=(1,)
    )

    dft_factory = make_qe_factory(
        params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudopotentials,
        base_dir=f"_pilot_{material}_vacancy_dft_labels_index1_qe_workdir",
        command=QE_COMMAND,
    )
    label_result = label_selected_images(
        result_index1,
        dft_calculator_factory=dft_factory,
        output_dir=f"_pilot_{material}_vacancy_dft_labels_index1",
    )
    print(f"Requested: {label_result.metadata['requested_count']}")
    print(f"Labeled  : {label_result.metadata['labeled_count']}")
    print(f"Failed   : {label_result.failed_indices}")
    for label in label_result.labels:
        print(
            f"  image {label.index:02d}  "
            f"DFT_rel={label.dft_relative_energy_eV:+.6f} eV  "
            f"MLIP_rel={label.mlip_relative_energy_eV:+.6f} eV  "
            f"E_disagreement={label.relative_energy_disagreement_eV:+.6f} eV"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python _label_missing_index1.py <material>")
    main(sys.argv[1].lower())
