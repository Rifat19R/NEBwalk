"""Parameterized Stage 1-3 active-learning pilot for MACE-MP-0 vacancy
systems: relax endpoints under MACE-MP-0, run NEB + peak_plus_neighbors
selection, QE/PBE single-point DFT label the selected + reference images,
compute an isolated-atom QE reference, and export the material's own
canonical dataset directly into datasets/<Symbol>/ via
nebwalk.finetune.export_mace_training_set() (nebwalk.datasets-schema,
validated on write).

This is a throwaway pilot script, not a permanent example. Run materials
ONE AT A TIME, not concurrently -- each material writes to its own
datasets/<Symbol>/ directory, so concurrent runs are safe from file
collisions but QE itself is not set up here for concurrent execution.

Run:
    python examples/_pilot_vacancy_dft_labeling.py <material>

where <material> is one of: al cu ag ni pd au pt w mo si mg li fe
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

from ase import Atoms
from vacancy_benchmark_suite import (
    SYSTEMS,
    mace_model_for,
    make_mace_calc,
    make_vacancy_endpoints,
    qe_params_for,
    relax_endpoint,
)

from nebwalk import MLIPActiveNEBConfig, NEBRunConfig, run_mlip_assisted_neb
from nebwalk.finetune import (
    compute_isolated_atom_reference,
    export_mace_training_set,
    generate_finetune_command,
    save_isolated_atom_reference,
    summarize_training_set,
)
from nebwalk.label import label_selected_images
from nebwalk.qe import QEParams, make_qe_factory
from nebwalk.validate import qe_settings_hash, qe_settings_record

SSSP_DIR = "/mnt/d/Rifat_kh/SSSP_1.3.0_PBE_efficiency"
DAC_DIR = "/home/duets/DAC_ML_v2/pseudo"
QE_COMMAND = os.environ.get(
    "QE_COMMAND",
    "mpirun --oversubscribe -np 6 /home/duets/q-e-qe-7.4.1/PW/src/pw.x",
)
DATASETS_DIR = Path("datasets")

# A 20 A cubic vacuum cell (the original choice) made plane-wave FFT cost
# for a single isolated atom explode -- observed directly at ~20-30 minutes
# per SCF iteration for Cu, next to seconds for a normal bulk-cell
# calculation. A single atom's wavefunction/density decays over a few A, so
# 10 A of separation is still generously isolated (no periodic-image
# interaction) at a small fraction of the cost.
ISOLATED_ATOM_CELL_ANGSTROM = 10.0

# material key -> (pseudo_dir, pseudo_filename). Cu/Si/Mg/Fe/Al are the
# SSSP_1.3.0_PBE_efficiency curated set already used for al_vacancy_qe.py.
# The other 8 have no SSSP entry available on this machine; their
# pseudopotentials come from a separate local project (DAC_ML_v2) and are
# NOT independently cutoff-converged here -- see qe_params_for_material().
PSEUDOPOTENTIAL_BY_MATERIAL: dict[str, tuple[str, str]] = {
    "al": (SSSP_DIR, "Al.pbe-n-kjpaw_psl.1.0.0.UPF"),
    "cu": (SSSP_DIR, "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf"),
    "ag": (DAC_DIR, "Ag_ONCV_PBE-1.0.oncvpsp.upf"),
    "ni": (DAC_DIR, "ni_pbe_v1.4.uspp.F.UPF"),
    "pd": (DAC_DIR, "Pd_ONCV_PBE-1.0.oncvpsp.upf"),
    "au": (DAC_DIR, "Au_ONCV_PBE-1.0.oncvpsp.upf"),
    "pt": (DAC_DIR, "pt_pbe_v1.4.uspp.F.UPF"),
    "w": (DAC_DIR, "W_pbe_v1.2.uspp.F.UPF"),
    "mo": (DAC_DIR, "Mo_ONCV_PBE-1.0.oncvpsp.upf"),
    "si": (SSSP_DIR, "Si.pbe-n-rrkjus_psl.1.0.0.UPF"),
    "mg": (SSSP_DIR, "Mg.pbe-n-kjpaw_psl.0.3.0.UPF"),
    "li": (DAC_DIR, "li_pbe_v1.4.uspp.F.UPF"),
    "fe": (SSSP_DIR, "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF"),
}

# Extra pseudopotential-provenance notes for the two heavy/relativistic
# elements, read directly from each UPF's PP_HEADER/PP_INFO (not asserted
# from memory): relativistic treatment, semicore inclusion, and family, so
# these get the same "consult before trusting" documentation as everything
# else here rather than being asserted from memory.
PSEUDOPOTENTIAL_NOTES: dict[str, str] = {
    "au": (
        "Au_ONCV_PBE-1.0.oncvpsp.upf: relativistic=\"scalar\" (scalar-"
        "relativistic core, no explicit spin-orbit coupling -- standard for "
        "non-collinear-off pw.x). z_valence=19 (5s^2 5p^6 5d^10 6s^1: "
        "semicore 5s/5p included as valence, not frozen into the core). "
        "Same ONCV pseudopotential generation family/generation as Ag/Pd/Mo "
        "used elsewhere in this run set."
    ),
    "pt": (
        "pt_pbe_v1.4.uspp.F.UPF: \"Scalar-Relativistic Calculation\" per "
        "PP_INFO (no spin-orbit coupling). z_valence=16 (5p^6 5d^9.5 6s^0.5 "
        "reference config: USPP generation uses a symmetrized fractional-"
        "occupation reference, not the atom's true 5d^9 6s^1 ground state --"
        " that only affects pseudopotential generation, not this run's own "
        "SCF, which is free to find the true ground state). USPP family "
        "(GBRV-style), NOT the same family as the ONCV set used for "
        "Ag/Pd/Au/Mo -- consistent with Ni/W/Li in this run set instead."
    ),
}


# Isolated-atom ground states are open-shell (Hund's rule) for every element
# here except Pd (4d^10) and Mg (3s^2) -- NOT the same question as whether
# the BULK crystal is magnetic. Bulk Al/Cu/Ag/Au/Mo/W/Si/Li are correctly
# non-magnetic (nspin=1 is right for their vacancy-path calculations), but
# their free atoms are not: Al is 3s^2 3p^1, Cu/Ag/Au are (n)d^10(n+1)s^1,
# Si is 3s^2 3p^2, W is 5d^4 6s^2, Mo is the half-filled-shell case
# 4d^5 5s^1, Li is 2s^1, Pt is 5d^9 6s^1, Ni is 3d^8 4s^2 (^3F term), Fe is
# 3d^6 4s^2 (^5D term). Ni and Fe's BULK crystals are already correctly
# magnetic (nspin=2 via system.magnetic), but that alone is not sufficient
# for their ISOLATED atoms: a starting_magnetization seed with no
# tot_magnetization constraint collapsed to exactly 0 Bohr mag/cell for Fe's
# isolated atom (observed directly, along with SCF failing to converge in
# 300 iterations while oscillating) -- the same failure mode fixed for
# Al/Cu/Ag below applies to Ni/Fe's isolated atoms too, independent of their
# bulk-crystal magnetism. Value is unpaired-electron count (= 2S); used to
# force nspin=2, seed a starting_magnetization fraction (2S / z_valence),
# and constrain tot_magnetization=2S for the isolated-atom calculation
# specifically -- the bulk calculation (already correct) is untouched.
ATOMIC_UNPAIRED_ELECTRONS: dict[str, int] = {
    "al": 1,
    "cu": 1,
    "ag": 1,
    "au": 1,
    "li": 1,
    "si": 2,
    "pt": 2,
    "ni": 2,
    "w": 4,
    "fe": 4,
    "mo": 6,
    # pd, mg: 0 (closed shell) -- omitted, nspin=1 is correct as-is.
}

# QEParams doesn't track z_valence (that's a pseudopotential-file property).
# Read from each UPF's PP_HEADER (z_valence="..."/"Z valence") for the
# materials in ATOMIC_UNPAIRED_ELECTRONS, to convert unpaired-electron count
# into a starting_magnetization fraction.
Z_VALENCE: dict[str, float] = {
    "al": 3.0,
    "cu": 11.0,
    "ag": 19.0,
    "au": 19.0,
    "li": 3.0,
    "si": 4.0,
    "pt": 16.0,
    "ni": 18.0,
    "w": 14.0,
    "fe": 16.0,
    "mo": 14.0,
}


def isolated_atom_qe_params(material: str, system, base_params: QEParams) -> QEParams:
    """QEParams for the isolated-atom reference, correcting for atomic spin.

    Starts from ``base_params`` (already gamma-point-only via the caller)
    and, for elements whose free-atom ground state is open-shell, forces
    nspin=2 with a tot_magnetization constraint -- unconditionally, even if
    base_params already has nspin=2 from bulk-crystal magnetism (Ni/Fe),
    because that seed-only setup is not sufficient for the isolated-atom
    case on its own. See ATOMIC_UNPAIRED_ELECTRONS for the physical
    justification.

    A ``starting_magnetization`` seed alone is not robust here: for a single
    weakly-magnetic atom, SCF mixing routinely relaxes the seed straight
    back to zero within a handful of iterations (observed directly for Al
    and Cu -- both collapsed to ~0 Bohr mag/cell by iteration 5), silently
    reproducing the same wrong (non-magnetic) answer this fix exists to
    correct. ``tot_magnetization`` instead *constrains* the total
    majority-minus-minority spin count to the known atomic value for the
    whole SCF cycle, which cannot collapse the same way; it maps to
    ``extra_system`` (the &SYSTEM namelist).
    """
    unpaired = ATOMIC_UNPAIRED_ELECTRONS.get(material)
    if not unpaired:
        return base_params
    return dataclasses.replace(
        base_params,
        nspin=2,
        starting_magnetization={system.symbol: unpaired / Z_VALENCE[material]},
        extra_system={
            **base_params.extra_system,
            "tot_magnetization": float(unpaired),
        },
    )


# Ni's image_002/003/004 QE labels all hit QE's default 100-iteration SCF cap
# without reaching conv_thr, even after automatic mixing_beta-reducing
# retries -- a smaller mixing_beta needs MORE iterations to converge, not
# fewer, so retrying with a fixed 100-iteration cap made it strictly harder
# to finish in time, not easier. Residual oscillation was tiny (~1e-5 Ry),
# so this is a step-budget problem, not an instability problem. Raise the
# cap for both magnetic materials rather than wait to hit the same issue
# on Fe.
ELECTRON_MAXSTEP_OVERRIDE: dict[str, int] = {"ni": 300, "fe": 300}


def qe_params_for_material(material: str, system) -> QEParams:
    """qe_params_for(), except:

    - norm-conserving (ONCV) pseudopotentials get the standard 4x
      ecutrho:ecutwfc ratio instead of the 8x ratio tuned for USPP/PAW --
      8x would just be wasteful for a norm-conserving potential, which has
      no augmentation charge to resolve.
    - magnetic materials get a raised SCF iteration cap (see
      ELECTRON_MAXSTEP_OVERRIDE).
    """
    _, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
    if "oncv" in pseudo_file.lower():
        params = dataclasses.replace(qe_params_for(system), ecutwfc=60.0, ecutrho=240.0)
    else:
        params = qe_params_for(system)

    max_step = ELECTRON_MAXSTEP_OVERRIDE.get(material)
    if max_step is not None:
        # electron_maxstep is a QE &ELECTRONS namelist parameter, not
        # &CONTROL -- extra_electrons maps to &ELECTRONS in nebwalk.qe.
        params = dataclasses.replace(
            params,
            extra_electrons={**params.extra_electrons, "electron_maxstep": max_step},
        )
    return params


def main(material: str) -> None:
    if material not in PSEUDOPOTENTIAL_BY_MATERIAL:
        known = ", ".join(sorted(PSEUDOPOTENTIAL_BY_MATERIAL))
        raise SystemExit(f"Unknown material {material!r}. Known: {known}")

    system = SYSTEMS[material]
    pseudo_dir, pseudo_file = PSEUDOPOTENTIAL_BY_MATERIAL[material]
    pseudopotentials = {system.symbol: pseudo_file}
    params = qe_params_for_material(material, system)
    curated = pseudo_dir == SSSP_DIR

    print("=" * 72)
    print(f"Material        : {system.symbol} ({system.description})")
    print(f"Pseudopotential : {pseudo_file}")
    if not curated:
        print(
            "                  NOT SSSP-curated -- cutoffs here are "
            "conservative defaults, not independently converged."
        )
    if material in PSEUDOPOTENTIAL_NOTES:
        print(f"Provenance      : {PSEUDOPOTENTIAL_NOTES[material]}")
    print("=" * 72)

    mace_model = mace_model_for(material)

    def mace_factory():
        return make_mace_calc(model=mace_model)

    print(f"\n[0] Relaxing {system.symbol} vacancy endpoints under MACE-MP-0")
    print(f"    Model        : {mace_model}")
    initial, final, nn_index, nn_distance = make_vacancy_endpoints(system)
    e_initial = relax_endpoint(initial, mace_factory, fmax=0.03, steps=300)
    e_final = relax_endpoint(final, mace_factory, fmax=0.03, steps=300)
    print(f"    NN distance : {nn_distance:.4f} A")
    print(f"    Endpoint dE : {e_final - e_initial:.6f} eV")

    print("\n[1] Running Stage 1: MACE-MP-0 NEB + peak_plus_neighbors selection")
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
    print(f"    Converged       : {result.neb_result.converged}")
    print(f"    MLIP barrier    : {result.mlip_barrier:.6f} eV")
    print(f"    Reference       : {system.reference_label}")
    print(f"    Selected indices: {result.selected_indices}")

    print("\n[2] Running Stage 2: QE/PBE single-point DFT labeling of selected images")
    dft_factory = make_qe_factory(
        params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudopotentials,
        base_dir=f"_pilot_{material}_vacancy_dft_labels_qe_workdir",
        command=QE_COMMAND,
    )
    label_result = label_selected_images(
        result,
        dft_calculator_factory=dft_factory,
        output_dir=f"_pilot_{material}_vacancy_dft_labels",
    )
    print(f"\nRequested       : {label_result.metadata['requested_count']}")
    print(f"Labeled         : {label_result.metadata['labeled_count']}")
    print(f"Failed indices  : {label_result.failed_indices}")
    for label in label_result.labels:
        tag = " (reference)" if label.is_reference else ""
        print(
            f"  image {label.index:02d}{tag}  "
            f"DFT_rel={label.dft_relative_energy_eV:+.6f} eV  "
            f"MLIP_rel={label.mlip_relative_energy_eV:+.6f} eV  "
            f"E_disagreement={label.relative_energy_disagreement_eV:+.6f} eV  "
            f"max|dF|={label.max_force_disagreement_eV_A:.6f} eV/A"
        )

    if label_result.metadata["labeled_count"] == 0:
        print(f"\nNo images labeled for {material} -- skipping Stage 3 export.")
        return

    print("\n[3] Running Stage 3: isolated-atom reference + accumulating training set")
    isolated_params = isolated_atom_qe_params(
        material, system, dataclasses.replace(params, kpts=(1, 1, 1))
    )
    if material in ATOMIC_UNPAIRED_ELECTRONS:
        print(
            f"    Isolated-atom spin: nspin=2, "
            f"starting_magnetization={isolated_params.starting_magnetization} "
            f"({ATOMIC_UNPAIRED_ELECTRONS[material]} unpaired e- in the free-atom "
            "ground state -- see ATOMIC_UNPAIRED_ELECTRONS)"
        )
    elif isolated_params.nspin == 2:
        print(
            f"    Isolated-atom spin: nspin=2 (bulk crystal magnetism, "
            f"starting_magnetization={isolated_params.starting_magnetization})"
        )
    isolated_factory = make_qe_factory(
        isolated_params,
        pseudo_dir=pseudo_dir,
        pseudopotentials=pseudopotentials,
        base_dir=f"_pilot_{material}_isolated_atom_qe_workdir",
        command=QE_COMMAND,
    )
    c = ISOLATED_ATOM_CELL_ANGSTROM
    isolated_atom = Atoms(
        system.symbol, positions=[[0.0, 0.0, 0.0]], cell=[c, c, c], pbc=True
    )
    isolated_ref = compute_isolated_atom_reference(isolated_atom, isolated_factory)
    print(f"    Isolated {system.symbol} (QE/PBE): {isolated_ref.energy_eV:.6f} eV")

    bulk_hash = qe_settings_hash(qe_settings_record(pseudo_dir, pseudo_file, params))
    isolated_hash = qe_settings_hash(
        qe_settings_record(pseudo_dir, pseudo_file, isolated_params)
    )
    material_dir = DATASETS_DIR / system.symbol
    artifact = export_mace_training_set(
        images=result.neb_result.neb.images,
        labels=label_result.labels,
        output_path=material_dir / f"{material}_train.extxyz",
        path_id=f"{material}_vacancy",
        dft_settings_hash=bulk_hash,
        campaign_id="vacancy_migration_single_shot",
        calculator_name="quantum_espresso",
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
    summary = summarize_training_set(artifact.path)
    print(f"    Training set : {artifact.path}")
    print(f"    Summary      : {summary}")
    print(
        "    NOTE: run examples/_finalize_datasets.py separately to add "
        "the full validation checkpoint (cohesive energy, SCF convergence, "
        "MACE/nebwalk.datasets loader checks) and update datasets/manifest.json."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python _pilot_vacancy_dft_labeling.py <material>")
    main(sys.argv[1].lower())
