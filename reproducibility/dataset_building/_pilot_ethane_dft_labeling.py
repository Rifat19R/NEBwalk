"""Pilot run for Stage 2 of the active-learning loop: DFT-label the images
Stage 1 (MLIP-assisted NEB + uncertainty-disagreement selection) picked out
for the ethane/Egret-1t torsion.

This is a throwaway pilot script, not a permanent example -- it exists to
validate nebwalk.label.label_selected_images() against a real QE run before
deciding whether/how to turn this into a documented example.

Run:
    python reproducibility/dataset_building/_pilot_ethane_dft_labeling.py
"""

from __future__ import annotations

import os

from ase import Atoms
from ase.optimize import BFGS
from organic_disagreement_common import (
    ethane,
    make_egret,
    run_organic_disagreement_example,
)

from nebwalk.label import label_selected_images
from nebwalk.qe import QEParams, make_qe_factory

ENDPOINT_RELAX_FMAX = 0.05

PSEUDO_DIR = "/mnt/d/Rifat_kh/SSSP_1.3.0_PBE_efficiency"
PSEUDOPOTENTIALS = {
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
}
QE_COMMAND = os.environ.get(
    "QE_COMMAND",
    "mpirun --oversubscribe -np 6 /home/duets/q-e-qe-7.4.1/PW/src/pw.x",
)

QE_PARAMS = QEParams(
    ecutwfc=60.0,
    ecutrho=480.0,
    kpts=(1, 1, 1),
    occupations="fixed",
    conv_thr=1.0e-8,
    mixing_beta=0.4,
    extra_system={"input_dft": "PBE"},
)


def relax_endpoint(atoms: Atoms, fmax: float = ENDPOINT_RELAX_FMAX) -> Atoms:
    """Relax a hand-built endpoint under Egret-1t before it enters the NEB.

    ethane() builds idealized, non-equilibrium geometries from fixed bond
    lengths/angles, and NEB never moves endpoint images (forces are zeroed
    there by design). Without this step, DFT-vs-MLIP force comparisons at
    the endpoints would measure "how wrong is this idealized geometry" for
    both methods, not "how wrong is the MLIP" -- a confound, not a labeling
    signal.
    """
    relaxed = atoms.copy()
    relaxed.calc = make_egret()
    BFGS(relaxed, logfile=None).run(fmax=fmax, steps=200)
    return relaxed


def main() -> None:
    print("[0] Relaxing endpoints under Egret-1t before NEB")
    initial = relax_endpoint(ethane(60.0))
    final = relax_endpoint(ethane(180.0))

    print("[1] Running Stage 1: Egret-1t NEB + uncertainty-disagreement selection")
    result = run_organic_disagreement_example(
        title="Ethane methyl torsion (DFT labeling pilot)",
        initial=initial,
        final=final,
        output_dir="_pilot_ethane_selected",
    )

    print("\n[2] Running Stage 2: QE/PBE single-point DFT labeling of selected images")
    dft_factory = make_qe_factory(
        QE_PARAMS,
        pseudo_dir=PSEUDO_DIR,
        pseudopotentials=PSEUDOPOTENTIALS,
        base_dir="_pilot_ethane_dft_labels_qe_workdir",
        command=QE_COMMAND,
    )

    label_result = label_selected_images(
        result,
        dft_calculator_factory=dft_factory,
        output_dir="_pilot_ethane_dft_labels",
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
    print(f"\nOutput dir: {label_result.output_dir}")


if __name__ == "__main__":
    main()
