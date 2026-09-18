"""DFT single-point labeling for MLIP-selected NEB images.

This is stage 2 of the active-learning loop (stage 1 is MLIP-assisted
selection in :mod:`NEBwalk.active`). It computes reference energies and
forces at the *exact* MLIP-relaxed geometry of each selected image, so the
label scores the same configuration the MLIP actually got wrong. It does not
re-relax structures, build a fine-tuning-ready training set, or retrain any
model -- see the module docstring in a future ``NEBwalk.finetune`` for those
stages.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from numpy.typing import NDArray

from .active import MLIPActiveNEBResult
from .recovery import NoOpRecoveryStrategy, RecoveryExhausted, run_with_recovery

FloatArray = NDArray[np.float64]

LABELING_DISCLOSURE = (
    "DFT labels were computed at the MLIP-relaxed geometry of each selected\n"
    "image (single-point, not re-relaxed). This folder is a labeled dataset,\n"
    "not a fine-tuning-ready training set: units, isolated-atom reference\n"
    "energies, and train/validation splitting for MACE-style fine-tuning are\n"
    "a separate step.\n\n"
    "energy_eV / mlip_energy_eV are raw absolute totals from two different\n"
    "codes and are NOT comparable to each other -- QE and an MLIP have\n"
    "unrelated absolute energy zeros. Only *_relative_energy_eV (each zeroed\n"
    "against its own value at the reference image) and\n"
    "relative_energy_disagreement_eV are physically meaningful for judging\n"
    "how wrong the MLIP is. max_force_disagreement_eV_A needs no such\n"
    "zeroing -- forces have no reference-energy ambiguity."
)


@dataclass(frozen=True)
class DFTLabel:
    """One DFT single-point label for a selected NEB image.

    Absolute energies (``energy_eV``, ``mlip_energy_eV``) are kept only for
    transparency/debugging -- QE and an MLIP have unrelated absolute energy
    zeros, so they are not comparable to each other. ``*_relative_energy_eV``
    is each method's energy zeroed against its own value at the labeling
    run's reference image, matching the convention in
    :func:`NEBwalk.uncertainty.compute_cross_model_disagreement`.
    """

    index: int
    energy_eV: float
    forces_eV_A: FloatArray
    mlip_energy_eV: float
    dft_relative_energy_eV: float
    mlip_relative_energy_eV: float
    relative_energy_disagreement_eV: float
    max_force_disagreement_eV_A: float
    is_reference: bool = False

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["forces_eV_A"] = self.forces_eV_A.tolist()
        return d


@dataclass(frozen=True)
class NEBLabelingResult:
    """Result bundle for a DFT labeling pass over selected images.

    Named ``NEBLabelingResult`` (not ``LabelingResult``) to stay distinct from
    :class:`NEBwalk.labeling.LabelingResult`, the campaign-level labeling
    result used by :mod:`NEBwalk.campaign` -- the two are unrelated types
    that happened to want the same short name.
    """

    labels: tuple[DFTLabel, ...]
    failed_indices: tuple[int, ...]
    reference_index: int
    output_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _nebwalk_version() -> str:
    try:
        return importlib_metadata.version("NEBwalk")
    except importlib_metadata.PackageNotFoundError:
        return "0.10.0"


def _label_one_image(
    atoms: Atoms,
    index: int,
    dft_calculator_factory: Callable[[], Any],
    recovery_strategy: Any | None,
) -> tuple[float, FloatArray]:
    """Single-point DFT energy+forces at a fixed geometry, with QE recovery."""
    labeled = atoms.copy()
    labeled.calc = dft_calculator_factory()
    strategy = (
        recovery_strategy
        or getattr(labeled.calc, "recovery_strategy", None)
        or NoOpRecoveryStrategy()
    )

    def compute_fn(eval_atoms: Atoms, _params: dict) -> tuple[float, FloatArray]:
        energy = float(eval_atoms.get_potential_energy())
        forces = np.asarray(eval_atoms.get_forces(), dtype=float)
        return energy, forces

    return run_with_recovery(compute_fn, strategy, labeled, {}, index, [])


def label_selected_images(
    result: MLIPActiveNEBResult,
    dft_calculator_factory: Callable[[], Any],
    output_dir: str | Path | None = None,
    recovery_strategy: Any | None = None,
    reference_index: int = 0,
) -> NEBLabelingResult:
    """Run DFT single-point labeling on an MLIP-assisted NEB's selected images.

    Labels are computed at the exact MLIP-relaxed geometry of each selected
    image, not re-relaxed at the DFT level. Active learning corrects the
    MLIP's prediction error at configurations it actually visits along its
    own path; labeling a re-optimized structure would score a different
    point than the one the MLIP got wrong.

    ``reference_index`` is DFT-labeled in addition to the selected images
    (even if it was not itself selected) so that both DFT and MLIP energies
    can be zeroed against a common baseline before differencing -- raw
    absolute energies from an MLIP and from QE are not comparable. If the
    reference image's DFT calculation fails, every requested image is
    reported as failed, since no valid relative baseline exists.

    QE images that fail are retried through the same recovery machinery used
    during NEB optimization (see :mod:`NEBwalk.recovery`); images that are
    still unrecoverable are reported in ``NEBLabelingResult.failed_indices``
    rather than aborting the whole batch.
    """
    images = result.neb_result.neb.images
    if not (0 <= reference_index < len(images)):
        raise ValueError(
            f"reference_index must be in [0, {len(images) - 1}], got {reference_index}"
        )

    requested_indices = {selected.index for selected in result.selected_images}
    requested_indices.add(reference_index)

    dft_energy_forces: dict[int, tuple[float, FloatArray]] = {}
    failed: list[int] = []
    for index in sorted(requested_indices):
        try:
            dft_energy_forces[index] = _label_one_image(
                images[index], index, dft_calculator_factory, recovery_strategy
            )
        except RecoveryExhausted:
            failed.append(index)

    labels: list[DFTLabel] = []
    if reference_index in dft_energy_forces:
        dft_ref_energy, _ = dft_energy_forces[reference_index]
        mlip_ref_energy = float(images[reference_index].get_potential_energy())

        selected_by_index = {s.index: s for s in result.selected_images}
        label_indices = sorted(requested_indices - set(failed))
        for index in label_indices:
            energy, forces = dft_energy_forces[index]
            mlip_energy = (
                selected_by_index[index].energy
                if index in selected_by_index
                else float(images[index].get_potential_energy())
            )
            mlip_forces = np.asarray(images[index].get_forces(), dtype=float)
            dft_relative = energy - dft_ref_energy
            mlip_relative = mlip_energy - mlip_ref_energy
            labels.append(
                DFTLabel(
                    index=index,
                    energy_eV=energy,
                    forces_eV_A=forces,
                    mlip_energy_eV=mlip_energy,
                    dft_relative_energy_eV=dft_relative,
                    mlip_relative_energy_eV=mlip_relative,
                    relative_energy_disagreement_eV=dft_relative - mlip_relative,
                    max_force_disagreement_eV_A=float(
                        np.max(np.abs(forces - mlip_forces))
                    ),
                    is_reference=(index == reference_index),
                )
            )
    else:
        # Reference DFT calculation failed: no valid baseline for anyone.
        failed = sorted(requested_indices)

    metadata: dict[str, Any] = {
        "nebwalk_version": _nebwalk_version(),
        "stage": "dft_labeling",
        "reference_index": reference_index,
        "labeled_count": len(labels),
        "requested_count": len(requested_indices),
        "failed_indices": failed,
    }

    out_dir = None
    if output_dir is not None:
        out_dir = _export_labels(images, labels, output_dir, metadata)

    return NEBLabelingResult(
        labels=tuple(labels),
        failed_indices=tuple(failed),
        reference_index=reference_index,
        output_dir=out_dir,
        metadata=metadata,
    )


def _export_labels(
    images: Sequence[Atoms],
    labels: Sequence[DFTLabel],
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for label in labels:
        labeled_atoms = images[label.index].copy()
        labeled_atoms.calc = SinglePointCalculator(
            labeled_atoms,
            energy=label.energy_eV,
            forces=label.forces_eV_A,
        )
        write(out / f"dft_label_{label.index:02d}.xyz", labeled_atoms)

    payload: dict[str, Any] = {
        "schema": "nebwalk.dft_labels.v1",
        "labels": [label.to_json() for label in labels],
        "metadata": metadata,
    }
    with (out / "dft_labels.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    readme = out / "README.md"
    readme.write_text(
        f"# NEBwalk DFT labels\n\n{LABELING_DISCLOSURE}\n",
        encoding="utf-8",
    )
    return out


__all__ = [
    "DFTLabel",
    "NEBLabelingResult",
    "label_selected_images",
]
