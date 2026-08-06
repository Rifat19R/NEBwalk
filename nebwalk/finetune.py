"""MACE-compatible training-set export for DFT-labeled NEB images.

This is stage 3 of the active-learning loop (stage 1: MLIP-assisted
selection in :mod:`nebwalk.active`; stage 2: DFT labeling in
:mod:`nebwalk.label`). It formats DFT labels as :mod:`nebwalk.datasets`
records -- the same canonical, checksum-verified, structure-hash-validated
schema :mod:`nebwalk.campaign`/:mod:`nebwalk.labeling` use -- so a
DFT-labeled NEB path is written and validated through one real, shared code
path rather than a second bespoke writer, and computes a correct
isolated-atom energy reference for the labeling run's own DFT setup.

It does not invoke MACE, choose hyperparameters, or run any training --
fine-tuning itself stays external to nebwalk; see
:func:`generate_finetune_command` for a documented, reviewable command
template rather than an executed one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read, write
from numpy.typing import NDArray

from .datasets import DatasetArtifact, compute_structure_hash, write_dataset
from .label import DFTLabel
from .recovery import NoOpRecoveryStrategy, run_with_recovery

FloatArray = NDArray[np.float64]

MIN_RECOMMENDED_CONFIGS = 50

TRAINING_SET_DISCLOSURE = (
    "This file was exported by nebwalk.finetune.export_mace_training_set()\n"
    "via nebwalk.datasets.write_dataset() -- it passed that module's real\n"
    "geometry/duplicate/provenance validation, not just a bespoke check\n"
    "local to this module. It is correctly *formatted* for MACE, but a\n"
    "handful of configs from a single NEB path is NOT enough data to\n"
    "responsibly fine-tune a foundation model -- it validates that the\n"
    "labeling-to-training pipeline works, not that the resulting checkpoint\n"
    "would be trustworthy. Real fine-tuning needs many more labeled\n"
    "configurations across multiple paths/systems/compositions before\n"
    "training on this data for anything beyond a pipeline smoke test.\n\n"
    "Energies/forces are stored as REF_energy/REF_forces (atoms.info /\n"
    "atoms.arrays), not attached via an ASE calculator with the plain\n"
    "energy/forces keys -- MACE's own loader warns that the plain keys are\n"
    "no longer safe to share with ASE (since ASE 3.23) and recommends the\n"
    "REF_ prefix, which also matches mace.data.utils.DefaultKeys.\n\n"
    "This file does NOT contain an IsolatedAtom entry: an isolated atom in a\n"
    "vacuum cell is evaluated at different QE settings (gamma-only, no\n"
    "smearing) than the bulk path, so it has a different dft_settings_hash\n"
    "and nebwalk.datasets.write_dataset() correctly refuses to mix that into\n"
    "one dataset file. Its energy reference is instead saved alongside this\n"
    "file (see isolated_atom_reference.json / save_isolated_atom_reference())\n"
    "and should be passed to mace_run_train explicitly via --E0s, not fit\n"
    "from this file with --E0s=average and not taken from the foundation\n"
    "model's own built-in E0s -- different codes and pseudopotential\n"
    "families (e.g. VASP/PAW vs. Quantum ESPRESSO/PSL) have unrelated\n"
    "absolute energy zeros even under the same nominal functional, so\n"
    "reusing the foundation model's E0s here would reintroduce the same\n"
    "reference-energy mismatch nebwalk.label guards against one stage\n"
    "earlier."
)


@dataclass(frozen=True)
class IsolatedAtomReference:
    """A single isolated-atom DFT single-point, for a correct E0 reference.

    Computed at the same DFT code/pseudopotential as the path labels it will
    accompany, but at its own (cheaper, gamma-only) QE settings -- see
    :func:`compute_isolated_atom_reference`. Kept out of the path's
    :mod:`nebwalk.datasets` file for that reason; see
    :func:`save_isolated_atom_reference`.
    """

    symbol: str
    energy_eV: float
    forces_eV_A: FloatArray
    atoms: Atoms


def _dft_single_point(
    atoms: Atoms,
    dft_calculator_factory: Callable[[], Any],
    recovery_strategy: Any | None,
) -> tuple[float, FloatArray]:
    """Single-point DFT energy+forces at a fixed geometry, with QE recovery."""
    evaluated = atoms.copy()
    evaluated.calc = dft_calculator_factory()
    strategy = recovery_strategy or getattr(
        evaluated.calc, "recovery_strategy", None
    ) or NoOpRecoveryStrategy()

    def compute_fn(eval_atoms: Atoms, _params: dict) -> tuple[float, FloatArray]:
        energy = float(eval_atoms.get_potential_energy())
        forces = np.asarray(eval_atoms.get_forces(), dtype=float)
        return energy, forces

    return run_with_recovery(compute_fn, strategy, evaluated, {}, 0, [])


def compute_isolated_atom_reference(
    atom: Atoms,
    dft_calculator_factory: Callable[[], Any],
    recovery_strategy: Any | None = None,
) -> IsolatedAtomReference:
    """Run a DFT single-point on one isolated atom, for a correct E0 reference.

    ``atom`` must already be a single-atom ``Atoms`` object in a vacuum cell
    large enough that periodic images don't interact -- the caller chooses
    cell size and QE k-point/occupation settings (gamma-only, no smearing
    dependence on whatever the bulk system used).
    """
    if len(atom) != 1:
        raise ValueError(f"expected a single-atom Atoms object, got {len(atom)} atoms")

    energy, forces = _dft_single_point(atom, dft_calculator_factory, recovery_strategy)
    symbol = atom.get_chemical_symbols()[0]

    labeled_atom = atom.copy()
    labeled_atom.info["REF_energy"] = energy
    labeled_atom.new_array("REF_forces", forces)
    labeled_atom.info["config_type"] = "IsolatedAtom"

    return IsolatedAtomReference(
        symbol=symbol, energy_eV=energy, forces_eV_A=forces, atoms=labeled_atom
    )


def save_isolated_atom_reference(
    ref: IsolatedAtomReference,
    output_path: str | Path,
    dft_settings_hash: str,
) -> Path:
    """Persist an isolated-atom E0 reference as a self-contained JSON record.

    Stores the full geometry alongside the energy/forces so the record
    remains meaningful even if the vacuum-cell convention used to compute it
    changes later, plus the QE settings hash it was computed under (distinct
    from the bulk path's hash -- see :data:`TRAINING_SET_DISCLOSURE`).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nebwalk.isolated_atom_reference.v1",
        "symbol": ref.symbol,
        "energy_eV": ref.energy_eV,
        "forces_eV_A": np.asarray(ref.forces_eV_A, dtype=float).tolist(),
        "positions_A": ref.atoms.positions.tolist(),
        "cell_A": ref.atoms.cell.array.tolist(),
        "pbc": np.asarray(ref.atoms.pbc, dtype=bool).tolist(),
        "dft_settings_hash": dft_settings_hash,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def load_isolated_atom_reference(path: str | Path) -> IsolatedAtomReference:
    """Read back a JSON record written by :func:`save_isolated_atom_reference`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    atoms = Atoms(
        payload["symbol"],
        positions=payload["positions_A"],
        cell=payload["cell_A"],
        pbc=payload["pbc"],
    )
    forces = np.asarray(payload["forces_eV_A"], dtype=float)
    atoms.info["REF_energy"] = payload["energy_eV"]
    atoms.new_array("REF_forces", forces)
    atoms.info["config_type"] = "IsolatedAtom"
    return IsolatedAtomReference(
        symbol=payload["symbol"],
        energy_eV=payload["energy_eV"],
        forces_eV_A=forces,
        atoms=atoms,
    )


def export_mace_training_set(
    images: Sequence[Atoms],
    labels: Sequence[DFTLabel],
    output_path: str | Path,
    *,
    path_id: str,
    dft_settings_hash: str,
    campaign_id: str = "vacancy_migration_single_shot",
    calculator_name: str = "quantum_espresso",
    iteration: int = 0,
    selection_reason_overrides: Mapping[int, str] | None = None,
) -> DatasetArtifact:
    """Write DFT-labeled images as a canonical :mod:`nebwalk.datasets` file.

    Each label becomes one ``config_type=Default`` frame carrying every
    field :data:`nebwalk.datasets.REQUIRED_INFO_KEYS` requires
    (``structure_hash`` via :func:`nebwalk.datasets.compute_structure_hash`,
    ``selection_reason`` derived from ``label.is_reference``, and so on), then
    :func:`nebwalk.datasets.write_dataset` validates, checksums, and writes
    it atomically -- this is the same writer :mod:`nebwalk.labeling` and
    :mod:`nebwalk.campaign` use, so a file written here is guaranteed to load
    back through :func:`nebwalk.datasets.load_dataset` too.

    Every non-reference label defaults to ``selection_reason="peak_plus_
    neighbors"`` -- correct only if every label really was chosen by that
    strategy. If some labels were added later for other reasons (e.g.
    completing full-path DFT coverage of an image the selection strategy
    never picked), pass ``selection_reason_overrides={label.index: "..."}``
    so the dataset's own provenance stays accurate; downstream code that
    needs to know exactly which images a selection strategy chose (e.g. an
    active-vs-random training-data comparison) depends on this being honest,
    not just on the file loading successfully.

    All labels must share one ``dft_settings_hash`` (they come from the same
    QE setup) -- do not call this once per material with a mixed-settings
    label list. Isolated-atom E0 references are handled separately; see
    :func:`save_isolated_atom_reference` and :data:`TRAINING_SET_DISCLOSURE`
    for why they cannot live in this same file.
    """
    overrides = selection_reason_overrides or {}
    frames: list[Atoms] = []
    for label in labels:
        atoms = images[label.index].copy()
        atoms.info["REF_energy"] = label.energy_eV
        atoms.new_array("REF_forces", np.asarray(label.forces_eV_A, dtype=float))
        atoms.info["config_type"] = "Default"
        atoms.info["campaign_id"] = campaign_id
        atoms.info["iteration"] = iteration
        atoms.info["path_id"] = path_id
        atoms.info["image_index"] = label.index
        if label.index in overrides:
            atoms.info["selection_reason"] = overrides[label.index]
        elif label.is_reference:
            atoms.info["selection_reason"] = "reference"
        else:
            atoms.info["selection_reason"] = "peak_plus_neighbors"
        atoms.info["calculator_name"] = calculator_name
        atoms.info["dft_settings_hash"] = dft_settings_hash
        atoms.info["structure_hash"] = compute_structure_hash(atoms)
        frames.append(atoms)

    artifact = write_dataset(
        output_path,
        frames,
        manifest_metadata={"stage": "mace_training_export", "path_id": path_id},
    )

    readme = artifact.path.parent / f"{artifact.path.stem}_README.md"
    readme.write_text(
        "# nebwalk MACE training-set export\n\n"
        f"{artifact.summary.n_configurations} configuration(s), path_id="
        f"{path_id!r}, elements: {', '.join(artifact.summary.elements)}.\n"
        f"sha256: {artifact.checksum}\n\n"
        f"{TRAINING_SET_DISCLOSURE}\n",
        encoding="utf-8",
    )
    return artifact


def summarize_training_set(path: str | Path) -> dict[str, Any]:
    """Read back an exported training file and summarize its contents.

    Loads through :func:`nebwalk.datasets.load_dataset`, so this also
    re-validates the file (geometry, duplicates, provenance) on every call.
    """
    from .datasets import load_dataset

    frames = load_dataset(path)
    elements = sorted({s for c in frames for s in c.get_chemical_symbols()})
    return {
        "total_configs": len(frames),
        "path_configs": sum(
            1 for c in frames if c.info.get("config_type") == "Default"
        ),
        "elements_present": elements,
        "path_ids": sorted({str(c.info.get("path_id")) for c in frames}),
    }


def generate_finetune_command(
    train_file: str | Path,
    isolated_atom_references: Sequence[IsolatedAtomReference] = (),
    foundation_model: str = "medium",
    valid_fraction: float = 0.1,
    device: str = "cuda",
    max_num_epochs: int = 200,
    name: str = "nebwalk_finetune",
) -> str:
    """Return a documented ``mace_run_train`` command template.

    If ``isolated_atom_references`` is given, E0s are passed explicitly as
    ``--E0s='{atomic_number: energy_eV, ...}'`` -- ``--E0s=average`` is
    unreliable (it is also rejected outright under multihead fine-tuning)
    and the foundation model's own built-in E0s use a different DFT
    reference level than this dataset; see :data:`TRAINING_SET_DISCLOSURE`.
    Without references, falls back to ``--E0s=average`` and the caller is
    responsible for that choice being appropriate.

    This does not run MACE and does not tune hyperparameters -- it is a
    reviewable starting point. Flag names vary across mace-torch versions;
    check ``mace_run_train --help`` for your installed version before
    running it, and do not run it at all until the dataset satisfies
    :data:`TRAINING_SET_DISCLOSURE`.
    """
    if isolated_atom_references:
        e0s = {
            atomic_numbers[ref.symbol]: ref.energy_eV
            for ref in isolated_atom_references
        }
        e0s_line = f"    --E0s='{json.dumps(e0s)}' \\\n"
    else:
        e0s_line = "    --E0s=average \\\n"
    return (
        "mace_run_train \\\n"
        f"    --name={name} \\\n"
        f"    --foundation_model={foundation_model} \\\n"
        f"    --train_file={train_file} \\\n"
        f"    --valid_fraction={valid_fraction} \\\n"
        "    --energy_key=REF_energy \\\n"
        "    --forces_key=REF_forces \\\n"
        f"{e0s_line}"
        f"    --device={device} \\\n"
        f"    --max_num_epochs={max_num_epochs} \\\n"
        "    --batch_size=1\n"
    )


def combine_training_sets(
    material_paths: Mapping[str, str | Path], output_path: str | Path
) -> Path:
    """Concatenate several per-material training files into one combined file.

    This is a deliberately permissive, non-validated concatenation -- unlike
    :func:`export_mace_training_set`, it does not go through
    :func:`nebwalk.datasets.write_dataset`, because different materials
    legitimately have different ``dft_settings_hash`` values (different
    pseudopotentials/cutoffs per element), which that writer's validation
    refuses to mix into one canonical dataset. This reads each material's own
    file and writes a new combined file -- it never modifies or merges the
    per-material sources in place. Each material's extxyz stays the single
    source of truth for its own data; the combined file is a derived,
    regeneratable view for actually running ``mace_run_train`` against
    multiple elements at once. Every config gets a ``source_material`` info
    key so provenance survives the concatenation.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    combined: list[Atoms] = []
    for material, path in material_paths.items():
        read_configs = read(path, index=":")
        configs = read_configs if isinstance(read_configs, list) else [read_configs]
        for config in configs:
            config.info["source_material"] = material
        combined.extend(configs)

    write(out, combined, format="extxyz")
    return out


__all__ = [
    "MIN_RECOMMENDED_CONFIGS",
    "TRAINING_SET_DISCLOSURE",
    "IsolatedAtomReference",
    "compute_isolated_atom_reference",
    "save_isolated_atom_reference",
    "load_isolated_atom_reference",
    "export_mace_training_set",
    "summarize_training_set",
    "combine_training_sets",
    "generate_finetune_command",
]
