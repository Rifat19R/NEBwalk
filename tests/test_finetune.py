"""Tests for MACE-compatible training-set export (active-learning stage 3)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from nebwalk.datasets import DatasetArtifact, load_dataset
from nebwalk.finetune import (
    IsolatedAtomReference,
    combine_training_sets,
    compute_isolated_atom_reference,
    export_mace_training_set,
    generate_finetune_command,
    load_isolated_atom_reference,
    save_isolated_atom_reference,
    summarize_training_set,
)
from nebwalk.label import DFTLabel
from nebwalk.recovery import FailureType, NoOpRecoveryStrategy, RecoveryExhausted


class ConstantCalculator:
    """Fixed energy/forces calculator, mirroring tests/test_label.py."""

    def __init__(self, energy: float, forces: np.ndarray) -> None:
        self.energy = float(energy)
        self.forces = np.asarray(forces, dtype=float)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.energy

    def get_forces(self, atoms=None):
        return self.forces.copy()


class AlwaysFailsCalculator:
    def get_potential_energy(self, atoms=None, force_consistent=False):
        raise RuntimeError("QE process failure")

    def get_forces(self, atoms=None):
        raise RuntimeError("QE process failure")


class _NonRetryableStrategy(NoOpRecoveryStrategy):
    def classify(self, error, raw_output=None):
        return FailureType.PROCESS_FAILURE


def _image(symbol: str = "Al", offset: float = 0.0) -> Atoms:
    return Atoms(
        symbol, positions=[[offset, 0.0, 0.0]], cell=[10, 10, 10], pbc=True
    )


def _isolated_ref(symbol: str, energy: float) -> IsolatedAtomReference:
    atom = Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)
    return compute_isolated_atom_reference(
        atom,
        dft_calculator_factory=lambda: ConstantCalculator(energy, np.zeros((1, 3))),
    )


def _label(index: int, energy: float, forces: np.ndarray) -> DFTLabel:
    return DFTLabel(
        index=index,
        energy_eV=energy,
        forces_eV_A=forces,
        mlip_energy_eV=energy + 0.1,
        dft_relative_energy_eV=energy + 100.0,
        mlip_relative_energy_eV=energy + 100.1,
        relative_energy_disagreement_eV=-0.1,
        max_force_disagreement_eV_A=0.05,
        is_reference=(index == 0),
    )


def test_export_mace_training_set_writes_valid_extxyz(tmp_path):
    images = [_image(offset=0.0), _image(offset=0.3)]
    labels = [
        _label(0, -100.0, np.array([[0.0, 0.0, 0.0]])),
        _label(1, -99.5, np.array([[0.1, 0.0, 0.0]])),
    ]

    artifact = export_mace_training_set(
        images,
        labels,
        tmp_path / "train.extxyz",
        path_id="al_vacancy",
        dft_settings_hash="a" * 64,
    )

    assert isinstance(artifact, DatasetArtifact)
    configs = read(artifact.path, index=":")
    assert len(configs) == 2
    by_index = {c.info["image_index"]: c for c in configs}
    assert by_index[0].info["REF_energy"] == pytest.approx(-100.0)
    assert by_index[1].info["REF_energy"] == pytest.approx(-99.5)
    np.testing.assert_allclose(by_index[1].arrays["REF_forces"], [[0.1, 0.0, 0.0]])
    assert by_index[0].info["config_type"] == "Default"
    assert by_index[0].info["selection_reason"] == "reference"
    assert by_index[1].info["selection_reason"] == "peak_plus_neighbors"
    assert by_index[0].info["path_id"] == "al_vacancy"
    assert by_index[0].calc is None


def test_export_mace_training_set_selection_reason_overrides(tmp_path):
    images = [_image(offset=0.0), _image(offset=0.3), _image(offset=0.6)]
    labels = [
        _label(0, -100.0, np.array([[0.0, 0.0, 0.0]])),
        _label(1, -99.8, np.array([[0.05, 0.0, 0.0]])),
        _label(2, -99.5, np.array([[0.1, 0.0, 0.0]])),
    ]

    artifact = export_mace_training_set(
        images,
        labels,
        tmp_path / "train.extxyz",
        path_id="al_vacancy",
        dft_settings_hash="1" * 64,
        selection_reason_overrides={1: "full_path_completion"},
    )

    by_index = {c.info["image_index"]: c for c in read(artifact.path, index=":")}
    assert by_index[0].info["selection_reason"] == "reference"
    assert by_index[1].info["selection_reason"] == "full_path_completion"
    assert by_index[2].info["selection_reason"] == "peak_plus_neighbors"


def test_export_mace_training_set_loads_through_real_datasets_validator(tmp_path):
    images = [_image()]
    labels = [_label(0, -100.0, np.array([[0.0, 0.0, 0.0]]))]

    artifact = export_mace_training_set(
        images,
        labels,
        tmp_path / "train.extxyz",
        path_id="al_vacancy",
        dft_settings_hash="b" * 64,
    )

    reloaded = load_dataset(artifact.path)
    assert len(reloaded) == 1
    assert reloaded[0].info["dft_settings_hash"] == "b" * 64
    assert artifact.summary.dft_settings_hashes == ("b" * 64,)


def test_export_mace_training_set_writes_disclosure_readme(tmp_path):
    images = [_image()]
    labels = [_label(0, -100.0, np.array([[0.0, 0.0, 0.0]]))]

    artifact = export_mace_training_set(
        images,
        labels,
        tmp_path / "train.extxyz",
        path_id="al_vacancy",
        dft_settings_hash="c" * 64,
    )

    readme = artifact.path.parent / "train_README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "NOT enough data" in text
    assert "does NOT contain an IsolatedAtom entry" in text


def test_compute_isolated_atom_reference_rejects_multi_atom_input():
    dimer = Atoms("Al2", positions=[[0, 0, 0], [2, 0, 0]], cell=[10, 10, 10], pbc=True)

    with pytest.raises(ValueError, match="single-atom"):
        compute_isolated_atom_reference(
            dimer,
            dft_calculator_factory=lambda: ConstantCalculator(0.0, np.zeros((2, 3))),
        )


def test_compute_isolated_atom_reference_returns_symbol_and_energy():
    atom = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[15, 15, 15], pbc=True)

    ref = compute_isolated_atom_reference(
        atom,
        dft_calculator_factory=lambda: ConstantCalculator(
            -3.7, np.array([[1e-6, 0.0, 0.0]])
        ),
    )

    assert isinstance(ref, IsolatedAtomReference)
    assert ref.symbol == "Cu"
    assert ref.energy_eV == pytest.approx(-3.7)
    assert ref.atoms.info["config_type"] == "IsolatedAtom"
    assert len(ref.atoms) == 1


def test_compute_isolated_atom_reference_propagates_unrecoverable_failure():
    atom = Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[15, 15, 15], pbc=True)

    with pytest.raises(RecoveryExhausted):
        compute_isolated_atom_reference(
            atom,
            dft_calculator_factory=AlwaysFailsCalculator,
            recovery_strategy=_NonRetryableStrategy(),
        )


def test_save_and_load_isolated_atom_reference_round_trips(tmp_path):
    ref = _isolated_ref("Ni", -1234.5)

    out = save_isolated_atom_reference(
        ref, tmp_path / "isolated_atom_reference.json", dft_settings_hash="d" * 64
    )
    reloaded = load_isolated_atom_reference(out)

    assert reloaded.symbol == "Ni"
    assert reloaded.energy_eV == pytest.approx(-1234.5)
    np.testing.assert_allclose(reloaded.forces_eV_A, ref.forces_eV_A)
    np.testing.assert_allclose(reloaded.atoms.cell.array, ref.atoms.cell.array)
    assert reloaded.atoms.info["config_type"] == "IsolatedAtom"


def test_generate_finetune_command_defaults_to_e0s_average():
    command = generate_finetune_command(
        train_file="train.extxyz",
        foundation_model="medium",
        device="cuda",
        name="al_vacancy_test",
    )

    assert "mace_run_train" in command
    assert "--E0s=average" in command
    assert "--foundation_model=medium" in command
    assert "--train_file=train.extxyz" in command
    assert "--device=cuda" in command
    assert "--name=al_vacancy_test" in command


def test_generate_finetune_command_uses_explicit_e0s_when_references_given():
    ref = _isolated_ref("Al", -533.9)

    command = generate_finetune_command(
        train_file="train.extxyz", isolated_atom_references=[ref]
    )

    assert "--E0s=average" not in command
    assert "--E0s=" in command
    assert "13" in command  # Al's atomic number
    assert "-533.9" in command


def test_combine_training_sets_concatenates_without_touching_sources(tmp_path):
    al_path = tmp_path / "al" / "al_train.extxyz"
    cu_path = tmp_path / "cu" / "cu_train.extxyz"
    export_mace_training_set(
        images=[_image("Al")],
        labels=[_label(0, -1000.0, np.array([[0.0, 0.0, 0.0]]))],
        output_path=al_path,
        path_id="al_vacancy",
        dft_settings_hash="e" * 64,
    )
    export_mace_training_set(
        images=[_image("Cu")],
        labels=[_label(0, -900.0, np.array([[0.0, 0.0, 0.0]]))],
        output_path=cu_path,
        path_id="cu_vacancy",
        dft_settings_hash="f" * 64,
    )
    al_configs_before = read(al_path, index=":")

    combined_path = combine_training_sets(
        {"al": al_path, "cu": cu_path}, tmp_path / "combined.extxyz"
    )

    combined = read(combined_path, index=":")
    assert len(combined) == 2
    sources = {c.info["source_material"] for c in combined}
    assert sources == {"al", "cu"}
    # Source files must be untouched by combining.
    assert len(read(al_path, index=":")) == len(al_configs_before)
    assert "source_material" not in read(al_path, index=":")[0].info


def test_summarize_training_set_reflects_real_dataset_contents(tmp_path):
    artifact = export_mace_training_set(
        images=[_image("Al", offset=0.0), _image("Al", offset=0.3)],
        labels=[
            _label(0, -1000.0, np.array([[0.0, 0.0, 0.0]])),
            _label(1, -999.5, np.array([[0.0, 0.0, 0.0]])),
        ],
        output_path=tmp_path / "train.extxyz",
        path_id="al_vacancy",
        dft_settings_hash="0a" * 32,
    )

    summary = summarize_training_set(artifact.path)

    assert summary["total_configs"] == 2
    assert summary["path_configs"] == 2
    assert summary["elements_present"] == ["Al"]
    assert summary["path_ids"] == ["al_vacancy"]
