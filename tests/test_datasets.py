"""Canonical active-learning dataset tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from NEBwalk.datasets import (
    DatasetValidationError,
    compute_dataset_checksum,
    compute_structure_hash,
    deduplicate_structures,
    load_dataset,
    merge_datasets,
    split_dataset_by_group,
    summarize_dataset,
    validate_dataset,
    write_dataset,
)


def _frame(
    path_id: str = "path-0",
    image_index: int = 0,
    *,
    energy: float = -1.0,
    settings: str = "qe-settings-a",
) -> Atoms:
    atoms = Atoms(
        "HHe",
        positions=[[0.0, 0.0, 0.0], [0.75 + 0.01 * image_index, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    atoms.info.update(
        {
            "REF_energy": energy,
            "config_type": "neb_image",
            "campaign_id": "campaign-test",
            "iteration": 0,
            "path_id": path_id,
            "image_index": image_index,
            "selection_reason": "peak",
            "calculator_name": "QE/PBE",
            "dft_settings_hash": settings,
        }
    )
    atoms.arrays["REF_forces"] = np.full((len(atoms), 3), image_index * 0.01)
    atoms.info["structure_hash"] = compute_structure_hash(atoms)
    return atoms


def test_extxyz_round_trip_preserves_labels_metadata_and_constraints(tmp_path):
    frame = _frame()
    frame.set_constraint(FixAtoms(indices=[0]))

    artifact = write_dataset(tmp_path / "dataset.extxyz", [frame])
    loaded = load_dataset(artifact.path)

    assert loaded[0].info["REF_energy"] == pytest.approx(-1.0)
    np.testing.assert_allclose(loaded[0].arrays["REF_forces"], np.zeros((2, 3)))
    assert loaded[0].info["path_id"] == "path-0"
    assert loaded[0].constraints
    assert artifact.checksum == compute_dataset_checksum(artifact.path)


@pytest.mark.parametrize("missing", ["REF_energy", "REF_forces", "path_id"])
def test_required_labels_and_metadata_are_enforced(missing):
    frame = _frame()
    if missing in frame.info:
        del frame.info[missing]
    else:
        del frame.arrays[missing]

    with pytest.raises(DatasetValidationError, match=missing):
        validate_dataset([frame])


def test_malformed_force_shape_is_rejected():
    frame = _frame()
    frame.arrays["REF_forces"] = np.zeros((2, 2))

    with pytest.raises(DatasetValidationError, match="shape"):
        validate_dataset([frame])


@pytest.mark.parametrize("field", ["energy", "forces", "positions", "cell"])
def test_nonfinite_data_is_rejected(field):
    frame = _frame()
    if field == "energy":
        frame.info["REF_energy"] = np.nan
    elif field == "forces":
        frame.arrays["REF_forces"][0, 0] = np.inf
    elif field == "positions":
        frame.positions[0, 0] = np.nan
    else:
        frame.cell[0, 0] = np.inf
    with pytest.raises((DatasetValidationError, ValueError), match="nonfinite|NaN"):
        validate_dataset([frame])


def test_dangerously_short_distance_is_rejected():
    frame = _frame()
    frame.positions[1] = [0.1, 0.0, 0.0]
    frame.info["structure_hash"] = compute_structure_hash(frame)

    with pytest.raises(DatasetValidationError, match="dangerously short"):
        validate_dataset([frame])


def test_periodic_axis_requires_nonzero_cell_vector():
    frame = _frame()
    frame.pbc = [True, False, False]
    frame.cell[0] = 0.0
    frame.info["structure_hash"] = compute_structure_hash(frame)

    with pytest.raises(DatasetValidationError, match="zero cell"):
        validate_dataset([frame])


def test_element_order_and_cell_must_match_within_path():
    reference = _frame("same-path", 0)
    reordered = _frame("same-path", 1)
    reordered.symbols = reordered.symbols[::-1]
    reordered.info["structure_hash"] = compute_structure_hash(reordered)
    with pytest.raises(DatasetValidationError, match="element order"):
        validate_dataset([reference, reordered])

    changed_cell = _frame("same-path", 1)
    changed_cell.cell[0, 0] += 1.0
    changed_cell.info["structure_hash"] = compute_structure_hash(changed_cell)
    with pytest.raises(DatasetValidationError, match="cell/PBC"):
        validate_dataset([reference, changed_cell])


def test_atom_count_must_match_within_path():
    reference = _frame("same-path", 0)
    changed = _frame("same-path", 1)
    changed += Atoms("H", positions=[[4.0, 4.0, 4.0]])
    changed.arrays["REF_forces"] = np.zeros((len(changed), 3))
    changed.info["structure_hash"] = compute_structure_hash(changed)

    with pytest.raises(DatasetValidationError, match="atom count"):
        validate_dataset([reference, changed])


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("path_id", "", "malformed string metadata"),
        ("calculator_name", {"name": "QE"}, "malformed string metadata"),
        ("iteration", 1.5, "iteration must be an integer"),
        ("image_index", -1, "image_index must be >= 0"),
        ("structure_hash", "not-a-hash", "64-character SHA-256"),
    ],
)
def test_malformed_metadata_is_rejected(key, value, message):
    frame = _frame()
    frame.info[key] = value

    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset([frame])


def test_duplicate_detection_and_deduplication():
    frame = _frame()

    with pytest.raises(DatasetValidationError, match="duplicate"):
        validate_dataset([frame, frame.copy()])
    unique = deduplicate_structures([frame, frame.copy()])
    assert len(unique) == 1


def test_duplicate_with_conflicting_label_is_rejected():
    first = _frame(energy=-1.0)
    conflict = first.copy()
    conflict.info["REF_energy"] = -2.0

    with pytest.raises(DatasetValidationError, match="conflicting"):
        deduplicate_structures([first, conflict])


def test_mixed_dft_settings_are_rejected():
    frames = [
        _frame("path-a", 0, settings="settings-a"),
        _frame("path-b", 1, settings="settings-b"),
    ]

    with pytest.raises(DatasetValidationError, match="incompatible"):
        validate_dataset(frames)


def test_structure_hash_is_deterministic_and_geometry_sensitive():
    first = _frame()
    second = first.copy()
    assert compute_structure_hash(first) == compute_structure_hash(second)

    second.positions[1, 0] += 0.01
    assert compute_structure_hash(first) != compute_structure_hash(second)


def test_dataset_checksum_is_order_independent_for_frames():
    frames = [_frame("path-a", 0), _frame("path-b", 1)]

    assert compute_dataset_checksum(frames) == compute_dataset_checksum(frames[::-1])


def test_write_dataset_is_deterministic_and_has_atomic_manifest(tmp_path):
    frames = [_frame("path-b", 1), _frame("path-a", 0)]

    first = write_dataset(tmp_path / "first.extxyz", frames)
    second = write_dataset(tmp_path / "second.extxyz", frames[::-1])

    assert first.checksum == second.checksum
    payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "nebwalk.dataset.v1"
    assert payload["sha256"] == first.checksum
    assert not list(tmp_path.glob(".*.json.*"))


def test_group_split_is_deterministic_and_has_no_path_leakage(tmp_path):
    frames = []
    for path in range(10):
        for image in range(2):
            frame = _frame(f"path-{path}", image, energy=-1.0 + 0.01 * image)
            frame.positions[:, 1] += path * 0.05
            frame.info["structure_hash"] = compute_structure_hash(frame)
            frames.append(frame)

    first = split_dataset_by_group(frames, tmp_path / "split-a", seed=42)
    second = split_dataset_by_group(frames, tmp_path / "split-b", seed=42)

    assert first.group_assignments == second.group_assignments
    assert first.train.checksum == second.train.checksum
    assert first.valid.checksum == second.valid.checksum
    assert first.test.checksum == second.test.checksum
    split_groups = {
        name: {atoms.info["path_id"] for atoms in load_dataset(artifact.path)}
        for name, artifact in (
            ("train", first.train),
            ("valid", first.valid),
            ("test", first.test),
        )
    }
    assert split_groups["train"].isdisjoint(split_groups["valid"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["valid"].isdisjoint(split_groups["test"])


def test_group_split_rejects_too_few_groups_for_nonempty_partitions(tmp_path):
    frames = [_frame("only-path", 0)]

    with pytest.raises(DatasetValidationError, match="cannot populate"):
        split_dataset_by_group(frames, tmp_path / "split")


def test_merge_preserves_inputs_and_removes_identical_duplicates():
    first = _frame("path-a", 0)
    second = _frame("path-b", 1)
    original_position = first.positions.copy()

    merged = merge_datasets([first], [first.copy(), second])

    assert len(merged) == 2
    np.testing.assert_allclose(first.positions, original_position)
    assert summarize_dataset(merged).n_configurations == 2
