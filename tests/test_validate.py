"""Tests for the per-material checkpoint/validation bundle."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms

from nebwalk.finetune import IsolatedAtomReference, export_mace_training_set
from nebwalk.label import DFTLabel
from nebwalk.qe import QEParams
from nebwalk.validate import (
    MaterialValidationSummary,
    QESettingsRecord,
    build_material_validation_summary,
    check_finite,
    check_qe_scf_converged,
    cohesive_energy_check,
    mace_loader_read_test,
    nebwalk_dataset_validation_check,
    qe_settings_hash,
    qe_settings_record,
    sha256_of_file,
    write_manifest,
)


def _label(
    index: int, energy: float, forces: np.ndarray, is_reference: bool = False
) -> DFTLabel:
    return DFTLabel(
        index=index,
        energy_eV=energy,
        forces_eV_A=forces,
        mlip_energy_eV=energy + 0.1,
        dft_relative_energy_eV=0.0 if is_reference else 0.5,
        mlip_relative_energy_eV=0.0 if is_reference else 0.4,
        relative_energy_disagreement_eV=0.1,
        max_force_disagreement_eV_A=0.05,
        is_reference=is_reference,
    )


def _isolated_ref(symbol: str, energy: float) -> IsolatedAtomReference:
    atom = Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)
    atom.info["REF_energy"] = energy
    atom.new_array("REF_forces", np.zeros((1, 3)))
    atom.info["config_type"] = "IsolatedAtom"
    return IsolatedAtomReference(
        symbol=symbol, energy_eV=energy, forces_eV_A=np.zeros((1, 3)), atoms=atom
    )


def _export(tmp_path, images, labels, dft_settings_hash="a" * 64, path_id="al_vacancy"):
    return export_mace_training_set(
        images,
        labels,
        tmp_path / "train.extxyz",
        path_id=path_id,
        dft_settings_hash=dft_settings_hash,
    )


def test_sha256_of_file_matches_known_digest(tmp_path):
    path = tmp_path / "pseudo.upf"
    path.write_text("hello world", encoding="utf-8")

    digest = sha256_of_file(path)

    assert digest == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )
    assert len(digest) == 64


def test_qe_settings_record_captures_smearing_and_magnetism(tmp_path):
    pseudo = tmp_path / "Ni.UPF"
    pseudo.write_text("pseudo data", encoding="utf-8")
    params = QEParams(
        ecutwfc=50.0,
        ecutrho=400.0,
        kpts=(2, 2, 2),
        occupations="smearing",
        smearing="marzari-vanderbilt",
        degauss=0.02,
        conv_thr=1e-7,
        mixing_beta=0.3,
        nspin=2,
        starting_magnetization={"Ni": 0.3},
    )

    record = qe_settings_record(tmp_path, "Ni.UPF", params)

    assert record.pseudopotential_sha256 == sha256_of_file(pseudo)
    assert record.nspin == 2
    assert record.starting_magnetization == {"Ni": 0.3}
    assert record.smearing == "marzari-vanderbilt"
    assert record.degauss == pytest.approx(0.02)


def test_qe_settings_record_omits_smearing_for_fixed_occupations(tmp_path):
    pseudo = tmp_path / "Si.UPF"
    pseudo.write_text("pseudo data", encoding="utf-8")
    params = QEParams(occupations="fixed", nspin=1)

    record = qe_settings_record(tmp_path, "Si.UPF", params)

    assert record.smearing is None
    assert record.degauss is None
    assert record.starting_magnetization is None


def test_qe_settings_hash_is_deterministic_and_settings_sensitive(tmp_path):
    pseudo = tmp_path / "Al.UPF"
    pseudo.write_text("pseudo data", encoding="utf-8")
    bulk = qe_settings_record(tmp_path, "Al.UPF", QEParams(kpts=(4, 4, 4)))
    bulk_again = qe_settings_record(tmp_path, "Al.UPF", QEParams(kpts=(4, 4, 4)))
    isolated = qe_settings_record(tmp_path, "Al.UPF", QEParams(kpts=(1, 1, 1)))

    assert qe_settings_hash(bulk) == qe_settings_hash(bulk_again)
    assert qe_settings_hash(bulk) != qe_settings_hash(isolated)
    assert len(qe_settings_hash(bulk)) == 64


def test_check_finite_passes_clean_data():
    labels = [_label(0, -100.0, np.array([[0.0, 0.0, 0.0]]), is_reference=True)]
    ref = _isolated_ref("Al", -533.9)

    assert check_finite(labels, [ref]) == []


def test_check_finite_catches_nan_energy_and_forces():
    labels = [
        _label(0, float("nan"), np.array([[0.0, 0.0, 0.0]]), is_reference=True),
        _label(1, -99.0, np.array([[float("inf"), 0.0, 0.0]])),
    ]

    problems = check_finite(labels)

    assert any("image 0" in p and "energy" in p for p in problems)
    assert any("image 1" in p and "forces" in p for p in problems)


def test_check_finite_catches_bad_isolated_atom():
    ref = _isolated_ref("Al", float("nan"))

    problems = check_finite([], [ref])

    assert any("isolated atom Al" in p and "energy" in p for p in problems)


def test_cohesive_energy_check_accepts_normal_value():
    ref = _isolated_ref("Al", -533.92)
    result = cohesive_energy_check(
        reference_energy_eV=31 * -533.92 + 31 * -3.48,
        n_atoms=31,
        isolated_atom_reference=ref,
    )

    assert result["flags"] == []
    assert result["cohesive_energy_eV_per_atom"] == pytest.approx(-3.48, abs=1e-6)


def test_cohesive_energy_check_flags_unbound_result():
    ref = _isolated_ref("Al", -533.92)
    result = cohesive_energy_check(
        reference_energy_eV=31 * -533.92 + 31 * 1.0,
        n_atoms=31,
        isolated_atom_reference=ref,
    )

    assert any("not negative" in flag for flag in result["flags"])


def test_cohesive_energy_check_flags_implausibly_large():
    ref = _isolated_ref("Al", -533.92)
    result = cohesive_energy_check(
        reference_energy_eV=31 * -533.92 + 31 * -50.0,
        n_atoms=31,
        isolated_atom_reference=ref,
    )

    assert any("implausibly large" in flag for flag in result["flags"])


def test_mace_loader_read_test_reads_real_export(tmp_path):
    pytest.importorskip("mace")

    images = [Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)]
    labels = [_label(0, -100.0, np.array([[0.0, 0.0, 0.0]]), is_reference=True)]
    artifact = _export(tmp_path, images, labels)

    result = mace_loader_read_test(artifact.path)

    assert result["ok"] is True
    assert result["n_configs_loaded"] == 1
    assert result["config_types"] == ["Default"]


def test_nebwalk_dataset_validation_check_accepts_real_export(tmp_path):
    images = [Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)]
    labels = [_label(0, -100.0, np.array([[0.0, 0.0, 0.0]]), is_reference=True)]
    artifact = _export(tmp_path, images, labels)

    result = nebwalk_dataset_validation_check(artifact.path)

    assert result["ok"] is True
    assert result["error"] is None
    assert result["n_configurations"] == 1
    assert result["elements"] == ["Al"]


def test_nebwalk_dataset_validation_check_reports_missing_file(tmp_path):
    result = nebwalk_dataset_validation_check(tmp_path / "does_not_exist.extxyz")

    assert result["ok"] is False
    assert result["error"]


def test_build_material_validation_summary_writes_files(tmp_path):
    pytest.importorskip("mace")

    pseudo = tmp_path / "Al.UPF"
    pseudo.write_text("pseudo data", encoding="utf-8")
    images = [Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)]
    ref_label = _label(
        0, 31 * -533.92 + 31 * -3.48, np.zeros((1, 3)), is_reference=True
    )
    labels = [ref_label]
    iso_ref = _isolated_ref("Al", -533.92)
    artifact = _export(tmp_path, images, labels)
    params = QEParams(ecutwfc=50.0, ecutrho=400.0, kpts=(2, 2, 2))

    summary = build_material_validation_summary(
        material="al",
        symbol="Al",
        labels=labels,
        failed_indices=[],
        reference_label=ref_label,
        n_atoms=31,
        isolated_atom_reference=iso_ref,
        pseudo_dir=tmp_path,
        pseudo_file="Al.UPF",
        params=params,
        training_set_path=artifact.path,
        output_dir=tmp_path / "out",
    )

    assert summary.passed is True
    assert summary.nebwalk_dataset_ok is True
    assert summary.nebwalk_dataset_error is None
    assert (tmp_path / "out" / "validation_summary.json").exists()
    assert (tmp_path / "out" / "VALIDATION_SUMMARY.md").exists()

    payload = json.loads((tmp_path / "out" / "validation_summary.json").read_text())
    assert payload["material"] == "al"
    assert payload["qe_settings"]["pseudopotential_sha256"] == sha256_of_file(pseudo)
    assert payload["nebwalk_dataset_ok"] is True

    markdown = (tmp_path / "out" / "VALIDATION_SUMMARY.md").read_text()
    assert "PASSED" in markdown
    assert "nebwalk.datasets schema validation" in markdown


def _write_pwo(path, converged: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "     total energy              =    -100.0 Ry\n"
        + (
            "\n     End of self-consistent calculation\n"
            if converged
            else "     convergence NOT achieved after 100 iterations: stopping\n"
        )
        + "\n     JOB DONE.\n"
    )
    path.write_text(body, encoding="utf-8")


def test_check_qe_scf_converged_reports_clean_workdir(tmp_path):
    _write_pwo(tmp_path / "image_000" / "espresso.pwo", converged=True)
    _write_pwo(tmp_path / "image_001" / "espresso.pwo", converged=True)

    result = check_qe_scf_converged(tmp_path)

    assert result["checked"] == 2
    assert result["problems"] == []
    assert result["converged_cleanly"] is True


def test_check_qe_scf_converged_flags_non_converged_image(tmp_path):
    _write_pwo(tmp_path / "image_000" / "espresso.pwo", converged=True)
    _write_pwo(tmp_path / "image_001" / "espresso.pwo", converged=False)

    result = check_qe_scf_converged(tmp_path)

    assert result["checked"] == 2
    assert result["converged_cleanly"] is False
    assert any("image_001" in p for p in result["problems"])
    assert not any("image_000" in p for p in result["problems"])


def test_build_material_validation_summary_fails_on_scf_non_convergence(tmp_path):
    pytest.importorskip("mace")

    pseudo = tmp_path / "Al.UPF"
    pseudo.write_text("pseudo data", encoding="utf-8")
    images = [Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)]
    ref_label = _label(
        0, 31 * -533.92 + 31 * -3.48, np.zeros((1, 3)), is_reference=True
    )
    labels = [ref_label]
    iso_ref = _isolated_ref("Al", -533.92)
    artifact = _export(tmp_path, images, labels)
    params = QEParams(ecutwfc=50.0, ecutrho=400.0, kpts=(2, 2, 2))
    _write_pwo(tmp_path / "qe_workdir" / "image_000" / "espresso.pwo", converged=False)

    summary = build_material_validation_summary(
        material="al",
        symbol="Al",
        labels=labels,
        failed_indices=[],
        reference_label=ref_label,
        n_atoms=31,
        isolated_atom_reference=iso_ref,
        pseudo_dir=tmp_path,
        pseudo_file="Al.UPF",
        params=params,
        training_set_path=artifact.path,
        output_dir=tmp_path / "out",
        qe_workdirs=[tmp_path / "qe_workdir"],
    )

    assert summary.passed is False
    assert summary.scf_converged_cleanly is False
    assert any("image_000" in p for p in summary.scf_problems)
    markdown = (tmp_path / "out" / "VALIDATION_SUMMARY.md").read_text()
    assert "FAILED" in markdown


def test_build_material_validation_summary_fails_on_broken_dataset_file(tmp_path):
    pytest.importorskip("mace")
    from ase.io import write as ase_write

    pseudo = tmp_path / "Al.UPF"
    pseudo.write_text("pseudo data", encoding="utf-8")
    ref_label = _label(
        0, 31 * -533.92 + 31 * -3.48, np.zeros((1, 3)), is_reference=True
    )
    labels = [ref_label]
    iso_ref = _isolated_ref("Al", -533.92)
    params = QEParams(ecutwfc=50.0, ecutrho=400.0, kpts=(2, 2, 2))

    # Well-formed enough for ASE/MACE's own loader (has REF_energy/REF_forces),
    # but missing nebwalk.datasets.REQUIRED_INFO_KEYS (e.g. structure_hash,
    # path_id) -- this is the realistic failure mode the schema gate exists
    # to catch, distinct from a file being unreadable outright.
    broken_path = tmp_path / "broken.extxyz"
    incomplete = Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    incomplete.info["REF_energy"] = -533.92
    incomplete.new_array("REF_forces", np.zeros((1, 3)))
    ase_write(broken_path, [incomplete], format="extxyz")

    summary = build_material_validation_summary(
        material="al",
        symbol="Al",
        labels=labels,
        failed_indices=[],
        reference_label=ref_label,
        n_atoms=31,
        isolated_atom_reference=iso_ref,
        pseudo_dir=tmp_path,
        pseudo_file="Al.UPF",
        params=params,
        training_set_path=broken_path,
        output_dir=tmp_path / "out",
    )

    assert summary.nebwalk_dataset_ok is False
    assert summary.passed is False


def _make_summary(
    material: str, symbol: str, passed: bool
) -> MaterialValidationSummary:
    return MaterialValidationSummary(
        material=material,
        symbol=symbol,
        n_labeled=4 if passed else 1,
        n_failed=0 if passed else 3,
        qe_settings=_dummy_qe_settings(),
        finite_check_problems=[],
        cohesive_energy_eV_per_atom=-3.0,
        cohesive_energy_flags=[],
        mace_loader_ok=True,
        mace_loader_n_configs=5,
        mace_loader_config_types=["Default"],
        scf_converged_cleanly=passed,
        scf_problems=[] if passed else ["image_001: SCF did not converge"],
        scf_images_checked=3,
        nebwalk_dataset_ok=passed,
        nebwalk_dataset_error=None if passed else "dataset validation failed",
        passed=passed,
    )


def _dummy_qe_settings() -> QESettingsRecord:
    return QESettingsRecord(
        pseudopotential_file="X.UPF",
        pseudopotential_sha256="deadbeef",
        ecutwfc=50.0,
        ecutrho=400.0,
        kpts=(2, 2, 2),
        occupations="smearing",
        smearing="marzari-vanderbilt",
        degauss=0.02,
        nspin=1,
        starting_magnetization=None,
        conv_thr=1e-7,
        mixing_beta=0.3,
    )


def test_write_manifest_indexes_materials_without_merging(tmp_path):
    summaries = [
        _make_summary("al", "Al", passed=True),
        _make_summary("ni", "Ni", passed=False),
    ]
    dataset_paths = {
        "al": "datasets/Al/al_train.extxyz",
        "ni": "datasets/Ni/ni_train.extxyz",
    }

    manifest_path = write_manifest(summaries, dataset_paths, tmp_path / "manifest.json")

    payload = json.loads(manifest_path.read_text())
    assert payload["n_materials"] == 2
    assert payload["n_passed"] == 1
    by_material = {e["material"]: e for e in payload["materials"]}
    assert by_material["al"]["passed"] is True
    assert by_material["ni"]["passed"] is False
    assert by_material["al"]["dataset_path"] == "datasets/Al/al_train.extxyz"
