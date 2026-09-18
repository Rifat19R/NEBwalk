"""Tests for reproducibility bundle export."""

from __future__ import annotations

import hashlib
import json

from ase import Atoms
from ase.calculators.emt import EMT

import NEBwalk
from NEBwalk import NEBRunConfig, ReproBundle, run_neb_calculation, save_bundle


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _inputs():
    initial = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("Al", positions=[[0.1, 0.0, 0.0]])
    config = NEBRunConfig(
        n_images=3,
        interpolation="linear",
        max_steps=1,
        fmax=10.0,
        verbose=False,
    )
    return initial, final, config


def _result():
    initial, final, config = _inputs()
    result = run_neb_calculation(
        initial,
        final,
        EMT,
        config=config,
    )
    return initial, final, config, result


def _save(tmp_path, **kwargs):
    initial, final, config, result = _result()
    output_dir = tmp_path / "repro"
    options = {
        "include_env": False,
        "compress": False,
        **kwargs,
    }
    bundle = save_bundle(
        result,
        initial,
        final,
        config,
        output_dir=output_dir,
        **options,
    )
    return output_dir, bundle, config, result


def test_save_bundle_creates_all_files(tmp_path):
    initial, final, config, result = _result()
    output_dir = tmp_path / "repro"

    save_bundle(
        result,
        initial,
        final,
        config,
        output_dir=output_dir,
        compress=False,
        include_env=True,
    )

    expected = {
        "initial.xyz",
        "final.xyz",
        "neb_path.xyz",
        "neb_config.json",
        "calc_params.json",
        "results.json",
        "convergence_history.json",
        "environment.txt",
        "rerun_template.py",
        "manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected


def test_manifest_checksums_match(tmp_path):
    output_dir, _, _, _ = _save(tmp_path)

    manifest = json.loads((output_dir / "manifest.json").read_text())

    for filename, digest in manifest["files"].items():
        assert _sha256(output_dir / filename) == digest


def test_results_json_values(tmp_path):
    output_dir, _, config, result = _save(tmp_path)

    payload = json.loads((output_dir / "results.json").read_text())

    assert payload["schema"] == "nebwalk.results.v1"
    assert isinstance(payload["converged"], bool)
    assert isinstance(payload["forward_barrier_eV"], float)
    assert payload["forward_barrier_eV"] >= 0.0
    assert len(payload["image_energies_eV"]) == len(result.neb.images)
    assert len(payload["image_energies_eV"]) == config.n_images + 2


def test_neb_config_json_roundtrip(tmp_path):
    output_dir, _, config, _ = _save(tmp_path)

    payload = json.loads((output_dir / "neb_config.json").read_text())
    clean = {
        key: value
        for key, value in payload.items()
        if key not in ("schema", "nebwalk_version")
    }
    roundtrip = NEBRunConfig(**clean)

    assert roundtrip == config


def test_calc_params_stored_correctly(tmp_path):
    output_dir, _, _, _ = _save(
        tmp_path,
        calc_params={"type": "EMT", "device": "cpu"},
    )

    payload = json.loads((output_dir / "calc_params.json").read_text())

    assert payload["params"] == {"type": "EMT", "device": "cpu"}


def test_calc_params_none_writes_empty_dict(tmp_path):
    output_dir, _, _, _ = _save(tmp_path, calc_params=None)

    payload = json.loads((output_dir / "calc_params.json").read_text())

    assert payload["params"] == {}


def test_tarball_created_when_compress_true(tmp_path):
    initial, final, config, result = _result()
    output_dir = tmp_path / "repro"

    bundle = save_bundle(
        result,
        initial,
        final,
        config,
        output_dir=output_dir,
        compress=True,
        include_env=False,
    )

    assert bundle.tarball == tmp_path / "repro.tar.gz"
    assert bundle.tarball.exists()


def test_no_tarball_when_compress_false(tmp_path):
    output_dir, bundle, _, _ = _save(tmp_path, compress=False)

    assert bundle.tarball is None
    assert not output_dir.with_suffix(".tar.gz").exists()


def test_rerun_template_is_valid_python(tmp_path):
    output_dir, _, _, _ = _save(tmp_path)
    source = (output_dir / "rerun_template.py").read_text()

    compile(source, str(output_dir / "rerun_template.py"), "exec")


def test_rerun_template_contains_nebwalk_version(tmp_path):
    output_dir, _, _, _ = _save(tmp_path)

    assert NEBwalk.__version__ in (output_dir / "rerun_template.py").read_text()


def test_engine_reproduce_dir_kwarg(tmp_path):
    initial, final, config = _inputs()
    output_dir = tmp_path / "repro"

    run_neb_calculation(
        initial,
        final,
        EMT,
        config=config,
        reproduce_dir=output_dir,
        calc_params={"type": "EMT"},
    )

    assert output_dir.exists()
    assert (output_dir / "results.json").exists()


def test_engine_no_reproduce_dir_no_side_effects(tmp_path, monkeypatch):
    initial, final, config = _inputs()
    monkeypatch.chdir(tmp_path)

    run_neb_calculation(initial, final, EMT, config=config)

    assert not (tmp_path / "nebwalk_repro").exists()


def test_save_bundle_returns_reprobundle(tmp_path):
    output_dir, bundle, _, _ = _save(tmp_path)

    assert isinstance(bundle, ReproBundle)
    assert bundle.output_dir == output_dir
    assert bundle.nebwalk_version == NEBwalk.__version__


def test_environment_txt_written_when_include_env_true(tmp_path):
    initial, final, config, result = _result()
    output_dir = tmp_path / "repro"

    save_bundle(
        result,
        initial,
        final,
        config,
        output_dir=output_dir,
        compress=False,
        include_env=True,
    )

    assert (output_dir / "environment.txt").exists()
    assert (output_dir / "environment.txt").read_text()


def test_environment_txt_absent_when_include_env_false(tmp_path):
    output_dir, _, _, _ = _save(tmp_path, include_env=False)

    assert not (output_dir / "environment.txt").exists()


def test_convergence_history_json_structure(tmp_path):
    output_dir, _, _, _ = _save(tmp_path)

    payload = json.loads((output_dir / "convergence_history.json").read_text())

    assert isinstance(payload, list)
    assert payload
    for item in payload:
        assert {"step", "fmax", "energies"} <= item.keys()
        assert isinstance(item["fmax"], float)
        assert item["fmax"] >= 0.0
