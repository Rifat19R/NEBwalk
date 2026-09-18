"""Tests for the MLIP-assisted NEB workflow layer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from ase import Atoms

from NEBwalk import NEBRunConfig
from NEBwalk.active import (
    MLIPActiveNEBConfig,
    SelectedImage,
    export_selected_images,
    run_mlip_assisted_neb,
)


def _images(n_images: int = 5) -> list[Atoms]:
    return [Atoms("Al", positions=[[0.1 * idx, 0.0, 0.0]]) for idx in range(n_images)]


def test_active_config_defaults():
    config = MLIPActiveNEBConfig()

    assert config.selection_strategy == "peak_plus_neighbors"
    assert config.n_select == 3
    assert config.include_endpoints is False
    assert config.export_selected is True
    assert config.export_formats == ("xyz", "traj", "json")


def test_selected_image_stores_metadata():
    selected = SelectedImage(index=2, energy=-3.0, relative_energy=0.4)

    assert selected.index == 2
    assert selected.energy == pytest.approx(-3.0)
    assert selected.relative_energy == pytest.approx(0.4)
    assert selected.reason == "peak_plus_neighbors"


def test_export_selected_images_creates_requested_files(tmp_path):
    selected = (
        SelectedImage(index=1, energy=-1.0, relative_energy=0.1),
        SelectedImage(index=2, energy=-0.8, relative_energy=0.3),
    )

    output_dir = export_selected_images(
        images=_images(),
        selected=selected,
        output_dir=tmp_path / "selected",
        export_formats=("xyz", "traj", "json"),
        metadata={"stage": "mlip_assisted_neb"},
    )

    assert (output_dir / "selected_images.json").exists()
    assert (output_dir / "selected_images.traj").exists()
    assert (output_dir / "selected_image_01.xyz").exists()
    assert (output_dir / "selected_image_02.xyz").exists()
    assert (output_dir / "README.md").exists()


def test_export_selected_images_json_schema(tmp_path):
    selected = (SelectedImage(index=1, energy=-1.0, relative_energy=0.1),)

    output_dir = export_selected_images(
        images=_images(),
        selected=selected,
        output_dir=tmp_path / "selected",
        export_formats=("json",),
        metadata={"nebwalk_version": "0.7.0"},
    )

    payload = json.loads((output_dir / "selected_images.json").read_text())
    assert payload["schema"] == "nebwalk.selected_images.v1"
    assert payload["selection_strategy"] == "peak_plus_neighbors"
    assert payload["selected_indices"] == [1]
    assert payload["selected_images"][0]["index"] == 1
    assert "DFT/QE refinement" in payload["notes"]
    assert payload["metadata"]["nebwalk_version"] == "0.7.0"


def test_export_readme_reflects_strategy(tmp_path):
    selected = (
        SelectedImage(
            index=1,
            energy=0.5,
            relative_energy=0.5,
            reason="custom_strategy",
        ),
    )

    output_dir = export_selected_images(
        images=_images(),
        selected=selected,
        output_dir=tmp_path / "selected",
        export_formats=("json",),
    )
    text = (output_dir / "README.md").read_text(encoding="utf-8")

    assert "custom_strategy" in text
    assert "peak_plus_neighbors" not in text


def test_export_selected_images_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="unsupported export format"):
        export_selected_images(
            images=_images(),
            selected=(SelectedImage(index=1, energy=0.0, relative_energy=0.0),),
            output_dir=tmp_path / "selected",
            export_formats=("cif",),
        )


def test_run_mlip_assisted_neb_returns_selection_and_export(monkeypatch, tmp_path):
    energies = [0.0, 0.1, 0.5, 0.2, 0.0]
    images = _images()
    neb = SimpleNamespace(images=images, get_energies=lambda: energies)
    neb_result = SimpleNamespace(
        neb=neb,
        converged=True,
        barrier=0.5,
        reverse_barrier=0.5,
        reaction_energy=0.0,
    )

    def fake_runner(initial, final, calculator_factory, config):
        assert isinstance(config, NEBRunConfig)
        assert calculator_factory() == "calc"
        return neb_result

    monkeypatch.setattr("NEBwalk.active.run_neb_calculation", fake_runner)

    result = run_mlip_assisted_neb(
        initial=images[0],
        final=images[-1],
        mlip_calculator_factory=lambda: "calc",
        neb_config=NEBRunConfig(n_images=3, interpolation="linear"),
        active_config=MLIPActiveNEBConfig(
            n_select=3,
            output_dir=tmp_path / "selected",
        ),
    )

    assert result.neb_result is neb_result
    assert result.selected_indices == (1, 2, 3)
    assert [item.index for item in result.selected_images] == [1, 2, 3]
    assert result.output_dir == tmp_path / "selected"
    assert result.mlip_barrier == pytest.approx(0.5)
    assert (tmp_path / "selected" / "selected_images.json").exists()


def test_run_mlip_assisted_neb_can_skip_export(monkeypatch):
    energies = [0.0, 0.1, 0.5, 0.2, 0.0]
    images = _images()
    neb = SimpleNamespace(images=images, get_energies=lambda: energies)
    neb_result = SimpleNamespace(neb=neb, barrier=0.5)

    monkeypatch.setattr(
        "NEBwalk.active.run_neb_calculation",
        lambda initial, final, calculator_factory, config: neb_result,
    )

    result = run_mlip_assisted_neb(
        initial=images[0],
        final=images[-1],
        mlip_calculator_factory=lambda: "calc",
        active_config=MLIPActiveNEBConfig(export_selected=False),
    )

    assert result.output_dir is None
    assert result.selected_indices == (1, 2, 3)
