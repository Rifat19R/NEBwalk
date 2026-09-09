"""MLIP-assisted NEB workflow utilities."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from ase.io import write

from .engine import NEBRunConfig, run_neb_calculation
from .selection import select_images, select_peak_plus_neighbors
from .uncertainty import DisagreementResult, compute_cross_model_disagreement

DISAGREEMENT_DISCLOSURE = (
    "Selection used cross-model disagreement between two independently-trained MLIPs\n"
    "as an uncertainty proxy, not a calibrated uncertainty quantification method (no\n"
    "committee/ensemble was trained). This proxy is only meaningful where both\n"
    "calculators are within their validated chemical domain; consult nebwalk's\n"
    "documented domain-failure list before trusting results outside that domain."
)


@dataclass(frozen=True)
class MLIPActiveNEBConfig:
    """Configuration for the MLIP-assisted NEB workflow layer."""

    selection_strategy: str = "peak_plus_neighbors"
    n_select: int = 3
    include_endpoints: bool = False
    export_selected: bool = True
    output_dir: str | Path = "nebwalk_mlip_round0"
    export_formats: tuple[str, ...] = ("xyz", "traj", "json")
    metadata: dict[str, Any] = field(default_factory=dict)
    secondary_calculator_factory: Callable[[], Any] | None = None
    disagreement_metric: str = "force_disagreement"


@dataclass(frozen=True)
class SelectedImage:
    """Metadata for an image selected for higher-level refinement."""

    index: int
    energy: float
    relative_energy: float
    reason: str = "peak_plus_neighbors"
    energy_disagreement: float | None = None
    force_disagreement: float | None = None


@dataclass(frozen=True)
class MLIPActiveNEBResult:
    """Result bundle for an MLIP-assisted NEB run."""

    neb_result: Any
    selected_images: tuple[SelectedImage, ...]
    selected_indices: tuple[int, ...]
    output_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mlip_barrier(self) -> float:
        return float(self.neb_result.barrier)

    @property
    def barrier(self) -> float:
        return self.mlip_barrier


def _nebwalk_version() -> str:
    try:
        return importlib_metadata.version("nebwalk")
    except importlib_metadata.PackageNotFoundError:
        return "0.7.1"


def _validate_export_formats(export_formats: Sequence[str]) -> tuple[str, ...]:
    formats = tuple(export_formats)
    supported = {"xyz", "traj", "json"}
    unsupported = sorted(set(formats) - supported)
    if unsupported:
        raise ValueError(f"unsupported export format(s): {', '.join(unsupported)}")
    return formats


def export_selected_images(
    images: Sequence[Any],
    selected: Sequence[SelectedImage],
    output_dir: str | Path,
    export_formats: Sequence[str] = ("xyz", "traj", "json"),
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export selected NEB images and selection metadata."""
    formats = _validate_export_formats(export_formats)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    selected_atoms = [images[item.index].copy() for item in selected]

    if "xyz" in formats:
        for item, atoms in zip(selected, selected_atoms):
            write(out / f"selected_image_{item.index:02d}.xyz", atoms)

    if "traj" in formats:
        write(out / "selected_images.traj", selected_atoms)

    if "json" in formats:
        payload: dict[str, Any] = {
            "schema": "nebwalk.selected_images.v1",
            "selection_strategy": selected[0].reason if selected else None,
            "selected_indices": [item.index for item in selected],
            "selected_images": [asdict(item) for item in selected],
            "notes": (
                "Selected images are intended for higher-level DFT/QE refinement."
            ),
        }
        if metadata is not None:
            payload["metadata"] = metadata
        with (out / "selected_images.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    readme = out / "README.md"
    strategy = selected[0].reason if selected else "unknown"
    readme_text = (
        "# nebwalk selected images\n\n"
        "These images were selected from an MLIP-assisted NEB path using the "
        f"`{strategy}` strategy.\n\n"
        "They are intended for DFT/QE refinement, single-point validation, or "
        "later labeling in an active-learning workflow.\n\n"
        "This folder does not contain a complete DFT workflow by itself.\n"
    )
    if selected and selected[0].reason == "uncertainty_disagreement":
        readme_text += f"\n{DISAGREEMENT_DISCLOSURE}\n"
    readme.write_text(readme_text, encoding="utf-8")
    return out


def _eligible_valid_count(
    disagreements: Sequence[DisagreementResult],
    include_endpoints: bool,
) -> int:
    start = 0 if include_endpoints else 1
    stop = len(disagreements) if include_endpoints else len(disagreements) - 1
    return sum(
        1 for result in disagreements if start <= result.index < stop and result.valid
    )


def _disagreement_by_index(
    disagreements: Sequence[DisagreementResult],
) -> dict[int, DisagreementResult]:
    return {result.index: result for result in disagreements}


def run_mlip_assisted_neb(
    initial: Any,
    final: Any,
    mlip_calculator_factory: Any,
    neb_config: NEBRunConfig | None = None,
    active_config: MLIPActiveNEBConfig | None = None,
) -> MLIPActiveNEBResult:
    """Run an MLIP-assisted NEB and export barrier-sensitive images."""
    neb_cfg = neb_config or NEBRunConfig()
    active_cfg = active_config or MLIPActiveNEBConfig()

    neb_result = run_neb_calculation(
        initial=initial,
        final=final,
        calculator_factory=mlip_calculator_factory,
        config=neb_cfg,
    )

    images = neb_result.neb.images
    energies = [float(energy) for energy in neb_result.neb.get_energies()]
    result_metadata = {
        "nebwalk_version": _nebwalk_version(),
        "stage": "mlip_assisted_neb",
        **active_cfg.metadata,
    }
    disagreements: list[DisagreementResult] | None = None
    selection_reason = active_cfg.selection_strategy

    if active_cfg.selection_strategy == "uncertainty_disagreement":
        if active_cfg.secondary_calculator_factory is None:
            raise ValueError(
                "uncertainty_disagreement strategy requires "
                "secondary_calculator_factory"
            )
        disagreements = compute_cross_model_disagreement(
            images,
            active_cfg.secondary_calculator_factory,
        )
        valid_count = _eligible_valid_count(
            disagreements,
            active_cfg.include_endpoints,
        )
        if valid_count < active_cfg.n_select:
            selected_indices = select_peak_plus_neighbors(
                energies=energies,
                n_select=active_cfg.n_select,
                include_endpoints=active_cfg.include_endpoints,
            )
            selection_reason = "peak_plus_neighbors"
            result_metadata["selection_fallback"] = (
                "insufficient_valid_disagreement_results"
            )
        else:
            selected_indices = select_images(
                energies=energies,
                strategy=active_cfg.selection_strategy,
                n_select=active_cfg.n_select,
                include_endpoints=active_cfg.include_endpoints,
                disagreements=disagreements,
                disagreement_metric=active_cfg.disagreement_metric,
            )
    else:
        selected_indices = select_images(
            energies=energies,
            strategy=active_cfg.selection_strategy,
            n_select=active_cfg.n_select,
            include_endpoints=active_cfg.include_endpoints,
        )

    reference_energy = energies[0]
    disagreement_map = _disagreement_by_index(disagreements or [])
    use_disagreement_metadata = selection_reason == "uncertainty_disagreement"
    selected = tuple(
        SelectedImage(
            index=idx,
            energy=energies[idx],
            relative_energy=energies[idx] - reference_energy,
            reason=selection_reason,
            energy_disagreement=(
                disagreement_map[idx].energy_disagreement
                if idx in disagreement_map and use_disagreement_metadata
                else None
            ),
            force_disagreement=(
                disagreement_map[idx].force_disagreement
                if idx in disagreement_map and use_disagreement_metadata
                else None
            ),
        )
        for idx in selected_indices
    )
    output_dir = None
    if active_cfg.export_selected:
        output_dir = export_selected_images(
            images=images,
            selected=selected,
            output_dir=active_cfg.output_dir,
            export_formats=active_cfg.export_formats,
            metadata=result_metadata,
        )

    return MLIPActiveNEBResult(
        neb_result=neb_result,
        selected_images=selected,
        selected_indices=tuple(selected_indices),
        output_dir=output_dir,
        metadata=result_metadata,
    )
