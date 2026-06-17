"""Shared high-level NEB calculation engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ase import Atoms

from .interpolate import geodesic_interpolate, idpp_interpolate, linear_interpolate
from .neb import NEB

InterpolatorName = Literal["linear", "idpp", "geodesic"]


@dataclass(frozen=True)
class NEBRunConfig:
    """Configuration for one NEB calculation."""

    n_images: int = 7
    interpolation: InterpolatorName = "idpp"
    k: float = 0.1
    k_min: float | None = None
    climb: bool = False
    climb_delay: int = 100
    n_workers: int = 1
    fmax: float = 0.05
    max_steps: int = 500
    verbose: bool = True


@dataclass(frozen=True)
class NEBRunResult:
    """Result bundle from the shared NEB engine."""

    neb: NEB
    converged: bool
    barrier: float
    reverse_barrier: float
    reaction_energy: float


def build_images(
    initial: Atoms,
    final: Atoms,
    n_images: int,
    interpolation: InterpolatorName = "idpp",
) -> list[Atoms]:
    """Build path images using selected interpolation method."""
    interpolators = {
        "linear": linear_interpolate,
        "idpp": idpp_interpolate,
        "geodesic": geodesic_interpolate,
    }
    return interpolators[interpolation](initial, final, n_images=n_images)


def attach_calculators(
    images: Sequence[Atoms],
    calculator_factory: Callable[[], Any],
) -> None:
    """Attach one fresh calculator instance to each image."""
    for image in images:
        image.calc = calculator_factory()


def run_neb_calculation(
    initial: Atoms,
    final: Atoms,
    calculator_factory: Callable[[], Any],
    config: NEBRunConfig | None = None,
    prepare_images: Callable[[list[Atoms]], None] | None = None,
    *,
    reproduce_dir: str | Path | None = None,
    calc_params: dict[str, Any] | None = None,
) -> NEBRunResult:
    """Prepare, run, and summarize one NEB calculation."""
    cfg = config or NEBRunConfig()
    images = build_images(initial, final, cfg.n_images, cfg.interpolation)
    attach_calculators(images, calculator_factory)
    if prepare_images is not None:
        prepare_images(images)

    neb = NEB(
        images,
        k=cfg.k,
        k_min=cfg.k_min,
        climb=cfg.climb,
        climb_delay=cfg.climb_delay,
        n_workers=cfg.n_workers,
    )
    converged = neb.optimize(
        fmax=cfg.fmax,
        max_steps=cfg.max_steps,
        verbose=cfg.verbose,
    )
    result = NEBRunResult(
        neb=neb,
        converged=converged,
        barrier=neb.get_barrier(),
        reverse_barrier=neb.get_reverse_barrier(),
        reaction_energy=neb.get_reaction_energy(),
    )
    if reproduce_dir is not None:
        from .reproduce import save_bundle

        save_bundle(
            result,
            initial,
            final,
            cfg,
            output_dir=reproduce_dir,
            calc_params=calc_params,
            recovery_log=getattr(result.neb, "recovery_log", None),
        )
        print(f"Reproducibility bundle saved to {reproduce_dir}")
    return result
