"""Reproducibility bundle support for NEBwalk calculations."""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from .engine import NEBRunConfig, NEBRunResult
from .recovery import RecoveryAttempt

__all__ = ["ReproBundle", "save_bundle"]


@dataclass(frozen=True)
class ReproBundle:
    """Immutable record of one NEBwalk reproducibility bundle."""

    output_dir: Path
    tarball: Path | None
    manifest: dict[str, str]
    nebwalk_version: str
    timestamp: str


def save_bundle(
    result: NEBRunResult,
    initial: Atoms,
    final: Atoms,
    config: NEBRunConfig,
    *,
    output_dir: str | Path = "nebwalk_repro",
    calc_params: dict[str, Any] | None = None,
    compress: bool = True,
    include_env: bool = True,
    recovery_log: list[RecoveryAttempt] | None = None,
) -> ReproBundle:
    """Write a self-contained reproducibility bundle for a completed NEB run."""
    from . import __version__

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "")
    )

    _write_structures(initial, final, result.neb, out)
    _write_config_json(config, out, __version__)
    _write_calc_params_json(calc_params, out, __version__)
    _write_results_json(
        result,
        result.neb,
        out,
        __version__,
        timestamp,
        recovery_log=recovery_log,
    )
    _write_convergence_history(result.neb, out)

    env_text = _capture_env(include_env)
    if env_text is not None:
        (out / "environment.txt").write_text(env_text, encoding="utf-8")

    _write_rerun_template(out, __version__, timestamp)
    manifest = _write_manifest(out, __version__, timestamp)
    tarball = _make_tarball(out) if compress else None

    return ReproBundle(
        output_dir=out,
        tarball=tarball,
        manifest=manifest,
        nebwalk_version=__version__,
        timestamp=timestamp,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_env(include_env: bool) -> str | None:
    if not include_env:
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout or "# environment capture returned no output\n"
    except Exception:
        return "# environment capture failed\n"


def _write_structures(initial: Atoms, final: Atoms, neb: Any, output_dir: Path) -> None:
    write(output_dir / "initial.xyz", initial, format="extxyz")
    write(output_dir / "final.xyz", final, format="extxyz")
    write(output_dir / "neb_path.xyz", neb.images, format="extxyz")


def _write_config_json(
    config: NEBRunConfig,
    output_dir: Path,
    version: str,
) -> None:
    payload = {
        "schema": "nebwalk.neb_config.v1",
        "nebwalk_version": version,
        **asdict(config),
    }
    _write_json(output_dir / "neb_config.json", payload)


def _write_calc_params_json(
    calc_params: dict[str, Any] | None,
    output_dir: Path,
    version: str,
) -> None:
    payload = {
        "schema": "nebwalk.calc_params.v1",
        "nebwalk_version": version,
        "params": calc_params or {},
    }
    _write_json(output_dir / "calc_params.json", payload)


def _write_results_json(
    result: NEBRunResult,
    neb: Any,
    output_dir: Path,
    version: str,
    timestamp: str,
    recovery_log: list[RecoveryAttempt] | None = None,
) -> None:
    energies = [float(energy) for energy in neb.get_energies()]
    reference = energies[0] if energies else 0.0
    payload = {
        "schema": "nebwalk.results.v1",
        "nebwalk_version": version,
        "timestamp_utc": timestamp,
        "converged": bool(result.converged),
        "forward_barrier_eV": float(result.barrier),
        "reverse_barrier_eV": float(result.reverse_barrier),
        "reaction_energy_eV": float(result.reaction_energy),
        "image_energies_eV": energies,
        "image_energies_relative_eV": [
            float(energy - reference) for energy in energies
        ],
    }
    if recovery_log:
        payload["recovery_log"] = [attempt.to_json() for attempt in recovery_log]
    _write_json(output_dir / "results.json", payload)


def _write_convergence_history(neb: Any, output_dir: Path) -> None:
    history = []
    for idx, item in enumerate(getattr(neb, "history", [])):
        energies = item.get("energies", [])
        history.append(
            {
                "step": int(item.get("step", idx)),
                "fmax": float(item.get("fmax", 0.0)),
                "energies": [float(energy) for energy in energies],
            }
        )
    _write_json(output_dir / "convergence_history.json", history)


def _write_rerun_template(output_dir: Path, version: str, timestamp: str) -> None:
    template = '''#!/usr/bin/env python3
"""
NEBwalk reproducibility rerun template.
Generated by NEBwalk {version} on {timestamp}.

Instructions:
1. Install the same NEBwalk version:
       pip install NEBwalk=={version}
2. Fill in your calculator factory below (marked TODO).
3. Run:  python rerun_template.py
4. Compare the printed barrier with original_barrier in results.json.
"""

from ase.io import read
from NEBwalk import run_neb_calculation, NEBRunConfig
import json, pathlib

HERE = pathlib.Path(__file__).parent


# TODO: replace this with your actual calculator factory
def calculator_factory():
    raise NotImplementedError(
        "Edit this function to return a fresh calculator instance.\\n"
        "Original calc_params are recorded in calc_params.json."
    )


def main():
    initial = read(HERE / "initial.xyz")
    final = read(HERE / "final.xyz")

    with open(HERE / "neb_config.json") as fh:
        cfg_dict = json.load(fh)

    config = NEBRunConfig(
        n_images=cfg_dict["n_images"],
        interpolation=cfg_dict["interpolation"],
        k=cfg_dict["k"],
        k_min=cfg_dict["k_min"],
        climb=cfg_dict["climb"],
        climb_delay=cfg_dict["climb_delay"],
        fmax=cfg_dict["fmax"],
        max_steps=cfg_dict["max_steps"],
    )

    result = run_neb_calculation(initial, final, calculator_factory, config=config)

    with open(HERE / "results.json") as fh:
        original = json.load(fh)

    delta = abs(result.barrier - original["forward_barrier_eV"])
    print(f"Original barrier : {original['forward_barrier_eV']:.4f} eV")
    print(f"Recomputed barrier: {result.barrier:.4f} eV")
    print(f"Difference        : {delta:.4f} eV")
    if delta > 0.005:
        print("WARNING: barrier differs by > 5 meV - check environment or calc_params.")
    else:
        print("PASS: barriers agree within 5 meV.")


if __name__ == "__main__":
    main()
'''
    template = template.replace("{version}", version)
    template = template.replace("{timestamp}", timestamp)
    (output_dir / "rerun_template.py").write_text(template, encoding="utf-8")


def _write_manifest(output_dir: Path, version: str, timestamp: str) -> dict[str, str]:
    filenames = [
        "initial.xyz",
        "final.xyz",
        "neb_path.xyz",
        "neb_config.json",
        "calc_params.json",
        "results.json",
        "convergence_history.json",
        "rerun_template.py",
    ]
    manifest = {name: _sha256(output_dir / name) for name in filenames}
    payload = {
        "schema": "nebwalk.manifest.v1",
        "nebwalk_version": version,
        "timestamp_utc": timestamp,
        "files": manifest,
    }
    _write_json(output_dir / "manifest.json", payload)
    return manifest


def _make_tarball(output_dir: Path) -> Path:
    tarball = output_dir.with_suffix(".tar.gz")
    if tarball.exists():
        tarball.unlink()
    subprocess.run(
        [sys.executable, "-m", "tarfile", "-c", str(tarball), output_dir.name],
        cwd=output_dir.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return tarball


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
