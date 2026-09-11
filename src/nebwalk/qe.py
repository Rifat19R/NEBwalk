"""Quantum ESPRESSO calculator helpers for nebwalk.

The functions here intentionally keep Quantum ESPRESSO optional. Importing
``nebwalk`` does not import ASE's Espresso calculator or require ``pw.x``;
callers opt in by creating a QE calculator factory.
"""

from __future__ import annotations

import random
import re
import shlex
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from os import X_OK, access
from pathlib import Path
from typing import Any

import numpy as np
from ase import units
from ase.geometry import find_mic

from .recovery import MAX_RETRIES, FailureType


def _espresso_placeholder(*args: Any, **kwargs: Any) -> Any:
    """Placeholder replaced lazily by ASE's Espresso class."""
    raise RuntimeError("ASE Espresso calculator was not loaded.")


Espresso = _espresso_placeholder
EspressoProfile = None  # set lazily alongside Espresso (ASE >= 3.23)


@dataclass(frozen=True)
class QEParams:
    """Common Quantum ESPRESSO parameters for NEB image calculations."""

    ecutwfc: float = 40.0
    ecutrho: float = 320.0
    kpts: tuple[int, int, int] = (1, 1, 1)
    koffset: tuple[int, int, int] = (0, 0, 0)
    occupations: str = "smearing"
    smearing: str = "marzari-vanderbilt"
    degauss: float = 0.02
    conv_thr: float = 1.0e-8
    mixing_beta: float = 0.3
    nspin: int = 1
    starting_magnetization: Mapping[str | int, float] | None = None
    extra_control: Mapping[str, Any] = field(default_factory=dict)
    extra_system: Mapping[str, Any] = field(default_factory=dict)
    extra_electrons: Mapping[str, Any] = field(default_factory=dict)


class QERecoveryStrategy:
    def __init__(
        self,
        seed: int = 0,
        max_displacement_A: float = 0.05,
        mic_tolerance_A: float = 0.3,
    ) -> None:
        self._rng = random.Random(seed)
        self.max_displacement_A = max_displacement_A
        self.mic_tolerance_A = mic_tolerance_A

    def classify(self, error, raw_output=None):
        text = raw_output or str(error)
        # Check for non-convergence before "JOB DONE." -- QE prints
        # "JOB DONE." on exit even when the SCF cycle never converged, so
        # checking JOB DONE. first would misclassify a real convergence
        # failure as FailureType.UNKNOWN and skip the retryable path.
        if "convergence NOT achieved" in text:
            return FailureType.CONVERGENCE_FAILURE
        if "JOB DONE." in text:
            return FailureType.UNKNOWN
        geometry_markers = (
            "S matrix not positive definite",
            "negative bond length",
            "atoms too close",
            "NaN",
        )
        if any(marker in text for marker in geometry_markers):
            return FailureType.GEOMETRY_INSTABILITY
        return FailureType.PROCESS_FAILURE

    def propose_retry(self, failure_type, attempt, atoms, calc_params):
        new_params = dict(calc_params)

        if failure_type == FailureType.CONVERGENCE_FAILURE:
            base_beta = calc_params.get("mixing_beta", 0.7)
            new_params["mixing_beta"] = base_beta * (0.5**attempt)
            if attempt >= MAX_RETRIES:
                new_params["degauss"] = calc_params.get("degauss", 0.01) + 0.01
            return new_params, atoms

        if failure_type == FailureType.GEOMETRY_INSTABILITY:
            new_atoms = atoms.copy()
            new_atoms.calc = atoms.calc
            disp = self._rng.uniform
            displacement = np.array(
                [
                    [
                        disp(-self.max_displacement_A, self.max_displacement_A)
                        for _ in range(3)
                    ]
                    for _ in range(len(new_atoms))
                ]
            )
            new_atoms.positions = new_atoms.positions + displacement
            return new_params, new_atoms

        return None

    def validate_recovered_geometry(self, recovered_atoms, target_atoms) -> bool:
        _, dist = find_mic(
            recovered_atoms.positions - target_atoms.positions,
            recovered_atoms.cell,
            recovered_atoms.pbc,
        )
        return float(np.max(dist)) <= self.mic_tolerance_A


def _read_qe_output(image_dir: Path) -> str:
    chunks: list[str] = []
    for pattern in ("*.pwo", "*.out", "*.err"):
        for path in sorted(image_dir.glob(pattern)):
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def _parse_qe_energy_forces(raw_output: str) -> tuple[float, np.ndarray]:
    """Parse final QE energy and forces when ASE output parsing fails."""
    energy_ry: float | None = None
    forces_ry_bohr: list[list[float]] = []
    in_forces = False

    for line in raw_output.splitlines():
        if "total energy" in line and "=" in line and "Ry" in line:
            match = re.search(r"=\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*Ry", line)
            if match:
                energy_ry = float(match.group(1))

        if "Forces acting on atoms" in line:
            forces_ry_bohr = []
            in_forces = True
            continue

        if in_forces and line.lstrip().startswith("atom") and "force =" in line:
            values = line.split("force =", 1)[1].split()
            if len(values) >= 3:
                forces_ry_bohr.append(
                    [float(values[0]), float(values[1]), float(values[2])]
                )
            continue

        if in_forces and forces_ry_bohr and line.strip().startswith("Total force"):
            in_forces = False

    if energy_ry is None or not forces_ry_bohr:
        raise ValueError("could not parse QE energy/forces from completed output")

    energy_ev = energy_ry * units.Ry
    forces_ev_ang = np.array(forces_ry_bohr, dtype=float) * units.Ry / units.Bohr
    return float(energy_ev), forces_ev_ang


def _attach_qe_output_capture(calc: Any, image_dir: Path) -> Any:
    fallback_results: dict[str, Any] = {}

    def completed_qe_fallback(exc: Exception) -> tuple[float, np.ndarray]:
        raw_output = _read_qe_output(image_dir)
        if not hasattr(exc, "qe_output"):
            setattr(exc, "qe_output", raw_output)
        if "JOB DONE" not in raw_output:
            raise exc
        if "convergence NOT achieved" in raw_output:
            # QE writes "JOB DONE." on exit regardless of whether the SCF
            # cycle actually converged -- never fabricate a "successful"
            # energy/forces result from a non-converged run. Re-raise so
            # run_with_recovery's classify() can route this to a real
            # convergence-failure retry instead of silently succeeding.
            raise exc
        if "energy" not in fallback_results or "forces" not in fallback_results:
            energy, forces = _parse_qe_energy_forces(raw_output)
            fallback_results["energy"] = energy
            fallback_results["forces"] = forces
            if hasattr(calc, "results"):
                calc.results.update(fallback_results)
        return fallback_results["energy"], fallback_results["forces"]

    for method_name in ("get_potential_energy", "get_forces"):
        method = getattr(calc, method_name, None)
        if method is None:
            continue

        def wrapped(*args, _method=method, _name=method_name, **kwargs):
            try:
                return _method(*args, **kwargs)
            except Exception as exc:
                energy, forces = completed_qe_fallback(exc)
                if _name == "get_potential_energy":
                    return energy
                return forces

        setattr(calc, method_name, wrapped)
    return calc


def _command_binaries(command: str) -> list[str]:
    """Return executable names worth checking from a QE command string."""
    tokens = shlex.split(command)
    if not tokens:
        return ["pw.x"]

    binaries = [tokens[0]]
    flags_with_values = {
        "-np",
        "-n",
        "--np",
        "--n",
        "-machinefile",
        "-hostfile",
        "--host",
        "--hostfile",
    }
    positionals: list[str] = []
    skip_next = False
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in flags_with_values:
            skip_next = True
            continue
        if token.startswith("-") or token.isdigit() or "=" in token:
            continue
        positionals.append(token)

    for token in reversed(positionals):
        name = Path(token).name
        if token.endswith(".x") or "." in name or "/" in token or "\\" in token:
            binaries.append(token)
            break

    unique: list[str] = []
    for binary in binaries:
        if binary not in unique:
            unique.append(binary)
    return unique


def _binary_is_available(binary: str) -> bool:
    """Check PATH binaries and explicit executable paths."""
    path = Path(binary).expanduser()
    has_path_separator = "/" in binary or "\\" in binary
    if path.is_absolute() or has_path_separator:
        return path.is_file() and access(path, X_OK)
    return shutil.which(binary) is not None


def validate_qe_setup(
    pseudo_dir: str | Path,
    pseudopotentials: Mapping[str, str],
    command: str = "pw.x",
) -> None:
    """Validate QE executable access and required pseudopotential files.

    Raises
    ------
    FileNotFoundError
        If the pseudopotential directory, a requested UPF file, or a required
        executable is unavailable.
    ValueError
        If the pseudopotential mapping is empty.
    """
    pseudo_path = Path(pseudo_dir).expanduser()
    if not pseudo_path.is_dir():
        raise FileNotFoundError(
            f"QE pseudopotential directory not found: {pseudo_path}"
        )
    if not pseudopotentials:
        raise ValueError("At least one pseudopotential must be provided.")

    missing_pseudos = [
        pseudo_name
        for pseudo_name in pseudopotentials.values()
        if not (pseudo_path / pseudo_name).is_file()
    ]
    if missing_pseudos:
        missing = ", ".join(missing_pseudos)
        raise FileNotFoundError(f"Missing QE pseudopotential file(s): {missing}")

    missing_bins = []
    for binary in _command_binaries(command):
        if not _binary_is_available(binary):
            missing_bins.append(binary)
    if missing_bins:
        missing = ", ".join(missing_bins)
        raise FileNotFoundError(f"QE executable not found or not executable: {missing}")


def _species_index(
    species: str | int,
    pseudopotentials: Mapping[str, str],
) -> int:
    """Map chemical symbol or explicit index to QE species index."""
    if isinstance(species, int):
        return species
    symbols = list(pseudopotentials)
    if species not in pseudopotentials:
        known = ", ".join(symbols)
        raise ValueError(f"Magnetization species {species!r} not in pseudos: {known}")
    return symbols.index(species) + 1


def _input_data(
    params: QEParams,
    pseudo_dir: Path,
    outdir: Path,
    pseudopotentials: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Build ASE Espresso input_data with NEB-safe defaults."""
    control: dict[str, Any] = {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "tprnfor": True,
        "tstress": False,
        "disk_io": "low",
        "pseudo_dir": str(pseudo_dir),
        "outdir": str(outdir),
    }
    control.update(dict(params.extra_control))

    system: dict[str, Any] = {
        "ecutwfc": params.ecutwfc,
        "ecutrho": params.ecutrho,
        "occupations": params.occupations,
        "nspin": params.nspin,
    }
    if params.occupations == "smearing":
        system["smearing"] = params.smearing
        system["degauss"] = params.degauss
    if params.nspin == 2 and params.starting_magnetization:
        for species, value in params.starting_magnetization.items():
            idx = _species_index(species, pseudopotentials)
            system[f"starting_magnetization({idx})"] = value
    system.update(dict(params.extra_system))

    electrons: dict[str, Any] = {
        "conv_thr": params.conv_thr,
        "mixing_beta": params.mixing_beta,
    }
    electrons.update(dict(params.extra_electrons))

    return {"control": control, "system": system, "electrons": electrons}


def make_qe_factory(
    params: QEParams,
    pseudo_dir: str | Path,
    pseudopotentials: Mapping[str, str],
    base_dir: str | Path = "neb_qe_workdir",
    command: str = "pw.x",
    recovery_strategy: Any | None = None,
) -> Callable[[], Any]:
    """Create a zero-argument factory returning independent QE calculators.

    Each factory call creates a fresh image directory, which avoids ASE/QE file
    collisions during serial and parallel NEB force evaluations.
    """
    global Espresso, EspressoProfile

    if not pseudopotentials:
        raise ValueError("At least one pseudopotential must be provided.")
    if Espresso is _espresso_placeholder:
        from ase.calculators.espresso import Espresso as ASEEspresso

        Espresso = ASEEspresso
        try:
            from ase.calculators.espresso import EspressoProfile as _EP

            EspressoProfile = _EP
        except ImportError:
            pass

    pseudo_path = Path(pseudo_dir).expanduser().resolve()
    base_path = Path(base_dir).expanduser().resolve()
    counter = {"image": 0}
    strategy = (
        recovery_strategy if recovery_strategy is not None else QERecoveryStrategy()
    )

    def factory() -> Any:
        image_idx = counter["image"]
        counter["image"] += 1

        image_dir = base_path / f"image_{image_idx:03d}"
        outdir = image_dir / "tmp"
        outdir.mkdir(parents=True, exist_ok=True)

        if EspressoProfile is not None:
            _profile = EspressoProfile(command, str(pseudo_path))
            calc = Espresso(
                profile=_profile,
                input_data=_input_data(params, pseudo_path, outdir, pseudopotentials),
                pseudopotentials=dict(pseudopotentials),
                kpts=params.kpts,
                koffset=params.koffset,
                directory=str(image_dir),
            )
        else:
            calc = Espresso(
                input_data=_input_data(params, pseudo_path, outdir, pseudopotentials),
                pseudopotentials=dict(pseudopotentials),
                kpts=params.kpts,
                koffset=params.koffset,
                directory=str(image_dir),
                label="espresso",
                command=f"{command} -in PREFIX.pwi > PREFIX.pwo",
            )
        calc.recovery_strategy = strategy
        return _attach_qe_output_capture(calc, image_dir)

    factory.recovery_strategy = strategy  # type: ignore[attr-defined]
    return factory


__all__ = ["QEParams", "QERecoveryStrategy", "make_qe_factory", "validate_qe_setup"]
