"""Per-material checkpoint/validation bundle for the active-learning pipeline.

A run finishing (Stage 1 NEB converged, Stage 2 QE labels came back, Stage 3
export succeeded) is not the same claim as the data being validated. This
module freezes a reviewable checkpoint once a material's run completes:
pseudopotential identity (name + SHA-256), the exact QE settings used, a
finite-value check on every energy/force, a cohesive-energy sanity check,
and a real MACE-loader read test (not just ASE's generic extxyz reader) --
written to one JSON/README pair alongside that material's dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .finetune import IsolatedAtomReference
from .label import DFTLabel
from .qe import QEParams


def check_qe_scf_converged(qe_workdir: str | Path) -> dict[str, Any]:
    """Scan a QE workdir's per-image ``.pwo`` files for SCF non-convergence.

    ASE's own Espresso reader (and nebwalk's own fallback parser in
    ``nebwalk.qe``) will happily return a final energy/forces block even
    when QE printed "convergence NOT achieved ... stopping" before "JOB
    DONE." -- QE writes "JOB DONE." on exit regardless of whether the SCF
    cycle actually converged, so "reading the output didn't raise" is not
    the same claim as "the SCF converged." This is a distinct check.
    """
    workdir = Path(qe_workdir)
    problems: list[str] = []
    checked = 0
    for image_dir in sorted(workdir.glob("image_*")):
        pwo_files = sorted(image_dir.glob("*.pwo"))
        if not pwo_files:
            continue
        checked += 1
        text = pwo_files[0].read_text(encoding="utf-8", errors="replace")
        if "convergence NOT achieved" in text:
            problems.append(
                f"{image_dir.name}: SCF did not converge (see {pwo_files[0]})"
            )
    return {
        "checked": checked,
        "problems": problems,
        "converged_cleanly": not problems,
    }


def sha256_of_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class QESettingsRecord:
    """The exact QE identity/settings a material's labels were computed with."""

    pseudopotential_file: str
    pseudopotential_sha256: str
    ecutwfc: float
    ecutrho: float
    kpts: tuple[int, int, int]
    occupations: str
    smearing: str | None
    degauss: float | None
    nspin: int
    starting_magnetization: dict[str | int, float] | None
    conv_thr: float
    mixing_beta: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def qe_settings_hash(record: QESettingsRecord) -> str:
    """Deterministic SHA-256 identity for a QE settings record.

    Two materials (or a material's bulk path vs. its isolated-atom
    reference) that used genuinely different QE settings must get different
    hashes -- this feeds ``dft_settings_hash`` in
    :mod:`nebwalk.datasets`-schema exports, which
    :func:`nebwalk.datasets.validate_dataset` uses to reject silently mixing
    incompatible DFT setups into one dataset file.
    """
    encoded = json.dumps(
        record.to_json(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qe_settings_record(
    pseudo_dir: str | Path, pseudo_file: str, params: QEParams
) -> QESettingsRecord:
    """Build a QESettingsRecord, hashing the pseudopotential file in place."""
    pseudo_path = Path(pseudo_dir) / pseudo_file
    return QESettingsRecord(
        pseudopotential_file=pseudo_file,
        pseudopotential_sha256=sha256_of_file(pseudo_path),
        ecutwfc=params.ecutwfc,
        ecutrho=params.ecutrho,
        kpts=(params.kpts[0], params.kpts[1], params.kpts[2]),
        occupations=params.occupations,
        smearing=params.smearing if params.occupations == "smearing" else None,
        degauss=params.degauss if params.occupations == "smearing" else None,
        nspin=params.nspin,
        starting_magnetization=(
            dict(params.starting_magnetization)
            if params.starting_magnetization
            else None
        ),
        conv_thr=params.conv_thr,
        mixing_beta=params.mixing_beta,
    )


def check_finite(
    labels: Sequence[DFTLabel],
    isolated_atom_references: Sequence[IsolatedAtomReference] = (),
) -> list[str]:
    """Return problems found in labeled energies/forces; empty = all finite."""
    problems: list[str] = []
    for label in labels:
        if not np.isfinite(label.energy_eV):
            problems.append(f"image {label.index}: non-finite energy")
        if not np.all(np.isfinite(label.forces_eV_A)):
            problems.append(f"image {label.index}: non-finite forces")
    for ref in isolated_atom_references:
        if not np.isfinite(ref.energy_eV):
            problems.append(f"isolated atom {ref.symbol}: non-finite energy")
        if not np.all(np.isfinite(ref.forces_eV_A)):
            problems.append(f"isolated atom {ref.symbol}: non-finite forces")
    return problems


def cohesive_energy_check(
    reference_energy_eV: float,
    n_atoms: int,
    isolated_atom_reference: IsolatedAtomReference,
) -> dict[str, Any]:
    """Cohesive energy per atom for the labeling run's relaxed reference config.

    This is a sanity check, not a pass/fail gate -- true cohesive energies
    vary widely by element and structure. It only flags clearly-broken
    results (unbound, or implausibly large in magnitude), so an obviously
    wrong setup doesn't silently enter the dataset.
    """
    cohesive_per_atom = (
        reference_energy_eV / n_atoms - isolated_atom_reference.energy_eV
    )
    flags: list[str] = []
    if cohesive_per_atom >= 0:
        flags.append(
            "cohesive energy is not negative (unbound) -- likely a setup error"
        )
    if cohesive_per_atom < -20.0:
        flags.append("cohesive energy magnitude implausibly large (< -20 eV/atom)")
    return {"cohesive_energy_eV_per_atom": cohesive_per_atom, "flags": flags}


def mace_loader_read_test(extxyz_path: str | Path) -> dict[str, Any]:
    """Confirm MACE's own extxyz parser -- not just ASE's -- accepts the file.

    Reads with REF_energy/REF_forces (mace.data.utils.DefaultKeys), matching
    what :func:`nebwalk.finetune.export_mace_training_set` writes. Raises
    RuntimeError if mace-torch is not installed or the file is rejected --
    this check is meant to fail loudly, not be skipped silently.
    """
    try:
        from mace.data.utils import KeySpecification, load_from_xyz
    except ImportError as exc:
        raise RuntimeError(
            "mace-torch is required for mace_loader_read_test(); "
            'install with pip install "nebwalk[mace]"'
        ) from exc

    key_spec = KeySpecification.from_defaults()
    _atomic_energies, configs = load_from_xyz(
        str(extxyz_path), key_specification=key_spec
    )

    return {
        "ok": True,
        "n_configs_loaded": len(configs),
        "config_types": sorted({c.config_type for c in configs}),
    }


def nebwalk_dataset_validation_check(training_set_path: str | Path) -> dict[str, Any]:
    """Confirm the exported file passes :mod:`nebwalk.datasets`'s real validator.

    :mod:`nebwalk.campaign` and :mod:`nebwalk.labeling` are built entirely
    around that module's schema (structure-hash identity, per-path_id
    consistency, duplicate/settings-mixing checks) -- passing this module's
    own checks above is not the same claim as being accepted by the
    validator the rest of the active-learning pipeline actually relies on,
    so this is a required, independent gate rather than a duplicate of them.
    """
    from .datasets import DatasetValidationError, load_dataset, summarize_dataset

    try:
        frames = load_dataset(training_set_path)
        summary = summarize_dataset(frames)
    except (DatasetValidationError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "error": None,
        "n_configurations": summary.n_configurations,
        "elements": list(summary.elements),
        "path_ids": list(summary.path_ids),
        "dft_settings_hashes": list(summary.dft_settings_hashes),
    }


@dataclass
class MaterialValidationSummary:
    """One material's frozen validation checkpoint."""

    material: str
    symbol: str
    n_labeled: int
    n_failed: int
    qe_settings: QESettingsRecord
    finite_check_problems: list[str]
    cohesive_energy_eV_per_atom: float
    cohesive_energy_flags: list[str]
    mace_loader_ok: bool
    mace_loader_n_configs: int
    mace_loader_config_types: list[str]
    scf_converged_cleanly: bool
    scf_problems: list[str]
    nebwalk_dataset_ok: bool
    nebwalk_dataset_error: str | None
    passed: bool

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["qe_settings"] = self.qe_settings.to_json()
        return d


def build_material_validation_summary(
    material: str,
    symbol: str,
    labels: Sequence[DFTLabel],
    failed_indices: Sequence[int],
    reference_label: DFTLabel,
    n_atoms: int,
    isolated_atom_reference: IsolatedAtomReference,
    pseudo_dir: str | Path,
    pseudo_file: str,
    params: QEParams,
    training_set_path: str | Path,
    output_dir: str | Path,
    qe_workdirs: Sequence[str | Path] = (),
) -> MaterialValidationSummary:
    """Run every checkpoint for one material and freeze the result.

    Writes ``validation_summary.json`` and ``VALIDATION_SUMMARY.md`` into
    ``output_dir``. ``passed`` is False if any finite-value problem,
    cohesive-energy flag, or SCF non-convergence was found, or the MACE
    loader rejected the file -- a run "completing" without error is not the
    same as this passing. Pass every QE workdir this material touched
    (labeling + isolated-atom) in ``qe_workdirs`` to check SCF convergence;
    an empty sequence skips that check (backward compatible, but a material
    with no workdir checked is not the same as one confirmed converged).
    """
    qe_settings = qe_settings_record(pseudo_dir, pseudo_file, params)
    finite_problems = check_finite(labels, [isolated_atom_reference])
    cohesive = cohesive_energy_check(
        reference_label.energy_eV, n_atoms, isolated_atom_reference
    )
    loader_result = mace_loader_read_test(training_set_path)
    dataset_check = nebwalk_dataset_validation_check(training_set_path)

    scf_problems: list[str] = []
    for workdir in qe_workdirs:
        scf_problems.extend(check_qe_scf_converged(workdir)["problems"])

    passed = (
        not finite_problems
        and not cohesive["flags"]
        and loader_result["ok"]
        and not scf_problems
        and dataset_check["ok"]
    )

    summary = MaterialValidationSummary(
        material=material,
        symbol=symbol,
        n_labeled=len(labels),
        n_failed=len(failed_indices),
        qe_settings=qe_settings,
        finite_check_problems=finite_problems,
        cohesive_energy_eV_per_atom=cohesive["cohesive_energy_eV_per_atom"],
        cohesive_energy_flags=cohesive["flags"],
        mace_loader_ok=loader_result["ok"],
        mace_loader_n_configs=loader_result["n_configs_loaded"],
        mace_loader_config_types=loader_result["config_types"],
        scf_converged_cleanly=not scf_problems,
        scf_problems=scf_problems,
        nebwalk_dataset_ok=dataset_check["ok"],
        nebwalk_dataset_error=dataset_check["error"],
        passed=passed,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_json(), handle, indent=2)
        handle.write("\n")

    (out / "VALIDATION_SUMMARY.md").write_text(
        _render_summary_markdown(summary), encoding="utf-8"
    )

    return summary


def _render_summary_markdown(summary: MaterialValidationSummary) -> str:
    qe = summary.qe_settings
    status = "PASSED" if summary.passed else "FAILED"
    smearing_note = f" ({qe.smearing}, degauss={qe.degauss})" if qe.smearing else ""
    magnetic_note = (
        f", starting_magnetization: {qe.starting_magnetization}"
        if qe.starting_magnetization
        else ""
    )
    cohesive_note = (
        f" -- FLAGGED: {'; '.join(summary.cohesive_energy_flags)}"
        if summary.cohesive_energy_flags
        else ""
    )
    finite_note = (
        "OK"
        if not summary.finite_check_problems
        else "FAILED: " + "; ".join(summary.finite_check_problems)
    )
    lines = [
        f"# {summary.material} ({summary.symbol}) validation summary: {status}",
        "",
        f"- Labeled images: {summary.n_labeled} (failed: {summary.n_failed})",
        f"- Pseudopotential: {qe.pseudopotential_file}",
        f"  SHA-256: {qe.pseudopotential_sha256}",
        f"- ecutwfc/ecutrho: {qe.ecutwfc}/{qe.ecutrho} Ry",
        f"- kpts: {qe.kpts}, occupations: {qe.occupations}{smearing_note}",
        f"- nspin: {qe.nspin}{magnetic_note}",
        f"- Cohesive energy: {summary.cohesive_energy_eV_per_atom:.4f} eV/atom"
        f"{cohesive_note}",
        f"- Finite-value check: {finite_note}",
        f"- MACE loader read test: {'OK' if summary.mace_loader_ok else 'FAILED'} "
        f"({summary.mace_loader_n_configs} configs loaded, "
        f"config_types={summary.mace_loader_config_types})",
        "- QE SCF convergence: "
        + (
            "OK"
            if summary.scf_converged_cleanly
            else "FAILED: " + "; ".join(summary.scf_problems)
        ),
        "- nebwalk.datasets schema validation (real load_dataset() call): "
        + (
            "OK"
            if summary.nebwalk_dataset_ok
            else f"FAILED: {summary.nebwalk_dataset_error}"
        ),
    ]
    return "\n".join(lines) + "\n"


def write_manifest(
    summaries: Sequence[MaterialValidationSummary],
    dataset_paths: dict[str, str | Path],
    output_path: str | Path,
) -> Path:
    """Write a top-level index over per-material datasets.

    This is a manifest, not a merged dataset -- each material's own
    directory (and its own validation_summary.json) remains the source of
    truth. See :func:`nebwalk.finetune.combine_training_sets` for producing
    an actual combined training file when one is needed for
    ``mace_run_train``.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        {
            "material": s.material,
            "symbol": s.symbol,
            "dataset_path": str(dataset_paths[s.material]),
            "n_labeled": s.n_labeled,
            "n_failed": s.n_failed,
            "passed": s.passed,
        }
        for s in summaries
    ]
    payload = {
        "schema": "nebwalk.dataset_manifest.v1",
        "materials": entries,
        "n_materials": len(entries),
        "n_passed": sum(1 for e in entries if e["passed"]),
    }
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return out


__all__ = [
    "sha256_of_file",
    "check_qe_scf_converged",
    "QESettingsRecord",
    "qe_settings_record",
    "qe_settings_hash",
    "check_finite",
    "cohesive_energy_check",
    "mace_loader_read_test",
    "nebwalk_dataset_validation_check",
    "MaterialValidationSummary",
    "build_material_validation_summary",
    "write_manifest",
]
