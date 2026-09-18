"""Resumable closed-loop active learning for NEB pathways."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms
from ase.io import read, write

from .campaign_state import (
    CampaignLock,
    CampaignStage,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
)
from .candidate_selection import (
    CandidateInput,
    CandidateSelectionConfig,
    select_active_learning_candidates,
)
from .datasets import (
    DatasetArtifact,
    compute_dataset_checksum,
    compute_structure_hash,
    load_dataset,
    merge_datasets,
    split_dataset_by_group,
    summarize_dataset,
    write_dataset,
)
from .engine import NEBRunConfig, run_neb_calculation
from .labeling import LabelingResult, ReferenceLabeler
from .mlip import (
    CommitteeEvaluator,
    FineTuningBackend,
    FineTuningConfig,
    ModelRegistry,
    ModelRegistryEntry,
    ModelRegistryError,
    TrainedModelArtifact,
    TrainingRun,
)


@dataclass(frozen=True)
class StoppingCriteriaConfig:
    """Objective stopping thresholds; missing configured metrics never pass."""

    maximum_iterations: int = 10
    maximum_force_uncertainty: float | None = None
    maximum_relative_energy_uncertainty: float | None = None
    maximum_committee_barrier_std: float | None = None
    maximum_selected_dft_energy_error: float | None = None
    maximum_selected_dft_force_error: float | None = None
    minimum_new_configurations: int = 1
    minimum_metric_improvement: float | None = None
    consecutive_successful_iterations: int = 2

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be >= 1")
        if self.minimum_new_configurations < 0:
            raise ValueError("minimum_new_configurations must be >= 0")
        if self.consecutive_successful_iterations < 1:
            raise ValueError("consecutive_successful_iterations must be >= 1")


@dataclass(frozen=True)
class StoppingDecision:
    """Auditable campaign stop/continue decision."""

    stop: bool
    reason: str
    metrics: Mapping[str, float]
    satisfied: Mapping[str, bool]


def evaluate_stopping_criteria(
    config: StoppingCriteriaConfig,
    metrics: Mapping[str, float],
    *,
    iteration: int,
    previous_validation_metric: float | None = None,
    current_validation_metric: float | None = None,
    consecutive_successes: int = 0,
) -> StoppingDecision:
    """Evaluate configured thresholds without treating missing data as success."""
    thresholds = {
        "max_force_uncertainty": config.maximum_force_uncertainty,
        "max_relative_energy_uncertainty": config.maximum_relative_energy_uncertainty,
        "committee_barrier_std": config.maximum_committee_barrier_std,
        "selected_dft_energy_error": config.maximum_selected_dft_energy_error,
        "selected_dft_force_error": config.maximum_selected_dft_force_error,
    }
    satisfied: dict[str, bool] = {}
    for metric, threshold in thresholds.items():
        if threshold is not None:
            value = metrics.get(metric)
            satisfied[metric] = (
                value is not None and math.isfinite(value) and value <= threshold
            )
    new_count = metrics.get("new_configurations")
    satisfied["minimum_new_configurations"] = (
        new_count is not None and new_count >= config.minimum_new_configurations
    )
    if config.minimum_metric_improvement is not None:
        satisfied["minimum_metric_improvement"] = (
            previous_validation_metric is not None
            and current_validation_metric is not None
            and previous_validation_metric - current_validation_metric
            >= config.minimum_metric_improvement
        )
    if iteration >= config.maximum_iterations:
        return StoppingDecision(True, "maximum_iterations", dict(metrics), satisfied)
    all_satisfied = bool(satisfied) and all(satisfied.values())
    next_consecutive = consecutive_successes + 1 if all_satisfied else 0
    if all_satisfied and next_consecutive >= config.consecutive_successful_iterations:
        return StoppingDecision(
            True, "objective_thresholds_satisfied", dict(metrics), satisfied
        )
    reason = (
        "awaiting_consecutive_success" if all_satisfied else "criteria_not_satisfied"
    )
    return StoppingDecision(False, reason, dict(metrics), satisfied)


@dataclass(frozen=True)
class CampaignPath:
    """Stable path identifier and relaxed endpoint structures."""

    path_id: str
    initial: Atoms
    final: Atoms


@dataclass(frozen=True)
class CampaignPathResult:
    """One MLIP NEB path and optional committee diagnostics."""

    path_id: str
    images: tuple[Atoms, ...]
    energies: tuple[float, ...]
    barrier: float
    candidate_inputs: tuple[CandidateInput, ...] = ()
    committee_barrier_std: float | None = None


class CampaignPathRunner(Protocol):
    """Injected MLIP/committee NEB runner used by the campaign."""

    def __call__(
        self,
        paths: Sequence[CampaignPath],
        active_model: ModelRegistryEntry | None,
        output_dir: Path,
    ) -> Sequence[CampaignPathResult]: ...


@dataclass(frozen=True)
class MACEPathRunner:
    """Production path runner that imports MACE only when a path is executed."""

    foundation_model: str
    neb_config: NEBRunConfig = field(default_factory=NEBRunConfig)
    device: str = "cpu"
    default_dtype: str = "float64"
    committee_factory_provider: (
        Callable[[ModelRegistryEntry | None], Sequence[Callable[[], Any]]] | None
    ) = None
    minimum_valid_members: int = 2

    def _calculator_factory(
        self, active_model: ModelRegistryEntry | None
    ) -> Callable[[], Any]:
        model = str(active_model.path) if active_model else self.foundation_model

        def factory() -> Any:
            try:
                from mace.calculators import MACECalculator
            except ImportError as exc:
                raise RuntimeError(
                    "MACE path execution requires the optional 'mace' extra"
                ) from exc
            return MACECalculator(
                model_paths=model,
                device=self.device,
                default_dtype=self.default_dtype,
            )

        return factory

    def __call__(
        self,
        paths: Sequence[CampaignPath],
        active_model: ModelRegistryEntry | None,
        output_dir: Path,
    ) -> Sequence[CampaignPathResult]:
        results = []
        calculator_factory = self._calculator_factory(active_model)
        committee = None
        if self.committee_factory_provider is not None:
            factories = self.committee_factory_provider(active_model)
            if factories:
                committee = CommitteeEvaluator(
                    factories, minimum_valid_members=self.minimum_valid_members
                )
        for path in paths:
            neb_result = run_neb_calculation(
                path.initial,
                path.final,
                calculator_factory,
                config=self.neb_config,
                reproduce_dir=output_dir / path.path_id / "reproducibility",
                calc_params={
                    "backend": "MACE",
                    "model": (
                        str(active_model.path)
                        if active_model is not None
                        else self.foundation_model
                    ),
                    "device": self.device,
                    "default_dtype": self.default_dtype,
                },
            )
            images = tuple(image.copy() for image in neb_result.neb.images)
            energies = tuple(neb_result.neb.get_energies())
            candidate_inputs: tuple[CandidateInput, ...] = ()
            barrier_std = None
            if committee is not None:
                path_diagnostics = committee.evaluate_path(neb_result.neb.images)
                candidate_inputs = tuple(
                    CandidateInput(item.image_index, committee=item)
                    for item in path_diagnostics.images
                )
                barrier_std = path_diagnostics.barrier_std
            results.append(
                CampaignPathResult(
                    path.path_id,
                    images,
                    energies,
                    neb_result.barrier,
                    candidate_inputs,
                    barrier_std,
                )
            )
        return results


@dataclass(frozen=True)
class ActiveLearningConfig:
    """Compact serializable configuration for a complete campaign."""

    campaign_id: str
    campaign_dir: Path
    fine_tuning: FineTuningConfig
    candidate_selection: CandidateSelectionConfig = field(
        default_factory=CandidateSelectionConfig
    )
    stopping: StoppingCriteriaConfig = field(default_factory=StoppingCriteriaConfig)
    committee_size: int = 3
    committee_seeds: tuple[int, ...] = (11, 29, 47)
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 0
    model_selection_metric: str = "force_component_rmse"
    full_final_qe_neb: bool = False

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id is required")
        if self.committee_size < 1:
            raise ValueError("committee_size must be >= 1")
        if len(self.committee_seeds) < self.committee_size:
            raise ValueError("committee_seeds must cover committee_size")


@dataclass(frozen=True)
class ActiveLearningResult:
    """Current durable outcome of a campaign run."""

    state: CampaignState
    dataset_path: Path | None
    active_model: ModelRegistryEntry | None
    campaign_dir: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Atoms):
        raise TypeError("Atoms must be persisted as extxyz, not JSON")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ActiveLearningCampaign:
    """Small resumable orchestrator around injected NEB, QE, and training layers."""

    TERMINAL = {
        CampaignStage.CONVERGED,
        CampaignStage.MAX_ITERATIONS,
    }

    def __init__(
        self,
        config: ActiveLearningConfig,
        trainer: FineTuningBackend,
        labeler: ReferenceLabeler,
        *,
        path_runner: CampaignPathRunner | None = None,
        final_neb_validator: Callable[[Sequence[CampaignPath], Path], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.config = config
        self.trainer = trainer
        self.labeler = labeler
        self.path_runner = path_runner
        self.final_neb_validator = final_neb_validator
        self.root = Path(config.campaign_dir)
        self.store = CampaignStateStore(self.root)
        self.registry = ModelRegistry(self.root / "models" / "registry.json")

    def _initialize(self) -> CampaignState:
        state = self.store.initialize(self.config.campaign_id)
        config_path = self.root / "campaign_config.json"
        serialized = _json_safe(asdict(self.config))
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != serialized:
                raise CampaignStateError(
                    "campaign configuration differs from persisted run"
                )
        else:
            _write_json(config_path, serialized)
            state = self.store.record_artifact(state, config_path)
        for directory in ("dataset", "models", "reports", "final_validation", "paths"):
            (self.root / directory).mkdir(exist_ok=True)
        return state

    def _save_paths(self, paths: Sequence[CampaignPath]) -> None:
        if not paths:
            raise ValueError("campaign requires at least one path")
        seen: set[str] = set()
        manifest = []
        for path in paths:
            if not path.path_id or path.path_id in seen:
                raise ValueError("campaign path IDs must be unique and nonempty")
            seen.add(path.path_id)
            endpoint_path = self.root / "paths" / f"{path.path_id}.extxyz"
            if endpoint_path.exists():
                loaded = read(endpoint_path, index=":", format="extxyz")
                frames = [loaded] if isinstance(loaded, Atoms) else list(loaded)
                if len(frames) != 2:
                    raise CampaignStateError("persisted endpoint file is malformed")
                if any(
                    compute_structure_hash(a) != compute_structure_hash(b)
                    for a, b in zip(frames, (path.initial, path.final))
                ):
                    raise CampaignStateError(
                        "provided paths differ from persisted endpoints"
                    )
            else:
                write(endpoint_path, [path.initial, path.final], format="extxyz")
            manifest.append({"path_id": path.path_id, "file": endpoint_path.name})
        _write_json(self.root / "paths" / "manifest.json", {"paths": manifest})

    def _load_paths(self) -> list[CampaignPath]:
        manifest = json.loads(
            (self.root / "paths" / "manifest.json").read_text(encoding="utf-8")
        )
        paths = []
        for item in manifest["paths"]:
            loaded = read(
                self.root / "paths" / item["file"], index=":", format="extxyz"
            )
            frames = [loaded] if isinstance(loaded, Atoms) else list(loaded)
            paths.append(CampaignPath(item["path_id"], frames[0], frames[1]))
        return paths

    def _iteration_dir(self, state: CampaignState) -> Path:
        return self.root / f"iteration_{state.iteration:03d}"

    @staticmethod
    def _clear_stale_attempt(output_dir: Path) -> None:
        """Remove a directory left by a previously failed train/evaluate call.

        ``MaceTrainer.train``/``evaluate`` refuse to run into a non-empty
        directory (to avoid silently mixing two runs' artifacts). On resume
        after a recorded failure, the completion marker (model_manifest.json
        / the expected result file) is absent by definition, so any leftover
        directory here is from the failed attempt, not a completed run --
        safe to clear so the retry can proceed instead of raising
        FileExistsError and permanently wedging the campaign.
        """
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def _active_model(self, state: CampaignState) -> ModelRegistryEntry | None:
        return (
            self.registry.get_model(state.active_model_id)
            if state.active_model_id is not None
            else None
        )

    def _run_and_select(
        self, state: CampaignState, paths: Sequence[CampaignPath], bootstrap: bool
    ) -> tuple[Path, dict[str, float]]:
        if self.path_runner is None:
            raise RuntimeError("campaign requires an injected path_runner")
        iteration_dir = self._iteration_dir(state)
        run_dir = iteration_dir / "neb"
        run_dir.mkdir(parents=True, exist_ok=True)
        results = list(self.path_runner(paths, self._active_model(state), run_dir))
        if {result.path_id for result in results} != {path.path_id for path in paths}:
            raise RuntimeError("path_runner results do not match configured paths")
        selected_atoms: list[Atoms] = []
        selected_records: list[dict[str, Any]] = []
        force_uncertainties: list[float] = []
        energy_uncertainties: list[float] = []
        barrier_stds: list[float] = []
        profile_manifest: list[dict[str, Any]] = []
        for result in sorted(results, key=lambda item: item.path_id):
            if len(result.images) != len(result.energies):
                raise RuntimeError("path image and energy counts differ")
            path_file = run_dir / f"{result.path_id}.extxyz"
            images = [image.copy() for image in result.images]
            for image in images:
                image.calc = None
            write(path_file, images, format="extxyz")
            selection = select_active_learning_candidates(
                list(result.energies),
                list(result.candidate_inputs),
                self.config.candidate_selection,
                bootstrap=bootstrap,
            )
            for candidate in selection.selected:
                atoms = result.images[candidate.image_index].copy()
                atoms.calc = None
                atoms.info.update(
                    {
                        "campaign_id": self.config.campaign_id,
                        "iteration": state.iteration,
                        "path_id": result.path_id,
                        "image_index": candidate.image_index,
                        "selection_reason": "+".join(candidate.reasons),
                        "MLIP_energy": result.energies[candidate.image_index],
                    }
                )
                try:
                    mlip_forces = np.asarray(
                        result.images[candidate.image_index].get_forces(), dtype=float
                    )
                except (RuntimeError, AttributeError):
                    pass
                else:
                    if (
                        mlip_forces.shape == (len(atoms), 3)
                        and np.isfinite(mlip_forces).all()
                    ):
                        atoms.arrays["MLIP_forces"] = mlip_forces
                selected_atoms.append(atoms)
                selected_records.append(
                    {
                        "path_id": result.path_id,
                        "image_index": candidate.image_index,
                        "reasons": list(candidate.reasons),
                        "score": candidate.score,
                    }
                )
            for item in result.candidate_inputs:
                if item.committee is not None:
                    force_uncertainties.append(item.committee.max_atom_force_std)
                    if item.committee.relative_energy_std is not None:
                        energy_uncertainties.append(item.committee.relative_energy_std)
            if result.committee_barrier_std is not None:
                barrier_stds.append(result.committee_barrier_std)
            profile_manifest.append(
                {
                    "path_id": result.path_id,
                    "energies_eV": list(result.energies),
                    "barrier_eV": result.barrier,
                    "path_file": str(path_file.relative_to(iteration_dir)),
                    "selection_fallback": selection.fallback_reason,
                    "diagnostics": list(selection.diagnostics),
                }
            )

        unique: dict[str, Atoms] = {}
        existing_hashes: set[str] = set()
        master = self._latest_dataset_path()
        if master is not None:
            existing_hashes = {
                str(frame.info["structure_hash"]) for frame in load_dataset(master)
            }
        filtered_records = []
        for atoms, record in zip(selected_atoms, selected_records):
            structure_hash = compute_structure_hash(atoms)
            if structure_hash in existing_hashes or structure_hash in unique:
                continue
            unique[structure_hash] = atoms
            filtered_records.append(record)
        candidates_path = iteration_dir / "selected_candidates.extxyz"
        if unique:
            write(candidates_path, list(unique.values()), format="extxyz")
        else:
            candidates_path.write_text("", encoding="utf-8")
        _write_json(
            iteration_dir / "selected_candidates.json",
            {
                "candidates": filtered_records,
                "profiles": profile_manifest,
                "new_unique_count": len(unique),
            },
        )
        metrics = {"new_configurations": float(len(unique))}
        if force_uncertainties:
            metrics["max_force_uncertainty"] = max(force_uncertainties)
        if energy_uncertainties:
            metrics["max_relative_energy_uncertainty"] = max(energy_uncertainties)
        if barrier_stds:
            metrics["committee_barrier_std"] = max(barrier_stds)
        _write_json(iteration_dir / "iteration_metrics.json", metrics)
        return candidates_path, metrics

    def _latest_dataset_path(self) -> Path | None:
        datasets = sorted((self.root / "dataset").glob("dataset_v*.extxyz"))
        return datasets[-1] if datasets else None

    def _candidate_structures(self, state: CampaignState) -> list[Atoms]:
        path = self._iteration_dir(state) / "selected_candidates.extxyz"
        if not path.exists() or path.stat().st_size == 0:
            return []
        loaded = read(path, index=":", format="extxyz")
        return [loaded] if isinstance(loaded, Atoms) else list(loaded)

    def _update_dataset(self, state: CampaignState, result: LabelingResult) -> Path:
        if result.dataset is None:
            raise RuntimeError(
                "labeling produced no successful reference configurations"
            )
        new_frames = load_dataset(result.dataset.path)
        energy_errors: list[float] = []
        force_errors: list[float] = []
        grouped: dict[str, list[Atoms]] = {}
        for frame in new_frames:
            grouped.setdefault(str(frame.info["path_id"]), []).append(frame)
            if "MLIP_forces" in frame.arrays:
                difference = np.asarray(frame.arrays["MLIP_forces"]) - np.asarray(
                    frame.arrays["REF_forces"]
                )
                force_errors.append(float(np.max(np.linalg.norm(difference, axis=1))))
        for frames in grouped.values():
            comparable = [frame for frame in frames if "MLIP_energy" in frame.info]
            if not comparable:
                continue
            reference = min(
                comparable, key=lambda frame: int(frame.info["image_index"])
            )
            ref_dft = float(reference.info["REF_energy"])
            ref_mlip = float(reference.info["MLIP_energy"])
            for frame in comparable:
                dft_relative = float(frame.info["REF_energy"]) - ref_dft
                mlip_relative = float(frame.info["MLIP_energy"]) - ref_mlip
                energy_errors.append(abs(dft_relative - mlip_relative))
        metrics_path = self._iteration_dir(state) / "iteration_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        if energy_errors:
            metrics["selected_dft_energy_error"] = max(energy_errors)
        if force_errors:
            metrics["selected_dft_force_error"] = max(force_errors)
        _write_json(metrics_path, metrics)
        previous = self._latest_dataset_path()
        merged = merge_datasets(previous, new_frames) if previous else new_frames
        version = len(list((self.root / "dataset").glob("dataset_v*.extxyz"))) + 1
        dataset_path = self.root / "dataset" / f"dataset_v{version:03d}.extxyz"
        write_dataset(
            dataset_path,
            merged,
            manifest_metadata={
                "campaign_id": self.config.campaign_id,
                "iteration": state.iteration,
            },
        )
        split_dataset_by_group(
            merged,
            self.root / "dataset" / f"split_v{version:03d}",
            ratios=self.config.split_ratios,
            seed=self.config.split_seed,
        )
        return dataset_path

    def _latest_split(self):
        split_dirs = sorted((self.root / "dataset").glob("split_v*"))
        if not split_dirs:
            raise RuntimeError("campaign has no dataset split")
        from .datasets import DatasetArtifact, DatasetSplit, summarize_dataset

        directory = split_dirs[-1]

        def artifact(name: str) -> DatasetArtifact:
            path = directory / f"{name}.extxyz"
            frames = load_dataset(path)
            from .datasets import compute_dataset_checksum

            return DatasetArtifact(
                path,
                compute_dataset_checksum(path),
                path.with_suffix(path.suffix + ".manifest.json"),
                summarize_dataset(frames),
            )

        payload = json.loads((directory / "split_manifest.json").read_text())
        return DatasetSplit(
            artifact("train"),
            artifact("valid"),
            artifact("test"),
            int(payload["seed"]),
            str(payload["group_key"]),
            payload["group_assignments"],
            directory / "split_manifest.json",
        )

    def _train_committee(self, state: CampaignState) -> list[TrainedModelArtifact]:
        split = self._latest_split()
        active = self._active_model(state)
        foundation = (
            str(active.path) if active else self.config.fine_tuning.foundation_model
        )
        artifacts = []
        for member, seed in enumerate(
            self.config.committee_seeds[: self.config.committee_size]
        ):
            member_dir = self._iteration_dir(state) / "models" / f"member_{member:02d}"
            if (member_dir / "model_manifest.json").is_file():
                artifacts.append(self._artifact_from_member_dir(member_dir))
                continue
            self._clear_stale_attempt(member_dir)
            config = replace(
                self.config.fine_tuning,
                name=f"{self.config.fine_tuning.name}_i{state.iteration:03d}_m{member:02d}",
                foundation_model=foundation,
                seed=seed,
            )
            artifacts.append(
                self.trainer.train(
                    split,
                    config,
                    member_dir,
                )
            )
        return artifacts

    def _artifact_from_member_dir(self, member_dir: Path) -> TrainedModelArtifact:
        manifest = json.loads((member_dir / "model_manifest.json").read_text())
        run = TrainingRun(
            tuple(manifest["command"]),
            0,
            "succeeded",
            member_dir / "stdout.log",
            member_dir / "stderr.log",
        )
        return TrainedModelArtifact(
            Path(manifest["model_path"]),
            manifest["sha256"],
            member_dir / "model_manifest.json",
            run,
        )

    def _load_training_artifacts(
        self, state: CampaignState
    ) -> list[TrainedModelArtifact]:
        artifacts = []
        for member_dir in sorted(
            (self._iteration_dir(state) / "models").glob("member_*")
        ):
            artifacts.append(self._artifact_from_member_dir(member_dir))
        if len(artifacts) != self.config.committee_size:
            raise RuntimeError("incomplete trained committee artifacts")
        return artifacts

    def _validate_models(
        self, state: CampaignState, artifacts: Sequence[TrainedModelArtifact]
    ) -> ModelRegistryEntry:
        split = self._latest_split()
        dataset_path = self._latest_dataset_path()
        if dataset_path is None:
            raise RuntimeError("model validation requires a campaign dataset")
        entries = []
        for member, (seed, artifact) in enumerate(
            zip(self.config.committee_seeds, artifacts)
        ):
            model_id = f"{self.config.campaign_id}-i{state.iteration:03d}-m{member:02d}"
            try:
                existing = self.registry.get_model(model_id)
            except ModelRegistryError:
                pass
            else:
                if not self.registry.verify_model_checksum(model_id):
                    raise RuntimeError(f"registered model is corrupted: {model_id}")
                entries.append(existing)
                continue
            output_dir = (
                self._iteration_dir(state) / "evaluation" / f"member_{member:02d}"
            )
            self._clear_stale_attempt(output_dir)
            evaluation = self.trainer.evaluate(artifact, split.valid_file, output_dir)
            if evaluation.n_failures:
                raise RuntimeError("model evaluation contains failed configurations")
            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "model_id": model_id,
                    "validation_metrics": asdict(evaluation.metrics),
                    "validation_status": "succeeded",
                }
            )
            _write_json(artifact.manifest_path, manifest)
            entries.append(
                self.registry.register_model(
                    artifact,
                    model_id=model_id,
                    foundation_model=self.config.fine_tuning.foundation_model,
                    fine_tuning_protocol=self.config.fine_tuning.protocol.value,
                    dataset_version=dataset_path.stem,
                    campaign_id=self.config.campaign_id,
                    iteration=state.iteration,
                    seed=seed,
                    metrics=evaluation.metrics,
                )
            )
        scored = []
        for entry in entries:
            value = getattr(entry.metrics, self.config.model_selection_metric, None)
            if value is None or not math.isfinite(value):
                raise RuntimeError(
                    "model selection metric is missing or nonfinite: "
                    f"{self.config.model_selection_metric} ({entry.model_id})"
                )
            scored.append((float(value), entry.model_id, entry))
        best = min(scored, key=lambda item: (item[0], item[1]))[2]
        active = self.registry.mark_model_active(best.model_id)
        test_output = self._iteration_dir(state) / "evaluation" / "selected_test"
        test_record = test_output / "post_selection_test_metrics.json"
        if not test_record.is_file():
            artifact_by_id = {
                (
                    f"{self.config.campaign_id}-i{state.iteration:03d}-m{member:02d}"
                ): artifact
                for member, artifact in enumerate(artifacts)
            }
            self._clear_stale_attempt(test_output)
            test_evaluation = self.trainer.evaluate(
                artifact_by_id[best.model_id], split.test_file, test_output
            )
            if test_evaluation.n_failures:
                raise RuntimeError("post-selection test evaluation contains failures")
            _write_json(
                test_record,
                {
                    "schema": "nebwalk.post_selection_test.v1",
                    "model_id": best.model_id,
                    "selection_dataset": "validation",
                    "evaluation_dataset": "test",
                    "metrics": asdict(test_evaluation.metrics),
                    "n_configurations": test_evaluation.n_configurations,
                    "n_failures": test_evaluation.n_failures,
                },
            )
        return active

    def _step(
        self, state: CampaignState, paths: Sequence[CampaignPath]
    ) -> CampaignState:
        if state.stage is CampaignStage.INITIALIZED:
            return self.store.advance(state, CampaignStage.BOOTSTRAPPING)
        if state.stage in {CampaignStage.BOOTSTRAPPING, CampaignStage.NEB_RUNNING}:
            bootstrap = state.stage is CampaignStage.BOOTSTRAPPING
            candidates, metrics = self._run_and_select(state, paths, bootstrap)
            if not bootstrap and metrics["new_configurations"] == 0:
                return self.store.advance(
                    state,
                    CampaignStage.CONVERGED,
                    stopping_reason="no_new_unique_configurations",
                )
            state = self.store.record_artifact(state, candidates)
            next_stage = (
                CampaignStage.CANDIDATES_READY if bootstrap else CampaignStage.SELECTION
            )
            return self.store.advance(state, next_stage)
        if state.stage in {CampaignStage.CANDIDATES_READY, CampaignStage.SELECTION}:
            return self.store.advance(state, CampaignStage.LABELING)
        if state.stage is CampaignStage.LABELING:
            structures = self._candidate_structures(state)
            if not structures:
                return self.store.advance(
                    state,
                    CampaignStage.CONVERGED,
                    stopping_reason="no_new_unique_configurations",
                )
            result = self.labeler.label(
                structures,
                self._iteration_dir(state) / "labels",
                {"campaign_id": self.config.campaign_id, "iteration": state.iteration},
            )
            if result.successful_count == 0:
                raise RuntimeError("all reference labels failed")
            return self.store.advance(state, CampaignStage.LABELS_READY)
        if state.stage is CampaignStage.LABELS_READY:
            labels_dir = self._iteration_dir(state) / "labels"
            manifest = json.loads((labels_dir / "label_manifest.json").read_text())
            from .labeling import ReferenceLabel

            labels_path = labels_dir / "labels.extxyz"
            labeled_frames = load_dataset(labels_path)
            dataset = DatasetArtifact(
                path=labels_path,
                checksum=compute_dataset_checksum(labels_path),
                manifest_path=labels_path.with_suffix(
                    labels_path.suffix + ".manifest.json"
                ),
                summary=summarize_dataset(labeled_frames),
            )

            label_result = LabelingResult(
                labels=tuple(
                    ReferenceLabel(
                        input_index=item["input_index"],
                        input_structure_hash=item["input_structure_hash"],
                        structure_hash=item["structure_hash"],
                        energy_eV=item["energy_eV"],
                        raw_work_directory=Path(item["raw_work_directory"]),
                        geometry_changed_during_recovery=item[
                            "geometry_changed_during_recovery"
                        ],
                        recovery_attempts=item["recovery_attempts"],
                        maximum_displacement_A=item.get("maximum_displacement_A", 0.0),
                        recovered_geometry_accepted=item.get(
                            "recovered_geometry_accepted", False
                        ),
                    )
                    for item in manifest["labels"]
                ),
                failures=(),
                dataset=dataset,
                output_dir=labels_dir,
                settings_hash=manifest["qe_settings_hash"],
            )
            dataset_path = self._update_dataset(state, label_result)
            state = self.store.record_artifact(state, dataset_path)
            return self.store.advance(state, CampaignStage.DATASET_UPDATED)
        if state.stage is CampaignStage.DATASET_UPDATED:
            return self.store.advance(state, CampaignStage.TRAINING)
        if state.stage is CampaignStage.TRAINING:
            self._train_committee(state)
            return self.store.advance(state, CampaignStage.MODELS_READY)
        if state.stage is CampaignStage.MODELS_READY:
            return self.store.advance(state, CampaignStage.MODEL_VALIDATION)
        if state.stage is CampaignStage.MODEL_VALIDATION:
            best = self._validate_models(state, self._load_training_artifacts(state))
            if state.iteration == 0:
                return self.store.advance(
                    state,
                    CampaignStage.NEB_RUNNING,
                    iteration=1,
                    active_model_id=best.model_id,
                )
            return self.store.advance(
                state,
                CampaignStage.STOPPING_CHECK,
                active_model_id=best.model_id,
            )
        if state.stage is CampaignStage.STOPPING_CHECK:
            metrics = json.loads(
                (self._iteration_dir(state) / "iteration_metrics.json").read_text()
            )
            decision = evaluate_stopping_criteria(
                self.config.stopping,
                metrics,
                iteration=state.iteration,
                consecutive_successes=state.consecutive_successes,
            )
            _write_json(
                self._iteration_dir(state) / "stopping_decision.json",
                asdict(decision),
            )
            all_satisfied = bool(decision.satisfied) and all(
                decision.satisfied.values()
            )
            consecutive = state.consecutive_successes + 1 if all_satisfied else 0
            if decision.stop:
                terminal = (
                    CampaignStage.MAX_ITERATIONS
                    if decision.reason == "maximum_iterations"
                    else CampaignStage.CONVERGED
                )
                return self.store.advance(
                    state,
                    terminal,
                    consecutive_successes=consecutive,
                    stopping_reason=decision.reason,
                )
            return self.store.advance(
                state,
                CampaignStage.NEB_RUNNING,
                iteration=state.iteration + 1,
                consecutive_successes=consecutive,
            )
        raise CampaignStateError(f"cannot advance campaign from {state.stage.value}")

    def run(self, paths: Sequence[CampaignPath] | None = None) -> ActiveLearningResult:
        """Run or continue until an objective terminal state is reached."""
        with CampaignLock(self.root):
            state = self._initialize()
            if paths is not None:
                self._save_paths(paths)
            persisted_paths = self._load_paths()
            while state.stage not in self.TERMINAL:
                if state.stage is CampaignStage.FAILED:
                    raise CampaignStateError("campaign is failed; call resume()")
                try:
                    state = self._step(state, persisted_paths)
                except Exception as exc:
                    self.store.record_failure(state, exc)
                    raise
            from .reporting import write_campaign_summary

            write_campaign_summary(self.root)
            return self._result(state)

    def initialize(self, paths: Sequence[CampaignPath]) -> ActiveLearningResult:
        """Create durable configuration and endpoint artifacts without running work."""
        with CampaignLock(self.root):
            state = self._initialize()
            self._save_paths(paths)
            return self._result(state)

    def run_iteration(self) -> ActiveLearningResult:
        """Advance exactly one durable stage for controlled execution or debugging."""
        with CampaignLock(self.root):
            state = self._initialize()
            paths = self._load_paths()
            if state.stage in self.TERMINAL:
                return self._result(state)
            try:
                state = self._step(state, paths)
            except Exception as exc:
                self.store.record_failure(state, exc)
                raise
            return self._result(state)

    def resume(self, *, force: bool = False) -> ActiveLearningResult:
        """Resume safely; force records intent but never invalidates artifacts."""
        state = self.store.load()
        if state.stage is CampaignStage.FAILED:
            if state.failed_stage is None:
                raise CampaignStateError(
                    "failed state does not record its failed stage"
                )
            state = self.store.save(
                replace(
                    state,
                    stage=state.failed_stage,
                    failed_stage=None,
                    failure=None,
                )
            )
        if force:
            self.store.event(
                "forced_resume_requested",
                {
                    "stage": state.stage.value,
                    "behavior": (
                        "resume_from_current_durable_stage_without_artifact_invalidation"
                    ),
                },
            )
        return self.run()

    def status(self) -> CampaignState:
        """Return integrity-verified current state without changing it."""
        return self.store.load()

    def validate_final(self) -> Mapping[str, Any]:
        """Run explicit QE checks around each latest predicted saddle."""
        with CampaignLock(self.root):
            state = self.store.load()
            if state.stage not in self.TERMINAL:
                raise CampaignStateError("final validation requires a stopped campaign")
            iteration = state.iteration
            profiles_path = (
                self.root / f"iteration_{iteration:03d}" / "selected_candidates.json"
            )
            if not profiles_path.exists() and iteration > 0:
                iteration -= 1
                profiles_path = (
                    self.root
                    / f"iteration_{iteration:03d}"
                    / "selected_candidates.json"
                )
            profiles = json.loads(profiles_path.read_text())["profiles"]
            structures: list[Atoms] = []
            checked: list[dict[str, Any]] = []
            for profile in profiles:
                path_file = (
                    self.root / f"iteration_{iteration:03d}" / profile["path_file"]
                )
                loaded = read(path_file, index=":", format="extxyz")
                images = [loaded] if isinstance(loaded, Atoms) else list(loaded)
                energies = profile["energies_eV"]
                saddle = int(np.argmax(energies))
                indices = sorted({0, len(images) - 1, saddle - 1, saddle, saddle + 1})
                indices = [index for index in indices if 0 <= index < len(images)]
                for index in indices:
                    atoms = images[index]
                    atoms.info.update(
                        {
                            "campaign_id": self.config.campaign_id,
                            "iteration": iteration,
                            "path_id": profile["path_id"],
                            "image_index": index,
                            "selection_reason": "final_qe_validation",
                        }
                    )
                    structures.append(atoms)
                checked.append(
                    {
                        "path_id": profile["path_id"],
                        "mlip_barrier_eV": profile["barrier_eV"],
                        "checked_indices": indices,
                    }
                )
            labels = self.labeler.label(
                structures,
                self.root / "final_validation" / "single_points",
                {"campaign_id": self.config.campaign_id},
            )
            full_qe = None
            if self.config.full_final_qe_neb:
                if self.final_neb_validator is None:
                    raise RuntimeError(
                        "full final QE NEB requested but no validator provided"
                    )
                full_qe = dict(
                    self.final_neb_validator(
                        self._load_paths(),
                        self.root / "final_validation" / "full_qe_neb",
                    )
                )
            report = {
                "schema": "nebwalk.final_validation.v1",
                "validation_status": (
                    "complete" if labels.failed_count == 0 else "partial"
                ),
                "mlip_profiles": checked,
                "qe_checked_configurations": labels.successful_count,
                "qe_failed_configurations": labels.failed_count,
                "full_qe_neb": full_qe,
                "full_qe_neb_performed": full_qe is not None,
                "disclosure": (
                    "Sparse QE checks are not a full DFT NEB barrier. A DFT barrier is "
                    "reported only when full_qe_neb_performed is true."
                ),
            }
            _write_json(
                self.root / "final_validation" / "final_validation.json", report
            )
            from .reporting import write_campaign_summary

            write_campaign_summary(self.root)
            return report

    def _result(self, state: CampaignState) -> ActiveLearningResult:
        return ActiveLearningResult(
            state,
            self._latest_dataset_path(),
            self._active_model(state),
            self.root,
        )


__all__ = [
    "ActiveLearningCampaign",
    "ActiveLearningConfig",
    "ActiveLearningResult",
    "CampaignPath",
    "CampaignPathResult",
    "CampaignPathRunner",
    "MACEPathRunner",
    "StoppingCriteriaConfig",
    "StoppingDecision",
    "evaluate_stopping_criteria",
]
