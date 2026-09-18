"""Atomic campaign state and mocked closed-loop orchestration tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from NEBwalk.campaign import (
    ActiveLearningCampaign,
    ActiveLearningConfig,
    CampaignPath,
    CampaignPathResult,
    StoppingCriteriaConfig,
    evaluate_stopping_criteria,
)
from NEBwalk.campaign_state import (
    CampaignLock,
    CampaignLockError,
    CampaignStage,
    CampaignStateError,
    CampaignStateStore,
)
from NEBwalk.candidate_selection import CandidateSelectionConfig
from NEBwalk.datasets import compute_structure_hash, write_dataset
from NEBwalk.labeling import LabelingResult, ReferenceLabel
from NEBwalk.mlip import (
    FineTuningConfig,
    ModelEvaluation,
    ModelMetrics,
    TrainedModelArtifact,
    TrainingRun,
)


class FakeLabeler:
    def __init__(self):
        self.calls = 0

    def label(self, structures, output_dir, metadata=None):
        self.calls += 1
        output_dir = Path(output_dir)
        frames = []
        labels = []
        for index, source in enumerate(structures):
            frame = source.copy()
            structure_hash = compute_structure_hash(frame)
            energy = float(np.sum(frame.positions**2))
            frame.info.update(metadata or {})
            frame.info.update(
                {
                    "REF_energy": energy,
                    "config_type": "neb_image",
                    "campaign_id": frame.info.get("campaign_id", "campaign"),
                    "iteration": int(frame.info.get("iteration", 0)),
                    "path_id": str(frame.info["path_id"]),
                    "image_index": int(frame.info["image_index"]),
                    "selection_reason": str(frame.info["selection_reason"]),
                    "calculator_name": "FakeReference",
                    "dft_settings_hash": "fake-settings",
                    "structure_hash": structure_hash,
                }
            )
            frame.arrays["REF_forces"] = -2.0 * frame.positions
            frames.append(frame)
            labels.append(
                ReferenceLabel(
                    index,
                    structure_hash,
                    structure_hash,
                    energy,
                    output_dir / "raw" / str(index),
                    False,
                    0,
                )
            )
        artifact = write_dataset(output_dir / "labels.extxyz", frames)
        (output_dir / "raw").mkdir(exist_ok=True)
        (output_dir / "label_manifest.json").write_text(
            json.dumps(
                {
                    "qe_settings_hash": "fake-settings",
                    "labels": [
                        {
                            **asdict(label),
                            "raw_work_directory": str(label.raw_work_directory),
                        }
                        for label in labels
                    ],
                }
            ),
            encoding="utf-8",
        )
        return LabelingResult(tuple(labels), (), artifact, output_dir, "fake-settings")


class FakeTrainer:
    def __init__(self, fail_once_at_call=None):
        self.train_calls = 0
        self.evaluate_calls = 0
        self.evaluation_datasets = []
        self.fail_once_at_call = fail_once_at_call
        self.failed = False

    def validate_environment(self, protocol=None):
        raise NotImplementedError

    def train(self, dataset, config, output_dir):
        self.train_calls += 1
        if self.fail_once_at_call == self.train_calls and not self.failed:
            self.failed = True
            raise RuntimeError("injected training interruption")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = output_dir / f"{config.name}.model"
        model.write_bytes(f"model-{config.seed}".encode())
        checksum = hashlib.sha256(model.read_bytes()).hexdigest()
        manifest = {
            "model_path": str(model),
            "sha256": checksum,
            "command": ["fake_train", "--seed", str(config.seed)],
        }
        (output_dir / "model_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (output_dir / "stdout.log").write_text("ok", encoding="utf-8")
        (output_dir / "stderr.log").write_text("", encoding="utf-8")
        run = TrainingRun(
            tuple(manifest["command"]),
            0,
            "succeeded",
            output_dir / "stdout.log",
            output_dir / "stderr.log",
        )
        return TrainedModelArtifact(
            model, checksum, output_dir / "model_manifest.json", run
        )

    def evaluate(self, model, dataset, output_dir):
        self.evaluate_calls += 1
        self.evaluation_datasets.append(Path(dataset).name)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metric = 0.1 + int(model.model_path.stem.rsplit("_m", 1)[-1]) * 0.01
        return ModelEvaluation(
            ModelMetrics(force_component_rmse=metric),
            n_configurations=1,
            n_failures=0,
            output_dir=output_dir,
        )


class FakePathRunner:
    def __init__(self):
        self.calls = 0

    def __call__(self, paths, active_model, output_dir):
        self.calls += 1
        shift = 0.0 if active_model is None else 0.02
        results = []
        for path_index, path in enumerate(paths):
            y = path_index * 2.0
            images = (
                Atoms("H", positions=[[0.0 + shift, y, 0.0]]),
                Atoms("H", positions=[[0.5 + shift, y, 0.0]]),
                Atoms("H", positions=[[1.0 + shift, y, 0.0]]),
            )
            results.append(
                CampaignPathResult(
                    path.path_id,
                    images,
                    (0.0, 1.0, 0.0),
                    1.0,
                )
            )
        return results


def _paths():
    return [
        CampaignPath(
            f"path-{index}",
            Atoms("H", positions=[[0.0, index * 2.0, 0.0]]),
            Atoms("H", positions=[[1.0, index * 2.0, 0.0]]),
        )
        for index in range(3)
    ]


def _config(tmp_path):
    return ActiveLearningConfig(
        campaign_id="campaign-test",
        campaign_dir=tmp_path / "campaign",
        fine_tuning=FineTuningConfig(name="active", foundation_model="foundation"),
        candidate_selection=CandidateSelectionConfig(
            n_select=3,
            peak_neighbors=1,
            include_endpoints_during_bootstrap=True,
            diversity=False,
        ),
        stopping=StoppingCriteriaConfig(
            maximum_iterations=4,
            minimum_new_configurations=1,
            consecutive_successful_iterations=2,
        ),
        split_ratios=(1 / 3, 1 / 3, 1 / 3),
    )


def test_stopping_never_converges_with_missing_configured_metrics():
    decision = evaluate_stopping_criteria(
        StoppingCriteriaConfig(maximum_force_uncertainty=0.1),
        {"new_configurations": 2.0},
        iteration=1,
        consecutive_successes=10,
    )
    assert decision.stop is False
    assert decision.satisfied["max_force_uncertainty"] is False


def test_stopping_requires_consecutive_successes_and_handles_maximum():
    config = StoppingCriteriaConfig(
        maximum_iterations=3,
        maximum_force_uncertainty=0.2,
        consecutive_successful_iterations=2,
    )
    metrics = {"new_configurations": 2.0, "max_force_uncertainty": 0.1}
    first = evaluate_stopping_criteria(config, metrics, iteration=1)
    second = evaluate_stopping_criteria(
        config, metrics, iteration=2, consecutive_successes=1
    )
    maximum = evaluate_stopping_criteria(config, metrics, iteration=3)
    assert first.stop is False
    assert second.reason == "objective_thresholds_satisfied"
    assert maximum.reason == "maximum_iterations"


def test_campaign_lock_rejects_second_writer(tmp_path):
    first = CampaignLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(CampaignLockError, match="locked"):
            CampaignLock(tmp_path).acquire()
    finally:
        first.release()


def test_corrupted_state_and_artifact_are_detected(tmp_path):
    store = CampaignStateStore(tmp_path)
    state = store.initialize("test")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("valid")
    state = store.record_artifact(state, artifact)
    artifact.write_text("corrupt")
    with pytest.raises(CampaignStateError, match="integrity"):
        store.load()

    store.state_path.write_text("not-json")
    with pytest.raises(CampaignStateError, match="corrupted"):
        store.load()


def test_mock_campaign_bootstraps_retrains_and_stops_on_no_new_candidates(tmp_path):
    trainer = FakeTrainer()
    labeler = FakeLabeler()
    runner = FakePathRunner()
    campaign = ActiveLearningCampaign(
        _config(tmp_path), trainer, labeler, path_runner=runner
    )

    result = campaign.run(_paths())

    assert result.state.stage is CampaignStage.CONVERGED
    assert result.state.stopping_reason == "no_new_unique_configurations"
    assert result.active_model is not None
    assert result.dataset_path.is_file()
    assert trainer.train_calls == 6  # bootstrap and one update, three members each
    assert trainer.evaluate_calls == 8
    assert trainer.evaluation_datasets == [
        "valid.extxyz",
        "valid.extxyz",
        "valid.extxyz",
        "test.extxyz",
        "valid.extxyz",
        "valid.extxyz",
        "valid.extxyz",
        "test.extxyz",
    ]
    post_selection = (
        result.campaign_dir
        / "iteration_001"
        / "evaluation"
        / "selected_test"
        / "post_selection_test_metrics.json"
    )
    assert json.loads(post_selection.read_text())["selection_dataset"] == "validation"
    assert labeler.calls == 2
    assert runner.calls == 3
    assert (result.campaign_dir / "iteration_000").is_dir()
    assert (result.campaign_dir / "iteration_001").is_dir()
    assert (result.campaign_dir / "iteration_002").is_dir()


def test_failed_training_resumes_without_overwriting_completed_member(tmp_path):
    trainer = FakeTrainer(fail_once_at_call=2)
    campaign = ActiveLearningCampaign(
        _config(tmp_path), trainer, FakeLabeler(), path_runner=FakePathRunner()
    )

    with pytest.raises(RuntimeError, match="injected"):
        campaign.run(_paths())
    assert campaign.status().stage is CampaignStage.FAILED
    first_model = next(
        (tmp_path / "campaign" / "iteration_000" / "models" / "member_00").glob(
            "*.model"
        )
    )
    original = first_model.read_bytes()

    result = campaign.resume()

    assert result.state.stage is CampaignStage.CONVERGED
    assert first_model.read_bytes() == original
    assert trainer.train_calls == 7  # one failed attempt; completed member was reused


def test_run_iteration_advances_exactly_one_stage(tmp_path):
    campaign = ActiveLearningCampaign(
        _config(tmp_path), FakeTrainer(), FakeLabeler(), path_runner=FakePathRunner()
    )
    initialized = campaign.initialize(_paths())
    assert initialized.state.stage is CampaignStage.INITIALIZED

    advanced = campaign.run_iteration()
    assert advanced.state.stage is CampaignStage.BOOTSTRAPPING


def test_fresh_campaign_instance_resumes_every_durable_major_stage(tmp_path):
    config = _config(tmp_path)
    trainer = FakeTrainer()
    labeler = FakeLabeler()
    runner = FakePathRunner()
    campaign = ActiveLearningCampaign(config, trainer, labeler, path_runner=runner)
    campaign.initialize(_paths())
    observed = []

    for _ in range(40):
        campaign = ActiveLearningCampaign(config, trainer, labeler, path_runner=runner)
        result = campaign.run_iteration()
        observed.append(result.state.stage)
        if result.state.stage in {
            CampaignStage.CONVERGED,
            CampaignStage.MAX_ITERATIONS,
        }:
            break

    assert CampaignStage.CANDIDATES_READY in observed
    assert CampaignStage.LABELING in observed
    assert CampaignStage.DATASET_UPDATED in observed
    assert CampaignStage.TRAINING in observed
    assert CampaignStage.MODEL_VALIDATION in observed
    assert CampaignStage.NEB_RUNNING in observed
    assert observed[-1] is CampaignStage.CONVERGED


def test_final_validation_is_explicitly_not_a_full_qe_barrier(tmp_path):
    campaign = ActiveLearningCampaign(
        _config(tmp_path), FakeTrainer(), FakeLabeler(), path_runner=FakePathRunner()
    )
    campaign.run(_paths())

    report = campaign.validate_final()

    assert report["validation_status"] == "complete"
    assert report["full_qe_neb_performed"] is False
    assert report["full_qe_neb"] is None
    assert "not a full DFT NEB barrier" in report["disclosure"]


def test_labeling_interruption_resumes_without_repeating_completed_stages(tmp_path):
    class FailOnceLabeler(FakeLabeler):
        def label(self, structures, output_dir, metadata=None):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("injected labeling interruption")
            return super().label(structures, output_dir, metadata)

    labeler = FailOnceLabeler()
    campaign = ActiveLearningCampaign(
        _config(tmp_path), FakeTrainer(), labeler, path_runner=FakePathRunner()
    )
    with pytest.raises(RuntimeError, match="labeling interruption"):
        campaign.run(_paths())
    assert campaign.status().failed_stage is CampaignStage.LABELING

    assert campaign.resume().state.stage is CampaignStage.CONVERGED


def test_neb_interruption_resumes_from_neb_stage(tmp_path):
    class FailOnceRunner(FakePathRunner):
        def __call__(self, paths, active_model, output_dir):
            if self.calls == 1:
                self.calls += 1
                raise RuntimeError("injected NEB interruption")
            return super().__call__(paths, active_model, output_dir)

    runner = FailOnceRunner()
    campaign = ActiveLearningCampaign(
        _config(tmp_path), FakeTrainer(), FakeLabeler(), path_runner=runner
    )
    with pytest.raises(RuntimeError, match="NEB interruption"):
        campaign.run(_paths())
    assert campaign.status().failed_stage is CampaignStage.NEB_RUNNING

    assert campaign.resume().state.stage is CampaignStage.CONVERGED


def test_model_validation_interruption_resumes_from_validation_stage(tmp_path):
    class FailOnceValidationTrainer(FakeTrainer):
        def evaluate(self, model, dataset, output_dir):
            if self.evaluate_calls == 0:
                self.evaluate_calls += 1
                raise RuntimeError("injected validation interruption")
            return super().evaluate(model, dataset, output_dir)

    trainer = FailOnceValidationTrainer()
    campaign = ActiveLearningCampaign(
        _config(tmp_path), trainer, FakeLabeler(), path_runner=FakePathRunner()
    )
    with pytest.raises(RuntimeError, match="validation interruption"):
        campaign.run(_paths())
    assert campaign.status().failed_stage is CampaignStage.MODEL_VALIDATION

    assert campaign.resume().state.stage is CampaignStage.CONVERGED


def test_dataset_merge_and_stopping_check_resume_from_durable_state(tmp_path):
    config = _config(tmp_path)
    campaign = ActiveLearningCampaign(
        config, FakeTrainer(), FakeLabeler(), path_runner=FakePathRunner()
    )
    campaign.initialize(_paths())
    while campaign.status().stage is not CampaignStage.DATASET_UPDATED:
        campaign.run_iteration()
    resumed = ActiveLearningCampaign(
        config, FakeTrainer(), FakeLabeler(), path_runner=FakePathRunner()
    )
    assert resumed.run_iteration().state.stage is CampaignStage.TRAINING

    while resumed.status().stage is not CampaignStage.STOPPING_CHECK:
        resumed.run_iteration()
    final_instance = ActiveLearningCampaign(
        config, FakeTrainer(), FakeLabeler(), path_runner=FakePathRunner()
    )
    assert final_instance.run_iteration().state.stage in {
        CampaignStage.NEB_RUNNING,
        CampaignStage.MAX_ITERATIONS,
        CampaignStage.CONVERGED,
    }


def test_force_resume_records_non_destructive_behavior_for_terminal_campaign(tmp_path):
    campaign = ActiveLearningCampaign(
        _config(tmp_path), FakeTrainer(), FakeLabeler(), path_runner=FakePathRunner()
    )
    result = campaign.run(_paths())
    state_before = campaign.status()

    forced = campaign.resume(force=True)

    assert forced.state.stage is result.state.stage
    assert forced.state.completed_artifacts == state_before.completed_artifacts
    events = [
        json.loads(line)
        for line in (result.campaign_dir / "events.jsonl").read_text().splitlines()
    ]
    event = next(item for item in events if item["event"] == "forced_resume_requested")
    assert "without_artifact_invalidation" in event["behavior"]
