"""Fine-tuning contracts, MACE CLI adapter, metrics, and registry tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.io import write

from NEBwalk.datasets import compute_structure_hash, split_dataset_by_group
from NEBwalk.mlip import (
    FineTuningConfig,
    FineTuningProtocol,
    MACEBackendError,
    MACECompatibilityError,
    MACEEvaluationError,
    MACEFineTuningBackend,
    MACETrainingError,
    ModelMetrics,
    ModelRegistry,
    ModelRegistryError,
    TrainedModelArtifact,
    TrainingRun,
    compute_model_metrics,
    compute_pathway_metrics,
)
from tests.test_datasets import _frame

HELP = """--name --foundation_model --train_file --valid_file --test_file
--lora --lora_rank --lora_alpha --multiheads_finetuning
--pt_train_file --num_samples_pt
"""


def _executable(tmp_path: Path, name: str = "mace_run_train") -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _split(tmp_path: Path):
    frames = []
    for path_index in range(3):
        frame = _frame(f"path-{path_index}", 0)
        frame.positions[:, 1] += path_index * 0.1
        frame.info["structure_hash"] = compute_structure_hash(frame)
        frames.append(frame)
    return split_dataset_by_group(frames, tmp_path / "split", seed=2)


def _completed(argv, *, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


@pytest.mark.parametrize(
    ("protocol", "kwargs", "expected_flags"),
    [
        (FineTuningProtocol.NAIVE, {}, ("--E0s", "estimated")),
        (
            FineTuningProtocol.LORA,
            {"lora_rank": 8, "lora_alpha": 16.0},
            ("--lora", "--lora_rank", "--lora_alpha"),
        ),
        (
            FineTuningProtocol.MULTIHEAD_REPLAY,
            {"replay_dataset": Path("replay.xyz"), "num_replay_samples": 20},
            ("--multiheads_finetuning", "--pt_train_file", "--num_samples_pt"),
        ),
    ],
)
def test_command_generation_for_all_protocols(
    tmp_path, protocol, kwargs, expected_flags
):
    executable = _executable(tmp_path)
    backend = MACEFineTuningBackend(str(executable))
    config = FineTuningConfig(
        name="model",
        foundation_model="MACE-MP-0",
        protocol=protocol,
        **kwargs,
    )

    command = backend.build_train_command(
        _split(tmp_path), config, tmp_path / "run", str(executable)
    )

    assert command[0] == str(executable)
    assert all(flag in command for flag in expected_flags)
    assert "estimated" in command
    assert not any(";" in argument for argument in command)
    for switch in ("--ema", "--amsgrad"):
        if switch in command:
            assert (switch, "True") not in zip(command, command[1:])


def test_presence_only_mace_flags_do_not_receive_boolean_values(tmp_path):
    executable = _executable(tmp_path)
    config = FineTuningConfig(
        name="model",
        foundation_model="MACE-MP-0",
        ema=True,
        amsgrad=True,
        restart_latest=True,
    )

    command = MACEFineTuningBackend(str(executable)).build_train_command(
        _split(tmp_path), config, tmp_path / "run", str(executable)
    )

    for switch in ("--ema", "--amsgrad", "--restart_latest"):
        assert command.count(switch) == 1
        assert (switch, "True") not in zip(command, command[1:])


def test_fine_tuning_config_rejects_incomplete_protocol_settings():
    with pytest.raises(ValueError, match="lora_rank"):
        FineTuningConfig(
            name="bad",
            foundation_model="foundation",
            protocol=FineTuningProtocol.LORA,
        )
    with pytest.raises(ValueError, match="replay_dataset"):
        FineTuningConfig(
            name="bad",
            foundation_model="foundation",
            protocol=FineTuningProtocol.MULTIHEAD_REPLAY,
        )


def test_missing_executable_is_explicit(tmp_path):
    backend = MACEFineTuningBackend(str(tmp_path / "missing"))
    with pytest.raises(MACEBackendError, match="not found"):
        backend.validate_environment()


def test_capability_flag_missing_is_explicit(tmp_path):
    executable = _executable(tmp_path)

    def runner(argv, **kwargs):
        return _completed(argv, stdout="--name --train_file")

    backend = MACEFineTuningBackend(str(executable), runner=runner)
    with pytest.raises(MACECompatibilityError, match="--lora"):
        backend.validate_environment(FineTuningProtocol.LORA)


def test_environment_and_training_never_use_shell(tmp_path):
    executable = _executable(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if argv[-1] == "--help":
            return _completed(argv, stdout=HELP)
        model_dir = Path(argv[argv.index("--model_dir") + 1])
        name = argv[argv.index("--name") + 1]
        (model_dir / f"{name}.model").write_bytes(b"trained-model")
        return _completed(argv, stdout="done")

    backend = MACEFineTuningBackend(str(executable), runner=runner)
    artifact = backend.train(
        _split(tmp_path),
        FineTuningConfig(name="safe", foundation_model="foundation", seed=7),
        tmp_path / "training",
    )

    assert artifact.model_path.is_file()
    assert artifact.model_path.is_absolute()
    assert artifact.training_run.status == "succeeded"
    assert all(kwargs["shell"] is False for _, kwargs in calls)
    assert all(kwargs["check"] is False for _, kwargs in calls)
    assert calls[0][1]["timeout"] == 60.0
    assert calls[1][1]["timeout"] is None
    for filename in (
        "training_config.json",
        "command.json",
        "stdout.log",
        "stderr.log",
        "environment.json",
        "dataset_manifest.json",
        "training_result.json",
        "model_manifest.json",
    ):
        assert (tmp_path / "training" / filename).exists()
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sha256"] == artifact.checksum
    assert manifest["seed"] == 7
    assert manifest["model_id"] == "safe"
    assert manifest["e0s"] == "estimated"
    assert manifest["atomic_reference_strategy"] == "configured_by_training_run"
    assert manifest["split_manifest_sha256"]
    assert manifest["validation_metrics"] is None
    assert manifest["creation_time"].endswith("+00:00")


def test_training_nonzero_return_code_is_not_swallowed(tmp_path):
    executable = _executable(tmp_path)

    def runner(argv, **kwargs):
        if argv[-1] == "--help":
            return _completed(argv, stdout=HELP)
        return _completed(argv, returncode=3, stderr="training exploded")

    backend = MACEFineTuningBackend(str(executable), runner=runner)
    output = tmp_path / "failed"
    with pytest.raises(MACETrainingError, match="code 3"):
        backend.train(
            _split(tmp_path),
            FineTuningConfig(name="failed", foundation_model="foundation"),
            output,
        )
    assert (
        json.loads((output / "training_result.json").read_text())["status"] == "failed"
    )
    assert "training exploded" in (output / "stderr.log").read_text()


def test_training_timeout_is_recorded_and_reported(tmp_path):
    executable = _executable(tmp_path)

    def runner(argv, **kwargs):
        if argv[-1] == "--help":
            return _completed(argv, stdout=HELP)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    backend = MACEFineTuningBackend(
        str(executable), runner=runner, training_timeout_seconds=2.0
    )
    output = tmp_path / "timeout"
    with pytest.raises(MACETrainingError, match="exceeded 2.0 seconds"):
        backend.train(
            _split(tmp_path),
            FineTuningConfig(name="timeout", foundation_model="foundation"),
            output,
        )
    result = json.loads((output / "training_result.json").read_text())
    assert result["status"] == "timed_out"


def test_training_success_without_model_is_failure(tmp_path):
    executable = _executable(tmp_path)

    def runner(argv, **kwargs):
        return _completed(argv, stdout=HELP if argv[-1] == "--help" else "done")

    backend = MACEFineTuningBackend(str(executable), runner=runner)
    with pytest.raises(MACETrainingError, match="no unambiguous"):
        backend.train(
            _split(tmp_path),
            FineTuningConfig(name="missing", foundation_model="foundation"),
            tmp_path / "missing-model",
        )


def test_output_collision_requires_explicit_restart(tmp_path):
    executable = _executable(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "old.txt").write_text("old")
    backend = MACEFineTuningBackend(str(executable))

    with pytest.raises(FileExistsError, match="not empty"):
        backend.train(
            _split(tmp_path),
            FineTuningConfig(name="model", foundation_model="foundation"),
            output,
        )


def test_restart_command_is_explicit(tmp_path):
    executable = _executable(tmp_path)
    config = FineTuningConfig(
        name="restart",
        foundation_model="foundation",
        restart_latest=True,
    )
    command = MACEFineTuningBackend(str(executable)).build_train_command(
        _split(tmp_path), config, tmp_path / "run", str(executable)
    )
    assert command.count("--restart_latest") == 1
    assert ("--restart_latest", "True") not in zip(command, command[1:])


def test_model_and_pathway_metrics_have_expected_values():
    metrics = compute_model_metrics(
        [-2.0],
        [-1.0],
        [2],
        [np.zeros((2, 3))],
        [np.ones((2, 3))],
    )
    assert metrics.energy_mae_per_atom == pytest.approx(0.5)
    assert metrics.force_component_rmse == pytest.approx(1.0)
    assert metrics.force_vector_rmse == pytest.approx(np.sqrt(3.0))
    assert metrics.max_atom_force_error == pytest.approx(np.sqrt(3.0))

    pathway = compute_pathway_metrics([0.0, 1.0, 0.2], [5.0, 5.8, 5.1])
    assert pathway["barrier_error"] == pytest.approx(-0.2)
    assert pathway["reaction_energy_error"] == pytest.approx(-0.1)
    assert pathway["transition_state_index_difference"] == 0


def _artifact(tmp_path: Path, name: str = "model") -> TrainedModelArtifact:
    model = tmp_path / f"{name}.model"
    model.write_bytes(name.encode())
    import hashlib

    checksum = hashlib.sha256(model.read_bytes()).hexdigest()
    run = TrainingRun((), 0, "succeeded", tmp_path / "out", tmp_path / "err")
    return TrainedModelArtifact(model, checksum, tmp_path / "manifest.json", run)


def _register(registry, artifact, model_id, metric):
    return registry.register_model(
        artifact,
        model_id=model_id,
        foundation_model="foundation",
        fine_tuning_protocol="naive",
        dataset_version="dataset-v1",
        campaign_id="campaign",
        iteration=0,
        seed=1,
        metrics=ModelMetrics(force_component_rmse=metric),
        license_provenance_notes="Foundation model terms reviewed",
    )


def test_model_registry_selects_validation_metric_and_verifies_checksum(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    _register(registry, _artifact(tmp_path, "worse"), "worse", 0.3)
    _register(registry, _artifact(tmp_path, "better"), "better", 0.1)

    assert registry.best_model().model_id == "better"
    assert registry.mark_model_active("better").status == "active"
    assert registry.list_models(status="active")[0].model_id == "better"
    assert registry.verify_model_checksum("better") is True

    registry.get_model("better").path.write_bytes(b"corrupted")
    assert registry.verify_model_checksum("better") is False


def test_registry_rejects_training_loss_and_duplicate_ids(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    artifact = _artifact(tmp_path)
    _register(registry, artifact, "model", 0.1)

    with pytest.raises(ModelRegistryError, match="training loss"):
        registry.best_model("training_loss")
    with pytest.raises(ModelRegistryError, match="already"):
        _register(registry, artifact, "model", 0.2)


def test_evaluation_failure_is_explicit(tmp_path):
    executable = _executable(tmp_path, "mace_eval_configs")
    artifact = _artifact(tmp_path)

    def runner(argv, **kwargs):
        return _completed(argv, returncode=4, stderr="bad model")

    backend = MACEFineTuningBackend(
        str(executable), evaluation_executable=str(executable), runner=runner
    )
    with pytest.raises(MACEEvaluationError, match="code 4"):
        backend.evaluate(artifact, _split(tmp_path).test_file, tmp_path / "evaluation")


def test_evaluation_calculates_metrics_from_official_cli_output(tmp_path):
    executable = _executable(tmp_path, "mace_eval_configs")
    artifact = _artifact(tmp_path)
    split = _split(tmp_path)

    def runner(argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        predictions = __import__(
            "NEBwalk.datasets", fromlist=["load_dataset"]
        ).load_dataset(split.test_file)
        for frame in predictions:
            frame.info["MACE_energy"] = frame.info["REF_energy"] + 0.2
            frame.arrays["MACE_forces"] = frame.arrays["REF_forces"] + 0.1
        write(output, predictions, format="extxyz")
        return _completed(argv)

    backend = MACEFineTuningBackend(
        str(executable), evaluation_executable=str(executable), runner=runner
    )
    evaluation = backend.evaluate(artifact, split.test_file, tmp_path / "evaluation")

    assert evaluation.n_failures == 0
    assert evaluation.metrics.energy_mae_per_atom == pytest.approx(0.1)
    assert evaluation.metrics.force_component_mae == pytest.approx(0.1)
    assert (evaluation.output_dir / "training_metrics.json").exists()
