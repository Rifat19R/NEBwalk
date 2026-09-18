"""Argparse command-line interface for datasets, MACE, and campaigns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from .campaign import (
    ActiveLearningCampaign,
    ActiveLearningConfig,
    CampaignPath,
    MACEPathRunner,
    StoppingCriteriaConfig,
)
from .campaign_state import CampaignStateStore
from .candidate_selection import CandidateSelectionConfig
from .datasets import (
    load_dataset,
    load_dataset_split,
    split_dataset_by_group,
    summarize_dataset,
    validate_dataset,
)
from .engine import NEBRunConfig
from .labeling import QEReferenceLabeler
from .mlip import (
    FineTuningConfig,
    FineTuningProtocol,
    MACEFineTuningBackend,
    TrainedModelArtifact,
    TrainingRun,
)
from .qe import QEParams


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _fine_tuning(payload: dict[str, Any]) -> FineTuningConfig:
    values = dict(payload)
    values["protocol"] = FineTuningProtocol(values.get("protocol", "naive"))
    if values.get("replay_dataset") is not None:
        values["replay_dataset"] = Path(values["replay_dataset"])
    if "extra_args" in values:
        values["extra_args"] = tuple(values["extra_args"])
    return FineTuningConfig(**values)


def _model_artifact(manifest_path: Path) -> TrainedModelArtifact:
    manifest = _json(manifest_path)
    run = TrainingRun(
        tuple(manifest.get("command", [])),
        0,
        "succeeded",
        manifest_path.parent / "stdout.log",
        manifest_path.parent / "stderr.log",
    )
    return TrainedModelArtifact(
        Path(manifest["model_path"]),
        manifest["sha256"],
        manifest_path,
        run,
    )


def _campaign_components(config_path: Path):
    payload = _json(config_path)
    active = dict(payload["active_learning"])
    campaign_dir = Path(active.pop("campaign_dir"))
    if not campaign_dir.is_absolute():
        campaign_dir = (config_path.parent / campaign_dir).resolve()
    fine_tuning = _fine_tuning(active.pop("fine_tuning"))
    selection = CandidateSelectionConfig(**active.pop("candidate_selection", {}))
    stopping = StoppingCriteriaConfig(**active.pop("stopping", {}))
    if "committee_seeds" in active:
        active["committee_seeds"] = tuple(active["committee_seeds"])
    if "split_ratios" in active:
        active["split_ratios"] = tuple(active["split_ratios"])
    config = ActiveLearningConfig(
        campaign_dir=campaign_dir,
        fine_tuning=fine_tuning,
        candidate_selection=selection,
        stopping=stopping,
        **active,
    )
    qe = dict(payload["qe"])
    qe_params = QEParams(**qe.pop("params", {}))
    labeler = QEReferenceLabeler(qe_params, **qe)
    backend = MACEFineTuningBackend(**payload.get("mace_backend", {}))
    runner_values = dict(payload.get("path_runner", {}))
    neb_config = NEBRunConfig(**runner_values.pop("neb_config", {}))
    runner = MACEPathRunner(neb_config=neb_config, **runner_values)
    campaign = ActiveLearningCampaign(config, backend, labeler, path_runner=runner)
    paths = []
    for item in payload.get("paths", []):
        initial = read(config_path.parent / item["initial"])
        final = read(config_path.parent / item["final"])
        if not isinstance(initial, Atoms) or not isinstance(final, Atoms):
            raise ValueError("campaign endpoint files must contain one structure each")
        paths.append(CampaignPath(item["path_id"], initial, final))
    return campaign, paths


def _dataset_command(args: argparse.Namespace) -> int:
    frames = load_dataset(args.dataset, validate=False)
    if args.dataset_command == "validate":
        summary = validate_dataset(frames)
        _print({"valid": True, **summary.__dict__})
    elif args.dataset_command == "summarize":
        _print(summarize_dataset(frames).__dict__)
    else:
        split = split_dataset_by_group(
            frames,
            args.output_dir,
            group_key=args.group_key,
            ratios=tuple(args.ratios),
            seed=args.seed,
        )
        _print(
            {
                "train": split.train_file,
                "valid": split.valid_file,
                "test": split.test_file,
                "manifest": split.manifest_path,
            }
        )
    return 0


def _mlip_command(args: argparse.Namespace) -> int:
    backend = MACEFineTuningBackend(
        executable=args.executable,
        evaluation_executable=getattr(
            args, "evaluation_executable", "mace_eval_configs"
        ),
        probe_accelerator=getattr(args, "probe_accelerator", False),
    )
    if args.mlip_command == "check":
        environment = backend.validate_environment(FineTuningProtocol(args.protocol))
        _print(environment.__dict__)
    elif args.mlip_command == "finetune":
        artifact = backend.train(
            load_dataset_split(args.split_dir),
            _fine_tuning(_json(args.config)),
            Path(args.output_dir),
        )
        _print({"model": artifact.model_path, "sha256": artifact.checksum})
    else:
        evaluation = backend.evaluate(
            _model_artifact(Path(args.model_manifest)),
            Path(args.dataset),
            Path(args.output_dir),
        )
        _print(
            {
                "metrics": evaluation.metrics.__dict__,
                "n_configurations": evaluation.n_configurations,
                "n_failures": evaluation.n_failures,
            }
        )
    return 0


def _campaign_command(args: argparse.Namespace) -> int:
    if args.campaign_command == "status":
        _print(CampaignStateStore(args.campaign_dir).load().__dict__)
        return 0
    campaign, paths = _campaign_components(Path(args.config))
    if args.campaign_command == "init":
        result = campaign.initialize(paths)
    elif args.campaign_command == "run":
        result = campaign.run(paths or None)
    elif args.campaign_command == "resume":
        result = campaign.resume(force=args.force)
    else:
        _print(campaign.validate_final())
        return 0
    _print(result.state.__dict__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser without importing optional training frameworks."""
    parser = argparse.ArgumentParser(prog="NEBwalk")
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    for name in ("validate", "summarize"):
        command = dataset_commands.add_parser(name)
        command.add_argument("dataset")
    split = dataset_commands.add_parser("split")
    split.add_argument("dataset")
    split.add_argument("output_dir")
    split.add_argument("--group-key", default="path_id")
    split.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    split.add_argument("--seed", type=int, default=0)

    mlip = commands.add_parser("mlip")
    mlip_commands = mlip.add_subparsers(dest="mlip_command", required=True)
    check = mlip_commands.add_parser("check")
    check.add_argument("--executable", default="mace_run_train")
    check.add_argument(
        "--protocol",
        choices=[item.value for item in FineTuningProtocol],
        default="naive",
    )
    check.add_argument("--probe-accelerator", action="store_true")
    finetune = mlip_commands.add_parser("finetune")
    finetune.add_argument("split_dir")
    finetune.add_argument("config")
    finetune.add_argument("output_dir")
    finetune.add_argument("--executable", default="mace_run_train")
    finetune.set_defaults(evaluation_executable="mace_eval_configs")
    evaluate = mlip_commands.add_parser("evaluate")
    evaluate.add_argument("model_manifest")
    evaluate.add_argument("dataset")
    evaluate.add_argument("output_dir")
    evaluate.add_argument("--executable", default="mace_run_train")
    evaluate.add_argument("--evaluation-executable", default="mace_eval_configs")

    campaign = commands.add_parser("campaign")
    campaign_commands = campaign.add_subparsers(dest="campaign_command", required=True)
    for name in ("init", "run", "validate-final"):
        command = campaign_commands.add_parser(name)
        command.add_argument("config")
    resume = campaign_commands.add_parser("resume")
    resume.add_argument("config")
    resume.add_argument("--force", action="store_true")
    status = campaign_commands.add_parser("status")
    status.add_argument("campaign_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and convert operational failures into useful exit status 1."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset":
            return _dataset_command(args)
        if args.command == "mlip":
            return _mlip_command(args)
        return _campaign_command(args)
    except Exception as exc:
        print(f"NEBwalk: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
