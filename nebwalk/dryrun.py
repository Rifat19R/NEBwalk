"""Deterministic no-external-software campaign used for orchestration sanity tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase import Atoms

from .campaign import (
    ActiveLearningCampaign,
    ActiveLearningConfig,
    CampaignPath,
    CampaignPathResult,
    StoppingCriteriaConfig,
)
from .candidate_selection import CandidateSelectionConfig
from .datasets import compute_structure_hash, write_dataset
from .labeling import LabelingResult, ReferenceLabel
from .mlip import (
    FineTuningConfig,
    ModelEvaluation,
    ModelMetrics,
    TrainedModelArtifact,
    TrainingRun,
)


class DryRunReferenceLabeler:
    """Analytical harmonic reference labeler for workflow tests only."""

    def label(self, structures, output_dir, metadata=None):
        output_dir = Path(output_dir)
        frames = []
        labels = []
        for index, source in enumerate(structures):
            atoms = source.copy()
            structure_hash = compute_structure_hash(atoms)
            energy = float(np.sum(atoms.positions**2))
            atoms.info.update(metadata or {})
            atoms.info.update(
                {
                    "REF_energy": energy,
                    "config_type": "dry_run",
                    "campaign_id": atoms.info.get("campaign_id", "dry-run"),
                    "iteration": int(atoms.info.get("iteration", 0)),
                    "path_id": str(atoms.info["path_id"]),
                    "image_index": int(atoms.info["image_index"]),
                    "selection_reason": str(atoms.info["selection_reason"]),
                    "calculator_name": "analytical-dry-run",
                    "dft_settings_hash": "dry-run-reference-v1",
                    "structure_hash": structure_hash,
                }
            )
            atoms.arrays["REF_forces"] = -2.0 * atoms.positions
            frames.append(atoms)
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
                    "qe_settings_hash": "dry-run-reference-v1",
                    "labels": [
                        {
                            **asdict(label),
                            "raw_work_directory": str(label.raw_work_directory),
                        }
                        for label in labels
                    ],
                    "failures": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return LabelingResult(
            tuple(labels), (), artifact, output_dir, "dry-run-reference-v1"
        )


class DryRunTrainer:
    """Deterministic artifact-producing trainer; it does not train an ML model."""

    def validate_environment(self, protocol=None):
        raise NotImplementedError("dry-run trainer has no external environment")

    def train(self, dataset, config, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = output_dir / f"{config.name}.model"
        model.write_bytes(f"dry-model-seed-{config.seed}".encode())
        checksum = hashlib.sha256(model.read_bytes()).hexdigest()
        command = ("dry_run_train", "--seed", str(config.seed))
        manifest = {
            "model_path": str(model),
            "sha256": checksum,
            "command": list(command),
        }
        (output_dir / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        stdout = output_dir / "stdout.log"
        stderr = output_dir / "stderr.log"
        stdout.write_text("dry run complete", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        return TrainedModelArtifact(
            model,
            checksum,
            output_dir / "model_manifest.json",
            TrainingRun(command, 0, "succeeded", stdout, stderr),
            backend="dry-run",
        )

    def evaluate(self, model, dataset, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        member = int(model.model_path.stem.rsplit("_m", 1)[-1])
        return ModelEvaluation(
            ModelMetrics(force_component_rmse=0.1 + member * 0.01),
            1,
            0,
            output_dir,
        )


class DryRunPathRunner:
    """Three-image deterministic path runner with one post-bootstrap geometry."""

    def __call__(self, paths, active_model, output_dir):
        shift = 0.0 if active_model is None else 0.02
        return [
            CampaignPathResult(
                path.path_id,
                tuple(
                    Atoms("H", positions=[[x + shift, index * 2.0, 0.0]])
                    for x in (0.0, 0.5, 1.0)
                ),
                (0.0, 1.0, 0.0),
                1.0,
            )
            for index, path in enumerate(paths)
        ]


def run_dry_campaign(output_dir: str | Path):
    """Run bootstrap and retraining to an explicit no-new-candidate stop."""
    output = Path(output_dir)
    config = ActiveLearningConfig(
        campaign_id="nebwalk-dry-run",
        campaign_dir=output,
        fine_tuning=FineTuningConfig(name="dry", foundation_model="dry-foundation"),
        candidate_selection=CandidateSelectionConfig(
            n_select=3,
            peak_neighbors=1,
            diversity=False,
        ),
        stopping=StoppingCriteriaConfig(maximum_iterations=4),
        split_ratios=(1 / 3, 1 / 3, 1 / 3),
    )
    paths = [
        CampaignPath(
            f"path-{index}",
            Atoms("H", positions=[[0.0, index * 2.0, 0.0]]),
            Atoms("H", positions=[[1.0, index * 2.0, 0.0]]),
        )
        for index in range(3)
    ]
    campaign = ActiveLearningCampaign(
        config,
        DryRunTrainer(),
        DryRunReferenceLabeler(),
        path_runner=DryRunPathRunner(),
    )
    return campaign.run(paths)


__all__ = [
    "DryRunPathRunner",
    "DryRunReferenceLabeler",
    "DryRunTrainer",
    "run_dry_campaign",
]
