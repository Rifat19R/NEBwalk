"""Deterministic JSON, Markdown, and CSV campaign reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .campaign_state import CampaignStateStore


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_iteration_report(
    campaign_dir: str | Path, iteration: int
) -> tuple[Path, Path]:
    """Collect one immutable iteration's selection, labels, metrics, and models."""
    root = Path(campaign_dir)
    directory = root / f"iteration_{iteration:03d}"
    candidates = _read_json(directory / "selected_candidates.json", {})
    metrics = _read_json(directory / "iteration_metrics.json", {})
    labels = _read_json(directory / "labels" / "label_manifest.json", {})
    stopping = _read_json(directory / "stopping_decision.json", None)
    models = [
        _read_json(path, {})
        for path in sorted((directory / "models").glob("member_*/model_manifest.json"))
    ]
    evaluations = [
        _read_json(path, {})
        for path in sorted(
            (directory / "evaluation").glob("member_*/training_metrics.json")
        )
    ]
    report = {
        "schema": "nebwalk.iteration_report.v1",
        "iteration": iteration,
        "metrics": metrics,
        "selection": candidates,
        "labeling": labels,
        "models": models,
        "model_evaluations": evaluations,
        "stopping_decision": stopping,
    }
    json_path = directory / "iteration_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected_count = candidates.get("new_unique_count", 0)
    success_count = len(labels.get("labels", []))
    failure_count = len(labels.get("failures", []))
    stopping_reason = (
        stopping.get("reason", "not evaluated") if stopping else "not evaluated"
    )
    markdown = (
        f"# Iteration {iteration}\n\n"
        f"- New unique candidates: {selected_count}\n"
        f"- Successful reference labels: {success_count}\n"
        f"- Failed reference labels: {failure_count}\n"
        f"- Trained models: {len(models)}\n"
        f"- Stopping decision: {stopping_reason}\n\n"
        "This report records workflow outputs and does not claim calibrated "
        "uncertainty or publication-grade accuracy.\n"
    )
    markdown_path = directory / "iteration_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    csv_path = directory / "committee_uncertainty.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path_id",
                "image_index",
                "energy_eV",
                "uncertainty",
                "score",
                "selected",
                "reasons",
            ),
        )
        writer.writeheader()
        for profile in candidates.get("profiles", []):
            for item in profile.get("diagnostics", []):
                writer.writerow(
                    {
                        "path_id": profile["path_id"],
                        "image_index": item["image_index"],
                        "energy_eV": item["energy_eV"],
                        "uncertainty": item["uncertainty"],
                        "score": item["score"],
                        "selected": item["selected"],
                        "reasons": "+".join(item["reasons"]),
                    }
                )
    return json_path, markdown_path


def write_campaign_summary(campaign_dir: str | Path) -> tuple[Path, Path]:
    """Write campaign-wide barrier history, stopping status, and provenance links."""
    root = Path(campaign_dir)
    state = CampaignStateStore(root).load()
    iteration_reports = []
    barrier_history = []
    for directory in sorted(root.glob("iteration_*")):
        if not directory.is_dir():
            continue
        iteration = int(directory.name.rsplit("_", 1)[1])
        report_path, _ = write_iteration_report(root, iteration)
        report = _read_json(report_path, {})
        iteration_reports.append(str(report_path.relative_to(root)))
        for profile in report.get("selection", {}).get("profiles", []):
            barrier_history.append(
                {
                    "iteration": iteration,
                    "path_id": profile["path_id"],
                    "mlip_barrier_eV": profile["barrier_eV"],
                }
            )
    final_validation = _read_json(
        root / "final_validation" / "final_validation.json", None
    )
    summary = {
        "schema": "nebwalk.campaign_summary.v1",
        "campaign_id": state.campaign_id,
        "stage": state.stage.value,
        "iteration": state.iteration,
        "active_model_id": state.active_model_id,
        "stopping_reason": state.stopping_reason,
        "barrier_history": barrier_history,
        "iteration_reports": iteration_reports,
        "final_validation": final_validation,
    }
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)
    json_path = reports_dir / "campaign_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = reports_dir / "campaign_summary.md"
    markdown_path.write_text(
        "# Active-learning campaign summary\n\n"
        f"- Campaign: `{state.campaign_id}`\n"
        f"- State: `{state.stage.value}`\n"
        f"- Iteration: {state.iteration}\n"
        f"- Stopping reason: `{state.stopping_reason}`\n"
        f"- Active model: `{state.active_model_id}`\n"
        f"- Recorded pathway barriers: {len(barrier_history)}\n\n"
        "Committee disagreement is an uncertainty proxy, not calibrated "
        "uncertainty. Final DFT claims require the separately recorded QE "
        "validation stage.\n",
        encoding="utf-8",
    )
    return json_path, markdown_path


__all__ = ["write_campaign_summary", "write_iteration_report"]
