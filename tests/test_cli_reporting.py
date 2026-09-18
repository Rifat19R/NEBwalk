"""CLI, reporting, and zero-external-software dry-run tests."""

from __future__ import annotations

import json
import sys

from NEBwalk.cli import build_parser, main
from NEBwalk.datasets import compute_structure_hash, write_dataset
from NEBwalk.dryrun import run_dry_campaign
from NEBwalk.reporting import write_campaign_summary, write_iteration_report
from tests.test_datasets import _frame


def test_cli_parser_does_not_import_torch_or_mace(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "mace", raising=False)

    parser = build_parser()
    args = parser.parse_args(["dataset", "validate", "data.extxyz"])

    assert args.command == "dataset"
    assert "torch" not in sys.modules
    assert "mace" not in sys.modules


def test_dataset_validate_and_summarize_cli(tmp_path, capsys):
    dataset = write_dataset(tmp_path / "data.extxyz", [_frame()]).path

    assert main(["dataset", "validate", str(dataset)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["n_configurations"] == 1

    assert main(["dataset", "summarize", str(dataset)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_atoms"] == 2


def test_cli_operational_error_returns_nonzero(tmp_path, capsys):
    code = main(["dataset", "validate", str(tmp_path / "missing.extxyz")])

    assert code == 1
    assert "FileNotFoundError" in capsys.readouterr().err


def test_dry_campaign_runs_end_to_end_and_writes_reports(tmp_path):
    result = run_dry_campaign(tmp_path / "dry")

    assert result.state.stage.value == "converged"
    assert result.state.stopping_reason == "no_new_unique_configurations"
    summary_json, summary_md = write_campaign_summary(result.campaign_dir)
    assert summary_json.is_file()
    assert summary_md.is_file()
    summary = json.loads(summary_json.read_text())
    assert summary["active_model_id"]
    assert len(summary["barrier_history"]) == 9


def test_iteration_reporting_exports_uncertainty_csv(tmp_path):
    result = run_dry_campaign(tmp_path / "report")

    report_json, report_md = write_iteration_report(result.campaign_dir, 0)

    assert report_json.is_file()
    assert report_md.is_file()
    assert (
        result.campaign_dir / "iteration_000" / "committee_uncertainty.csv"
    ).is_file()
    payload = json.loads(report_json.read_text())
    assert payload["schema"] == "nebwalk.iteration_report.v1"


def test_structure_hash_helper_remains_deterministic_in_cli_fixture():
    frame = _frame()
    assert compute_structure_hash(frame) == frame.info["structure_hash"]
