from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from intentbench.ie4 import (
    IE4PilotReport,
    RunSummary,
    comparison_evidence,
    render_markdown,
)
from intentbench.reporting import BootstrapComparisonArtifact
from intentbench.schemas import ModelRole
from intentbench.verdict import Verdict, decide


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_report() -> IE4PilotReport:
    path = repository_root() / "experiments/ie4-pilot/report.json"
    return IE4PilotReport.model_validate_json(path.read_text(encoding="utf-8"))


def test_generated_report_has_one_primary_authority_and_reproducible_markdown() -> None:
    report = load_report()
    authorities = [item for item in report.comparisons if item.verdict_authority]
    assert [item.role for item in authorities] == [ModelRole.PRIMARY_DECISION]
    assert report.global_verdict == authorities[0].verdict
    assert report.global_verdict.verdict is Verdict.INCONCLUSIVE
    assert report.global_verdict.reasons == ["budget_confounded"]
    assert report.pilot_status == "awaiting_semantic_audit"
    assert report.semantic_audit.affects_verdict is False
    assert len(report.runs) == 9

    markdown_path = repository_root() / "experiments/ie4-pilot/report.md"
    assert render_markdown(report) == markdown_path.read_text(encoding="utf-8")


def test_generated_comparisons_reproduce_their_frozen_verdict_inputs() -> None:
    report = load_report()
    runs = {run.run_id: run for run in report.runs}
    root = repository_root()
    for summary in report.comparisons:
        bootstrap = BootstrapComparisonArtifact.model_validate_json(
            (root / summary.artifacts["bootstrap"].path).read_text(encoding="utf-8")
        )
        evidence = comparison_evidence(
            comparison=bootstrap,
            budget=summary.budget,
            b2b=runs[summary.left_run_id],
            b3=runs[summary.right_run_id],
        )
        assert decide(evidence) == summary.verdict


def test_run_summary_rejects_claimed_complete_latency_without_percentiles() -> None:
    primary = next(run for run in load_report().runs if run.run_id == "primary_decision/B3")
    payload = primary.model_dump(mode="json")
    payload["latency_p95_ms"] = None
    with pytest.raises(ValidationError, match="latency completeness"):
        RunSummary.model_validate(payload)
