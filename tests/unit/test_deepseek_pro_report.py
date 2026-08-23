from __future__ import annotations

import json
from pathlib import Path

import pytest

from intentbench.deepseek_pro import (
    DeepSeekProRobustnessReport,
    render_deepseek_pro_markdown,
)
from intentbench.freeze import sha256_file
from intentbench.verdict import Verdict


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_deepseek_pro_report_records_frozen_contract_outcome() -> None:
    root = repository_root()
    report_path = root / "experiments/ie3-deepseek-pro-robustness/report.json"
    report = DeepSeekProRobustnessReport.model_validate_json(report_path.read_text())

    assert report.global_verdict_authority is False
    assert report.global_verdict_effect == "none"
    assert report.conclusion == "no_rescue_under_frozen_contract"
    assert report.comparison.diagnostic_verdict.verdict is Verdict.INCONCLUSIVE
    assert report.comparison.diagnostic_verdict.reasons == [
        "budget_confounded",
        "incomplete_cost",
    ]

    b2b, b3 = report.runs
    assert (b2b.success_count, b2b.failure_count) == (31, 81)
    assert (b3.success_count, b3.failure_count) == (8, 104)
    assert b2b.incomplete_call_count == b2b.incomplete_calls_within_one_token_of_cap == 81
    assert b3.incomplete_call_count == b3.incomplete_calls_within_one_token_of_cap == 104
    assert b3.formed_intention_success_count == 15
    assert b2b.hierarchical_exact_match == pytest.approx(31 / 112)
    assert b3.hierarchical_exact_match == pytest.approx(8 / 112)

    audit = report.semantic_audit_coverage
    assert (audit.formed_intention_success_count, audit.formed_intention_failure_count) == (4, 20)
    assert audit.successful_case_ids == [
        "kir-pilot-v2-0061",
        "kir-pilot-v2-0073",
        "kir-pilot-v2-0096",
        "kir-pilot-v2-0130",
    ]
    assert audit.decision == "do_not_add_pro_audit"


def test_deepseek_pro_report_is_rendered_and_hash_bound() -> None:
    root = repository_root()
    output_dir = root / "experiments/ie3-deepseek-pro-robustness"
    report = DeepSeekProRobustnessReport.model_validate_json(
        (output_dir / "report.json").read_text()
    )

    assert render_deepseek_pro_markdown(report) == (output_dir / "README.md").read_text()
    for digest in report.source_artifacts.values():
        assert sha256_file(root / digest.path) == digest.sha256
    for run in report.runs:
        for digest in run.artifacts.values():
            assert sha256_file(root / digest.path) == digest.sha256
    assert sha256_file(root / report.comparison.artifact.path) == report.comparison.artifact.sha256

    payload = json.loads((output_dir / "report.json").read_text())
    assert payload["flash_reference"][0]["pro_minus_flash_hem"] == pytest.approx(
        -0.39285714285714285
    )
