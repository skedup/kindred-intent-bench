from __future__ import annotations

from pathlib import Path

import pytest

from intentbench.budget_attribution import (
    BudgetAttributionReport,
    build_budget_attribution_report,
    render_markdown,
)
from intentbench.freeze import sha256_file


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_report() -> BudgetAttributionReport:
    path = repository_root() / "experiments/ie45-budget-attribution/report.json"
    return BudgetAttributionReport.model_validate_json(path.read_text(encoding="utf-8"))


def test_primary_budget_attribution_reproduces_frozen_arm_and_stage_totals() -> None:
    report = load_report()
    b2b, b3 = report.arms

    assert report.boundary.provider_calls == 0
    assert report.boundary.changes_ie4_verdict is False
    assert report.boundary.supports_component_level_token_causality is False
    assert len(report.cases) == 112

    assert b2b.total_usage.model_dump() == {
        "input_tokens": 704_427,
        "output_tokens": 17_827,
        "total_tokens": 722_254,
        "retry_input_tokens": 0,
        "retry_output_tokens": 0,
    }
    assert b3.total_usage.model_dump() == {
        "input_tokens": 555_139,
        "output_tokens": 18_158,
        "total_tokens": 573_297,
        "retry_input_tokens": 0,
        "retry_output_tokens": 0,
    }
    assert b2b.incremental_provider_call_count == 112
    assert b3.incremental_provider_call_count == 224
    assert [stage.total_usage.total_tokens for stage in b2b.stages] == [372_827, 349_427]
    assert [stage.total_usage.total_tokens for stage in b3.stages] == [195_569, 377_728]

    gap = report.gap
    assert gap.registered_status == "budget-confounded"
    assert gap.absolute_total_token_difference_rate == pytest.approx(0.20623907932666347)
    assert gap.b3_minus_b2b_tokens.model_dump() == {
        "input_tokens": -149_288,
        "output_tokens": 331,
        "retry_input_tokens": 0,
        "retry_output_tokens": 0,
        "total_tokens": -148_957,
    }
    assert [pair.b3_minus_b2b_tokens.total_tokens for pair in gap.stage_pairs] == [
        -177_258,
        28_301,
    ]
    assert gap.b3_minus_b2b_cost_usd == pytest.approx(-0.11072475)
    assert gap.paired_latency_difference_ms.mean == pytest.approx(-247.70997505610077)
    assert gap.paired_latency_difference_ms.p50 == pytest.approx(-171.68862489052117)
    assert gap.paired_latency_difference_ms.p95 == pytest.approx(295.08631709031755)


def test_budget_attribution_is_rendered_and_hash_bound() -> None:
    root = repository_root()
    report = load_report()

    assert render_markdown(report) == (
        root / "experiments/ie45-budget-attribution/report.md"
    ).read_text(encoding="utf-8")
    for digest in report.source_artifacts.values():
        assert sha256_file(root / digest.path) == digest.sha256


def test_budget_attribution_regeneration_is_byte_identical(tmp_path: Path) -> None:
    root = repository_root()
    output_dir = tmp_path / "ie45-budget-attribution"
    arguments = {
        "matrix_path": root / "experiments/ie3-openai-recovery/seven-run-matrix.json",
        "experiment_path": root / "configs/kir-pilot-v2-ie3-experiment.yaml",
        "output_dir": output_dir,
        "repository_root": root,
    }

    created = build_budget_attribution_report(**arguments)
    unchanged = build_budget_attribution_report(**arguments)

    assert created["status"] == "created"
    assert created["provider_calls"] == 0
    assert unchanged["status"] == "unchanged"
    assert (output_dir / "report.json").read_bytes() == (
        root / "experiments/ie45-budget-attribution/report.json"
    ).read_bytes()
    assert (output_dir / "report.md").read_bytes() == (
        root / "experiments/ie45-budget-attribution/report.md"
    ).read_bytes()
