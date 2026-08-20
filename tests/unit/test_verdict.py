from __future__ import annotations

import pytest

from intentbench.verdict import (
    ComparisonEvidence,
    MetricInterval,
    Verdict,
    decide,
    token_difference_rate,
)


def evidence(**updates: object) -> ComparisonEvidence:
    payload: dict[str, object] = {
        "hem": MetricInterval(point=0.05, lower=0.01, upper=0.09),
        "id_intent_macro_f1": MetricInterval(point=0.0, lower=-0.03, upper=0.03),
        "near_oos_recall": MetricInterval(point=0.0, lower=-0.05, upper=0.05),
        "no_intent_recall": MetricInterval(point=0.0, lower=-0.05, upper=0.05),
        "b2b_context_distractor_error_count": 1,
        "b3_context_distractor_error_count": 1,
        "b2b_schema_invalid_count": 0,
        "b3_schema_invalid_count": 0,
        "b2b_total_tokens": 100,
        "b3_total_tokens": 110,
    }
    payload.update(updates)
    return ComparisonEvidence.model_validate(payload)


def test_promising_accepts_registered_inclusive_boundaries() -> None:
    result = decide(evidence())
    assert result.verdict is Verdict.PROMISING
    assert result.token_difference_rate == pytest.approx(0.10)


def test_invalid_run_has_no_tri_state_verdict() -> None:
    result = decide(evidence(run_integrity_valid=False))
    assert result.valid_comparison is False
    assert result.verdict is None


def test_contract_failure_precedes_negative() -> None:
    result = decide(
        evidence(
            b3_total_tokens=111,
            hem=MetricInterval(point=-0.1, lower=-0.2, upper=0.0),
        )
    )
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "budget_confounded" in result.reasons


def test_negative_uses_exact_upper_and_count_comparisons() -> None:
    assert (
        decide(evidence(hem=MetricInterval(point=-0.01, lower=-0.1, upper=0.0))).verdict
        is Verdict.NEGATIVE
    )
    assert decide(evidence(b3_context_distractor_error_count=2)).verdict is Verdict.NEGATIVE
    assert decide(evidence(b3_schema_invalid_count=1)).verdict is Verdict.NEGATIVE


def test_ci_crossing_noninferiority_gate_is_inconclusive() -> None:
    result = decide(
        evidence(id_intent_macro_f1=MetricInterval(point=-0.03, lower=-0.031, upper=0.02))
    )
    assert result.verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize(
    ("field", "interval"),
    [
        ("hem", MetricInterval(point=0.01, lower=0.0, upper=0.02)),
        (
            "near_oos_recall",
            MetricInterval(point=0.0, lower=-0.051, upper=0.01),
        ),
        (
            "no_intent_recall",
            MetricInterval(point=0.0, lower=-0.051, upper=0.01),
        ),
    ],
)
def test_each_ci_lower_gate_can_make_result_inconclusive(
    field: str, interval: MetricInterval
) -> None:
    assert decide(evidence(**{field: interval})).verdict is Verdict.INCONCLUSIVE


def test_missing_usage_metadata_is_inconclusive() -> None:
    result = decide(evidence(usage_complete=False))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "incomplete_usage_metadata" in result.reasons


def test_negative_strict_upper_boundaries_are_not_over_applied() -> None:
    id_result = decide(
        evidence(
            id_intent_macro_f1=MetricInterval(point=-0.03, lower=-0.03, upper=-0.03),
        )
    )
    assert id_result.verdict is Verdict.PROMISING
    near_result = decide(
        evidence(near_oos_recall=MetricInterval(point=-0.05, lower=-0.05, upper=-0.05))
    )
    assert near_result.verdict is Verdict.INCONCLUSIVE
    no_intent_result = decide(
        evidence(no_intent_recall=MetricInterval(point=-0.05, lower=-0.05, upper=-0.05))
    )
    assert no_intent_result.verdict is Verdict.PROMISING


def test_token_formula_counts_zero_denominator_exactly() -> None:
    assert token_difference_rate(0, 5) == 5.0


def test_percentile_interval_may_exclude_point_estimate() -> None:
    interval = MetricInterval(point=0.2, lower=-0.1, upper=0.1)
    assert interval.point == 0.2
