"""Pre-registered, mutually exclusive B3-vs-B2b verdict."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, model_validator

from intentbench.schemas import StrictModel


class Verdict(str, Enum):
    PROMISING = "promising"
    INCONCLUSIVE = "inconclusive"
    NEGATIVE = "negative"


class MetricInterval(StrictModel):
    point: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_order(self) -> MetricInterval:
        if not all(math.isfinite(value) for value in (self.point, self.lower, self.upper)):
            raise ValueError("metric interval values must be finite")
        if self.lower > self.upper:
            raise ValueError("metric interval must satisfy lower <= upper")
        return self


class ComparisonEvidence(StrictModel):
    run_integrity_valid: bool = True
    contract_aligned: bool = True
    required_cis_estimable: bool = True
    hem: MetricInterval
    id_intent_macro_f1: MetricInterval
    near_oos_recall: MetricInterval
    no_intent_recall: MetricInterval
    b2b_context_distractor_error_count: int = Field(ge=0)
    b3_context_distractor_error_count: int = Field(ge=0)
    b2b_schema_invalid_count: int = Field(ge=0)
    b3_schema_invalid_count: int = Field(ge=0)
    b2b_total_tokens: int | None = Field(default=None, ge=0)
    b3_total_tokens: int | None = Field(default=None, ge=0)
    usage_complete: bool = True
    latency_complete: bool = True
    cost_complete: bool = True


class VerdictResult(StrictModel):
    valid_comparison: bool
    verdict: Verdict | None
    reasons: list[str]
    token_difference_rate: float | None


def token_difference_rate(b2b_total: int, b3_total: int) -> float:
    return abs(b3_total - b2b_total) / max(b2b_total, 1)


def decide(evidence: ComparisonEvidence) -> VerdictResult:
    if not evidence.run_integrity_valid:
        return VerdictResult(
            valid_comparison=False,
            verdict=None,
            reasons=["invalid_run"],
            token_difference_rate=None,
        )

    token_difference: float | None = None
    contract_reasons: list[str] = []
    if evidence.b2b_total_tokens is None or evidence.b3_total_tokens is None:
        contract_reasons.append("missing_token_totals")
    else:
        token_difference = token_difference_rate(
            evidence.b2b_total_tokens, evidence.b3_total_tokens
        )
        if token_difference > 0.10:
            contract_reasons.append("budget_confounded")
    if not evidence.usage_complete:
        contract_reasons.append("incomplete_usage_metadata")
    if not evidence.cost_complete:
        contract_reasons.append("incomplete_cost")
    if not evidence.latency_complete:
        contract_reasons.append("incomplete_latency")
    if not evidence.required_cis_estimable:
        contract_reasons.append("required_ci_unavailable")
    if not evidence.contract_aligned:
        contract_reasons.append("comparison_contract_mismatch")
    if contract_reasons:
        return VerdictResult(
            valid_comparison=True,
            verdict=Verdict.INCONCLUSIVE,
            reasons=contract_reasons,
            token_difference_rate=token_difference,
        )

    negative_reasons: list[str] = []
    if evidence.hem.upper <= 0:
        negative_reasons.append("hem_not_improved")
    if evidence.id_intent_macro_f1.upper < -0.03:
        negative_reasons.append("id_intent_regressed")
    if evidence.near_oos_recall.upper < -0.05:
        negative_reasons.append("near_oos_regressed")
    if evidence.no_intent_recall.upper < -0.05:
        negative_reasons.append("no_intent_regressed")
    if evidence.b3_context_distractor_error_count > evidence.b2b_context_distractor_error_count:
        negative_reasons.append("more_context_distractor_errors")
    if evidence.b3_schema_invalid_count > evidence.b2b_schema_invalid_count:
        negative_reasons.append("more_schema_invalid_predictions")
    if negative_reasons:
        return VerdictResult(
            valid_comparison=True,
            verdict=Verdict.NEGATIVE,
            reasons=negative_reasons,
            token_difference_rate=token_difference,
        )

    inconclusive_reasons: list[str] = []
    if evidence.hem.lower <= 0:
        inconclusive_reasons.append("hem_ci_crosses_zero")
    if evidence.id_intent_macro_f1.lower < -0.03:
        inconclusive_reasons.append("id_intent_noninferiority_unproven")
    if evidence.near_oos_recall.lower < -0.05:
        inconclusive_reasons.append("near_oos_noninferiority_unproven")
    if evidence.no_intent_recall.lower < -0.05:
        inconclusive_reasons.append("no_intent_noninferiority_unproven")
    if evidence.near_oos_recall.point < 0:
        inconclusive_reasons.append("near_oos_point_regressed")
    if inconclusive_reasons:
        return VerdictResult(
            valid_comparison=True,
            verdict=Verdict.INCONCLUSIVE,
            reasons=inconclusive_reasons,
            token_difference_rate=token_difference,
        )

    return VerdictResult(
        valid_comparison=True,
        verdict=Verdict.PROMISING,
        reasons=["all_pre_registered_gates_passed"],
        token_difference_rate=token_difference,
    )
