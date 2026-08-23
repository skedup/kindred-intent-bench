"""Deterministic IE4.5A budget attribution from frozen primary call artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, ValidationError, model_validator

from intentbench.b2 import IE3LLMExperimentConfig
from intentbench.b2a import PricingConfig
from intentbench.freeze import ArtifactDigest, sha256_file
from intentbench.matrix import MatrixCell, SevenRunMatrixArtifact
from intentbench.schemas import ModelRole, PredictionStatus, RunManifest, StrictModel, TokenUsage
from intentbench.two_stage import LLMCallRecord

REPORT_FILENAME = "report.json"
REPORT_MARKDOWN_FILENAME = "report.md"
PRIMARY_ROLE: Literal["primary_decision"] = "primary_decision"
PRIMARY_MODEL = "google_generative_language/gemini-3.6-flash"
REQUESTED_MODEL = "gemini-3.6-flash"
Arm = Literal["B2b", "B3"]
Stage = Literal["call_1_one_stage", "call_2_verifier", "stage_a", "stage_b"]
B2bStage = Literal["call_1_one_stage", "call_2_verifier"]
B3Stage = Literal["stage_a", "stage_b"]
ARMS: tuple[Arm, Arm] = ("B2b", "B3")


class BudgetAttributionError(ValueError):
    """Frozen IE4.5A sources or deterministic outputs violate the contract."""


class DistributionSummary(StrictModel):
    count: int = Field(ge=1)
    minimum: float = Field(ge=0)
    mean: float = Field(ge=0)
    p50: float = Field(ge=0)
    p95: float = Field(ge=0)
    maximum: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> DistributionSummary:
        if not self.minimum <= self.p50 <= self.p95 <= self.maximum:
            raise ValueError("distribution percentiles must be ordered")
        return self


class SignedDistributionSummary(StrictModel):
    count: int = Field(ge=1)
    minimum: float
    mean: float
    p50: float
    p95: float
    maximum: float

    @model_validator(mode="after")
    def validate_order(self) -> SignedDistributionSummary:
        if not self.minimum <= self.p50 <= self.p95 <= self.maximum:
            raise ValueError("signed distribution percentiles must be ordered")
        return self


class CallBudget(StrictModel):
    stage: Stage
    call_index: Literal[1, 2]
    status: PredictionStatus
    error_type: str | None
    usage: TokenUsage
    latency_ms: float = Field(ge=0)


class CaseArmBudget(StrictModel):
    arm: Arm
    calls: list[CallBudget] = Field(min_length=2, max_length=2)
    total_usage: TokenUsage
    estimated_cost_usd: float = Field(ge=0)
    end_to_end_latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_calls(self) -> CaseArmBudget:
        if [call.call_index for call in self.calls] != [1, 2]:
            raise ValueError("case arm calls must use indexes one and two")
        if any(call.stage not in _expected_stages(self.arm) for call in self.calls):
            raise ValueError("case arm contains a stage from another arm")
        expected = _sum_usage([call.usage for call in self.calls])
        if self.total_usage != expected:
            raise ValueError("case total usage differs from its call sum")
        if self.end_to_end_latency_ms != sum(call.latency_ms for call in self.calls):
            raise ValueError("case latency differs from its call sum")
        return self


class SignedTokenDifference(StrictModel):
    input_tokens: int
    output_tokens: int
    retry_input_tokens: int
    retry_output_tokens: int
    total_tokens: int

    @model_validator(mode="after")
    def validate_total(self) -> SignedTokenDifference:
        if self.total_tokens != (
            self.input_tokens
            + self.output_tokens
            + self.retry_input_tokens
            + self.retry_output_tokens
        ):
            raise ValueError("signed token difference fields do not sum to total")
        return self


class CaseBudgetAttribution(StrictModel):
    case_id: str
    b2b: CaseArmBudget
    b3: CaseArmBudget
    b3_minus_b2b_tokens: SignedTokenDifference
    b3_minus_b2b_cost_usd: float
    b3_minus_b2b_latency_ms: float


class StageBudgetSummary(StrictModel):
    arm: Arm
    stage: Stage
    call_index: Literal[1, 2]
    call_count: Literal[112] = 112
    success_count: int = Field(ge=0, le=112)
    failure_count: int = Field(ge=0, le=112)
    usage_complete: Literal[True] = True
    total_usage: TokenUsage
    estimated_cost_usd: float = Field(ge=0)
    tokens_per_call: DistributionSummary
    latency_ms: DistributionSummary

    @model_validator(mode="after")
    def validate_counts(self) -> StageBudgetSummary:
        if self.success_count + self.failure_count != self.call_count:
            raise ValueError("stage success/failure counts must cover every call")
        return self


class ArmBudgetSummary(StrictModel):
    arm: Arm
    model: Literal["google_generative_language/gemini-3.6-flash"] = (
        "google_generative_language/gemini-3.6-flash"
    )
    case_count: Literal[112] = 112
    logical_call_count: Literal[224] = 224
    incremental_provider_call_count: int = Field(ge=0, le=224)
    success_call_count: int = Field(ge=0, le=224)
    failure_call_count: int = Field(ge=0, le=224)
    usage_complete: Literal[True] = True
    total_usage: TokenUsage
    estimated_cost_usd: float = Field(ge=0)
    per_case_total_tokens: DistributionSummary
    per_case_end_to_end_latency_ms: DistributionSummary
    stages: list[StageBudgetSummary]

    @model_validator(mode="after")
    def validate_summary(self) -> ArmBudgetSummary:
        if self.success_call_count + self.failure_call_count != self.logical_call_count:
            raise ValueError("arm success/failure counts must cover every call")
        if [stage.stage for stage in self.stages] != list(_expected_stages(self.arm)):
            raise ValueError("arm stage summaries differ from the registered order")
        if _sum_usage([stage.total_usage for stage in self.stages]) != self.total_usage:
            raise ValueError("arm usage differs from its stage totals")
        return self


class StagePairDifference(StrictModel):
    call_index: Literal[1, 2]
    b2b_stage: Literal["call_1_one_stage", "call_2_verifier"]
    b3_stage: Literal["stage_a", "stage_b"]
    b3_minus_b2b_tokens: SignedTokenDifference
    b3_minus_b2b_cost_usd: float


class BudgetGapSummary(StrictModel):
    difference: Literal["B3_minus_B2b"] = "B3_minus_B2b"
    b3_minus_b2b_tokens: SignedTokenDifference
    absolute_total_token_difference_rate: float = Field(ge=0)
    registered_maximum_rate: float = Field(default=0.1, ge=0)
    registered_status: Literal["budget-confounded"] = "budget-confounded"
    b3_minus_b2b_cost_usd: float
    paired_latency_difference_ms: SignedDistributionSummary
    stage_pairs: list[StagePairDifference]

    @model_validator(mode="after")
    def validate_stage_pairs(self) -> BudgetGapSummary:
        if self.registered_maximum_rate != 0.1:
            raise ValueError("IE4.5A must preserve the registered 10% budget threshold")
        if [pair.call_index for pair in self.stage_pairs] != [1, 2]:
            raise ValueError("budget gap must contain both call positions in order")
        combined = _sum_signed([pair.b3_minus_b2b_tokens for pair in self.stage_pairs])
        if combined != self.b3_minus_b2b_tokens:
            raise ValueError("stage-pair differences do not sum to the arm difference")
        return self


class AttributionBoundary(StrictModel):
    provider_calls: Literal[0] = 0
    changes_ie4_verdict: Literal[False] = False
    supports_component_level_token_causality: Literal[False] = False
    registered_taxonomy_visibility: dict[Stage, Literal["full", "hidden", "not_separable"]]

    @model_validator(mode="after")
    def validate_visibility(self) -> AttributionBoundary:
        if self.registered_taxonomy_visibility != {
            "call_1_one_stage": "full",
            "call_2_verifier": "full",
            "stage_a": "hidden",
            "stage_b": "full",
        }:
            raise ValueError("registered taxonomy visibility drifted")
        return self


class BudgetAttributionReport(StrictModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["ie45a_primary_budget_attribution"] = "ie45a_primary_budget_attribution"
    experiment_kind: Literal["post_ie4_offline_diagnostic"] = "post_ie4_offline_diagnostic"
    dataset_version: str
    taxonomy_version: str
    split: Literal["test"] = "test"
    model_role: Literal["primary_decision"] = "primary_decision"
    model: Literal["google_generative_language/gemini-3.6-flash"] = (
        "google_generative_language/gemini-3.6-flash"
    )
    pricing: PricingConfig
    arms: list[ArmBudgetSummary]
    gap: BudgetGapSummary
    cases: list[CaseBudgetAttribution]
    boundary: AttributionBoundary
    source_artifacts: dict[str, ArtifactDigest]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_universe(self) -> BudgetAttributionReport:
        if [arm.arm for arm in self.arms] != ["B2b", "B3"]:
            raise ValueError("budget attribution arms must use B2b then B3")
        if len(self.cases) != 112 or len({case.case_id for case in self.cases}) != 112:
            raise ValueError("budget attribution must contain 112 unique paired cases")
        return self


def _expected_stages(arm: Arm) -> tuple[Stage, Stage]:
    if arm == "B2b":
        return "call_1_one_stage", "call_2_verifier"
    return "stage_a", "stage_b"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode()


def _relative(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    root = repository_root.resolve()
    if not resolved.is_relative_to(root):
        raise BudgetAttributionError(f"IE4.5A source is outside repository: {path}")
    return resolved.relative_to(root).as_posix()


def _source(path: Path, repository_root: Path) -> ArtifactDigest:
    return ArtifactDigest(path=_relative(path, repository_root), sha256=sha256_file(path))


def _read_model(path: Path, model: type[Any]) -> Any:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        raise BudgetAttributionError(f"invalid IE4.5A artifact: {path}") from exc


def _read_calls(path: Path) -> list[LLMCallRecord]:
    records: list[LLMCallRecord] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise BudgetAttributionError(f"{path}:{line_number}: blank call row")
            records.append(LLMCallRecord.model_validate_json(line))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        if isinstance(exc, BudgetAttributionError):
            raise
        raise BudgetAttributionError(f"invalid IE4.5A calls: {path}") from exc
    return records


def _load_config(path: Path) -> IE3LLMExperimentConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return IE3LLMExperimentConfig.model_validate(payload["llm"])
    except (OSError, UnicodeError, KeyError, TypeError, yaml.YAMLError, ValidationError) as exc:
        raise BudgetAttributionError(f"invalid IE4.5A experiment config: {path}") from exc


def _sum_usage(usages: Sequence[TokenUsage]) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        retry_input_tokens=sum(usage.retry_input_tokens for usage in usages),
        retry_output_tokens=sum(usage.retry_output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )


def _difference(left: TokenUsage, right: TokenUsage) -> SignedTokenDifference:
    """Return left minus right using the report's B3-minus-B2b sign convention."""

    return SignedTokenDifference(
        input_tokens=left.input_tokens - right.input_tokens,
        output_tokens=left.output_tokens - right.output_tokens,
        retry_input_tokens=left.retry_input_tokens - right.retry_input_tokens,
        retry_output_tokens=left.retry_output_tokens - right.retry_output_tokens,
        total_tokens=left.total_tokens - right.total_tokens,
    )


def _sum_signed(values: Sequence[SignedTokenDifference]) -> SignedTokenDifference:
    return SignedTokenDifference(
        input_tokens=sum(value.input_tokens for value in values),
        output_tokens=sum(value.output_tokens for value in values),
        retry_input_tokens=sum(value.retry_input_tokens for value in values),
        retry_output_tokens=sum(value.retry_output_tokens for value in values),
        total_tokens=sum(value.total_tokens for value in values),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise BudgetAttributionError("percentile requires values and quantile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _distribution(values: Sequence[float]) -> DistributionSummary:
    return DistributionSummary(
        count=len(values),
        minimum=min(values),
        mean=sum(values) / len(values),
        p50=_percentile(values, 0.5),
        p95=_percentile(values, 0.95),
        maximum=max(values),
    )


def _signed_distribution(values: Sequence[float]) -> SignedDistributionSummary:
    return SignedDistributionSummary(
        count=len(values),
        minimum=min(values),
        mean=sum(values) / len(values),
        p50=_percentile(values, 0.5),
        p95=_percentile(values, 0.95),
        maximum=max(values),
    )


def _cost(usage: TokenUsage, pricing: PricingConfig) -> float:
    input_tokens = usage.input_tokens + usage.retry_input_tokens
    output_tokens = usage.output_tokens + usage.retry_output_tokens
    return (
        input_tokens * pricing.input_per_million_tokens
        + output_tokens * pricing.output_per_million_tokens
    ) / 1_000_000


def _validate_matrix_source(
    *,
    matrix: SevenRunMatrixArtifact,
    source_key: str,
    path: Path,
    repository_root: Path,
) -> None:
    expected = matrix.source_artifacts.get(source_key)
    if expected is None or expected != _source(path, repository_root):
        raise BudgetAttributionError(f"matrix source binding drifted: {source_key}")


def _arm_sources(
    *, matrix: SevenRunMatrixArtifact, arm: Arm, repository_root: Path
) -> tuple[MatrixCell, Path, Path]:
    cell = next(
        (
            item
            for item in matrix.cells
            if item.role is ModelRole.PRIMARY_DECISION and item.arm == arm
        ),
        None,
    )
    if cell is None:
        raise BudgetAttributionError(f"primary matrix cell is missing: {arm}")
    run_dir = repository_root / cell.run_directory
    calls_path = run_dir / "calls.jsonl"
    manifest_path = run_dir / "run-manifest.json"
    _validate_matrix_source(
        matrix=matrix,
        source_key=f"primary_decision_{arm}_calls.jsonl",
        path=calls_path,
        repository_root=repository_root,
    )
    if sha256_file(manifest_path) != cell.run_manifest_sha256:
        raise BudgetAttributionError(f"primary {arm} manifest differs from matrix")
    return cell, calls_path, manifest_path


def _validated_arm_calls(
    *, arm: Arm, calls: list[LLMCallRecord], manifest: RunManifest
) -> tuple[list[str], dict[str, list[LLMCallRecord]]]:
    if manifest.model != PRIMARY_MODEL or manifest.model_role is not ModelRole.PRIMARY_DECISION:
        raise BudgetAttributionError(f"primary {arm} manifest identity drifted")
    if manifest.metadata.get("arm") != arm:
        raise BudgetAttributionError(f"primary {arm} manifest arm drifted")
    if any(
        call.arm != arm
        or call.requested_model != REQUESTED_MODEL
        or call.reported_model != REQUESTED_MODEL
        for call in calls
    ):
        raise BudgetAttributionError(f"primary {arm} call identity drifted")
    if any(call.usage is None for call in calls):
        raise BudgetAttributionError(f"primary {arm} usage is incomplete")
    by_case: dict[str, list[LLMCallRecord]] = defaultdict(list)
    case_order: list[str] = []
    for call in calls:
        if call.case_id not in by_case:
            case_order.append(call.case_id)
        by_case[call.case_id].append(call)
    if len(case_order) != 112 or len(calls) != 224:
        raise BudgetAttributionError(f"primary {arm} must contain 112 cases and 224 calls")
    expected_stages = list(_expected_stages(arm))
    for case_id in case_order:
        case_calls = by_case[case_id]
        if [call.call_index for call in case_calls] != [1, 2]:
            raise BudgetAttributionError(f"{case_id}: primary {arm} call indexes drifted")
        if [call.stage for call in case_calls] != expected_stages:
            raise BudgetAttributionError(f"{case_id}: primary {arm} stages drifted")
    return case_order, by_case


def _call_budget(call: LLMCallRecord) -> CallBudget:
    assert call.usage is not None
    return CallBudget(
        stage=call.stage,
        call_index=call.call_index,
        status=call.status,
        error_type=call.error_type,
        usage=call.usage,
        latency_ms=call.latency_ms,
    )


def _case_arm_budget(
    *, arm: Arm, calls: list[LLMCallRecord], pricing: PricingConfig
) -> CaseArmBudget:
    budgets = [_call_budget(call) for call in calls]
    usage = _sum_usage([call.usage for call in budgets])
    return CaseArmBudget(
        arm=arm,
        calls=budgets,
        total_usage=usage,
        estimated_cost_usd=_cost(usage, pricing),
        end_to_end_latency_ms=sum(call.latency_ms for call in budgets),
    )


def _stage_summary(
    *, arm: Arm, stage: Stage, calls: list[LLMCallRecord], pricing: PricingConfig
) -> StageBudgetSummary:
    selected = [call for call in calls if call.stage == stage]
    if len(selected) != 112 or any(call.usage is None for call in selected):
        raise BudgetAttributionError(f"primary {arm}/{stage} must contain 112 complete calls")
    usages = [call.usage for call in selected]
    assert all(usage is not None for usage in usages)
    complete_usages = [usage for usage in usages if usage is not None]
    total_usage = _sum_usage(complete_usages)
    return StageBudgetSummary(
        arm=arm,
        stage=stage,
        call_index=selected[0].call_index,
        success_count=sum(call.status is PredictionStatus.SUCCESS for call in selected),
        failure_count=sum(call.status is not PredictionStatus.SUCCESS for call in selected),
        total_usage=total_usage,
        estimated_cost_usd=_cost(total_usage, pricing),
        tokens_per_call=_distribution([float(usage.total_tokens) for usage in complete_usages]),
        latency_ms=_distribution([call.latency_ms for call in selected]),
    )


def _arm_summary(
    *,
    arm: Arm,
    calls: list[LLMCallRecord],
    case_budgets: list[CaseArmBudget],
    manifest: RunManifest,
    cell: MatrixCell,
    pricing: PricingConfig,
) -> ArmBudgetSummary:
    usages = [call.usage for call in calls]
    assert all(usage is not None for usage in usages)
    total_usage = _sum_usage([usage for usage in usages if usage is not None])
    if manifest.total_usage != total_usage:
        raise BudgetAttributionError(f"primary {arm} manifest usage differs from calls")
    if (
        cell.call_count != len(calls)
        or cell.input_tokens != total_usage.input_tokens
        or cell.output_tokens != total_usage.output_tokens
        or cell.total_tokens != total_usage.total_tokens
    ):
        raise BudgetAttributionError(f"primary {arm} matrix totals differ from calls")
    cost = _cost(total_usage, pricing)
    if manifest.estimated_cost != cost or cell.estimated_cost != cost:
        raise BudgetAttributionError(f"primary {arm} registered cost differs from calls")
    provider_calls = manifest.metadata.get("provider_calls")
    if not isinstance(provider_calls, int):
        raise BudgetAttributionError(f"primary {arm} provider-call count is missing")
    return ArmBudgetSummary(
        arm=arm,
        incremental_provider_call_count=provider_calls,
        success_call_count=sum(call.status is PredictionStatus.SUCCESS for call in calls),
        failure_call_count=sum(call.status is not PredictionStatus.SUCCESS for call in calls),
        total_usage=total_usage,
        estimated_cost_usd=cost,
        per_case_total_tokens=_distribution(
            [float(case.total_usage.total_tokens) for case in case_budgets]
        ),
        per_case_end_to_end_latency_ms=_distribution(
            [case.end_to_end_latency_ms for case in case_budgets]
        ),
        stages=[
            _stage_summary(arm=arm, stage=stage, calls=calls, pricing=pricing)
            for stage in _expected_stages(arm)
        ],
    )


def _stage_pair_difference(
    *,
    call_index: Literal[1, 2],
    b2b: StageBudgetSummary,
    b3: StageBudgetSummary,
) -> StagePairDifference:
    return StagePairDifference(
        call_index=call_index,
        b2b_stage=cast(B2bStage, b2b.stage),
        b3_stage=cast(B3Stage, b3.stage),
        b3_minus_b2b_tokens=_difference(b3.total_usage, b2b.total_usage),
        b3_minus_b2b_cost_usd=b3.estimated_cost_usd - b2b.estimated_cost_usd,
    )


def render_markdown(report: BudgetAttributionReport) -> str:
    """Render a concise explanation only from the validated attribution artifact."""

    b2b, b3 = report.arms
    gap = report.gap
    token_gap = gap.b3_minus_b2b_tokens
    paired_latency = gap.paired_latency_difference_ms
    lines = [
        "# IE4.5A Primary Budget Attribution",
        "",
        "> Offline diagnostic only: provider_calls=0 and the IE4 verdict is unchanged.",
        "",
        "## Observed arm totals",
        "",
        "| Arm | Logical calls | Incremental provider calls | Input tokens | Output tokens | "
        "Retry tokens | Total tokens | Estimated cost USD | P50 case latency ms | "
        "P95 case latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report.arms:
        retry_tokens = arm.total_usage.retry_input_tokens + arm.total_usage.retry_output_tokens
        lines.append(
            f"| {arm.arm} | {arm.logical_call_count} | "
            f"{arm.incremental_provider_call_count} | {arm.total_usage.input_tokens:,} | "
            f"{arm.total_usage.output_tokens:,} | {retry_tokens:,} | "
            f"{arm.total_usage.total_tokens:,} | {arm.estimated_cost_usd:.8f} | "
            f"{arm.per_case_end_to_end_latency_ms.p50:.1f} | "
            f"{arm.per_case_end_to_end_latency_ms.p95:.1f} |"
        )
    lines.extend(
        [
            "",
            "B3 minus B2b is "
            f"{token_gap.total_tokens:+,} total tokens "
            f"({gap.absolute_total_token_difference_rate:.2%} absolute difference), "
            f"{gap.b3_minus_b2b_cost_usd:+.8f} USD, "
            f"{paired_latency.p50:+.1f} ms at paired P50, and "
            f"{paired_latency.p95:+.1f} ms at paired P95.",
            "",
            "## Stage-pair attribution",
            "",
            "| Position | B2b stage | B3 stage | Δ input | Δ output | Δ retry | "
            "Δ total | Δ cost USD |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in gap.stage_pairs:
        difference = pair.b3_minus_b2b_tokens
        retry_difference = difference.retry_input_tokens + difference.retry_output_tokens
        lines.append(
            f"| {pair.call_index} | {pair.b2b_stage} | {pair.b3_stage} | "
            f"{difference.input_tokens:+,} | {difference.output_tokens:+,} | "
            f"{retry_difference:+,} | {difference.total_tokens:+,} | "
            f"{pair.b3_minus_b2b_cost_usd:+.8f} |"
        )
    lines.extend(
        [
            "",
            "The observed gap is driven by the first call: Stage A uses substantially fewer "
            "input tokens than the one-stage B2b call, while Stage B is larger than the B2b "
            "verifier and partially offsets that saving. Output usage is nearly equal overall.",
            "",
            "## Contract boundary",
            "",
            "- The registered contract says B2b sees the full taxonomy in both calls, while "
            "B3 hides it in Stage A and exposes it in Stage B.",
            "- API usage supports exact arm, stage, and case accounting. It cannot uniquely "
            "split input tokens into taxonomy, prompt instructions, schema, few-shot, and "
            "context components.",
            "- The first-call difference is therefore associated with the registered "
            "architecture, but it is not a component-level causal estimate of taxonomy tokens.",
            "- B2b counts both logical calls for fair resource accounting even though its "
            "first-call results were reused from the B2a cache; incremental provider-call "
            "counts are reported separately.",
            "- This artifact does not modify or recompute the IE4 global verdict.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run intentbench ie45 budget-attribution",
            "```",
        ]
    )
    assert b2b.arm == "B2b" and b3.arm == "B3"
    return "\n".join(lines) + "\n"


def _write_artifact(path: Path, payload: bytes) -> Literal["created", "unchanged"]:
    if path.exists():
        if path.read_bytes() != payload:
            raise BudgetAttributionError(
                f"IE4.5A output differs from deterministic regeneration: {path}"
            )
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return "created"


def build_budget_attribution_report(
    *,
    matrix_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Build IE4.5A only from frozen cached primary artifacts."""

    matrix: SevenRunMatrixArtifact = _read_model(matrix_path, SevenRunMatrixArtifact)
    if sha256_file(experiment_path) != matrix.experiment_lock_sha256:
        raise BudgetAttributionError("experiment config differs from the matrix lock")
    config = _load_config(experiment_path)
    pricing = config.pricing
    if pricing.model != REQUESTED_MODEL:
        raise BudgetAttributionError("pricing model differs from the primary calls")

    arm_inputs: dict[Arm, tuple[MatrixCell, Path, Path, list[LLMCallRecord], RunManifest]] = {}
    case_orders: dict[Arm, list[str]] = {}
    calls_by_case: dict[Arm, dict[str, list[LLMCallRecord]]] = {}
    for arm in ARMS:
        cell, calls_path, manifest_path = _arm_sources(
            matrix=matrix, arm=arm, repository_root=repository_root
        )
        calls = _read_calls(calls_path)
        manifest: RunManifest = _read_model(manifest_path, RunManifest)
        case_order, grouped = _validated_arm_calls(arm=arm, calls=calls, manifest=manifest)
        arm_inputs[arm] = (cell, calls_path, manifest_path, calls, manifest)
        case_orders[arm] = case_order
        calls_by_case[arm] = grouped
    if case_orders["B2b"] != case_orders["B3"]:
        raise BudgetAttributionError("primary B2b/B3 case order differs")

    case_rows: list[CaseBudgetAttribution] = []
    arm_case_budgets: dict[Arm, list[CaseArmBudget]] = {"B2b": [], "B3": []}
    for case_id in case_orders["B2b"]:
        b2b = _case_arm_budget(arm="B2b", calls=calls_by_case["B2b"][case_id], pricing=pricing)
        b3 = _case_arm_budget(arm="B3", calls=calls_by_case["B3"][case_id], pricing=pricing)
        arm_case_budgets["B2b"].append(b2b)
        arm_case_budgets["B3"].append(b3)
        case_rows.append(
            CaseBudgetAttribution(
                case_id=case_id,
                b2b=b2b,
                b3=b3,
                b3_minus_b2b_tokens=_difference(b3.total_usage, b2b.total_usage),
                b3_minus_b2b_cost_usd=b3.estimated_cost_usd - b2b.estimated_cost_usd,
                b3_minus_b2b_latency_ms=(b3.end_to_end_latency_ms - b2b.end_to_end_latency_ms),
            )
        )

    arm_summaries: list[ArmBudgetSummary] = []
    source_artifacts = {
        "matrix": _source(matrix_path, repository_root),
        "experiment": _source(experiment_path, repository_root),
    }
    for arm in ARMS:
        cell, calls_path, manifest_path, calls, manifest = arm_inputs[arm]
        arm_summaries.append(
            _arm_summary(
                arm=arm,
                calls=calls,
                case_budgets=arm_case_budgets[arm],
                manifest=manifest,
                cell=cell,
                pricing=pricing,
            )
        )
        source_artifacts[f"primary_{arm}_calls"] = _source(calls_path, repository_root)
        source_artifacts[f"primary_{arm}_manifest"] = _source(manifest_path, repository_root)

    b2b_summary, b3_summary = arm_summaries
    comparison = next(
        item for item in matrix.budget_comparisons if item.role is ModelRole.PRIMARY_DECISION
    )
    if comparison.maximum_allowed != 0.1 or comparison.status != "budget-confounded":
        raise BudgetAttributionError("primary matrix no longer records the frozen budget confound")
    stage_pairs = [
        _stage_pair_difference(call_index=1, b2b=b2b_summary.stages[0], b3=b3_summary.stages[0]),
        _stage_pair_difference(call_index=2, b2b=b2b_summary.stages[1], b3=b3_summary.stages[1]),
    ]
    gap = BudgetGapSummary(
        b3_minus_b2b_tokens=_difference(b3_summary.total_usage, b2b_summary.total_usage),
        absolute_total_token_difference_rate=comparison.absolute_difference_rate,
        registered_maximum_rate=comparison.maximum_allowed,
        registered_status=comparison.status,
        b3_minus_b2b_cost_usd=(b3_summary.estimated_cost_usd - b2b_summary.estimated_cost_usd),
        paired_latency_difference_ms=_signed_distribution(
            [case.b3_minus_b2b_latency_ms for case in case_rows]
        ),
        stage_pairs=stage_pairs,
    )
    expected_rate = abs(gap.b3_minus_b2b_tokens.total_tokens) / max(
        b2b_summary.total_usage.total_tokens, 1
    )
    if gap.absolute_total_token_difference_rate != expected_rate:
        raise BudgetAttributionError("matrix budget rate differs from call-level attribution")

    report = BudgetAttributionReport(
        dataset_version=matrix.dataset_version,
        taxonomy_version=matrix.taxonomy_version,
        pricing=pricing,
        arms=arm_summaries,
        gap=gap,
        cases=case_rows,
        boundary=AttributionBoundary(
            registered_taxonomy_visibility={
                "call_1_one_stage": "full",
                "call_2_verifier": "full",
                "stage_a": config.b3.stage_a_taxonomy_visibility,
                "stage_b": config.b3.stage_b_taxonomy_visibility,
            }
        ),
        source_artifacts=source_artifacts,
        limitations=[
            "API usage metadata permits exact case/stage accounting but not exact tokenizer-level "
            "decomposition of taxonomy, instructions, schema, few-shot, and context components.",
            "The registered taxonomy visibility supports an architectural association, not a "
            "component-level causal estimate of taxonomy token savings.",
            "B2b logical cost includes the reused B2a-compatible first call; incremental provider "
            "calls are reported separately from full arm resource accounting.",
            "This post-IE4 diagnostic makes no provider calls and has no effect on the frozen IE4 "
            "global verdict.",
        ],
    )
    report_payload = _json_bytes(report.model_dump(mode="json"))
    markdown_payload = render_markdown(report).encode()
    report_status = _write_artifact(output_dir / REPORT_FILENAME, report_payload)
    markdown_status = _write_artifact(output_dir / REPORT_MARKDOWN_FILENAME, markdown_payload)
    return {
        "status": "created" if "created" in (report_status, markdown_status) else "unchanged",
        "provider_calls": 0,
        "case_count": len(case_rows),
        "b2b_total_tokens": b2b_summary.total_usage.total_tokens,
        "b3_total_tokens": b3_summary.total_usage.total_tokens,
        "b3_minus_b2b_total_tokens": gap.b3_minus_b2b_tokens.total_tokens,
        "absolute_total_token_difference_rate": gap.absolute_total_token_difference_rate,
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "output_dir": str(output_dir),
    }
