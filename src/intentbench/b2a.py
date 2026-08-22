"""Registered one-stage prompt, supervision, and output contract for B2a."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from intentbench.adapters.base import GenerationResult
from intentbench.b0 import serialize_context
from intentbench.generation_cache import (
    CALL_CONTRACT_FIELDS,
    CachedGeneration,
    GenerationRequest,
    canonical_json,
    sha256_json,
    sha256_text,
)
from intentbench.providers import LLMModelSpec
from intentbench.schemas import (
    Case,
    Decision,
    Prediction,
    PredictionStatus,
    StrictModel,
    Taxonomy,
    TokenUsage,
)
from intentbench.taxonomy import TaxonomyError, validate_prediction

PROMPT_PLACEHOLDERS = (
    "{{TAXONOMY_JSON}}",
    "{{FEW_SHOTS_JSONL}}",
    "{{CONTEXT_JSON}}",
)


class SharedFewShotConfig(StrictModel):
    selection_algorithm: Literal["manual_risk_balanced_cluster_distinct_v1"]
    selection_receipt_artifact: str = Field(min_length=1)
    case_ids: list[str] = Field(min_length=4)
    required_decision_counts: dict[Decision, int]
    evidence_quote_in_prompt: Literal[False] = False
    review_fields_in_prompt: Literal[False] = False

    @model_validator(mode="after")
    def validate_selection(self) -> SharedFewShotConfig:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("shared few-shot IDs must be unique")
        if set(self.required_decision_counts) != set(Decision):
            raise ValueError("shared few-shot counts must register every decision")
        if any(count < 1 for count in self.required_decision_counts.values()):
            raise ValueError("shared few-shot decision counts must be positive")
        if sum(self.required_decision_counts.values()) != len(self.case_ids):
            raise ValueError("shared few-shot counts must sum to the registered IDs")
        return self


class LLMCacheConfig(StrictModel):
    format: Literal["normalized-generation-record-v2"]
    key_fields: list[str]
    cache_provider_failures: Literal[True] = True

    @model_validator(mode="after")
    def validate_key(self) -> LLMCacheConfig:
        if self.key_fields != list(CALL_CONTRACT_FIELDS):
            raise ValueError("LLM cache key fields differ from the registered order")
        return self


class FewShotSelectionReceipt(StrictModel):
    schema_version: Literal[1] = 1
    receipt_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: Literal["dev"]
    dev_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_algorithm: Literal["manual_risk_balanced_cluster_distinct_v1"]
    case_ids: list[str] = Field(min_length=4)
    required_decision_counts: dict[Decision, int]
    global_constraints: list[str] = Field(min_length=1)
    coverage_policy: str = Field(min_length=1)
    case_rationales: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_receipt(self) -> FewShotSelectionReceipt:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("few-shot receipt case IDs must be unique")
        if set(self.case_rationales) != set(self.case_ids):
            raise ValueError("few-shot receipt rationales must cover exactly the selected IDs")
        if any(not reasons for reasons in self.case_rationales.values()):
            raise ValueError("every selected few-shot case needs at least one rationale")
        return self


class PromptCandidateReceipt(StrictModel):
    prompt_version: str = Field(pattern=r"^v[0-9]+$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    full_dev_hem: float = Field(ge=0, le=1)
    full_dev_decision_macro_f1: float = Field(ge=0, le=1)
    case_id_held_out_count: int = Field(ge=1)
    case_id_held_out_hem: float = Field(ge=0, le=1)
    case_id_held_out_decision_macro_f1: float = Field(ge=0, le=1)
    cluster_held_out_count: int = Field(ge=1)
    cluster_held_out_hem: float = Field(ge=0, le=1)
    cluster_held_out_decision_macro_f1: float = Field(ge=0, le=1)
    diagnostic_errors: list[str]


class PromptSelectionReceipt(StrictModel):
    schema_version: Literal[1] = 1
    receipt_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: Literal["dev"]
    dev_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_metric_order: list[str]
    revision_budget: Literal[1]
    provider: str
    model: str
    fixed_call_policy: dict[str, Any]
    candidates: list[PromptCandidateReceipt] = Field(min_length=2)
    selected_prompt_version: str = Field(pattern=r"^v[0-9]+$")
    revision_scope: list[str] = Field(min_length=1)
    stopping_rule: str = Field(min_length=1)
    excluded_diagnostic_run: str = Field(min_length=1)
    interpretation_limit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidates(self) -> PromptSelectionReceipt:
        if self.selection_metric_order != [
            "hierarchical_exact_match",
            "decision_macro_f1",
        ]:
            raise ValueError("Prompt selection metric order differs from the registered contract")
        versions = [candidate.prompt_version for candidate in self.candidates]
        if len(versions) != len(set(versions)):
            raise ValueError("Prompt selection candidate versions must be unique")
        if self.selected_prompt_version not in versions:
            raise ValueError("selected Prompt version is not a registered candidate")
        return self


class B2AConfig(StrictModel):
    prompt_artifact: str = Field(min_length=1)
    prompt_selection_receipt_artifact: str = Field(min_length=1)
    model_role: Literal["primary_decision"]
    calls_per_case: Literal[1]
    request_max_output_tokens: int = Field(ge=1)
    retry_policy: Literal["none"]
    schema_repair_policy: Literal["reject"]
    selected_prompt_version: str = Field(pattern=r"^v[0-9]+$")


class PricingConfig(StrictModel):
    provider: Literal["google_generative_language"]
    model: str
    tier: Literal["standard_paid_list"]
    currency: Literal["USD"]
    snapshot_date: str
    input_per_million_tokens: float = Field(ge=0)
    output_per_million_tokens: float = Field(ge=0)
    source: str


class LLMExperimentConfig(StrictModel):
    input_serializer: Literal["context-json-v1"]
    prompt_renderer: Literal["literal-three-placeholder-v1"]
    response_schema: Literal["prediction-semantic-fields-v1"]
    shared_few_shot: SharedFewShotConfig
    cache: LLMCacheConfig
    b2a: B2AConfig
    pricing: PricingConfig

    @model_validator(mode="after")
    def validate_pricing_model(self) -> LLMExperimentConfig:
        if self.pricing.model != "gemini-3.6-flash":
            raise ValueError("B2a pricing must bind the registered primary model")
        return self


def validate_few_shot_cases(cases: Sequence[Case], config: SharedFewShotConfig) -> tuple[Case, ...]:
    case_by_id = {case.id: case for case in cases}
    missing = set(config.case_ids) - set(case_by_id)
    if missing:
        raise ValueError(f"shared few-shot IDs are not in frozen dev: {sorted(missing)}")
    selected = tuple(case_by_id[case_id] for case_id in config.case_ids)
    clusters = [case.bootstrap_cluster_id for case in selected]
    if any(cluster is None for cluster in clusters) or len(clusters) != len(set(clusters)):
        raise ValueError("shared few-shot cases must use distinct assigned bootstrap clusters")
    counts = Counter(case.gold.decision for case in selected)
    if dict(counts) != config.required_decision_counts:
        raise ValueError("shared few-shot decision counts differ from the registered balance")
    return selected


def validate_few_shot_receipt(
    *,
    receipt: FewShotSelectionReceipt,
    config: SharedFewShotConfig,
    selected_cases: Sequence[Case],
    dataset_version: str,
    dev_sha256: str,
) -> None:
    if receipt.dataset_version != dataset_version or receipt.dev_sha256 != dev_sha256:
        raise ValueError("few-shot receipt is bound to different frozen dev inputs")
    if receipt.selection_algorithm != config.selection_algorithm:
        raise ValueError("few-shot receipt selection algorithm differs from experiment")
    if receipt.case_ids != config.case_ids:
        raise ValueError("few-shot receipt case order differs from experiment")
    if receipt.required_decision_counts != config.required_decision_counts:
        raise ValueError("few-shot receipt decision balance differs from experiment")
    if [case.id for case in selected_cases] != receipt.case_ids:
        raise ValueError("few-shot receipt and loaded selection differ")


def validate_prompt_selection_receipt(
    *,
    receipt: PromptSelectionReceipt,
    config: B2AConfig,
    dataset_version: str,
    dev_sha256: str,
    provider: str,
    model: str,
    prompt_sha256: str,
) -> None:
    if receipt.dataset_version != dataset_version or receipt.dev_sha256 != dev_sha256:
        raise ValueError("Prompt selection receipt is bound to different frozen dev inputs")
    if receipt.provider != provider or receipt.model != model:
        raise ValueError("Prompt selection receipt model identity differs")
    if receipt.selected_prompt_version != config.selected_prompt_version:
        raise ValueError("selected Prompt version differs from its receipt")
    selected = next(
        candidate
        for candidate in receipt.candidates
        if candidate.prompt_version == receipt.selected_prompt_version
    )
    if selected.prompt_sha256 != prompt_sha256:
        raise ValueError("selected Prompt hash differs from its receipt")


def _few_shot_reason(case: Case) -> str:
    if case.gold.decision is Decision.IN_SCOPE:
        return f"当前主要行动唯一对应 {case.gold.target_intent}"
    if case.gold.decision is Decision.OOS:
        return "当前主要行动明确; 但不属于 Activity 目录"
    if case.gold.decision is Decision.NO_INTENT:
        return "输入没有表达当前可执行的下一行动"
    return "存在当前行动倾向; 但候选或对象尚未消解"


def few_shot_payload(cases: Sequence[Case]) -> list[dict[str, object]]:
    return [
        {
            "context": case.context.model_dump(mode="json"),
            "output": {
                "decision": case.gold.decision.value,
                "predicted_intent": case.gold.target_intent,
                "candidate_intents": [],
                "slots": case.gold.slots.model_dump(mode="json"),
                "reason_short": _few_shot_reason(case),
            },
        }
        for case in cases
    ]


def taxonomy_prompt_payload(taxonomy: Taxonomy) -> dict[str, object]:
    return {
        "taxonomy_version": taxonomy.taxonomy_version,
        "intents": [intent.model_dump(mode="json") for intent in taxonomy.intents],
    }


def render_b2a_prompt(
    *,
    template: str,
    taxonomy: Taxonomy,
    few_shots: Sequence[Case],
    context_json: str,
) -> str:
    for placeholder in PROMPT_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(f"B2a Prompt must contain {placeholder} exactly once")
    rendered = template.replace(
        "{{TAXONOMY_JSON}}", canonical_json(taxonomy_prompt_payload(taxonomy))
    )
    rendered = rendered.replace(
        "{{FEW_SHOTS_JSONL}}",
        "\n".join(canonical_json(item) for item in few_shot_payload(few_shots)),
    )
    return rendered.replace("{{CONTEXT_JSON}}", context_json)


def prediction_response_schema(taxonomy: Taxonomy) -> dict[str, Any]:
    intent_names = [intent.name for intent in taxonomy.intents]
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": [decision.value for decision in Decision]},
            "predicted_intent": {
                "anyOf": [{"type": "string", "enum": intent_names}, {"type": "null"}]
            },
            "candidate_intents": {
                "type": "array",
                "items": {"type": "string", "enum": intent_names},
            },
            "slots": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "desired_experience": nullable_string,
                    "object": nullable_string,
                    "horizon": {
                        "type": "string",
                        "enum": ["now", "later", "unspecified"],
                    },
                },
                "required": ["desired_experience", "object", "horizon"],
            },
            "reason_short": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "required": [
            "decision",
            "predicted_intent",
            "candidate_intents",
            "slots",
            "reason_short",
        ],
    }


def prediction_from_generation(
    *, case_id: str, result: GenerationResult, taxonomy: Taxonomy
) -> Prediction:
    payload = {
        "case_id": case_id,
        "status": PredictionStatus.SUCCESS.value,
        **result.value,
        "latency_ms": result.latency_ms,
        "usage": TokenUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
        ).model_dump(mode="json"),
        "raw_response_sha256": result.raw_response_sha256,
    }
    try:
        prediction = Prediction.model_validate(payload)
        validate_prediction(prediction, taxonomy)
        return prediction
    except (ValidationError, TaxonomyError, ValueError) as exc:
        return Prediction(
            case_id=case_id,
            status=PredictionStatus.SCHEMA_INVALID,
            error_type=f"local_contract:{type(exc).__name__}",
            latency_ms=result.latency_ms,
            usage=TokenUsage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
            ),
            raw_response_sha256=result.raw_response_sha256,
        )


def normalize_cached_one_stage_prediction(
    *,
    case_id: str,
    cached: CachedGeneration,
    taxonomy: Taxonomy,
    spec: LLMModelSpec,
) -> Prediction:
    """Apply the one-stage result contract shared by B2a and B2b call-1."""

    if cached.result is None:
        usage = None
        if (
            cached.input_tokens is not None
            and cached.output_tokens is not None
            and cached.total_tokens is not None
        ):
            usage = TokenUsage(
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                total_tokens=cached.total_tokens,
            )
        return Prediction(
            case_id=case_id,
            status=(
                PredictionStatus.SCHEMA_INVALID
                if cached.error_type == "schema_invalid"
                else PredictionStatus.PROVIDER_FAILURE
            ),
            error_type=cached.error_type or "provider_failure",
            latency_ms=cached.latency_ms,
            usage=usage,
            raw_response_sha256=cached.raw_response_sha256,
        )

    result = cached.result
    usage = TokenUsage(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
    )
    reported_identity = (result.reported_model or "").removeprefix("models/")
    if reported_identity != spec.model:
        return Prediction(
            case_id=case_id,
            status=PredictionStatus.PROVIDER_FAILURE,
            error_type="model_identity_mismatch",
            latency_ms=result.latency_ms,
            usage=usage,
            raw_response_sha256=result.raw_response_sha256,
        )
    if result.structured_output_mode != spec.structured_output_mode:
        return Prediction(
            case_id=case_id,
            status=PredictionStatus.SCHEMA_INVALID,
            error_type="structured_output_mode_mismatch",
            latency_ms=result.latency_ms,
            usage=usage,
            raw_response_sha256=result.raw_response_sha256,
        )
    if not set(spec.expected_usage_fields) <= set(result.usage_metadata):
        return Prediction(
            case_id=case_id,
            status=PredictionStatus.PROVIDER_FAILURE,
            error_type="usage_contract_mismatch",
            latency_ms=result.latency_ms,
            usage=usage,
            raw_response_sha256=result.raw_response_sha256,
        )
    return prediction_from_generation(case_id=case_id, result=result, taxonomy=taxonomy)


def one_stage_adapter_config_sha256(
    *,
    config: LLMExperimentConfig,
    spec: LLMModelSpec,
    prompt_sha256: str,
    few_shot_sha256: str,
    schema_sha256: str,
) -> str:
    """Hash only fields material to a B2a-compatible call, excluding other arms/pricing."""

    return sha256_json(
        {
            "contract_version": "one-stage-adapter-config-v2",
            "input_serializer": config.input_serializer,
            "prompt_renderer": config.prompt_renderer,
            "response_schema": config.response_schema,
            "prompt_sha256": prompt_sha256,
            "few_shot_payload_sha256": few_shot_sha256,
            "response_schema_sha256": schema_sha256,
            "provider_request": {
                "provider": spec.provider,
                "model": spec.model,
                "adapter": spec.adapter,
                "adapter_revision": spec.adapter_revision,
                "structured_output_mode": spec.structured_output_mode,
                "expected_usage_fields": spec.expected_usage_fields,
                "max_output_tokens": config.b2a.request_max_output_tokens,
                "generation_parameters": spec.generation_parameters,
            },
            "call_policy": {
                "calls_per_case": config.b2a.calls_per_case,
                "retry_policy": config.b2a.retry_policy,
                "schema_repair_policy": config.b2a.schema_repair_policy,
            },
        }
    )


def build_one_stage_request(
    *,
    context_json: str,
    prompt: str,
    taxonomy_sha256: str,
    prompt_template_sha256: str,
    few_shot_sha256: str,
    response_schema: dict[str, Any],
    spec: LLMModelSpec,
    max_output_tokens: int,
) -> GenerationRequest:
    return GenerationRequest(
        normalized_context=context_json,
        prompt=prompt,
        taxonomy_sha256=taxonomy_sha256,
        prompt_template_sha256=prompt_template_sha256,
        few_shot_payload_sha256=few_shot_sha256,
        response_schema=response_schema,
        model=spec.model,
        adapter=spec.adapter,
        adapter_revision=spec.adapter_revision,
        structured_output_mode=spec.structured_output_mode,
        max_output_tokens=max_output_tokens,
        generation_parameters=dict(spec.generation_parameters),
    )


def estimate_cost(usage: TokenUsage, pricing: PricingConfig) -> float:
    return (
        usage.input_tokens * pricing.input_per_million_tokens
        + usage.output_tokens * pricing.output_per_million_tokens
    ) / 1_000_000


def normalized_context_sha256(case: Case) -> str:
    return sha256_text(serialize_context(case.context))


def few_shot_payload_sha256(cases: Sequence[Case]) -> str:
    return sha256_json(few_shot_payload(cases))


def response_schema_sha256(taxonomy: Taxonomy) -> str:
    return sha256_json(prediction_response_schema(taxonomy))


def decision_counts(cases: Sequence[Case]) -> Mapping[Decision, int]:
    return Counter(case.gold.decision for case in cases)
