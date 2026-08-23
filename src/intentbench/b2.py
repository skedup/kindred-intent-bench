"""Registered two-call contracts for the B2b verifier and B3 semantic decomposition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, model_validator

from intentbench.b2a import LLMExperimentConfig, prediction_response_schema
from intentbench.generation_cache import canonical_json, sha256_json, sha256_text
from intentbench.schemas import Case, Decision, Horizon, StrictModel, Taxonomy

B2B_PROMPT_PLACEHOLDERS = (
    "{{TAXONOMY_JSON}}",
    "{{FEW_SHOTS_JSONL}}",
    "{{CONTEXT_JSON}}",
    "{{DRAFT_PREDICTION_JSON}}",
)
B3_STAGE_A_PROMPT_PLACEHOLDERS = (
    "{{FEW_SHOTS_JSONL}}",
    "{{CONTEXT_JSON}}",
)
B3_STAGE_B_PROMPT_PLACEHOLDERS = (
    "{{TAXONOMY_JSON}}",
    "{{FEW_SHOTS_JSONL}}",
    "{{FORMED_INTENTION_JSON}}",
)
TAXONOMY_HIDDEN_SHA256 = sha256_text("taxonomy-hidden-from-b3-stage-a-v1")


def _few_shot_reason(case: Case) -> str:
    if case.gold.decision is Decision.IN_SCOPE:
        return f"当前主要行动唯一对应 {case.gold.target_intent}"
    if case.gold.decision is Decision.OOS:
        return "当前主要行动明确; 但不属于 Activity 目录"
    if case.gold.decision is Decision.NO_INTENT:
        return "输入没有表达当前可执行的下一行动"
    return "存在当前行动倾向; 但候选或对象尚未消解"


def routing_few_shot_payload(cases: Sequence[Case]) -> list[dict[str, object]]:
    """Mirror B2a supervision without importing the parent experiment config module."""

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


class B2BConfig(StrictModel):
    verifier_prompt_artifact: str = Field(min_length=1)
    prompt_selection_receipt_artifact: str = Field(min_length=1)
    selected_prompt_version: Literal["v2"]
    calls_per_case: Literal[2]
    request_max_output_tokens_per_call: int = Field(ge=1)
    retry_policy: Literal["none"]
    schema_repair_policy: Literal["reject"]
    call_1_policy: Literal["require_existing_one_stage_v2"]
    call_2_semantics: Literal["same_channel_verifier"]


class B3Config(StrictModel):
    stage_a_prompt_artifact: str = Field(min_length=1)
    stage_b_prompt_artifact: str = Field(min_length=1)
    stage_a_supervision_artifact: str = Field(min_length=1)
    prompt_selection_receipt_artifact: str = Field(min_length=1)
    selected_prompt_version: Literal["v2"]
    formed_intention_contract: Literal["formed-intention-evidence-v2"]
    calls_per_case: Literal[2]
    request_max_output_tokens_per_call: int = Field(ge=1)
    retry_policy: Literal["none"]
    schema_repair_policy: Literal["reject"]
    stage_a_taxonomy_visibility: Literal["hidden"]
    stage_b_taxonomy_visibility: Literal["full"]
    stage_b_context_policy: Literal["formed_intention_only"]


class IE3LLMExperimentConfig(LLMExperimentConfig):
    b2b: B2BConfig
    b3: B3Config

    @model_validator(mode="after")
    def validate_two_call_fairness(self) -> IE3LLMExperimentConfig:
        budgets = {
            self.b2a.request_max_output_tokens,
            self.b2b.request_max_output_tokens_per_call,
            self.b3.request_max_output_tokens_per_call,
        }
        if len(budgets) != 1:
            raise ValueError("B2a/B2b/B3 per-call output budgets must match")
        if (
            self.b2a.retry_policy != self.b2b.retry_policy
            or self.b2a.retry_policy != self.b3.retry_policy
            or self.b2a.schema_repair_policy != self.b2b.schema_repair_policy
            or self.b2a.schema_repair_policy != self.b3.schema_repair_policy
        ):
            raise ValueError("LLM arms must align retry and schema-repair policies")
        if self.b2b.prompt_selection_receipt_artifact != self.b3.prompt_selection_receipt_artifact:
            raise ValueError("B2b/B3 must share one IE3 Prompt selection receipt")
        return self


class FormedIntention(StrictModel):
    """Taxonomy-free representation of intention evidence already present in context."""

    evidence_status: Literal["clear", "ambiguous", "deferred", "none"]
    action: str | None = Field(default=None, max_length=120)
    object: str | None = Field(default=None, max_length=120)
    desired_experience: str | None = Field(default=None, max_length=160)
    qualifiers: list[str] = Field(max_length=6)
    alternative_actions: list[str] = Field(default_factory=list, max_length=4)
    horizon: Horizon
    reason_short: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_semantics(self) -> FormedIntention:
        if len(self.alternative_actions) != len(set(self.alternative_actions)):
            raise ValueError("alternative_actions must be unique")
        if len(self.qualifiers) != len(set(self.qualifiers)):
            raise ValueError("qualifiers must be unique")
        if self.evidence_status == "none":
            if (
                any(
                    value is not None
                    for value in (self.action, self.object, self.desired_experience)
                )
                or self.alternative_actions
            ):
                raise ValueError("none evidence cannot contain positive intention fields")
        elif self.evidence_status in {"clear", "deferred"}:
            if self.action is None or self.alternative_actions:
                raise ValueError("clear/deferred evidence requires one action and no alternatives")
        elif self.action is None and len(self.alternative_actions) < 2:
            raise ValueError("ambiguous evidence needs an action or at least two alternatives")
        if self.evidence_status == "deferred" and self.horizon is not Horizon.LATER:
            raise ValueError("deferred evidence requires horizon=later")
        return self


class StageASupervisionExample(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    output: FormedIntention


class StageASupervisionReceipt(StrictModel):
    schema_version: Literal[1] = 1
    receipt_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: Literal["dev"]
    dev_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_receipt_artifact: str = Field(min_length=1)
    representation_contract: Literal["formed-intention-evidence-v1", "formed-intention-evidence-v2"]
    taxonomy_visible: Literal[False]
    annotation_policy: str = Field(min_length=1)
    examples: list[StageASupervisionExample] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_examples(self) -> StageASupervisionReceipt:
        ids = [example.case_id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("stage-A supervision case IDs must be unique")
        return self


class IE3PromptCandidateReceipt(StrictModel):
    version: str = Field(pattern=r"^v[0-9]+$")
    prompt_sha256: dict[str, str]
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hierarchical_exact_match: float = Field(ge=0, le=1)
    decision_macro_f1: float = Field(ge=0, le=1)
    schema_invalid_count: int = Field(ge=0)
    failed_prediction_count: int = Field(ge=0)
    badcase_ids: list[str]
    provider_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_hashes(self) -> IE3PromptCandidateReceipt:
        if not self.prompt_sha256 or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.prompt_sha256.values()
        ):
            raise ValueError("every Prompt candidate needs valid Prompt hashes")
        if len(self.badcase_ids) != len(set(self.badcase_ids)):
            raise ValueError("Prompt candidate badcase IDs must be unique")
        return self


class IE3ArmPromptSelection(StrictModel):
    selected_version: str = Field(pattern=r"^v[0-9]+$")
    candidates: list[IE3PromptCandidateReceipt] = Field(min_length=2)
    revision_scope: list[str] = Field(min_length=1)
    stopping_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> IE3ArmPromptSelection:
        versions = [candidate.version for candidate in self.candidates]
        if len(versions) != len(set(versions)):
            raise ValueError("IE3 Prompt candidate versions must be unique")
        if self.selected_version not in versions:
            raise ValueError("selected IE3 Prompt version is not a candidate")
        return self


class IE3PromptSelectionReceipt(StrictModel):
    schema_version: Literal[1] = 1
    receipt_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: Literal["dev"]
    dev_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str
    model: str
    revision_budget: Literal[1]
    selection_order: list[str]
    arms: dict[Literal["B2b", "B3"], IE3ArmPromptSelection]
    interpretation_limit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> IE3PromptSelectionReceipt:
        if set(self.arms) != {"B2b", "B3"}:
            raise ValueError("IE3 Prompt receipt must contain B2b and B3")
        if self.selection_order != [
            "schema_invalid_count_asc",
            "failed_prediction_count_asc",
            "hierarchical_exact_match_desc",
            "decision_macro_f1_desc",
        ]:
            raise ValueError("IE3 Prompt selection order differs from the registered contract")
        return self


def validate_ie3_prompt_selection_receipt(
    *,
    receipt: IE3PromptSelectionReceipt,
    config: IE3LLMExperimentConfig,
    dataset_version: str,
    dev_sha256: str,
    primary_provider: str,
    primary_model: str,
    selected_prompt_hashes: Mapping[str, Mapping[str, str]],
) -> None:
    if receipt.dataset_version != dataset_version or receipt.dev_sha256 != dev_sha256:
        raise ValueError("IE3 Prompt receipt is bound to different frozen dev inputs")
    if receipt.provider != primary_provider or receipt.model != primary_model:
        raise ValueError("IE3 Prompt receipt is bound to a different primary model")
    if config.b2b.prompt_selection_receipt_artifact != config.b3.prompt_selection_receipt_artifact:
        raise ValueError("IE3 arms reference different Prompt selection receipts")
    arm_versions: tuple[tuple[Literal["B2b", "B3"], str], ...] = (
        ("B2b", config.b2b.selected_prompt_version),
        ("B3", config.b3.selected_prompt_version),
    )
    for arm, selected_version in arm_versions:
        arm_receipt = receipt.arms[arm]
        if arm_receipt.selected_version != selected_version:
            raise ValueError(f"{arm} selected Prompt version differs from its receipt")
        selected = next(
            candidate
            for candidate in arm_receipt.candidates
            if candidate.version == selected_version
        )
        if selected.prompt_sha256 != dict(selected_prompt_hashes[arm]):
            raise ValueError(f"{arm} selected Prompt hashes differ from its receipt")


def validate_stage_a_supervision(
    *,
    receipt: StageASupervisionReceipt,
    selected_cases: Sequence[Case],
    dataset_version: str,
    dev_sha256: str,
    selection_receipt_artifact: str,
) -> None:
    selected_ids = [case.id for case in selected_cases]
    receipt_ids = [example.case_id for example in receipt.examples]
    if receipt.dataset_version != dataset_version or receipt.dev_sha256 != dev_sha256:
        raise ValueError("stage-A supervision is bound to different frozen dev inputs")
    if receipt.selection_receipt_artifact != selection_receipt_artifact:
        raise ValueError("stage-A supervision references a different few-shot receipt")
    if receipt_ids != selected_ids:
        raise ValueError("stage-A supervision must preserve shared few-shot order and IDs")


def formed_intention_response_schema() -> dict[str, Any]:
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_status": {
                "type": "string",
                "enum": ["clear", "ambiguous", "deferred", "none"],
            },
            "action": nullable_string,
            "object": nullable_string,
            "desired_experience": nullable_string,
            "qualifiers": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
            "alternative_actions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "horizon": {
                "type": "string",
                "enum": [horizon.value for horizon in Horizon],
            },
            "reason_short": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "required": [
            "evidence_status",
            "action",
            "object",
            "desired_experience",
            "qualifiers",
            "alternative_actions",
            "horizon",
            "reason_short",
        ],
    }


def two_call_prediction_response_schema(
    taxonomy: Taxonomy, *, openai_strict_compatible: bool = False
) -> dict[str, Any]:
    """Tighten ambiguous candidates within the selected Provider's JSON Schema subset."""

    schema = prediction_response_schema(taxonomy)
    intent_names = [intent.name for intent in taxonomy.intents]
    item_schema = {"type": "string", "enum": intent_names}
    empty_array_schema: dict[str, Any] = {"type": "array", "maxItems": 0}
    multiple_array_schema: dict[str, Any] = {
        "type": "array",
        "items": item_schema,
        "minItems": 2,
        "uniqueItems": True,
    }
    if openai_strict_compatible:
        # OpenAI Structured Outputs supports minItems/maxItems but not uniqueItems.
        # Keep item typing on both anyOf branches and enforce uniqueness locally in Prediction.
        empty_array_schema["items"] = item_schema
        multiple_array_schema.pop("uniqueItems")
    schema["properties"]["candidate_intents"] = {
        "anyOf": [empty_array_schema, multiple_array_schema]
    }
    return schema


def stage_a_few_shot_payload(
    cases: Sequence[Case], receipt: StageASupervisionReceipt
) -> list[dict[str, object]]:
    by_id = {example.case_id: example.output for example in receipt.examples}
    return [
        {
            "context": case.context.model_dump(mode="json"),
            "output": by_id[case.id].model_dump(mode="json"),
        }
        for case in cases
    ]


def stage_b_few_shot_payload(
    cases: Sequence[Case], receipt: StageASupervisionReceipt
) -> list[dict[str, object]]:
    formed_by_id = {example.case_id: example.output for example in receipt.examples}
    routing_by_id = {
        case.id: item["output"]
        for case, item in zip(cases, routing_few_shot_payload(cases), strict=True)
    }
    return [
        {
            "formed_intention": formed_by_id[case.id].model_dump(mode="json"),
            "output": routing_by_id[case.id],
        }
        for case in cases
    ]


def _render_literal_template(
    *, template: str, placeholders: Sequence[str], replacements: Mapping[str, str]
) -> str:
    for placeholder in placeholders:
        if template.count(placeholder) != 1:
            raise ValueError(f"Prompt must contain {placeholder} exactly once")
    rendered = template
    for placeholder in placeholders:
        rendered = rendered.replace(placeholder, replacements[placeholder])
    return rendered


def render_b2b_verifier_prompt(
    *,
    template: str,
    taxonomy: Taxonomy,
    few_shots: Sequence[Case],
    context_json: str,
    draft_prediction_json: str,
) -> str:
    return _render_literal_template(
        template=template,
        placeholders=B2B_PROMPT_PLACEHOLDERS,
        replacements={
            "{{TAXONOMY_JSON}}": canonical_json(taxonomy_prompt_payload(taxonomy)),
            "{{FEW_SHOTS_JSONL}}": "\n".join(
                canonical_json(item) for item in routing_few_shot_payload(few_shots)
            ),
            "{{CONTEXT_JSON}}": context_json,
            "{{DRAFT_PREDICTION_JSON}}": draft_prediction_json,
        },
    )


def render_b3_stage_a_prompt(
    *,
    template: str,
    few_shots: Sequence[Case],
    receipt: StageASupervisionReceipt,
    context_json: str,
) -> str:
    return _render_literal_template(
        template=template,
        placeholders=B3_STAGE_A_PROMPT_PLACEHOLDERS,
        replacements={
            "{{FEW_SHOTS_JSONL}}": "\n".join(
                canonical_json(item) for item in stage_a_few_shot_payload(few_shots, receipt)
            ),
            "{{CONTEXT_JSON}}": context_json,
        },
    )


def render_b3_stage_b_prompt(
    *,
    template: str,
    taxonomy: Taxonomy,
    few_shots: Sequence[Case],
    receipt: StageASupervisionReceipt,
    formed_intention_json: str,
) -> str:
    return _render_literal_template(
        template=template,
        placeholders=B3_STAGE_B_PROMPT_PLACEHOLDERS,
        replacements={
            "{{TAXONOMY_JSON}}": canonical_json(taxonomy_prompt_payload(taxonomy)),
            "{{FEW_SHOTS_JSONL}}": "\n".join(
                canonical_json(item) for item in stage_b_few_shot_payload(few_shots, receipt)
            ),
            "{{FORMED_INTENTION_JSON}}": formed_intention_json,
        },
    )


def verifier_few_shot_sha256(cases: Sequence[Case]) -> str:
    return sha256_json(routing_few_shot_payload(cases))


def stage_a_few_shot_sha256(cases: Sequence[Case], receipt: StageASupervisionReceipt) -> str:
    return sha256_json(stage_a_few_shot_payload(cases, receipt))


def stage_b_few_shot_sha256(cases: Sequence[Case], receipt: StageASupervisionReceipt) -> str:
    return sha256_json(stage_b_few_shot_payload(cases, receipt))
