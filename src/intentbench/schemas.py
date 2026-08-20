"""Versioned contracts shared by datasets, runners, and the evaluator."""

from __future__ import annotations

import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

INTENT_NAMES: tuple[str, ...] = (
    "create_picture",
    "dine_out",
    "eat_at_home",
    "play_xiaohongshu",
    "reach_out_to_user",
    "rest",
    "take_a_walk",
    "visit_cultural_place",
)


class StrictModel(BaseModel):
    """Reject silent contract drift in every serialized artifact."""

    model_config = ConfigDict(extra="forbid")


class Decision(str, Enum):
    IN_SCOPE = "in_scope"
    OOS = "oos"
    NO_INTENT = "no_intent"
    AMBIGUOUS = "ambiguous"


class Horizon(str, Enum):
    NOW = "now"
    LATER = "later"
    UNSPECIFIED = "unspecified"


class Split(str, Enum):
    DEV = "dev"
    TEST = "test"


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    PROVIDER_FAILURE = "provider_failure"
    SCHEMA_INVALID = "schema_invalid"


class ModelRole(str, Enum):
    PRIMARY_DECISION = "primary_decision"
    WEAK_DECISION = "weak_decision"
    CROSS_PROVIDER_REFERENCE = "cross_provider_reference"
    EMBEDDING = "embedding"


class Slots(StrictModel):
    desired_experience: str | None
    object: str | None
    horizon: Horizon


class ConversationTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Context(StrictModel):
    state_summary: str = Field(min_length=1)
    conversation: list[ConversationTurn] = Field(default_factory=list)
    recent_activities: list[str] = Field(default_factory=list)


class Gold(StrictModel):
    decision: Decision
    target_intent: str | None
    evidence_quote: str = Field(min_length=1)
    near_oos_sibling_intents: list[str] = Field(default_factory=list)
    slots: Slots

    @model_validator(mode="after")
    def validate_truth_table(self) -> Gold:
        if self.decision is Decision.IN_SCOPE:
            if self.target_intent is None:
                raise ValueError("in_scope Gold requires target_intent")
        elif self.target_intent is not None:
            raise ValueError("only in_scope Gold may carry target_intent")
        if len(self.near_oos_sibling_intents) != len(set(self.near_oos_sibling_intents)):
            raise ValueError("near_oos_sibling_intents must be unique")
        return self


def normalize_evidence_text(value: str) -> str:
    """Normalize only representation, never paraphrase evidence."""

    return unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")


class Case(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    split: Split
    context: Context
    gold: Gold
    tags: list[str] = Field(default_factory=list)
    scenario_family_id: str = Field(min_length=1)
    contrast_group_id: str = Field(min_length=1)
    paraphrase_cluster_id: str = Field(min_length=1)
    split_group_id: str | None = None
    bootstrap_cluster_id: str | None = None
    source: Literal["human_authored", "llm_assisted_human_reviewed", "synthetic_fixture"]
    annotator_id: str = Field(min_length=1)
    adjudication_status: Literal["draft", "reviewed", "adjudicated"]
    annotation_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_contract(self) -> Case:
        carriers = [self.context.state_summary]
        carriers.extend(turn.content for turn in self.context.conversation)
        quote = normalize_evidence_text(self.gold.evidence_quote)
        if not any(quote in normalize_evidence_text(carrier) for carrier in carriers):
            raise ValueError(
                "gold.evidence_quote must be a normalized contiguous context substring"
            )

        is_near_oos = "near_oos" in self.tags
        if is_near_oos:
            if self.gold.decision is not Decision.OOS:
                raise ValueError("near_oos tag requires Gold decision=oos")
            if not self.gold.near_oos_sibling_intents:
                raise ValueError("near_oos requires at least one sibling intent")
            if not self.contrast_group_id:
                raise ValueError("near_oos requires contrast_group_id")
        elif self.gold.near_oos_sibling_intents:
            raise ValueError("only near_oos cases may declare sibling intents")

        if (self.split_group_id is None) != (self.bootstrap_cluster_id is None):
            raise ValueError("split_group_id and bootstrap_cluster_id must be assigned together")
        if self.split_group_id is not None and self.bootstrap_cluster_id != self.split_group_id:
            raise ValueError("bootstrap_cluster_id must equal split_group_id")
        return self


class TokenUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    retry_input_tokens: int = Field(default=0, ge=0)
    retry_output_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> TokenUsage:
        billed = (
            self.input_tokens
            + self.output_tokens
            + self.retry_input_tokens
            + self.retry_output_tokens
        )
        if self.total_tokens != billed:
            raise ValueError(
                "total_tokens must include input, output, retry, and failed-call usage"
            )
        return self


class Prediction(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    status: PredictionStatus = PredictionStatus.SUCCESS
    decision: Decision | None = None
    predicted_intent: str | None = None
    candidate_intents: list[str] | None = None
    slots: Slots | None = None
    reason_short: str | None = Field(default=None, min_length=1, max_length=200)
    error_type: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    usage: TokenUsage | None = None
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_prediction_contract(self) -> Prediction:
        semantic = (
            self.decision,
            self.predicted_intent,
            self.candidate_intents,
            self.slots,
            self.reason_short,
        )
        if self.status is not PredictionStatus.SUCCESS:
            if any(value is not None for value in semantic):
                raise ValueError("failed predictions must not contain semantic fields")
            if not self.error_type:
                raise ValueError("failed predictions require error_type")
            return self

        if self.error_type is not None:
            raise ValueError("successful predictions must not contain error_type")
        if self.decision is None or self.candidate_intents is None:
            raise ValueError("successful predictions require decision and candidate_intents")
        if self.slots is None or self.reason_short is None:
            raise ValueError("successful predictions require slots and reason_short")
        if len(self.candidate_intents) != len(set(self.candidate_intents)):
            raise ValueError("candidate_intents must be unique")

        if self.decision is Decision.IN_SCOPE:
            if self.predicted_intent is None or self.candidate_intents:
                raise ValueError("in_scope requires one target and no candidates")
        else:
            if self.predicted_intent is not None:
                raise ValueError("rejection decisions must not contain predicted_intent")
            if self.decision is Decision.AMBIGUOUS:
                if len(self.candidate_intents) == 1:
                    raise ValueError(
                        "ambiguous candidates must be empty or contain at least two intents"
                    )
            elif self.candidate_intents:
                raise ValueError("only ambiguous may contain candidate_intents")
        return self


class IntentDefinition(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    definition: str = Field(min_length=1)
    inclusion_criteria: list[str] = Field(min_length=1)
    exclusion_criteria: list[str] = Field(min_length=1)
    canonical_examples: list[str] = Field(min_length=3, max_length=5)
    sibling_intents: list[str] = Field(default_factory=list)


class Taxonomy(StrictModel):
    schema_version: Literal[1]
    taxonomy_version: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+-v[0-9]+$")
    intents: list[IntentDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_intents(self) -> Taxonomy:
        names = [intent.name for intent in self.intents]
        if len(names) != len(set(names)):
            raise ValueError("taxonomy intent names must be unique")
        known = set(names)
        for intent in self.intents:
            if intent.name in intent.sibling_intents:
                raise ValueError(f"{intent.name} cannot be its own sibling")
            unknown = set(intent.sibling_intents) - known
            if unknown:
                raise ValueError(f"{intent.name} has unknown siblings: {sorted(unknown)}")
            for sibling in intent.sibling_intents:
                sibling_definition = self.intents[names.index(sibling)]
                if intent.name not in sibling_definition.sibling_intents:
                    raise ValueError(
                        f"sibling relation must be symmetric: {intent.name} -> {sibling}"
                    )
        return self


class RunParameters(StrictModel):
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_calls: int = Field(ge=1)
    max_output_tokens_per_call: int = Field(ge=1)
    retry_policy: str
    schema_repair_policy: str


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_version: str
    adapter: str
    adapter_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str
    model_role: ModelRole | Literal["none"]
    primary_decision_model: str
    embedding_model: str | None
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluator_version: str
    parameters: RunParameters
    pricing_snapshot_date: str
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    git_commit: str
    total_usage: TokenUsage | None = None
    estimated_cost: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
