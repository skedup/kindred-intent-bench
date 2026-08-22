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

MULTI_TURN_PATTERN_TAGS: tuple[str, ...] = (
    "multi_turn_override",
    "multi_turn_recency",
    "multi_turn_resolution",
    "multi_turn_reconsideration",
)
WEAK_MODEL_PROBE_COMPONENT_TAGS: frozenset[str] = frozenset(
    {"near_oos", "brandless_xhs", "rest_eat_confusion"}
)
DIAGNOSTIC_SLICE_TAGS: tuple[str, ...] = (
    "single_turn",
    "multi_turn",
    *MULTI_TURN_PATTERN_TAGS,
    "near_oos",
    "far_oos",
    "hard_negative",
    "hypothesized_weak_model_probe",
    "context_distractor",
    "context_control",
    "brandless_xhs",
    "rest_eat_confusion",
    "slot_required",
    "quiet_control",
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


class ContextPerturbation(StrictModel):
    """A registered background-only delta shared by a context pair."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    state_summary_prefix: str = Field(min_length=1)
    recent_activities_added: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_added_activities(self) -> ContextPerturbation:
        if len(self.recent_activities_added) != len(set(self.recent_activities_added)):
            raise ValueError("context perturbation activities must be unique")
        return self


class OpenIntentCandidate(StrictModel):
    """Non-routing signal for an activity idea the frozen catalog cannot safely execute."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    evidence_quote: str = Field(min_length=1)
    horizon: Horizon
    catalog_status: Literal["not_in_catalog", "underspecified_for_catalog"]
    routing_effect: Literal["none"] = "none"
    observability: Literal["log_candidate"] = "log_candidate"


RoutingLabelSource = Literal[
    "blind_human_review",
    "adjudicated_human",
    "adjudicated_draft",
    "policy_dual_channel_draft",
]


class ReviewProvenance(StrictModel):
    """Per-case lineage from the label-blind response through optional adjudication."""

    review_item_id: str = Field(pattern=r"^review-(initial|blind-retest)-[0-9]{4}$")
    reviewer_id: str = Field(min_length=1)
    reviewed_at: datetime
    routing_label_source: RoutingLabelSource
    supporting_fields_source: Literal["llm_assisted_draft"] = "llm_assisted_draft"
    adjudication_packet_item_id: str | None = Field(
        default=None, pattern=r"^adjudication-[0-9]{4}$"
    )
    adjudication_action: str | None = Field(default=None, min_length=1)
    adjudicator_id: str | None = Field(default=None, min_length=1)
    adjudicated_at: datetime | None = None
    policy_resolution_id: str | None = Field(default=None, pattern=r"^policy-[0-9]{4}$")

    @model_validator(mode="after")
    def validate_lineage(self) -> ReviewProvenance:
        if self.reviewed_at.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        adjudication_fields = (
            self.adjudication_packet_item_id,
            self.adjudication_action,
            self.adjudicator_id,
            self.adjudicated_at,
        )
        if any(value is not None for value in adjudication_fields):
            if any(value is None for value in adjudication_fields):
                raise ValueError("adjudication provenance fields must be present together")
            assert self.adjudicated_at is not None
            if self.adjudicated_at.tzinfo is None:
                raise ValueError("adjudicated_at must include a timezone")
        elif self.routing_label_source != "blind_human_review":
            raise ValueError("non-review routing sources require adjudication provenance")
        if self.routing_label_source == "policy_dual_channel_draft":
            if self.policy_resolution_id is None:
                raise ValueError("dual-channel routing requires policy_resolution_id")
        elif self.policy_resolution_id is not None:
            raise ValueError("only dual-channel routing may carry policy_resolution_id")
        return self


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


def validate_diagnostic_tag_contract(
    *,
    tags: list[str],
    gold: Gold,
    hard_negative_against: list[str],
    context_perturbation: ContextPerturbation | None,
) -> None:
    """Validate per-case diagnostic semantics shared by drafts and formal cases."""

    if len(tags) != len(set(tags)):
        raise ValueError("diagnostic tags must be unique")
    tag_set = set(tags)

    context_tags = {"context_control", "context_distractor"} & tag_set
    if len(context_tags) > 1:
        raise ValueError("a case cannot be both context control and distractor")
    if bool(context_tags) != (context_perturbation is not None):
        raise ValueError("context pair tags require exactly one registered perturbation")

    is_near_oos = "near_oos" in tag_set
    if is_near_oos:
        if gold.decision is not Decision.OOS:
            raise ValueError("near_oos tag requires Gold decision=oos")
        if not gold.near_oos_sibling_intents:
            raise ValueError("near_oos requires at least one sibling intent")
        if "hard_negative" not in tag_set:
            raise ValueError("near_oos must also be a hard_negative")
    elif gold.near_oos_sibling_intents:
        raise ValueError("only near_oos cases may declare sibling intents")

    if len(hard_negative_against) != len(set(hard_negative_against)):
        raise ValueError("hard_negative_against must be unique")
    if "hard_negative" in tag_set:
        if not hard_negative_against:
            raise ValueError("hard_negative requires at least one competing intent")
        if gold.decision is Decision.IN_SCOPE:
            raise ValueError("hard_negative cannot use in_scope Gold")
        if gold.decision is Decision.AMBIGUOUS and len(hard_negative_against) < 2:
            raise ValueError("ambiguous hard_negative requires at least two competing intents")
    elif hard_negative_against:
        raise ValueError("only hard_negative cases may declare competing intents")
    if is_near_oos and set(hard_negative_against) != set(gold.near_oos_sibling_intents):
        raise ValueError("near_oos hard-negative competitors must equal sibling intents")

    is_probe = "hypothesized_weak_model_probe" in tag_set
    has_probe_component = bool(WEAK_MODEL_PROBE_COMPONENT_TAGS & tag_set)
    if is_probe != has_probe_component:
        raise ValueError("hypothesized weak-model probe must equal its registered component tags")
    if "quiet_control" in tag_set and gold.decision is not Decision.NO_INTENT:
        raise ValueError("quiet_control requires Gold decision=no_intent")


class Case(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    split: Split
    context: Context
    gold: Gold
    tags: list[str] = Field(default_factory=list)
    hard_negative_against: list[str] = Field(default_factory=list)
    context_perturbation: ContextPerturbation | None = None
    scenario_family_id: str = Field(min_length=1)
    contrast_group_id: str = Field(min_length=1)
    paraphrase_cluster_id: str = Field(min_length=1)
    split_group_id: str | None = None
    bootstrap_cluster_id: str | None = None
    open_intent_candidate: OpenIntentCandidate | None = None
    review_provenance: ReviewProvenance | None = None
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

        validate_diagnostic_tag_contract(
            tags=self.tags,
            gold=self.gold,
            hard_negative_against=self.hard_negative_against,
            context_perturbation=self.context_perturbation,
        )

        if (self.split_group_id is None) != (self.bootstrap_cluster_id is None):
            raise ValueError("split_group_id and bootstrap_cluster_id must be assigned together")
        if self.split_group_id is not None and self.bootstrap_cluster_id != self.split_group_id:
            raise ValueError("bootstrap_cluster_id must equal split_group_id")
        if self.source == "llm_assisted_human_reviewed" and self.review_provenance is None:
            raise ValueError("LLM-assisted reviewed cases require review_provenance")
        if self.open_intent_candidate is not None:
            open_quote = normalize_evidence_text(self.open_intent_candidate.evidence_quote)
            if not any(open_quote in normalize_evidence_text(carrier) for carrier in carriers):
                raise ValueError("open_intent_candidate.evidence_quote must be a context substring")
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
    max_calls: int = Field(ge=0)
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
