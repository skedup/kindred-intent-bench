"""Executable IE1.1 annotation-pack contracts; never formal Gold data."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from intentbench.schemas import (
    Context,
    Decision,
    Gold,
    StrictModel,
    normalize_evidence_text,
)
from intentbench.taxonomy import load_taxonomy

ExampleKind = Literal["positive", "negative", "adjudication"]
BoundaryTag = Literal[
    "multi_turn",
    "context_distractor",
    "brandless_xhs",
    "rest_eat_confusion",
]
DatasetCardDisclosure = Literal[
    "synthetic_balanced_dataset",
    "non_blind_frozen_test",
    "no_production_data",
    "single_annotator_limit",
    "llm_assistance_provenance",
    "external_validity_limit",
    "test_access_and_tuning_boundary",
]

REQUIRED_BOUNDARY_TAGS: frozenset[str] = frozenset(
    {"multi_turn", "context_distractor", "brandless_xhs", "rest_eat_confusion"}
)
REQUIRED_DATASET_CARD_DISCLOSURES: frozenset[str] = frozenset(
    {
        "synthetic_balanced_dataset",
        "non_blind_frozen_test",
        "no_production_data",
        "single_annotator_limit",
        "llm_assistance_provenance",
        "external_validity_limit",
        "test_access_and_tuning_boundary",
    }
)


def _ordered_decision_pair(left: Decision, right: Decision) -> tuple[str, str]:
    if left.value < right.value:
        return left.value, right.value
    return right.value, left.value


REQUIRED_ADJUDICATION_DECISION_PAIRS: frozenset[tuple[str, str]] = frozenset(
    _ordered_decision_pair(left, right) for left, right in combinations(Decision, 2)
)


def _normalized_text(value: str) -> str:
    return " ".join(normalize_evidence_text(value).split()).casefold()


def _evidence_is_in_context(context: Context, evidence_quote: str) -> bool:
    carriers = [context.state_summary]
    carriers.extend(turn.content for turn in context.conversation)
    quote = normalize_evidence_text(evidence_quote)
    return any(quote in normalize_evidence_text(carrier) for carrier in carriers)


def _context_signature(context: Context) -> tuple[object, ...]:
    return (
        _normalized_text(context.state_summary),
        tuple((turn.role, _normalized_text(turn.content)) for turn in context.conversation),
        tuple(_normalized_text(activity) for activity in context.recent_activities),
    )


class CompetingLabel(StrictModel):
    """A complete alternative Gold label considered during adjudication."""

    decision: Decision
    target_intent: str | None

    @model_validator(mode="after")
    def validate_target_truth_table(self) -> CompetingLabel:
        if self.decision is Decision.IN_SCOPE:
            if self.target_intent is None:
                raise ValueError("in_scope competing label requires a target")
        elif self.target_intent is not None:
            raise ValueError("rejection competing label cannot carry a target")
        return self


class GuideExample(StrictModel):
    """A guideline example, deliberately lacking case/split identity."""

    example_id: str = Field(pattern=r"^guide-[a-z0-9-]{5,80}$")
    boundary_decision: Decision
    kind: ExampleKind
    context: Context
    gold: Gold
    competing_labels: list[CompetingLabel] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_example_contract(self) -> GuideExample:
        if not _evidence_is_in_context(self.context, self.gold.evidence_quote):
            raise ValueError("guide evidence_quote must be a contiguous context substring")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("guide tags must be unique")
        competing_keys = [(label.decision, label.target_intent) for label in self.competing_labels]
        if len(competing_keys) != len(set(competing_keys)):
            raise ValueError("competing_labels must be unique")

        if self.kind == "positive":
            if self.gold.decision is not self.boundary_decision:
                raise ValueError("positive guide must resolve to its boundary_decision")
            if self.competing_labels:
                raise ValueError("positive guide cannot declare competing_labels")
        elif self.kind == "negative":
            if self.gold.decision is self.boundary_decision:
                raise ValueError("negative guide must resolve away from its boundary_decision")
            if self.competing_labels:
                raise ValueError("negative guide cannot declare competing_labels")
        else:
            if self.gold.decision is not self.boundary_decision:
                raise ValueError("adjudication guide must resolve to its boundary_decision")
            if len(self.competing_labels) < 2:
                raise ValueError("adjudication guide requires at least two competing labels")
            selected_key = (self.gold.decision, self.gold.target_intent)
            if selected_key not in competing_keys:
                raise ValueError("competing labels must include the selected Gold label")

        is_near_oos = "near_oos" in self.tags
        if is_near_oos:
            if self.gold.decision is not Decision.OOS:
                raise ValueError("near_oos guide must resolve to oos")
            if not self.gold.near_oos_sibling_intents:
                raise ValueError("near_oos guide requires sibling intents")
        elif self.gold.near_oos_sibling_intents:
            raise ValueError("only near_oos guides may declare sibling intents")
        return self


class IntentComparisonExample(StrictModel):
    text: str = Field(min_length=1)
    expected_decision: Decision
    expected_target_intent: str | None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_truth_table(self) -> IntentComparisonExample:
        if self.expected_decision is Decision.IN_SCOPE:
            if self.expected_target_intent is None:
                raise ValueError("in_scope intent comparison requires a target")
        elif self.expected_target_intent is not None:
            raise ValueError("rejection intent comparison cannot carry a target")
        return self


class IntentBoundary(StrictModel):
    intent: str
    inclusion_examples: list[IntentComparisonExample] = Field(min_length=2, max_length=2)
    exclusion_examples: list[IntentComparisonExample] = Field(min_length=2, max_length=2)
    annotation_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inclusion_and_exclusion(self) -> IntentBoundary:
        examples = [*self.inclusion_examples, *self.exclusion_examples]
        signatures = [_normalized_text(example.text) for example in examples]
        if len(signatures) != len(set(signatures)):
            raise ValueError("intent comparison texts must be unique within a boundary")
        if any(
            example.expected_decision is not Decision.IN_SCOPE
            or example.expected_target_intent != self.intent
            for example in self.inclusion_examples
        ):
            raise ValueError("all inclusion examples must resolve to their declared intent")
        if any(
            example.expected_decision is Decision.IN_SCOPE
            and example.expected_target_intent == self.intent
            for example in self.exclusion_examples
        ):
            raise ValueError("exclusion examples must resolve away from their declared intent")
        return self


class NearOOSSiblingPair(StrictModel):
    pair_id: str = Field(pattern=r"^near-oos-[a-z0-9-]{3,80}$")
    sibling_intent: str
    oos_text: str = Field(min_length=1)
    sibling_in_scope_text: str = Field(min_length=1)
    decisive_difference: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pair_contrast(self) -> NearOOSSiblingPair:
        if normalize_evidence_text(self.oos_text) == normalize_evidence_text(
            self.sibling_in_scope_text
        ):
            raise ValueError("near-OOS pair texts must differ")
        return self


class AdjudicationContract(StrictModel):
    workflow_steps: list[str] = Field(min_length=4)
    disagreement_record_fields: list[str] = Field(min_length=7)
    unresolved_action: Literal["exclude_from_freeze"]
    single_annotator_recheck_min_days: int = Field(ge=3)

    @model_validator(mode="after")
    def validate_unique_fields(self) -> AdjudicationContract:
        if len(self.disagreement_record_fields) != len(set(self.disagreement_record_fields)):
            raise ValueError("disagreement_record_fields must be unique")
        required = {
            "case_id",
            "annotator_id",
            "proposed_label",
            "evidence_quote",
            "rationale",
            "resolution",
            "adjudicator_id",
        }
        if not required.issubset(self.disagreement_record_fields):
            raise ValueError("adjudication record lacks required fields")
        return self


class DatasetCardContract(StrictModel):
    required_disclosures: list[DatasetCardDisclosure]

    @model_validator(mode="after")
    def validate_disclosures(self) -> DatasetCardContract:
        if len(self.required_disclosures) != len(set(self.required_disclosures)):
            raise ValueError("dataset-card disclosures must be unique")
        if set(self.required_disclosures) != REQUIRED_DATASET_CARD_DISCLOSURES:
            raise ValueError("dataset-card disclosures do not match the IE1.1 contract")
        return self


class AnnotationPack(StrictModel):
    schema_version: Literal[2]
    pack_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+-v[0-9]+$")
    status: Literal["approved_for_ie1_authoring"]
    dataset_version: str
    taxonomy_version: str
    decision_examples: list[GuideExample] = Field(min_length=36, max_length=36)
    intent_boundaries: list[IntentBoundary] = Field(min_length=8, max_length=8)
    near_oos_pairs: list[NearOOSSiblingPair] = Field(min_length=8, max_length=8)
    required_boundary_tags: list[BoundaryTag]
    adjudication: AdjudicationContract
    dataset_card: DatasetCardContract

    @model_validator(mode="after")
    def validate_pack_coverage(self) -> AnnotationPack:
        example_ids = [example.example_id for example in self.decision_examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("guide example IDs must be unique")
        context_signatures = [
            _context_signature(example.context) for example in self.decision_examples
        ]
        if len(context_signatures) != len(set(context_signatures)):
            raise ValueError("guide contexts must be unique")
        coverage = Counter(
            (example.boundary_decision, example.kind) for example in self.decision_examples
        )
        for decision in Decision:
            for kind in ("positive", "negative", "adjudication"):
                if coverage[(decision, kind)] != 3:
                    raise ValueError(
                        f"guide coverage requires exactly 3 {decision.value}/{kind} examples"
                    )

        adjudication_examples = [
            example for example in self.decision_examples if example.kind == "adjudication"
        ]
        observed_decision_pairs: set[tuple[str, str]] = set()
        for example in adjudication_examples:
            competing_decisions = sorted(
                {label.decision for label in example.competing_labels},
                key=lambda decision: decision.value,
            )
            observed_decision_pairs.update(
                _ordered_decision_pair(left, right)
                for left, right in combinations(competing_decisions, 2)
            )
        if not REQUIRED_ADJUDICATION_DECISION_PAIRS.issubset(observed_decision_pairs):
            raise ValueError("adjudication guides must cover all six decision pairs")
        has_in_scope_intent_dispute = any(
            len(
                {
                    label.target_intent
                    for label in example.competing_labels
                    if label.decision is Decision.IN_SCOPE
                }
            )
            >= 2
            for example in adjudication_examples
        )
        if not has_in_scope_intent_dispute:
            raise ValueError("adjudication guides require an in-scope intent dispute")

        boundary_tags = set(self.required_boundary_tags)
        if boundary_tags != REQUIRED_BOUNDARY_TAGS:
            raise ValueError("required_boundary_tags do not match the IE1.1 contract")
        observed_tags = {tag for example in self.decision_examples for tag in example.tags}
        if not boundary_tags.issubset(observed_tags):
            raise ValueError("guide examples do not cover every required boundary tag")

        brandless_examples = [
            example for example in self.decision_examples if "brandless_xhs" in example.tags
        ]
        if any(
            example.gold.decision is not Decision.IN_SCOPE
            or example.gold.target_intent != "play_xiaohongshu"
            or "小红书" in example.context.state_summary
            or any("小红书" in turn.content for turn in example.context.conversation)
            for example in brandless_examples
        ):
            raise ValueError("brandless_xhs guides must map brandless text to play_xiaohongshu")
        multi_turn_examples = [
            example for example in self.decision_examples if "multi_turn" in example.tags
        ]
        if any(not example.context.conversation for example in multi_turn_examples):
            raise ValueError("multi_turn guides require conversation turns")

        intents = [boundary.intent for boundary in self.intent_boundaries]
        if len(intents) != len(set(intents)):
            raise ValueError("intent boundary names must be unique")
        pair_ids = [pair.pair_id for pair in self.near_oos_pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("near-OOS pair IDs must be unique")
        return self


class DraftCase(StrictModel):
    """Pre-split authoring record that cannot pass the formal Case schema."""

    id: str = Field(pattern=r"^draft-[a-z0-9-]{3,73}$")
    context: Context
    gold: Gold
    tags: list[str] = Field(default_factory=list)
    scenario_family_id: str = Field(min_length=1)
    contrast_group_id: str = Field(min_length=1)
    paraphrase_cluster_id: str = Field(min_length=1)
    source: Literal["human_authored", "llm_assisted_human_reviewed"]
    annotator_id: str = Field(min_length=1)
    adjudication_status: Literal["draft", "reviewed", "adjudicated"]
    annotation_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_draft_contract(self) -> DraftCase:
        if not _evidence_is_in_context(self.context, self.gold.evidence_quote):
            raise ValueError("draft evidence_quote must be a contiguous context substring")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("draft tags must be unique")
        is_near_oos = "near_oos" in self.tags
        if is_near_oos:
            if self.gold.decision is not Decision.OOS:
                raise ValueError("near_oos draft requires Gold decision=oos")
            if not self.gold.near_oos_sibling_intents:
                raise ValueError("near_oos draft requires sibling intents")
        elif self.gold.near_oos_sibling_intents:
            raise ValueError("only near_oos drafts may declare sibling intents")
        return self


class CaseAuthoringTemplate(StrictModel):
    schema_version: Literal[1]
    template_kind: Literal["non_dataset_case_authoring_template"]
    instructions: list[str] = Field(min_length=5)
    case: DraftCase

    @model_validator(mode="after")
    def validate_non_dataset_template(self) -> CaseAuthoringTemplate:
        if len(self.instructions) != len(set(self.instructions)):
            raise ValueError("authoring template instructions must be unique")
        if not self.case.id.startswith("draft-template-"):
            raise ValueError("authoring template ID must use draft-template- prefix")
        if self.case.adjudication_status != "draft":
            raise ValueError("authoring template must remain draft")
        return self


def load_annotation_pack(path: Path) -> AnnotationPack:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid annotation pack YAML: {path}") from exc
    return AnnotationPack.model_validate(payload)


def load_case_authoring_template(path: Path) -> CaseAuthoringTemplate:
    return CaseAuthoringTemplate.model_validate_json(path.read_text(encoding="utf-8"))


def validate_annotation_artifacts(
    pack_path: Path, taxonomy_path: Path, template_path: Path
) -> dict[str, object]:
    pack = load_annotation_pack(pack_path)
    taxonomy = load_taxonomy(taxonomy_path)
    template = load_case_authoring_template(template_path)
    known_intents = {intent.name for intent in taxonomy.intents}
    if pack.taxonomy_version != taxonomy.taxonomy_version:
        raise ValueError("annotation pack and taxonomy versions differ")
    boundary_intents = {boundary.intent for boundary in pack.intent_boundaries}
    if boundary_intents != known_intents:
        raise ValueError("annotation pack must define every taxonomy intent exactly once")
    pair_intents = {pair.sibling_intent for pair in pack.near_oos_pairs}
    if len(pack.near_oos_pairs) != len(known_intents) or pair_intents != known_intents:
        raise ValueError("near-OOS pairs must cover every taxonomy intent exactly once")

    for example in pack.decision_examples:
        target = example.gold.target_intent
        if target is not None and target not in known_intents:
            raise ValueError(f"guide {example.example_id} has unknown target intent")
        unknown_siblings = set(example.gold.near_oos_sibling_intents) - known_intents
        if unknown_siblings:
            raise ValueError(f"guide {example.example_id} has unknown sibling intents")
        unknown_competing_targets = {
            label.target_intent
            for label in example.competing_labels
            if label.target_intent is not None
        } - known_intents
        if unknown_competing_targets:
            raise ValueError(f"guide {example.example_id} has unknown competing targets")
    for boundary in pack.intent_boundaries:
        examples = [*boundary.inclusion_examples, *boundary.exclusion_examples]
        unknown_targets = {
            example.expected_target_intent
            for example in examples
            if example.expected_target_intent is not None
        } - known_intents
        if unknown_targets:
            raise ValueError(f"intent boundary {boundary.intent} has unknown targets")

    template_target = template.case.gold.target_intent
    if template_target is not None and template_target not in known_intents:
        raise ValueError("case authoring template has unknown target intent")
    unknown_template_siblings = set(template.case.gold.near_oos_sibling_intents) - known_intents
    if unknown_template_siblings:
        raise ValueError("case authoring template has unknown sibling intents")

    return {
        "status": "valid",
        "pack_id": pack.pack_id,
        "decision_example_count": len(pack.decision_examples),
        "intent_boundary_count": len(pack.intent_boundaries),
        "near_oos_pair_count": len(pack.near_oos_pairs),
        "required_boundary_tags": sorted(pack.required_boundary_tags),
        "template_case_id": template.case.id,
    }
