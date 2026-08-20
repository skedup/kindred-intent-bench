"""IE1.2 candidate-dataset loading and coverage validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from intentbench.annotation import DraftCase
from intentbench.schemas import Decision, Taxonomy, normalize_evidence_text

EXPECTED_CANDIDATE_COUNT: Final = 160
EXPECTED_DECISION_COUNTS: Final = {
    Decision.IN_SCOPE: 80,
    Decision.OOS: 32,
    Decision.NO_INTENT: 24,
    Decision.AMBIGUOUS: 24,
}
MINIMUM_TAG_COUNTS: Final = {
    "multi_turn": 40,
    "hard_negative": 40,
    "context_distractor": 24,
    "weak_model_trap": 32,
    "brandless_xhs": 8,
    "rest_eat_confusion": 16,
}
ALLOWED_TAGS: Final = frozenset(
    {
        "single_turn",
        "multi_turn",
        "near_oos",
        "far_oos",
        "hard_negative",
        "weak_model_trap",
        "context_distractor",
        "brandless_xhs",
        "rest_eat_confusion",
        "slot_required",
        "quiet_control",
    }
)


class CandidateDatasetError(ValueError):
    """The pre-split, pre-freeze IE1.2 candidate set violates its contract."""


def load_candidate_cases(path: Path) -> list[DraftCase]:
    cases: list[DraftCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise CandidateDatasetError(f"{path}:{line_number}: blank JSONL lines are forbidden")
        try:
            payload = json.loads(raw_line)
            cases.append(DraftCase.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise CandidateDatasetError(f"{path}:{line_number}: invalid candidate case") from exc
    return cases


def _normalized_context(case: DraftCase) -> tuple[object, ...]:
    def normalize(value: str) -> str:
        return " ".join(normalize_evidence_text(value).split()).casefold()

    return (
        normalize(case.context.state_summary),
        tuple((turn.role, normalize(turn.content)) for turn in case.context.conversation),
    )


def validate_candidate_cases(cases: Sequence[DraftCase], taxonomy: Taxonomy) -> dict[str, object]:
    if len(cases) != EXPECTED_CANDIDATE_COUNT:
        raise CandidateDatasetError("IE1.2 candidate set must contain exactly 160 cases")

    expected_ids = [f"draft-kir-pilot-{index:04d}" for index in range(1, 161)]
    actual_ids = [case.id for case in cases]
    if actual_ids != expected_ids:
        raise CandidateDatasetError("candidate IDs must be unique, contiguous, and ordered")

    contexts = [_normalized_context(case) for case in cases]
    if len(contexts) != len(set(contexts)):
        raise CandidateDatasetError("candidate contexts must be unique")

    known_intents = {intent.name for intent in taxonomy.intents}
    decision_counts = Counter(case.gold.decision for case in cases)
    if decision_counts != Counter(EXPECTED_DECISION_COUNTS):
        raise CandidateDatasetError("candidate decision distribution differs from IE1.2")

    intent_counts = Counter(
        case.gold.target_intent for case in cases if case.gold.decision is Decision.IN_SCOPE
    )
    if intent_counts != Counter({intent: 10 for intent in known_intents}):
        raise CandidateDatasetError("each in-scope intent must have exactly 10 candidates")

    by_contrast: dict[str, list[DraftCase]] = defaultdict(list)
    by_paraphrase: dict[str, list[DraftCase]] = defaultdict(list)
    for case in cases:
        by_contrast[case.contrast_group_id].append(case)
        by_paraphrase[case.paraphrase_cluster_id].append(case)
        if case.source != "llm_assisted_pending_human_review":
            raise CandidateDatasetError(f"{case.id}: candidate provenance must remain pending")
        if case.adjudication_status != "draft":
            raise CandidateDatasetError(
                f"{case.id}: candidate must remain draft before human review"
            )
        if any(
            "replace-" in relation
            for relation in (
                case.scenario_family_id,
                case.contrast_group_id,
                case.paraphrase_cluster_id,
            )
        ):
            raise CandidateDatasetError(f"{case.id}: unresolved relationship placeholder")

        unknown_tags = set(case.tags) - ALLOWED_TAGS
        if unknown_tags:
            raise CandidateDatasetError(f"{case.id}: unknown tags {sorted(unknown_tags)}")
        turn_tags = {"single_turn", "multi_turn"} & set(case.tags)
        if len(turn_tags) != 1:
            raise CandidateDatasetError(f"{case.id}: requires exactly one turn-shape tag")
        if ("multi_turn" in turn_tags) != bool(case.context.conversation):
            raise CandidateDatasetError(f"{case.id}: turn-shape tag disagrees with context")

        target = case.gold.target_intent
        if target is not None and target not in known_intents:
            raise CandidateDatasetError(f"{case.id}: unknown target intent {target}")
        unknown_siblings = set(case.gold.near_oos_sibling_intents) - known_intents
        if unknown_siblings:
            raise CandidateDatasetError(f"{case.id}: unknown near-OOS siblings")
        unknown_recent = set(case.context.recent_activities) - known_intents
        if unknown_recent:
            raise CandidateDatasetError(f"{case.id}: unknown recent activities")

        is_oos = case.gold.decision is Decision.OOS
        oos_tags = {"near_oos", "far_oos"} & set(case.tags)
        if is_oos and len(oos_tags) != 1:
            raise CandidateDatasetError(f"{case.id}: OOS requires exactly one near/far tag")
        if not is_oos and oos_tags:
            raise CandidateDatasetError(f"{case.id}: near/far tags are OOS-only")
        if "quiet_control" in case.tags and case.gold.decision is not Decision.NO_INTENT:
            raise CandidateDatasetError(f"{case.id}: quiet_control is reserved for no_intent")
        if "hard_negative" in case.tags and case.gold.decision is Decision.IN_SCOPE:
            raise CandidateDatasetError(f"{case.id}: hard_negative cannot be in_scope")
        if (
            "rest_eat_confusion" in case.tags
            and case.gold.decision is Decision.IN_SCOPE
            and target not in {"rest", "eat_at_home"}
        ):
            raise CandidateDatasetError(f"{case.id}: invalid rest/eat slice membership")

        if "brandless_xhs" in case.tags:
            carriers = [case.context.state_summary]
            carriers.extend(turn.content for turn in case.context.conversation)
            if (
                case.gold.decision is not Decision.IN_SCOPE
                or target != "play_xiaohongshu"
                or any("小红书" in carrier for carrier in carriers)
            ):
                raise CandidateDatasetError(f"{case.id}: invalid brandless_xhs candidate")

    near_oos = [case for case in cases if "near_oos" in case.tags]
    far_oos = [case for case in cases if "far_oos" in case.tags]
    if len(near_oos) != 16 or len(far_oos) != 16:
        raise CandidateDatasetError("OOS candidates must split into 16 near and 16 far")
    sibling_counts = Counter(
        sibling for case in near_oos for sibling in case.gold.near_oos_sibling_intents
    )
    if sibling_counts != Counter({intent: 2 for intent in known_intents}):
        raise CandidateDatasetError("near-OOS candidates require two siblings per intent")
    for case in near_oos:
        contrast_cases = by_contrast[case.contrast_group_id]
        linked_targets = {
            candidate.gold.target_intent
            for candidate in contrast_cases
            if candidate.gold.decision is Decision.IN_SCOPE
        }
        if not set(case.gold.near_oos_sibling_intents) <= linked_targets:
            raise CandidateDatasetError(f"{case.id}: near-OOS sibling lacks contrast pair")
        if len(contrast_cases) != 2:
            raise CandidateDatasetError(f"{case.id}: near-OOS contrast must be an exact pair")

    for cluster_id, cluster_cases in by_paraphrase.items():
        if len(cluster_cases) == 1:
            continue
        labels = {(case.gold.decision, case.gold.target_intent) for case in cluster_cases}
        if len(labels) != 1:
            raise CandidateDatasetError(f"paraphrase cluster {cluster_id} crosses Gold labels")

    tag_counts = Counter(tag for case in cases for tag in case.tags)
    for tag, minimum in MINIMUM_TAG_COUNTS.items():
        if tag_counts[tag] < minimum:
            raise CandidateDatasetError(f"tag {tag} requires at least {minimum} candidates")

    return {
        "status": "valid_pending_human_review",
        "candidate_count": len(cases),
        "decision_counts": {decision.value: decision_counts[decision] for decision in Decision},
        "in_scope_intent_counts": dict(sorted(intent_counts.items())),
        "oos_partition": {"far_oos": len(far_oos), "near_oos": len(near_oos)},
        "required_tag_counts": {tag: tag_counts[tag] for tag in sorted(MINIMUM_TAG_COUNTS)},
    }


def validate_candidate_artifact(cases_path: Path, taxonomy_path: Path) -> dict[str, object]:
    from intentbench.taxonomy import load_taxonomy

    return validate_candidate_cases(load_candidate_cases(cases_path), load_taxonomy(taxonomy_path))
