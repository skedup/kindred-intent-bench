"""IE1.2 candidate-dataset loading and coverage validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from intentbench.annotation import DraftCase
from intentbench.schemas import (
    DIAGNOSTIC_SLICE_TAGS,
    MULTI_TURN_PATTERN_TAGS,
    Decision,
    Taxonomy,
    normalize_evidence_text,
)

EXPECTED_CANDIDATE_COUNT: Final = 160
EXPECTED_DECISION_COUNTS: Final = {
    Decision.IN_SCOPE: 80,
    Decision.OOS: 32,
    Decision.NO_INTENT: 24,
    Decision.AMBIGUOUS: 24,
}
EXPECTED_TAG_COUNTS: Final = {
    "multi_turn": 40,
    "hard_negative": 40,
    "context_distractor": 24,
    "context_control": 24,
    "hypothesized_weak_model_probe": 40,
    "brandless_xhs": 8,
    "rest_eat_confusion": 16,
}
EXPECTED_CONTEXT_DECISIONS: Final = Counter(
    {Decision.IN_SCOPE: 8, Decision.OOS: 4, Decision.NO_INTENT: 8, Decision.AMBIGUOUS: 4}
)
EXPECTED_MULTI_TURN_PATTERNS: Final = {tag: 10 for tag in MULTI_TURN_PATTERN_TAGS}
ALLOWED_TAGS: Final = frozenset(DIAGNOSTIC_SLICE_TAGS)


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
    by_scenario: dict[str, list[DraftCase]] = defaultdict(list)
    for case in cases:
        by_contrast[case.contrast_group_id].append(case)
        by_paraphrase[case.paraphrase_cluster_id].append(case)
        by_scenario[case.scenario_family_id].append(case)
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
        pattern_tags = set(EXPECTED_MULTI_TURN_PATTERNS) & set(case.tags)
        if "multi_turn" in turn_tags and len(pattern_tags) != 1:
            raise CandidateDatasetError(f"{case.id}: multi_turn requires one pattern tag")
        if "single_turn" in turn_tags and pattern_tags:
            raise CandidateDatasetError(f"{case.id}: single_turn cannot use multi-turn patterns")

        target = case.gold.target_intent
        if target is not None and target not in known_intents:
            raise CandidateDatasetError(f"{case.id}: unknown target intent {target}")
        unknown_siblings = set(case.gold.near_oos_sibling_intents) - known_intents
        if unknown_siblings:
            raise CandidateDatasetError(f"{case.id}: unknown near-OOS siblings")
        unknown_recent = set(case.context.recent_activities) - known_intents
        if unknown_recent:
            raise CandidateDatasetError(f"{case.id}: unknown recent activities")
        unknown_competitors = set(case.hard_negative_against) - known_intents
        if unknown_competitors:
            raise CandidateDatasetError(f"{case.id}: unknown hard-negative intents")
        if case.context_perturbation is not None:
            unknown_added = set(case.context_perturbation.recent_activities_added) - known_intents
            if unknown_added:
                raise CandidateDatasetError(f"{case.id}: unknown perturbation activities")

        is_oos = case.gold.decision is Decision.OOS
        oos_tags = {"near_oos", "far_oos"} & set(case.tags)
        if is_oos and len(oos_tags) != 1:
            raise CandidateDatasetError(f"{case.id}: OOS requires exactly one near/far tag")
        if not is_oos and oos_tags:
            raise CandidateDatasetError(f"{case.id}: near/far tags are OOS-only")
        if "quiet_control" in case.tags and case.gold.decision is not Decision.NO_INTENT:
            raise CandidateDatasetError(f"{case.id}: quiet_control is reserved for no_intent")
        if "hard_negative" in case.tags:
            if case.gold.decision is Decision.IN_SCOPE:
                raise CandidateDatasetError(f"{case.id}: hard_negative cannot be in_scope")
            if not case.hard_negative_against:
                raise CandidateDatasetError(f"{case.id}: hard_negative requires competing intents")
            if case.gold.decision is Decision.AMBIGUOUS and len(case.hard_negative_against) < 2:
                raise CandidateDatasetError(
                    f"{case.id}: ambiguous hard_negative needs at least two competitors"
                )
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

        if "multi_turn" in case.tags:
            if case.context.conversation[-1].role != "assistant":
                raise CandidateDatasetError(
                    f"{case.id}: multi_turn evidence must end in a Kindred turn"
                )
            if normalize_evidence_text(case.gold.evidence_quote) not in normalize_evidence_text(
                case.context.conversation[-1].content
            ):
                raise CandidateDatasetError(
                    f"{case.id}: multi_turn final Kindred turn must carry evidence"
                )

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
        sibling_case = next(
            candidate
            for candidate in contrast_cases
            if candidate.gold.decision is Decision.IN_SCOPE
        )
        if set(case.hard_negative_against) != set(case.gold.near_oos_sibling_intents):
            raise CandidateDatasetError(
                f"{case.id}: near-OOS hard-negative competitors must equal siblings"
            )
        if case.scenario_family_id != sibling_case.scenario_family_id:
            raise CandidateDatasetError(f"{case.id}: near-OOS pair must share scenario family")
        if ("multi_turn" in case.tags) != ("multi_turn" in sibling_case.tags):
            raise CandidateDatasetError(f"{case.id}: near-OOS pair must share turn shape")
        if case.context.recent_activities != sibling_case.context.recent_activities:
            raise CandidateDatasetError(f"{case.id}: near-OOS pair changes recent activity")
        case_turns = case.context.conversation[:-1]
        sibling_turns = sibling_case.context.conversation[:-1]
        if case_turns != sibling_turns:
            raise CandidateDatasetError(f"{case.id}: near-OOS pair changes dialogue frame")

    context_distractors = [case for case in cases if "context_distractor" in case.tags]
    context_controls = [case for case in cases if "context_control" in case.tags]
    context_groups = {case.contrast_group_id for case in (*context_distractors, *context_controls)}
    for group_id in context_groups:
        pair = by_contrast[group_id]
        if len(pair) != 2:
            raise CandidateDatasetError(f"{group_id}: context contrast must be an exact pair")
        distractors = [case for case in pair if "context_distractor" in case.tags]
        controls = [case for case in pair if "context_control" in case.tags]
        if len(distractors) != 1 or len(controls) != 1:
            raise CandidateDatasetError(
                f"{group_id}: context pair requires one control and one distractor"
            )
        distractor = distractors[0]
        control = controls[0]
        if distractor.scenario_family_id != control.scenario_family_id:
            raise CandidateDatasetError(f"{group_id}: context pair must share scenario family")
        if (
            distractor.gold != control.gold
            or distractor.hard_negative_against != control.hard_negative_against
            or distractor.context_perturbation != control.context_perturbation
            or distractor.context.conversation != control.context.conversation
            or ("multi_turn" in distractor.tags) != ("multi_turn" in control.tags)
        ):
            raise CandidateDatasetError(
                f"{group_id}: context pair changes more than background context"
            )
        distractor_tags = set(distractor.tags) - {"context_distractor"}
        control_tags = set(control.tags) - {"context_control"}
        if distractor_tags != control_tags:
            raise CandidateDatasetError(f"{group_id}: context pair changes diagnostic tags")
        if control.context.recent_activities:
            raise CandidateDatasetError(f"{group_id}: context control must have no recent activity")
        perturbation = control.context_perturbation
        if perturbation is None:
            raise CandidateDatasetError(f"{group_id}: context pair lacks registered perturbation")
        if distractor.context.state_summary != (
            f"{perturbation.state_summary_prefix}{control.context.state_summary}"
        ):
            raise CandidateDatasetError(
                f"{group_id}: context distractor does not apply the registered state delta"
            )
        if distractor.context.recent_activities != [
            *control.context.recent_activities,
            *perturbation.recent_activities_added,
        ]:
            raise CandidateDatasetError(
                f"{group_id}: context distractor does not apply the registered activity delta"
            )

    repeated_user_turns: dict[str, list[DraftCase]] = defaultdict(list)
    for case in cases:
        if "multi_turn" not in case.tags:
            continue
        if len(by_scenario[case.scenario_family_id]) < 2:
            raise CandidateDatasetError(f"{case.id}: multi_turn scenario skeleton must be grouped")
        for turn in case.context.conversation:
            if turn.role == "user":
                repeated_user_turns[normalize_evidence_text(turn.content)].append(case)
    for text, turn_cases in repeated_user_turns.items():
        if len(turn_cases) >= 2 and len({case.gold.decision for case in turn_cases}) == 1:
            raise CandidateDatasetError(f"multi-turn user cue maps to one Gold decision: {text}")

    for cluster_id, cluster_cases in by_paraphrase.items():
        if len(cluster_cases) == 1:
            continue
        labels = {(case.gold.decision, case.gold.target_intent) for case in cluster_cases}
        if len(labels) != 1:
            raise CandidateDatasetError(f"paraphrase cluster {cluster_id} crosses Gold labels")

    tag_counts = Counter(tag for case in cases for tag in case.tags)
    for tag, expected in EXPECTED_TAG_COUNTS.items():
        if tag_counts[tag] != expected:
            raise CandidateDatasetError(f"tag {tag} requires exactly {expected} candidates")
    for tag, expected in EXPECTED_MULTI_TURN_PATTERNS.items():
        if tag_counts[tag] != expected:
            raise CandidateDatasetError(f"multi-turn pattern {tag} requires exactly {expected}")

    hard_negative_decisions = Counter(
        case.gold.decision for case in cases if "hard_negative" in case.tags
    )
    if hard_negative_decisions != Counter(
        {Decision.OOS: 16, Decision.NO_INTENT: 16, Decision.AMBIGUOUS: 8}
    ):
        raise CandidateDatasetError("hard-negative decision composition differs from IE1.2")

    for context_tag in ("context_distractor", "context_control"):
        context_cases = [case for case in cases if context_tag in case.tags]
        if Counter(case.gold.decision for case in context_cases) != EXPECTED_CONTEXT_DECISIONS:
            raise CandidateDatasetError(f"{context_tag} decision composition differs from IE1.2")
        if Counter("multi_turn" in case.tags for case in context_cases) != Counter(
            {True: 12, False: 12}
        ):
            raise CandidateDatasetError(f"{context_tag} must balance single and multi turn")

    rest_eat_cases = [case for case in cases if "rest_eat_confusion" in case.tags]
    rest_eat_labels = Counter(
        case.gold.target_intent
        if case.gold.decision is Decision.IN_SCOPE
        else case.gold.decision.value
        for case in rest_eat_cases
    )
    if rest_eat_labels != Counter({"rest": 4, "eat_at_home": 4, "no_intent": 4, "ambiguous": 4}):
        raise CandidateDatasetError("rest/eat slice requires four cases in each boundary bucket")
    if Counter("multi_turn" in case.tags for case in rest_eat_cases) != Counter(
        {True: 8, False: 8}
    ):
        raise CandidateDatasetError("rest/eat slice must balance single and multi turn")

    expected_probe_ids = {
        case.id
        for case in cases
        if {"near_oos", "brandless_xhs", "rest_eat_confusion"} & set(case.tags)
    }
    actual_probe_ids = {case.id for case in cases if "hypothesized_weak_model_probe" in case.tags}
    if actual_probe_ids != expected_probe_ids:
        raise CandidateDatasetError(
            "hypothesized weak-model probe must equal its pre-registered component slices"
        )

    overlap_counts = {
        "hard_negative&hypothesized_weak_model_probe": sum(
            {"hard_negative", "hypothesized_weak_model_probe"} <= set(case.tags) for case in cases
        ),
        "multi_turn&context_control": sum(
            {"multi_turn", "context_control"} <= set(case.tags) for case in cases
        ),
        "multi_turn&context_distractor": sum(
            {"multi_turn", "context_distractor"} <= set(case.tags) for case in cases
        ),
        "near_oos&hard_negative&hypothesized_weak_model_probe": sum(
            {"near_oos", "hard_negative", "hypothesized_weak_model_probe"} <= set(case.tags)
            for case in cases
        ),
    }

    return {
        "status": "valid_pending_human_review",
        "candidate_count": len(cases),
        "decision_counts": {decision.value: decision_counts[decision] for decision in Decision},
        "in_scope_intent_counts": dict(sorted(intent_counts.items())),
        "oos_partition": {"far_oos": len(far_oos), "near_oos": len(near_oos)},
        "required_tag_counts": {tag: tag_counts[tag] for tag in sorted(EXPECTED_TAG_COUNTS)},
        "multi_turn_pattern_counts": {
            tag: tag_counts[tag] for tag in sorted(EXPECTED_MULTI_TURN_PATTERNS)
        },
        "slice_overlap_counts": overlap_counts,
    }


def validate_candidate_artifact(cases_path: Path, taxonomy_path: Path) -> dict[str, object]:
    from intentbench.taxonomy import load_taxonomy

    return validate_candidate_cases(load_candidate_cases(cases_path), load_taxonomy(taxonomy_path))
