"""Taxonomy loading and cross-artifact validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from intentbench.schemas import Case, Decision, Prediction, Taxonomy


class TaxonomyError(ValueError):
    """Raised when an artifact violates the frozen taxonomy."""


def load_taxonomy(path: Path) -> Taxonomy:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    taxonomy = Taxonomy.model_validate(payload)
    names = tuple(intent.name for intent in taxonomy.intents)
    if names != tuple(sorted(names)):
        raise TaxonomyError("taxonomy intents must be sorted by name for stable hashing")
    return taxonomy


def validate_cases(cases: Iterable[Case], taxonomy: Taxonomy) -> None:
    case_list = list(cases)
    ids = [case.id for case in case_list]
    if len(ids) != len(set(ids)):
        raise TaxonomyError("case ids must be unique")

    known = {intent.name for intent in taxonomy.intents}
    by_contrast: dict[str, list[Case]] = {}
    for case in case_list:
        by_contrast.setdefault(case.contrast_group_id, []).append(case)
        target = case.gold.target_intent
        if target is not None and target not in known:
            raise TaxonomyError(f"{case.id}: unknown Gold target {target}")
        unknown_siblings = set(case.gold.near_oos_sibling_intents) - known
        if unknown_siblings:
            raise TaxonomyError(f"{case.id}: unknown near-OOS siblings {sorted(unknown_siblings)}")
        unknown_competitors = set(case.hard_negative_against) - known
        if unknown_competitors:
            raise TaxonomyError(
                f"{case.id}: unknown hard-negative intents {sorted(unknown_competitors)}"
            )
        if case.context_perturbation is not None:
            unknown_added = set(case.context_perturbation.recent_activities_added) - known
            if unknown_added:
                raise TaxonomyError(
                    f"{case.id}: unknown perturbation activities {sorted(unknown_added)}"
                )

    for case in case_list:
        if "near_oos" not in case.tags:
            continue
        linked_targets = {
            candidate.gold.target_intent
            for candidate in by_contrast[case.contrast_group_id]
            if candidate.gold.decision is Decision.IN_SCOPE
        }
        if not set(case.gold.near_oos_sibling_intents) <= linked_targets:
            raise TaxonomyError(
                f"{case.id}: near-OOS siblings require linked in-scope cases in contrast group"
            )

    context_group_ids = {
        case.contrast_group_id
        for case in case_list
        if {"context_control", "context_distractor"} & set(case.tags)
    }
    for group_id in context_group_ids:
        pair = by_contrast[group_id]
        controls = [case for case in pair if "context_control" in case.tags]
        distractors = [case for case in pair if "context_distractor" in case.tags]
        if len(pair) != 2 or len(controls) != 1 or len(distractors) != 1:
            raise TaxonomyError(
                f"{group_id}: context contrast requires exactly one control and distractor"
            )
        control = controls[0]
        distractor = distractors[0]
        if (
            control.scenario_family_id != distractor.scenario_family_id
            or control.gold != distractor.gold
            or control.hard_negative_against != distractor.hard_negative_against
            or control.context_perturbation != distractor.context_perturbation
            or control.context.conversation != distractor.context.conversation
            or ("multi_turn" in control.tags) != ("multi_turn" in distractor.tags)
            or (set(control.tags) - {"context_control"})
            != (set(distractor.tags) - {"context_distractor"})
            or control.context.recent_activities
            or control.context == distractor.context
        ):
            raise TaxonomyError(
                f"{group_id}: context contrast changes more than background context"
            )
        perturbation = control.context_perturbation
        if perturbation is None:
            raise TaxonomyError(f"{group_id}: context contrast lacks registered perturbation")
        if distractor.context.state_summary != (
            f"{perturbation.state_summary_prefix}{control.context.state_summary}"
        ) or distractor.context.recent_activities != [
            *control.context.recent_activities,
            *perturbation.recent_activities_added,
        ]:
            raise TaxonomyError(
                f"{group_id}: context contrast does not match its registered perturbation"
            )


def validate_prediction(prediction: Prediction, taxonomy: Taxonomy) -> None:
    known = {intent.name for intent in taxonomy.intents}
    referenced = set(prediction.candidate_intents or [])
    if prediction.predicted_intent is not None:
        referenced.add(prediction.predicted_intent)
    unknown = referenced - known
    if unknown:
        raise TaxonomyError(f"prediction references unknown intents: {sorted(unknown)}")
