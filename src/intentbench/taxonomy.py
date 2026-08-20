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


def validate_prediction(prediction: Prediction, taxonomy: Taxonomy) -> None:
    known = {intent.name for intent in taxonomy.intents}
    referenced = set(prediction.candidate_intents or [])
    if prediction.predicted_intent is not None:
        referenced.add(prediction.predicted_intent)
    unknown = referenced - known
    if unknown:
        raise TaxonomyError(f"prediction references unknown intents: {sorted(unknown)}")
