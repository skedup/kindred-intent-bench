"""Evaluator enforcing full-Gold-universe and invalid-output policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from intentbench.metrics import (
    confusion_matrix,
    decision_macro_f1,
    decision_prf,
    failure_counts,
    hem,
    hierarchical_exact_match,
    intent_macro_f1,
    per_intent_recall,
)
from intentbench.schemas import Case, Decision, Prediction, PredictionStatus, Taxonomy
from intentbench.taxonomy import validate_cases, validate_prediction


class RunContractError(ValueError):
    """A run cannot receive an authoritative metric or verdict."""


def index_predictions(
    predictions: Sequence[Prediction], gold_ids: set[str] | None = None
) -> dict[str, Prediction]:
    indexed: dict[str, Prediction] = {}
    duplicates: set[str] = set()
    for prediction in predictions:
        if prediction.case_id in indexed:
            duplicates.add(prediction.case_id)
        indexed[prediction.case_id] = prediction
    if duplicates:
        raise RunContractError(f"duplicate prediction ids: {sorted(duplicates)}")
    if gold_ids is not None:
        extra = set(indexed) - gold_ids
        if extra:
            raise RunContractError(f"extra prediction ids: {sorted(extra)}")
    return indexed


def evaluate_run(
    cases: Sequence[Case], predictions: Sequence[Prediction], taxonomy: Taxonomy
) -> dict[str, Any]:
    validate_cases(cases, taxonomy)
    gold_ids = {case.id for case in cases}
    if len(gold_ids) != len(cases):
        raise RunContractError("duplicate Gold case ids")
    indexed = index_predictions(predictions, gold_ids)
    for prediction in predictions:
        validate_prediction(prediction, taxonomy)

    intents = tuple(intent.name for intent in taxonomy.intents)
    decision_metrics = {
        decision.value: dict(
            zip(
                ("precision", "recall", "f1"),
                decision_prf(cases, indexed, decision),
                strict=True,
            )
        )
        for decision in Decision
    }
    near_oos = [case for case in cases if "near_oos" in case.tags]
    near_oos_recall = (
        sum(
            indexed.get(case.id) is not None
            and indexed[case.id].status is PredictionStatus.SUCCESS
            and indexed[case.id].decision is Decision.OOS
            for case in near_oos
        )
        / len(near_oos)
        if near_oos
        else 0.0
    )

    sibling_pairs = {
        (near_case.contrast_group_id, sibling)
        for near_case in near_oos
        for sibling in near_case.gold.near_oos_sibling_intents
    }
    sibling_cases = [
        case
        for case in cases
        if case.gold.decision is Decision.IN_SCOPE
        and (case.contrast_group_id, case.gold.target_intent) in sibling_pairs
    ]
    sibling_false_reject_rate = (
        sum(
            indexed.get(case.id) is not None
            and indexed[case.id].status is PredictionStatus.SUCCESS
            and indexed[case.id].decision is Decision.OOS
            for case in sibling_cases
        )
        / len(sibling_cases)
        if sibling_cases
        else 0.0
    )

    context_distractor_error_count = sum(
        "context_distractor" in case.tags
        and hierarchical_exact_match(case, indexed.get(case.id)) == 0
        for case in cases
    )
    failures = failure_counts(cases, indexed)
    return {
        "valid_run": True,
        "gold_case_count": len(cases),
        "prediction_count": len(predictions),
        "hierarchical_exact_match": hem(cases, indexed),
        "decision_macro_f1": decision_macro_f1(cases, indexed),
        "in_scope_intent_macro_f1": intent_macro_f1(cases, indexed, intents),
        "decision": decision_metrics,
        "per_intent_recall": per_intent_recall(cases, indexed, intents),
        "near_oos_recall": near_oos_recall,
        "sibling_id_false_reject_rate": sibling_false_reject_rate,
        "context_distractor_error_count": context_distractor_error_count,
        "schema_invalid_count": failures["schema_invalid"],
        "failure_counts": failures,
        "confusion": confusion_matrix(cases, indexed),
    }
