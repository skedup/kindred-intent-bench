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
from intentbench.schemas import (
    DIAGNOSTIC_SLICE_TAGS,
    Case,
    Decision,
    Prediction,
    PredictionStatus,
    Taxonomy,
)
from intentbench.taxonomy import validate_cases, validate_prediction

DIAGNOSTIC_SLICES = DIAGNOSTIC_SLICE_TAGS


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
    context_controls = [case for case in cases if "context_control" in case.tags]
    context_control_error_count = sum(
        hierarchical_exact_match(case, indexed.get(case.id)) == 0 for case in context_controls
    )
    controls_by_contrast = {case.contrast_group_id: case for case in context_controls}
    context_pairs = [
        (controls_by_contrast[case.contrast_group_id], case)
        for case in cases
        if "context_distractor" in case.tags and case.contrast_group_id in controls_by_contrast
    ]

    def prediction_signature(case: Case) -> tuple[object, ...]:
        prediction = indexed.get(case.id)
        if prediction is None:
            return ("missing",)
        return (
            prediction.status,
            prediction.decision,
            prediction.predicted_intent,
            tuple(sorted(prediction.candidate_intents or [])),
        )

    context_flip_count = sum(
        prediction_signature(control) != prediction_signature(distractor)
        for control, distractor in context_pairs
    )
    context_adverse_flip_count = sum(
        hierarchical_exact_match(control, indexed.get(control.id)) == 1
        and hierarchical_exact_match(distractor, indexed.get(distractor.id)) == 0
        for control, distractor in context_pairs
    )

    slice_metrics: dict[str, dict[str, float | int]] = {}
    slice_confusions: dict[str, dict[str, dict[str, int]]] = {}
    for tag in DIAGNOSTIC_SLICES:
        slice_cases = [case for case in cases if tag in case.tags]
        cluster_ids = {
            case.bootstrap_cluster_id
            for case in slice_cases
            if case.bootstrap_cluster_id is not None
        }
        decision_correct = sum(
            indexed.get(case.id) is not None
            and indexed[case.id].status is PredictionStatus.SUCCESS
            and indexed[case.id].decision is case.gold.decision
            for case in slice_cases
        )
        slice_metrics[tag] = {
            "case_count": len(slice_cases),
            "cluster_count": len(cluster_ids),
            "hierarchical_exact_match": hem(slice_cases, indexed) if slice_cases else 0.0,
            "decision_accuracy": decision_correct / len(slice_cases) if slice_cases else 0.0,
        }
        slice_confusions[tag] = confusion_matrix(slice_cases, indexed)

    slice_overlap_matrix = {
        left: {
            right: sum({left, right} <= set(case.tags) for case in cases)
            for right in DIAGNOSTIC_SLICES
        }
        for left in DIAGNOSTIC_SLICES
    }
    hard_negative_cases = [case for case in cases if "hard_negative" in case.tags]
    hard_negative_target_hit_count = sum(
        indexed.get(case.id) is not None
        and indexed[case.id].status is PredictionStatus.SUCCESS
        and indexed[case.id].decision is Decision.IN_SCOPE
        and indexed[case.id].predicted_intent in case.hard_negative_against
        for case in hard_negative_cases
    )

    def required_slots_complete(case: Case) -> bool:
        prediction = indexed.get(case.id)
        if (
            prediction is None
            or prediction.status is not PredictionStatus.SUCCESS
            or prediction.slots is None
        ):
            return False
        return (
            case.gold.slots.desired_experience is None
            or prediction.slots.desired_experience is not None
        ) and (case.gold.slots.object is None or prediction.slots.object is not None)

    def required_horizon_correct(case: Case) -> bool:
        prediction = indexed.get(case.id)
        return bool(
            prediction is not None
            and prediction.status is PredictionStatus.SUCCESS
            and prediction.slots is not None
            and prediction.slots.horizon is case.gold.slots.horizon
        )

    slot_required_cases = [case for case in cases if "slot_required" in case.tags]
    slot_required_complete_count = sum(map(required_slots_complete, slot_required_cases))
    slot_required_horizon_correct_count = sum(map(required_horizon_correct, slot_required_cases))
    quiet_control_cases = [case for case in cases if "quiet_control" in case.tags]
    quiet_control_correct_count = sum(
        indexed.get(case.id) is not None
        and indexed[case.id].status is PredictionStatus.SUCCESS
        and indexed[case.id].decision is Decision.NO_INTENT
        for case in quiet_control_cases
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
        "context_control_error_count": context_control_error_count,
        "context_distractor_pair_count": len(context_pairs),
        "context_distractor_flip_rate": (
            context_flip_count / len(context_pairs) if context_pairs else 0.0
        ),
        "context_distractor_adverse_flip_rate": (
            context_adverse_flip_count / len(context_pairs) if context_pairs else 0.0
        ),
        "context_distractor_error_delta": (
            context_distractor_error_count - context_control_error_count
        ),
        "hard_negative_target_hit_count": hard_negative_target_hit_count,
        "hard_negative_target_hit_rate": (
            hard_negative_target_hit_count / len(hard_negative_cases)
            if hard_negative_cases
            else 0.0
        ),
        "slot_required_completeness": (
            slot_required_complete_count / len(slot_required_cases) if slot_required_cases else 0.0
        ),
        "slot_required_horizon_accuracy": (
            slot_required_horizon_correct_count / len(slot_required_cases)
            if slot_required_cases
            else 0.0
        ),
        "quiet_control_no_intent_recall": (
            quiet_control_correct_count / len(quiet_control_cases) if quiet_control_cases else 0.0
        ),
        "slice_metrics": slice_metrics,
        "slice_confusions": slice_confusions,
        "slice_overlap_matrix": slice_overlap_matrix,
        "schema_invalid_count": failures["schema_invalid"],
        "failure_counts": failures,
        "confusion": confusion_matrix(cases, indexed),
    }
