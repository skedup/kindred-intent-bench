"""Deterministic metrics over the complete Gold universe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from intentbench.schemas import (
    INTENT_NAMES,
    Case,
    Decision,
    Prediction,
    PredictionStatus,
)

INVALID_LABEL = "invalid"
DECISION_LABELS: tuple[str, ...] = tuple(decision.value for decision in Decision)


def prediction_decision_label(prediction: Prediction | None) -> str:
    if prediction is None or prediction.status is not PredictionStatus.SUCCESS:
        return INVALID_LABEL
    assert prediction.decision is not None
    return prediction.decision.value


def prediction_intent_label(prediction: Prediction | None) -> str:
    if (
        prediction is None
        or prediction.status is not PredictionStatus.SUCCESS
        or prediction.decision is not Decision.IN_SCOPE
        or prediction.predicted_intent is None
    ):
        return INVALID_LABEL
    return prediction.predicted_intent


def hierarchical_exact_match(case: Case, prediction: Prediction | None) -> int:
    if prediction is None or prediction.status is not PredictionStatus.SUCCESS:
        return 0
    if prediction.decision is not case.gold.decision:
        return 0
    if case.gold.decision is Decision.IN_SCOPE:
        return int(prediction.predicted_intent == case.gold.target_intent)
    return 1


def precision_recall_f1(
    gold_labels: Sequence[str], prediction_labels: Sequence[str], label: str
) -> tuple[float, float, float]:
    if len(gold_labels) != len(prediction_labels):
        raise ValueError("gold and prediction label sequences must be aligned")
    true_positive = sum(
        gold == label and prediction == label
        for gold, prediction in zip(gold_labels, prediction_labels, strict=True)
    )
    false_positive = sum(
        gold != label and prediction == label
        for gold, prediction in zip(gold_labels, prediction_labels, strict=True)
    )
    false_negative = sum(
        gold == label and prediction != label
        for gold, prediction in zip(gold_labels, prediction_labels, strict=True)
    )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def fixed_macro_f1(
    gold_labels: Sequence[str], prediction_labels: Sequence[str], labels: Sequence[str]
) -> float:
    if not labels:
        raise ValueError("fixed label set must not be empty")
    return sum(
        precision_recall_f1(gold_labels, prediction_labels, label)[2] for label in labels
    ) / len(labels)


def hem(cases: Sequence[Case], predictions: Mapping[str, Prediction]) -> float:
    if not cases:
        return 0.0
    return sum(hierarchical_exact_match(case, predictions.get(case.id)) for case in cases) / len(
        cases
    )


def decision_macro_f1(cases: Sequence[Case], predictions: Mapping[str, Prediction]) -> float:
    gold_labels = [case.gold.decision.value for case in cases]
    predicted_labels = [prediction_decision_label(predictions.get(case.id)) for case in cases]
    return fixed_macro_f1(gold_labels, predicted_labels, DECISION_LABELS)


def intent_macro_f1(
    cases: Sequence[Case],
    predictions: Mapping[str, Prediction],
    intent_names: Sequence[str] = INTENT_NAMES,
) -> float:
    in_scope = [case for case in cases if case.gold.decision is Decision.IN_SCOPE]
    gold_labels = [str(case.gold.target_intent) for case in in_scope]
    predicted_labels = [prediction_intent_label(predictions.get(case.id)) for case in in_scope]
    return fixed_macro_f1(gold_labels, predicted_labels, intent_names)


def decision_prf(
    cases: Sequence[Case], predictions: Mapping[str, Prediction], decision: Decision
) -> tuple[float, float, float]:
    gold_labels = [case.gold.decision.value for case in cases]
    predicted_labels = [prediction_decision_label(predictions.get(case.id)) for case in cases]
    return precision_recall_f1(gold_labels, predicted_labels, decision.value)


def per_intent_recall(
    cases: Sequence[Case], predictions: Mapping[str, Prediction], intent_names: Sequence[str]
) -> dict[str, float]:
    in_scope = [case for case in cases if case.gold.decision is Decision.IN_SCOPE]
    gold_labels = [str(case.gold.target_intent) for case in in_scope]
    predicted_labels = [prediction_intent_label(predictions.get(case.id)) for case in in_scope]
    return {
        intent: precision_recall_f1(gold_labels, predicted_labels, intent)[1]
        for intent in intent_names
    }


def confusion_matrix(
    cases: Sequence[Case], predictions: Mapping[str, Prediction]
) -> dict[str, dict[str, int]]:
    columns = (*DECISION_LABELS, INVALID_LABEL)
    matrix = {gold: {prediction: 0 for prediction in columns} for gold in DECISION_LABELS}
    for case in cases:
        matrix[case.gold.decision.value][prediction_decision_label(predictions.get(case.id))] += 1
    return matrix


def failure_counts(cases: Sequence[Case], predictions: Mapping[str, Prediction]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in cases:
        prediction = predictions.get(case.id)
        if prediction is None:
            counts["missing"] += 1
        elif prediction.status is PredictionStatus.PROVIDER_FAILURE:
            counts["provider_failure"] += 1
        elif prediction.status is PredictionStatus.SCHEMA_INVALID:
            counts["schema_invalid"] += 1
    return {
        "missing": counts["missing"],
        "provider_failure": counts["provider_failure"],
        "schema_invalid": counts["schema_invalid"],
    }
