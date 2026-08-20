"""Frozen-embedding B1 baseline with explicit open-set gates."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from intentbench.schemas import Decision

Vector = Sequence[float]


def l2_normalize(vector: Vector) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("embedding must have a finite non-zero L2 norm")
    return tuple(value / norm for value in vector)


def normalized_centroid(vectors: Iterable[Vector]) -> tuple[float, ...]:
    normalized = [l2_normalize(vector) for vector in vectors]
    if not normalized:
        raise ValueError("a prototype needs at least one embedding")
    dimension = len(normalized[0])
    if any(len(vector) != dimension for vector in normalized):
        raise ValueError("prototype embeddings must have one dimension")
    centroid = tuple(
        sum(vector[index] for vector in normalized) / len(normalized) for index in range(dimension)
    )
    return l2_normalize(centroid)


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("cosine inputs must have one dimension")
    left_unit = l2_normalize(left)
    right_unit = l2_normalize(right)
    return sum(a * b for a, b in zip(left_unit, right_unit, strict=True))


@dataclass(frozen=True)
class B1Thresholds:
    tau_actionability: float
    tau_oos_margin: float
    tau_activity_min: float
    tau_ambiguity: float

    def ordered_tuple(self) -> tuple[float, ...]:
        values = vars(self)
        return tuple(values[name] for name in sorted(values))


@dataclass(frozen=True)
class B1Scores:
    activity_scores: Mapping[str, float]
    activity_score: float
    second_activity_score: float
    no_intent_score: float
    oos_score: float
    actionability_score: float

    @property
    def top_intent(self) -> str:
        return min(
            (
                intent
                for intent, score in self.activity_scores.items()
                if score == self.activity_score
            ),
            default="",
        )


def score_embedding(
    query: Vector,
    activity_centroids: Mapping[str, Vector],
    no_intent_centroid: Vector,
    oos_centroid: Vector,
) -> B1Scores:
    if len(activity_centroids) < 2:
        raise ValueError("B1 ambiguity gate needs at least two activity prototypes")
    activity_scores = {
        intent: cosine_similarity(query, centroid)
        for intent, centroid in activity_centroids.items()
    }
    ranked = sorted(activity_scores.values(), reverse=True)
    activity_score = ranked[0]
    no_intent_score = cosine_similarity(query, no_intent_centroid)
    oos_score = cosine_similarity(query, oos_centroid)
    return B1Scores(
        activity_scores=activity_scores,
        activity_score=activity_score,
        second_activity_score=ranked[1],
        no_intent_score=no_intent_score,
        oos_score=oos_score,
        actionability_score=max(activity_score, oos_score) - no_intent_score,
    )


def route_scores(scores: B1Scores, thresholds: B1Thresholds) -> tuple[Decision, str | None]:
    """Apply actionability, OOS, ambiguity, then in-scope gates in that order."""

    if scores.actionability_score < thresholds.tau_actionability:
        return Decision.NO_INTENT, None
    if (
        scores.oos_score - scores.activity_score >= thresholds.tau_oos_margin
        or scores.activity_score < thresholds.tau_activity_min
    ):
        return Decision.OOS, None
    if scores.activity_score - scores.second_activity_score < thresholds.tau_ambiguity:
        return Decision.AMBIGUOUS, None
    return Decision.IN_SCOPE, scores.top_intent


def select_thresholds(
    candidates: Iterable[B1Thresholds],
    score_candidate: Callable[[B1Thresholds], tuple[float, float]],
) -> B1Thresholds:
    """Maximize dev HEM, then decision Macro-F1, then smallest sorted-parameter tuple."""

    scored = [(score_candidate(candidate), candidate) for candidate in candidates]
    if not scored:
        raise ValueError("threshold grid must not be empty")
    return min(
        scored,
        key=lambda item: (-item[0][0], -item[0][1], item[1].ordered_tuple()),
    )[1]
