"""Frozen-embedding B1 baseline with explicit open-set gates."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

from pydantic import Field, model_validator

from intentbench.schemas import Case, Decision, Split, StrictModel

Vector = Sequence[float]


class PrototypeSelectionConfig(StrictModel):
    algorithm: Literal["tag_stratified_unique_cluster_round_robin_case_id_v1"]
    oos_maximum: int = Field(ge=1, le=8)
    oos_strata: list[str] = Field(min_length=1)
    no_intent_maximum: int = Field(ge=1, le=8)
    no_intent_strata: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_strata(self) -> PrototypeSelectionConfig:
        for strata in (self.oos_strata, self.no_intent_strata):
            if len(strata) != len(set(strata)):
                raise ValueError("prototype strata must be unique")
            if "__other__" not in strata:
                raise ValueError("prototype strata must contain the __other__ fallback")
        return self


class B1ThresholdGrid(StrictModel):
    tau_actionability: list[float] = Field(min_length=1)
    tau_oos_margin: list[float] = Field(min_length=1)
    tau_activity_min: list[float] = Field(min_length=1)
    tau_ambiguity: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grid(self) -> B1ThresholdGrid:
        for values in vars(self).values():
            if len(values) != len(set(values)) or any(not math.isfinite(value) for value in values):
                raise ValueError("B1 threshold axes must contain unique finite values")
        return self


class B1Config(StrictModel):
    input_serializer: Literal["context-json-v1"]
    vector_normalization: Literal["l2_each_then_centroid_then_l2"]
    similarity: Literal["cosine"]
    activity_prototypes: Literal["taxonomy_canonical_examples_only"]
    oos_prototypes: Literal["dev_gold_oos_max_8"]
    no_intent_prototypes: Literal["dev_gold_no_intent_max_8"]
    ambiguous_prototype: Literal["none"]
    prototype_selection: PrototypeSelectionConfig
    scores: dict[str, str]
    gate_order: list[str]
    threshold_grid: B1ThresholdGrid
    selection_order: list[str]
    selected_thresholds: dict[str, float] | None
    selected_prototype_case_ids: dict[str, list[str]] | None

    @model_validator(mode="after")
    def validate_registered_contract(self) -> B1Config:
        if self.gate_order != ["actionability", "oos", "ambiguity", "in_scope"]:
            raise ValueError("B1 gate order must be actionability, OOS, ambiguity, in-scope")
        if self.selection_order != [
            "dev_hierarchical_exact_match_desc",
            "dev_decision_macro_f1_desc",
            "sorted_parameter_tuple_asc",
        ]:
            raise ValueError("B1 threshold selection order violates the registered contract")
        threshold_names = {
            "tau_actionability",
            "tau_oos_margin",
            "tau_activity_min",
            "tau_ambiguity",
        }
        if (
            self.selected_thresholds is not None
            and set(self.selected_thresholds) != threshold_names
        ):
            raise ValueError("registered B1 thresholds must contain the complete parameter tuple")
        if self.selected_prototype_case_ids is not None:
            if set(self.selected_prototype_case_ids) != {"oos", "no_intent"}:
                raise ValueError("registered B1 prototypes must contain OOS and no-intent IDs")
            if any(not case_ids for case_ids in self.selected_prototype_case_ids.values()):
                raise ValueError("registered B1 prototype ID lists must not be empty")
        return self


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

    def as_dict(self) -> dict[str, float]:
        return {
            "tau_actionability": self.tau_actionability,
            "tau_oos_margin": self.tau_oos_margin,
            "tau_activity_min": self.tau_activity_min,
            "tau_ambiguity": self.tau_ambiguity,
        }


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

    @property
    def ranked_intents(self) -> tuple[str, ...]:
        return tuple(
            intent
            for intent, _score in sorted(
                self.activity_scores.items(), key=lambda item: (-item[1], item[0])
            )
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


def threshold_candidates(grid: B1ThresholdGrid) -> tuple[B1Thresholds, ...]:
    return tuple(
        B1Thresholds(*values)
        for values in product(
            grid.tau_actionability,
            grid.tau_oos_margin,
            grid.tau_activity_min,
            grid.tau_ambiguity,
        )
    )


def select_prototype_case_ids(
    cases: Sequence[Case],
    *,
    decision: Decision,
    maximum: int,
    strata: Sequence[str],
) -> tuple[str, ...]:
    """Select registered dev-Gold prototypes by tag strata and then case ID."""

    if decision not in {Decision.OOS, Decision.NO_INTENT}:
        raise ValueError("B1 rejection prototypes only support OOS and no-intent")
    if maximum < 1 or maximum > 8:
        raise ValueError("B1 rejection prototype maximum must be in [1, 8]")
    if "__other__" not in strata or len(strata) != len(set(strata)):
        raise ValueError("prototype strata need one unique __other__ fallback")
    candidates = [case for case in cases if case.gold.decision is decision]
    if not candidates:
        raise ValueError(f"dev has no {decision.value} prototype candidates")
    if any(case.split is not Split.DEV for case in candidates):
        raise ValueError("B1 prototype selection accepts dev Gold only")
    if any(case.bootstrap_cluster_id is None for case in candidates):
        raise ValueError("B1 prototype selection requires assigned dev clusters")

    representative_by_cluster: dict[str, Case] = {}
    for case in sorted(candidates, key=lambda item: item.id):
        assert case.bootstrap_cluster_id is not None
        representative_by_cluster.setdefault(case.bootstrap_cluster_id, case)
    representatives = list(representative_by_cluster.values())

    buckets: dict[str, list[str]] = {stratum: [] for stratum in strata}
    for case in representatives:
        stratum = next(
            (
                candidate
                for candidate in strata
                if candidate != "__other__" and candidate in case.tags
            ),
            "__other__",
        )
        buckets[stratum].append(case.id)

    selected: list[str] = []
    while len(selected) < min(maximum, len(representatives)):
        added = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < maximum:
                selected.append(buckets[stratum].pop(0))
                added = True
        if not added:
            break
    return tuple(selected)
