import pytest

from intentbench.b1 import (
    B1Scores,
    B1Thresholds,
    normalized_centroid,
    route_scores,
    select_thresholds,
)
from intentbench.schemas import Decision


def scores(*, actionability: float, activity: float, second: float, oos: float) -> B1Scores:
    return B1Scores(
        activity_scores={"rest": activity, "take_a_walk": second},
        activity_score=activity,
        second_activity_score=second,
        no_intent_score=max(activity, oos) - actionability,
        oos_score=oos,
        actionability_score=actionability,
    )


THRESHOLDS = B1Thresholds(
    tau_actionability=0.1,
    tau_oos_margin=0.05,
    tau_activity_min=0.4,
    tau_ambiguity=0.05,
)


def test_centroid_normalizes_examples_before_and_after_mean() -> None:
    assert normalized_centroid([[10.0, 0.0], [0.0, 2.0]]) == pytest.approx(
        (
            2**-0.5,
            2**-0.5,
        )
    )


def test_b1_gate_order_and_boundaries() -> None:
    assert route_scores(
        scores(actionability=0.09, activity=0.9, second=0.1, oos=0.0), THRESHOLDS
    ) == (Decision.NO_INTENT, None)
    assert route_scores(
        scores(actionability=0.1, activity=0.5, second=0.1, oos=0.55), THRESHOLDS
    ) == (Decision.OOS, None)
    assert route_scores(
        scores(actionability=0.2, activity=0.5, second=0.46, oos=0.0), THRESHOLDS
    ) == (Decision.AMBIGUOUS, None)
    assert route_scores(
        scores(actionability=0.2, activity=0.5, second=0.44, oos=0.0), THRESHOLDS
    ) == (Decision.IN_SCOPE, "rest")


def test_threshold_selection_has_deterministic_lexical_tie_break() -> None:
    larger = B1Thresholds(0.2, 0.0, 0.4, 0.05)
    smaller = B1Thresholds(0.1, 0.0, 0.4, 0.05)
    selected = select_thresholds([larger, smaller], lambda _thresholds: (0.8, 0.7))
    assert selected is smaller
