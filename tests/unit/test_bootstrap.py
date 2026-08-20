from __future__ import annotations

from collections.abc import Callable

import pytest

from intentbench.bootstrap import (
    BOOTSTRAP_SEED,
    InsufficientClustersError,
    SplitLeakageError,
    assign_relationship_clusters,
    paired_cluster_bootstrap,
    type7_quantile,
    validate_split_integrity,
)
from intentbench.metrics import hem
from intentbench.schemas import Case, Decision, Prediction


def prediction(case_id: str, decision: Decision) -> Prediction:
    return Prediction.model_validate(
        {
            "case_id": case_id,
            "decision": decision,
            "predicted_intent": None,
            "candidate_intents": [],
            "slots": {
                "desired_experience": None,
                "object": None,
                "horizon": "now",
            },
            "reason_short": "fixture",
        }
    )


def test_relationship_components_are_transitive_and_ignore_source(
    make_case: Callable[..., Case],
) -> None:
    cases = [
        make_case("kir-test-001", scenario="shared-scenario"),
        make_case("kir-test-002", scenario="shared-scenario", contrast="bridge-contrast"),
        make_case("kir-test-003", contrast="bridge-contrast"),
        make_case("kir-test-004"),
    ]
    clustered = assign_relationship_clusters(cases)
    groups = {case.id: case.split_group_id for case in clustered}
    assert groups["kir-test-001"] == "sg-kir-test-001"
    assert groups["kir-test-001"] == groups["kir-test-002"] == groups["kir-test-003"]
    assert groups["kir-test-004"] != groups["kir-test-001"]
    assert all(case.bootstrap_cluster_id == case.split_group_id for case in clustered)


def test_type7_quantile_uses_linear_interpolation() -> None:
    assert type7_quantile([0.0, 10.0], 0.25) == 2.5


def test_paired_cluster_bootstrap_is_deterministic_and_paired(
    make_case: Callable[..., Case],
) -> None:
    cases = [make_case(f"kir-test-{index:03d}", cluster=f"sg-{index:03d}") for index in range(1, 9)]
    left = [prediction(case.id, Decision.OOS) for case in cases]
    right = [prediction(case.id, Decision.NO_INTENT) for case in cases]
    first = paired_cluster_bootstrap(cases, left, right, hem, iterations=200)
    second = paired_cluster_bootstrap(cases, left, right, hem, iterations=200)
    assert first == second
    assert first.seed == BOOTSTRAP_SEED
    assert (first.point, first.lower, first.upper) == (1.0, 1.0, 1.0)


def test_required_slice_needs_eight_clusters(make_case: Callable[..., Case]) -> None:
    cases = [make_case(f"kir-test-{index:03d}", cluster=f"sg-{index:03d}") for index in range(1, 8)]
    predictions = [prediction(case.id, Decision.NO_INTENT) for case in cases]
    with pytest.raises(InsufficientClustersError, match="minimum is 8"):
        paired_cluster_bootstrap(cases, predictions, predictions, hem, iterations=10)


def test_split_integrity_rejects_connected_cases_across_dev_and_test(
    make_case: Callable[..., Case],
) -> None:
    cases = assign_relationship_clusters(
        [
            make_case("kir-test-001", scenario="shared"),
            make_case("kir-test-002", scenario="shared"),
        ]
    )
    second = cases[1].model_dump()
    second["split"] = "dev"
    cases[1] = Case.model_validate(second)
    with pytest.raises(SplitLeakageError, match="cross splits"):
        validate_split_integrity(cases)
