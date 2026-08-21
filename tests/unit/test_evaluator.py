from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from intentbench.evaluator import RunContractError, evaluate_run
from intentbench.schemas import DIAGNOSTIC_SLICE_TAGS, Case, Decision, Prediction, PredictionStatus
from intentbench.taxonomy import load_taxonomy


def prediction(case_id: str, decision: Decision, target: str | None = None) -> Prediction:
    return Prediction.model_validate(
        {
            "case_id": case_id,
            "decision": decision,
            "predicted_intent": target,
            "candidate_intents": [],
            "slots": {
                "desired_experience": None,
                "object": None,
                "horizon": "now",
            },
            "reason_short": "fixture",
        }
    )


def test_failures_stay_in_complete_gold_universe(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    cases = [
        make_case("kir-test-001", Decision.IN_SCOPE, target_intent="rest"),
        make_case("kir-test-002", Decision.OOS),
        make_case("kir-test-003", Decision.NO_INTENT),
        make_case("kir-test-004", Decision.AMBIGUOUS),
    ]
    predictions = [
        prediction("kir-test-001", Decision.IN_SCOPE, "rest"),
        Prediction(
            case_id="kir-test-002",
            status=PredictionStatus.PROVIDER_FAILURE,
            error_type="timeout",
        ),
        Prediction(
            case_id="kir-test-003",
            status=PredictionStatus.SCHEMA_INVALID,
            error_type="invalid_json",
        ),
    ]
    result = evaluate_run(cases, predictions, taxonomy)
    assert result["gold_case_count"] == 4
    assert result["prediction_count"] == 3
    assert result["hierarchical_exact_match"] == 0.25
    assert result["failure_counts"] == {
        "missing": 1,
        "provider_failure": 1,
        "schema_invalid": 1,
    }
    confusion = result["confusion"]
    assert confusion["oos"]["invalid"] == 1
    assert confusion["no_intent"]["invalid"] == 1
    assert confusion["ambiguous"]["invalid"] == 1


@pytest.mark.parametrize("kind", ["duplicate", "extra"])
def test_duplicate_or_extra_prediction_invalidates_run(
    make_case: Callable[..., Case], kind: str
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    cases = [make_case("kir-test-001")]
    predictions = [prediction("kir-test-001", Decision.NO_INTENT)]
    if kind == "duplicate":
        predictions.append(prediction("kir-test-001", Decision.NO_INTENT))
    else:
        predictions.append(prediction("kir-test-999", Decision.NO_INTENT))
    with pytest.raises(RunContractError, match=kind):
        evaluate_run(cases, predictions, taxonomy)


def test_intent_macro_f1_uses_only_gold_in_scope_and_all_eight_labels(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    cases = [
        make_case(f"kir-test-{index:03d}", Decision.IN_SCOPE, target_intent=intent)
        for index, intent in enumerate(
            (definition.name for definition in taxonomy.intents), start=1
        )
    ]
    cases.append(make_case("kir-test-099", Decision.OOS))
    predictions = [
        prediction(case.id, Decision.IN_SCOPE, case.gold.target_intent)
        for case in cases
        if case.gold.decision is Decision.IN_SCOPE
    ]
    predictions.append(prediction("kir-test-099", Decision.NO_INTENT))
    result = evaluate_run(cases, predictions, taxonomy)
    assert result["in_scope_intent_macro_f1"] == 1.0


def test_near_oos_and_sibling_false_reject_are_exact(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    cases = [
        make_case(
            "kir-test-001",
            Decision.OOS,
            tags=["near_oos", "hard_negative", "hypothesized_weak_model_probe"],
            siblings=["take_a_walk"],
            hard_negative_against=["take_a_walk"],
            contrast="walk-vs-run",
        ),
        make_case(
            "kir-test-002",
            Decision.IN_SCOPE,
            target_intent="take_a_walk",
            contrast="walk-vs-run",
        ),
    ]
    predictions = [
        prediction("kir-test-001", Decision.OOS),
        prediction("kir-test-002", Decision.OOS),
    ]
    result = evaluate_run(cases, predictions, taxonomy)
    assert result["near_oos_recall"] == 1.0
    assert result["sibling_id_false_reject_rate"] == 1.0


def test_context_pairs_and_generic_slice_metrics_are_reported(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    cases = [
        make_case(
            "kir-test-001",
            Decision.IN_SCOPE,
            target_intent="rest",
            tags=["context_control"],
            scenario="context-rest",
            contrast="context-rest",
            cluster="sg-context-rest",
        ),
        make_case(
            "kir-test-002",
            Decision.IN_SCOPE,
            target_intent="rest",
            state_summary="窗外正在下雨。我现在没有具体想做的事情。",
            tags=["context_distractor"],
            scenario="context-rest",
            contrast="context-rest",
            cluster="sg-context-rest",
        ),
    ]
    predictions = [
        prediction("kir-test-001", Decision.IN_SCOPE, "rest"),
        prediction("kir-test-002", Decision.OOS),
    ]
    result = evaluate_run(cases, predictions, taxonomy)
    assert result["context_control_error_count"] == 0
    assert result["context_distractor_error_count"] == 1
    assert result["context_distractor_pair_count"] == 1
    assert result["context_distractor_flip_rate"] == 1.0
    assert result["context_distractor_adverse_flip_rate"] == 1.0
    assert result["context_distractor_error_delta"] == 1
    assert result["slice_metrics"]["context_distractor"] == {
        "case_count": 1,
        "cluster_count": 1,
        "hierarchical_exact_match": 0.0,
        "decision_accuracy": 0.0,
    }
    assert result["slice_confusions"]["context_distractor"]["in_scope"]["oos"] == 1
    assert result["slice_overlap_matrix"]["context_control"]["context_control"] == 1
    assert result["slice_overlap_matrix"]["context_control"]["context_distractor"] == 0
    assert set(result["slice_metrics"]) == set(DIAGNOSTIC_SLICE_TAGS)
    assert set(result["slice_confusions"]) == set(DIAGNOSTIC_SLICE_TAGS)
    assert set(result["slice_overlap_matrix"]) == set(DIAGNOSTIC_SLICE_TAGS)


def test_slot_and_quiet_control_specialty_metrics_are_reported(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    slot_case = make_case(
        "kir-test-001",
        Decision.IN_SCOPE,
        target_intent="rest",
        tags=["slot_required"],
    )
    payload = slot_case.model_dump(mode="json")
    payload["gold"]["slots"]["desired_experience"] = "安静休息"
    payload["gold"]["slots"]["object"] = "休息时段"
    slot_case = Case.model_validate(payload)
    quiet_case = make_case(
        "kir-test-002",
        Decision.NO_INTENT,
        tags=["quiet_control"],
    )
    result = evaluate_run(
        [slot_case, quiet_case],
        [
            prediction("kir-test-001", Decision.IN_SCOPE, "rest"),
            prediction("kir-test-002", Decision.NO_INTENT),
        ],
        taxonomy,
    )
    assert result["slot_required_completeness"] == 0.0
    assert result["slot_required_horizon_accuracy"] == 1.0
    assert result["quiet_control_no_intent_recall"] == 1.0


def test_hard_negative_reports_competing_intent_hits(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    case = make_case(
        "kir-test-001",
        Decision.NO_INTENT,
        tags=["hard_negative"],
        hard_negative_against=["rest"],
    )
    result = evaluate_run([case], [prediction("kir-test-001", Decision.IN_SCOPE, "rest")], taxonomy)
    assert result["hard_negative_target_hit_count"] == 1
    assert result["hard_negative_target_hit_rate"] == 1.0
