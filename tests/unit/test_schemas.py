from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from intentbench.schemas import Case, Decision, Prediction, PredictionStatus, RunManifest


def successful_prediction(decision: Decision, target: str | None = None) -> Prediction:
    candidates = ["rest", "take_a_walk"] if decision is Decision.AMBIGUOUS else []
    return Prediction.model_validate(
        {
            "case_id": "kir-test-001",
            "status": "success",
            "decision": decision,
            "predicted_intent": target,
            "candidate_intents": candidates,
            "slots": {
                "desired_experience": None,
                "object": None,
                "horizon": "now",
            },
            "reason_short": "可审计的短解释",
        }
    )


@pytest.mark.parametrize("decision", list(Decision))
def test_prediction_truth_table_accepts_all_decisions(decision: Decision) -> None:
    target = "rest" if decision is Decision.IN_SCOPE else None
    assert successful_prediction(decision, target).decision is decision


def test_prediction_truth_table_rejects_target_on_oos() -> None:
    payload = successful_prediction(Decision.OOS).model_dump()
    payload["predicted_intent"] = "rest"
    with pytest.raises(ValidationError, match="must not contain predicted_intent"):
        Prediction.model_validate(payload)


def test_failed_prediction_has_no_semantic_fields() -> None:
    failure = Prediction(
        case_id="kir-test-001",
        status=PredictionStatus.SCHEMA_INVALID,
        error_type="invalid_json",
    )
    assert failure.decision is None
    payload = failure.model_dump()
    payload["reason_short"] = "silently repaired"
    with pytest.raises(ValidationError, match="must not contain semantic fields"):
        Prediction.model_validate(payload)


def test_reason_short_unicode_limit() -> None:
    payload = successful_prediction(Decision.NO_INTENT).model_dump()
    payload["reason_short"] = "意" * 201
    with pytest.raises(ValidationError):
        Prediction.model_validate(payload)


def test_evidence_uses_nfkc_contiguous_substring(
    make_case: Callable[..., Case],
) -> None:
    case = make_case("kir-test-001", text="我想看ＡＢＣ展览")
    payload = case.model_dump()
    payload["gold"]["evidence_quote"] = "ABC"
    assert Case.model_validate(payload).id == case.id
    payload["gold"]["evidence_quote"] = "不存在的意图"
    with pytest.raises(ValidationError, match="contiguous context substring"):
        Case.model_validate(payload)


def test_near_oos_requires_oos_siblings_and_matching_clusters(
    make_case: Callable[..., Case],
) -> None:
    valid = make_case(
        "kir-test-002",
        Decision.OOS,
        tags=["near_oos", "hard_negative", "hypothesized_weak_model_probe"],
        siblings=["take_a_walk"],
        hard_negative_against=["take_a_walk"],
    )
    assert valid.gold.near_oos_sibling_intents == ["take_a_walk"]
    payload = valid.model_dump()
    payload["gold"]["near_oos_sibling_intents"] = []
    with pytest.raises(ValidationError, match="at least one sibling"):
        Case.model_validate(payload)
    payload = valid.model_dump()
    payload["bootstrap_cluster_id"] = "different"
    payload["split_group_id"] = "same"
    with pytest.raises(ValidationError, match="must equal split_group_id"):
        Case.model_validate(payload)


def test_hard_negative_requires_explicit_competing_intents(
    make_case: Callable[..., Case],
) -> None:
    valid = make_case(
        "kir-test-003",
        Decision.NO_INTENT,
        tags=["hard_negative"],
        hard_negative_against=["rest"],
    )
    assert valid.hard_negative_against == ["rest"]
    payload = valid.model_dump()
    payload["hard_negative_against"] = []
    with pytest.raises(ValidationError, match="requires at least one competing intent"):
        Case.model_validate(payload)


def test_formal_case_rejects_ambiguous_hard_negative_with_one_competitor(
    make_case: Callable[..., Case],
) -> None:
    with pytest.raises(ValidationError, match="at least two competing intents"):
        make_case(
            "kir-test-004",
            Decision.AMBIGUOUS,
            tags=["hard_negative"],
            hard_negative_against=["rest"],
        )


def test_formal_case_rejects_arbitrary_hypothesized_probe(
    make_case: Callable[..., Case],
) -> None:
    with pytest.raises(ValidationError, match="registered component tags"):
        make_case(
            "kir-test-005",
            Decision.NO_INTENT,
            tags=["hypothesized_weak_model_probe"],
        )


def test_formal_near_oos_competitors_must_equal_siblings(
    make_case: Callable[..., Case],
) -> None:
    with pytest.raises(ValidationError, match="must equal sibling intents"):
        make_case(
            "kir-test-006",
            Decision.OOS,
            tags=["near_oos", "hard_negative", "hypothesized_weak_model_probe"],
            siblings=["take_a_walk"],
            hard_negative_against=["rest"],
        )


def test_run_manifest_carries_reproduction_and_budget_identity() -> None:
    digest = "a" * 64
    manifest = RunManifest.model_validate(
        {
            "run_id": "fixture-run",
            "dataset_version": "kir-pilot-v1",
            "dataset_sha256": digest,
            "dataset_freeze_sha256": digest,
            "experiment_lock_sha256": digest,
            "taxonomy_version": "kindred-activity-intents-v1",
            "adapter": "llm_two_stage",
            "adapter_config_sha256": digest,
            "model": "google/gemini-3.6-flash",
            "model_role": "primary_decision",
            "primary_decision_model": "google/gemini-3.6-flash",
            "embedding_model": None,
            "prompt_sha256": digest,
            "raw_response_sha256": digest,
            "evaluator_version": "intent-eval-v1",
            "parameters": {
                "temperature": None,
                "top_p": None,
                "seed": None,
                "max_calls": 2,
                "max_output_tokens_per_call": 400,
                "retry_policy": "none",
                "schema_repair_policy": "reject",
            },
            "pricing_snapshot_date": "2026-08-20",
            "dependency_lock_sha256": digest,
            "started_at": "2026-08-20T00:00:00Z",
            "git_commit": "deadbeef",
        }
    )
    assert manifest.model_role.value == "primary_decision"
