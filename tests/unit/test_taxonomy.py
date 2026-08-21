from collections.abc import Callable
from pathlib import Path

import pytest

from intentbench.schemas import INTENT_NAMES, Case, Decision, Prediction
from intentbench.taxonomy import TaxonomyError, load_taxonomy, validate_cases, validate_prediction


def test_frozen_taxonomy_has_exactly_eight_complete_intents() -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    assert tuple(intent.name for intent in taxonomy.intents) == INTENT_NAMES
    assert all(3 <= len(intent.canonical_examples) <= 5 for intent in taxonomy.intents)
    assert all(intent.inclusion_criteria for intent in taxonomy.intents)
    assert all(intent.exclusion_criteria for intent in taxonomy.intents)


def test_unknown_prediction_target_is_rejected_by_taxonomy() -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    prediction = Prediction.model_validate(
        {
            "case_id": "kir-test-001",
            "decision": Decision.IN_SCOPE,
            "predicted_intent": "invented_activity",
            "candidate_intents": [],
            "slots": {
                "desired_experience": None,
                "object": None,
                "horizon": "now",
            },
            "reason_short": "unknown target",
        }
    )
    with pytest.raises(TaxonomyError, match="unknown intents"):
        validate_prediction(prediction, taxonomy)


def test_formal_context_contrast_requires_one_matched_pair(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    control = make_case(
        "kir-test-001",
        Decision.IN_SCOPE,
        target_intent="rest",
        tags=["context_control"],
        scenario="context-rest",
        contrast="context-rest",
    )
    distractor = make_case(
        "kir-test-002",
        Decision.IN_SCOPE,
        target_intent="take_a_walk",
        tags=["context_distractor"],
        scenario="context-rest",
        contrast="context-rest",
    )
    with pytest.raises(TaxonomyError, match="changes more than background"):
        validate_cases([control, distractor], taxonomy)


def test_formal_context_contrast_must_apply_registered_delta(
    make_case: Callable[..., Case],
) -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v1.yaml"))
    control = make_case(
        "kir-test-001",
        Decision.IN_SCOPE,
        target_intent="rest",
        tags=["context_control"],
        scenario="context-rest",
        contrast="context-rest",
    )
    distractor = make_case(
        "kir-test-002",
        Decision.IN_SCOPE,
        target_intent="rest",
        state_summary=(
            "窗外正在下雨。我现在没有具体想做的事情。不过最终取消原计划并决定留在家里做饭。"
        ),
        tags=["context_distractor"],
        scenario="context-rest",
        contrast="context-rest",
    )
    with pytest.raises(TaxonomyError, match="registered perturbation"):
        validate_cases([control, distractor], taxonomy)
