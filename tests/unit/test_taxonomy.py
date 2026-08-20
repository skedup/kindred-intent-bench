from pathlib import Path

import pytest

from intentbench.schemas import INTENT_NAMES, Decision, Prediction
from intentbench.taxonomy import TaxonomyError, load_taxonomy, validate_prediction


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
