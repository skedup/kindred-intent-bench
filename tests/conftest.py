from __future__ import annotations

from collections.abc import Callable

import pytest

from intentbench.schemas import Case, Decision, Gold, Horizon, Slots


@pytest.fixture
def make_case() -> Callable[..., Case]:
    def factory(
        case_id: str,
        decision: Decision = Decision.NO_INTENT,
        *,
        target_intent: str | None = None,
        text: str = "我现在没有具体想做的事情。",
        tags: list[str] | None = None,
        siblings: list[str] | None = None,
        scenario: str | None = None,
        contrast: str | None = None,
        paraphrase: str | None = None,
        cluster: str | None = None,
    ) -> Case:
        if decision is Decision.IN_SCOPE and target_intent is None:
            target_intent = "rest"
        payload = {
            "id": case_id,
            "split": "test",
            "context": {
                "state_summary": text,
                "conversation": [],
                "recent_activities": [],
            },
            "gold": Gold(
                decision=decision,
                target_intent=target_intent,
                evidence_quote=text,
                near_oos_sibling_intents=siblings or [],
                slots=Slots(
                    desired_experience=None,
                    object=None,
                    horizon=Horizon.NOW,
                ),
            ),
            "tags": tags or [],
            "scenario_family_id": scenario or f"scenario-{case_id}",
            "contrast_group_id": contrast or f"contrast-{case_id}",
            "paraphrase_cluster_id": paraphrase or f"paraphrase-{case_id}",
            "split_group_id": cluster,
            "bootstrap_cluster_id": cluster,
            "source": "synthetic_fixture",
            "annotator_id": "test",
            "adjudication_status": "reviewed",
            "annotation_note": "unit test fixture",
        }
        return Case.model_validate(payload)

    return factory
