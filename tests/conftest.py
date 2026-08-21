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
        state_summary: str | None = None,
        tags: list[str] | None = None,
        siblings: list[str] | None = None,
        hard_negative_against: list[str] | None = None,
        context_perturbation: dict[str, object] | None = None,
        scenario: str | None = None,
        contrast: str | None = None,
        paraphrase: str | None = None,
        cluster: str | None = None,
    ) -> Case:
        if decision is Decision.IN_SCOPE and target_intent is None:
            target_intent = "rest"
        resolved_tags = tags or []
        resolved_contrast = contrast or f"contrast-{case_id}"
        if context_perturbation is None and {
            "context_control",
            "context_distractor",
        } & set(resolved_tags):
            context_perturbation = {
                "id": f"perturbation-{resolved_contrast}",
                "state_summary_prefix": "窗外正在下雨。",
                "recent_activities_added": [],
            }
        payload = {
            "id": case_id,
            "split": "test",
            "context": {
                "state_summary": state_summary or text,
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
            "tags": resolved_tags,
            "hard_negative_against": hard_negative_against or [],
            "context_perturbation": context_perturbation,
            "scenario_family_id": scenario or f"scenario-{case_id}",
            "contrast_group_id": resolved_contrast,
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
