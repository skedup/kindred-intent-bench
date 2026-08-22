from __future__ import annotations

from pathlib import Path

import yaml

from intentbench.b0 import B0Config, current_decision_text, predict_b0
from intentbench.schemas import Context, ConversationTurn, Decision, Horizon


def registered_config() -> B0Config:
    payload = yaml.safe_load(Path("configs/kir-pilot-v2-experiment.yaml").read_text())
    return B0Config.model_validate(payload["b0"])


def context(text: str) -> Context:
    return Context(state_summary=text, conversation=[], recent_activities=[])


def test_b0_registered_rules_cover_all_four_route_decisions() -> None:
    config = registered_config()
    examples = {
        "在家煮碗面吃。": (Decision.IN_SCOPE, "eat_at_home"),
        "打开主机玩一局电子游戏。": (Decision.OOS, None),
        "现在没有下一步计划。": (Decision.NO_INTENT, None),
        "散步还是自己画一张都可以。": (Decision.AMBIGUOUS, None),
    }
    for index, (text, expected) in enumerate(examples.items(), 1):
        prediction = predict_b0(
            case_id=f"fixture-b0-{index:02d}", context=context(text), config=config
        )
        assert (prediction.decision, prediction.predicted_intent) == expected


def test_b0_horizon_uses_latest_decision_instead_of_stale_history() -> None:
    value = Context(
        state_summary="正在考虑下一项活动。",
        conversation=[
            ConversationTurn(role="assistant", content="我原来想晚点再决定。"),
            ConversationTurn(role="user", content="以你此刻的决定为准。"),
            ConversationTurn(role="assistant", content="现在去美术馆现场看展。"),
        ],
        recent_activities=[],
    )
    assert current_decision_text(value) == "现在去美术馆现场看展。"
    prediction = predict_b0(case_id="fixture-b0-recency", context=value, config=registered_config())
    assert prediction.decision is Decision.IN_SCOPE
    assert prediction.slots is not None
    assert prediction.slots.horizon is Horizon.NOW
