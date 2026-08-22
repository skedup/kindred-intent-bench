"""Frozen lexical B0 baseline over the public Context contract only."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from intentbench.schemas import Context, Decision, Horizon, Prediction, Slots, StrictModel


class B0Config(StrictModel):
    ruleset_version: Literal["lexical-open-set-v1"]
    input_serializer: Literal["context-json-v1"]
    gate_order: list[str]
    no_intent_patterns: list[str] = Field(min_length=1)
    oos_patterns: list[str] = Field(min_length=1)
    ambiguity_patterns: list[str] = Field(min_length=1)
    future_patterns: list[str] = Field(min_length=1)
    in_scope_patterns: dict[str, list[str]] = Field(min_length=1)
    mixed_oos_choice_is_ambiguous: Literal[True] = True
    unmatched_default: Literal["no_intent"] = "no_intent"

    @model_validator(mode="after")
    def validate_rules(self) -> B0Config:
        if self.gate_order != ["actionability", "oos", "ambiguity", "in_scope"]:
            raise ValueError("B0 gate order must be actionability, OOS, ambiguity, in-scope")
        patterns: list[str] = [
            *self.no_intent_patterns,
            *self.oos_patterns,
            *self.ambiguity_patterns,
            *self.future_patterns,
        ]
        for values in self.in_scope_patterns.values():
            if not values:
                raise ValueError("each B0 in-scope intent needs at least one pattern")
            patterns.extend(values)
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid B0 regex: {pattern}") from exc
        return self


def serialize_context(context: Context) -> str:
    """Serialize exactly Context, excluding every Gold and review-only field."""

    return json.dumps(
        context.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def current_decision_text(context: Context) -> str:
    """Use the latest assistant decision when present; otherwise use state summary."""

    for turn in reversed(context.conversation):
        if turn.role == "assistant":
            return turn.content
    return context.state_summary


def _matches(patterns: Sequence[str], text: str) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def _intent_hits(patterns: Mapping[str, Sequence[str]], text: str) -> list[str]:
    return sorted(intent for intent, values in patterns.items() if _matches(values, text))


def _horizon(context: Context, config: B0Config) -> Horizon:
    text = current_decision_text(context)
    return Horizon.LATER if _matches(config.future_patterns, text) else Horizon.NOW


def predict_b0(*, case_id: str, context: Context, config: B0Config) -> Prediction:
    """Apply the registered actionability → OOS → ambiguity → in-scope rules."""

    text = serialize_context(context)
    slots = Slots(desired_experience=None, object=None, horizon=_horizon(context, config))

    if _matches(config.no_intent_patterns, text):
        return Prediction(
            case_id=case_id,
            decision=Decision.NO_INTENT,
            predicted_intent=None,
            candidate_intents=[],
            slots=slots,
            reason_short="命中冻结的无当前行动规则",
        )

    intents = _intent_hits(config.in_scope_patterns, text)
    oos = _matches(config.oos_patterns, text)
    ambiguous = _matches(config.ambiguity_patterns, text)

    if oos and not (config.mixed_oos_choice_is_ambiguous and ambiguous):
        return Prediction(
            case_id=case_id,
            decision=Decision.OOS,
            predicted_intent=None,
            candidate_intents=[],
            slots=slots,
            reason_short="命中冻结的目录外行动规则",
        )

    if ambiguous or len(intents) > 1:
        return Prediction(
            case_id=case_id,
            decision=Decision.AMBIGUOUS,
            predicted_intent=None,
            candidate_intents=intents if len(intents) >= 2 else [],
            slots=slots,
            reason_short="存在未消解选择或多个活动规则同时命中",
        )

    if len(intents) == 1:
        return Prediction(
            case_id=case_id,
            decision=Decision.IN_SCOPE,
            predicted_intent=intents[0],
            candidate_intents=[],
            slots=slots,
            reason_short="唯一命中冻结的目录内活动规则",
        )

    if oos:
        return Prediction(
            case_id=case_id,
            decision=Decision.OOS,
            predicted_intent=None,
            candidate_intents=[],
            slots=slots,
            reason_short="未消解选择中包含目录外行动",
        )

    return Prediction(
        case_id=case_id,
        decision=Decision.NO_INTENT,
        predicted_intent=None,
        candidate_intents=[],
        slots=slots,
        reason_short="没有命中冻结的行动规则",
    )
