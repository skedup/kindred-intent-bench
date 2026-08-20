from __future__ import annotations

import json

import httpx
import pytest

from intentbench.adapters.base import ProviderError
from intentbench.adapters.deepseek import DeepSeekChatClient


def test_chat_completions_uses_json_mode_and_preserves_usage_contract() -> None:
    response_schema = {
        "type": "object",
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high"
        assert (
            json.dumps(response_schema, separators=(",", ":")) in payload["messages"][0]["content"]
        )
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "in_scope",
                                    "predicted_intent": "take_a_walk",
                                    "reason_short": "想散步",
                                }
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            },
        )

    with DeepSeekChatClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        result = client.generate_json(
            model="deepseek-v4-flash",
            prompt="synthetic",
            response_schema=response_schema,
            max_output_tokens=512,
            generation_parameters={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        )

    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (12, 8, 20)
    assert result.usage_metadata["reasoning_tokens"] == 3
    assert result.reported_model == "deepseek-v4-flash"
    assert result.structured_output_mode == "json_object_plus_local_schema_validation"


def test_provider_error_does_not_persist_response_body() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(400, json={"error": {"message": "sensitive body"}})
    )
    with (
        DeepSeekChatClient(api_key="test-key", transport=transport) as client,
        pytest.raises(ProviderError) as caught,
    ):
        client.generate_json(
            model="deepseek-v4-flash",
            prompt="synthetic",
            response_schema={"type": "object"},
            max_output_tokens=512,
        )
    assert "sensitive body" not in str(caught.value)


def test_json_object_that_violates_schema_is_rejected_locally() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"decision": 42})},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )
    )
    with (
        DeepSeekChatClient(api_key="test-key", transport=transport) as client,
        pytest.raises(ProviderError, match="requested schema") as caught,
    ):
        client.generate_json(
            model="deepseek-v4-flash",
            prompt="synthetic",
            response_schema={
                "type": "object",
                "properties": {"decision": {"type": "string"}},
                "required": ["decision"],
            },
            max_output_tokens=512,
        )
    assert caught.value.error_type == "schema_invalid"
