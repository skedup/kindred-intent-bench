from __future__ import annotations

import json

import httpx
import pytest

from intentbench.adapters.base import ProviderError
from intentbench.adapters.openai import OpenAIResponsesClient


def test_responses_uses_strict_json_schema_and_preserves_usage_contract() -> None:
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.6-luna"
        assert payload["text"]["format"] == {
            "type": "json_schema",
            "name": "intent_readiness",
            "strict": True,
            "schema": response_schema,
        }
        assert payload["reasoning"] == {"effort": "high"}
        assert payload["store"] is False
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "model": "gpt-5.6-luna",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "decision": "in_scope",
                                        "predicted_intent": "take_a_walk",
                                        "reason_short": "想散步",
                                    }
                                ),
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 15,
                    "output_tokens": 10,
                    "total_tokens": 25,
                    "output_tokens_details": {"reasoning_tokens": 4},
                },
            },
        )

    with OpenAIResponsesClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    ) as client:
        result = client.generate_json(
            model="gpt-5.6-luna",
            prompt="synthetic",
            response_schema=response_schema,
            max_output_tokens=512,
            generation_parameters={"reasoning": {"effort": "high"}},
        )

    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (15, 10, 25)
    assert result.usage_metadata["reasoning_tokens"] == 4
    assert result.reported_model == "gpt-5.6-luna"
    assert result.structured_output_mode == "provider_json_schema"


def test_provider_error_does_not_persist_response_body() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(400, json={"error": {"message": "sensitive body"}})
    )
    with (
        OpenAIResponsesClient(api_key="test-key", transport=transport) as client,
        pytest.raises(ProviderError) as caught,
    ):
        client.generate_json(
            model="gpt-5.6-luna",
            prompt="synthetic",
            response_schema={"type": "object"},
            max_output_tokens=512,
        )
    assert "sensitive body" not in str(caught.value)
