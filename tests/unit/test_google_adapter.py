from __future__ import annotations

import json

import httpx
import pytest

from intentbench.adapters.base import ProviderError
from intentbench.adapters.google import GoogleGenerativeLanguageClient


def test_generation_uses_json_schema_and_preserves_usage_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-key"
        payload = json.loads(request.content)
        config = payload["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"]["type"] == "object"
        assert "temperature" not in config
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "decision": "in_scope",
                                            "predicted_intent": "take_a_walk",
                                            "reason_short": "想散步",
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 3,
                    "totalTokenCount": 18,
                },
                "modelVersion": "gemini-3.6-flash",
            },
        )

    with GoogleGenerativeLanguageClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    ) as client:
        result = client.generate_json(
            model="gemini-3.6-flash",
            prompt="synthetic",
            response_schema={"type": "object"},
            max_output_tokens=512,
        )
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (10, 8, 18)
    assert result.reported_model == "gemini-3.6-flash"
    assert result.structured_output_mode == "provider_json_schema"


def test_provider_error_does_not_persist_response_body() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            400,
            json={"error": {"status": "INVALID_ARGUMENT", "message": "sensitive body"}},
        )
    )
    with (
        GoogleGenerativeLanguageClient(api_key="test-key", transport=transport) as client,
        pytest.raises(ProviderError) as caught,
    ):
        client.generate_json(
            model="gemini-3.6-flash",
            prompt="synthetic",
            response_schema={"type": "object"},
            max_output_tokens=512,
        )
    assert "sensitive body" not in str(caught.value)
