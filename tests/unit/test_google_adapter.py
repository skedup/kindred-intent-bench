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


def test_schema_failure_preserves_billable_usage_without_raw_body() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "candidates": [{"finishReason": "MAX_TOKENS"}],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "thoughtsTokenCount": 500,
                    "totalTokenCount": 600,
                },
                "modelVersion": "gemini-3.6-flash",
            },
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
    error = caught.value
    assert error.error_type == "schema_invalid"
    assert (error.input_tokens, error.output_tokens, error.total_tokens) == (100, 500, 600)
    assert error.reported_model == "gemini-3.6-flash"
    assert error.raw_response_sha256 is not None
    assert "MAX_TOKENS" not in str(error)


def test_embedding_sends_registered_similarity_task_and_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload == {
            "model": "models/gemini-embedding-001",
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": 3,
            "content": {"parts": [{"text": "想去散步"}]},
        }
        return httpx.Response(200, json={"embedding": {"values": [1.0, 2.0, 3.0]}})

    with GoogleGenerativeLanguageClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    ) as client:
        result = client.embed(
            model="gemini-embedding-001",
            text="想去散步",
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=3,
        )
    assert result.vector == (1.0, 2.0, 3.0)
