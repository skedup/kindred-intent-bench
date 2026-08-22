"""Narrow OpenAI Responses API adapter using provider-enforced JSON Schema."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx

from intentbench.adapters.base import GenerationResult, ProviderError


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIResponsesClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
        generation_parameters: dict[str, Any] | None = None,
    ) -> GenerationResult:
        reserved = {"model", "input", "text", "max_output_tokens", "store", "stream"}
        parameters = dict(generation_parameters or {})
        conflicts = reserved & set(parameters)
        if conflicts:
            raise ValueError(f"generation_parameters override reserved fields: {sorted(conflicts)}")
        payload: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "intent_readiness",
                    "strict": True,
                    "schema": response_schema,
                }
            },
            "max_output_tokens": max_output_tokens,
            "store": False,
            "stream": False,
            **parameters,
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/responses", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", "provider request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("transport_error", "provider transport failed") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if response.is_error:
            error_type = "rate_limit" if response.status_code == 429 else "provider_http_error"
            raise ProviderError(
                error_type,
                f"provider returned HTTP {response.status_code}",
                response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError("invalid_json_response", "provider returned non-JSON") from exc
        usage_raw = data.get("usage")
        if not isinstance(usage_raw, dict):
            raise ProviderError("usage_missing", "generation response lacks usage")
        try:
            input_tokens = int(usage_raw["input_tokens"])
            output_tokens = int(usage_raw["output_tokens"])
            total_tokens = int(usage_raw["total_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("usage_missing", "required token usage fields are absent") from exc
        if min(input_tokens, output_tokens, total_tokens) < 0:
            raise ProviderError("usage_invalid", "provider token totals must be non-negative")
        if total_tokens != input_tokens + output_tokens:
            raise ProviderError("usage_invalid", "provider token totals are inconsistent")
        numeric_usage = {
            key: int(item)
            for key, item in usage_raw.items()
            if isinstance(item, int) and not isinstance(item, bool)
        }
        output_details = usage_raw.get("output_tokens_details")
        if isinstance(output_details, dict):
            reasoning_tokens = output_details.get("reasoning_tokens")
            if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
                numeric_usage["reasoning_tokens"] = reasoning_tokens
        reported_model_raw = data.get("model")
        reported_model = reported_model_raw if isinstance(reported_model_raw, str) else None
        raw_response_sha256 = hashlib.sha256(response.content).hexdigest()

        def response_error(error_type: str, message: str) -> ProviderError:
            return ProviderError(
                error_type,
                message,
                reported_model=reported_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                raw_response_sha256=raw_response_sha256,
                usage_metadata=numeric_usage,
                structured_output_mode="provider_json_schema",
            )

        if data.get("status") != "completed":
            raise response_error("incomplete", "provider response status is not completed")
        try:
            output_texts = [
                content["text"]
                for item in data["output"]
                if item.get("type") == "message"
                for content in item.get("content", [])
                if content.get("type") == "output_text"
            ]
            if len(output_texts) != 1:
                raise response_error("schema_invalid", "response must contain one output_text")
            value = json.loads(output_texts[0])
        except ProviderError:
            raise
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise response_error(
                "schema_invalid", "structured response was absent or invalid"
            ) from exc
        if not isinstance(value, dict):
            raise response_error("schema_invalid", "structured response must be a JSON object")
        return GenerationResult(
            value=value,
            requested_model=model,
            reported_model=reported_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            raw_response_sha256=raw_response_sha256,
            usage_metadata=numeric_usage,
            structured_output_mode="provider_json_schema",
        )
