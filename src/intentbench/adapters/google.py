"""Minimal Gemini Developer API adapter for readiness and later runners."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import httpx

from intentbench.adapters.base import EmbeddingResult, GenerationResult, ProviderError


class GoogleGenerativeLanguageClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GoogleGenerativeLanguageClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], bytes, float]:
        started = time.perf_counter()
        try:
            response = self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", "provider request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("transport_error", "provider transport failed") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if response.is_error:
            status = response.status_code
            error_type = "rate_limit" if status == 429 else "provider_http_error"
            try:
                provider_status = response.json().get("error", {}).get("status", "unknown")
            except (ValueError, AttributeError):
                provider_status = "unparseable"
            raise ProviderError(
                error_type, f"provider returned {status} ({provider_status})", status
            )
        try:
            return response.json(), response.content, latency_ms
        except ValueError as exc:
            raise ProviderError("invalid_json_response", "provider returned non-JSON") from exc

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
        generation_parameters: dict[str, Any] | None = None,
    ) -> GenerationResult:
        generation_config = dict(generation_parameters or {})
        generation_config.update(
            {
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            }
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        data, raw, latency_ms = self._post(f"/v1beta/models/{model}:generateContent", payload)
        usage_raw = data.get("usageMetadata")
        if not isinstance(usage_raw, dict):
            raise ProviderError("usage_missing", "generation response lacks usageMetadata")
        try:
            input_tokens = int(usage_raw["promptTokenCount"])
            total_tokens = int(usage_raw["totalTokenCount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("usage_missing", "required token usage fields are absent") from exc
        output_tokens = total_tokens - input_tokens
        if input_tokens < 0 or output_tokens < 0:
            raise ProviderError("usage_invalid", "provider token totals are inconsistent")
        numeric_usage = {
            key: int(value)
            for key, value in usage_raw.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        raw_response_sha256 = hashlib.sha256(raw).hexdigest()
        reported_model_raw = data.get("modelVersion")
        reported_model = reported_model_raw if isinstance(reported_model_raw, str) else None
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            value = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "schema_invalid",
                "structured response was absent or invalid",
                reported_model=reported_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                raw_response_sha256=raw_response_sha256,
                usage_metadata=numeric_usage,
                structured_output_mode="provider_json_schema",
            ) from exc
        if not isinstance(value, dict):
            raise ProviderError(
                "schema_invalid",
                "structured response must be a JSON object",
                reported_model=reported_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                raw_response_sha256=raw_response_sha256,
                usage_metadata=numeric_usage,
                structured_output_mode="provider_json_schema",
            )
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

    def embed(
        self,
        *,
        model: str,
        text: str,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
    ) -> EmbeddingResult:
        payload: dict[str, Any] = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
        }
        if task_type is not None:
            payload["taskType"] = task_type
        if output_dimensionality is not None:
            if output_dimensionality <= 0:
                raise ValueError("output_dimensionality must be positive")
            payload["outputDimensionality"] = output_dimensionality
        data, raw, latency_ms = self._post(f"/v1beta/models/{model}:embedContent", payload)
        try:
            values = tuple(float(value) for value in data["embedding"]["values"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "embedding_invalid", "embedding vector is absent or invalid"
            ) from exc
        if not values or not all(math.isfinite(value) for value in values):
            raise ProviderError(
                "embedding_invalid", "embedding vector must be finite and non-empty"
            )
        return EmbeddingResult(
            vector=values,
            requested_model=model,
            latency_ms=latency_ms,
            raw_response_sha256=hashlib.sha256(raw).hexdigest(),
        )
