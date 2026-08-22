"""Narrow DeepSeek Chat Completions adapter with strict local schema validation."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from intentbench.adapters.base import GenerationResult, ProviderError


class DeepSeekChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
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

    def __enter__(self) -> DeepSeekChatClient:
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
        reserved = {"model", "messages", "response_format", "max_tokens", "stream"}
        parameters = dict(generation_parameters or {})
        conflicts = reserved & set(parameters)
        if conflicts:
            raise ValueError(f"generation_parameters override reserved fields: {sorted(conflicts)}")
        try:
            Draft202012Validator.check_schema(response_schema)
        except SchemaError as exc:
            raise ValueError("response_schema is not valid JSON Schema 2020-12") from exc
        schema_text = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return one JSON object only. It must match this JSON Schema exactly: "
                        f"{schema_text}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_output_tokens,
            "stream": False,
            **parameters,
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/chat/completions", json=payload)
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
            input_tokens = int(usage_raw["prompt_tokens"])
            output_tokens = int(usage_raw["completion_tokens"])
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
        details = usage_raw.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens")
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
                structured_output_mode="json_object_plus_local_schema_validation",
            )

        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise response_error("incomplete", "provider response did not finish with stop")
            content = choice["message"]["content"]
            value = json.loads(content)
        except ProviderError:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise response_error(
                "schema_invalid", "structured response was absent or invalid"
            ) from exc
        if not isinstance(value, dict):
            raise response_error("schema_invalid", "structured response must be a JSON object")
        try:
            Draft202012Validator(response_schema).validate(value)
        except ValidationError as exc:
            raise response_error(
                "schema_invalid", "structured response did not match the requested schema"
            ) from exc
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
            structured_output_mode="json_object_plus_local_schema_validation",
        )
