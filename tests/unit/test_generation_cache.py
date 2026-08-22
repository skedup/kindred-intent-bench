from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from intentbench.adapters.base import GenerationResult, ProviderError
from intentbench.generation_cache import (
    GenerationCache,
    GenerationCacheError,
    GenerationRequest,
)


class CountingClient:
    def __init__(self, *, fail: bool = False, accounted_failure: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.accounted_failure = accounted_failure

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
        generation_parameters: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self.calls += 1
        if self.accounted_failure:
            raise ProviderError(
                "schema_invalid",
                "sanitized",
                reported_model=model,
                input_tokens=10,
                output_tokens=500,
                total_tokens=510,
                latency_ms=4.0,
                raw_response_sha256="9" * 64,
                usage_metadata={"promptTokenCount": 10, "totalTokenCount": 510},
                structured_output_mode="provider_json_schema",
            )
        if self.fail:
            raise ProviderError("rate_limit", "sanitized", 429)
        return GenerationResult(
            value={"answer": prompt},
            requested_model=model,
            reported_model=model,
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            latency_ms=3.0,
            raw_response_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            usage_metadata={"promptTokenCount": 10, "totalTokenCount": 12},
            structured_output_mode="provider_json_schema",
        )


def _request(**updates: object) -> GenerationRequest:
    request = GenerationRequest(
        normalized_context='{"state_summary":"想散步"}',
        prompt="rendered prompt",
        taxonomy_sha256="2" * 64,
        prompt_template_sha256="3" * 64,
        few_shot_payload_sha256="4" * 64,
        response_schema={"type": "object"},
        model="gemini-3.6-flash",
        adapter="google_generate_content_v1beta",
        adapter_revision="v2",
        structured_output_mode="provider_json_schema",
        max_output_tokens=512,
        generation_parameters={"thinkingConfig": {"thinkingLevel": "minimal"}},
    )
    return request.model_copy(update=updates)


def test_generation_cache_first_call_then_offline_hit(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path / "cache")
    client = CountingClient()
    request = _request()
    first = cache.get_or_create(client=client, request=request)
    second = cache.get_or_create(client=client, request=request)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.result == second.result
    assert first.cache_key == second.cache_key
    assert first.origin == "provider_call"
    assert client.calls == 1


def test_required_call1_cache_never_falls_back_to_provider(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path / "cache")
    request = _request()
    with pytest.raises(GenerationCacheError, match="is absent"):
        cache.require_existing(request=request)

    client = CountingClient()
    created = cache.get_or_create(client=client, request=request)
    required = cache.require_existing(request=request)
    assert required.cache_hit is True
    assert required.cache_key == created.cache_key
    assert client.calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalized_context", '{"state_summary":"different"}'),
        ("prompt", "different rendered prompt"),
        ("taxonomy_sha256", "a" * 64),
        ("prompt_template_sha256", "b" * 64),
        ("few_shot_payload_sha256", "c" * 64),
        ("response_schema", {"type": "array"}),
        ("model", "different-model"),
        ("adapter", "different-adapter"),
        ("adapter_revision", "v3"),
        ("structured_output_mode", "different-mode"),
        ("max_output_tokens", 1024),
        ("generation_parameters", {"thinkingConfig": {"thinkingLevel": "low"}}),
    ],
)
def test_each_material_raw_request_field_changes_cache_key(field: str, value: object) -> None:
    original = _request().contract().sha256
    assert _request(**{field: value}).contract().sha256 != original


def test_generation_cache_persists_sanitized_provider_failure(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path / "cache")
    client = CountingClient(fail=True)
    request = _request()
    first = cache.get_or_create(client=client, request=request)
    second = cache.get_or_create(client=client, request=request)
    assert first.result is None
    assert first.error_type == "rate_limit"
    assert first.http_status == 429
    assert second.cache_hit is True
    assert second.error_type == "rate_limit"
    assert client.calls == 1


def test_generation_cache_preserves_failed_call_accounting(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path / "cache")
    client = CountingClient(accounted_failure=True)
    request = _request()
    first = cache.get_or_create(client=client, request=request)
    second = cache.get_or_create(client=client, request=request)
    assert first.result is None
    assert first.error_type == "schema_invalid"
    assert (first.input_tokens, first.output_tokens, first.total_tokens) == (10, 500, 510)
    assert first.raw_response_sha256 == "9" * 64
    assert second.cache_hit is True
    assert second.total_tokens == 510
    assert client.calls == 1


def test_validated_migration_and_reference_bundle_restore_are_offline(tmp_path: Path) -> None:
    request = _request()
    result = CountingClient().generate_json(
        model=request.model,
        prompt=request.prompt,
        response_schema=request.response_schema,
        max_output_tokens=request.max_output_tokens,
        generation_parameters=request.generation_parameters,
    )
    source = GenerationCache(tmp_path / "source")
    migrated = source.seed_validated_success(
        request=request,
        result=result,
        source_cache_key="d" * 64,
    )
    assert migrated.cache_hit is True
    assert migrated.origin == "validated_contract_migration"
    assert migrated.source_cache_key == "d" * 64

    records = source.records([migrated.cache_key])
    restored = GenerationCache(tmp_path / "restored")
    assert restored.import_records(records) == 1
    assert restored.import_records(records) == 0
    assert restored.require_existing(request=request).result == result
