"""Content-addressed normalized generation cache for one-stage and verifier reuse."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from intentbench.adapters.base import GenerationResult, ProviderError, StructuredGenerationClient
from intentbench.schemas import StrictModel

CALL_CONTRACT_FIELDS = (
    "schema_version",
    "normalized_context_sha256",
    "prompt_instance_sha256",
    "taxonomy_sha256",
    "prompt_template_sha256",
    "few_shot_payload_sha256",
    "response_schema_sha256",
    "model",
    "adapter",
    "adapter_revision",
    "structured_output_mode",
    "max_output_tokens",
    "generation_parameters_sha256",
)


class GenerationCacheError(ValueError):
    """A generation cache record disagrees with the current call contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


class GenerationCallContract(StrictModel):
    """Immutable identity derived from every material one-stage request field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    normalized_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_instance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    few_shot_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    adapter_revision: str = Field(pattern=r"^v[0-9]+$")
    structured_output_mode: str = Field(min_length=1)
    max_output_tokens: int = Field(ge=1)
    generation_parameters_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class GenerationRequest(StrictModel):
    """Raw request plus provenance; hashes are always computed internally."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized_context: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    taxonomy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    few_shot_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema: dict[str, Any]
    model: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    adapter_revision: str = Field(pattern=r"^v[0-9]+$")
    structured_output_mode: str = Field(min_length=1)
    max_output_tokens: int = Field(ge=1)
    generation_parameters: dict[str, Any]

    def contract(self) -> GenerationCallContract:
        return GenerationCallContract(
            normalized_context_sha256=sha256_text(self.normalized_context),
            prompt_instance_sha256=sha256_text(self.prompt),
            taxonomy_sha256=self.taxonomy_sha256,
            prompt_template_sha256=self.prompt_template_sha256,
            few_shot_payload_sha256=self.few_shot_payload_sha256,
            response_schema_sha256=sha256_json(self.response_schema),
            model=self.model,
            adapter=self.adapter,
            adapter_revision=self.adapter_revision,
            structured_output_mode=self.structured_output_mode,
            max_output_tokens=self.max_output_tokens,
            generation_parameters_sha256=sha256_json(self.generation_parameters),
        )


class GenerationCacheRecord(StrictModel):
    schema_version: Literal[2] = 2
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_contract: GenerationCallContract
    origin: Literal["provider_call", "validated_contract_migration"]
    source_cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Literal["success", "provider_failure"]
    value: dict[str, Any] | None
    reported_model: str | None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    usage_metadata: dict[str, int] | None
    structured_output_mode: str | None
    error_type: str | None
    http_status: int | None

    @model_validator(mode="after")
    def validate_record(self) -> GenerationCacheRecord:
        if self.cache_key != self.call_contract.sha256:
            raise ValueError("generation cache key differs from the embedded call contract")
        if self.call_contract_sha256 != self.call_contract.sha256:
            raise ValueError("generation call-contract hash is inconsistent")
        if self.origin == "provider_call" and self.source_cache_key is not None:
            raise ValueError("a direct Provider call cannot carry a migration source")
        if self.origin == "validated_contract_migration" and self.source_cache_key is None:
            raise ValueError("a migrated record requires its source cache key")

        accounting_fields = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.raw_response_sha256,
            self.usage_metadata,
        )
        if self.status == "success":
            if self.value is None or any(value is None for value in accounting_fields):
                raise ValueError("successful generation cache record lacks response fields")
            if self.structured_output_mode is None:
                raise ValueError("successful generation cache record lacks output mode")
            if self.error_type is not None or self.http_status is not None:
                raise ValueError("successful generation cache record cannot carry an error")
        else:
            if self.value is not None:
                raise ValueError("failed generation cache record cannot carry a value")
            if not self.error_type:
                raise ValueError("failed generation cache record requires error_type")
            if any(value is not None for value in accounting_fields) and any(
                value is None for value in accounting_fields
            ):
                raise ValueError("failed generation accounting must be complete or absent")
        if self.input_tokens is not None:
            assert self.output_tokens is not None
            assert self.total_tokens is not None
            if self.total_tokens != self.input_tokens + self.output_tokens:
                raise ValueError("generation cache token totals are inconsistent")
        return self


@dataclass(frozen=True)
class CachedGeneration:
    result: GenerationResult | None
    call_contract: GenerationCallContract
    origin: Literal["provider_call", "validated_contract_migration"]
    source_cache_key: str | None
    reported_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw_response_sha256: str | None
    usage_metadata: dict[str, int] | None
    structured_output_mode: str | None
    error_type: str | None
    http_status: int | None
    latency_ms: float
    cache_key: str
    call_contract_sha256: str
    cache_hit: bool


class GenerationCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, cache_key: str) -> Path:
        return self.directory / f"{cache_key}.json"

    def require_existing(self, *, request: GenerationRequest) -> CachedGeneration:
        """Load an exact cached call or fail; B2b call-1 must use this path."""

        contract = request.contract()
        path = self._path(contract.sha256)
        if not path.is_file():
            raise GenerationCacheError(
                "required one-stage-compatible generation cache record is absent"
            )
        record = self._read_path(path)
        if record.call_contract != contract:
            raise GenerationCacheError("generation cache identity mismatch")
        return self._to_result(record, cache_hit=True)

    def get_or_create(
        self,
        *,
        client: StructuredGenerationClient,
        request: GenerationRequest,
    ) -> CachedGeneration:
        contract = request.contract()
        if self._path(contract.sha256).is_file():
            return self.require_existing(request=request)

        started = time.perf_counter()
        try:
            result = client.generate_json(
                model=request.model,
                prompt=request.prompt,
                response_schema=request.response_schema,
                max_output_tokens=request.max_output_tokens,
                generation_parameters=request.generation_parameters,
            )
            if result.requested_model != request.model:
                raise GenerationCacheError("adapter returned a different requested model")
            record = self._success_record(
                contract=contract,
                result=result,
                origin="provider_call",
                source_cache_key=None,
            )
        except ProviderError as exc:
            record = GenerationCacheRecord(
                cache_key=contract.sha256,
                call_contract_sha256=contract.sha256,
                call_contract=contract,
                origin="provider_call",
                source_cache_key=None,
                status="provider_failure",
                value=None,
                reported_model=exc.reported_model,
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
                total_tokens=exc.total_tokens,
                latency_ms=exc.latency_ms
                if exc.latency_ms is not None
                else (time.perf_counter() - started) * 1000,
                raw_response_sha256=exc.raw_response_sha256,
                usage_metadata=exc.usage_metadata,
                structured_output_mode=exc.structured_output_mode,
                error_type=exc.error_type,
                http_status=exc.status_code,
            )
        self._write(record)
        return self._to_result(record, cache_hit=False)

    def seed_validated_success(
        self,
        *,
        request: GenerationRequest,
        result: GenerationResult,
        source_cache_key: str,
    ) -> CachedGeneration:
        """Migrate a validated normalized success without repeating a paid call."""

        contract = request.contract()
        if result.requested_model != request.model:
            raise GenerationCacheError("migration source requested a different model")
        if self._path(contract.sha256).is_file():
            return self.require_existing(request=request)
        record = self._success_record(
            contract=contract,
            result=result,
            origin="validated_contract_migration",
            source_cache_key=source_cache_key,
        )
        self._write(record)
        return self._to_result(record, cache_hit=True)

    def records(self, cache_keys: list[str]) -> list[GenerationCacheRecord]:
        if len(cache_keys) != len(set(cache_keys)):
            raise GenerationCacheError("cache bundle keys must be unique")
        return [self._read_path(self._path(cache_key)) for cache_key in cache_keys]

    def import_records(self, records: list[GenerationCacheRecord]) -> int:
        """Restore a safe normalized reference bundle into a local cache."""

        imported = 0
        for record in records:
            path = self._path(record.cache_key)
            if path.is_file():
                if self._read_path(path) != record:
                    raise GenerationCacheError(
                        "existing cache record differs from reference bundle"
                    )
                continue
            self._write(record)
            imported += 1
        return imported

    @staticmethod
    def _success_record(
        *,
        contract: GenerationCallContract,
        result: GenerationResult,
        origin: Literal["provider_call", "validated_contract_migration"],
        source_cache_key: str | None,
    ) -> GenerationCacheRecord:
        return GenerationCacheRecord(
            cache_key=contract.sha256,
            call_contract_sha256=contract.sha256,
            call_contract=contract,
            origin=origin,
            source_cache_key=source_cache_key,
            status="success",
            value=result.value,
            reported_model=result.reported_model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            latency_ms=result.latency_ms,
            raw_response_sha256=result.raw_response_sha256,
            usage_metadata=result.usage_metadata,
            structured_output_mode=result.structured_output_mode,
            error_type=None,
            http_status=None,
        )

    def _read_path(self, path: Path) -> GenerationCacheRecord:
        if not path.is_file():
            raise GenerationCacheError(f"generation cache record is absent: {path.name}")
        try:
            return GenerationCacheRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GenerationCacheError(f"invalid generation cache record: {path}") from exc

    def _write(self, record: GenerationCacheRecord) -> None:
        payload = (canonical_json(record.model_dump(mode="json")) + "\n").encode()
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{record.cache_key}-",
            suffix=".tmp",
            dir=self.directory,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, self._path(record.cache_key))
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _to_result(record: GenerationCacheRecord, *, cache_hit: bool) -> CachedGeneration:
        result: GenerationResult | None = None
        if record.status == "success":
            assert record.value is not None
            assert record.input_tokens is not None
            assert record.output_tokens is not None
            assert record.total_tokens is not None
            assert record.raw_response_sha256 is not None
            assert record.usage_metadata is not None
            assert record.structured_output_mode is not None
            result = GenerationResult(
                value=record.value,
                requested_model=record.call_contract.model,
                reported_model=record.reported_model,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                total_tokens=record.total_tokens,
                latency_ms=record.latency_ms,
                raw_response_sha256=record.raw_response_sha256,
                usage_metadata=record.usage_metadata,
                structured_output_mode=record.structured_output_mode,
            )
        return CachedGeneration(
            result=result,
            call_contract=record.call_contract,
            origin=record.origin,
            source_cache_key=record.source_cache_key,
            reported_model=record.reported_model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            raw_response_sha256=record.raw_response_sha256,
            usage_metadata=record.usage_metadata,
            structured_output_mode=record.structured_output_mode,
            error_type=record.error_type,
            http_status=record.http_status,
            latency_ms=record.latency_ms,
            cache_key=record.cache_key,
            call_contract_sha256=record.call_contract_sha256,
            cache_hit=cache_hit,
        )
