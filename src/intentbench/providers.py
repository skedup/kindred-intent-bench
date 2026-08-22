"""Synthetic-only multi-environment Provider readiness checks."""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from intentbench.adapters.base import ProviderError, StructuredGenerationClient
from intentbench.adapters.deepseek import DeepSeekChatClient
from intentbench.adapters.google import GoogleGenerativeLanguageClient
from intentbench.adapters.openai import OpenAIResponsesClient
from intentbench.freeze import sha256_file
from intentbench.schemas import Decision, StrictModel

ProviderName = Literal["google_generative_language", "deepseek", "openai"]
ExperimentArm = Literal["B2a", "B2b", "B3"]
ReadinessRole = Literal[
    "primary_decision", "weak_decision", "cross_provider_reference", "embedding"
]
READINESS_ROLES: tuple[ReadinessRole, ...] = (
    "primary_decision",
    "weak_decision",
    "cross_provider_reference",
    "embedding",
)
READINESS_SECRET_POLICY = "environment_variable_name_only; no secret or raw response persisted"


class LLMModelSpec(StrictModel):
    provider: ProviderName
    model: str
    adapter: str
    adapter_revision: str = Field(default="v2", pattern=r"^v[0-9]+$")
    base_url: str
    api_key_env: str
    timeout_seconds: float = Field(gt=0)
    request_max_output_tokens: int = Field(gt=0)
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    unsupported_generation_parameters: list[str] = Field(default_factory=list)
    expected_usage_fields: list[str]
    structured_output_mode: Literal[
        "provider_json_schema", "json_object_plus_local_schema_validation"
    ]
    required_arms: list[ExperimentArm] = Field(min_length=1)
    verdict_authority: bool
    capability_context_window: int | None = None
    capability_max_output_tokens: int | None = None

    @model_validator(mode="after")
    def validate_provider_contract(self) -> LLMModelSpec:
        expected = {
            "google_generative_language": (
                "google_generate_content_v1beta",
                "provider_json_schema",
            ),
            "deepseek": (
                "deepseek_chat_completions_json_v1",
                "json_object_plus_local_schema_validation",
            ),
            "openai": ("openai_responses_json_schema_v1", "provider_json_schema"),
        }
        adapter, output_mode = expected[self.provider]
        if self.adapter != adapter or self.structured_output_mode != output_mode:
            raise ValueError("provider, adapter, and structured_output_mode disagree")
        if len(self.required_arms) != len(set(self.required_arms)):
            raise ValueError("required_arms must be unique")
        if len(self.expected_usage_fields) != len(set(self.expected_usage_fields)):
            raise ValueError("expected_usage_fields must be unique")
        return self


class EmbeddingModelSpec(StrictModel):
    provider: Literal["google_generative_language"]
    model: str
    adapter: str
    base_url: str
    api_key_env: str
    timeout_seconds: float = Field(gt=0)
    expected_dimension: int = Field(gt=0)
    task_type: Literal["SEMANTIC_SIMILARITY"] | None = None
    output_dimensionality: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_dimension(self) -> EmbeddingModelSpec:
        if (
            self.output_dimensionality is not None
            and self.output_dimensionality != self.expected_dimension
        ):
            raise ValueError("embedding output and expected dimensions must agree")
        return self


class ReadinessModels(StrictModel):
    authority: dict[str, Any]
    runtime_candidate_pool: list[dict[str, Any]]
    selection_rationale: dict[str, str]
    comparison_matrix: dict[str, Any]
    primary_decision: LLMModelSpec
    weak_decision: LLMModelSpec
    cross_provider_reference: LLMModelSpec
    embedding: EmbeddingModelSpec

    @model_validator(mode="after")
    def validate_preregistered_matrix(self) -> ReadinessModels:
        role_contracts = (
            (
                self.primary_decision,
                "google_generative_language",
                ["B2a", "B2b", "B3"],
                True,
            ),
            (self.weak_decision, "deepseek", ["B2b", "B3"], False),
            (self.cross_provider_reference, "openai", ["B2b", "B3"], False),
        )
        for spec, provider, arms, verdict_authority in role_contracts:
            if (
                spec.provider != provider
                or spec.required_arms != arms
                or spec.verdict_authority is not verdict_authority
            ):
                raise ValueError("LLM role violates the preregistered provider/arm authority")
        expected_matrix = {
            "primary_confirmatory": {
                "roles": ["primary_decision"],
                "arms": ["B2a", "B2b", "B3"],
                "comparison": "B3_minus_B2b",
                "global_verdict_authority": True,
            },
            "cross_provider_replication": {
                "roles": ["weak_decision", "cross_provider_reference"],
                "arms": ["B2b", "B3"],
                "comparison": "within_model_B3_minus_B2b",
                "pooled_score_or_verdict": False,
                "primary_model_switch_after_test": "forbidden",
            },
        }
        if self.comparison_matrix != expected_matrix:
            raise ValueError("comparison_matrix violates the preregistered seven-run contract")
        if self.embedding.adapter != "google_embed_content_v1beta":
            raise ValueError("embedding adapter violates the preregistered contract")
        return self


class ReadinessFixture(StrictModel):
    schema_version: Literal[1]
    fixture_id: str
    prompt: str
    expected: dict[str, str]
    embedding_text: str


class SmokeOutput(StrictModel):
    decision: Decision
    predicted_intent: str | None
    reason_short: str = Field(min_length=1, max_length=200)


SMOKE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": [decision.value for decision in Decision]},
        "predicted_intent": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "create_picture",
                        "dine_out",
                        "eat_at_home",
                        "play_xiaohongshu",
                        "reach_out_to_user",
                        "rest",
                        "take_a_walk",
                        "visit_cultural_place",
                    ],
                },
                {"type": "null"},
            ]
        },
        "reason_short": {"type": "string", "minLength": 1, "maxLength": 200},
    },
    "required": ["decision", "predicted_intent", "reason_short"],
}


def load_readiness_inputs(
    experiment_path: Path, fixture_path: Path
) -> tuple[ReadinessModels, ReadinessFixture]:
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    models = ReadinessModels.model_validate(experiment["models"])
    fixture = ReadinessFixture.model_validate_json(fixture_path.read_text(encoding="utf-8"))
    return models, fixture


def _generation_client(spec: LLMModelSpec, api_key: str) -> StructuredGenerationClient:
    common: dict[str, Any] = {
        "api_key": api_key,
        "base_url": spec.base_url,
        "timeout_seconds": spec.timeout_seconds,
    }
    if spec.provider == "google_generative_language":
        return GoogleGenerativeLanguageClient(**common)
    if spec.provider == "deepseek":
        return DeepSeekChatClient(**common)
    return OpenAIResponsesClient(**common)


def _llm_smoke(role: str, spec: LLMModelSpec, fixture: ReadinessFixture) -> dict[str, Any]:
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        return {
            "role": role,
            "provider": spec.provider,
            "adapter": spec.adapter,
            "requested_model": spec.model,
            "status": "failed",
            "error_type": "credential_missing",
        }
    client = _generation_client(spec, api_key)
    try:
        result = client.generate_json(
            model=spec.model,
            prompt=fixture.prompt,
            response_schema=SMOKE_RESPONSE_SCHEMA,
            max_output_tokens=spec.request_max_output_tokens,
            generation_parameters=spec.generation_parameters,
        )
        output = SmokeOutput.model_validate(result.value)
        expected_match = output.decision.value == fixture.expected.get(
            "decision"
        ) and output.predicted_intent == fixture.expected.get("predicted_intent")
        reported_identity = (result.reported_model or "").removeprefix("models/")
        identity_match = reported_identity == spec.model
        output_contract_match = result.structured_output_mode == spec.structured_output_mode
        usage_contract_match = set(spec.expected_usage_fields).issubset(result.usage_metadata)
        status = (
            "passed"
            if expected_match and identity_match and output_contract_match and usage_contract_match
            else "failed"
        )
        return {
            "role": role,
            "provider": spec.provider,
            "adapter": spec.adapter,
            "requested_model": result.requested_model,
            "reported_model": result.reported_model,
            "required_arms": spec.required_arms,
            "verdict_authority": spec.verdict_authority,
            "status": status,
            "connectivity": True,
            "structured_output_valid": True,
            "structured_output_mode": result.structured_output_mode,
            "structured_output_contract_match": output_contract_match,
            "fixture_expectation_match": expected_match,
            "identity_match": identity_match,
            "timeout_seconds": spec.timeout_seconds,
            "latency_ms": round(result.latency_ms, 3),
            "usage_available": True,
            "usage_contract_match": usage_contract_match,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "provider_usage_fields": sorted(result.usage_metadata),
            "raw_response_sha256": result.raw_response_sha256,
        }
    except ProviderError as exc:
        return {
            "role": role,
            "provider": spec.provider,
            "adapter": spec.adapter,
            "requested_model": spec.model,
            "status": "failed",
            "error_type": exc.error_type,
            "http_status": exc.status_code,
        }
    except (ValueError, KeyError) as exc:
        return {
            "role": role,
            "provider": spec.provider,
            "adapter": spec.adapter,
            "requested_model": spec.model,
            "status": "failed",
            "error_type": type(exc).__name__,
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _embedding_smoke(spec: EmbeddingModelSpec, fixture: ReadinessFixture) -> dict[str, Any]:
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        return {
            "role": "embedding",
            "provider": spec.provider,
            "adapter": spec.adapter,
            "requested_model": spec.model,
            "status": "failed",
            "error_type": "credential_missing",
        }
    try:
        with GoogleGenerativeLanguageClient(
            api_key=api_key,
            base_url=spec.base_url,
            timeout_seconds=spec.timeout_seconds,
        ) as client:
            result = client.embed(
                model=spec.model,
                text=fixture.embedding_text,
                task_type=spec.task_type,
                output_dimensionality=spec.output_dimensionality,
            )
        dimension_match = len(result.vector) == spec.expected_dimension
        return {
            "role": "embedding",
            "provider": spec.provider,
            "adapter": spec.adapter,
            "requested_model": result.requested_model,
            "reported_model": None,
            "identity_evidence": "model_specific_request_endpoint",
            "identity_reported_by_provider": False,
            "status": "passed" if dimension_match else "failed",
            "connectivity": True,
            "finite_vector": True,
            "dimension": len(result.vector),
            "expected_dimension": spec.expected_dimension,
            "dimension_match": dimension_match,
            "timeout_seconds": spec.timeout_seconds,
            "latency_ms": round(result.latency_ms, 3),
            "raw_response_sha256": result.raw_response_sha256,
        }
    except ProviderError as exc:
        return {
            "role": "embedding",
            "provider": spec.provider,
            "adapter": spec.adapter,
            "requested_model": spec.model,
            "status": "failed",
            "error_type": exc.error_type,
            "http_status": exc.status_code,
        }


def _role_specs(
    models: ReadinessModels,
) -> dict[ReadinessRole, LLMModelSpec | EmbeddingModelSpec]:
    return {
        "primary_decision": models.primary_decision,
        "weak_decision": models.weak_decision,
        "cross_provider_reference": models.cross_provider_reference,
        "embedding": models.embedding,
    }


def run_provider_readiness(
    experiment_path: Path,
    fixture_path: Path,
    *,
    environment_label: str,
    selected_roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    models, fixture = load_readiness_inputs(experiment_path, fixture_path)
    roles = tuple(selected_roles) if selected_roles else READINESS_ROLES
    if len(roles) != len(set(roles)):
        raise ValueError("readiness roles must be unique")
    unknown = set(roles) - set(READINESS_ROLES)
    if unknown:
        raise ValueError(f"unknown readiness roles: {sorted(unknown)}")
    specs = _role_specs(models)
    results: list[dict[str, Any]] = []
    runtime = {"system": platform.system(), "machine": platform.machine()}
    for role in roles:
        spec = specs[role]  # type: ignore[index]
        result = (
            _embedding_smoke(spec, fixture)
            if role == "embedding" and isinstance(spec, EmbeddingModelSpec)
            else _llm_smoke(role, spec, fixture)  # type: ignore[arg-type]
        )
        result["execution_environment"] = environment_label
        result["runtime"] = runtime
        results.append(result)
    return {
        "schema_version": 2,
        "record_kind": (
            "provider_readiness"
            if tuple(roles) == READINESS_ROLES
            else "provider_readiness_partial"
        ),
        "status": "passed" if all(result["status"] == "passed" for result in results) else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "fixture_id": fixture.fixture_id,
        "fixture_scope": "synthetic_dev_only",
        "fixture_sha256": sha256_file(fixture_path),
        "experiment_config_sha256": sha256_file(experiment_path),
        "model_authority": models.authority,
        "required_roles": list(READINESS_ROLES),
        "selected_roles": list(roles),
        "secret_policy": READINESS_SECRET_POLICY,
        "results": results,
    }


def _validate_sha256(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"readiness {field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"readiness {field} must be a SHA-256 hex digest") from exc


def _validate_non_negative_number(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"readiness {field} must be a non-negative number")


def _validate_passed_result(
    role: ReadinessRole,
    result: dict[str, Any],
    spec: LLMModelSpec | EmbeddingModelSpec,
) -> None:
    expected_common = {
        "role": role,
        "provider": spec.provider,
        "adapter": spec.adapter,
        "requested_model": spec.model,
        "status": "passed",
    }
    for field, expected in expected_common.items():
        if result.get(field) != expected:
            raise ValueError(f"readiness {role} result disagrees on {field}")
    if (
        not isinstance(result.get("execution_environment"), str)
        or not result["execution_environment"]
    ):
        raise ValueError(f"readiness {role} lacks execution_environment")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict) or not all(
        isinstance(runtime.get(field), str) and runtime[field] for field in ("system", "machine")
    ):
        raise ValueError(f"readiness {role} lacks runtime identity")
    _validate_non_negative_number(result.get("latency_ms"), field=f"{role}.latency_ms")
    _validate_sha256(result.get("raw_response_sha256"), field=f"{role}.raw_response_sha256")

    if isinstance(spec, EmbeddingModelSpec):
        expected_embedding: dict[str, Any] = {
            "reported_model": None,
            "identity_evidence": "model_specific_request_endpoint",
            "identity_reported_by_provider": False,
            "connectivity": True,
            "finite_vector": True,
            "dimension": spec.expected_dimension,
            "expected_dimension": spec.expected_dimension,
            "dimension_match": True,
        }
        for field, expected in expected_embedding.items():
            if result.get(field) != expected:
                raise ValueError(f"readiness embedding result disagrees on {field}")
        return

    expected_llm: dict[str, Any] = {
        "required_arms": spec.required_arms,
        "verdict_authority": spec.verdict_authority,
        "connectivity": True,
        "structured_output_valid": True,
        "structured_output_mode": spec.structured_output_mode,
        "structured_output_contract_match": True,
        "fixture_expectation_match": True,
        "identity_match": True,
        "usage_available": True,
        "usage_contract_match": True,
    }
    for field, expected in expected_llm.items():
        if result.get(field) != expected:
            raise ValueError(f"readiness {role} result disagrees on {field}")
    reported_model = result.get("reported_model")
    if not isinstance(reported_model, str) or reported_model.removeprefix("models/") != spec.model:
        raise ValueError(f"readiness {role} reported_model does not match the config")
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        _validate_non_negative_number(result.get(field), field=f"{role}.{field}")
    if result["total_tokens"] != result["input_tokens"] + result["output_tokens"]:
        raise ValueError(f"readiness {role} token totals are inconsistent")
    usage_fields = result.get("provider_usage_fields")
    if not isinstance(usage_fields, list) or not all(
        isinstance(field, str) for field in usage_fields
    ):
        raise ValueError(f"readiness {role} provider_usage_fields are invalid")
    if not set(spec.expected_usage_fields).issubset(usage_fields):
        raise ValueError(f"readiness {role} lacks configured usage fields")


def _validate_partial_record(
    record: dict[str, Any],
    *,
    models: ReadinessModels,
    fixture: ReadinessFixture,
    experiment_config_sha256: str,
    fixture_sha256: str,
) -> datetime:
    expected_top_level = {
        "schema_version": 2,
        "record_kind": "provider_readiness_partial",
        "status": "passed",
        "fixture_id": fixture.fixture_id,
        "fixture_scope": "synthetic_dev_only",
        "fixture_sha256": fixture_sha256,
        "experiment_config_sha256": experiment_config_sha256,
        "model_authority": models.authority,
        "required_roles": list(READINESS_ROLES),
        "secret_policy": READINESS_SECRET_POLICY,
    }
    for field, expected in expected_top_level.items():
        if record.get(field) != expected:
            raise ValueError(f"readiness partial disagrees with current {field}")
    checked_at_raw = record.get("checked_at")
    if not isinstance(checked_at_raw, str):
        raise ValueError("readiness partial lacks checked_at")
    try:
        checked_at = datetime.fromisoformat(checked_at_raw)
    except ValueError as exc:
        raise ValueError("readiness partial checked_at is not ISO-8601") from exc
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("readiness partial checked_at must include a timezone")

    selected_roles = record.get("selected_roles")
    results = record.get("results")
    if not isinstance(selected_roles, list) or not selected_roles:
        raise ValueError("readiness partial must select at least one role")
    if len(selected_roles) != len(set(selected_roles)):
        raise ValueError("readiness partial selected_roles must be unique")
    if not isinstance(results, list) or len(results) != len(selected_roles):
        raise ValueError("readiness partial results do not match selected_roles")
    role_specs = _role_specs(models)
    for selected_role, result in zip(selected_roles, results, strict=True):
        if selected_role not in READINESS_ROLES:
            raise ValueError(f"readiness partial has unknown role: {selected_role}")
        if not isinstance(result, dict):
            raise ValueError(f"readiness {selected_role} result must be an object")
        role = cast(ReadinessRole, selected_role)
        _validate_passed_result(role, result, role_specs[role])
    return checked_at


def merge_provider_readiness(
    records: Sequence[dict[str, Any]],
    *,
    experiment_path: Path,
    fixture_path: Path,
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one readiness record is required")
    models, fixture = load_readiness_inputs(experiment_path, fixture_path)
    experiment_config_sha256 = sha256_file(experiment_path)
    fixture_sha256 = sha256_file(fixture_path)
    checked_at_values: list[datetime] = []
    for record in records:
        checked_at_values.append(
            _validate_partial_record(
                record,
                models=models,
                fixture=fixture,
                experiment_config_sha256=experiment_config_sha256,
                fixture_sha256=fixture_sha256,
            )
        )
    results: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for record in records:
        for result in record["results"]:
            role = result["role"]
            if role in seen_roles:
                raise ValueError(f"duplicate readiness role: {role}")
            seen_roles.add(role)
            results.append(result)
    required_roles = set(READINESS_ROLES)
    if seen_roles != required_roles:
        raise ValueError(
            f"readiness roles incomplete: missing={sorted(required_roles - seen_roles)} "
            f"extra={sorted(seen_roles - required_roles)}"
        )
    role_order = {role: index for index, role in enumerate(READINESS_ROLES)}
    results.sort(key=lambda result: role_order[result["role"]])
    return {
        "schema_version": 2,
        "record_kind": "provider_readiness",
        "status": "passed",
        "checked_at": max(checked_at_values).isoformat(),
        "fixture_id": fixture.fixture_id,
        "fixture_scope": "synthetic_dev_only",
        "fixture_sha256": fixture_sha256,
        "experiment_config_sha256": experiment_config_sha256,
        "model_authority": models.authority,
        "required_roles": list(READINESS_ROLES),
        "selected_roles": list(READINESS_ROLES),
        "secret_policy": READINESS_SECRET_POLICY,
        "results": results,
    }


def load_readiness_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"readiness record must be an object: {path}")
    return value


def write_readiness_record(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
