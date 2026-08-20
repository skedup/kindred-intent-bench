"""Synthetic-only Provider readiness checks."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from intentbench.adapters.google import GoogleGenerativeLanguageClient, ProviderError
from intentbench.schemas import Decision, StrictModel


class LLMModelSpec(StrictModel):
    provider: Literal["google_generative_language"]
    model: str
    adapter: str
    base_url: str
    api_key_env: str
    timeout_seconds: float = Field(gt=0)
    request_max_output_tokens: int = Field(gt=0)
    capability_context_window: int | None = None
    capability_max_output_tokens: int | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    unsupported_generation_parameters: list[str] = Field(default_factory=list)
    expected_usage_fields: list[str]


class EmbeddingModelSpec(StrictModel):
    provider: Literal["google_generative_language"]
    model: str
    adapter: str
    base_url: str
    api_key_env: str
    timeout_seconds: float = Field(gt=0)
    expected_dimension: int = Field(gt=0)


class ReadinessModels(StrictModel):
    authority: dict[str, Any]
    runtime_candidate_pool: list[dict[str, Any]]
    selection_rationale: dict[str, str]
    primary_decision: LLMModelSpec
    weak_decision: LLMModelSpec
    embedding: EmbeddingModelSpec


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


def _llm_smoke(role: str, spec: LLMModelSpec, fixture: ReadinessFixture) -> dict[str, Any]:
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        return {
            "role": role,
            "provider": spec.provider,
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
        status = "passed" if expected_match and identity_match else "failed"
        return {
            "role": role,
            "provider": spec.provider,
            "requested_model": result.requested_model,
            "reported_model": result.reported_model,
            "status": status,
            "connectivity": True,
            "structured_output_valid": True,
            "fixture_expectation_match": expected_match,
            "identity_match": identity_match,
            "timeout_seconds": spec.timeout_seconds,
            "latency_ms": round(result.latency_ms, 3),
            "usage_available": True,
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
            "requested_model": spec.model,
            "status": "failed",
            "error_type": exc.error_type,
            "http_status": exc.status_code,
        }
    except (ValueError, KeyError) as exc:
        return {
            "role": role,
            "provider": spec.provider,
            "requested_model": spec.model,
            "status": "failed",
            "error_type": type(exc).__name__,
        }


def _embedding_smoke(spec: EmbeddingModelSpec, fixture: ReadinessFixture) -> dict[str, Any]:
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        return {
            "role": "embedding",
            "provider": spec.provider,
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
            result = client.embed(model=spec.model, text=fixture.embedding_text)
        dimension_match = len(result.vector) == spec.expected_dimension
        return {
            "role": "embedding",
            "provider": spec.provider,
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
            "requested_model": spec.model,
            "status": "failed",
            "error_type": exc.error_type,
            "http_status": exc.status_code,
        }


def run_provider_readiness(
    experiment_path: Path, fixture_path: Path, *, environment_label: str
) -> dict[str, Any]:
    models, fixture = load_readiness_inputs(experiment_path, fixture_path)
    results = [
        _llm_smoke("primary_decision", models.primary_decision, fixture),
        _llm_smoke("weak_decision", models.weak_decision, fixture),
        _embedding_smoke(models.embedding, fixture),
    ]
    return {
        "schema_version": 1,
        "status": "passed" if all(result["status"] == "passed" for result in results) else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "fixture_id": fixture.fixture_id,
        "fixture_scope": "synthetic_dev_only",
        "model_authority": models.authority,
        "execution_environment": environment_label,
        "runtime": {"system": platform.system(), "machine": platform.machine()},
        "secret_policy": "environment_variable_name_only; no secret or raw response persisted",
        "results": results,
    }


def write_readiness_record(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
