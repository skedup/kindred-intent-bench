"""IE3 two-call B2b/B3 dev runners and auditable per-call artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from intentbench import __version__
from intentbench.adapters.base import StructuredGenerationClient
from intentbench.b0 import serialize_context
from intentbench.b2 import (
    TAXONOMY_HIDDEN_SHA256,
    FormedIntention,
    IE3LLMExperimentConfig,
    IE3PromptSelectionReceipt,
    StageASupervisionReceipt,
    formed_intention_response_schema,
    render_b2b_verifier_prompt,
    render_b3_stage_a_prompt,
    render_b3_stage_b_prompt,
    stage_a_few_shot_sha256,
    stage_b_few_shot_sha256,
    two_call_prediction_response_schema,
    validate_ie3_prompt_selection_receipt,
    validate_stage_a_supervision,
    verifier_few_shot_sha256,
)
from intentbench.b2a import (
    build_one_stage_request,
    estimate_cost,
    few_shot_payload_sha256,
    normalize_cached_one_stage_prediction,
    prediction_response_schema,
    render_b2a_prompt,
    validate_few_shot_cases,
)
from intentbench.baselines import (
    PREDICTIONS_FILENAME,
    BaselineRunError,
    _git_identity,
    _json_bytes,
    _jsonl_bytes,
    _load_bound_inputs,
    _load_experiment,
    _percentile,
    _sha256_bytes,
    _sha256_json,
    _write_run_bundle,
)
from intentbench.evaluator import evaluate_run
from intentbench.freeze import ArtifactDigest, ExperimentLock, sha256_file, verify_test_guard
from intentbench.generation_cache import (
    CachedGeneration,
    GenerationCache,
    GenerationRequest,
    canonical_json,
    sha256_json,
)
from intentbench.providers import LLMModelSpec, ReadinessModels
from intentbench.reporting import EVALUATOR_VERSION, bind_frozen_split
from intentbench.schemas import (
    Case,
    ModelRole,
    Prediction,
    PredictionStatus,
    RunManifest,
    RunParameters,
    Split,
    StrictModel,
    Taxonomy,
    TokenUsage,
)
from intentbench.taxonomy import load_taxonomy

CALLS_FILENAME = "calls.jsonl"
FORMED_INTENTIONS_FILENAME = "formed-intentions.jsonl"
CALL_1_PROVENANCE_FILENAME = "call-1-provenance.json"
LLM_ROLE_NAMES = (
    "primary_decision",
    "weak_decision",
    "cross_provider_reference",
)
LLMRoleName = Literal["primary_decision", "weak_decision", "cross_provider_reference"]


class LLMCallRecord(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str
    arm: Literal["B2b", "B3"]
    stage: Literal["call_1_one_stage", "call_2_verifier", "stage_a", "stage_b"]
    call_index: Literal[1, 2]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_model: str
    reported_model: str | None
    status: PredictionStatus
    usage: TokenUsage | None
    latency_ms: float = Field(ge=0)
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_type: str | None
    cache_origin: Literal["provider_call", "validated_contract_migration"]
    source_cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    one_stage_cache_compatible: bool = False

    @model_validator(mode="after")
    def validate_status(self) -> LLMCallRecord:
        expected_stage = {
            ("B2b", 1): "call_1_one_stage",
            ("B2b", 2): "call_2_verifier",
            ("B3", 1): "stage_a",
            ("B3", 2): "stage_b",
        }[(self.arm, self.call_index)]
        if self.stage != expected_stage:
            raise ValueError("call stage differs from arm/index contract")
        if self.status is PredictionStatus.SUCCESS and self.error_type is not None:
            raise ValueError("successful call record cannot carry error_type")
        if self.status is PredictionStatus.SUCCESS and (
            self.usage is None or self.raw_response_sha256 is None
        ):
            raise ValueError("successful call record requires usage and raw response identity")
        if self.status is not PredictionStatus.SUCCESS and not self.error_type:
            raise ValueError("failed call record requires error_type")
        if self.one_stage_cache_compatible != (self.stage == "call_1_one_stage"):
            raise ValueError("only B2b call-1 can claim one-stage cache compatibility")
        if self.cache_origin == "provider_call" and self.source_cache_key is not None:
            raise ValueError("direct calls cannot carry migration provenance")
        if self.cache_origin == "validated_contract_migration" and self.source_cache_key is None:
            raise ValueError("migrated calls require a source cache key")
        return self


class FormedIntentionRecord(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str
    status: PredictionStatus
    formed_intention: FormedIntention | None
    error_type: str | None
    usage: TokenUsage | None
    latency_ms: float = Field(ge=0)
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> FormedIntentionRecord:
        if self.status is PredictionStatus.SUCCESS:
            if self.formed_intention is None or self.error_type is not None:
                raise ValueError("successful formed intention requires value and no error")
        elif self.formed_intention is not None or not self.error_type:
            raise ValueError("failed formed intention requires only an error")
        return self


class Call1ProvenanceEntry(StrictModel):
    case_id: str
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_origin: Literal["provider_call", "validated_contract_migration"]
    source_cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class Call1Provenance(StrictModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["b2b_one_stage_compatible_call_1"] = "b2b_one_stage_compatible_call_1"
    role: LLMRoleName
    model: str
    one_stage_call_contract_version: Literal[2] = 2
    cache_policy: Literal["require_existing_one_stage_v2"] = "require_existing_one_stage_v2"
    entries: list[Call1ProvenanceEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> Call1Provenance:
        ids = [entry.case_id for entry in self.entries]
        keys = [entry.cache_key for entry in self.entries]
        if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
            raise ValueError("call-1 provenance cases and cache keys must be unique")
        for entry in self.entries:
            if entry.cache_origin == "provider_call" and entry.source_cache_key is not None:
                raise ValueError("direct provenance entries cannot carry a migration source")
            if (
                entry.cache_origin == "validated_contract_migration"
                and entry.source_cache_key is None
            ):
                raise ValueError("migrated provenance entries require a source cache key")
        return self


def _spec_for_role(models: ReadinessModels, role: LLMRoleName) -> LLMModelSpec:
    return {
        "primary_decision": models.primary_decision,
        "weak_decision": models.weak_decision,
        "cross_provider_reference": models.cross_provider_reference,
    }[role]


def _artifact_path(
    *, name: str, experiment: Any, repository_root: Path
) -> tuple[Path, ArtifactDigest]:
    digest = experiment.artifacts.get(name)
    if digest is None:
        raise BaselineRunError(f"registered artifact is missing: {name}")
    return (repository_root.resolve() / digest.path).resolve(), digest


def _usage_from_cached(cached: CachedGeneration) -> TokenUsage | None:
    if cached.input_tokens is None or cached.output_tokens is None or cached.total_tokens is None:
        return None
    return TokenUsage(
        input_tokens=cached.input_tokens,
        output_tokens=cached.output_tokens,
        total_tokens=cached.total_tokens,
    )


def _status_and_error(cached: CachedGeneration) -> tuple[PredictionStatus, str | None]:
    if cached.result is not None:
        return PredictionStatus.SUCCESS, None
    if cached.error_type == "schema_invalid":
        return PredictionStatus.SCHEMA_INVALID, cached.error_type
    return PredictionStatus.PROVIDER_FAILURE, cached.error_type or "provider_failure"


def _call_record(
    *,
    case_id: str,
    arm: Literal["B2b", "B3"],
    stage: Literal["call_1_one_stage", "call_2_verifier", "stage_a", "stage_b"],
    call_index: Literal[1, 2],
    cached: CachedGeneration,
    status: PredictionStatus | None = None,
    error_type: str | None = None,
) -> LLMCallRecord:
    raw_status, raw_error = _status_and_error(cached)
    return LLMCallRecord(
        case_id=case_id,
        arm=arm,
        stage=stage,
        call_index=call_index,
        cache_key=cached.cache_key,
        call_contract_sha256=cached.call_contract_sha256,
        requested_model=cached.call_contract.model,
        reported_model=cached.reported_model,
        status=status or raw_status,
        usage=_usage_from_cached(cached),
        latency_ms=cached.latency_ms,
        raw_response_sha256=cached.raw_response_sha256,
        error_type=error_type if status is not None else raw_error,
        cache_origin=cached.origin,
        source_cache_key=cached.source_cache_key,
        one_stage_cache_compatible=stage == "call_1_one_stage",
    )


def _aggregate_call_usage(calls: Sequence[LLMCallRecord]) -> TokenUsage | None:
    if not calls or any(call.usage is None for call in calls):
        return None
    usages = [call.usage for call in calls]
    assert all(usage is not None for usage in usages)
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages if usage is not None),
        output_tokens=sum(usage.output_tokens for usage in usages if usage is not None),
        total_tokens=sum(usage.total_tokens for usage in usages if usage is not None),
    )


def _finalize_prediction(prediction: Prediction, calls: Sequence[LLMCallRecord]) -> Prediction:
    raw_identity = sha256_json(
        [
            call.raw_response_sha256 or f"{call.call_contract_sha256}:{call.error_type}"
            for call in calls
        ]
    )
    return prediction.model_copy(
        update={
            "usage": _aggregate_call_usage(calls),
            "latency_ms": sum(call.latency_ms for call in calls),
            "raw_response_sha256": raw_identity,
        }
    )


def _failed_prediction(
    *, case_id: str, error_type: str, calls: Sequence[LLMCallRecord]
) -> Prediction:
    return _finalize_prediction(
        Prediction(
            case_id=case_id,
            status=PredictionStatus.PROVIDER_FAILURE,
            error_type=error_type,
        ),
        calls,
    )


def _generation_request(
    *,
    normalized_input: str,
    prompt: str,
    taxonomy_sha256: str,
    prompt_template_sha256: str,
    few_shot_sha256: str,
    response_schema: dict[str, Any],
    spec: LLMModelSpec,
    max_output_tokens: int,
) -> GenerationRequest:
    return GenerationRequest(
        normalized_context=normalized_input,
        prompt=prompt,
        taxonomy_sha256=taxonomy_sha256,
        prompt_template_sha256=prompt_template_sha256,
        few_shot_payload_sha256=few_shot_sha256,
        response_schema=response_schema,
        model=spec.model,
        adapter=spec.adapter,
        adapter_revision=spec.adapter_revision,
        structured_output_mode=spec.structured_output_mode,
        max_output_tokens=max_output_tokens,
        generation_parameters=dict(spec.generation_parameters),
    )


def _load_contract(
    *,
    split: Split,
    role: LLMRoleName,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    repository_root: Path,
) -> tuple[
    list[Case],
    Taxonomy,
    Any,
    IE3LLMExperimentConfig,
    ReadinessModels,
    LLMModelSpec,
    tuple[Case, ...],
    ExperimentLock,
]:
    if split is Split.DEV:
        cases, taxonomy, dataset, experiment, _payload = _load_bound_inputs(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            repository_root=repository_root,
        )
        dev_cases = cases
    else:
        dataset, experiment = verify_test_guard(
            dataset_manifest_path,
            experiment_path,
            repository_root,
        )
        cases, _test_dataset = bind_frozen_split(
            split=Split.TEST,
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            repository_root=repository_root,
        )
        dev_cases, _dev_dataset = bind_frozen_split(
            split=Split.DEV,
            cases_path=dev_cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            repository_root=repository_root,
        )
        taxonomy = load_taxonomy(taxonomy_path)
        _commit, dirty = _git_identity(repository_root)
        if dirty:
            raise BaselineRunError("formal test run requires a clean committed worktree")
    try:
        config = IE3LLMExperimentConfig.model_validate(experiment.llm)
        models = ReadinessModels.model_validate(experiment.models)
    except ValidationError as exc:
        raise BaselineRunError("experiment two-call/model config is invalid") from exc
    spec = _spec_for_role(models, role)
    budget = config.b2b.request_max_output_tokens_per_call
    if budget != config.b3.request_max_output_tokens_per_call:
        raise BaselineRunError("B2b/B3 per-call budgets differ")
    if spec.request_max_output_tokens != budget:
        raise BaselineRunError(f"{role} model and two-call output budgets differ")
    few_shots = validate_few_shot_cases(dev_cases, config.shared_few_shot)
    prompt_receipt_path, _prompt_receipt_digest = _artifact_path(
        name=config.b2b.prompt_selection_receipt_artifact,
        experiment=experiment,
        repository_root=repository_root,
    )
    try:
        prompt_receipt = IE3PromptSelectionReceipt.model_validate(
            yaml.safe_load(prompt_receipt_path.read_text(encoding="utf-8"))
        )
        validate_ie3_prompt_selection_receipt(
            receipt=prompt_receipt,
            config=config,
            dataset_version=dataset.dataset_version,
            dev_sha256=sha256_file(dev_cases_path),
            primary_provider=models.primary_decision.provider,
            primary_model=models.primary_decision.model,
            selected_prompt_hashes={
                "B2b": {
                    "verifier": experiment.artifacts[config.b2b.verifier_prompt_artifact].sha256
                },
                "B3": {
                    "stage_a": experiment.artifacts[config.b3.stage_a_prompt_artifact].sha256,
                    "stage_b": experiment.artifacts[config.b3.stage_b_prompt_artifact].sha256,
                },
            },
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise BaselineRunError("IE3 Prompt selection receipt is invalid") from exc
    return cases, taxonomy, dataset, config, models, spec, few_shots, experiment


def _one_stage_inputs(
    *,
    case: Case,
    taxonomy: Taxonomy,
    taxonomy_path: Path,
    experiment: Any,
    config: IE3LLMExperimentConfig,
    spec: LLMModelSpec,
    few_shots: Sequence[Case],
    repository_root: Path,
) -> GenerationRequest:
    prompt_path, prompt_digest = _artifact_path(
        name=config.b2a.prompt_artifact,
        experiment=experiment,
        repository_root=repository_root,
    )
    context_json = serialize_context(case.context)
    prompt = render_b2a_prompt(
        template=prompt_path.read_text(encoding="utf-8"),
        taxonomy=taxonomy,
        few_shots=few_shots,
        context_json=context_json,
    )
    return build_one_stage_request(
        context_json=context_json,
        prompt=prompt,
        taxonomy_sha256=sha256_file(taxonomy_path),
        prompt_template_sha256=prompt_digest.sha256,
        few_shot_sha256=few_shot_payload_sha256(few_shots),
        response_schema=prediction_response_schema(taxonomy),
        spec=spec,
        max_output_tokens=config.b2a.request_max_output_tokens,
    )


def _prepare_one_stage_cache(
    *,
    split: Split,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Populate one-stage-compatible cache before B2b; this is not a scoring arm."""

    cases, taxonomy, _dataset, config, models, spec, few_shots, experiment = _load_contract(
        split=split,
        role=role,
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    del models
    cache = GenerationCache(cache_dir)
    cache_hits = 0
    provider_calls = 0
    success_count = 0
    for case in cases:
        request = _one_stage_inputs(
            case=case,
            taxonomy=taxonomy,
            taxonomy_path=taxonomy_path,
            experiment=experiment,
            config=config,
            spec=spec,
            few_shots=few_shots,
            repository_root=repository_root,
        )
        cached = cache.get_or_create(client=client, request=request)
        cache_hits += int(cached.cache_hit)
        provider_calls += int(not cached.cache_hit)
        success_count += int(cached.result is not None)
    return {
        "role": role,
        "model": spec.model,
        "split": split.value,
        "case_count": len(cases),
        "success_count": success_count,
        "cache_hits": cache_hits,
        "provider_calls": provider_calls,
        "scoring_arm_created": False,
    }


def prepare_one_stage_dev_cache(
    *,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    return _prepare_one_stage_cache(
        split=Split.DEV,
        client=client,
        role=role,
        cache_dir=cache_dir,
        cases_path=cases_path,
        dev_cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )


def prepare_one_stage_test_cache(
    *,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    return _prepare_one_stage_cache(
        split=Split.TEST,
        client=client,
        role=role,
        cache_dir=cache_dir,
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )


def _manifest_and_write(
    *,
    split: Split,
    arm: Literal["B2b", "B3"],
    role: LLMRoleName,
    spec: LLMModelSpec,
    cases: Sequence[Case],
    taxonomy: Taxonomy,
    dataset: Any,
    models: ReadinessModels,
    config: IE3LLMExperimentConfig,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    repository_root: Path,
    output_dir: Path,
    predictions: Sequence[Prediction],
    calls: Sequence[LLMCallRecord],
    extra_payloads: dict[str, bytes],
    prompt_hashes: dict[str, str],
    cache_hits: int,
    provider_calls: int,
) -> dict[str, object]:
    predictions_payload = _jsonl_bytes(
        [prediction.model_dump(mode="json") for prediction in predictions]
    )
    calls_payload = _jsonl_bytes([call.model_dump(mode="json") for call in calls])
    total_usage = _aggregate_call_usage(calls)
    estimated_cost = (
        estimate_cost(total_usage, config.pricing)
        if role == "primary_decision" and total_usage is not None
        else None
    )
    latencies = [prediction.latency_ms or 0 for prediction in predictions]
    raw_response_sha256 = _sha256_json(
        [prediction.raw_response_sha256 for prediction in predictions]
    )
    adapter_config_sha256 = _sha256_json(
        {
            "arm": arm,
            "role": role,
            "model": spec.model_dump(mode="json"),
            "arm_config": getattr(config, arm.lower()).model_dump(mode="json"),
            "prompt_hashes": prompt_hashes,
            "shared_few_shot_case_ids": config.shared_few_shot.case_ids,
        }
    )
    dependency = _load_experiment(experiment_path)[0].artifacts.get("dependency_lock")
    if dependency is None:
        raise BaselineRunError("experiment lacks dependency_lock artifact")
    commit, dirty = _git_identity(repository_root)
    manifest = RunManifest(
        run_id=(
            f"{dataset.dataset_version}-{arm.lower()}-{role}-{split.value}-"
            f"{adapter_config_sha256[:12]}"
        ),
        dataset_version=dataset.dataset_version,
        dataset_sha256=sha256_file(cases_path),
        dataset_freeze_sha256=sha256_file(dataset_manifest_path),
        experiment_lock_sha256=sha256_file(experiment_path),
        taxonomy_version=taxonomy.taxonomy_version,
        adapter=spec.adapter,
        adapter_config_sha256=adapter_config_sha256,
        model=f"{spec.provider}/{spec.model}",
        model_role=ModelRole(role),
        primary_decision_model=(
            f"{models.primary_decision.provider}/{models.primary_decision.model}"
        ),
        embedding_model=None,
        raw_response_sha256=raw_response_sha256,
        prompt_sha256=sha256_json(prompt_hashes),
        evaluator_version=EVALUATOR_VERSION,
        parameters=RunParameters(
            temperature=None,
            top_p=None,
            seed=None,
            max_calls=2,
            max_output_tokens_per_call=config.b2b.request_max_output_tokens_per_call,
            retry_policy=config.b2b.retry_policy,
            schema_repair_policy=config.b2b.schema_repair_policy,
        ),
        pricing_snapshot_date=(
            config.pricing.snapshot_date if role == "primary_decision" else "not_registered"
        ),
        dependency_lock_sha256=dependency.sha256,
        started_at=datetime.now(timezone.utc),
        git_commit=commit,
        total_usage=total_usage,
        estimated_cost=estimated_cost,
        metadata={
            "arm": arm,
            "split": split.value,
            "git_worktree_dirty": dirty,
            "gold_fields_available_to_adapter": False,
            "prediction_sha256": _sha256_bytes(predictions_payload),
            "calls_sha256": _sha256_bytes(calls_payload),
            "prompt_hashes": prompt_hashes,
            "shared_few_shot_case_ids": config.shared_few_shot.case_ids,
            "call_count": len(calls),
            "complete_two_call_case_count": sum(
                1 for case in cases if sum(call.case_id == case.id for call in calls) == 2
            ),
            "generation_cache_hits": cache_hits,
            "provider_calls": provider_calls,
            "latency_p50_ms": _percentile(latencies, 0.5),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "pricing_status": (
                "registered"
                if role == "primary_decision" and total_usage is not None
                else "not_registered"
            ),
            "usage_complete": total_usage is not None,
            "output_artifact_sha256": {
                PREDICTIONS_FILENAME: _sha256_bytes(predictions_payload),
                CALLS_FILENAME: _sha256_bytes(calls_payload),
                **{
                    name: _sha256_bytes(payload) for name, payload in sorted(extra_payloads.items())
                },
            },
            "intentbench_version": __version__,
        },
    )
    status = _write_run_bundle(
        output_dir=output_dir,
        deterministic_payloads={
            PREDICTIONS_FILENAME: predictions_payload,
            CALLS_FILENAME: calls_payload,
            **extra_payloads,
        },
        manifest=manifest,
    )
    metrics = evaluate_run(cases, predictions, taxonomy)
    return {
        "status": status,
        "arm": arm,
        "role": role,
        "split": split.value,
        "prediction_count": len(predictions),
        "hierarchical_exact_match": metrics["hierarchical_exact_match"],
        "decision_macro_f1": metrics["decision_macro_f1"],
        "total_input_tokens": total_usage.input_tokens if total_usage is not None else None,
        "total_output_tokens": total_usage.output_tokens if total_usage is not None else None,
        "estimated_cost_usd": estimated_cost,
        "generation_cache_hits": cache_hits,
        "provider_calls": provider_calls,
        "output_dir": str(output_dir),
    }


def _run_b2b(
    *,
    split: Split,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    cases, taxonomy, dataset, config, models, spec, few_shots, experiment = _load_contract(
        split=split,
        role=role,
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    verifier_path, verifier_digest = _artifact_path(
        name=config.b2b.verifier_prompt_artifact,
        experiment=experiment,
        repository_root=repository_root,
    )
    verifier_template = verifier_path.read_text(encoding="utf-8")
    response_schema = two_call_prediction_response_schema(
        taxonomy, openai_strict_compatible=spec.provider == "openai"
    )
    taxonomy_sha256 = sha256_file(taxonomy_path)
    verifier_shots_sha256 = verifier_few_shot_sha256(few_shots)
    cache = GenerationCache(cache_dir)
    predictions: list[Prediction] = []
    calls: list[LLMCallRecord] = []
    provenance_entries: list[Call1ProvenanceEntry] = []
    cache_hits = 0
    provider_calls = 0

    for case in cases:
        one_stage_request = _one_stage_inputs(
            case=case,
            taxonomy=taxonomy,
            taxonomy_path=taxonomy_path,
            experiment=experiment,
            config=config,
            spec=spec,
            few_shots=few_shots,
            repository_root=repository_root,
        )
        first = cache.require_existing(request=one_stage_request)
        cache_hits += 1
        draft = normalize_cached_one_stage_prediction(
            case_id=case.id,
            cached=first,
            taxonomy=taxonomy,
            spec=spec,
        )
        first_record = _call_record(
            case_id=case.id,
            arm="B2b",
            stage="call_1_one_stage",
            call_index=1,
            cached=first,
            status=draft.status,
            error_type=draft.error_type,
        )
        calls.append(first_record)
        provenance_entries.append(
            Call1ProvenanceEntry(
                case_id=case.id,
                cache_key=first.cache_key,
                call_contract_sha256=first.call_contract_sha256,
                cache_origin=first.origin,
                source_cache_key=first.source_cache_key,
            )
        )
        if draft.status is not PredictionStatus.SUCCESS:
            predictions.append(
                _finalize_prediction(
                    draft.model_copy(update={"error_type": f"call_1:{draft.error_type}"}),
                    [first_record],
                )
            )
            continue

        context_json = serialize_context(case.context)
        draft_json = canonical_json(
            draft.model_dump(
                mode="json",
                include={
                    "decision",
                    "predicted_intent",
                    "candidate_intents",
                    "slots",
                    "reason_short",
                },
            )
        )
        verifier_prompt = render_b2b_verifier_prompt(
            template=verifier_template,
            taxonomy=taxonomy,
            few_shots=few_shots,
            context_json=context_json,
            draft_prediction_json=draft_json,
        )
        second_request = _generation_request(
            normalized_input=canonical_json(
                {"context": case.context.model_dump(mode="json"), "draft": draft_json}
            ),
            prompt=verifier_prompt,
            taxonomy_sha256=taxonomy_sha256,
            prompt_template_sha256=verifier_digest.sha256,
            few_shot_sha256=verifier_shots_sha256,
            response_schema=response_schema,
            spec=spec,
            max_output_tokens=config.b2b.request_max_output_tokens_per_call,
        )
        second = cache.get_or_create(client=client, request=second_request)
        cache_hits += int(second.cache_hit)
        provider_calls += int(not second.cache_hit)
        verified = normalize_cached_one_stage_prediction(
            case_id=case.id,
            cached=second,
            taxonomy=taxonomy,
            spec=spec,
        )
        second_record = _call_record(
            case_id=case.id,
            arm="B2b",
            stage="call_2_verifier",
            call_index=2,
            cached=second,
            status=verified.status,
            error_type=verified.error_type,
        )
        calls.append(second_record)
        predictions.append(_finalize_prediction(verified, [first_record, second_record]))

    provenance = Call1Provenance(
        role=role,
        model=spec.model,
        entries=provenance_entries,
    )
    return _manifest_and_write(
        split=split,
        arm="B2b",
        role=role,
        spec=spec,
        cases=cases,
        taxonomy=taxonomy,
        dataset=dataset,
        models=models,
        config=config,
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
        output_dir=output_dir,
        predictions=predictions,
        calls=calls,
        extra_payloads={
            CALL_1_PROVENANCE_FILENAME: _json_bytes(provenance.model_dump(mode="json"))
        },
        prompt_hashes={
            "call_1": experiment.artifacts[config.b2a.prompt_artifact].sha256,
            "call_2": verifier_digest.sha256,
        },
        cache_hits=cache_hits,
        provider_calls=provider_calls,
    )


def run_b2b_dev(
    *,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    return _run_b2b(
        split=Split.DEV,
        client=client,
        role=role,
        cache_dir=cache_dir,
        cases_path=cases_path,
        dev_cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        output_dir=output_dir,
        repository_root=repository_root,
    )


def run_b2b_test(
    *,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    return _run_b2b(
        split=Split.TEST,
        client=client,
        role=role,
        cache_dir=cache_dir,
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        output_dir=output_dir,
        repository_root=repository_root,
    )


def _normalize_formed_intention(
    *, case_id: str, cached: CachedGeneration, spec: LLMModelSpec, taxonomy: Taxonomy
) -> FormedIntentionRecord:
    usage = _usage_from_cached(cached)
    if cached.result is None:
        return FormedIntentionRecord(
            case_id=case_id,
            status=(
                PredictionStatus.SCHEMA_INVALID
                if cached.error_type == "schema_invalid"
                else PredictionStatus.PROVIDER_FAILURE
            ),
            formed_intention=None,
            error_type=cached.error_type or "provider_failure",
            usage=usage,
            latency_ms=cached.latency_ms,
            raw_response_sha256=cached.raw_response_sha256,
        )
    result = cached.result
    reported_identity = (result.reported_model or "").removeprefix("models/")
    contract_error: str | None = None
    if reported_identity != spec.model:
        contract_error = "model_identity_mismatch"
    elif result.structured_output_mode != spec.structured_output_mode:
        contract_error = "structured_output_mode_mismatch"
    elif not set(spec.expected_usage_fields) <= set(result.usage_metadata):
        contract_error = "usage_contract_mismatch"
    if contract_error is not None:
        return FormedIntentionRecord(
            case_id=case_id,
            status=PredictionStatus.PROVIDER_FAILURE,
            formed_intention=None,
            error_type=contract_error,
            usage=usage,
            latency_ms=result.latency_ms,
            raw_response_sha256=result.raw_response_sha256,
        )
    try:
        formed = FormedIntention.model_validate(result.value)
        serialized = canonical_json(formed.model_dump(mode="json"))
        forbidden = [taxonomy.taxonomy_version, *(intent.name for intent in taxonomy.intents)]
        if any(term in serialized for term in forbidden):
            raise ValueError("stage-A output contains taxonomy identity")
    except (ValidationError, ValueError) as exc:
        return FormedIntentionRecord(
            case_id=case_id,
            status=PredictionStatus.SCHEMA_INVALID,
            formed_intention=None,
            error_type=f"local_contract:{type(exc).__name__}",
            usage=usage,
            latency_ms=result.latency_ms,
            raw_response_sha256=result.raw_response_sha256,
        )
    return FormedIntentionRecord(
        case_id=case_id,
        status=PredictionStatus.SUCCESS,
        formed_intention=formed,
        error_type=None,
        usage=usage,
        latency_ms=result.latency_ms,
        raw_response_sha256=result.raw_response_sha256,
    )


def _run_b3(
    *,
    split: Split,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    cases, taxonomy, dataset, config, models, spec, few_shots, experiment = _load_contract(
        split=split,
        role=role,
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    stage_a_path, stage_a_digest = _artifact_path(
        name=config.b3.stage_a_prompt_artifact,
        experiment=experiment,
        repository_root=repository_root,
    )
    stage_b_path, stage_b_digest = _artifact_path(
        name=config.b3.stage_b_prompt_artifact,
        experiment=experiment,
        repository_root=repository_root,
    )
    supervision_path, _supervision_digest = _artifact_path(
        name=config.b3.stage_a_supervision_artifact,
        experiment=experiment,
        repository_root=repository_root,
    )
    try:
        supervision = StageASupervisionReceipt.model_validate(
            yaml.safe_load(supervision_path.read_text(encoding="utf-8"))
        )
        validate_stage_a_supervision(
            receipt=supervision,
            selected_cases=few_shots,
            dataset_version=dataset.dataset_version,
            dev_sha256=sha256_file(dev_cases_path),
            selection_receipt_artifact=config.shared_few_shot.selection_receipt_artifact,
        )
        if supervision.representation_contract != config.b3.formed_intention_contract:
            raise ValueError("B3 formed-intention contract differs from supervision receipt")
    except (OSError, ValidationError, ValueError) as exc:
        raise BaselineRunError("B3 stage-A supervision receipt is invalid") from exc

    stage_a_template = stage_a_path.read_text(encoding="utf-8")
    stage_b_template = stage_b_path.read_text(encoding="utf-8")
    stage_a_schema = formed_intention_response_schema()
    stage_b_schema = two_call_prediction_response_schema(
        taxonomy, openai_strict_compatible=spec.provider == "openai"
    )
    taxonomy_sha256 = sha256_file(taxonomy_path)
    stage_a_shots_sha256 = stage_a_few_shot_sha256(few_shots, supervision)
    stage_b_shots_sha256 = stage_b_few_shot_sha256(few_shots, supervision)
    cache = GenerationCache(cache_dir)
    predictions: list[Prediction] = []
    formed_records: list[FormedIntentionRecord] = []
    calls: list[LLMCallRecord] = []
    cache_hits = 0
    provider_calls = 0

    for case in cases:
        context_json = serialize_context(case.context)
        stage_a_prompt = render_b3_stage_a_prompt(
            template=stage_a_template,
            few_shots=few_shots,
            receipt=supervision,
            context_json=context_json,
        )
        stage_a_request = _generation_request(
            normalized_input=context_json,
            prompt=stage_a_prompt,
            taxonomy_sha256=TAXONOMY_HIDDEN_SHA256,
            prompt_template_sha256=stage_a_digest.sha256,
            few_shot_sha256=stage_a_shots_sha256,
            response_schema=stage_a_schema,
            spec=spec,
            max_output_tokens=config.b3.request_max_output_tokens_per_call,
        )
        first = cache.get_or_create(client=client, request=stage_a_request)
        cache_hits += int(first.cache_hit)
        provider_calls += int(not first.cache_hit)
        formed_record = _normalize_formed_intention(
            case_id=case.id,
            cached=first,
            spec=spec,
            taxonomy=taxonomy,
        )
        formed_records.append(formed_record)
        first_call = _call_record(
            case_id=case.id,
            arm="B3",
            stage="stage_a",
            call_index=1,
            cached=first,
            status=formed_record.status,
            error_type=formed_record.error_type,
        )
        calls.append(first_call)
        if formed_record.status is not PredictionStatus.SUCCESS:
            predictions.append(
                _failed_prediction(
                    case_id=case.id,
                    error_type=f"stage_a:{formed_record.error_type}",
                    calls=[first_call],
                )
            )
            continue
        assert formed_record.formed_intention is not None
        formed_json = canonical_json(formed_record.formed_intention.model_dump(mode="json"))
        stage_b_prompt = render_b3_stage_b_prompt(
            template=stage_b_template,
            taxonomy=taxonomy,
            few_shots=few_shots,
            receipt=supervision,
            formed_intention_json=formed_json,
        )
        stage_b_request = _generation_request(
            normalized_input=canonical_json(
                {
                    "source_context_sha256": sha256_json(case.context.model_dump(mode="json")),
                    "formed_intention": formed_record.formed_intention.model_dump(mode="json"),
                }
            ),
            prompt=stage_b_prompt,
            taxonomy_sha256=taxonomy_sha256,
            prompt_template_sha256=stage_b_digest.sha256,
            few_shot_sha256=stage_b_shots_sha256,
            response_schema=stage_b_schema,
            spec=spec,
            max_output_tokens=config.b3.request_max_output_tokens_per_call,
        )
        second = cache.get_or_create(client=client, request=stage_b_request)
        cache_hits += int(second.cache_hit)
        provider_calls += int(not second.cache_hit)
        grounded = normalize_cached_one_stage_prediction(
            case_id=case.id,
            cached=second,
            taxonomy=taxonomy,
            spec=spec,
        )
        second_call = _call_record(
            case_id=case.id,
            arm="B3",
            stage="stage_b",
            call_index=2,
            cached=second,
            status=grounded.status,
            error_type=grounded.error_type,
        )
        calls.append(second_call)
        predictions.append(_finalize_prediction(grounded, [first_call, second_call]))

    formed_payload = _jsonl_bytes([record.model_dump(mode="json") for record in formed_records])
    return _manifest_and_write(
        split=split,
        arm="B3",
        role=role,
        spec=spec,
        cases=cases,
        taxonomy=taxonomy,
        dataset=dataset,
        models=models,
        config=config,
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
        output_dir=output_dir,
        predictions=predictions,
        calls=calls,
        extra_payloads={FORMED_INTENTIONS_FILENAME: formed_payload},
        prompt_hashes={"stage_a": stage_a_digest.sha256, "stage_b": stage_b_digest.sha256},
        cache_hits=cache_hits,
        provider_calls=provider_calls,
    )


def run_b3_dev(
    *,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    return _run_b3(
        split=Split.DEV,
        client=client,
        role=role,
        cache_dir=cache_dir,
        cases_path=cases_path,
        dev_cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        output_dir=output_dir,
        repository_root=repository_root,
    )


def run_b3_test(
    *,
    client: StructuredGenerationClient,
    role: LLMRoleName,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    return _run_b3(
        split=Split.TEST,
        client=client,
        role=role,
        cache_dir=cache_dir,
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        output_dir=output_dir,
        repository_root=repository_root,
    )
