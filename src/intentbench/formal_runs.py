"""Formal frozen-test B0/B1/B2a runners used by the IE3 matrix."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from intentbench import __version__
from intentbench.adapters.base import EmbeddingClient, StructuredGenerationClient
from intentbench.b0 import B0Config, predict_b0, serialize_context
from intentbench.b1 import (
    B1Config,
    B1Thresholds,
    normalized_centroid,
    score_embedding,
)
from intentbench.b2a import (
    estimate_cost,
    normalize_cached_one_stage_prediction,
)
from intentbench.baselines import (
    B2A_CACHE_BUNDLE_FILENAME,
    B2A_CACHE_PROVENANCE_FILENAME,
    B2A_CALLS_FILENAME,
    PREDICTIONS_FILENAME,
    B2ACacheProvenance,
    B2ACacheProvenanceEntry,
    B2ACallRecord,
    BaselineRunError,
    _git_identity,
    _json_bytes,
    _jsonl_bytes,
    _prediction_from_b1_scores,
    _prototype_context,
    _sha256_bytes,
    _sha256_json,
    _write_run_bundle,
)
from intentbench.embedding import EmbeddingCache
from intentbench.evaluator import evaluate_run
from intentbench.freeze import ExperimentLock, sha256_file, verify_test_guard
from intentbench.generation_cache import GenerationCache
from intentbench.providers import ReadinessModels
from intentbench.reporting import EVALUATOR_VERSION, bind_frozen_split
from intentbench.schemas import (
    Case,
    ModelRole,
    Prediction,
    RunManifest,
    RunParameters,
    Split,
    Taxonomy,
    TokenUsage,
)
from intentbench.taxonomy import load_taxonomy, validate_prediction
from intentbench.two_stage import (
    _load_contract,
    _one_stage_inputs,
)

B1_TEST_PROVENANCE_FILENAME = "b1-test-provenance.json"


class FormalRunError(BaselineRunError):
    """A formal frozen-test runner violates freeze or clean-run requirements."""


def _require_clean(repository_root: Path) -> tuple[str, bool]:
    commit, dirty = _git_identity(repository_root)
    if dirty:
        raise FormalRunError("formal test run requires a clean committed worktree")
    return commit, dirty


def _load_formal_inputs(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    repository_root: Path,
) -> tuple[list[Case], Taxonomy, Any, ExperimentLock, ReadinessModels]:
    dataset, experiment = verify_test_guard(
        dataset_manifest_path,
        experiment_path,
        repository_root,
    )
    cases, _dataset = bind_frozen_split(
        split=Split.TEST,
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
    )
    try:
        models = ReadinessModels.model_validate(experiment.models)
    except ValidationError as exc:
        raise FormalRunError("frozen model contract is invalid") from exc
    _require_clean(repository_root)
    return cases, load_taxonomy(taxonomy_path), dataset, experiment, models


def _strict_prediction_usage(predictions: list[Prediction]) -> TokenUsage | None:
    if any(prediction.usage is None for prediction in predictions):
        return None
    usages = [prediction.usage for prediction in predictions]
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages if usage is not None),
        output_tokens=sum(usage.output_tokens for usage in usages if usage is not None),
        retry_input_tokens=sum(usage.retry_input_tokens for usage in usages if usage is not None),
        retry_output_tokens=sum(usage.retry_output_tokens for usage in usages if usage is not None),
        total_tokens=sum(usage.total_tokens for usage in usages if usage is not None),
    )


def _formal_manifest(
    *,
    run_id: str,
    arm: Literal["B0", "B1", "B2a"],
    adapter: str,
    adapter_config_sha256: str,
    model: str,
    model_role: ModelRole | Literal["none"],
    embedding_model: str | None,
    dataset: Any,
    cases_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    taxonomy: Taxonomy,
    models: ReadinessModels,
    repository_root: Path,
    metadata: dict[str, Any],
    prompt_sha256: str | None = None,
    raw_response_sha256: str | None = None,
    parameters: RunParameters | None = None,
    pricing_snapshot_date: str = "not_applicable",
    total_usage: TokenUsage | None = None,
    estimated_cost: float | None = None,
) -> RunManifest:
    commit, dirty = _require_clean(repository_root)
    experiment = verify_test_guard(
        dataset_manifest_path,
        experiment_path,
        repository_root,
    )[1]
    dependency = experiment.artifacts.get("dependency_lock")
    if dependency is None:
        raise FormalRunError("frozen experiment lacks dependency lock")
    return RunManifest(
        run_id=run_id,
        dataset_version=dataset.dataset_version,
        dataset_sha256=sha256_file(cases_path),
        dataset_freeze_sha256=sha256_file(dataset_manifest_path),
        experiment_lock_sha256=sha256_file(experiment_path),
        taxonomy_version=taxonomy.taxonomy_version,
        adapter=adapter,
        adapter_config_sha256=adapter_config_sha256,
        model=model,
        model_role=model_role,
        primary_decision_model=(
            f"{models.primary_decision.provider}/{models.primary_decision.model}"
        ),
        embedding_model=embedding_model,
        prompt_sha256=prompt_sha256,
        raw_response_sha256=raw_response_sha256,
        evaluator_version=EVALUATOR_VERSION,
        parameters=parameters
        or RunParameters(
            temperature=None,
            top_p=None,
            seed=None,
            max_calls=0 if arm == "B0" else 1,
            max_output_tokens_per_call=1,
            retry_policy="none",
            schema_repair_policy="reject",
        ),
        pricing_snapshot_date=pricing_snapshot_date,
        dependency_lock_sha256=dependency.sha256,
        started_at=datetime.now(timezone.utc),
        git_commit=commit,
        total_usage=total_usage,
        estimated_cost=estimated_cost,
        metadata={
            "arm": arm,
            "split": "test",
            "git_worktree_dirty": dirty,
            "gold_fields_available_to_adapter": False,
            **metadata,
        },
    )


def run_b0_test(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    cases, taxonomy, dataset, experiment, models = _load_formal_inputs(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    try:
        config = B0Config.model_validate(experiment.b0)
    except ValidationError as exc:
        raise FormalRunError("frozen B0 config is invalid") from exc
    predictions = [
        predict_b0(case_id=case.id, context=case.context, config=config) for case in cases
    ]
    for prediction in predictions:
        validate_prediction(prediction, taxonomy)
    predictions_payload = _jsonl_bytes(
        [prediction.model_dump(mode="json") for prediction in predictions]
    )
    adapter_hash = _sha256_json(experiment.b0)
    manifest = _formal_manifest(
        run_id=f"{dataset.dataset_version}-b0-test-{adapter_hash[:12]}",
        arm="B0",
        adapter="lexical_open_set_v1",
        adapter_config_sha256=adapter_hash,
        model="none",
        model_role="none",
        embedding_model=None,
        dataset=dataset,
        cases_path=cases_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        taxonomy=taxonomy,
        models=models,
        repository_root=repository_root,
        metadata={
            "prediction_sha256": _sha256_bytes(predictions_payload),
            "provider_calls": 0,
            "intentbench_version": __version__,
        },
    )
    status = _write_run_bundle(
        output_dir=output_dir,
        deterministic_payloads={PREDICTIONS_FILENAME: predictions_payload},
        manifest=manifest,
    )
    metrics = evaluate_run(cases, predictions, taxonomy)
    return {
        "status": status,
        "arm": "B0",
        "split": "test",
        "prediction_count": len(predictions),
        "hierarchical_exact_match": metrics["hierarchical_exact_match"],
        "decision_macro_f1": metrics["decision_macro_f1"],
        "provider_calls": 0,
        "output_dir": str(output_dir),
    }


def run_b1_test(
    *,
    client: EmbeddingClient,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    cases, taxonomy, dataset, experiment, models = _load_formal_inputs(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    dev_cases, _dev_dataset = bind_frozen_split(
        split=Split.DEV,
        cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
    )
    try:
        config = B1Config.model_validate(experiment.b1)
        if config.selected_thresholds is None or config.selected_prototype_case_ids is None:
            raise ValueError("frozen B1 selections are absent")
        thresholds = B1Thresholds(**config.selected_thresholds)
    except (ValidationError, ValueError) as exc:
        raise FormalRunError("frozen B1 selection contract is invalid") from exc
    embedding = models.embedding
    embedding_adapter_hash = _sha256_json(embedding.model_dump(mode="json"))
    adapter_hash = _sha256_json(
        {"embedding": embedding.model_dump(mode="json"), "b1": experiment.b1}
    )
    cache = EmbeddingCache(cache_dir)
    cache_hits = 0
    provider_calls = 0

    def fetch(text: str) -> tuple[tuple[float, ...], str, float]:
        nonlocal cache_hits, provider_calls
        result = cache.get_or_create(
            client=client,
            text=text,
            model=embedding.model,
            adapter_config_sha256=embedding_adapter_hash,
            task_type=embedding.task_type,
            output_dimensionality=embedding.output_dimensionality,
            expected_dimension=embedding.expected_dimension,
        )
        cache_hits += int(result.cache_hit)
        provider_calls += int(not result.cache_hit)
        return result.vector, result.raw_response_sha256, result.latency_ms

    activity_centroids = {
        intent.name: normalized_centroid(
            fetch(serialize_context(_prototype_context(example)))[0]
            for example in intent.canonical_examples
        )
        for intent in taxonomy.intents
    }
    dev_by_id = {case.id: case for case in dev_cases}
    prototype_ids = config.selected_prototype_case_ids
    assert prototype_ids is not None
    try:
        oos_centroid = normalized_centroid(
            fetch(serialize_context(dev_by_id[case_id].context))[0]
            for case_id in prototype_ids["oos"]
        )
        no_intent_centroid = normalized_centroid(
            fetch(serialize_context(dev_by_id[case_id].context))[0]
            for case_id in prototype_ids["no_intent"]
        )
    except KeyError as exc:
        raise FormalRunError("B1 prototype ID is absent from frozen dev") from exc

    predictions: list[Prediction] = []
    raw_hashes: list[str] = []
    for case in cases:
        vector, raw_hash, latency = fetch(serialize_context(case.context))
        raw_hashes.append(raw_hash)
        scores = score_embedding(
            vector,
            activity_centroids=activity_centroids,
            no_intent_centroid=no_intent_centroid,
            oos_centroid=oos_centroid,
        )
        prediction = _prediction_from_b1_scores(
            case=case,
            scores=scores,
            thresholds=thresholds,
            query_raw_response_sha256=raw_hash,
            query_latency_ms=latency,
        )
        validate_prediction(prediction, taxonomy)
        predictions.append(prediction)

    predictions_payload = _jsonl_bytes(
        [prediction.model_dump(mode="json") for prediction in predictions]
    )
    provenance = {
        "schema_version": 1,
        "artifact_kind": "b1_frozen_test_provenance",
        "dev_sha256": sha256_file(dev_cases_path),
        "selected_thresholds": config.selected_thresholds,
        "selected_prototype_case_ids": prototype_ids,
        "selection_recomputed_on_test": False,
    }
    provenance_payload = _json_bytes(provenance)
    manifest = _formal_manifest(
        run_id=f"{dataset.dataset_version}-b1-test-{adapter_hash[:12]}",
        arm="B1",
        adapter="google_embedding_open_set_v1",
        adapter_config_sha256=adapter_hash,
        model=f"{embedding.provider}/{embedding.model}",
        model_role=ModelRole.EMBEDDING,
        embedding_model=embedding.model,
        dataset=dataset,
        cases_path=cases_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        taxonomy=taxonomy,
        models=models,
        repository_root=repository_root,
        raw_response_sha256=_sha256_json(sorted(raw_hashes)),
        metadata={
            "prediction_sha256": _sha256_bytes(predictions_payload),
            "provenance_sha256": _sha256_bytes(provenance_payload),
            "selected_thresholds": config.selected_thresholds,
            "selected_prototype_case_ids": prototype_ids,
            "selection_recomputed_on_test": False,
            "embedding_cache_hits": cache_hits,
            "provider_calls": provider_calls,
            "intentbench_version": __version__,
        },
    )
    status = _write_run_bundle(
        output_dir=output_dir,
        deterministic_payloads={
            PREDICTIONS_FILENAME: predictions_payload,
            B1_TEST_PROVENANCE_FILENAME: provenance_payload,
        },
        manifest=manifest,
    )
    metrics = evaluate_run(cases, predictions, taxonomy)
    return {
        "status": status,
        "arm": "B1",
        "split": "test",
        "prediction_count": len(predictions),
        "hierarchical_exact_match": metrics["hierarchical_exact_match"],
        "decision_macro_f1": metrics["decision_macro_f1"],
        "embedding_cache_hits": cache_hits,
        "provider_calls": provider_calls,
        "output_dir": str(output_dir),
    }


def run_b2a_test(
    *,
    client: StructuredGenerationClient,
    cache_dir: Path,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    (
        cases,
        taxonomy,
        dataset,
        config,
        models,
        spec,
        few_shots,
        experiment,
    ) = _load_contract(
        split=Split.TEST,
        role="primary_decision",
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    _require_clean(repository_root)
    cache = GenerationCache(cache_dir)
    predictions: list[Prediction] = []
    calls: list[B2ACallRecord] = []
    cache_hits = 0
    provider_calls = 0
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
        prediction = normalize_cached_one_stage_prediction(
            case_id=case.id,
            cached=cached,
            taxonomy=taxonomy,
            spec=spec,
        )
        predictions.append(prediction)
        calls.append(
            B2ACallRecord(
                case_id=case.id,
                cache_key=cached.cache_key,
                call_contract_sha256=cached.call_contract_sha256,
                prompt_instance_sha256=cached.call_contract.prompt_instance_sha256,
                requested_model=spec.model,
                reported_model=cached.reported_model,
                prediction_status=prediction.status,
                latency_ms=cached.latency_ms,
                usage=prediction.usage,
                raw_response_sha256=prediction.raw_response_sha256,
                error_type=prediction.error_type,
                cache_origin=cached.origin,
                source_cache_key=cached.source_cache_key,
                prediction_sha256=_sha256_json(prediction.model_dump(mode="json")),
            )
        )
    predictions_payload = _jsonl_bytes(
        [prediction.model_dump(mode="json") for prediction in predictions]
    )
    calls_payload = _jsonl_bytes([call.model_dump(mode="json") for call in calls])
    cache_records = cache.records([call.cache_key for call in calls])
    cache_payload = _jsonl_bytes([record.model_dump(mode="json") for record in cache_records])
    provenance = B2ACacheProvenance(
        portable_cache_bundle_sha256=_sha256_bytes(cache_payload),
        source_predictions_sha256=_sha256_bytes(predictions_payload),
        entries=[
            B2ACacheProvenanceEntry(
                case_id=call.case_id,
                destination_cache_key=call.cache_key,
                origin=call.cache_origin,
                source_cache_key=call.source_cache_key,
                prediction_sha256=call.prediction_sha256,
                raw_response_sha256=call.raw_response_sha256,
            )
            for call in calls
        ],
    )
    provenance_payload = _json_bytes(provenance.model_dump(mode="json"))
    total_usage = _strict_prediction_usage(predictions)
    estimated_cost = estimate_cost(total_usage, config.pricing) if total_usage is not None else None
    prompt_digest = experiment.artifacts[config.b2a.prompt_artifact]
    adapter_hash = _sha256_json(
        {
            "arm": "B2a",
            "model": spec.model_dump(mode="json"),
            "b2a": config.b2a.model_dump(mode="json"),
            "prompt_sha256": prompt_digest.sha256,
            "few_shot_case_ids": config.shared_few_shot.case_ids,
        }
    )
    manifest = _formal_manifest(
        run_id=f"{dataset.dataset_version}-b2a-primary-test-{adapter_hash[:12]}",
        arm="B2a",
        adapter=spec.adapter,
        adapter_config_sha256=adapter_hash,
        model=f"{spec.provider}/{spec.model}",
        model_role=ModelRole.PRIMARY_DECISION,
        embedding_model=None,
        dataset=dataset,
        cases_path=cases_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        taxonomy=taxonomy,
        models=models,
        repository_root=repository_root,
        prompt_sha256=prompt_digest.sha256,
        raw_response_sha256=_sha256_json(
            [prediction.raw_response_sha256 for prediction in predictions]
        ),
        parameters=RunParameters(
            temperature=None,
            top_p=None,
            seed=None,
            max_calls=1,
            max_output_tokens_per_call=config.b2a.request_max_output_tokens,
            retry_policy=config.b2a.retry_policy,
            schema_repair_policy=config.b2a.schema_repair_policy,
        ),
        pricing_snapshot_date=config.pricing.snapshot_date,
        total_usage=total_usage,
        estimated_cost=estimated_cost,
        metadata={
            "prediction_sha256": _sha256_bytes(predictions_payload),
            "calls_sha256": _sha256_bytes(calls_payload),
            "cache_bundle_sha256": _sha256_bytes(cache_payload),
            "cache_provenance_sha256": _sha256_bytes(provenance_payload),
            "one_stage_cache_compatible": True,
            "one_stage_call_contract_version": 2,
            "shared_few_shot_case_ids": config.shared_few_shot.case_ids,
            "generation_cache_hits": cache_hits,
            "provider_calls": provider_calls,
            "usage_complete": total_usage is not None,
            "intentbench_version": __version__,
        },
    )
    status = _write_run_bundle(
        output_dir=output_dir,
        deterministic_payloads={
            PREDICTIONS_FILENAME: predictions_payload,
            B2A_CALLS_FILENAME: calls_payload,
            B2A_CACHE_BUNDLE_FILENAME: cache_payload,
            B2A_CACHE_PROVENANCE_FILENAME: provenance_payload,
        },
        manifest=manifest,
    )
    metrics = evaluate_run(cases, predictions, taxonomy)
    return {
        "status": status,
        "arm": "B2a",
        "split": "test",
        "prediction_count": len(predictions),
        "hierarchical_exact_match": metrics["hierarchical_exact_match"],
        "decision_macro_f1": metrics["decision_macro_f1"],
        "generation_cache_hits": cache_hits,
        "provider_calls": provider_calls,
        "estimated_cost_usd": estimated_cost,
        "output_dir": str(output_dir),
    }
