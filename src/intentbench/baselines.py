"""Dev-only B0/B1/B2a runners with unified Prediction and RunManifest artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from intentbench import __version__
from intentbench.adapters.base import EmbeddingClient, StructuredGenerationClient
from intentbench.b0 import B0Config, current_decision_text, predict_b0, serialize_context
from intentbench.b1 import (
    B1Config,
    B1Scores,
    B1Thresholds,
    normalized_centroid,
    route_scores,
    score_embedding,
    select_prototype_case_ids,
    select_thresholds,
    threshold_candidates,
)
from intentbench.b2a import (
    FewShotSelectionReceipt,
    LLMExperimentConfig,
    PromptSelectionReceipt,
    build_one_stage_request,
    estimate_cost,
    few_shot_payload_sha256,
    normalize_cached_one_stage_prediction,
    one_stage_adapter_config_sha256,
    prediction_response_schema,
    render_b2a_prompt,
    response_schema_sha256,
    validate_few_shot_cases,
    validate_few_shot_receipt,
    validate_prompt_selection_receipt,
)
from intentbench.embedding import EmbeddingCache
from intentbench.evaluator import evaluate_run
from intentbench.freeze import ArtifactDigest, DatasetFreezeManifest, ExperimentLock, sha256_file
from intentbench.generation_cache import GenerationCache, sha256_json
from intentbench.metrics import decision_macro_f1, hem
from intentbench.providers import ReadinessModels
from intentbench.reporting import EVALUATOR_VERSION, bind_frozen_dev
from intentbench.schemas import (
    Case,
    Context,
    Decision,
    Horizon,
    ModelRole,
    Prediction,
    PredictionStatus,
    RunManifest,
    RunParameters,
    Slots,
    StrictModel,
    Taxonomy,
    TokenUsage,
)
from intentbench.taxonomy import load_taxonomy, validate_prediction

PREDICTIONS_FILENAME = "predictions.jsonl"
MANIFEST_FILENAME = "run-manifest.json"
B1_SELECTION_FILENAME = "b1-selection.json"
B1_SCORES_FILENAME = "b1-scores.jsonl"
B2A_CALLS_FILENAME = "b2a-calls.jsonl"
B2A_CACHE_BUNDLE_FILENAME = "b2a-cache.jsonl"
B2A_CACHE_PROVENANCE_FILENAME = "b2a-cache-provenance.json"
HORIZON_FUTURE_PATTERN = re.compile(r"(?:哪天|晚点|以后|明天|下周|不是现在)")


class BaselineRunError(ValueError):
    """A dev baseline input or output violates its registered contract."""


class B1CaseScore(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str
    activity_scores: dict[str, float]
    activity_score: float
    second_activity_score: float
    no_intent_score: float
    oos_score: float
    actionability_score: float
    ranked_intents: list[str] = Field(min_length=2)
    query_raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_latency_ms: float = Field(ge=0)


class B1SelectionArtifact(StrictModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["b1_dev_threshold_selection"] = "b1_dev_threshold_selection"
    dataset_version: str
    taxonomy_version: str
    split: Literal["dev"] = "dev"
    dataset_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: dict[str, ArtifactDigest]
    input_serializer: Literal["context-json-v1"]
    embedding_provider: str
    embedding_model: str
    embedding_adapter: str
    embedding_task_type: str | None
    embedding_dimension: int = Field(ge=1)
    embedding_adapter_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    activity_prototype_example_counts: dict[str, int]
    rejection_prototype_case_ids: dict[str, list[str]]
    prototype_selection_algorithm: str
    threshold_candidate_count: int = Field(ge=1)
    selected_thresholds: dict[str, float]
    selected_dev_hem: float = Field(ge=0, le=1)
    selected_dev_decision_macro_f1: float = Field(ge=0, le=1)
    aggregate_raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_prototypes(self) -> B1SelectionArtifact:
        if set(self.rejection_prototype_case_ids) != {"oos", "no_intent"}:
            raise ValueError("B1 selection must record OOS and no-intent prototype IDs")
        for case_ids in self.rejection_prototype_case_ids.values():
            if not case_ids or len(case_ids) > 8 or len(case_ids) != len(set(case_ids)):
                raise ValueError("B1 rejection prototype IDs must contain 1-8 unique cases")
        return self


class B2ACallRecord(StrictModel):
    schema_version: Literal[2] = 2
    case_id: str
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_instance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_model: str
    reported_model: str | None
    prediction_status: PredictionStatus
    latency_ms: float = Field(ge=0)
    usage: TokenUsage | None
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_type: str | None
    cache_origin: Literal["provider_call", "validated_contract_migration"]
    source_cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    one_stage_cache_compatible: Literal[True] = True

    @model_validator(mode="after")
    def validate_call(self) -> B2ACallRecord:
        if self.prediction_status is PredictionStatus.SUCCESS:
            if (
                self.error_type is not None
                or self.usage is None
                or self.raw_response_sha256 is None
            ):
                raise ValueError("successful B2a call lacks usage/raw identity or carries error")
        elif not self.error_type:
            raise ValueError("failed B2a call requires error_type")
        if self.cache_origin == "provider_call" and self.source_cache_key is not None:
            raise ValueError("direct B2a calls cannot carry migration provenance")
        if self.cache_origin == "validated_contract_migration" and self.source_cache_key is None:
            raise ValueError("migrated B2a calls require a source cache key")
        return self


class B2ACacheProvenanceEntry(StrictModel):
    case_id: str
    destination_cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["provider_call", "validated_contract_migration"]
    source_cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class B2ACacheProvenance(StrictModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["b2a_normalized_cache_provenance"] = "b2a_normalized_cache_provenance"
    call_contract_version: Literal[2] = 2
    normalization_contract: Literal["normalize_cached_one_stage_prediction_v1"] = (
        "normalize_cached_one_stage_prediction_v1"
    )
    portable_cache_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: list[B2ACacheProvenanceEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> B2ACacheProvenance:
        case_ids = [entry.case_id for entry in self.entries]
        keys = [entry.destination_cache_key for entry in self.entries]
        if len(case_ids) != len(set(case_ids)) or len(keys) != len(set(keys)):
            raise ValueError("B2a cache provenance cases and keys must be unique")
        for entry in self.entries:
            if entry.origin == "provider_call" and entry.source_cache_key is not None:
                raise ValueError("direct calls cannot carry migration provenance")
            if entry.origin == "validated_contract_migration" and entry.source_cache_key is None:
                raise ValueError("migrated calls require a source cache key")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode()


def _jsonl_bytes(values: Sequence[object]) -> bytes:
    return ("\n".join(_canonical_json(value) for value in values) + "\n").encode()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative_path(path: Path, repository_root: Path) -> str:
    resolved_root = repository_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise BaselineRunError(f"source artifact is outside repository root: {path}")
    return resolved.relative_to(resolved_root).as_posix()


def _source_digest(path: Path, repository_root: Path) -> ArtifactDigest:
    return ArtifactDigest(
        path=_relative_path(path, repository_root),
        sha256=sha256_file(path),
    )


def _load_experiment(path: Path) -> tuple[ExperimentLock, dict[str, Any]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("experiment must be an object")
        return ExperimentLock.model_validate(payload), payload
    except (OSError, ValidationError, ValueError) as exc:
        raise BaselineRunError(f"invalid experiment config: {path}") from exc


def _bind_experiment(
    *,
    experiment_path: Path,
    dataset_manifest_path: Path,
    dataset: DatasetFreezeManifest,
    repository_root: Path,
) -> tuple[ExperimentLock, dict[str, Any]]:
    experiment, payload = _load_experiment(experiment_path)
    if experiment.dataset_version != dataset.dataset_version:
        raise BaselineRunError("experiment and frozen dev dataset versions differ")
    if experiment.taxonomy_version != dataset.taxonomy_version:
        raise BaselineRunError("experiment and frozen dev taxonomy versions differ")
    dataset_freeze_sha256 = sha256_file(dataset_manifest_path)
    if experiment.dataset_freeze_sha256 != dataset_freeze_sha256:
        raise BaselineRunError("experiment references a different dataset freeze manifest")
    for name, digest in experiment.artifacts.items():
        artifact = (repository_root.resolve() / digest.path).resolve()
        if not artifact.is_relative_to(repository_root.resolve()) or not artifact.is_file():
            raise BaselineRunError(f"experiment artifact is unavailable: {name}")
        if sha256_file(artifact) != digest.sha256:
            raise BaselineRunError(f"experiment artifact hash differs: {name}")
    return experiment, payload


def _git_identity(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselineRunError("cannot resolve git reproduction identity") from exc
    return commit, dirty


def _primary_model(models: ReadinessModels) -> str:
    return f"{models.primary_decision.provider}/{models.primary_decision.model}"


def _manifest(
    *,
    run_id: str,
    arm: Literal["B0", "B1", "B2a"],
    adapter: str,
    adapter_config_sha256: str,
    model: str,
    model_role: Literal["none", "embedding", "primary_decision"],
    embedding_model: str | None,
    dataset: DatasetFreezeManifest,
    cases_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    taxonomy: Taxonomy,
    models: ReadinessModels,
    repository_root: Path,
    raw_response_sha256: str | None,
    metadata: dict[str, Any],
    prompt_sha256: str | None = None,
    parameters: RunParameters | None = None,
    pricing_snapshot_date: str = "not_applicable",
    total_usage: TokenUsage | None = None,
    estimated_cost: float | None = None,
) -> RunManifest:
    commit, dirty = _git_identity(repository_root)
    dependency = _load_experiment(experiment_path)[0].artifacts.get("dependency_lock")
    if dependency is None:
        raise BaselineRunError("experiment lacks dependency_lock artifact")
    resolved_role: ModelRole | Literal["none"] = "none"
    if model_role == "embedding":
        resolved_role = ModelRole.EMBEDDING
    elif model_role == "primary_decision":
        resolved_role = ModelRole.PRIMARY_DECISION
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
        model_role=resolved_role,
        primary_decision_model=_primary_model(models),
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
            "split": "dev",
            "git_worktree_dirty": dirty,
            "gold_fields_available_to_adapter": False,
            **metadata,
        },
    )


def _write_run_bundle(
    *,
    output_dir: Path,
    deterministic_payloads: Mapping[str, bytes],
    manifest: RunManifest,
) -> Literal["created", "unchanged"]:
    payloads = {
        **deterministic_payloads,
        MANIFEST_FILENAME: _json_bytes(manifest.model_dump(mode="json")),
    }
    paths = {name: output_dir / name for name in payloads}
    existing = {name for name, path in paths.items() if path.exists()}
    if existing:
        if existing != set(paths):
            raise BaselineRunError(f"partial baseline outputs already exist: {sorted(existing)}")
        changed = [
            name
            for name, payload in deterministic_payloads.items()
            if paths[name].read_bytes() != payload
        ]
        if changed:
            raise BaselineRunError(f"baseline regeneration differs: {sorted(changed)}")
        try:
            existing_manifest = RunManifest.model_validate_json(
                paths[MANIFEST_FILENAME].read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise BaselineRunError("existing run manifest is invalid") from exc
        volatile_metadata = {
            "embedding_cache_hits",
            "generation_cache_hits",
            "provider_calls",
            "git_worktree_dirty",
        }
        existing_identity = existing_manifest.model_dump(mode="json", exclude={"started_at"})
        current_identity = manifest.model_dump(mode="json", exclude={"started_at"})
        for identity in (existing_identity, current_identity):
            metadata = identity["metadata"]
            assert isinstance(metadata, dict)
            for field in volatile_metadata:
                metadata.pop(field, None)
        if existing_identity != current_identity:
            raise BaselineRunError("existing run manifest differs from current registered inputs")
        return "unchanged"

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output_dir) as temporary:
            stage = Path(temporary)
            for name, payload in payloads.items():
                (stage / name).write_bytes(payload)
            for name, path in paths.items():
                os.replace(stage / name, path)
                created.append(path)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return "created"


def _load_bound_inputs(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    repository_root: Path,
) -> tuple[list[Case], Taxonomy, DatasetFreezeManifest, ExperimentLock, dict[str, Any]]:
    cases, dataset = bind_frozen_dev(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
    )
    experiment, experiment_payload = _bind_experiment(
        experiment_path=experiment_path,
        dataset_manifest_path=dataset_manifest_path,
        dataset=dataset,
        repository_root=repository_root,
    )
    return cases, load_taxonomy(taxonomy_path), dataset, experiment, experiment_payload


def run_b0_dev(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    cases, taxonomy, dataset, experiment, payload = _load_bound_inputs(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    try:
        config = B0Config.model_validate(experiment.b0)
        models = ReadinessModels.model_validate(experiment.models)
    except ValidationError as exc:
        raise BaselineRunError("experiment B0/model config is invalid") from exc
    predictions = [
        predict_b0(case_id=case.id, context=case.context, config=config) for case in cases
    ]
    for prediction in predictions:
        validate_prediction(prediction, taxonomy)
    predictions_payload = _jsonl_bytes(
        [prediction.model_dump(mode="json") for prediction in predictions]
    )
    adapter_config_sha256 = _sha256_json(payload["b0"])
    metrics = evaluate_run(cases, predictions, taxonomy)
    manifest = _manifest(
        run_id=f"{dataset.dataset_version}-b0-dev-{adapter_config_sha256[:12]}",
        arm="B0",
        adapter="lexical_open_set_v1",
        adapter_config_sha256=adapter_config_sha256,
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
        raw_response_sha256=None,
        metadata={
            "input_serializer": config.input_serializer,
            "ruleset_version": config.ruleset_version,
            "gate_order": config.gate_order,
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
    return {
        "status": status,
        "arm": "B0",
        "split": "dev",
        "prediction_count": len(predictions),
        "hierarchical_exact_match": metrics["hierarchical_exact_match"],
        "decision_macro_f1": metrics["decision_macro_f1"],
        "provider_calls": 0,
        "output_dir": str(output_dir),
    }


def _prototype_context(example: str) -> Context:
    return Context(state_summary=example, conversation=[], recent_activities=[])


def _prediction_from_b1_scores(
    *,
    case: Case,
    scores: B1Scores,
    thresholds: B1Thresholds,
    query_raw_response_sha256: str,
    query_latency_ms: float,
) -> Prediction:
    decision, target = route_scores(scores, thresholds)
    candidates = list(scores.ranked_intents[:2]) if decision is Decision.AMBIGUOUS else []
    horizon = (
        Horizon.LATER
        if HORIZON_FUTURE_PATTERN.search(current_decision_text(case.context))
        else Horizon.NOW
    )
    return Prediction(
        case_id=case.id,
        decision=decision,
        predicted_intent=target,
        candidate_intents=candidates,
        slots=Slots(desired_experience=None, object=None, horizon=horizon),
        reason_short=(
            f"embedding门控:a={scores.actionability_score:.4f},"
            f"oos={scores.oos_score:.4f},top={scores.activity_score:.4f},"
            f"margin={scores.activity_score - scores.second_activity_score:.4f}"
        ),
        latency_ms=query_latency_ms,
        raw_response_sha256=query_raw_response_sha256,
    )


def run_b1_dev(
    *,
    client: EmbeddingClient,
    cache_dir: Path,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    cases, taxonomy, dataset, experiment, payload = _load_bound_inputs(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    try:
        config = B1Config.model_validate(experiment.b1)
        models = ReadinessModels.model_validate(experiment.models)
    except ValidationError as exc:
        raise BaselineRunError("experiment B1/model config is invalid") from exc
    embedding = models.embedding
    embedding_adapter_config_sha256 = _sha256_json(embedding.model_dump(mode="json"))
    b1_adapter_config_sha256 = _sha256_json(
        {"embedding": embedding.model_dump(mode="json"), "b1": payload["b1"]}
    )
    selection = config.prototype_selection
    oos_ids = select_prototype_case_ids(
        cases,
        decision=Decision.OOS,
        maximum=selection.oos_maximum,
        strata=selection.oos_strata,
    )
    no_intent_ids = select_prototype_case_ids(
        cases,
        decision=Decision.NO_INTENT,
        maximum=selection.no_intent_maximum,
        strata=selection.no_intent_strata,
    )
    selected_prototype_ids = {"oos": list(oos_ids), "no_intent": list(no_intent_ids)}
    if (
        config.selected_prototype_case_ids is not None
        and config.selected_prototype_case_ids != selected_prototype_ids
    ):
        raise BaselineRunError("registered B1 prototype IDs differ from deterministic selection")
    case_by_id = {case.id: case for case in cases}
    cache = EmbeddingCache(cache_dir)
    cache_hits = 0
    provider_calls = 0
    raw_response_hashes: list[str] = []

    def fetch(text: str) -> tuple[tuple[float, ...], str, float]:
        nonlocal cache_hits, provider_calls
        result = cache.get_or_create(
            client=client,
            text=text,
            model=embedding.model,
            adapter_config_sha256=embedding_adapter_config_sha256,
            task_type=embedding.task_type,
            output_dimensionality=embedding.output_dimensionality,
            expected_dimension=embedding.expected_dimension,
        )
        if result.cache_hit:
            cache_hits += 1
        else:
            provider_calls += 1
        raw_response_hashes.append(result.raw_response_sha256)
        return result.vector, result.raw_response_sha256, result.latency_ms

    activity_centroids: dict[str, tuple[float, ...]] = {}
    activity_counts: dict[str, int] = {}
    for intent in taxonomy.intents:
        vectors = [
            fetch(serialize_context(_prototype_context(example)))[0]
            for example in intent.canonical_examples
        ]
        activity_centroids[intent.name] = normalized_centroid(vectors)
        activity_counts[intent.name] = len(vectors)
    oos_centroid = normalized_centroid(
        fetch(serialize_context(case_by_id[case_id].context))[0] for case_id in oos_ids
    )
    no_intent_centroid = normalized_centroid(
        fetch(serialize_context(case_by_id[case_id].context))[0] for case_id in no_intent_ids
    )

    scores_by_id: dict[str, B1Scores] = {}
    query_identity: dict[str, tuple[str, float]] = {}
    score_records: list[B1CaseScore] = []
    for case in cases:
        vector, raw_response_sha256, latency_ms = fetch(serialize_context(case.context))
        scores = score_embedding(
            vector,
            activity_centroids=activity_centroids,
            no_intent_centroid=no_intent_centroid,
            oos_centroid=oos_centroid,
        )
        scores_by_id[case.id] = scores
        query_identity[case.id] = (raw_response_sha256, latency_ms)
        score_records.append(
            B1CaseScore(
                case_id=case.id,
                activity_scores=dict(scores.activity_scores),
                activity_score=scores.activity_score,
                second_activity_score=scores.second_activity_score,
                no_intent_score=scores.no_intent_score,
                oos_score=scores.oos_score,
                actionability_score=scores.actionability_score,
                ranked_intents=list(scores.ranked_intents),
                query_raw_response_sha256=raw_response_sha256,
                query_latency_ms=latency_ms,
            )
        )

    def predictions_for(thresholds: B1Thresholds) -> list[Prediction]:
        return [
            _prediction_from_b1_scores(
                case=case,
                scores=scores_by_id[case.id],
                thresholds=thresholds,
                query_raw_response_sha256=query_identity[case.id][0],
                query_latency_ms=query_identity[case.id][1],
            )
            for case in cases
        ]

    def score_candidate(thresholds: B1Thresholds) -> tuple[float, float]:
        indexed = {prediction.case_id: prediction for prediction in predictions_for(thresholds)}
        return hem(cases, indexed), decision_macro_f1(cases, indexed)

    candidates = threshold_candidates(config.threshold_grid)
    selected_thresholds = select_thresholds(candidates, score_candidate)
    if (
        config.selected_thresholds is not None
        and config.selected_thresholds != selected_thresholds.as_dict()
    ):
        raise BaselineRunError("registered B1 thresholds differ from deterministic dev selection")
    predictions = predictions_for(selected_thresholds)
    for prediction in predictions:
        validate_prediction(prediction, taxonomy)
    indexed = {prediction.case_id: prediction for prediction in predictions}
    selected_hem = hem(cases, indexed)
    selected_macro_f1 = decision_macro_f1(cases, indexed)
    aggregate_raw_response_sha256 = _sha256_json(sorted(raw_response_hashes))
    selection_artifact = B1SelectionArtifact(
        dataset_version=dataset.dataset_version,
        taxonomy_version=taxonomy.taxonomy_version,
        dataset_freeze_sha256=sha256_file(dataset_manifest_path),
        experiment_config_sha256=sha256_file(experiment_path),
        source_artifacts={
            "dev": _source_digest(cases_path, repository_root),
            "taxonomy": _source_digest(taxonomy_path, repository_root),
        },
        input_serializer=config.input_serializer,
        embedding_provider=embedding.provider,
        embedding_model=embedding.model,
        embedding_adapter=embedding.adapter,
        embedding_task_type=embedding.task_type,
        embedding_dimension=embedding.expected_dimension,
        embedding_adapter_config_sha256=embedding_adapter_config_sha256,
        activity_prototype_example_counts=activity_counts,
        rejection_prototype_case_ids=selected_prototype_ids,
        prototype_selection_algorithm=selection.algorithm,
        threshold_candidate_count=len(candidates),
        selected_thresholds=selected_thresholds.as_dict(),
        selected_dev_hem=selected_hem,
        selected_dev_decision_macro_f1=selected_macro_f1,
        aggregate_raw_response_sha256=aggregate_raw_response_sha256,
    )
    predictions_payload = _jsonl_bytes(
        [prediction.model_dump(mode="json") for prediction in predictions]
    )
    selection_payload = _json_bytes(selection_artifact.model_dump(mode="json"))
    scores_payload = _jsonl_bytes([record.model_dump(mode="json") for record in score_records])
    manifest = _manifest(
        run_id=f"{dataset.dataset_version}-b1-dev-{b1_adapter_config_sha256[:12]}",
        arm="B1",
        adapter="google_embedding_open_set_v1",
        adapter_config_sha256=b1_adapter_config_sha256,
        model=f"{embedding.provider}/{embedding.model}",
        model_role="embedding",
        embedding_model=embedding.model,
        dataset=dataset,
        cases_path=cases_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        taxonomy=taxonomy,
        models=models,
        repository_root=repository_root,
        raw_response_sha256=aggregate_raw_response_sha256,
        metadata={
            "input_serializer": config.input_serializer,
            "prototype_selection_algorithm": selection.algorithm,
            "prototype_case_ids": selected_prototype_ids,
            "selected_thresholds": selected_thresholds.as_dict(),
            "prediction_sha256": _sha256_bytes(predictions_payload),
            "selection_sha256": _sha256_bytes(selection_payload),
            "scores_sha256": _sha256_bytes(scores_payload),
            "embedding_cache_hits": cache_hits,
            "provider_calls": provider_calls,
            "intentbench_version": __version__,
        },
    )
    status = _write_run_bundle(
        output_dir=output_dir,
        deterministic_payloads={
            PREDICTIONS_FILENAME: predictions_payload,
            B1_SELECTION_FILENAME: selection_payload,
            B1_SCORES_FILENAME: scores_payload,
        },
        manifest=manifest,
    )
    return {
        "status": status,
        "arm": "B1",
        "split": "dev",
        "prediction_count": len(predictions),
        "hierarchical_exact_match": selected_hem,
        "decision_macro_f1": selected_macro_f1,
        "selected_thresholds": selected_thresholds.as_dict(),
        "prototype_case_ids": selected_prototype_ids,
        "embedding_cache_hits": cache_hits,
        "provider_calls": provider_calls,
        "output_dir": str(output_dir),
    }


def _aggregate_usage(predictions: Sequence[Prediction]) -> TokenUsage:
    usages = [prediction.usage for prediction in predictions if prediction.usage is not None]
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        retry_input_tokens=sum(usage.retry_input_tokens for usage in usages),
        retry_output_tokens=sum(usage.retry_output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("percentile needs values and a quantile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def run_b2a_dev(
    *,
    client: StructuredGenerationClient,
    cache_dir: Path,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    cases, taxonomy, dataset, experiment, _payload = _load_bound_inputs(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        repository_root=repository_root,
    )
    try:
        config = LLMExperimentConfig.model_validate(experiment.llm)
        models = ReadinessModels.model_validate(experiment.models)
    except ValidationError as exc:
        raise BaselineRunError("experiment LLM/model config is invalid") from exc
    spec = models.primary_decision
    if config.pricing.model != spec.model or config.pricing.provider != spec.provider:
        raise BaselineRunError("B2a pricing identity differs from the primary model")
    if config.b2a.request_max_output_tokens != spec.request_max_output_tokens:
        raise BaselineRunError("B2a output budget differs from the primary model contract")

    prompt_digest = experiment.artifacts.get(config.b2a.prompt_artifact)
    if prompt_digest is None:
        raise BaselineRunError("B2a prompt artifact is not registered in the experiment")
    prompt_path = (repository_root.resolve() / prompt_digest.path).resolve()
    template = prompt_path.read_text(encoding="utf-8")
    prompt_receipt_digest = experiment.artifacts.get(config.b2a.prompt_selection_receipt_artifact)
    if prompt_receipt_digest is None:
        raise BaselineRunError("Prompt selection receipt is not registered")
    prompt_receipt_path = (repository_root.resolve() / prompt_receipt_digest.path).resolve()
    try:
        prompt_receipt = PromptSelectionReceipt.model_validate(
            yaml.safe_load(prompt_receipt_path.read_text(encoding="utf-8"))
        )
        validate_prompt_selection_receipt(
            receipt=prompt_receipt,
            config=config.b2a,
            dataset_version=dataset.dataset_version,
            dev_sha256=sha256_file(cases_path),
            provider=models.primary_decision.provider,
            model=models.primary_decision.model,
            prompt_sha256=prompt_digest.sha256,
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise BaselineRunError("Prompt selection receipt is invalid") from exc
    few_shots = validate_few_shot_cases(cases, config.shared_few_shot)
    receipt_digest = experiment.artifacts.get(config.shared_few_shot.selection_receipt_artifact)
    if receipt_digest is None:
        raise BaselineRunError("few-shot selection receipt is not registered")
    receipt_path = (repository_root.resolve() / receipt_digest.path).resolve()
    try:
        receipt = FewShotSelectionReceipt.model_validate(
            yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
        )
        validate_few_shot_receipt(
            receipt=receipt,
            config=config.shared_few_shot,
            selected_cases=few_shots,
            dataset_version=dataset.dataset_version,
            dev_sha256=sha256_file(cases_path),
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise BaselineRunError("few-shot selection receipt is invalid") from exc
    few_shot_ids = [case.id for case in few_shots]
    few_shot_sha256 = few_shot_payload_sha256(few_shots)
    response_schema = prediction_response_schema(taxonomy)
    schema_sha256 = response_schema_sha256(taxonomy)
    taxonomy_sha256 = sha256_file(taxonomy_path)
    generation_parameters = dict(spec.generation_parameters)
    generation_parameters_sha256 = sha256_json(generation_parameters)
    adapter_config_sha256 = one_stage_adapter_config_sha256(
        config=config,
        spec=spec,
        prompt_sha256=prompt_digest.sha256,
        few_shot_sha256=few_shot_sha256,
        schema_sha256=schema_sha256,
    )
    cache = GenerationCache(cache_dir)
    predictions: list[Prediction] = []
    call_records: list[B2ACallRecord] = []
    provider_calls = 0
    cache_hits = 0

    for case in cases:
        context_json = serialize_context(case.context)
        prompt = render_b2a_prompt(
            template=template,
            taxonomy=taxonomy,
            few_shots=few_shots,
            context_json=context_json,
        )
        request = build_one_stage_request(
            context_json=context_json,
            prompt=prompt,
            taxonomy_sha256=taxonomy_sha256,
            prompt_template_sha256=prompt_digest.sha256,
            few_shot_sha256=few_shot_sha256,
            response_schema=response_schema,
            spec=spec,
            max_output_tokens=config.b2a.request_max_output_tokens,
        )
        cached = cache.get_or_create(
            client=client,
            request=request,
        )
        if cached.cache_hit:
            cache_hits += 1
        else:
            provider_calls += 1

        prediction = normalize_cached_one_stage_prediction(
            case_id=case.id,
            cached=cached,
            taxonomy=taxonomy,
            spec=spec,
        )
        predictions.append(prediction)
        call_records.append(
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

    metrics = evaluate_run(cases, predictions, taxonomy)
    indexed = {prediction.case_id: prediction for prediction in predictions}
    few_shot_id_set = set(few_shot_ids)
    case_id_held_out_cases = [case for case in cases if case.id not in few_shot_id_set]
    few_shot_clusters = {case.bootstrap_cluster_id for case in few_shots}
    cluster_held_out_cases = [
        case for case in cases if case.bootstrap_cluster_id not in few_shot_clusters
    ]
    case_id_held_out_hem = hem(case_id_held_out_cases, indexed)
    case_id_held_out_macro_f1 = decision_macro_f1(case_id_held_out_cases, indexed)
    cluster_held_out_hem = hem(cluster_held_out_cases, indexed)
    cluster_held_out_macro_f1 = decision_macro_f1(cluster_held_out_cases, indexed)
    total_usage = _aggregate_usage(predictions)
    estimated_cost = estimate_cost(total_usage, config.pricing)
    latencies = [record.latency_ms for record in call_records]
    aggregate_raw_response_sha256 = _sha256_json(
        [
            record.raw_response_sha256 or f"{record.call_contract_sha256}:{record.error_type}"
            for record in call_records
        ]
    )
    predictions_payload = _jsonl_bytes(
        [prediction.model_dump(mode="json") for prediction in predictions]
    )
    calls_payload = _jsonl_bytes([record.model_dump(mode="json") for record in call_records])
    cache_records = cache.records([record.cache_key for record in call_records])
    cache_bundle_payload = _jsonl_bytes(
        [record.model_dump(mode="json") for record in cache_records]
    )
    cache_provenance = B2ACacheProvenance(
        portable_cache_bundle_sha256=_sha256_bytes(cache_bundle_payload),
        source_predictions_sha256=_sha256_bytes(predictions_payload),
        entries=[
            B2ACacheProvenanceEntry(
                case_id=record.case_id,
                destination_cache_key=record.cache_key,
                origin=record.cache_origin,
                source_cache_key=record.source_cache_key,
                prediction_sha256=record.prediction_sha256,
                raw_response_sha256=record.raw_response_sha256,
            )
            for record in call_records
        ],
    )
    cache_provenance_payload = _json_bytes(cache_provenance.model_dump(mode="json"))
    implementation_paths = [
        Path("src/intentbench/adapters/base.py"),
        Path("src/intentbench/adapters/google.py"),
        Path("src/intentbench/b2a.py"),
        Path("src/intentbench/baselines.py"),
        Path("src/intentbench/generation_cache.py"),
        Path("src/intentbench/providers.py"),
        Path("src/intentbench/schemas.py"),
    ]
    manifest = _manifest(
        run_id=f"{dataset.dataset_version}-b2a-primary-dev-{adapter_config_sha256[:12]}",
        arm="B2a",
        adapter=spec.adapter,
        adapter_config_sha256=adapter_config_sha256,
        model=f"{spec.provider}/{spec.model}",
        model_role="primary_decision",
        embedding_model=None,
        dataset=dataset,
        cases_path=cases_path,
        dataset_manifest_path=dataset_manifest_path,
        experiment_path=experiment_path,
        taxonomy=taxonomy,
        models=models,
        repository_root=repository_root,
        raw_response_sha256=aggregate_raw_response_sha256,
        prompt_sha256=prompt_digest.sha256,
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
            "input_serializer": config.input_serializer,
            "prompt_renderer": config.prompt_renderer,
            "prompt_artifact": prompt_digest.model_dump(mode="json"),
            "prompt_selection_receipt": prompt_receipt_digest.model_dump(mode="json"),
            "few_shot_selection_receipt": receipt_digest.model_dump(mode="json"),
            "few_shot_case_ids": few_shot_ids,
            "few_shot_payload_sha256": few_shot_sha256,
            "few_shot_overlap_count": len(few_shot_ids),
            "few_shot_evidence_quote_in_prompt": False,
            "few_shot_review_fields_in_prompt": False,
            "few_shot_supervision_fields": ["decision", "target_intent", "slots"],
            "response_schema_sha256": schema_sha256,
            "generation_parameters_sha256": generation_parameters_sha256,
            "prediction_sha256": _sha256_bytes(predictions_payload),
            "calls_sha256": _sha256_bytes(calls_payload),
            "cache_bundle_sha256": _sha256_bytes(cache_bundle_payload),
            "cache_provenance_sha256": _sha256_bytes(cache_provenance_payload),
            "one_stage_cache_compatible": True,
            "one_stage_call_contract_version": 2,
            "generation_cache_hits": cache_hits,
            "provider_calls": provider_calls,
            "latency_p50_ms": _percentile(latencies, 0.5),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "case_id_held_out_case_count": len(case_id_held_out_cases),
            "case_id_held_out_hem": case_id_held_out_hem,
            "case_id_held_out_decision_macro_f1": case_id_held_out_macro_f1,
            "cluster_held_out_case_count": len(cluster_held_out_cases),
            "cluster_held_out_hem": cluster_held_out_hem,
            "cluster_held_out_decision_macro_f1": cluster_held_out_macro_f1,
            "implementation_artifacts": {
                path.as_posix(): sha256_file(repository_root / path)
                for path in implementation_paths
            },
            "pricing": config.pricing.model_dump(mode="json"),
            "intentbench_version": __version__,
        },
    )
    status = _write_run_bundle(
        output_dir=output_dir,
        deterministic_payloads={
            PREDICTIONS_FILENAME: predictions_payload,
            B2A_CALLS_FILENAME: calls_payload,
            B2A_CACHE_BUNDLE_FILENAME: cache_bundle_payload,
            B2A_CACHE_PROVENANCE_FILENAME: cache_provenance_payload,
        },
        manifest=manifest,
    )
    return {
        "status": status,
        "arm": "B2a",
        "split": "dev",
        "prediction_count": len(predictions),
        "hierarchical_exact_match": metrics["hierarchical_exact_match"],
        "decision_macro_f1": metrics["decision_macro_f1"],
        "case_id_held_out_case_count": len(case_id_held_out_cases),
        "case_id_held_out_hem": case_id_held_out_hem,
        "case_id_held_out_decision_macro_f1": case_id_held_out_macro_f1,
        "cluster_held_out_case_count": len(cluster_held_out_cases),
        "cluster_held_out_hem": cluster_held_out_hem,
        "cluster_held_out_decision_macro_f1": cluster_held_out_macro_f1,
        "few_shot_case_ids": few_shot_ids,
        "total_input_tokens": total_usage.input_tokens,
        "total_output_tokens": total_usage.output_tokens,
        "estimated_cost_usd": estimated_cost,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "generation_cache_hits": cache_hits,
        "provider_calls": provider_calls,
        "output_dir": str(output_dir),
    }
