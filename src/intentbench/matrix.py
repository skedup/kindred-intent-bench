"""IE3 seven-run matrix and per-model B2b/B3 budget-contract validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from intentbench.b2 import IE3LLMExperimentConfig
from intentbench.baselines import (
    B2A_CACHE_BUNDLE_FILENAME,
    B2A_CACHE_PROVENANCE_FILENAME,
    B2A_CALLS_FILENAME,
    B2ACacheProvenance,
    B2ACallRecord,
)
from intentbench.evaluator import evaluate_run, index_predictions
from intentbench.freeze import ArtifactDigest, ExperimentLock, sha256_file, verify_test_guard
from intentbench.providers import LLMModelSpec, ReadinessModels
from intentbench.reporting import EVALUATOR_VERSION, bind_frozen_split, load_predictions
from intentbench.schemas import (
    ModelRole,
    Prediction,
    PredictionStatus,
    RunManifest,
    Split,
    StrictModel,
    TokenUsage,
)
from intentbench.taxonomy import load_taxonomy, validate_prediction
from intentbench.two_stage import (
    CALL_1_PROVENANCE_FILENAME,
    CALLS_FILENAME,
    FORMED_INTENTIONS_FILENAME,
    Call1Provenance,
    LLMCallRecord,
)

EXPECTED_LLM_CELLS: tuple[tuple[ModelRole, Literal["B2a", "B2b", "B3"]], ...] = (
    (ModelRole.PRIMARY_DECISION, "B2a"),
    (ModelRole.PRIMARY_DECISION, "B2b"),
    (ModelRole.PRIMARY_DECISION, "B3"),
    (ModelRole.WEAK_DECISION, "B2b"),
    (ModelRole.WEAK_DECISION, "B3"),
    (ModelRole.CROSS_PROVIDER_REFERENCE, "B2b"),
    (ModelRole.CROSS_PROVIDER_REFERENCE, "B3"),
)


class MatrixValidationError(ValueError):
    """A formal run is absent or violates the preregistered matrix."""


class MatrixCell(StrictModel):
    role: ModelRole
    arm: Literal["B2a", "B2b", "B3"]
    run_directory: str
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_count: int = Field(ge=1)
    success_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    hierarchical_exact_match: float = Field(ge=0, le=1)
    decision_macro_f1: float = Field(ge=0, le=1)
    call_count: int = Field(ge=1)
    usage_complete: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)


class ModelBudgetComparison(StrictModel):
    role: ModelRole
    b2b_total_tokens: int = Field(ge=0)
    b3_total_tokens: int = Field(ge=0)
    absolute_difference_rate: float = Field(ge=0)
    maximum_allowed: float = Field(ge=0)
    usage_complete: bool
    status: Literal["valid", "budget-confounded", "usage-incomplete"]

    @model_validator(mode="after")
    def validate_status(self) -> ModelBudgetComparison:
        expected = (
            "usage-incomplete"
            if not self.usage_complete
            else (
                "budget-confounded"
                if self.absolute_difference_rate > self.maximum_allowed
                else "valid"
            )
        )
        if self.status != expected:
            raise ValueError("budget status disagrees with usage and threshold")
        return self


class SevenRunMatrixArtifact(StrictModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["ie3_seven_run_matrix"] = "ie3_seven_run_matrix"
    dataset_version: str
    taxonomy_version: str
    split: Literal["test"] = "test"
    dataset_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: dict[str, ArtifactDigest]
    cells: list[MatrixCell]
    budget_comparisons: list[ModelBudgetComparison]
    matrix_complete: Literal[True] = True

    @model_validator(mode="after")
    def validate_matrix(self) -> SevenRunMatrixArtifact:
        actual = [(cell.role, cell.arm) for cell in self.cells]
        if actual != list(EXPECTED_LLM_CELLS):
            raise ValueError("seven-run cells differ from preregistered order")
        if {item.role for item in self.budget_comparisons} != {
            ModelRole.PRIMARY_DECISION,
            ModelRole.WEAK_DECISION,
            ModelRole.CROSS_PROVIDER_REFERENCE,
        }:
            raise ValueError("budget comparisons must cover all three LLM roles")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode()


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root = root.resolve()
    if not resolved.is_relative_to(root):
        raise MatrixValidationError(f"matrix source is outside repository: {path}")
    return resolved.relative_to(root).as_posix()


def _source(path: Path, root: Path) -> ArtifactDigest:
    return ArtifactDigest(path=_relative(path, root), sha256=sha256_file(path))


def _sum_call_usage(calls: Sequence[LLMCallRecord | B2ACallRecord]) -> tuple[bool, TokenUsage]:
    complete = all(call.usage is not None for call in calls)
    usages = [call.usage for call in calls if call.usage is not None]
    return complete, TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        retry_input_tokens=sum(usage.retry_input_tokens for usage in usages),
        retry_output_tokens=sum(usage.retry_output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )


def budget_comparison(
    *,
    role: ModelRole,
    b2b_total_tokens: int,
    b3_total_tokens: int,
    maximum_allowed: float,
    usage_complete: bool,
) -> ModelBudgetComparison:
    difference = abs(b3_total_tokens - b2b_total_tokens) / max(b2b_total_tokens, 1)
    status: Literal["valid", "budget-confounded", "usage-incomplete"]
    if not usage_complete:
        status = "usage-incomplete"
    elif difference > maximum_allowed:
        status = "budget-confounded"
    else:
        status = "valid"
    return ModelBudgetComparison(
        role=role,
        b2b_total_tokens=b2b_total_tokens,
        b3_total_tokens=b3_total_tokens,
        absolute_difference_rate=difference,
        maximum_allowed=maximum_allowed,
        usage_complete=usage_complete,
        status=status,
    )


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    try:
        return [model.model_validate_json(line) for line in path.read_text().splitlines()]
    except (OSError, ValidationError, ValueError) as exc:
        raise MatrixValidationError(f"invalid matrix call artifact: {path}") from exc


def _model_spec(models: ReadinessModels, role: ModelRole) -> LLMModelSpec:
    return {
        ModelRole.PRIMARY_DECISION: models.primary_decision,
        ModelRole.WEAK_DECISION: models.weak_decision,
        ModelRole.CROSS_PROVIDER_REFERENCE: models.cross_provider_reference,
    }[role]


def _validate_two_call_case_contract(
    *, calls: Sequence[LLMCallRecord], predictions: Mapping[str, Prediction], gold_ids: set[str]
) -> None:
    by_case: dict[str, list[LLMCallRecord]] = {case_id: [] for case_id in gold_ids}
    for call in calls:
        if call.case_id not in by_case:
            raise MatrixValidationError("two-call artifact contains an extra case")
        by_case[call.case_id].append(call)
    for case_id, case_calls in by_case.items():
        if not case_calls or [call.call_index for call in case_calls] not in ([1], [1, 2]):
            raise MatrixValidationError(f"{case_id}: invalid two-call sequence")
        if any(
            call.usage is not None
            and (call.usage.retry_input_tokens or call.usage.retry_output_tokens)
            for call in case_calls
        ):
            raise MatrixValidationError(f"{case_id}: retry usage violates IE3 contract")
        prediction = predictions[case_id]
        if prediction.status is PredictionStatus.SUCCESS and len(case_calls) != 2:
            raise MatrixValidationError(f"{case_id}: successful prediction lacks two calls")
        if all(call.usage is not None for call in case_calls) and prediction.usage is not None:
            expected = _sum_call_usage(case_calls)[1]
            if prediction.usage != expected:
                raise MatrixValidationError(f"{case_id}: Prediction usage differs from call sum")


def _expected_adapter_identity(
    *,
    role: ModelRole,
    arm: Literal["B2a", "B2b", "B3"],
    spec: LLMModelSpec,
    config: IE3LLMExperimentConfig,
    experiment: ExperimentLock,
) -> tuple[str, str]:
    if arm == "B2a":
        prompt_sha256 = experiment.artifacts[config.b2a.prompt_artifact].sha256
        return (
            _sha256_json(
                {
                    "arm": "B2a",
                    "model": spec.model_dump(mode="json"),
                    "b2a": config.b2a.model_dump(mode="json"),
                    "prompt_sha256": prompt_sha256,
                    "few_shot_case_ids": config.shared_few_shot.case_ids,
                }
            ),
            prompt_sha256,
        )
    prompt_hashes = (
        {
            "call_1": experiment.artifacts[config.b2a.prompt_artifact].sha256,
            "call_2": experiment.artifacts[config.b2b.verifier_prompt_artifact].sha256,
        }
        if arm == "B2b"
        else {
            "stage_a": experiment.artifacts[config.b3.stage_a_prompt_artifact].sha256,
            "stage_b": experiment.artifacts[config.b3.stage_b_prompt_artifact].sha256,
        }
    )
    return (
        _sha256_json(
            {
                "arm": arm,
                "role": role.value,
                "model": spec.model_dump(mode="json"),
                "arm_config": getattr(config, arm.lower()).model_dump(mode="json"),
                "prompt_hashes": prompt_hashes,
                "shared_few_shot_case_ids": config.shared_few_shot.case_ids,
            }
        ),
        _sha256_json(prompt_hashes),
    )


def _validate_manifest_contract(
    *,
    manifest: RunManifest,
    role: ModelRole,
    arm: Literal["B2a", "B2b", "B3"],
    cases_path: Path,
    dataset_version: str,
    taxonomy_version: str,
    dataset_freeze_sha256: str,
    experiment_lock_sha256: str,
    models: ReadinessModels,
    config: IE3LLMExperimentConfig,
    experiment: ExperimentLock,
) -> None:
    spec = _model_spec(models, role)
    expected_adapter_hash, expected_prompt_hash = _expected_adapter_identity(
        role=role,
        arm=arm,
        spec=spec,
        config=config,
        experiment=experiment,
    )
    dependency = experiment.artifacts.get("dependency_lock")
    if dependency is None:
        raise MatrixValidationError("frozen experiment lacks dependency lock")
    expected_calls = 1 if arm == "B2a" else 2
    if (
        manifest.metadata.get("arm") != arm
        or manifest.metadata.get("split") != "test"
        or manifest.metadata.get("git_worktree_dirty") is not False
        or manifest.metadata.get("gold_fields_available_to_adapter") is not False
        or manifest.dataset_version != dataset_version
        or manifest.dataset_sha256 != sha256_file(cases_path)
        or manifest.taxonomy_version != taxonomy_version
        or manifest.model_role is not role
        or manifest.model != f"{spec.provider}/{spec.model}"
        or manifest.adapter != spec.adapter
        or manifest.adapter_config_sha256 != expected_adapter_hash
        or manifest.prompt_sha256 != expected_prompt_hash
        or manifest.dataset_freeze_sha256 != dataset_freeze_sha256
        or manifest.experiment_lock_sha256 != experiment_lock_sha256
        or manifest.primary_decision_model
        != f"{models.primary_decision.provider}/{models.primary_decision.model}"
        or manifest.evaluator_version != EVALUATOR_VERSION
        or manifest.dependency_lock_sha256 != dependency.sha256
        or manifest.parameters.max_calls != expected_calls
        or manifest.parameters.max_output_tokens_per_call != config.b2a.request_max_output_tokens
        or manifest.parameters.retry_policy != "none"
        or manifest.parameters.schema_repair_policy != "reject"
    ):
        raise MatrixValidationError(f"run manifest identity mismatch: {role.value}/{arm}")


def _validate_manifest_file_hash(
    *, manifest: RunManifest, metadata_field: str, path: Path, label: str
) -> None:
    if manifest.metadata.get(metadata_field) != sha256_file(path):
        raise MatrixValidationError(f"manifest {label} hash differs from output")


def _validate_primary_call1_reuse(*, runs_root: Path) -> None:
    b2a_calls = _read_jsonl(
        runs_root / ModelRole.PRIMARY_DECISION.value / "b2a" / B2A_CALLS_FILENAME,
        B2ACallRecord,
    )
    b2b_calls = _read_jsonl(
        runs_root / ModelRole.PRIMARY_DECISION.value / "b2b" / CALLS_FILENAME,
        LLMCallRecord,
    )
    first_calls = [call for call in b2b_calls if call.call_index == 1]
    if len(b2a_calls) != len(first_calls):
        raise MatrixValidationError("primary B2b call-1 universe differs from B2a")
    for one_stage, verifier_source in zip(b2a_calls, first_calls, strict=True):
        if (
            one_stage.case_id != verifier_source.case_id
            or one_stage.cache_key != verifier_source.cache_key
            or one_stage.call_contract_sha256 != verifier_source.call_contract_sha256
            or one_stage.requested_model != verifier_source.requested_model
            or one_stage.reported_model != verifier_source.reported_model
            or one_stage.prediction_status is not verifier_source.status
            or one_stage.usage != verifier_source.usage
            or one_stage.raw_response_sha256 != verifier_source.raw_response_sha256
            or one_stage.error_type != verifier_source.error_type
            or one_stage.cache_origin != verifier_source.cache_origin
            or one_stage.source_cache_key != verifier_source.source_cache_key
        ):
            raise MatrixValidationError("primary B2b call-1 does not exactly reuse B2a")


def _cell(
    *,
    role: ModelRole,
    arm: Literal["B2a", "B2b", "B3"],
    run_dir: Path,
    cases: Sequence[Any],
    cases_path: Path,
    taxonomy: Any,
    dataset_freeze_sha256: str,
    experiment_lock_sha256: str,
    models: ReadinessModels,
    config: IE3LLMExperimentConfig,
    experiment: ExperimentLock,
    repository_root: Path,
) -> tuple[MatrixCell, dict[str, ArtifactDigest], TokenUsage]:
    manifest_path = run_dir / "run-manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    try:
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise MatrixValidationError(f"invalid or missing run manifest: {run_dir}") from exc
    _validate_manifest_contract(
        manifest=manifest,
        role=role,
        arm=arm,
        cases_path=cases_path,
        dataset_version=experiment.dataset_version,
        taxonomy_version=experiment.taxonomy_version,
        dataset_freeze_sha256=dataset_freeze_sha256,
        experiment_lock_sha256=experiment_lock_sha256,
        models=models,
        config=config,
        experiment=experiment,
    )
    predictions = load_predictions(predictions_path)
    gold_ids = {case.id for case in cases}
    indexed = index_predictions(predictions, gold_ids)
    if set(indexed) != gold_ids:
        raise MatrixValidationError(f"prediction universe is incomplete: {role.value}/{arm}")
    for prediction in predictions:
        validate_prediction(prediction, taxonomy)
    _validate_manifest_file_hash(
        manifest=manifest,
        metadata_field="prediction_sha256",
        path=predictions_path,
        label="prediction",
    )

    source_artifacts = {
        "manifest": _source(manifest_path, repository_root),
        "predictions": _source(predictions_path, repository_root),
    }

    if arm == "B2a":
        calls_path = run_dir / B2A_CALLS_FILENAME
        calls = _read_jsonl(calls_path, B2ACallRecord)
        if [call.case_id for call in calls] != [case.id for case in cases]:
            raise MatrixValidationError("B2a call universe differs from frozen test")
        if any(
            call.usage is not None
            and (call.usage.retry_input_tokens or call.usage.retry_output_tokens)
            for call in calls
        ):
            raise MatrixValidationError("B2a retry usage violates IE3 contract")
        for call in calls:
            prediction = indexed[call.case_id]
            if (
                call.prediction_status is not prediction.status
                or call.usage != prediction.usage
                or call.raw_response_sha256 != prediction.raw_response_sha256
                or call.error_type != prediction.error_type
                or call.prediction_sha256 != _sha256_json(prediction.model_dump(mode="json"))
            ):
                raise MatrixValidationError(f"{call.case_id}: B2a call differs from prediction")
        _validate_manifest_file_hash(
            manifest=manifest,
            metadata_field="calls_sha256",
            path=calls_path,
            label="B2a calls",
        )
        source_artifacts["calls"] = _source(calls_path, repository_root)
        cache_path = run_dir / B2A_CACHE_BUNDLE_FILENAME
        provenance_path = run_dir / B2A_CACHE_PROVENANCE_FILENAME
        _validate_manifest_file_hash(
            manifest=manifest,
            metadata_field="cache_bundle_sha256",
            path=cache_path,
            label="B2a cache bundle",
        )
        _validate_manifest_file_hash(
            manifest=manifest,
            metadata_field="cache_provenance_sha256",
            path=provenance_path,
            label="B2a cache provenance",
        )
        try:
            b2a_provenance = B2ACacheProvenance.model_validate_json(
                provenance_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise MatrixValidationError("B2a cache provenance is invalid") from exc
        if (
            b2a_provenance.portable_cache_bundle_sha256 != sha256_file(cache_path)
            or b2a_provenance.source_predictions_sha256 != sha256_file(predictions_path)
            or [entry.case_id for entry in b2a_provenance.entries]
            != [call.case_id for call in calls]
            or [entry.destination_cache_key for entry in b2a_provenance.entries]
            != [call.cache_key for call in calls]
            or [entry.origin for entry in b2a_provenance.entries]
            != [call.cache_origin for call in calls]
            or [entry.source_cache_key for entry in b2a_provenance.entries]
            != [call.source_cache_key for call in calls]
            or [entry.prediction_sha256 for entry in b2a_provenance.entries]
            != [call.prediction_sha256 for call in calls]
            or [entry.raw_response_sha256 for entry in b2a_provenance.entries]
            != [call.raw_response_sha256 for call in calls]
        ):
            raise MatrixValidationError("B2a cache provenance differs from calls and outputs")
        source_artifacts["cache"] = _source(cache_path, repository_root)
        source_artifacts["cache_provenance"] = _source(provenance_path, repository_root)
    else:
        calls_path = run_dir / CALLS_FILENAME
        calls = _read_jsonl(calls_path, LLMCallRecord)
        _validate_two_call_case_contract(calls=calls, predictions=indexed, gold_ids=gold_ids)
        _validate_manifest_file_hash(
            manifest=manifest,
            metadata_field="calls_sha256",
            path=calls_path,
            label="two-call calls",
        )
        output_hashes = manifest.metadata.get("output_artifact_sha256")
        if not isinstance(output_hashes, dict):
            raise MatrixValidationError("two-call manifest lacks output artifact hashes")
        expected_extra = CALL_1_PROVENANCE_FILENAME if arm == "B2b" else FORMED_INTENTIONS_FILENAME
        expected_names = {"predictions.jsonl", CALLS_FILENAME, expected_extra}
        if set(output_hashes) != expected_names:
            raise MatrixValidationError("two-call output artifact set differs from contract")
        for name in sorted(expected_names):
            path = run_dir / name
            if output_hashes[name] != sha256_file(path):
                raise MatrixValidationError(f"two-call artifact hash differs: {name}")
            source_artifacts[name] = _source(path, repository_root)
        if arm == "B2b":
            try:
                b2b_provenance = Call1Provenance.model_validate_json(
                    (run_dir / CALL_1_PROVENANCE_FILENAME).read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as exc:
                raise MatrixValidationError("B2b call-1 provenance is invalid") from exc
            first_calls = [call for call in calls if call.call_index == 1]
            if (
                b2b_provenance.role != role.value
                or [entry.case_id for entry in b2b_provenance.entries]
                != [call.case_id for call in first_calls]
                or [entry.cache_key for entry in b2b_provenance.entries]
                != [call.cache_key for call in first_calls]
                or [entry.call_contract_sha256 for entry in b2b_provenance.entries]
                != [call.call_contract_sha256 for call in first_calls]
                or [entry.cache_origin for entry in b2b_provenance.entries]
                != [call.cache_origin for call in first_calls]
                or [entry.source_cache_key for entry in b2b_provenance.entries]
                != [call.source_cache_key for call in first_calls]
                or any(not call.one_stage_cache_compatible for call in first_calls)
            ):
                raise MatrixValidationError("B2b call-1 provenance differs from calls")

    usage_complete, total_usage = _sum_call_usage(calls)
    if usage_complete:
        if manifest.total_usage != total_usage:
            raise MatrixValidationError(f"manifest usage differs from calls: {role.value}/{arm}")
    elif manifest.total_usage is not None:
        raise MatrixValidationError(
            f"usage-incomplete run must not report aggregate usage: {role.value}/{arm}"
        )
    metrics = evaluate_run(cases, predictions, taxonomy)
    return (
        MatrixCell(
            role=role,
            arm=arm,
            run_directory=_relative(run_dir, repository_root),
            run_manifest_sha256=sha256_file(manifest_path),
            predictions_sha256=sha256_file(predictions_path),
            prediction_count=len(predictions),
            success_count=sum(
                prediction.status is PredictionStatus.SUCCESS for prediction in predictions
            ),
            failed_count=sum(
                prediction.status is not PredictionStatus.SUCCESS for prediction in predictions
            ),
            hierarchical_exact_match=metrics["hierarchical_exact_match"],
            decision_macro_f1=metrics["decision_macro_f1"],
            call_count=len(calls),
            usage_complete=usage_complete,
            input_tokens=total_usage.input_tokens,
            output_tokens=total_usage.output_tokens,
            total_tokens=total_usage.total_tokens,
            estimated_cost=manifest.estimated_cost,
        ),
        source_artifacts,
        total_usage,
    )


def build_seven_run_matrix(
    *,
    runs_root: Path,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    output_path: Path,
    repository_root: Path,
) -> dict[str, object]:
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
    taxonomy = load_taxonomy(taxonomy_path)
    try:
        models = ReadinessModels.model_validate(experiment.models)
        config = IE3LLMExperimentConfig.model_validate(experiment.llm)
        maximum = float(experiment.verdict["budget_difference_max"])
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        raise MatrixValidationError("frozen experiment matrix contract is invalid") from exc

    cells: list[MatrixCell] = []
    sources: dict[str, ArtifactDigest] = {
        "dataset_freeze": _source(dataset_manifest_path, repository_root),
        "experiment_lock": _source(experiment_path, repository_root),
    }
    usage_by_cell: dict[tuple[ModelRole, str], tuple[bool, TokenUsage]] = {}
    for role, arm in EXPECTED_LLM_CELLS:
        run_dir = runs_root / role.value / arm.lower()
        cell, cell_sources, usage = _cell(
            role=role,
            arm=arm,
            run_dir=run_dir,
            cases=cases,
            cases_path=cases_path,
            taxonomy=taxonomy,
            dataset_freeze_sha256=sha256_file(dataset_manifest_path),
            experiment_lock_sha256=sha256_file(experiment_path),
            models=models,
            config=config,
            experiment=experiment,
            repository_root=repository_root,
        )
        cells.append(cell)
        for name, source in sorted(cell_sources.items()):
            sources[f"{role.value}_{arm}_{name}"] = source
        usage_by_cell[(role, arm)] = (cell.usage_complete, usage)

    _validate_primary_call1_reuse(runs_root=runs_root)

    comparisons = []
    for role in (
        ModelRole.PRIMARY_DECISION,
        ModelRole.WEAK_DECISION,
        ModelRole.CROSS_PROVIDER_REFERENCE,
    ):
        b2b_complete, b2b_usage = usage_by_cell[(role, "B2b")]
        b3_complete, b3_usage = usage_by_cell[(role, "B3")]
        comparisons.append(
            budget_comparison(
                role=role,
                b2b_total_tokens=b2b_usage.total_tokens,
                b3_total_tokens=b3_usage.total_tokens,
                maximum_allowed=maximum,
                usage_complete=b2b_complete and b3_complete,
            )
        )

    artifact = SevenRunMatrixArtifact(
        dataset_version=dataset.dataset_version,
        taxonomy_version=dataset.taxonomy_version,
        dataset_freeze_sha256=sha256_file(dataset_manifest_path),
        experiment_lock_sha256=sha256_file(experiment_path),
        source_artifacts=sources,
        cells=cells,
        budget_comparisons=comparisons,
    )
    payload = _json_bytes(artifact.model_dump(mode="json"))
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise MatrixValidationError("seven-run matrix differs from deterministic regeneration")
        status: Literal["created", "unchanged"] = "unchanged"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output_path.name}.", dir=output_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        try:
            os.replace(temporary, output_path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        status = "created"
    return {
        "status": status,
        "cell_count": len(cells),
        "matrix_complete": True,
        "budget_status": {comparison.role.value: comparison.status for comparison in comparisons},
        "matrix_sha256": hashlib.sha256(payload).hexdigest(),
        "output_path": str(output_path),
    }
