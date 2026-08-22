"""Hash-bound, deterministic IE2.1 evaluation artifacts over frozen dev Gold."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, ValidationError, model_validator

from intentbench.bootstrap import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    MIN_CLUSTERS,
    BootstrapInterval,
    paired_cluster_bootstrap,
    validate_split_integrity,
)
from intentbench.evaluator import evaluate_run, index_predictions
from intentbench.freeze import ArtifactDigest, DatasetFreezeManifest, sha256_file
from intentbench.metrics import (
    INVALID_LABEL,
    hem,
    hierarchical_exact_match,
    intent_macro_f1,
)
from intentbench.schemas import (
    Case,
    Decision,
    Prediction,
    PredictionStatus,
    Split,
    StrictModel,
)
from intentbench.taxonomy import load_taxonomy, validate_cases, validate_prediction

EVALUATOR_VERSION: Final = "intentbench-ie2.1-v1"
CONFUSION_PREDICTION_LABELS: Final = (
    *tuple(decision.value for decision in Decision),
    INVALID_LABEL,
)
EVALUATION_FILENAMES: Final = {
    "metrics": "metrics.json",
    "confusion": "confusion.csv",
    "badcases": "badcases.jsonl",
}
BADCASE_PRIORITY: Final = (
    "missing_prediction",
    "provider_failure",
    "schema_invalid",
    "known_intent_false_reject",
    "oos_false_accept",
    "no_intent_overaction",
    "ambiguity_forced_choice",
    "decision_mismatch",
    "intent_misroute",
    "hard_negative_competitor_hit",
    "context_adverse_flip",
    "desired_experience_missing",
    "object_missing",
    "horizon_mismatch",
)
REQUIRED_DOMAIN_NAMES: Final = (
    "full_hem",
    "gold_in_scope_intent_macro_f1",
    "near_oos_recall",
    "no_intent_recall",
)


class EvaluationArtifactError(ValueError):
    """Frozen dev inputs or generated evaluation artifacts violate IE2.1."""


class EvaluationMetricsArtifact(StrictModel):
    schema_version: Literal[1] = 1
    evaluator_version: Literal["intentbench-ie2.1-v1"] = EVALUATOR_VERSION
    dataset_version: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    split: Literal["dev"] = "dev"
    dataset_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: dict[str, ArtifactDigest]
    metrics: dict[str, Any]


class BadcaseRecord(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str
    split: Literal["dev"] = "dev"
    bootstrap_cluster_id: str
    error_types: list[str] = Field(min_length=1)
    gold_decision: Decision
    gold_target_intent: str | None
    prediction_status: PredictionStatus | Literal["missing"]
    predicted_decision: Decision | None
    predicted_intent: str | None
    candidate_intents: list[str] | None
    gold_evidence_quote: str
    reason_short: str | None
    tags: list[str]
    open_intent_candidate_name: str | None

    @model_validator(mode="after")
    def validate_errors(self) -> BadcaseRecord:
        if len(self.error_types) != len(set(self.error_types)):
            raise ValueError("badcase error types must be unique")
        unknown = set(self.error_types) - set(BADCASE_PRIORITY)
        if unknown:
            raise ValueError(f"unknown badcase error types: {sorted(unknown)}")
        return self


class BootstrapIntervalArtifact(StrictModel):
    point: float
    lower: float
    upper: float
    confidence_level: float
    iterations: int = Field(ge=1)
    seed: int
    cluster_count: int = Field(ge=MIN_CLUSTERS)


class BootstrapDomainResult(StrictModel):
    metric: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    cluster_count: int = Field(ge=0)
    estimable: bool
    point: float
    interval: BootstrapIntervalArtifact | None = None
    unavailable_reason: Literal["insufficient_clusters"] | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> BootstrapDomainResult:
        if self.estimable:
            if self.interval is None or self.unavailable_reason is not None:
                raise ValueError("estimable bootstrap domain requires only an interval")
        elif self.interval is not None or self.unavailable_reason != "insufficient_clusters":
            raise ValueError("unestimable bootstrap domain requires its fixed reason")
        return self


class BootstrapComparisonArtifact(StrictModel):
    schema_version: Literal[1] = 1
    evaluator_version: Literal["intentbench-ie2.1-v1"] = EVALUATOR_VERSION
    dataset_version: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    split: Literal["dev"] = "dev"
    difference: Literal["right_minus_left"] = "right_minus_left"
    left_id: str = Field(min_length=1)
    right_id: str = Field(min_length=1)
    dataset_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: dict[str, ArtifactDigest]
    confidence_level: float
    iterations: int = Field(ge=1)
    seed: int
    minimum_clusters: int = Field(ge=1)
    required_domains: dict[str, BootstrapDomainResult]

    @model_validator(mode="after")
    def validate_domains(self) -> BootstrapComparisonArtifact:
        if set(self.required_domains) != set(REQUIRED_DOMAIN_NAMES):
            raise ValueError("bootstrap artifact must contain every registered domain")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode()


def _jsonl_bytes(values: Sequence[object]) -> bytes:
    if not values:
        return b""
    return ("\n".join(_canonical_json(value) for value in values) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative_path(path: Path, repository_root: Path) -> str:
    root = repository_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise EvaluationArtifactError(f"source artifact is outside repository root: {path}")
    return resolved.relative_to(root).as_posix()


def load_formal_cases(path: Path) -> list[Case]:
    """Load strict JSONL Case records without tolerating blank lines."""

    cases: list[Case] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationArtifactError(f"cannot read Case artifact: {path}") from exc
    if not lines:
        raise EvaluationArtifactError("Case artifact must not be empty")
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            raise EvaluationArtifactError(f"{path}:{line_number}: blank JSONL line")
        try:
            cases.append(Case.model_validate_json(raw_line))
        except ValidationError as exc:
            raise EvaluationArtifactError(f"{path}:{line_number}: invalid Case") from exc
    return cases


def load_predictions(path: Path) -> list[Prediction]:
    """Load a possibly incomplete prediction universe; evaluator handles missing IDs."""

    predictions: list[Prediction] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationArtifactError(f"cannot read prediction artifact: {path}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            raise EvaluationArtifactError(f"{path}:{line_number}: blank JSONL line")
        try:
            predictions.append(Prediction.model_validate_json(raw_line))
        except ValidationError as exc:
            raise EvaluationArtifactError(f"{path}:{line_number}: invalid Prediction") from exc
    return predictions


def _load_dataset_freeze(path: Path) -> DatasetFreezeManifest:
    try:
        manifest = DatasetFreezeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise EvaluationArtifactError(f"invalid dataset freeze manifest: {path}") from exc
    if manifest.status != "frozen":
        raise EvaluationArtifactError("dev evaluation requires a frozen dataset manifest")
    return manifest


def bind_frozen_dev(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
) -> tuple[list[Case], DatasetFreezeManifest]:
    manifest = _load_dataset_freeze(dataset_manifest_path)
    for name, supplied_path in (("dev", cases_path), ("taxonomy", taxonomy_path)):
        digest = manifest.artifacts.get(name)
        if digest is None:
            raise EvaluationArtifactError(f"dataset freeze lacks {name} artifact")
        expected_path = (repository_root.resolve() / digest.path).resolve()
        if expected_path != supplied_path.resolve():
            raise EvaluationArtifactError(f"supplied {name} path differs from dataset freeze")
        if sha256_file(supplied_path) != digest.sha256:
            raise EvaluationArtifactError(f"supplied {name} hash differs from dataset freeze")
    taxonomy = load_taxonomy(taxonomy_path)
    if taxonomy.taxonomy_version != manifest.taxonomy_version:
        raise EvaluationArtifactError("taxonomy version differs from dataset freeze")
    cases = load_formal_cases(cases_path)
    if any(case.split is not Split.DEV for case in cases):
        raise EvaluationArtifactError("IE2.1 evaluator accepts frozen dev Case records only")
    validate_split_integrity(cases)
    validate_cases(cases, taxonomy)
    return cases, manifest


def _source_digest(path: Path, repository_root: Path) -> ArtifactDigest:
    return ArtifactDigest(
        path=_relative_path(path, repository_root),
        sha256=sha256_file(path),
    )


def _confusion_csv(metrics: Mapping[str, Any]) -> bytes:
    matrix = metrics.get("confusion")
    if not isinstance(matrix, dict):
        raise EvaluationArtifactError("evaluator result lacks decision confusion")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("gold_label", "predicted_label", "count", "gold_count", "row_fraction"))
    for decision in Decision:
        gold_label = decision.value
        row = matrix[gold_label]
        gold_count = sum(int(row[label]) for label in CONFUSION_PREDICTION_LABELS)
        for predicted_label in CONFUSION_PREDICTION_LABELS:
            count = int(row[predicted_label])
            fraction = count / gold_count if gold_count else 0.0
            writer.writerow((gold_label, predicted_label, count, gold_count, f"{fraction:.6f}"))
    return output.getvalue().encode()


def _route_error(case: Case, prediction: Prediction | None) -> list[str]:
    if prediction is None:
        return ["missing_prediction"]
    if prediction.status is PredictionStatus.PROVIDER_FAILURE:
        return ["provider_failure"]
    if prediction.status is PredictionStatus.SCHEMA_INVALID:
        return ["schema_invalid"]
    if prediction.decision is not case.gold.decision:
        if case.gold.decision is Decision.IN_SCOPE and prediction.decision is Decision.OOS:
            return ["known_intent_false_reject"]
        if case.gold.decision is Decision.OOS and prediction.decision is Decision.IN_SCOPE:
            return ["oos_false_accept"]
        if case.gold.decision is Decision.NO_INTENT:
            return ["no_intent_overaction"]
        if case.gold.decision is Decision.AMBIGUOUS and prediction.decision is Decision.IN_SCOPE:
            return ["ambiguity_forced_choice"]
        return ["decision_mismatch"]
    if (
        case.gold.decision is Decision.IN_SCOPE
        and prediction.predicted_intent != case.gold.target_intent
    ):
        return ["intent_misroute"]
    return []


def _semantic_errors(case: Case, prediction: Prediction | None) -> list[str]:
    if (
        "slot_required" not in case.tags
        or prediction is None
        or prediction.status is not PredictionStatus.SUCCESS
        or prediction.slots is None
    ):
        return []
    errors: list[str] = []
    if (
        case.gold.slots.desired_experience is not None
        and prediction.slots.desired_experience is None
    ):
        errors.append("desired_experience_missing")
    if case.gold.slots.object is not None and prediction.slots.object is None:
        errors.append("object_missing")
    if prediction.slots.horizon is not case.gold.slots.horizon:
        errors.append("horizon_mismatch")
    return errors


def _badcases(cases: Sequence[Case], predictions: Sequence[Prediction]) -> list[BadcaseRecord]:
    indexed = index_predictions(predictions, {case.id for case in cases})
    controls = {case.contrast_group_id: case for case in cases if "context_control" in case.tags}
    records: list[BadcaseRecord] = []
    for case in cases:
        prediction = indexed.get(case.id)
        errors = [*_route_error(case, prediction), *_semantic_errors(case, prediction)]
        if (
            prediction is not None
            and prediction.status is PredictionStatus.SUCCESS
            and prediction.decision is Decision.IN_SCOPE
            and prediction.predicted_intent in case.hard_negative_against
        ):
            errors.append("hard_negative_competitor_hit")
        if "context_distractor" in case.tags:
            control = controls.get(case.contrast_group_id)
            if (
                control is not None
                and hierarchical_exact_match(control, indexed.get(control.id)) == 1
                and hierarchical_exact_match(case, prediction) == 0
            ):
                errors.append("context_adverse_flip")
        ordered_errors = [name for name in BADCASE_PRIORITY if name in set(errors)]
        if not ordered_errors:
            continue
        if case.bootstrap_cluster_id is None:
            raise EvaluationArtifactError(f"{case.id}: badcase lacks bootstrap cluster")
        records.append(
            BadcaseRecord(
                case_id=case.id,
                bootstrap_cluster_id=case.bootstrap_cluster_id,
                error_types=ordered_errors,
                gold_decision=case.gold.decision,
                gold_target_intent=case.gold.target_intent,
                prediction_status=(prediction.status if prediction is not None else "missing"),
                predicted_decision=(prediction.decision if prediction is not None else None),
                predicted_intent=(prediction.predicted_intent if prediction is not None else None),
                candidate_intents=(
                    prediction.candidate_intents if prediction is not None else None
                ),
                gold_evidence_quote=case.gold.evidence_quote,
                reason_short=(prediction.reason_short if prediction is not None else None),
                tags=case.tags,
                open_intent_candidate_name=(
                    case.open_intent_candidate.name
                    if case.open_intent_candidate is not None
                    else None
                ),
            )
        )
    priority = {name: index for index, name in enumerate(BADCASE_PRIORITY)}
    return sorted(records, key=lambda record: (priority[record.error_types[0]], record.case_id))


def _write_bundle(output_dir: Path, payloads: dict[str, bytes]) -> Literal["created", "unchanged"]:
    paths = {name: output_dir / EVALUATION_FILENAMES[name] for name in payloads}
    existing = {name for name, path in paths.items() if path.exists()}
    if existing:
        if existing != set(paths):
            raise EvaluationArtifactError(
                f"partial evaluation outputs already exist: {sorted(existing)}"
            )
        changed = [name for name, path in paths.items() if path.read_bytes() != payloads[name]]
        if changed:
            raise EvaluationArtifactError(
                f"evaluation outputs differ from deterministic regeneration: {changed}"
            )
        return "unchanged"
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix=".evaluate-", dir=output_dir) as temporary:
            stage = Path(temporary)
            for name, payload in payloads.items():
                (stage / EVALUATION_FILENAMES[name]).write_bytes(payload)
            for name, path in paths.items():
                os.replace(stage / EVALUATION_FILENAMES[name], path)
                created.append(path)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return "created"


def evaluate_frozen_dev(
    *,
    cases_path: Path,
    predictions_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Evaluate one cached prediction artifact and atomically write IE2.1 files."""

    cases, dataset = bind_frozen_dev(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
    )
    taxonomy = load_taxonomy(taxonomy_path)
    predictions = load_predictions(predictions_path)
    metrics = evaluate_run(cases, predictions, taxonomy)
    badcases = _badcases(cases, predictions)
    artifact = EvaluationMetricsArtifact(
        dataset_version=dataset.dataset_version,
        taxonomy_version=dataset.taxonomy_version,
        dataset_freeze_sha256=sha256_file(dataset_manifest_path),
        source_artifacts={
            "dev": _source_digest(cases_path, repository_root),
            "predictions": _source_digest(predictions_path, repository_root),
            "taxonomy": _source_digest(taxonomy_path, repository_root),
        },
        metrics=metrics,
    )
    payloads = {
        "metrics": _json_bytes(artifact.model_dump(mode="json")),
        "confusion": _confusion_csv(metrics),
        "badcases": _jsonl_bytes([record.model_dump(mode="json") for record in badcases]),
    }
    status = _write_bundle(output_dir, payloads)
    return {
        "status": status,
        "split": "dev",
        "gold_case_count": len(cases),
        "prediction_count": len(predictions),
        "badcase_count": len(badcases),
        "hierarchical_exact_match": metrics["hierarchical_exact_match"],
        "metrics_sha256": _sha256_bytes(payloads["metrics"]),
        "confusion_sha256": _sha256_bytes(payloads["confusion"]),
        "badcases_sha256": _sha256_bytes(payloads["badcases"]),
        "provider_calls": 0,
    }


ComparisonMetric = Callable[[Sequence[Case], Mapping[str, Prediction]], float]


def _decision_recall(decision: Decision) -> ComparisonMetric:
    def metric(cases: Sequence[Case], predictions: Mapping[str, Prediction]) -> float:
        if not cases:
            return 0.0
        return sum(
            predictions.get(case.id) is not None
            and predictions[case.id].status is PredictionStatus.SUCCESS
            and predictions[case.id].decision is decision
            for case in cases
        ) / len(cases)

    return metric


def _domain_specs(cases: Sequence[Case]) -> tuple[tuple[str, list[Case], ComparisonMetric], ...]:
    return (
        ("full_hem", list(cases), hem),
        (
            "gold_in_scope_intent_macro_f1",
            [case for case in cases if case.gold.decision is Decision.IN_SCOPE],
            intent_macro_f1,
        ),
        (
            "near_oos_recall",
            [case for case in cases if "near_oos" in case.tags],
            _decision_recall(Decision.OOS),
        ),
        (
            "no_intent_recall",
            [case for case in cases if case.gold.decision is Decision.NO_INTENT],
            _decision_recall(Decision.NO_INTENT),
        ),
    )


def _interval_artifact(interval: BootstrapInterval) -> BootstrapIntervalArtifact:
    return BootstrapIntervalArtifact(
        point=interval.point,
        lower=interval.lower,
        upper=interval.upper,
        confidence_level=interval.confidence_level,
        iterations=interval.iterations,
        seed=interval.seed,
        cluster_count=interval.cluster_count,
    )


def _compare_domain(
    *,
    cases: Sequence[Case],
    left_predictions: Sequence[Prediction],
    right_predictions: Sequence[Prediction],
    metric_name: str,
    metric: ComparisonMetric,
    iterations: int,
    seed: int,
) -> BootstrapDomainResult:
    gold_ids = {case.id for case in cases}
    left = index_predictions(left_predictions)
    right = index_predictions(right_predictions)
    point = metric(cases, right) - metric(cases, left)
    cluster_count = len(
        {case.bootstrap_cluster_id for case in cases if case.bootstrap_cluster_id is not None}
    )
    if cluster_count < MIN_CLUSTERS:
        return BootstrapDomainResult(
            metric=metric_name,
            case_count=len(cases),
            cluster_count=cluster_count,
            estimable=False,
            point=point,
            unavailable_reason="insufficient_clusters",
        )
    interval = paired_cluster_bootstrap(
        cases,
        [prediction for prediction in left_predictions if prediction.case_id in gold_ids],
        [prediction for prediction in right_predictions if prediction.case_id in gold_ids],
        metric,
        iterations=iterations,
        seed=seed,
        confidence_level=CONFIDENCE_LEVEL,
        min_clusters=MIN_CLUSTERS,
    )
    return BootstrapDomainResult(
        metric=metric_name,
        case_count=len(cases),
        cluster_count=cluster_count,
        estimable=True,
        point=point,
        interval=_interval_artifact(interval),
    )


def compare_frozen_dev(
    *,
    cases_path: Path,
    left_predictions_path: Path,
    right_predictions_path: Path,
    left_id: str,
    right_id: str,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    output_path: Path,
    repository_root: Path,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Create the registered right-minus-left paired dev comparison artifact."""

    if iterations < 1:
        raise EvaluationArtifactError("bootstrap iterations must be positive")
    cases, dataset = bind_frozen_dev(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
    )
    taxonomy = load_taxonomy(taxonomy_path)
    left_predictions = load_predictions(left_predictions_path)
    right_predictions = load_predictions(right_predictions_path)
    gold_ids = {case.id for case in cases}
    index_predictions(left_predictions, gold_ids)
    index_predictions(right_predictions, gold_ids)
    for prediction in (*left_predictions, *right_predictions):
        validate_prediction(prediction, taxonomy)
    domains = {
        name: _compare_domain(
            cases=domain_cases,
            left_predictions=left_predictions,
            right_predictions=right_predictions,
            metric_name=name,
            metric=metric,
            iterations=iterations,
            seed=seed,
        )
        for name, domain_cases, metric in _domain_specs(cases)
    }
    artifact = BootstrapComparisonArtifact(
        dataset_version=dataset.dataset_version,
        taxonomy_version=dataset.taxonomy_version,
        left_id=left_id,
        right_id=right_id,
        dataset_freeze_sha256=sha256_file(dataset_manifest_path),
        source_artifacts={
            "dev": _source_digest(cases_path, repository_root),
            "left_predictions": _source_digest(left_predictions_path, repository_root),
            "right_predictions": _source_digest(right_predictions_path, repository_root),
            "taxonomy": _source_digest(taxonomy_path, repository_root),
        },
        confidence_level=CONFIDENCE_LEVEL,
        iterations=iterations,
        seed=seed,
        minimum_clusters=MIN_CLUSTERS,
        required_domains=domains,
    )
    payload = _json_bytes(artifact.model_dump(mode="json"))
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise EvaluationArtifactError(
                "bootstrap output differs from deterministic regeneration"
            )
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
        "split": "dev",
        "difference": "right_minus_left",
        "left_id": left_id,
        "right_id": right_id,
        "required_domains": {
            name: {
                "estimable": result.estimable,
                "point": result.point,
                "cluster_count": result.cluster_count,
            }
            for name, result in domains.items()
        },
        "bootstrap_sha256": _sha256_bytes(payload),
        "provider_calls": 0,
    }
