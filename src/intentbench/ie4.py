"""Deterministic IE4 pilot report built only from frozen cached predictions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, ValidationError, model_validator

from intentbench.bootstrap import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED
from intentbench.freeze import ArtifactDigest, sha256_file
from intentbench.matrix import MatrixCell, ModelBudgetComparison, SevenRunMatrixArtifact
from intentbench.reporting import (
    BootstrapComparisonArtifact,
    BootstrapDomainResult,
    EvaluationMetricsArtifact,
    compare_frozen_split,
    evaluate_frozen_split,
    load_predictions,
)
from intentbench.schemas import (
    ModelRole,
    Prediction,
    PredictionStatus,
    RunManifest,
    Split,
    StrictModel,
)
from intentbench.verdict import ComparisonEvidence, MetricInterval, VerdictResult, decide

REPORT_SCHEMA_VERSION = 1
REPORT_FILENAME = "report.json"
REPORT_MARKDOWN_FILENAME = "report.md"
RunRole = Literal[
    "none", "embedding", "primary_decision", "weak_decision", "cross_provider_reference"
]
ArmName = Literal["B0", "B1", "B2a", "B2b", "B3"]


class IE4ReportError(ValueError):
    """Frozen IE4 inputs or deterministic report outputs violate the contract."""


class RunSummary(StrictModel):
    run_id: str
    role: RunRole
    arm: ArmName
    model: str
    verdict_authority: bool
    prediction_count: int = Field(ge=1)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    hierarchical_exact_match: float = Field(ge=0, le=1)
    decision_macro_f1: float = Field(ge=0, le=1)
    in_scope_intent_macro_f1: float = Field(ge=0, le=1)
    near_oos_recall: float = Field(ge=0, le=1)
    no_intent_recall: float = Field(ge=0, le=1)
    context_distractor_error_count: int = Field(ge=0)
    schema_invalid_count: int = Field(ge=0)
    latency_complete: bool
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    usage_complete: bool
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    pricing_status: Literal["registered", "not_registered", "not_applicable"]
    failure_counts: dict[str, int]
    badcase_count: int = Field(ge=0)
    badcase_error_counts: dict[str, int]
    artifacts: dict[str, ArtifactDigest]

    @model_validator(mode="after")
    def validate_counts(self) -> RunSummary:
        if self.success_count + self.failure_count != self.prediction_count:
            raise ValueError("run success/failure counts must cover every prediction")
        if self.latency_complete != (
            self.latency_p50_ms is not None and self.latency_p95_ms is not None
        ):
            raise ValueError("latency completeness disagrees with percentiles")
        if self.usage_complete != (self.total_tokens is not None):
            raise ValueError("usage completeness disagrees with token total")
        return self


class ComparisonSummary(StrictModel):
    role: ModelRole
    model: str
    verdict_authority: bool
    left_run_id: str
    right_run_id: str
    difference: Literal["B3_minus_B2b"] = "B3_minus_B2b"
    required_domains: dict[str, BootstrapDomainResult]
    budget: ModelBudgetComparison
    verdict: VerdictResult
    artifacts: dict[str, ArtifactDigest]


class SemanticAuditStatus(StrictModel):
    status: Literal["awaiting_human", "complete"]
    frozen_case_count: Literal[24] = 24
    audit_row_count: Literal[48] = 48
    audited_model_roles: tuple[Literal["primary_decision"], Literal["weak_decision"]] = (
        "primary_decision",
        "weak_decision",
    )
    rubric_fields: tuple[Literal["action"], Literal["object"], Literal["horizon"]] = (
        "action",
        "object",
        "horizon",
    )
    affects_verdict: Literal[False] = False
    workbook: ArtifactDigest
    csv: ArtifactDigest | None = None


class IE4PilotReport(StrictModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["ie4_pilot_report"] = "ie4_pilot_report"
    dataset_version: str
    taxonomy_version: str
    split: Literal["test"] = "test"
    auto_analysis_complete: Literal[True] = True
    pilot_status: Literal["awaiting_semantic_audit", "complete"]
    global_verdict_authority: Literal["primary_decision"] = "primary_decision"
    global_verdict: VerdictResult
    runs: list[RunSummary]
    comparisons: list[ComparisonSummary]
    semantic_audit: SemanticAuditStatus
    source_artifacts: dict[str, ArtifactDigest]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_authority(self) -> IE4PilotReport:
        authorities = [item for item in self.comparisons if item.verdict_authority]
        if len(authorities) != 1 or authorities[0].role is not ModelRole.PRIMARY_DECISION:
            raise ValueError("only primary_decision may be global verdict authority")
        if self.global_verdict != authorities[0].verdict:
            raise ValueError("global verdict must equal the primary comparison verdict")
        if self.pilot_status == "complete" and self.semantic_audit.status != "complete":
            raise ValueError("pilot cannot be complete before semantic audit")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode()


def _relative(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    root = repository_root.resolve()
    if not resolved.is_relative_to(root):
        raise IE4ReportError(f"IE4 source is outside repository: {path}")
    return resolved.relative_to(root).as_posix()


def _source(path: Path, repository_root: Path) -> ArtifactDigest:
    return ArtifactDigest(path=_relative(path, repository_root), sha256=sha256_file(path))


def _read_model(path: Path, model: type[Any]) -> Any:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        raise IE4ReportError(f"invalid IE4 input artifact: {path}") from exc


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise IE4ReportError("percentile requires values and quantile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _latency_summary(predictions: Sequence[Prediction]) -> tuple[bool, float | None, float | None]:
    values = [prediction.latency_ms for prediction in predictions]
    if any(value is None for value in values):
        return False, None, None
    numeric = [value for value in values if value is not None]
    return True, _percentile(numeric, 0.5), _percentile(numeric, 0.95)


def _badcase_error_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if raw_line:
                payload = json.loads(raw_line)
                counts.update(payload["error_types"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise IE4ReportError(f"invalid IE4 badcase artifact: {path}") from exc
    return dict(sorted(counts.items()))


def _run_summary(
    *,
    role: RunRole,
    arm: ArmName,
    run_dir: Path,
    evaluation_dir: Path,
    repository_root: Path,
    matrix_cell: MatrixCell | None,
) -> RunSummary:
    predictions_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "run-manifest.json"
    metrics_path = evaluation_dir / "metrics.json"
    confusion_path = evaluation_dir / "confusion.csv"
    badcases_path = evaluation_dir / "badcases.jsonl"
    predictions = load_predictions(predictions_path)
    manifest = _read_model(manifest_path, RunManifest)
    metrics_artifact = _read_model(metrics_path, EvaluationMetricsArtifact)
    metrics = metrics_artifact.metrics
    latency_complete, latency_p50, latency_p95 = _latency_summary(predictions)
    success_count = sum(prediction.status is PredictionStatus.SUCCESS for prediction in predictions)
    failure_count = len(predictions) - success_count

    if matrix_cell is not None:
        if (
            matrix_cell.role.value != role
            or matrix_cell.arm != arm
            or matrix_cell.prediction_count != len(predictions)
            or matrix_cell.success_count != success_count
            or matrix_cell.failed_count != failure_count
        ):
            raise IE4ReportError(f"matrix cell disagrees with {role}/{arm} artifacts")
        usage_complete = matrix_cell.usage_complete
        total_tokens = matrix_cell.total_tokens if usage_complete else None
        estimated_cost = matrix_cell.estimated_cost
    else:
        usage_complete = manifest.total_usage is not None
        total_tokens = manifest.total_usage.total_tokens if manifest.total_usage else None
        estimated_cost = manifest.estimated_cost

    pricing_status: Literal["registered", "not_registered", "not_applicable"]
    if estimated_cost is not None:
        pricing_status = "registered"
    elif arm == "B0":
        pricing_status = "not_applicable"
    else:
        pricing_status = "not_registered"

    no_intent = metrics["decision"]["no_intent"]["recall"]
    return RunSummary(
        run_id=f"{role}/{arm}",
        role=role,
        arm=arm,
        model=manifest.model,
        verdict_authority=role == ModelRole.PRIMARY_DECISION.value,
        prediction_count=len(predictions),
        success_count=success_count,
        failure_count=failure_count,
        hierarchical_exact_match=metrics["hierarchical_exact_match"],
        decision_macro_f1=metrics["decision_macro_f1"],
        in_scope_intent_macro_f1=metrics["in_scope_intent_macro_f1"],
        near_oos_recall=metrics["near_oos_recall"],
        no_intent_recall=no_intent,
        context_distractor_error_count=metrics["context_distractor_error_count"],
        schema_invalid_count=metrics["schema_invalid_count"],
        latency_complete=latency_complete,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        usage_complete=usage_complete,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
        pricing_status=pricing_status,
        failure_counts=metrics["failure_counts"],
        badcase_count=sum(1 for line in badcases_path.read_text().splitlines() if line),
        badcase_error_counts=_badcase_error_counts(badcases_path),
        artifacts={
            "manifest": _source(manifest_path, repository_root),
            "predictions": _source(predictions_path, repository_root),
            "metrics": _source(metrics_path, repository_root),
            "confusion": _source(confusion_path, repository_root),
            "badcases": _source(badcases_path, repository_root),
        },
    )


def _metric_interval(domain: BootstrapDomainResult) -> MetricInterval:
    if domain.interval is None:
        return MetricInterval(point=domain.point, lower=domain.point, upper=domain.point)
    return MetricInterval(
        point=domain.point,
        lower=domain.interval.lower,
        upper=domain.interval.upper,
    )


def comparison_evidence(
    *,
    comparison: BootstrapComparisonArtifact,
    budget: ModelBudgetComparison,
    b2b: RunSummary,
    b3: RunSummary,
) -> ComparisonEvidence:
    """Translate generated IE4 artifacts into the frozen tri-state verdict input."""

    domains = comparison.required_domains
    return ComparisonEvidence(
        run_integrity_valid=True,
        contract_aligned=True,
        required_cis_estimable=all(domain.estimable for domain in domains.values()),
        hem=_metric_interval(domains["full_hem"]),
        id_intent_macro_f1=_metric_interval(domains["gold_in_scope_intent_macro_f1"]),
        near_oos_recall=_metric_interval(domains["near_oos_recall"]),
        no_intent_recall=_metric_interval(domains["no_intent_recall"]),
        b2b_context_distractor_error_count=b2b.context_distractor_error_count,
        b3_context_distractor_error_count=b3.context_distractor_error_count,
        b2b_schema_invalid_count=b2b.schema_invalid_count,
        b3_schema_invalid_count=b3.schema_invalid_count,
        b2b_total_tokens=budget.b2b_total_tokens,
        b3_total_tokens=budget.b3_total_tokens,
        usage_complete=budget.usage_complete,
        latency_complete=b2b.latency_complete and b3.latency_complete,
        cost_complete=(b2b.estimated_cost_usd is not None and b3.estimated_cost_usd is not None),
    )


def _format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _format_interval(domain: BootstrapDomainResult) -> str:
    if domain.interval is None:
        return f"{domain.point:+.3f} (CI unavailable)"
    return f"{domain.point:+.3f} [{domain.interval.lower:+.3f}, {domain.interval.upper:+.3f}]"


def render_markdown(report: IE4PilotReport) -> str:
    """Render the human-facing report only from the validated structured artifact."""

    primary = next(item for item in report.comparisons if item.verdict_authority)
    verdict = primary.verdict.verdict.value if primary.verdict.verdict else "invalid_run"
    primary_hem = primary.required_domains["full_hem"]
    bootstrap_draws = primary_hem.interval.iterations if primary_hem.interval else None
    draw_description = f"{bootstrap_draws:,} draws" if bootstrap_draws else "unavailable CI"
    lines = [
        "# IE4 Pilot Report",
        "",
        "> Auto analysis is complete; the frozen 48-row semantic-preservation audit "
        "is awaiting human review.",
        "",
        "## Decision",
        "",
        f"Global verdict: **{verdict}** (`primary_decision` only). Reasons: "
        + ", ".join(f"`{reason}`" for reason in primary.verdict.reasons)
        + ".",
        "",
        "The weak-model and cross-provider results are diagnostic replications. "
        "They are not averaged, "
        "voted, or allowed to replace the preregistered primary model.",
        "",
        "## Frozen-test runs",
        "",
        "| Role | Arm | Model | HEM | Decision Macro-F1 | ID Macro-F1 | "
        "Near-OOS recall | No-intent recall | Failures | P50 ms | P95 ms | "
        "Tokens | Cost USD |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    run.role,
                    run.arm,
                    run.model,
                    _format_number(run.hierarchical_exact_match),
                    _format_number(run.decision_macro_f1),
                    _format_number(run.in_scope_intent_macro_f1),
                    _format_number(run.near_oos_recall),
                    _format_number(run.no_intent_recall),
                    str(run.failure_count),
                    _format_number(run.latency_p50_ms, 1),
                    _format_number(run.latency_p95_ms, 1),
                    _format_number(run.total_tokens),
                    _format_number(run.estimated_cost_usd, 4),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## B3 - B2b paired comparisons",
            "",
            "Intervals are 95% paired cluster-bootstrap intervals with the frozen seed "
            f"and {draw_description}.",
            "",
            "| Role | Authority | Δ HEM [95% CI] | Δ ID Macro-F1 [95% CI] | "
            "Δ near-OOS [95% CI] | Δ no-intent [95% CI] | Token difference | "
            "Verdict | Reasons |",
            "|---|---|---|---|---|---|---:|---|---|",
        ]
    )
    for item in report.comparisons:
        domains = item.required_domains
        item_verdict = item.verdict.verdict.value if item.verdict.verdict else "invalid_run"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.role.value,
                    "global" if item.verdict_authority else "diagnostic",
                    _format_interval(domains["full_hem"]),
                    _format_interval(domains["gold_in_scope_intent_macro_f1"]),
                    _format_interval(domains["near_oos_recall"]),
                    _format_interval(domains["no_intent_recall"]),
                    f"{item.budget.absolute_difference_rate:.1%}",
                    item_verdict,
                    ", ".join(item.verdict.reasons),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Audit and interpretation boundary",
            "",
            "- The 24 frozen cases are audited for action, object, and horizon preservation "
            "on B3 Stage A for the primary and weak roles (48 rows).",
            "- The audit is diagnostic only. It does not change the tri-state verdict and "
            "must not be used to tune the test prompts.",
            "- A budget-confounded verdict is a contract result: it blocks causal "
            "attribution to the B3 structure even when point estimates or intervals look "
            "favorable or unfavorable.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"


def _write_artifact(path: Path, payload: bytes) -> Literal["created", "unchanged"]:
    if path.exists():
        if path.read_bytes() != payload:
            raise IE4ReportError(f"IE4 output differs from deterministic regeneration: {path}")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return "created"


def build_ie4_pilot_report(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    matrix_path: Path,
    semantic_audit_workbook_path: Path,
    output_dir: Path,
    repository_root: Path,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Generate every automatic IE4 artifact without making provider calls."""

    if iterations < 1:
        raise IE4ReportError("bootstrap iterations must be positive")
    matrix = _read_model(matrix_path, SevenRunMatrixArtifact)
    if not semantic_audit_workbook_path.is_file():
        raise IE4ReportError("semantic-audit workbook is missing")

    baseline_specs: list[tuple[RunRole, ArmName, Path, MatrixCell | None]] = [
        ("none", "B0", repository_root / "experiments/ie3-test/baselines/b0", None),
        ("embedding", "B1", repository_root / "experiments/ie3-test/baselines/b1", None),
    ]
    llm_specs = [
        (
            cast(RunRole, cell.role.value),
            cell.arm,
            repository_root / cell.run_directory,
            cell,
        )
        for cell in matrix.cells
    ]
    specs = [*baseline_specs, *llm_specs]

    runs: list[RunSummary] = []
    for role, arm, run_dir, matrix_cell in specs:
        evaluation_dir = output_dir / "evaluations" / role / arm.lower()
        evaluate_frozen_split(
            split=Split.TEST,
            cases_path=cases_path,
            predictions_path=run_dir / "predictions.jsonl",
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            output_dir=evaluation_dir,
            repository_root=repository_root,
        )
        runs.append(
            _run_summary(
                role=role,
                arm=arm,
                run_dir=run_dir,
                evaluation_dir=evaluation_dir,
                repository_root=repository_root,
                matrix_cell=matrix_cell,
            )
        )

    runs_by_id = {run.run_id: run for run in runs}
    budgets = {item.role: item for item in matrix.budget_comparisons}
    comparisons: list[ComparisonSummary] = []
    for model_role in (
        ModelRole.PRIMARY_DECISION,
        ModelRole.WEAK_DECISION,
        ModelRole.CROSS_PROVIDER_REFERENCE,
    ):
        left = runs_by_id[f"{model_role.value}/B2b"]
        right = runs_by_id[f"{model_role.value}/B3"]
        comparison_path = output_dir / "comparisons" / f"{model_role.value}-b2b-vs-b3.json"
        compare_frozen_split(
            split=Split.TEST,
            cases_path=cases_path,
            left_predictions_path=repository_root
            / next(
                cell.run_directory
                for cell in matrix.cells
                if cell.role is model_role and cell.arm == "B2b"
            )
            / "predictions.jsonl",
            right_predictions_path=repository_root
            / next(
                cell.run_directory
                for cell in matrix.cells
                if cell.role is model_role and cell.arm == "B3"
            )
            / "predictions.jsonl",
            left_id=left.run_id,
            right_id=right.run_id,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            output_path=comparison_path,
            repository_root=repository_root,
            iterations=iterations,
            seed=seed,
        )
        comparison = _read_model(comparison_path, BootstrapComparisonArtifact)
        evidence = comparison_evidence(
            comparison=comparison,
            budget=budgets[model_role],
            b2b=left,
            b3=right,
        )
        comparisons.append(
            ComparisonSummary(
                role=model_role,
                model=left.model,
                verdict_authority=model_role is ModelRole.PRIMARY_DECISION,
                left_run_id=left.run_id,
                right_run_id=right.run_id,
                required_domains=comparison.required_domains,
                budget=budgets[model_role],
                verdict=decide(evidence),
                artifacts={"bootstrap": _source(comparison_path, repository_root)},
            )
        )

    primary = next(item for item in comparisons if item.verdict_authority)
    report = IE4PilotReport(
        dataset_version=matrix.dataset_version,
        taxonomy_version=matrix.taxonomy_version,
        pilot_status="awaiting_semantic_audit",
        global_verdict=primary.verdict,
        runs=runs,
        comparisons=comparisons,
        semantic_audit=SemanticAuditStatus(
            status="awaiting_human",
            workbook=_source(semantic_audit_workbook_path, repository_root),
        ),
        source_artifacts={
            "test": _source(cases_path, repository_root),
            "taxonomy": _source(taxonomy_path, repository_root),
            "dataset_freeze": _source(dataset_manifest_path, repository_root),
            "corrected_seven_run_matrix": _source(matrix_path, repository_root),
        },
        limitations=[
            "The 160-case dataset is synthetic, balanced, and non-blind; it is not "
            "production traffic.",
            "Only 112 frozen test cases contribute to formal test metrics; small slices "
            "retain wide intervals.",
            "Cached predictions make evaluation reproducible, but hosted model inference "
            "is not fully replayable.",
            "Weak and cross-provider pricing is not registered, so their cost gates "
            "remain incomplete.",
            "No production Kindred runtime, user data, or Activity authority is changed "
            "by this pilot.",
        ],
    )
    report_payload = _json_bytes(report.model_dump(mode="json"))
    markdown_payload = render_markdown(report).encode()
    report_status = _write_artifact(output_dir / REPORT_FILENAME, report_payload)
    markdown_status = _write_artifact(output_dir / REPORT_MARKDOWN_FILENAME, markdown_payload)
    return {
        "status": "created" if "created" in (report_status, markdown_status) else "unchanged",
        "provider_calls": 0,
        "run_count": len(runs),
        "comparison_count": len(comparisons),
        "global_verdict": (
            primary.verdict.verdict.value if primary.verdict.verdict is not None else None
        ),
        "global_verdict_reasons": primary.verdict.reasons,
        "pilot_status": report.pilot_status,
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "output_dir": str(output_dir),
    }
