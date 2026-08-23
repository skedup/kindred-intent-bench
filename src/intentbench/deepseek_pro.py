"""Post-freeze DeepSeek V4 Pro robustness reporting."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from intentbench.bootstrap import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED
from intentbench.freeze import (
    ArtifactDigest,
    DatasetFreezeManifest,
    sha256_file,
    verify_test_guard,
)
from intentbench.ie4 import IE4PilotReport
from intentbench.matrix import ModelBudgetComparison, budget_comparison
from intentbench.reporting import (
    BootstrapComparisonArtifact,
    BootstrapDomainResult,
    EvaluationMetricsArtifact,
    compare_frozen_split,
    evaluate_frozen_split,
    load_predictions,
)
from intentbench.schemas import ModelRole, PredictionStatus, RunManifest, Split, StrictModel
from intentbench.two_stage import FormedIntentionRecord, LLMCallRecord
from intentbench.verdict import ComparisonEvidence, MetricInterval, VerdictResult, decide

MODEL = "deepseek-v4-pro"
REPORT_FILENAME = "report.json"
REPORT_MARKDOWN_FILENAME = "README.md"


class DeepSeekProReportError(ValueError):
    """DeepSeek Pro robustness inputs or outputs violate the frozen contract."""


class ProRunSummary(StrictModel):
    arm: Literal["B2b", "B3"]
    model: Literal["deepseek-v4-pro"] = "deepseek-v4-pro"
    prediction_count: Literal[112] = 112
    success_count: int = Field(ge=0, le=112)
    failure_count: int = Field(ge=0, le=112)
    incomplete_prediction_count: int = Field(ge=0, le=112)
    hierarchical_exact_match: float = Field(ge=0, le=1)
    decision_macro_f1: float = Field(ge=0, le=1)
    in_scope_intent_macro_f1: float = Field(ge=0, le=1)
    near_oos_recall: float = Field(ge=0, le=1)
    no_intent_recall: float = Field(ge=0, le=1)
    context_distractor_error_count: int = Field(ge=0)
    schema_invalid_count: int = Field(ge=0)
    call_count: int = Field(ge=1)
    incomplete_call_count: int = Field(ge=0)
    incomplete_calls_within_one_token_of_cap: int = Field(ge=0)
    max_output_tokens_per_call: Literal[512] = 512
    formed_intention_success_count: int | None = Field(default=None, ge=0, le=112)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    usage_complete: Literal[True] = True
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: None = None
    artifacts: dict[str, ArtifactDigest]

    @model_validator(mode="after")
    def validate_counts(self) -> ProRunSummary:
        if self.success_count + self.failure_count != self.prediction_count:
            raise ValueError("Pro success/failure counts must cover the prediction universe")
        if self.incomplete_prediction_count > self.failure_count:
            raise ValueError("incomplete predictions cannot exceed failures")
        if self.incomplete_calls_within_one_token_of_cap > self.incomplete_call_count:
            raise ValueError("near-cap incomplete count cannot exceed incomplete calls")
        if (self.arm == "B3") != (self.formed_intention_success_count is not None):
            raise ValueError("only B3 carries formed-intention success count")
        return self


class FlashArmReference(StrictModel):
    arm: Literal["B2b", "B3"]
    model: Literal["deepseek/deepseek-v4-flash"] = "deepseek/deepseek-v4-flash"
    success_count: int = Field(ge=0, le=112)
    failure_count: int = Field(ge=0, le=112)
    hierarchical_exact_match: float = Field(ge=0, le=1)
    pro_minus_flash_hem: float
    pro_minus_flash_success_count: int


class ProComparisonSummary(StrictModel):
    difference: Literal["B3_minus_B2b"] = "B3_minus_B2b"
    required_domains: dict[str, BootstrapDomainResult]
    budget: ModelBudgetComparison
    verdict_authority: Literal[False] = False
    global_verdict_effect: Literal["none"] = "none"
    diagnostic_verdict: VerdictResult
    artifact: ArtifactDigest


class SemanticAuditCoverage(StrictModel):
    frozen_case_count: Literal[24] = 24
    formed_intention_success_count: int = Field(ge=0, le=24)
    formed_intention_failure_count: int = Field(ge=0, le=24)
    successful_case_ids: list[str]
    decision: Literal["do_not_add_pro_audit"] = "do_not_add_pro_audit"
    reason: Literal["insufficient_successful_stage_a_coverage"] = (
        "insufficient_successful_stage_a_coverage"
    )

    @model_validator(mode="after")
    def validate_counts(self) -> SemanticAuditCoverage:
        if self.formed_intention_success_count + self.formed_intention_failure_count != 24:
            raise ValueError("Pro semantic-audit coverage must cover all 24 frozen cases")
        if len(self.successful_case_ids) != self.formed_intention_success_count:
            raise ValueError("successful Pro semantic-audit IDs disagree with the count")
        if len(set(self.successful_case_ids)) != len(self.successful_case_ids):
            raise ValueError("successful Pro semantic-audit IDs must be unique")
        return self


class DeepSeekProRobustnessReport(StrictModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["deepseek_pro_post_freeze_robustness"] = (
        "deepseek_pro_post_freeze_robustness"
    )
    model: Literal["deepseek-v4-pro"] = "deepseek-v4-pro"
    experiment_kind: Literal["post_freeze_diagnostic"] = "post_freeze_diagnostic"
    global_verdict_authority: Literal[False] = False
    global_verdict_effect: Literal["none"] = "none"
    conclusion: Literal["no_rescue_under_frozen_contract"] = "no_rescue_under_frozen_contract"
    runs: list[ProRunSummary]
    comparison: ProComparisonSummary
    flash_reference: list[FlashArmReference]
    semantic_audit_coverage: SemanticAuditCoverage
    source_artifacts: dict[str, ArtifactDigest]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_arms(self) -> DeepSeekProRobustnessReport:
        if [run.arm for run in self.runs] != ["B2b", "B3"]:
            raise ValueError("Pro report must contain B2b then B3")
        if [run.arm for run in self.flash_reference] != ["B2b", "B3"]:
            raise ValueError("Flash reference must contain B2b then B3")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode()


def _relative(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    root = repository_root.resolve()
    if not resolved.is_relative_to(root):
        raise DeepSeekProReportError(f"source is outside repository: {path}")
    return resolved.relative_to(root).as_posix()


def _source(path: Path, repository_root: Path) -> ArtifactDigest:
    return ArtifactDigest(path=_relative(path, repository_root), sha256=sha256_file(path))


def _read_model(path: Path, model: type[Any]) -> Any:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        raise DeepSeekProReportError(f"invalid robustness artifact: {path}") from exc


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    try:
        return [model.model_validate_json(line) for line in path.read_text().splitlines()]
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        raise DeepSeekProReportError(f"invalid robustness JSONL: {path}") from exc


def _metric_interval(domain: BootstrapDomainResult) -> MetricInterval:
    if domain.interval is None:
        return MetricInterval(point=domain.point, lower=domain.point, upper=domain.point)
    return MetricInterval(
        point=domain.point,
        lower=domain.interval.lower,
        upper=domain.interval.upper,
    )


def _run_summary(
    *, arm: Literal["B2b", "B3"], run_dir: Path, evaluation_dir: Path, repository_root: Path
) -> ProRunSummary:
    predictions_path = run_dir / "predictions.jsonl"
    calls_path = run_dir / "calls.jsonl"
    manifest_path = run_dir / "run-manifest.json"
    metrics_path = evaluation_dir / "metrics.json"
    confusion_path = evaluation_dir / "confusion.csv"
    badcases_path = evaluation_dir / "badcases.jsonl"
    predictions = load_predictions(predictions_path)
    calls: list[LLMCallRecord] = _read_jsonl(calls_path, LLMCallRecord)
    manifest: RunManifest = _read_model(manifest_path, RunManifest)
    metrics_artifact: EvaluationMetricsArtifact = _read_model(
        metrics_path, EvaluationMetricsArtifact
    )
    metrics = metrics_artifact.metrics
    if (
        manifest.model != f"deepseek/{MODEL}"
        or manifest.metadata.get("git_worktree_dirty") is not False
    ):
        raise DeepSeekProReportError(f"{arm} manifest model or clean-worktree identity drifted")
    if any(call.requested_model != MODEL or call.reported_model != MODEL for call in calls):
        raise DeepSeekProReportError(f"{arm} call model identity drifted")
    if any(call.usage is None for call in calls):
        raise DeepSeekProReportError(f"{arm} usage is incomplete")
    output_cap = manifest.parameters.max_output_tokens_per_call
    if output_cap != 512:
        raise DeepSeekProReportError(f"{arm} output cap differs from frozen contract")
    success_count = sum(prediction.status is PredictionStatus.SUCCESS for prediction in predictions)
    incomplete_predictions = sum(
        prediction.error_type is not None and prediction.error_type.endswith("incomplete")
        for prediction in predictions
    )
    incomplete_calls = [call for call in calls if call.error_type == "incomplete"]
    near_cap = sum(
        call.usage is not None and call.usage.output_tokens >= output_cap - 1
        for call in incomplete_calls
    )
    formed_success: int | None = None
    artifacts = {
        "manifest": _source(manifest_path, repository_root),
        "predictions": _source(predictions_path, repository_root),
        "calls": _source(calls_path, repository_root),
        "metrics": _source(metrics_path, repository_root),
        "confusion": _source(confusion_path, repository_root),
        "badcases": _source(badcases_path, repository_root),
    }
    if arm == "B3":
        formed_path = run_dir / "formed-intentions.jsonl"
        formed: list[FormedIntentionRecord] = _read_jsonl(formed_path, FormedIntentionRecord)
        formed_success = sum(item.status is PredictionStatus.SUCCESS for item in formed)
        artifacts["formed_intentions"] = _source(formed_path, repository_root)
    total_usage = manifest.total_usage
    if total_usage is None:
        raise DeepSeekProReportError(f"{arm} manifest lacks total usage")
    return ProRunSummary(
        arm=arm,
        success_count=success_count,
        failure_count=len(predictions) - success_count,
        incomplete_prediction_count=incomplete_predictions,
        hierarchical_exact_match=metrics["hierarchical_exact_match"],
        decision_macro_f1=metrics["decision_macro_f1"],
        in_scope_intent_macro_f1=metrics["in_scope_intent_macro_f1"],
        near_oos_recall=metrics["near_oos_recall"],
        no_intent_recall=metrics["decision"]["no_intent"]["recall"],
        context_distractor_error_count=metrics["context_distractor_error_count"],
        schema_invalid_count=metrics["schema_invalid_count"],
        call_count=len(calls),
        incomplete_call_count=len(incomplete_calls),
        incomplete_calls_within_one_token_of_cap=near_cap,
        formed_intention_success_count=formed_success,
        latency_p50_ms=manifest.metadata["latency_p50_ms"],
        latency_p95_ms=manifest.metadata["latency_p95_ms"],
        total_tokens=total_usage.total_tokens,
        artifacts=artifacts,
    )


def _flash_reference(
    *, pro_runs: list[ProRunSummary], ie4_report_path: Path
) -> list[FlashArmReference]:
    ie4: IE4PilotReport = _read_model(ie4_report_path, IE4PilotReport)
    by_arm = {
        run.arm: run
        for run in ie4.runs
        if run.role == ModelRole.WEAK_DECISION.value and run.arm in ("B2b", "B3")
    }
    if set(by_arm) != {"B2b", "B3"}:
        raise DeepSeekProReportError("IE4 report lacks both frozen Flash reference arms")
    return [
        FlashArmReference(
            arm=pro.arm,
            success_count=by_arm[pro.arm].success_count,
            failure_count=by_arm[pro.arm].failure_count,
            hierarchical_exact_match=by_arm[pro.arm].hierarchical_exact_match,
            pro_minus_flash_hem=(
                pro.hierarchical_exact_match - by_arm[pro.arm].hierarchical_exact_match
            ),
            pro_minus_flash_success_count=pro.success_count - by_arm[pro.arm].success_count,
        )
        for pro in pro_runs
    ]


def _semantic_audit_coverage(
    *, dataset: DatasetFreezeManifest, formed_intentions_path: Path
) -> SemanticAuditCoverage:
    audit_ids = dataset.semantic_audit_case_ids
    if len(audit_ids) != 24 or len(set(audit_ids)) != 24:
        raise DeepSeekProReportError("dataset freeze must contain 24 unique semantic-audit IDs")
    formed: list[FormedIntentionRecord] = _read_jsonl(formed_intentions_path, FormedIntentionRecord)
    formed_by_id = {item.case_id: item for item in formed}
    if len(formed_by_id) != len(formed):
        raise DeepSeekProReportError("B3 formed intentions contain duplicate case IDs")
    missing = set(audit_ids) - set(formed_by_id)
    if missing:
        raise DeepSeekProReportError(f"B3 lacks semantic-audit cases: {sorted(missing)}")
    successful = [
        case_id for case_id in audit_ids if formed_by_id[case_id].status is PredictionStatus.SUCCESS
    ]
    return SemanticAuditCoverage(
        formed_intention_success_count=len(successful),
        formed_intention_failure_count=len(audit_ids) - len(successful),
        successful_case_ids=successful,
    )


def _format_interval(domain: BootstrapDomainResult) -> str:
    if domain.interval is None:
        return f"{domain.point:+.4f} (CI unavailable)"
    return f"{domain.point:+.4f} [{domain.interval.lower:+.4f}, {domain.interval.upper:+.4f}]"


def render_deepseek_pro_markdown(report: DeepSeekProRobustnessReport) -> str:
    comparison = report.comparison
    verdict = comparison.diagnostic_verdict.verdict
    verdict_text = verdict.value if verdict is not None else "invalid_run"
    domains = comparison.required_domains
    flash = {item.arm: item for item in report.flash_reference}
    lines = [
        "# DeepSeek V4 Pro post-freeze robustness",
        "",
        "## Outcome",
        "",
        "DeepSeek V4 Pro did **not** rescue the frozen weak-model result under the same "
        "512-token-per-call contract. This is a diagnostic result and has no effect on the "
        "Gemini primary verdict.",
        "",
        "| Model | Arm | Success | Failures | HEM | P50 ms | P95 ms | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.runs:
        lines.append(
            f"| DeepSeek V4 Pro | {run.arm} | {run.success_count}/112 | "
            f"{run.failure_count} | {run.hierarchical_exact_match:.4f} | "
            f"{run.latency_p50_ms:.1f} | {run.latency_p95_ms:.1f} | "
            f"{run.total_tokens:,} |"
        )
    for arm in ("B2b", "B3"):
        item = flash[arm]
        lines.append(
            f"| DeepSeek V4 Flash (frozen reference) | {arm} | {item.success_count}/112 | "
            f"{item.failure_count} | {item.hierarchical_exact_match:.4f} | — | — | — |"
        )
    lines.extend(
        [
            "",
            "Pro minus Flash is descriptive only: B2b HEM "
            f"{flash['B2b'].pro_minus_flash_hem:+.4f} and B3 HEM "
            f"{flash['B3'].pro_minus_flash_hem:+.4f}. The model comparison was added after "
            "the original freeze and is not a new verdict authority.",
            "",
            "## Within-Pro B3 minus B2b",
            "",
            f"Diagnostic verdict: **{verdict_text}**. Reasons: "
            + ", ".join(f"`{reason}`" for reason in comparison.diagnostic_verdict.reasons)
            + ".",
            "",
            "| Metric | Difference [95% paired cluster-bootstrap CI] |",
            "|---|---:|",
            f"| HEM | {_format_interval(domains['full_hem'])} |",
            "| ID intent Macro-F1 | "
            f"{_format_interval(domains['gold_in_scope_intent_macro_f1'])} |",
            f"| Near-OOS recall | {_format_interval(domains['near_oos_recall'])} |",
            f"| No-intent recall | {_format_interval(domains['no_intent_recall'])} |",
            "",
            f"B2b/B3 token difference is {comparison.budget.absolute_difference_rate:.2%}; "
            "pricing is not registered. The pre-registered contract therefore stops at "
            "`inconclusive` before structural attribution.",
            "",
            "## Output-cap evidence",
            "",
        ]
    )
    for run in report.runs:
        lines.append(
            f"- {run.arm}: {run.incomplete_call_count} incomplete calls; "
            f"{run.incomplete_calls_within_one_token_of_cap} were at 511-512+ output tokens "
            "under the 512-token cap."
        )
    lines.extend(
        [
            "",
            "This strongly associates the failures with output-budget exhaustion under the frozen "
            "reasoning-enabled configuration. It does not establish that Pro is intrinsically less "
            "capable than Flash; a larger-cap run would be a different, separately frozen "
            "experiment.",
            "",
            "## Semantic-audit decision",
            "",
            f"Only {report.semantic_audit_coverage.formed_intention_success_count}/24 frozen "
            "semantic-audit cases produced a successful Pro Stage A formed intention; "
            f"{report.semantic_audit_coverage.formed_intention_failure_count}/24 failed before "
            "there was an intention to judge. An additional Pro human-audit sheet is therefore "
            "not added: it would mainly duplicate the output-cap failure signal rather than "
            "measure semantic preservation. The preregistered 48-row Primary + Flash audit "
            "resumes unchanged.",
            "",
            "## Boundaries",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"


def _write_artifact(path: Path, payload: bytes) -> Literal["created", "unchanged"]:
    if path.exists():
        if path.read_bytes() != payload:
            raise DeepSeekProReportError(f"output differs from regeneration: {path}")
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


def build_deepseek_pro_report(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    runs_root: Path,
    ie4_report_path: Path,
    output_dir: Path,
    repository_root: Path,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Build the complete Pro diagnostic without making provider calls."""

    dataset, experiment = verify_test_guard(dataset_manifest_path, experiment_path, repository_root)
    maximum = float(experiment.verdict["budget_difference_max"])
    summaries: list[ProRunSummary] = []
    for arm in ("B2b", "B3"):
        arm_lower = arm.lower()
        run_dir = runs_root / arm_lower
        evaluation_dir = output_dir / "evaluation" / arm_lower
        evaluate_frozen_split(
            split=Split.TEST,
            cases_path=cases_path,
            predictions_path=run_dir / "predictions.jsonl",
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            output_dir=evaluation_dir,
            repository_root=repository_root,
        )
        summaries.append(
            _run_summary(
                arm=arm,
                run_dir=run_dir,
                evaluation_dir=evaluation_dir,
                repository_root=repository_root,
            )
        )
    b2b, b3 = summaries
    comparison_path = output_dir / "b2b-vs-b3-bootstrap.json"
    compare_frozen_split(
        split=Split.TEST,
        cases_path=cases_path,
        left_predictions_path=runs_root / "b2b/predictions.jsonl",
        right_predictions_path=runs_root / "b3/predictions.jsonl",
        left_id="deepseek_pro_robustness/B2b",
        right_id="deepseek_pro_robustness/B3",
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        output_path=comparison_path,
        repository_root=repository_root,
        iterations=iterations,
        seed=seed,
    )
    comparison: BootstrapComparisonArtifact = _read_model(
        comparison_path, BootstrapComparisonArtifact
    )
    budget = budget_comparison(
        role=ModelRole.WEAK_DECISION,
        b2b_total_tokens=b2b.total_tokens,
        b3_total_tokens=b3.total_tokens,
        maximum_allowed=maximum,
        usage_complete=True,
    )
    domains = comparison.required_domains
    evidence = ComparisonEvidence(
        required_cis_estimable=all(item.estimable for item in domains.values()),
        hem=_metric_interval(domains["full_hem"]),
        id_intent_macro_f1=_metric_interval(domains["gold_in_scope_intent_macro_f1"]),
        near_oos_recall=_metric_interval(domains["near_oos_recall"]),
        no_intent_recall=_metric_interval(domains["no_intent_recall"]),
        b2b_context_distractor_error_count=b2b.context_distractor_error_count,
        b3_context_distractor_error_count=b3.context_distractor_error_count,
        b2b_schema_invalid_count=b2b.schema_invalid_count,
        b3_schema_invalid_count=b3.schema_invalid_count,
        b2b_total_tokens=b2b.total_tokens,
        b3_total_tokens=b3.total_tokens,
        usage_complete=True,
        latency_complete=True,
        cost_complete=False,
    )
    report = DeepSeekProRobustnessReport(
        runs=summaries,
        comparison=ProComparisonSummary(
            required_domains=domains,
            budget=budget,
            diagnostic_verdict=decide(evidence),
            artifact=_source(comparison_path, repository_root),
        ),
        flash_reference=_flash_reference(pro_runs=summaries, ie4_report_path=ie4_report_path),
        semantic_audit_coverage=_semantic_audit_coverage(
            dataset=dataset,
            formed_intentions_path=runs_root / "b3/formed-intentions.jsonl",
        ),
        source_artifacts={
            "test": _source(cases_path, repository_root),
            "taxonomy": _source(taxonomy_path, repository_root),
            "dataset_freeze": _source(dataset_manifest_path, repository_root),
            "experiment_lock": _source(experiment_path, repository_root),
            "ie4_flash_reference": _source(ie4_report_path, repository_root),
        },
        limitations=[
            "This post-freeze diagnostic cannot replace the preregistered Flash weak role.",
            "The 512-token cap is held fixed for comparability but is poorly matched to "
            "Pro output.",
            "Provider pricing is not registered, so the diagnostic cost gate is incomplete.",
            "The synthetic balanced test set is not production traffic.",
            "No result changes Gemini primary authority or the existing IE4 global verdict.",
        ],
    )
    report_payload = _json_bytes(report.model_dump(mode="json"))
    markdown_payload = render_deepseek_pro_markdown(report).encode()
    report_status = _write_artifact(output_dir / REPORT_FILENAME, report_payload)
    markdown_status = _write_artifact(output_dir / REPORT_MARKDOWN_FILENAME, markdown_payload)
    return {
        "status": "created" if "created" in (report_status, markdown_status) else "unchanged",
        "provider_calls": 0,
        "conclusion": report.conclusion,
        "diagnostic_verdict": (
            report.comparison.diagnostic_verdict.verdict.value
            if report.comparison.diagnostic_verdict.verdict is not None
            else None
        ),
        "verdict_reasons": report.comparison.diagnostic_verdict.reasons,
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
    }
