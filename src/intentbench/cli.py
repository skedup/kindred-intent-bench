"""Command-line interface for contract validation and explicit live smoke."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import ParamSpec, TypeVar

import click

from intentbench import __version__
from intentbench.adapters.base import ProviderError, StructuredGenerationClient
from intentbench.adapters.deepseek import DeepSeekChatClient
from intentbench.adapters.google import GoogleGenerativeLanguageClient
from intentbench.adapters.openai import OpenAIResponsesClient
from intentbench.annotation import validate_annotation_artifacts
from intentbench.baselines import BaselineRunError, run_b0_dev, run_b1_dev, run_b2a_dev
from intentbench.dataset import CandidateDatasetError, validate_candidate_artifact
from intentbench.deepseek_pro import DeepSeekProReportError, build_deepseek_pro_report
from intentbench.embedding import EmbeddingCacheError
from intentbench.formal_runs import FormalRunError, run_b0_test, run_b1_test, run_b2a_test
from intentbench.freeze import FreezeGuardError, verify_test_guard
from intentbench.generation_cache import GenerationCacheError
from intentbench.grounding import (
    GroundingContractError,
    load_grounding_snapshot,
    validate_grounding_alignment,
)
from intentbench.ie4 import IE4ReportError, build_ie4_pilot_report
from intentbench.materialize import MaterializationContractError, materialize_adjudicated_cases
from intentbench.matrix import MatrixValidationError, build_seven_run_matrix
from intentbench.providers import (
    READINESS_ROLES,
    load_readiness_inputs,
    load_readiness_record,
    merge_provider_readiness,
    run_provider_readiness,
    write_readiness_record,
)
from intentbench.reporting import (
    EvaluationArtifactError,
    compare_frozen_dev,
    evaluate_frozen_dev,
)
from intentbench.review import (
    BlindReviewItem,
    IndependentReviewLabel,
    ReviewContractError,
    ReviewPass,
    ReviewWorksheetEntry,
    build_adjudication_packet,
    import_adjudication_workbook,
    import_review_workbook,
    init_review_workspace,
    inspect_review_workspace,
    load_blind_batch,
    load_response_batch,
    load_review_manifest,
    review_horizon_values,
    validate_review_label,
    write_response_batch,
)
from intentbench.schemas import Decision, Horizon, Slots, Taxonomy
from intentbench.split import DatasetSplitError, split_and_freeze_dataset
from intentbench.taxonomy import load_taxonomy
from intentbench.two_stage import (
    LLM_ROLE_NAMES,
    prepare_one_stage_dev_cache,
    prepare_one_stage_test_cache,
    run_b2b_dev,
    run_b2b_test,
    run_b3_dev,
    run_b3_test,
)

P = ParamSpec("P")
R = TypeVar("R")

REVIEW_DECISION_SHORTCUTS: tuple[tuple[str, Decision], ...] = (
    ("1", Decision.IN_SCOPE),
    ("2", Decision.OOS),
    ("3", Decision.NO_INTENT),
    ("4", Decision.AMBIGUOUS),
)


def _prompt_review_decision() -> Decision:
    """Accept quick numeric input without leaking it into stored artifacts."""
    shortcut_map = dict(REVIEW_DECISION_SHORTCUTS)
    textual_values = tuple(value.value for value in Decision)
    click.echo(
        "decision options: "
        + ", ".join(
            f"{shortcut}={decision.value}" for shortcut, decision in REVIEW_DECISION_SHORTCUTS
        )
    )
    raw_value = click.prompt(
        "decision",
        type=click.Choice((*shortcut_map, *textual_values), case_sensitive=False),
    )
    if raw_value in shortcut_map:
        return shortcut_map[raw_value]
    return Decision(raw_value)


def _prompt_review_entry(
    *,
    item: BlindReviewItem,
    entry: ReviewWorksheetEntry,
    taxonomy: Taxonomy,
    reviewer_id: str,
) -> ReviewWorksheetEntry:
    """Collect and validate one response before replacing the stored row."""
    click.echo(f"\n[{item.review_item_id}]")
    click.echo(json.dumps(item.context.model_dump(mode="json"), ensure_ascii=False, indent=2))
    decision = _prompt_review_decision()
    target_intent: str | None = None
    if decision is Decision.IN_SCOPE:
        target_intent = click.prompt(
            "target_intent",
            type=click.Choice(tuple(intent.name for intent in taxonomy.intents)),
        )
    evidence_quote = click.prompt("evidence_quote", type=str)
    desired_experience = (
        click.prompt("desired_experience (blank=null)", default="", show_default=False) or None
    )
    object_value = click.prompt("object (blank=null)", default="", show_default=False) or None
    horizon = click.prompt("horizon", type=click.Choice(review_horizon_values()))
    review_note = click.prompt("review_note (blank=null)", default="", show_default=False) or None
    label = IndependentReviewLabel(
        decision=decision,
        target_intent=target_intent,
        evidence_quote=evidence_quote,
        slots=Slots(
            desired_experience=desired_experience,
            object=object_value,
            horizon=Horizon(horizon),
        ),
    )
    validate_review_label(label, item, taxonomy)
    payload = entry.model_dump(mode="json")
    payload.update(
        {
            "reviewer_id": reviewer_id,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "label": label.model_dump(mode="json"),
            "review_note": review_note,
        }
    )
    return ReviewWorksheetEntry.model_validate(payload)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
def main() -> None:
    """Offline-first Kindred intent recognition workbench."""


@main.group("taxonomy")
def taxonomy_group() -> None:
    """Validate versioned intent taxonomies."""


@taxonomy_group.command("validate")
@click.argument("path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
def taxonomy_validate(path: Path) -> None:
    taxonomy = load_taxonomy(path)
    click.echo(
        json.dumps(
            {
                "status": "valid",
                "taxonomy_version": taxonomy.taxonomy_version,
                "intent_count": len(taxonomy.intents),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@main.group("grounding")
def grounding_group() -> None:
    """Validate a static snapshot of the authoritative Kindred runtime catalog."""


@grounding_group.command("validate")
@click.option(
    "--snapshot",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-grounding-v1.yaml"),
    show_default=True,
)
@click.option(
    "--taxonomy",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-intents-v2.yaml"),
    show_default=True,
)
def grounding_validate(snapshot: Path, taxonomy: Path) -> None:
    try:
        report = validate_grounding_alignment(
            load_grounding_snapshot(snapshot),
            load_taxonomy(taxonomy),
        )
    except (OSError, GroundingContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@main.group("annotations")
def annotations_group() -> None:
    """Validate the non-dataset IE1.1 annotation contract."""


@annotations_group.command("validate")
@click.option(
    "--pack",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kir-pilot-v2-annotation-pack.yaml"),
    show_default=True,
)
@click.option(
    "--taxonomy",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-intents-v2.yaml"),
    show_default=True,
)
@click.option(
    "--template",
    "template_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("templates/kir-pilot-v1-case.template.json"),
    show_default=True,
)
def annotations_validate(pack: Path, taxonomy: Path, template_path: Path) -> None:
    try:
        report = validate_annotation_artifacts(pack, taxonomy, template_path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@main.group("dataset")
def dataset_group() -> None:
    """Validate pre-split candidate data and later frozen datasets."""


def _review_artifact_options(function: Callable[P, R]) -> Callable[P, R]:
    function = click.option(
        "--cases",
        "cases_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("data/kir-pilot-v2/candidates.jsonl"),
        show_default=True,
    )(function)
    function = click.option(
        "--taxonomy",
        "taxonomy_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("configs/kindred-activity-intents-v2.yaml"),
        show_default=True,
    )(function)
    return function


@dataset_group.group("candidates")
def dataset_candidates_group() -> None:
    """Inspect IE1.2 candidates that still require human review."""


@dataset_candidates_group.command("validate")
@click.option(
    "--cases",
    "cases_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/candidates.jsonl"),
    show_default=True,
)
@click.option(
    "--taxonomy",
    "taxonomy_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-intents-v2.yaml"),
    show_default=True,
)
def dataset_candidates_validate(cases_path: Path, taxonomy_path: Path) -> None:
    try:
        report = validate_candidate_artifact(cases_path, taxonomy_path)
    except (OSError, CandidateDatasetError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_group.group("gold")
def dataset_gold_group() -> None:
    """Materialize human-reviewed, adjudicated pre-split Gold."""


@dataset_gold_group.command("materialize")
@_review_artifact_options
@click.option(
    "--review-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
@click.option(
    "--packet-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial/adjudication"),
    show_default=True,
)
@click.option(
    "--policy-resolution",
    "policy_resolution_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial/adjudication/policy-resolution.json"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/kir-pilot-v2/adjudicated"),
    show_default=True,
)
def dataset_gold_materialize(
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    packet_dir: Path,
    policy_resolution_path: Path,
    output_dir: Path,
) -> None:
    """Apply review, adjudication, and policy artifacts without assigning a split."""

    try:
        report = materialize_adjudicated_cases(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
            packet_dir=packet_dir,
            policy_resolution_path=policy_resolution_path,
            output_dir=output_dir,
        )
    except (OSError, MaterializationContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_gold_group.command("split-freeze")
@click.option(
    "--cases",
    "cases_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/adjudicated/cases.jsonl"),
    show_default=True,
)
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/adjudicated/materialization-receipt.json"),
    show_default=True,
)
@click.option(
    "--taxonomy",
    "taxonomy_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-intents-v2.yaml"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/kir-pilot-v2"),
    show_default=True,
)
@click.option(
    "--repository-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("."),
    show_default=True,
)
def dataset_gold_split_freeze(
    cases_path: Path,
    receipt_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> None:
    """Create the deterministic group-safe formal split and freeze data artifacts."""

    try:
        report = split_and_freeze_dataset(
            cases_path=cases_path,
            receipt_path=receipt_path,
            taxonomy_path=taxonomy_path,
            output_dir=output_dir,
            repository_root=repository_root,
        )
    except (OSError, DatasetSplitError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_group.group("reviews")
def dataset_reviews_group() -> None:
    """Create and complete label-blind IE1.2 human-review passes."""


@dataset_reviews_group.command("init")
@_review_artifact_options
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
@click.option(
    "--review-pass",
    type=click.Choice(("initial", "blind_retest")),
    default="initial",
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=40, show_default=True)
def dataset_reviews_init(
    cases_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    review_pass: ReviewPass,
    batch_size: int,
) -> None:
    try:
        report = init_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            output_dir=output_dir,
            review_pass=review_pass,
            batch_size=batch_size,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_reviews_group.command("status")
@_review_artifact_options
@click.option(
    "--review-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
def dataset_reviews_status(cases_path: Path, taxonomy_path: Path, review_dir: Path) -> None:
    try:
        report = inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_reviews_group.command("validate")
@_review_artifact_options
@click.option(
    "--review-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
def dataset_reviews_validate(cases_path: Path, taxonomy_path: Path, review_dir: Path) -> None:
    try:
        report = inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
            require_complete=True,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_reviews_group.command("import-workbook")
@_review_artifact_options
@click.option(
    "--review-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
@click.option(
    "--workbook",
    "workbook_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the entire workbook without writing response JSONL or an import receipt.",
)
def dataset_reviews_import_workbook(
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    workbook_path: Path,
    dry_run: bool,
) -> None:
    """Import a hash-bound completed .xlsx review submission."""

    try:
        report = import_review_workbook(
            workbook_path=workbook_path,
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
            dry_run=dry_run,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_reviews_group.command("build-adjudication-packet")
@_review_artifact_options
@click.option(
    "--review-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial/adjudication"),
    show_default=True,
)
@click.option("--agreement-audit-count", type=click.IntRange(min=1), default=15, show_default=True)
def dataset_reviews_build_adjudication_packet(
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    output_dir: Path,
    agreement_audit_count: int,
) -> None:
    """Build the disclosed disagreement set plus a stratified agreement audit."""

    try:
        report = build_adjudication_packet(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
            output_dir=output_dir,
            agreement_audit_count=agreement_audit_count,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_reviews_group.command("import-adjudication-workbook")
@_review_artifact_options
@click.option(
    "--grounding",
    "grounding_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-grounding-v1.yaml"),
    show_default=True,
)
@click.option(
    "--packet-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial/adjudication"),
    show_default=True,
)
@click.option(
    "--workbook",
    "workbook_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option("--adjudicator-id", required=True, help="Stable non-secret adjudicator identifier.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the entire workbook without writing resolutions or an import receipt.",
)
def dataset_reviews_import_adjudication_workbook(
    cases_path: Path,
    taxonomy_path: Path,
    grounding_path: Path,
    packet_dir: Path,
    workbook_path: Path,
    adjudicator_id: str,
    dry_run: bool,
) -> None:
    """Import a hash-bound completed adjudication workbook."""

    try:
        report = import_adjudication_workbook(
            workbook_path=workbook_path,
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            grounding_path=grounding_path,
            packet_dir=packet_dir,
            adjudicator_id=adjudicator_id,
            dry_run=dry_run,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_reviews_group.command("run")
@_review_artifact_options
@click.option(
    "--review-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
@click.option("--batch", "batch_number", type=click.IntRange(min=1), required=True)
@click.option("--reviewer-id", required=True, help="Stable non-secret reviewer identifier.")
@click.option("--limit", type=click.IntRange(min=1), default=None)
def dataset_reviews_run(
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    batch_number: int,
    reviewer_id: str,
    limit: int | None,
) -> None:
    try:
        inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
        )
        manifest = load_review_manifest(review_dir)
        if batch_number > manifest.batch_count:
            raise ReviewContractError(f"batch must be between 1 and {manifest.batch_count}")
        taxonomy = load_taxonomy(taxonomy_path)
        items = load_blind_batch(review_dir, batch_number)
        entries = load_response_batch(review_dir, batch_number)
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    recorded = 0
    for index, (item, entry) in enumerate(zip(items, entries, strict=True)):
        if entry.is_complete:
            continue
        if limit is not None and recorded >= limit:
            break

        try:
            entries[index] = _prompt_review_entry(
                item=item,
                entry=entry,
                taxonomy=taxonomy,
                reviewer_id=reviewer_id,
            )
            write_response_batch(review_dir, batch_number, entries)
        except (ReviewContractError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        recorded += 1

    try:
        report = inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    report["recorded_this_run"] = recorded
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataset_reviews_group.command("revise")
@_review_artifact_options
@click.option(
    "--review-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/kir-pilot-v2/reviews/initial"),
    show_default=True,
)
@click.option("--review-item-id", required=True)
@click.option("--reviewer-id", required=True, help="Must match the original reviewer.")
def dataset_reviews_revise(
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    review_item_id: str,
    reviewer_id: str,
) -> None:
    try:
        inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
        )
        manifest = load_review_manifest(review_dir)
        taxonomy = load_taxonomy(taxonomy_path)
        match: tuple[int, int, BlindReviewItem, ReviewWorksheetEntry] | None = None
        for batch_number in range(1, manifest.batch_count + 1):
            items = load_blind_batch(review_dir, batch_number)
            entries = load_response_batch(review_dir, batch_number)
            for index, (item, entry) in enumerate(zip(items, entries, strict=True)):
                if item.review_item_id == review_item_id:
                    match = (batch_number, index, item, entry)
                    break
            if match is not None:
                break
        if match is None:
            raise ReviewContractError(f"unknown review item: {review_item_id}")
        batch_number, index, item, entry = match
        if not entry.is_complete:
            raise ReviewContractError(f"{review_item_id} is pending; use reviews run")
        if entry.reviewer_id != reviewer_id:
            raise ReviewContractError(
                f"{review_item_id} belongs to reviewer {entry.reviewer_id!r}, not {reviewer_id!r}"
            )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    assert entry.label is not None
    click.echo("Current stored response:")
    click.echo(
        json.dumps(
            {
                "label": entry.label.model_dump(mode="json"),
                "review_note": entry.review_note,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    click.confirm("Replace this response?", abort=True)
    try:
        revised_entry = _prompt_review_entry(
            item=item,
            entry=entry,
            taxonomy=taxonomy,
            reviewer_id=reviewer_id,
        )
        entries = load_response_batch(review_dir, batch_number)
        entries[index] = revised_entry
        write_response_batch(review_dir, batch_number, entries)
        report = inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
        )
    except (OSError, ReviewContractError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    report["revised_item_id"] = review_item_id
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@main.group("evaluate")
def evaluate_group() -> None:
    """Generate deterministic artifacts from cached frozen-dev predictions."""


def _frozen_dev_options(function: Callable[P, R]) -> Callable[P, R]:
    function = click.option(
        "--cases",
        "cases_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("data/kir-pilot-v2/dev.jsonl"),
        show_default=True,
    )(function)
    function = click.option(
        "--taxonomy",
        "taxonomy_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("configs/kindred-activity-intents-v2.yaml"),
        show_default=True,
    )(function)
    function = click.option(
        "--dataset-manifest",
        "dataset_manifest_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("data/kir-pilot-v2/freeze-manifest.json"),
        show_default=True,
    )(function)
    function = click.option(
        "--repository-root",
        type=click.Path(path_type=Path, exists=True, file_okay=False),
        default=Path("."),
        show_default=True,
    )(function)
    return function


@evaluate_group.command("dev")
@_frozen_dev_options
@click.option(
    "--predictions",
    "predictions_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
def evaluate_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    predictions_path: Path,
    output_dir: Path,
) -> None:
    """Write metrics, normalized confusion, and all deterministic dev badcases."""

    try:
        report = evaluate_frozen_dev(
            cases_path=cases_path,
            predictions_path=predictions_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            output_dir=output_dir,
            repository_root=repository_root,
        )
    except (OSError, EvaluationArtifactError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@evaluate_group.command("compare-dev")
@_frozen_dev_options
@click.option(
    "--left-predictions",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--right-predictions",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option("--left-id", required=True)
@click.option("--right-id", required=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
@click.option("--iterations", type=click.IntRange(min=1), default=10_000, show_default=True)
@click.option("--seed", type=int, default=20_260_820, show_default=True)
def evaluate_compare_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    left_predictions: Path,
    right_predictions: Path,
    left_id: str,
    right_id: str,
    output_path: Path,
    iterations: int,
    seed: int,
) -> None:
    """Write registered right-minus-left paired cluster-bootstrap dev results."""

    try:
        report = compare_frozen_dev(
            cases_path=cases_path,
            left_predictions_path=left_predictions,
            right_predictions_path=right_predictions,
            left_id=left_id,
            right_id=right_id,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            output_path=output_path,
            repository_root=repository_root,
            iterations=iterations,
            seed=seed,
        )
    except (OSError, EvaluationArtifactError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@main.group("baseline")
def baseline_group() -> None:
    """Run registered B0/B1 baselines over frozen dev only."""


def _baseline_dev_options(function: Callable[P, R]) -> Callable[P, R]:
    function = click.option(
        "--experiment",
        "experiment_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("configs/kir-pilot-v2-experiment.yaml"),
        show_default=True,
    )(function)
    return _frozen_dev_options(function)


@baseline_group.command("b0-dev")
@_baseline_dev_options
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
def baseline_b0_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    output_dir: Path,
) -> None:
    """Generate deterministic rule predictions and a unified dev manifest."""

    try:
        report = run_b0_dev(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            output_dir=output_dir,
            repository_root=repository_root,
        )
    except (OSError, BaselineRunError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@baseline_group.command("b1-dev")
@_baseline_dev_options
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path(".provider-cache/b1-embeddings-v2"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
def baseline_b1_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Embed frozen dev, calibrate the registered grid, and persist B1 artifacts."""

    try:
        models, _fixture = load_readiness_inputs(
            experiment_path,
            Path("tests/fixtures/provider-smoke.json"),
        )
        spec = models.embedding
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise BaselineRunError(f"missing embedding credential environment: {spec.api_key_env}")
        with GoogleGenerativeLanguageClient(
            api_key=api_key,
            base_url=spec.base_url,
            timeout_seconds=spec.timeout_seconds,
        ) as client:
            report = run_b1_dev(
                client=client,
                cache_dir=cache_dir,
                cases_path=cases_path,
                taxonomy_path=taxonomy_path,
                dataset_manifest_path=dataset_manifest_path,
                experiment_path=experiment_path,
                output_dir=output_dir,
                repository_root=repository_root,
            )
    except (OSError, BaselineRunError, EmbeddingCacheError, ProviderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@baseline_group.command("b2a-dev")
@_baseline_dev_options
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path(".provider-cache/b2a-primary-v1"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
def baseline_b2a_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Run the registered primary one-stage/one-call baseline over frozen dev."""

    try:
        models, _fixture = load_readiness_inputs(
            experiment_path,
            Path("tests/fixtures/provider-smoke.json"),
        )
        spec = models.primary_decision
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise BaselineRunError(f"missing primary credential environment: {spec.api_key_env}")
        with GoogleGenerativeLanguageClient(
            api_key=api_key,
            base_url=spec.base_url,
            timeout_seconds=spec.timeout_seconds,
        ) as client:
            report = run_b2a_dev(
                client=client,
                cache_dir=cache_dir,
                cases_path=cases_path,
                taxonomy_path=taxonomy_path,
                dataset_manifest_path=dataset_manifest_path,
                experiment_path=experiment_path,
                output_dir=output_dir,
                repository_root=repository_root,
            )
    except (OSError, BaselineRunError, GenerationCacheError, ProviderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@main.group("ie3")
def ie3_group() -> None:
    """Run registered IE3 dev selection and frozen-test experiments."""


def _ie3_role_option(function: Callable[P, R]) -> Callable[P, R]:
    return click.option(
        "--role",
        type=click.Choice(LLM_ROLE_NAMES, case_sensitive=True),
        required=True,
    )(function)


def _ie3_dev_options(function: Callable[P, R]) -> Callable[P, R]:
    function = click.option(
        "--experiment",
        "experiment_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("configs/kir-pilot-v2-ie3-experiment.yaml"),
        show_default=True,
    )(function)
    return _frozen_dev_options(function)


def _ie3_client(experiment_path: Path, role: str) -> tuple[StructuredGenerationClient, str]:
    models, _fixture = load_readiness_inputs(
        experiment_path,
        Path("tests/fixtures/provider-smoke.json"),
    )
    spec = {
        "primary_decision": models.primary_decision,
        "weak_decision": models.weak_decision,
        "cross_provider_reference": models.cross_provider_reference,
    }[role]
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        raise BaselineRunError(f"missing {role} credential environment: {spec.api_key_env}")
    if spec.provider == "google_generative_language":
        return (
            GoogleGenerativeLanguageClient(
                api_key=api_key,
                base_url=spec.base_url,
                timeout_seconds=spec.timeout_seconds,
            ),
            spec.model,
        )
    if spec.provider == "deepseek":
        return (
            DeepSeekChatClient(
                api_key=api_key,
                base_url=spec.base_url,
                timeout_seconds=spec.timeout_seconds,
            ),
            spec.model,
        )
    return (
        OpenAIResponsesClient(
            api_key=api_key,
            base_url=spec.base_url,
            timeout_seconds=spec.timeout_seconds,
        ),
        spec.model,
    )


@ie3_group.command("one-stage-cache-dev")
@_ie3_dev_options
@_ie3_role_option
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
def ie3_one_stage_cache_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
) -> None:
    """Populate B2b call-1 cache without creating a secondary B2a scoring arm."""

    client, _model = _ie3_client(experiment_path, role)
    try:
        report = prepare_one_stage_dev_cache(
            client=client,
            role=role,  # type: ignore[arg-type]
            cache_dir=cache_dir,
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            repository_root=repository_root,
        )
    except (OSError, BaselineRunError, GenerationCacheError, ProviderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _run_ie3_arm(
    *,
    arm: str,
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    client, _model = _ie3_client(experiment_path, role)
    runner = run_b2b_dev if arm == "B2b" else run_b3_dev
    try:
        report = runner(
            client=client,
            role=role,  # type: ignore[arg-type]
            cache_dir=cache_dir,
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            output_dir=output_dir,
            repository_root=repository_root,
        )
    except (OSError, BaselineRunError, GenerationCacheError, ProviderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _ie3_run_options(function: Callable[P, R]) -> Callable[P, R]:
    function = click.option(
        "--output-dir",
        type=click.Path(path_type=Path, file_okay=False),
        required=True,
    )(function)
    function = click.option(
        "--cache-dir",
        type=click.Path(path_type=Path, file_okay=False),
        required=True,
    )(function)
    function = _ie3_role_option(function)
    return _ie3_dev_options(function)


@ie3_group.command("b2b-dev")
@_ie3_run_options
def ie3_b2b_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Run the registered same-channel verifier over frozen dev."""

    _run_ie3_arm(
        arm="B2b",
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
        experiment_path=experiment_path,
        role=role,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


@ie3_group.command("b3-dev")
@_ie3_run_options
def ie3_b3_dev(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Run taxonomy-free evidence recovery followed by Activity grounding over dev."""

    _run_ie3_arm(
        arm="B3",
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
        experiment_path=experiment_path,
        role=role,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


def _ie3_test_options(function: Callable[P, R]) -> Callable[P, R]:
    function = click.option(
        "--experiment",
        "experiment_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("configs/kir-pilot-v2-ie3-experiment.yaml"),
        show_default=True,
    )(function)
    function = click.option(
        "--dev-cases",
        "dev_cases_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("data/kir-pilot-v2/dev.jsonl"),
        show_default=True,
    )(function)
    function = click.option(
        "--cases",
        "cases_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("data/kir-pilot-v2/test.jsonl"),
        show_default=True,
    )(function)
    function = click.option(
        "--taxonomy",
        "taxonomy_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("configs/kindred-activity-intents-v2.yaml"),
        show_default=True,
    )(function)
    function = click.option(
        "--dataset-manifest",
        "dataset_manifest_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("data/kir-pilot-v2/freeze-manifest.json"),
        show_default=True,
    )(function)
    return click.option(
        "--repository-root",
        type=click.Path(path_type=Path, exists=True, file_okay=False),
        default=Path("."),
        show_default=True,
    )(function)


def _ie3_test_run_options(function: Callable[P, R]) -> Callable[P, R]:
    function = click.option(
        "--output-dir",
        type=click.Path(path_type=Path, file_okay=False),
        required=True,
    )(function)
    function = click.option(
        "--cache-dir",
        type=click.Path(path_type=Path, file_okay=False),
        required=True,
    )(function)
    function = _ie3_role_option(function)
    return _ie3_test_options(function)


@ie3_group.command("one-stage-cache-test")
@_ie3_test_options
@_ie3_role_option
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
def ie3_one_stage_cache_test(
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
) -> None:
    """Populate frozen-test B2b call-1 cache without a secondary scoring arm."""

    client: StructuredGenerationClient | None = None
    try:
        verify_test_guard(dataset_manifest_path, experiment_path, repository_root)
        client, _model = _ie3_client(experiment_path, role)
        report = prepare_one_stage_test_cache(
            client=client,
            role=role,  # type: ignore[arg-type]
            cache_dir=cache_dir,
            cases_path=cases_path,
            dev_cases_path=dev_cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            repository_root=repository_root,
        )
    except (
        OSError,
        BaselineRunError,
        FreezeGuardError,
        GenerationCacheError,
        ProviderError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _run_ie3_test_arm(
    *,
    arm: str,
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    client: StructuredGenerationClient | None = None
    runner = run_b2b_test if arm == "B2b" else run_b3_test
    try:
        verify_test_guard(dataset_manifest_path, experiment_path, repository_root)
        client, _model = _ie3_client(experiment_path, role)
        report = runner(
            client=client,
            role=role,  # type: ignore[arg-type]
            cache_dir=cache_dir,
            cases_path=cases_path,
            dev_cases_path=dev_cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            output_dir=output_dir,
            repository_root=repository_root,
        )
    except (
        OSError,
        BaselineRunError,
        FreezeGuardError,
        GenerationCacheError,
        ProviderError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@ie3_group.command("b2b-test")
@_ie3_test_run_options
def ie3_b2b_test(
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Run the frozen-test same-channel verifier after both freeze guards pass."""

    _run_ie3_test_arm(
        arm="B2b",
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
        experiment_path=experiment_path,
        role=role,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


@ie3_group.command("b3-test")
@_ie3_test_run_options
def ie3_b3_test(
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    role: str,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Run frozen-test taxonomy-free evidence recovery and grounding."""

    _run_ie3_test_arm(
        arm="B3",
        cases_path=cases_path,
        dev_cases_path=dev_cases_path,
        taxonomy_path=taxonomy_path,
        dataset_manifest_path=dataset_manifest_path,
        repository_root=repository_root,
        experiment_path=experiment_path,
        role=role,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


@ie3_group.command("b0-test")
@_ie3_test_options
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
def ie3_b0_test(
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    output_dir: Path,
) -> None:
    """Run frozen lexical rules without reopening dev selection."""

    del dev_cases_path
    try:
        report = run_b0_test(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            output_dir=output_dir,
            repository_root=repository_root,
        )
    except (OSError, FormalRunError, FreezeGuardError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@ie3_group.command("b1-test")
@_ie3_test_options
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
def ie3_b1_test(
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Run frozen B1 prototypes and thresholds without test-time calibration."""

    try:
        verify_test_guard(dataset_manifest_path, experiment_path, repository_root)
        models, _fixture = load_readiness_inputs(
            experiment_path, Path("tests/fixtures/provider-smoke.json")
        )
        spec = models.embedding
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise FormalRunError(f"missing embedding credential: {spec.api_key_env}")
        with GoogleGenerativeLanguageClient(
            api_key=api_key,
            base_url=spec.base_url,
            timeout_seconds=spec.timeout_seconds,
        ) as client:
            report = run_b1_test(
                client=client,
                cache_dir=cache_dir,
                cases_path=cases_path,
                dev_cases_path=dev_cases_path,
                taxonomy_path=taxonomy_path,
                dataset_manifest_path=dataset_manifest_path,
                experiment_path=experiment_path,
                output_dir=output_dir,
                repository_root=repository_root,
            )
    except (
        OSError,
        EmbeddingCacheError,
        FormalRunError,
        FreezeGuardError,
        ProviderError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@ie3_group.command("b2a-test")
@_ie3_test_options
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
def ie3_b2a_test(
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Run the primary one-stage scoring arm and seed B2b call-1 cache."""

    client: StructuredGenerationClient | None = None
    try:
        verify_test_guard(dataset_manifest_path, experiment_path, repository_root)
        client, _model = _ie3_client(experiment_path, "primary_decision")
        report = run_b2a_test(
            client=client,
            cache_dir=cache_dir,
            cases_path=cases_path,
            dev_cases_path=dev_cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            output_dir=output_dir,
            repository_root=repository_root,
        )
    except (
        OSError,
        FormalRunError,
        FreezeGuardError,
        GenerationCacheError,
        ProviderError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@ie3_group.command("matrix-check")
@_ie3_test_options
@click.option(
    "--runs-root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True
)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def ie3_matrix_check(
    cases_path: Path,
    dev_cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    repository_root: Path,
    experiment_path: Path,
    runs_root: Path,
    output_path: Path,
) -> None:
    """Validate all seven formal LLM cells and per-model token budgets."""

    del dev_cases_path
    try:
        report = build_seven_run_matrix(
            runs_root=runs_root,
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            output_path=output_path,
            repository_root=repository_root,
        )
    except (OSError, FreezeGuardError, MatrixValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@main.group("ie4")
def ie4_group() -> None:
    """Generate the offline IE4 pilot report from frozen cached predictions."""


@ie4_group.command("report")
@click.option(
    "--cases",
    "cases_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/test.jsonl"),
    show_default=True,
)
@click.option(
    "--taxonomy",
    "taxonomy_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-intents-v2.yaml"),
    show_default=True,
)
@click.option(
    "--dataset-manifest",
    "dataset_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/freeze-manifest.json"),
    show_default=True,
)
@click.option(
    "--matrix",
    "matrix_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("experiments/ie3-openai-recovery/seven-run-matrix.json"),
    show_default=True,
)
@click.option(
    "--semantic-audit-workbook",
    "semantic_audit_workbook_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("outputs/2026-08-23-ie4-semantic-audit/semantic-audit.xlsx"),
    show_default=True,
)
@click.option(
    "--semantic-audit-process",
    "semantic_audit_process_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kir-pilot-v2-ie4-semantic-audit-process.yaml"),
    show_default=True,
)
@click.option(
    "--split-manifest",
    "split_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/split-manifest.json"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("experiments/ie4-pilot"),
    show_default=True,
)
@click.option(
    "--repository-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("."),
    show_default=True,
)
@click.option("--iterations", type=click.IntRange(min=1), default=10_000, show_default=True)
@click.option("--seed", type=int, default=20_260_820, show_default=True)
def ie4_report(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    matrix_path: Path,
    semantic_audit_workbook_path: Path,
    semantic_audit_process_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    repository_root: Path,
    iterations: int,
    seed: int,
) -> None:
    """Write evaluations, paired CIs, tri-state verdicts, and the pilot report."""

    try:
        report = build_ie4_pilot_report(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            matrix_path=matrix_path,
            semantic_audit_workbook_path=semantic_audit_workbook_path,
            semantic_audit_process_path=semantic_audit_process_path,
            split_manifest_path=split_manifest_path,
            output_dir=output_dir,
            repository_root=repository_root,
            iterations=iterations,
            seed=seed,
        )
    except (OSError, IE4ReportError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@ie4_group.command("deepseek-pro-report")
@click.option(
    "--cases",
    "cases_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/test.jsonl"),
    show_default=True,
)
@click.option(
    "--taxonomy",
    "taxonomy_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kindred-activity-intents-v2.yaml"),
    show_default=True,
)
@click.option(
    "--dataset-manifest",
    "dataset_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/kir-pilot-v2/freeze-manifest.json"),
    show_default=True,
)
@click.option(
    "--experiment",
    "experiment_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kir-pilot-v2-ie3-deepseek-pro-robustness.yaml"),
    show_default=True,
)
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("experiments/ie3-deepseek-pro-robustness/weak_decision"),
    show_default=True,
)
@click.option(
    "--ie4-report",
    "ie4_report_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("experiments/ie4-pilot/report.json"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("experiments/ie3-deepseek-pro-robustness"),
    show_default=True,
)
@click.option(
    "--repository-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("."),
    show_default=True,
)
@click.option("--iterations", type=click.IntRange(min=1), default=10_000, show_default=True)
@click.option("--seed", type=int, default=20_260_820, show_default=True)
def ie4_deepseek_pro_report(
    cases_path: Path,
    taxonomy_path: Path,
    dataset_manifest_path: Path,
    experiment_path: Path,
    runs_root: Path,
    ie4_report_path: Path,
    output_dir: Path,
    repository_root: Path,
    iterations: int,
    seed: int,
) -> None:
    """Build the post-freeze Pro diagnostic entirely from cached predictions."""

    try:
        report = build_deepseek_pro_report(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            dataset_manifest_path=dataset_manifest_path,
            experiment_path=experiment_path,
            runs_root=runs_root,
            ie4_report_path=ie4_report_path,
            output_dir=output_dir,
            repository_root=repository_root,
            iterations=iterations,
            seed=seed,
        )
    except (OSError, DeepSeekProReportError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@main.group("freeze")
def freeze_group() -> None:
    """Inspect the formal-test double-freeze guard."""


@freeze_group.command("verify")
@click.option(
    "--dataset-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--experiment-lock",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--repository-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path.cwd,
)
def freeze_verify(dataset_manifest: Path, experiment_lock: Path, repository_root: Path) -> None:
    try:
        dataset, experiment = verify_test_guard(dataset_manifest, experiment_lock, repository_root)
    except FreezeGuardError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "status": "verified",
                "dataset_version": dataset.dataset_version,
                "experiment_id": experiment.experiment_id,
            },
            sort_keys=True,
        )
    )


@main.group("providers")
def providers_group() -> None:
    """Run explicit synthetic-only Provider checks (networked)."""


@providers_group.command("check")
@click.option(
    "--fixture",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("tests/fixtures/provider-smoke.json"),
    show_default=True,
)
@click.option(
    "--config",
    "experiment_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kir-pilot-v1-experiment.yaml"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("configs/provider-readiness.json"),
    show_default=True,
)
@click.option(
    "--environment-label",
    default="manual-unspecified",
    show_default=True,
    help="Non-secret label describing where the manual smoke ran.",
)
@click.option(
    "--role",
    "selected_roles",
    type=click.Choice(READINESS_ROLES),
    multiple=True,
    help="Role to check; repeat for a partial, mergeable readiness record.",
)
def providers_check(
    fixture: Path,
    experiment_path: Path,
    output: Path,
    environment_label: str,
    selected_roles: tuple[str, ...],
) -> None:
    try:
        record = run_provider_readiness(
            experiment_path,
            fixture,
            environment_label=environment_label,
            selected_roles=selected_roles or None,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    write_readiness_record(record, output)
    click.echo(json.dumps(record, ensure_ascii=False, sort_keys=True))
    if record["status"] != "passed":
        raise click.ClickException("one or more Provider readiness checks failed")


@providers_group.command("merge")
@click.option(
    "--fixture",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("tests/fixtures/provider-smoke.json"),
    show_default=True,
)
@click.option(
    "--config",
    "experiment_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/kir-pilot-v1-experiment.yaml"),
    show_default=True,
)
@click.option(
    "--input",
    "input_paths",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    multiple=True,
    required=True,
    help="Partial readiness record; repeat until all configured roles are present.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("configs/provider-readiness.json"),
    show_default=True,
)
def providers_merge(
    fixture: Path,
    experiment_path: Path,
    input_paths: tuple[Path, ...],
    output: Path,
) -> None:
    try:
        record = merge_provider_readiness(
            [load_readiness_record(path) for path in input_paths],
            experiment_path=experiment_path,
            fixture_path=fixture,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    write_readiness_record(record, output)
    click.echo(json.dumps(record, ensure_ascii=False, sort_keys=True))
    if record["status"] != "passed":
        raise click.ClickException("merged Provider readiness contains failed checks")


if __name__ == "__main__":
    main()
