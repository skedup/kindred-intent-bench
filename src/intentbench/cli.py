"""Command-line interface for contract validation and explicit live smoke."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import ParamSpec, TypeVar

import click

from intentbench import __version__
from intentbench.annotation import validate_annotation_artifacts
from intentbench.dataset import CandidateDatasetError, validate_candidate_artifact
from intentbench.freeze import FreezeGuardError, verify_test_guard
from intentbench.grounding import (
    GroundingContractError,
    load_grounding_snapshot,
    validate_grounding_alignment,
)
from intentbench.materialize import MaterializationContractError, materialize_adjudicated_cases
from intentbench.providers import (
    READINESS_ROLES,
    load_readiness_record,
    merge_provider_readiness,
    run_provider_readiness,
    write_readiness_record,
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
