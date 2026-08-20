"""Command-line interface for contract validation and explicit live smoke."""

from __future__ import annotations

import json
from pathlib import Path

import click

from intentbench import __version__
from intentbench.freeze import FreezeGuardError, verify_test_guard
from intentbench.providers import (
    READINESS_ROLES,
    load_readiness_record,
    merge_provider_readiness,
    run_provider_readiness,
    write_readiness_record,
)
from intentbench.taxonomy import load_taxonomy


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
def providers_merge(input_paths: tuple[Path, ...], output: Path) -> None:
    try:
        record = merge_provider_readiness([load_readiness_record(path) for path in input_paths])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    write_readiness_record(record, output)
    click.echo(json.dumps(record, ensure_ascii=False, sort_keys=True))
    if record["status"] != "passed":
        raise click.ClickException("merged Provider readiness contains failed checks")


if __name__ == "__main__":
    main()
