"""Static provenance contract for the Kindred runtime Activity catalog."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import Field, model_validator

from intentbench.schemas import StrictModel, Taxonomy

SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


class GroundingContractError(ValueError):
    """Raised when a runtime snapshot cannot support a frozen taxonomy."""


class RuntimeAuthority(StrictModel):
    authority_kind: Literal["installed_derived_runtime"]
    environment: str = Field(min_length=1)
    distribution_name: Literal["kindred"]
    distribution_version: str = Field(min_length=1)
    internal_source_commit: str = Field(pattern=COMMIT_PATTERN)
    private_overlay_sha256: str = Field(pattern=SHA256_PATTERN)
    activity_loader: Literal["kindred.activity.skill.list_registered_activities"]
    action_loader: Literal["kindred.activity.action.list_registered_actions"]


class GroundedActivity(StrictModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str = Field(min_length=1)
    uses: list[str] = Field(min_length=1)
    terminal_when: str = Field(min_length=1)
    duration_hint: Literal["short", "medium", "long"]
    produces: Literal["always", "likely", "optional"]
    manifest_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    skill_path: str = Field(min_length=1)
    skill_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_sources_and_uses(self) -> GroundedActivity:
        _validate_relative_package_path(self.manifest_path, suffix="manifest.yaml")
        _validate_relative_package_path(self.skill_path, suffix="SKILL.md")
        if len(self.uses) != len(set(self.uses)):
            raise ValueError(f"activity {self.name!r} contains duplicate actions")
        return self


class GroundedAction(StrictModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str | None = None
    applies_when: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    manifest_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    skill_path: str = Field(min_length=1)
    skill_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_sources_and_capabilities(self) -> GroundedAction:
        _validate_relative_package_path(self.manifest_path, suffix="manifest.yaml")
        _validate_relative_package_path(self.skill_path, suffix="SKILL.md")
        if self.capabilities != sorted(set(self.capabilities)):
            raise ValueError(f"action {self.name!r} capabilities must be sorted and unique")
        return self


class ActivityGroundingSnapshot(StrictModel):
    schema_version: Literal[1]
    grounding_version: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+-v[0-9]+$")
    captured_at: datetime
    authority: RuntimeAuthority
    activities: list[GroundedActivity] = Field(min_length=1)
    actions: list[GroundedAction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> ActivityGroundingSnapshot:
        activity_names = [activity.name for activity in self.activities]
        action_names = [action.name for action in self.actions]
        if activity_names != sorted(set(activity_names)):
            raise ValueError("grounded activities must be sorted and unique")
        if action_names != sorted(set(action_names)):
            raise ValueError("grounded actions must be sorted and unique")
        known_actions = set(action_names)
        for activity in self.activities:
            unknown = set(activity.uses) - known_actions
            if unknown:
                raise ValueError(
                    f"activity {activity.name!r} references unknown actions {sorted(unknown)}"
                )
        return self


def _validate_relative_package_path(value: str, *, suffix: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name != suffix:
        raise ValueError(f"grounding source must be a relative package path ending in {suffix}")


def load_grounding_snapshot(path: Path) -> ActivityGroundingSnapshot:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ActivityGroundingSnapshot.model_validate(payload)


def validate_grounding_alignment(
    snapshot: ActivityGroundingSnapshot,
    taxonomy: Taxonomy,
) -> dict[str, object]:
    grounded_names = tuple(activity.name for activity in snapshot.activities)
    taxonomy_names = tuple(intent.name for intent in taxonomy.intents)
    if grounded_names != taxonomy_names:
        missing = sorted(set(grounded_names) - set(taxonomy_names))
        extra = sorted(set(taxonomy_names) - set(grounded_names))
        raise GroundingContractError(
            f"taxonomy/runtime Activity mismatch: missing={missing}, extra={extra}"
        )
    return {
        "status": "valid",
        "grounding_version": snapshot.grounding_version,
        "taxonomy_version": taxonomy.taxonomy_version,
        "activity_count": len(snapshot.activities),
        "action_count": len(snapshot.actions),
        "internal_source_commit": snapshot.authority.internal_source_commit,
        "private_overlay_sha256": snapshot.authority.private_overlay_sha256,
    }


__all__ = [
    "ActivityGroundingSnapshot",
    "GroundingContractError",
    "load_grounding_snapshot",
    "validate_grounding_alignment",
]
