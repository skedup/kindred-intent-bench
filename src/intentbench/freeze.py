"""Dataset and experiment double-freeze guard."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from intentbench.schemas import StrictModel


class FreezeGuardError(RuntimeError):
    """A formal test run was requested without a matching double freeze."""


class ArtifactDigest(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetFreezeManifest(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["draft", "frozen"]
    dataset_version: str
    taxonomy_version: str
    artifacts: dict[str, ArtifactDigest] = Field(default_factory=dict)
    semantic_audit_case_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def frozen_manifest_is_complete(self) -> DatasetFreezeManifest:
        if self.status == "frozen":
            required = {"taxonomy", "dev", "test", "split"}
            missing = required - set(self.artifacts)
            if missing:
                raise ValueError(f"frozen dataset manifest lacks {sorted(missing)}")
        return self


class ExperimentLock(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["draft", "frozen"]
    experiment_id: str
    dataset_version: str
    taxonomy_version: str
    dataset_freeze_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    artifacts: dict[str, ArtifactDigest] = Field(default_factory=dict)
    models: dict[str, Any]
    b0: dict[str, Any] = Field(default_factory=dict)
    b1: dict[str, Any]
    llm: dict[str, Any] = Field(default_factory=dict)
    bootstrap: dict[str, Any]
    verdict: dict[str, Any]

    @model_validator(mode="after")
    def frozen_lock_is_complete(self) -> ExperimentLock:
        if self.status == "frozen" and self.dataset_freeze_sha256 is None:
            raise ValueError("frozen experiment requires dataset_freeze_sha256")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_serialized(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    artifact = (root / relative_path).resolve()
    if not artifact.is_relative_to(root):
        raise FreezeGuardError(f"artifact escapes repository root: {relative_path}")
    return artifact


def _verify_artifacts(root: Path, artifacts: dict[str, ArtifactDigest], owner: str) -> None:
    for name, expected in sorted(artifacts.items()):
        artifact = _resolve_artifact(root, expected.path)
        if not artifact.is_file():
            raise FreezeGuardError(f"{owner} artifact missing: {name} ({expected.path})")
        actual = sha256_file(artifact)
        if actual != expected.sha256:
            raise FreezeGuardError(
                f"{owner} artifact hash mismatch: {name} expected={expected.sha256} actual={actual}"
            )


def verify_test_guard(
    dataset_manifest_path: Path, experiment_lock_path: Path, repository_root: Path
) -> tuple[DatasetFreezeManifest, ExperimentLock]:
    if not dataset_manifest_path.is_file():
        raise FreezeGuardError(f"dataset freeze manifest missing: {dataset_manifest_path}")
    if not experiment_lock_path.is_file():
        raise FreezeGuardError(f"experiment lock missing: {experiment_lock_path}")
    dataset = DatasetFreezeManifest.model_validate(_load_serialized(dataset_manifest_path))
    experiment = ExperimentLock.model_validate(_load_serialized(experiment_lock_path))
    if dataset.status != "frozen":
        raise FreezeGuardError("dataset manifest is not frozen")
    if experiment.status != "frozen":
        raise FreezeGuardError("experiment lock is not frozen")
    if dataset.dataset_version != experiment.dataset_version:
        raise FreezeGuardError("dataset version differs between the two freeze layers")
    if dataset.taxonomy_version != experiment.taxonomy_version:
        raise FreezeGuardError("taxonomy version differs between the two freeze layers")
    actual_dataset_manifest_hash = sha256_file(dataset_manifest_path)
    if experiment.dataset_freeze_sha256 != actual_dataset_manifest_hash:
        raise FreezeGuardError("experiment lock references a different dataset freeze manifest")
    _verify_artifacts(repository_root, dataset.artifacts, "dataset freeze")
    _verify_artifacts(repository_root, experiment.artifacts, "experiment lock")
    return dataset, experiment
