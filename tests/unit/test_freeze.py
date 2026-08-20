from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from intentbench.freeze import FreezeGuardError, sha256_file, verify_test_guard


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def frozen_pair(root: Path) -> tuple[Path, Path]:
    artifact_names = ("taxonomy", "dev", "test", "split")
    artifacts: dict[str, dict[str, str]] = {}
    for name in artifact_names:
        relative = f"data/{name}.txt"
        path = root / relative
        write(path, name)
        artifacts[name] = {"path": relative, "sha256": sha256_file(path)}
    dataset_path = root / "data/freeze-manifest.json"
    write(
        dataset_path,
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen",
                "dataset_version": "kir-pilot-v1",
                "taxonomy_version": "kindred-activity-intents-v1",
                "artifacts": artifacts,
                "semantic_audit_case_ids": [],
            }
        ),
    )
    lock_artifact = root / "uv.lock"
    write(lock_artifact, "frozen dependency graph")
    experiment_path = root / "configs/experiment.yaml"
    write(
        experiment_path,
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "frozen",
                "experiment_id": "kir-pilot-v1",
                "dataset_version": "kir-pilot-v1",
                "taxonomy_version": "kindred-activity-intents-v1",
                "dataset_freeze_sha256": sha256_file(dataset_path),
                "artifacts": {
                    "dependency_lock": {
                        "path": "uv.lock",
                        "sha256": sha256_file(lock_artifact),
                    }
                },
                "models": {},
                "b1": {},
                "bootstrap": {},
                "verdict": {},
            }
        ),
    )
    return dataset_path, experiment_path


def test_double_freeze_guard_accepts_matching_hashes(tmp_path: Path) -> None:
    dataset, experiment = frozen_pair(tmp_path)
    verified_dataset, verified_experiment = verify_test_guard(dataset, experiment, tmp_path)
    assert verified_dataset.status == "frozen"
    assert verified_experiment.status == "frozen"


def test_double_freeze_guard_rejects_hash_drift(tmp_path: Path) -> None:
    dataset, experiment = frozen_pair(tmp_path)
    write(tmp_path / "data/test.txt", "changed after freeze")
    with pytest.raises(FreezeGuardError, match="hash mismatch"):
        verify_test_guard(dataset, experiment, tmp_path)


def test_double_freeze_guard_rejects_draft(tmp_path: Path) -> None:
    dataset, experiment = frozen_pair(tmp_path)
    payload = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    payload["status"] = "draft"
    write(experiment, yaml.safe_dump(payload))
    with pytest.raises(FreezeGuardError, match="experiment lock is not frozen"):
        verify_test_guard(dataset, experiment, tmp_path)
