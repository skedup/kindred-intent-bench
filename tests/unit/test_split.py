from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest

from intentbench.bootstrap import validate_split_integrity
from intentbench.freeze import DatasetFreezeManifest, sha256_file
from intentbench.schemas import Case, Decision
from intentbench.split import (
    DEV_DECISION_TARGETS,
    DEV_INTENT_TARGET,
    INTENT_NAMES,
    DatasetSplitError,
    SplitManifest,
    split_and_freeze_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _seed_repository(root: Path) -> dict[str, Path]:
    cases = root / "data/kir-pilot-v2/adjudicated/cases.jsonl"
    receipt = root / "data/kir-pilot-v2/adjudicated/materialization-receipt.json"
    taxonomy = root / "configs/kindred-activity-intents-v2.yaml"
    for source, target in (
        (PROJECT_ROOT / "data/kir-pilot-v2/adjudicated/cases.jsonl", cases),
        (
            PROJECT_ROOT / "data/kir-pilot-v2/adjudicated/materialization-receipt.json",
            receipt,
        ),
        (PROJECT_ROOT / "configs/kindred-activity-intents-v2.yaml", taxonomy),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return {
        "cases": cases,
        "receipt": receipt,
        "taxonomy": taxonomy,
        "output": root / "data/kir-pilot-v2",
    }


def _load_cases(path: Path) -> list[Case]:
    return [Case.model_validate_json(line) for line in path.read_text().splitlines()]


def test_split_freeze_is_group_safe_exact_and_idempotent(tmp_path: Path) -> None:
    paths = _seed_repository(tmp_path)
    arguments = {
        "cases_path": paths["cases"],
        "receipt_path": paths["receipt"],
        "taxonomy_path": paths["taxonomy"],
        "output_dir": paths["output"],
        "repository_root": tmp_path,
    }
    created = split_and_freeze_dataset(**arguments)
    unchanged = split_and_freeze_dataset(**arguments)
    assert created["status"] == "created"
    assert unchanged == {**created, "status": "unchanged"}
    assert created["test_predictions_generated"] is False

    dev = _load_cases(paths["output"] / "dev.jsonl")
    test = _load_cases(paths["output"] / "test.jsonl")
    assert (len(dev), len(test)) == (48, 112)
    validate_split_integrity([*dev, *test])
    assert Counter(case.gold.decision for case in dev) == Counter(DEV_DECISION_TARGETS)
    assert Counter(
        case.gold.target_intent for case in dev if case.gold.decision is Decision.IN_SCOPE
    ) == Counter({intent: DEV_INTENT_TARGET for intent in INTENT_NAMES})
    assert sum("near_oos" in case.tags for case in dev) == 5

    manifest = SplitManifest.model_validate_json(
        (paths["output"] / "split-manifest.json").read_text()
    )
    assert manifest.required_test_cluster_counts == {
        "full": 77,
        "gold_in_scope": 43,
        "near_oos": 11,
        "no_intent": 11,
    }
    assert all(
        abs(manifest.diagnostic_dev_counts[feature] - target) <= 1
        for feature, target in manifest.diagnostic_target_counts.items()
    )
    audit_cases = [case for case in test if case.id in manifest.semantic_audit.case_ids]
    assert len(audit_cases) == 24
    assert Counter(case.gold.decision for case in audit_cases) == Counter(
        {
            Decision.IN_SCOPE: 12,
            Decision.OOS: 4,
            Decision.NO_INTENT: 4,
            Decision.AMBIGUOUS: 4,
        }
    )
    assert set(INTENT_NAMES) <= {
        case.gold.target_intent for case in audit_cases if case.gold.target_intent is not None
    }
    assert len({case.bootstrap_cluster_id for case in audit_cases}) == 24

    freeze_path = paths["output"] / "freeze-manifest.json"
    freeze = DatasetFreezeManifest.model_validate_json(freeze_path.read_text())
    assert freeze.status == "frozen"
    assert freeze.semantic_audit_case_ids == manifest.semantic_audit.case_ids
    for digest in freeze.artifacts.values():
        assert sha256_file(tmp_path / digest.path) == digest.sha256
    assert not list(tmp_path.rglob("*prediction*"))


def test_split_freeze_rejects_materialized_gold_hash_drift(tmp_path: Path) -> None:
    paths = _seed_repository(tmp_path)
    paths["cases"].write_bytes(paths["cases"].read_bytes() + b"\n")
    with pytest.raises(DatasetSplitError, match="differs from its materialization receipt"):
        split_and_freeze_dataset(
            cases_path=paths["cases"],
            receipt_path=paths["receipt"],
            taxonomy_path=paths["taxonomy"],
            output_dir=paths["output"],
            repository_root=tmp_path,
        )
