from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from intentbench.providers import READINESS_ROLES, merge_provider_readiness

EXPERIMENT_PATH = Path("configs/kir-pilot-v1-experiment.yaml")
FIXTURE_PATH = Path("tests/fixtures/provider-smoke.json")
READINESS_PATH = Path("configs/provider-readiness.json")


def partial(*roles: str) -> dict[str, Any]:
    full = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    results_by_role = {result["role"]: result for result in full["results"]}
    full["record_kind"] = "provider_readiness_partial"
    full["selected_roles"] = list(roles)
    full["results"] = [deepcopy(results_by_role[role]) for role in roles]
    return full


def merge(*records: dict[str, Any]) -> dict[str, Any]:
    return merge_provider_readiness(
        records,
        experiment_path=EXPERIMENT_PATH,
        fixture_path=FIXTURE_PATH,
    )


def test_merge_combines_multi_environment_records_in_canonical_order() -> None:
    merged = merge(
        partial("weak_decision", "cross_provider_reference"),
        partial("primary_decision", "embedding"),
    )
    assert merged["record_kind"] == "provider_readiness"
    assert merged["status"] == "passed"
    assert [result["role"] for result in merged["results"]] == list(READINESS_ROLES)


def test_merge_rejects_duplicate_or_missing_roles_and_hash_drift() -> None:
    with pytest.raises(ValueError, match="duplicate readiness role"):
        merge(partial("primary_decision"), partial("primary_decision"))
    with pytest.raises(ValueError, match="roles incomplete"):
        merge(partial("primary_decision"))

    drifted = deepcopy(partial("weak_decision", "cross_provider_reference"))
    drifted["experiment_config_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="current experiment_config_sha256"):
        merge(partial("primary_decision", "embedding"), drifted)


@pytest.mark.parametrize(
    ("role", "field", "replacement"),
    [
        ("weak_decision", "requested_model", "wrong-model"),
        ("cross_provider_reference", "verdict_authority", True),
        ("primary_decision", "structured_output_contract_match", False),
    ],
)
def test_merge_rejects_result_contract_drift(role: str, field: str, replacement: object) -> None:
    record = partial(role)
    record["results"][0][field] = replacement
    with pytest.raises(ValueError, match=field):
        merge(record, partial(*(item for item in READINESS_ROLES if item != role)))


def test_merge_rejects_stale_partials_after_current_inputs_change(tmp_path: Path) -> None:
    records = [
        partial("primary_decision", "embedding"),
        partial("weak_decision", "cross_provider_reference"),
    ]
    changed_config = tmp_path / "experiment.yaml"
    changed_config.write_text(EXPERIMENT_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current experiment_config_sha256"):
        merge_provider_readiness(
            records,
            experiment_path=changed_config,
            fixture_path=FIXTURE_PATH,
        )

    changed_fixture = tmp_path / "fixture.json"
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["embedding_text"] += "。"
    changed_fixture.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="current fixture_sha256"):
        merge_provider_readiness(
            records,
            experiment_path=EXPERIMENT_PATH,
            fixture_path=changed_fixture,
        )
