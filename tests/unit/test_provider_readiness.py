from __future__ import annotations

from copy import deepcopy

import pytest

from intentbench.providers import READINESS_ROLES, merge_provider_readiness


def partial(*roles: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "record_kind": "provider_readiness_partial",
        "status": "passed",
        "checked_at": "2026-08-21T00:00:00+00:00",
        "fixture_id": "fixture",
        "fixture_scope": "synthetic_dev_only",
        "fixture_sha256": "a" * 64,
        "experiment_config_sha256": "b" * 64,
        "model_authority": {"source": "kindred"},
        "required_roles": list(READINESS_ROLES),
        "selected_roles": list(roles),
        "secret_policy": "environment_variable_name_only",
        "results": [{"role": role, "status": "passed"} for role in roles],
    }


def test_merge_combines_multi_environment_records_in_canonical_order() -> None:
    merged = merge_provider_readiness(
        [
            partial("weak_decision", "cross_provider_reference"),
            partial("primary_decision", "embedding"),
        ]
    )
    assert merged["record_kind"] == "provider_readiness"
    assert merged["status"] == "passed"
    assert [result["role"] for result in merged["results"]] == list(READINESS_ROLES)


def test_merge_rejects_duplicate_or_missing_roles_and_hash_drift() -> None:
    with pytest.raises(ValueError, match="duplicate readiness role"):
        merge_provider_readiness([partial("primary_decision"), partial("primary_decision")])
    with pytest.raises(ValueError, match="roles incomplete"):
        merge_provider_readiness([partial("primary_decision")])

    drifted = deepcopy(partial("weak_decision", "cross_provider_reference"))
    drifted["experiment_config_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="experiment_config_sha256"):
        merge_provider_readiness([partial("primary_decision", "embedding"), drifted])
