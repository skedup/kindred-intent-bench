from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_deepseek_pro_lock_changes_only_registered_robustness_fields() -> None:
    root = repository_root()
    base = yaml.safe_load((root / "configs/kir-pilot-v2-ie3-experiment.yaml").read_text())
    pro = yaml.safe_load(
        (root / "configs/kir-pilot-v2-ie3-deepseek-pro-robustness.yaml").read_text()
    )
    normalized = deepcopy(pro)
    normalized["experiment_id"] = base["experiment_id"]
    normalized["models"]["selection_rationale"]["weak_decision"] = base["models"][
        "selection_rationale"
    ]["weak_decision"]
    normalized["models"]["weak_decision"]["model"] = base["models"]["weak_decision"]["model"]

    assert normalized == base
    assert pro["status"] == "frozen"
    assert pro["models"]["weak_decision"]["model"] == "deepseek-v4-pro"
    assert pro["models"]["weak_decision"]["verdict_authority"] is False


def test_deepseek_pro_readiness_binds_exact_model_and_config_hash() -> None:
    root = repository_root()
    config_path = root / "configs/kir-pilot-v2-ie3-deepseek-pro-robustness.yaml"
    readiness = json.loads(
        (root / "experiments/ie3-deepseek-pro-robustness/provider-readiness.json").read_text()
    )

    assert readiness["status"] == "passed"
    assert readiness["selected_roles"] == ["weak_decision"]
    assert (
        readiness["experiment_config_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert len(readiness["results"]) == 1
    result = readiness["results"][0]
    assert result["requested_model"] == "deepseek-v4-pro"
    assert result["reported_model"] == "deepseek-v4-pro"
    assert result["identity_match"] is True
    assert result["structured_output_contract_match"] is True
    assert result["usage_contract_match"] is True
    assert readiness["secret_policy"] == (
        "environment_variable_name_only; no secret or raw response persisted"
    )
