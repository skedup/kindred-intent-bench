from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from intentbench.b0 import B0Config
from intentbench.b1 import B1Config
from intentbench.b2a import LLMExperimentConfig, validate_few_shot_cases
from intentbench.bootstrap import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    MIN_CLUSTERS,
)
from intentbench.freeze import ExperimentLock, sha256_file
from intentbench.providers import ReadinessModels, load_readiness_inputs
from intentbench.reporting import load_formal_cases
from intentbench.schemas import Decision

EXPERIMENT_PATH = Path("configs/kir-pilot-v1-experiment.yaml")
FIXTURE_PATH = Path("tests/fixtures/provider-smoke.json")
V2_EXPERIMENT_PATH = Path("configs/kir-pilot-v2-experiment.yaml")


def test_experiment_draft_registers_runtime_candidates_and_exact_statistics() -> None:
    payload = yaml.safe_load(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    lock = ExperimentLock.model_validate(payload)
    assert lock.status == "draft"
    assert lock.dataset_freeze_sha256 is None
    assert payload["bootstrap"] == {
        "unit": "bootstrap_cluster_id",
        "paired": True,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "percentile",
        "quantile_method": "type_7_linear_interpolation",
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "minimum_clusters_per_required_slice": MIN_CLUSTERS,
    }
    providers = {item["provider"] for item in payload["models"]["runtime_candidate_pool"]}
    assert providers == {"google", "x-ai", "deepseek", "openai"}


def test_selected_models_and_safe_readiness_record_are_consistent() -> None:
    models, fixture = load_readiness_inputs(EXPERIMENT_PATH, FIXTURE_PATH)
    assert models.primary_decision.model == "gemini-3.6-flash"
    assert models.weak_decision.model == "deepseek-v4-flash"
    assert models.cross_provider_reference.model == "gpt-5.6-luna"
    assert models.embedding.model == "gemini-embedding-001"
    assert models.primary_decision.timeout_seconds == 30.0
    assert models.primary_decision.verdict_authority is True
    assert models.primary_decision.required_arms == ["B2a", "B2b", "B3"]
    assert models.weak_decision.required_arms == ["B2b", "B3"]
    assert models.cross_provider_reference.required_arms == ["B2b", "B3"]
    assert models.weak_decision.verdict_authority is False
    assert models.cross_provider_reference.verdict_authority is False
    assert models.comparison_matrix["cross_provider_replication"] == {
        "roles": ["weak_decision", "cross_provider_reference"],
        "arms": ["B2b", "B3"],
        "comparison": "within_model_B3_minus_B2b",
        "pooled_score_or_verdict": False,
        "primary_model_switch_after_test": "forbidden",
    }
    assert fixture.fixture_id == "synthetic-intent-readiness-v1"

    readiness_text = Path("configs/provider-readiness.json").read_text(encoding="utf-8")
    readiness = json.loads(readiness_text)
    assert readiness["schema_version"] == 2
    assert readiness["status"] == "passed"
    assert readiness["experiment_config_sha256"] == sha256_file(EXPERIMENT_PATH)
    assert readiness["fixture_sha256"] == sha256_file(FIXTURE_PATH)
    assert readiness["selected_roles"] == [
        "primary_decision",
        "weak_decision",
        "cross_provider_reference",
        "embedding",
    ]
    assert [result["requested_model"] for result in readiness["results"]] == [
        models.primary_decision.model,
        models.weak_decision.model,
        models.cross_provider_reference.model,
        models.embedding.model,
    ]
    assert "AIza" not in readiness_text
    assert "GEMINI_API_KEY=" not in readiness_text
    embedding_result = readiness["results"][3]
    assert embedding_result["identity_evidence"] == "model_specific_request_endpoint"
    assert embedding_result["identity_reported_by_provider"] is False


def test_model_role_and_global_verdict_authority_cannot_drift_silently() -> None:
    payload = yaml.safe_load(EXPERIMENT_PATH.read_text(encoding="utf-8"))["models"]
    drifted = deepcopy(payload)
    drifted["weak_decision"]["verdict_authority"] = True
    with pytest.raises(ValidationError, match="provider/arm authority"):
        ReadinessModels.model_validate(drifted)

    drifted = deepcopy(payload)
    drifted["comparison_matrix"]["cross_provider_replication"]["pooled_score_or_verdict"] = True
    with pytest.raises(ValidationError, match="seven-run contract"):
        ReadinessModels.model_validate(drifted)


def test_v2_experiment_binds_frozen_dev_and_selected_b1_contract() -> None:
    payload = yaml.safe_load(V2_EXPERIMENT_PATH.read_text(encoding="utf-8"))
    lock = ExperimentLock.model_validate(payload)
    b0 = B0Config.model_validate(payload["b0"])
    b1 = B1Config.model_validate(payload["b1"])
    llm = LLMExperimentConfig.model_validate(payload["llm"])
    models = ReadinessModels.model_validate(payload["models"])
    assert lock.status == "draft"
    assert lock.dataset_version == "kir-pilot-v2"
    assert lock.dataset_freeze_sha256 == sha256_file(Path("data/kir-pilot-v2/freeze-manifest.json"))
    assert b0.gate_order == ["actionability", "oos", "ambiguity", "in_scope"]
    assert models.embedding.task_type == "SEMANTIC_SIMILARITY"
    assert models.embedding.output_dimensionality == 3072
    assert b1.selected_thresholds == {
        "tau_actionability": 0.0,
        "tau_oos_margin": 0.0,
        "tau_activity_min": 0.3,
        "tau_ambiguity": 0.02,
    }

    dev_cases = {case.id: case for case in load_formal_cases(Path("data/kir-pilot-v2/dev.jsonl"))}
    assert b1.selected_prototype_case_ids is not None
    all_ids = [case_id for values in b1.selected_prototype_case_ids.values() for case_id in values]
    assert set(all_ids) <= set(dev_cases)
    clusters = [dev_cases[case_id].bootstrap_cluster_id for case_id in all_ids]
    assert len(clusters) == len(set(clusters))

    few_shots = validate_few_shot_cases(list(dev_cases.values()), llm.shared_few_shot)
    assert len(few_shots) == 12
    assert {case.gold.decision for case in few_shots} == set(Decision)
    assert payload["artifacts"]["b2a_prompt"]["sha256"] == sha256_file(
        Path("prompts/b2a-one-stage/v2.txt")
    )
    assert payload["artifacts"]["b2a_few_shot_selection"]["sha256"] == sha256_file(
        Path("configs/kir-pilot-v2-b2a-few-shot-selection.yaml")
    )
    assert payload["artifacts"]["b2a_prompt_selection"]["sha256"] == sha256_file(
        Path("configs/kir-pilot-v2-b2a-prompt-selection.yaml")
    )
    assert models.primary_decision.adapter_revision == "v2"
    assert models.weak_decision.adapter_revision == "v2"
    assert models.cross_provider_reference.adapter_revision == "v2"
