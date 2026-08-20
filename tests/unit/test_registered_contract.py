from __future__ import annotations

import json
from pathlib import Path

import yaml

from intentbench.bootstrap import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    MIN_CLUSTERS,
)
from intentbench.freeze import ExperimentLock
from intentbench.providers import load_readiness_inputs

EXPERIMENT_PATH = Path("configs/kir-pilot-v1-experiment.yaml")
FIXTURE_PATH = Path("tests/fixtures/provider-smoke.json")


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
    assert models.weak_decision.model == "gemini-3.5-flash"
    assert models.embedding.model == "gemini-embedding-001"
    assert models.primary_decision.timeout_seconds == 30.0
    assert fixture.fixture_id == "synthetic-intent-readiness-v1"

    readiness_text = Path("configs/provider-readiness.json").read_text(encoding="utf-8")
    readiness = json.loads(readiness_text)
    assert readiness["status"] == "passed"
    assert [result["requested_model"] for result in readiness["results"]] == [
        models.primary_decision.model,
        models.weak_decision.model,
        models.embedding.model,
    ]
    assert "AIza" not in readiness_text
    assert "GEMINI_API_KEY=" not in readiness_text
    embedding_result = readiness["results"][2]
    assert embedding_result["identity_evidence"] == "model_specific_request_endpoint"
    assert embedding_result["identity_reported_by_provider"] is False
