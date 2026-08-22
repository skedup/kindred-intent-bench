from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from intentbench.adapters.base import EmbeddingResult, GenerationResult
from intentbench.baselines import (
    B1_SELECTION_FILENAME,
    B2A_CACHE_BUNDLE_FILENAME,
    B2A_CACHE_PROVENANCE_FILENAME,
    B2A_CALLS_FILENAME,
    MANIFEST_FILENAME,
    PREDICTIONS_FILENAME,
    run_b0_dev,
    run_b1_dev,
    run_b2a_dev,
)
from intentbench.reporting import load_predictions
from intentbench.schemas import RunManifest


class DeterministicEmbeddingClient:
    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self,
        *,
        model: str,
        text: str,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
    ) -> EmbeddingResult:
        self.calls += 1
        digest = hashlib.sha256(text.encode()).digest()
        dimension = output_dimensionality or 16
        vector = tuple((digest[index % len(digest)] + 1) / 256 for index in range(dimension))
        return EmbeddingResult(
            vector=vector,
            requested_model=model,
            latency_ms=1.0,
            raw_response_sha256=hashlib.sha256(b"raw:" + text.encode()).hexdigest(),
        )


class NoCallEmbeddingClient:
    def embed(
        self,
        *,
        model: str,
        text: str,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
    ) -> EmbeddingResult:
        raise AssertionError("complete cache rerun must not call the Provider")


class DeterministicGenerationClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
        generation_parameters: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self.calls += 1
        return GenerationResult(
            value={
                "decision": "no_intent",
                "predicted_intent": None,
                "candidate_intents": [],
                "slots": {
                    "desired_experience": None,
                    "object": None,
                    "horizon": "unspecified",
                },
                "reason_short": "没有当前行动",
            },
            requested_model=model,
            reported_model=model,
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            latency_ms=1.0,
            raw_response_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            usage_metadata={"promptTokenCount": 100, "totalTokenCount": 110},
            structured_output_mode="provider_json_schema",
        )


class NoCallGenerationClient:
    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
        generation_parameters: dict[str, Any] | None = None,
    ) -> GenerationResult:
        raise AssertionError("complete generation cache rerun must not call the Provider")


def compact_experiment(tmp_path: Path) -> Path:
    source = Path("configs/kir-pilot-v2-experiment.yaml")
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["models"]["embedding"]["expected_dimension"] = 16
    payload["models"]["embedding"]["output_dimensionality"] = 16
    payload["b1"]["selected_thresholds"] = None
    payload["b1"]["selected_prototype_case_ids"] = None
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def common_arguments(tmp_path: Path, experiment: Path) -> dict[str, Path]:
    return {
        "cases_path": Path("data/kir-pilot-v2/dev.jsonl"),
        "taxonomy_path": Path("configs/kindred-activity-intents-v2.yaml"),
        "dataset_manifest_path": Path("data/kir-pilot-v2/freeze-manifest.json"),
        "experiment_path": experiment,
        "repository_root": Path("."),
        "output_dir": tmp_path / "run",
    }


def test_b0_runner_writes_complete_unified_dev_contract(tmp_path: Path) -> None:
    report = run_b0_dev(**common_arguments(tmp_path, Path("configs/kir-pilot-v2-experiment.yaml")))
    predictions = load_predictions(tmp_path / "run" / PREDICTIONS_FILENAME)
    manifest = RunManifest.model_validate_json(
        (tmp_path / "run" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert report["prediction_count"] == 48
    assert len(predictions) == 48
    assert manifest.metadata["gold_fields_available_to_adapter"] is False
    assert manifest.metadata["provider_calls"] == 0


def test_b1_runner_selects_dev_only_prototypes_and_cache_rerun_is_offline(
    tmp_path: Path,
) -> None:
    experiment = compact_experiment(tmp_path)
    arguments = common_arguments(tmp_path, experiment)
    cache_dir = tmp_path / "cache"
    client = DeterministicEmbeddingClient()
    first = run_b1_dev(client=client, cache_dir=cache_dir, **arguments)
    assert first["prediction_count"] == 48
    assert first["provider_calls"] == client.calls
    assert client.calls > 0
    dev_ids = {
        prediction.case_id
        for prediction in load_predictions(tmp_path / "run" / PREDICTIONS_FILENAME)
    }
    prototype_ids = first["prototype_case_ids"]
    assert isinstance(prototype_ids, dict)
    assert set(prototype_ids["oos"]) <= dev_ids
    assert set(prototype_ids["no_intent"]) <= dev_ids
    assert (tmp_path / "run" / B1_SELECTION_FILENAME).is_file()

    second = run_b1_dev(client=NoCallEmbeddingClient(), cache_dir=cache_dir, **arguments)
    assert second["status"] == "unchanged"
    assert second["provider_calls"] == 0


def test_b2a_runner_has_one_call_per_case_and_cache_rerun_is_offline(tmp_path: Path) -> None:
    arguments = common_arguments(tmp_path, Path("configs/kir-pilot-v2-experiment.yaml"))
    cache_dir = tmp_path / "generation-cache"
    client = DeterministicGenerationClient()
    first = run_b2a_dev(client=client, cache_dir=cache_dir, **arguments)
    predictions = load_predictions(tmp_path / "run" / PREDICTIONS_FILENAME)
    call_lines = (tmp_path / "run" / B2A_CALLS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert first["prediction_count"] == 48
    assert first["provider_calls"] == 48
    assert client.calls == 48
    assert len(predictions) == 48
    assert len(call_lines) == 48
    assert first["case_id_held_out_case_count"] == 36
    assert first["cluster_held_out_case_count"] == 30
    cache_lines = (
        (tmp_path / "run" / B2A_CACHE_BUNDLE_FILENAME).read_text(encoding="utf-8").splitlines()
    )
    assert len(cache_lines) == 48
    assert (tmp_path / "run" / B2A_CACHE_PROVENANCE_FILENAME).is_file()

    second = run_b2a_dev(client=NoCallGenerationClient(), cache_dir=cache_dir, **arguments)
    assert second["status"] == "unchanged"
    assert second["generation_cache_hits"] == 48
    assert second["provider_calls"] == 0
