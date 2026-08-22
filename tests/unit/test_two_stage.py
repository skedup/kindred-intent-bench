from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from intentbench.adapters.base import GenerationResult
from intentbench.baselines import BaselineRunError
from intentbench.freeze import FreezeGuardError
from intentbench.generation_cache import GenerationCacheError
from intentbench.reporting import load_predictions
from intentbench.two_stage import (
    CALL_1_PROVENANCE_FILENAME,
    CALLS_FILENAME,
    FORMED_INTENTIONS_FILENAME,
    LLMCallRecord,
    prepare_one_stage_dev_cache,
    run_b2b_dev,
    run_b2b_test,
    run_b3_dev,
    run_b3_test,
)


class TwoStageGenerationClient:
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
        if "你是意图证据抽取器" in prompt:
            value: dict[str, Any] = {
                "evidence_status": "none",
                "action": None,
                "object": None,
                "desired_experience": None,
                "qualifiers": [],
                "alternative_actions": [],
                "horizon": "unspecified",
                "reason_short": "没有当前行动证据",
            }
        else:
            value = {
                "decision": "no_intent",
                "predicted_intent": None,
                "candidate_intents": [],
                "slots": {
                    "desired_experience": None,
                    "object": None,
                    "horizon": "unspecified",
                },
                "reason_short": "没有当前行动",
            }
        return GenerationResult(
            value=value,
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
        raise AssertionError("complete cache rerun must not call the Provider")


def _arguments(tmp_path: Path, output_name: str) -> dict[str, object]:
    return {
        "role": "primary_decision",
        "cache_dir": tmp_path / "cache",
        "cases_path": Path("data/kir-pilot-v2/dev.jsonl"),
        "taxonomy_path": Path("configs/kindred-activity-intents-v2.yaml"),
        "dataset_manifest_path": Path("data/kir-pilot-v2/freeze-manifest.json"),
        "experiment_path": Path("configs/kir-pilot-v2-ie3-experiment.yaml"),
        "output_dir": tmp_path / output_name,
        "repository_root": Path("."),
    }


def test_b2b_requires_existing_call_1_and_never_falls_back_to_provider(tmp_path: Path) -> None:
    client = TwoStageGenerationClient()
    with pytest.raises(GenerationCacheError, match="required one-stage-compatible"):
        run_b2b_dev(client=client, **_arguments(tmp_path, "b2b"))
    assert client.calls == 0


def test_formal_test_runner_rejects_draft_experiment_before_any_provider_call(
    tmp_path: Path,
) -> None:
    client = TwoStageGenerationClient()
    with pytest.raises(FreezeGuardError, match="experiment lock is not frozen"):
        run_b2b_test(
            client=client,
            role="primary_decision",
            cache_dir=tmp_path / "cache",
            cases_path=Path("data/kir-pilot-v2/test.jsonl"),
            dev_cases_path=Path("data/kir-pilot-v2/dev.jsonl"),
            taxonomy_path=Path("configs/kindred-activity-intents-v2.yaml"),
            dataset_manifest_path=Path("data/kir-pilot-v2/freeze-manifest.json"),
            experiment_path=Path("configs/kir-pilot-v2-ie3-experiment.yaml"),
            output_dir=tmp_path / "test-run",
            repository_root=Path("."),
        )
    assert client.calls == 0


def test_formal_two_stage_runner_rejects_dirty_worktree_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = yaml.safe_load(Path("configs/kir-pilot-v2-ie3-experiment.yaml").read_text())
    payload["status"] = "frozen"
    frozen_experiment = tmp_path / "frozen-experiment.yaml"
    frozen_experiment.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr("intentbench.two_stage._git_identity", lambda _root: ("a" * 40, True))
    client = TwoStageGenerationClient()
    with pytest.raises(BaselineRunError, match="clean committed worktree"):
        run_b3_test(
            client=client,
            role="primary_decision",
            cache_dir=tmp_path / "cache",
            cases_path=Path("data/kir-pilot-v2/test.jsonl"),
            dev_cases_path=Path("data/kir-pilot-v2/dev.jsonl"),
            taxonomy_path=Path("configs/kindred-activity-intents-v2.yaml"),
            dataset_manifest_path=Path("data/kir-pilot-v2/freeze-manifest.json"),
            experiment_path=frozen_experiment,
            output_dir=tmp_path / "test-run",
            repository_root=Path("."),
        )
    assert client.calls == 0


def test_call_record_rejects_arm_stage_mismatch() -> None:
    with pytest.raises(ValidationError, match="arm/index contract"):
        LLMCallRecord(
            case_id="case-001",
            arm="B3",
            stage="call_1_one_stage",
            call_index=1,
            cache_key="a" * 64,
            call_contract_sha256="b" * 64,
            requested_model="model",
            reported_model="model",
            status="success",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            latency_ms=1.0,
            raw_response_sha256="c" * 64,
            error_type=None,
            cache_origin="provider_call",
            source_cache_key=None,
            one_stage_cache_compatible=False,
        )


def test_b2b_reuses_call_1_and_persists_two_call_provenance(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, "b2b")
    prep_arguments = {key: value for key, value in arguments.items() if key != "output_dir"}
    client = TwoStageGenerationClient()
    prepared = prepare_one_stage_dev_cache(client=client, **prep_arguments)
    assert prepared["provider_calls"] == 48
    assert prepared["scoring_arm_created"] is False

    report = run_b2b_dev(client=client, **arguments)
    calls = [
        json.loads(line) for line in (tmp_path / "b2b" / CALLS_FILENAME).read_text().splitlines()
    ]
    provenance = json.loads(
        (tmp_path / "b2b" / CALL_1_PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    assert report["prediction_count"] == 48
    assert report["provider_calls"] == 48
    assert len(calls) == 96
    assert len(provenance["entries"]) == 48
    assert all(call["one_stage_cache_compatible"] for call in calls[::2])
    assert all(not call["one_stage_cache_compatible"] for call in calls[1::2])

    rerun = run_b2b_dev(client=NoCallGenerationClient(), **arguments)
    assert rerun["status"] == "unchanged"
    assert rerun["provider_calls"] == 0
    assert rerun["generation_cache_hits"] == 96


def test_b3_persists_taxonomy_free_stage_a_log_and_cache_reruns(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, "b3")
    client = TwoStageGenerationClient()
    report = run_b3_dev(client=client, **arguments)
    predictions = load_predictions(tmp_path / "b3" / "predictions.jsonl")
    formed = [
        json.loads(line)
        for line in (tmp_path / "b3" / FORMED_INTENTIONS_FILENAME).read_text().splitlines()
    ]
    calls = (tmp_path / "b3" / CALLS_FILENAME).read_text().splitlines()
    assert report["prediction_count"] == 48
    assert report["provider_calls"] == 96
    assert len(predictions) == len(formed) == 48
    assert len(calls) == 96
    assert all(record["status"] == "success" for record in formed)
    assert all(record["formed_intention"]["evidence_status"] == "none" for record in formed)

    rerun = run_b3_dev(client=NoCallGenerationClient(), **arguments)
    assert rerun["status"] == "unchanged"
    assert rerun["provider_calls"] == 0
    assert rerun["generation_cache_hits"] == 96
