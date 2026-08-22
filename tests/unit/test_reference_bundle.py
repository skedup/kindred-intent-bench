from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from intentbench.b0 import serialize_context
from intentbench.b2a import (
    LLMExperimentConfig,
    build_one_stage_request,
    few_shot_payload_sha256,
    normalize_cached_one_stage_prediction,
    prediction_response_schema,
    render_b2a_prompt,
    validate_few_shot_cases,
)
from intentbench.baselines import B2ACacheProvenance, B2ACallRecord
from intentbench.freeze import ExperimentLock, sha256_file
from intentbench.generation_cache import GenerationCache, GenerationCacheRecord
from intentbench.providers import ReadinessModels
from intentbench.reporting import load_formal_cases, load_predictions
from intentbench.schemas import RunManifest
from intentbench.taxonomy import load_taxonomy

REFERENCE = Path("experiments/reference/kir-pilot-v2-ie2-final")


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_selected_b2a_reference_bundle_is_portable_and_cross_bound() -> None:
    run = REFERENCE / "b2a"
    manifest = RunManifest.model_validate_json(
        (run / "run-manifest.json").read_text(encoding="utf-8")
    )
    calls = [
        B2ACallRecord.model_validate_json(line)
        for line in (run / "b2a-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cache_records = [
        GenerationCacheRecord.model_validate_json(line)
        for line in (run / "b2a-cache.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    provenance = B2ACacheProvenance.model_validate_json(
        (run / "b2a-cache-provenance.json").read_text(encoding="utf-8")
    )
    predictions = load_predictions(run / "predictions.jsonl")
    dev_ids = {case.id for case in load_formal_cases(Path("data/kir-pilot-v2/dev.jsonl"))}

    assert len(calls) == len(cache_records) == len(predictions) == 48
    assert {prediction.case_id for prediction in predictions} == dev_ids
    assert [call.cache_key for call in calls] == [record.cache_key for record in cache_records]
    assert all(record.origin == "validated_contract_migration" for record in cache_records)
    assert all(record.source_cache_key is not None for record in cache_records)
    assert provenance.portable_cache_bundle_sha256 == sha256_file(run / "b2a-cache.jsonl")
    assert provenance.source_predictions_sha256 == sha256_file(run / "predictions.jsonl")
    assert [entry.destination_cache_key for entry in provenance.entries] == [
        call.cache_key for call in calls
    ]
    indexed = {prediction.case_id: prediction for prediction in predictions}
    assert all(
        call.prediction_sha256 == _sha256_json(indexed[call.case_id].model_dump(mode="json"))
        for call in calls
    )

    assert manifest.experiment_lock_sha256 == sha256_file(
        Path("configs/kir-pilot-v2-experiment.yaml")
    )
    implementations = manifest.metadata["implementation_artifacts"]
    assert isinstance(implementations, dict)
    assert all(sha256_file(Path(path)) == digest for path, digest in implementations.items())


def test_reference_evaluations_and_comparison_bind_versioned_predictions() -> None:
    for arm in ("b0", "b1", "b2a"):
        metrics = json.loads(
            (REFERENCE / arm / "evaluation" / "metrics.json").read_text(encoding="utf-8")
        )
        prediction_digest = metrics["source_artifacts"]["predictions"]["sha256"]
        assert prediction_digest == sha256_file(REFERENCE / arm / "predictions.jsonl")

    comparison = json.loads((REFERENCE / "b1-vs-b2a-bootstrap.json").read_text(encoding="utf-8"))
    assert comparison["source_artifacts"]["left_predictions"]["sha256"] == sha256_file(
        REFERENCE / "b1" / "predictions.jsonl"
    )
    assert comparison["source_artifacts"]["right_predictions"]["sha256"] == sha256_file(
        REFERENCE / "b2a" / "predictions.jsonl"
    )
    assert comparison["required_domains"]["full_hem"]["point"] == 0.22916666666666663


def test_portable_bundle_restores_exact_b2b_call1_without_provider(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        Path("configs/kir-pilot-v2-experiment.yaml").read_text(encoding="utf-8")
    )
    experiment = ExperimentLock.model_validate(payload)
    config = LLMExperimentConfig.model_validate(payload["llm"])
    spec = ReadinessModels.model_validate(payload["models"]).primary_decision
    cases = load_formal_cases(Path("data/kir-pilot-v2/dev.jsonl"))
    taxonomy_path = Path("configs/kindred-activity-intents-v2.yaml")
    taxonomy = load_taxonomy(taxonomy_path)
    few_shots = validate_few_shot_cases(cases, config.shared_few_shot)
    prompt_digest = experiment.artifacts[config.b2a.prompt_artifact]
    template = Path(prompt_digest.path).read_text(encoding="utf-8")
    schema = prediction_response_schema(taxonomy)
    expected = {
        prediction.case_id: prediction
        for prediction in load_predictions(REFERENCE / "b2a" / "predictions.jsonl")
    }
    records = [
        GenerationCacheRecord.model_validate_json(line)
        for line in (REFERENCE / "b2a" / "b2a-cache.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    restored = GenerationCache(tmp_path / "restored")
    assert restored.import_records(records) == 48

    for case in cases:
        context_json = serialize_context(case.context)
        prompt = render_b2a_prompt(
            template=template,
            taxonomy=taxonomy,
            few_shots=few_shots,
            context_json=context_json,
        )
        request = build_one_stage_request(
            context_json=context_json,
            prompt=prompt,
            taxonomy_sha256=sha256_file(taxonomy_path),
            prompt_template_sha256=prompt_digest.sha256,
            few_shot_sha256=few_shot_payload_sha256(few_shots),
            response_schema=schema,
            spec=spec,
            max_output_tokens=config.b2a.request_max_output_tokens,
        )
        cached = restored.require_existing(request=request)
        actual = normalize_cached_one_stage_prediction(
            case_id=case.id,
            cached=cached,
            taxonomy=taxonomy,
            spec=spec,
        )
        assert actual == expected[case.id]
