from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from intentbench.adapters.base import GenerationResult
from intentbench.b2a import (
    FewShotSelectionReceipt,
    LLMExperimentConfig,
    estimate_cost,
    few_shot_payload,
    normalize_cached_one_stage_prediction,
    one_stage_adapter_config_sha256,
    prediction_from_generation,
    render_b2a_prompt,
    validate_few_shot_cases,
    validate_few_shot_receipt,
)
from intentbench.freeze import sha256_file
from intentbench.generation_cache import CachedGeneration, GenerationRequest
from intentbench.providers import ReadinessModels
from intentbench.reporting import load_formal_cases
from intentbench.schemas import Case, Decision, PredictionStatus, TokenUsage
from intentbench.taxonomy import load_taxonomy


def _inputs() -> tuple[LLMExperimentConfig, list[Case]]:
    payload = yaml.safe_load(
        Path("configs/kir-pilot-v2-experiment.yaml").read_text(encoding="utf-8")
    )
    config = LLMExperimentConfig.model_validate(payload["llm"])
    cases = load_formal_cases(Path("data/kir-pilot-v2/dev.jsonl"))
    return config, cases


def _generation(value: dict[str, Any]) -> GenerationResult:
    return GenerationResult(
        value=value,
        requested_model="gemini-3.6-flash",
        reported_model="gemini-3.6-flash",
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        latency_ms=12.5,
        raw_response_sha256="a" * 64,
        usage_metadata={"promptTokenCount": 100, "totalTokenCount": 110},
        structured_output_mode="provider_json_schema",
    )


def test_registered_few_shots_are_balanced_cluster_distinct_and_minimal() -> None:
    config, cases = _inputs()
    selected = validate_few_shot_cases(cases, config.shared_few_shot)
    assert len(selected) == 12
    assert len({case.bootstrap_cluster_id for case in selected}) == 12
    assert {
        decision: sum(case.gold.decision is decision for case in selected) for decision in Decision
    } == {decision: 3 for decision in Decision}

    payload = few_shot_payload(selected)
    assert all(set(item) == {"context", "output"} for item in payload)
    assert all(
        set(item["output"])
        == {
            "decision",
            "predicted_intent",
            "candidate_intents",
            "slots",
            "reason_short",
        }
        for item in payload
        if isinstance(item["output"], dict)
    )


def test_registered_few_shot_receipt_binds_selection_and_frozen_dev() -> None:
    config, cases = _inputs()
    selected = validate_few_shot_cases(cases, config.shared_few_shot)
    receipt = FewShotSelectionReceipt.model_validate(
        yaml.safe_load(
            Path("configs/kir-pilot-v2-b2a-few-shot-selection.yaml").read_text(encoding="utf-8")
        )
    )
    validate_few_shot_receipt(
        receipt=receipt,
        config=config.shared_few_shot,
        selected_cases=selected,
        dataset_version="kir-pilot-v2",
        dev_sha256=sha256_file(Path("data/kir-pilot-v2/dev.jsonl")),
    )


def test_prompt_rendering_exposes_no_case_or_review_metadata() -> None:
    config, cases = _inputs()
    selected = validate_few_shot_cases(cases, config.shared_few_shot)
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v2.yaml"))
    template = Path("prompts/b2a-one-stage/v2.txt").read_text(encoding="utf-8")
    rendered = render_b2a_prompt(
        template=template,
        taxonomy=taxonomy,
        few_shots=selected,
        context_json='{"conversation":[],"recent_activities":[],"state_summary":"现在没有安排"}',
    )
    assert "{{" not in rendered
    assert "kir-pilot-v2-" not in rendered
    assert "evidence_quote" not in rendered
    assert "review_provenance" not in rendered
    assert "annotation_note" not in rendered


def test_generation_conversion_enforces_prediction_truth_table() -> None:
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v2.yaml"))
    valid = _generation(
        {
            "decision": "in_scope",
            "predicted_intent": "take_a_walk",
            "candidate_intents": [],
            "slots": {
                "desired_experience": "晒太阳",
                "object": None,
                "horizon": "now",
            },
            "reason_short": "当前明确想出去走走",
        }
    )
    prediction = prediction_from_generation(
        case_id="kir-pilot-v2-0001", result=valid, taxonomy=taxonomy
    )
    assert prediction.status is PredictionStatus.SUCCESS
    assert prediction.decision is Decision.IN_SCOPE
    assert prediction.predicted_intent == "take_a_walk"

    invalid = _generation(
        {
            **valid.value,
            "decision": "oos",
            "predicted_intent": "take_a_walk",
        }
    )
    rejected = prediction_from_generation(
        case_id="kir-pilot-v2-0001", result=invalid, taxonomy=taxonomy
    )
    assert rejected.status is PredictionStatus.SCHEMA_INVALID
    assert rejected.decision is None
    assert rejected.error_type == "local_contract:ValidationError"


def test_cost_uses_registered_input_and_output_rates() -> None:
    config, _ = _inputs()
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000)
    assert estimate_cost(usage, config.pricing) == 4.5


def test_one_stage_adapter_identity_excludes_pricing_and_other_arm_metadata() -> None:
    payload = yaml.safe_load(
        Path("configs/kir-pilot-v2-experiment.yaml").read_text(encoding="utf-8")
    )
    config = LLMExperimentConfig.model_validate(payload["llm"])
    spec = ReadinessModels.model_validate(payload["models"]).primary_decision
    inputs = {
        "prompt_sha256": "1" * 64,
        "few_shot_sha256": "2" * 64,
        "schema_sha256": "3" * 64,
    }
    original = one_stage_adapter_config_sha256(config=config, spec=spec, **inputs)
    different_pricing = config.model_copy(
        update={"pricing": config.pricing.model_copy(update={"input_per_million_tokens": 999.0})}
    )
    unrelated_arm = spec.model_copy(update={"required_arms": ["B2a"]})
    assert (
        one_stage_adapter_config_sha256(config=different_pricing, spec=unrelated_arm, **inputs)
        == original
    )
    changed_request = spec.model_copy(
        update={"generation_parameters": {"thinkingConfig": {"thinkingLevel": "low"}}}
    )
    assert (
        one_stage_adapter_config_sha256(config=config, spec=changed_request, **inputs) != original
    )


def _cached(result: GenerationResult) -> CachedGeneration:
    contract = GenerationRequest(
        normalized_context='{"state_summary":"想散步"}',
        prompt="prompt",
        taxonomy_sha256="1" * 64,
        prompt_template_sha256="2" * 64,
        few_shot_payload_sha256="3" * 64,
        response_schema={"type": "object"},
        model="gemini-3.6-flash",
        adapter="google_generate_content_v1beta",
        adapter_revision="v2",
        structured_output_mode="provider_json_schema",
        max_output_tokens=512,
        generation_parameters={},
    ).contract()
    return CachedGeneration(
        result=result,
        call_contract=contract,
        origin="provider_call",
        source_cache_key=None,
        reported_model=result.reported_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        raw_response_sha256=result.raw_response_sha256,
        usage_metadata=result.usage_metadata,
        structured_output_mode=result.structured_output_mode,
        error_type=None,
        http_status=None,
        latency_ms=result.latency_ms,
        cache_key=contract.sha256,
        call_contract_sha256=contract.sha256,
        cache_hit=True,
    )


def test_shared_one_stage_normalizer_rejects_identity_mode_and_usage_drift() -> None:
    payload = yaml.safe_load(
        Path("configs/kir-pilot-v2-experiment.yaml").read_text(encoding="utf-8")
    )
    spec = ReadinessModels.model_validate(payload["models"]).primary_decision
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v2.yaml"))
    valid = _generation(
        {
            "decision": "in_scope",
            "predicted_intent": "take_a_walk",
            "candidate_intents": [],
            "slots": {"desired_experience": None, "object": None, "horizon": "now"},
            "reason_short": "当前明确想散步",
        }
    )
    wrong_model = GenerationResult(**{**valid.__dict__, "reported_model": "other-model"})
    wrong_mode = GenerationResult(**{**valid.__dict__, "structured_output_mode": "other"})
    wrong_usage = GenerationResult(**{**valid.__dict__, "usage_metadata": {}})
    assert (
        normalize_cached_one_stage_prediction(
            case_id="kir-pilot-v2-0001",
            cached=_cached(wrong_model),
            taxonomy=taxonomy,
            spec=spec,
        ).error_type
        == "model_identity_mismatch"
    )
    assert (
        normalize_cached_one_stage_prediction(
            case_id="kir-pilot-v2-0001",
            cached=_cached(wrong_mode),
            taxonomy=taxonomy,
            spec=spec,
        ).error_type
        == "structured_output_mode_mismatch"
    )
    assert (
        normalize_cached_one_stage_prediction(
            case_id="kir-pilot-v2-0001",
            cached=_cached(wrong_usage),
            taxonomy=taxonomy,
            spec=spec,
        ).error_type
        == "usage_contract_mismatch"
    )
