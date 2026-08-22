from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from intentbench.b2 import (
    FormedIntention,
    IE3LLMExperimentConfig,
    StageASupervisionReceipt,
    render_b3_stage_a_prompt,
    render_b3_stage_b_prompt,
    validate_stage_a_supervision,
)
from intentbench.b2a import validate_few_shot_cases
from intentbench.freeze import sha256_file
from intentbench.reporting import load_formal_cases
from intentbench.taxonomy import load_taxonomy

EXPERIMENT = Path("configs/kir-pilot-v2-ie3-experiment.yaml")
SUPERVISION = Path("configs/kir-pilot-v2-b3-stage-a-supervision-v2.yaml")


def _contract() -> tuple[IE3LLMExperimentConfig, StageASupervisionReceipt]:
    experiment = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    config = IE3LLMExperimentConfig.model_validate(experiment["llm"])
    receipt = StageASupervisionReceipt.model_validate(
        yaml.safe_load(SUPERVISION.read_text(encoding="utf-8"))
    )
    return config, receipt


def test_ie3_contract_aligns_calls_budgets_and_shared_supervision() -> None:
    config, receipt = _contract()
    cases = load_formal_cases(Path("data/kir-pilot-v2/dev.jsonl"))
    selected = validate_few_shot_cases(cases, config.shared_few_shot)
    validate_stage_a_supervision(
        receipt=receipt,
        selected_cases=selected,
        dataset_version="kir-pilot-v2",
        dev_sha256=sha256_file(Path("data/kir-pilot-v2/dev.jsonl")),
        selection_receipt_artifact=config.shared_few_shot.selection_receipt_artifact,
    )
    assert config.b2b.calls_per_case == config.b3.calls_per_case == 2
    assert (
        config.b2a.request_max_output_tokens
        == config.b2b.request_max_output_tokens_per_call
        == config.b3.request_max_output_tokens_per_call
        == 512
    )
    assert [example.case_id for example in receipt.examples] == config.shared_few_shot.case_ids


def test_formed_intention_truth_table_rejects_invented_positive_none() -> None:
    with pytest.raises(ValidationError, match="none evidence"):
        FormedIntention(
            evidence_status="none",
            action="去散步",
            object=None,
            desired_experience=None,
            qualifiers=[],
            alternative_actions=[],
            horizon="now",
            reason_short="没有计划",
        )
    with pytest.raises(ValidationError, match="at least two alternatives"):
        FormedIntention(
            evidence_status="ambiguous",
            action=None,
            object=None,
            desired_experience=None,
            qualifiers=[],
            alternative_actions=["去散步"],
            horizon="now",
            reason_short="还没有选择",
        )


def test_stage_a_prompt_has_no_taxonomy_and_stage_b_has_no_current_raw_context() -> None:
    config, receipt = _contract()
    cases = load_formal_cases(Path("data/kir-pilot-v2/dev.jsonl"))
    selected = validate_few_shot_cases(cases, config.shared_few_shot)
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v2.yaml"))
    stage_a = render_b3_stage_a_prompt(
        template=Path("prompts/b3-stage-a-evidence/v2.txt").read_text(encoding="utf-8"),
        few_shots=selected,
        receipt=receipt,
        context_json='{"state_summary":"当前唯一哨兵上下文"}',
    )
    assert taxonomy.taxonomy_version not in stage_a
    assert all(intent.name not in stage_a for intent in taxonomy.intents)
    assert "target_intent" not in stage_a
    assert '"decision"' not in stage_a

    stage_b = render_b3_stage_b_prompt(
        template=Path("prompts/b3-stage-b-grounding/v2.txt").read_text(encoding="utf-8"),
        taxonomy=taxonomy,
        few_shots=selected,
        receipt=receipt,
        formed_intention_json=(
            '{"action":"散步","alternative_actions":[],"desired_experience":null,'
            '"evidence_status":"clear","horizon":"now","object":null,'
            '"qualifiers":[],"reason_short":"明确想散步"}'
        ),
    )
    assert "当前唯一哨兵上下文" not in stage_b
    assert taxonomy.taxonomy_version in stage_b
    assert "take_a_walk" in stage_b


def test_prompt_selection_receipt_hashes_committed_dev_evidence() -> None:
    receipt = yaml.safe_load(
        Path("configs/kir-pilot-v2-ie3-prompt-selection.yaml").read_text(encoding="utf-8")
    )
    directory_prefix = {"B2b": "primary-b2b", "B3": "primary-b3"}
    for arm, arm_receipt in receipt["arms"].items():
        for candidate in arm_receipt["candidates"]:
            run_dir = Path("experiments/ie3-dev") / (
                f"{directory_prefix[arm]}-{candidate['version']}"
            )
            assert sha256_file(run_dir / "predictions.jsonl") == candidate["prediction_sha256"]
            assert sha256_file(run_dir / "run-manifest.json") == candidate["run_manifest_sha256"]
            manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
            assert manifest["total_usage"]["input_tokens"] == candidate["input_tokens"]
            assert manifest["total_usage"]["output_tokens"] == candidate["output_tokens"]
            assert manifest["estimated_cost"] == candidate["estimated_cost_usd"]
