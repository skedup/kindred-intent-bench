from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from intentbench.annotation import DraftCase
from intentbench.cli import main
from intentbench.dataset import (
    CandidateDatasetError,
    load_candidate_cases,
    validate_candidate_artifact,
    validate_candidate_cases,
)
from intentbench.schemas import Case
from intentbench.taxonomy import load_taxonomy

ROOT = Path(__file__).parents[2]
CASES_PATH = ROOT / "data/kir-pilot-v1/candidates.jsonl"
TAXONOMY_PATH = ROOT / "configs/kindred-activity-intents-v1.yaml"


def candidate_payloads() -> list[dict[str, object]]:
    return [case.model_dump(mode="json") for case in load_candidate_cases(CASES_PATH)]


def parse_candidates(payloads: list[dict[str, object]]) -> list[DraftCase]:
    return [DraftCase.model_validate(payload) for payload in payloads]


def test_candidate_artifact_has_exact_ie12_distribution() -> None:
    report = validate_candidate_artifact(CASES_PATH, TAXONOMY_PATH)
    assert report == {
        "status": "valid_pending_human_review",
        "candidate_count": 160,
        "decision_counts": {
            "in_scope": 80,
            "oos": 32,
            "no_intent": 24,
            "ambiguous": 24,
        },
        "in_scope_intent_counts": {
            "create_picture": 10,
            "dine_out": 10,
            "eat_at_home": 10,
            "play_xiaohongshu": 10,
            "reach_out_to_user": 10,
            "rest": 10,
            "take_a_walk": 10,
            "visit_cultural_place": 10,
        },
        "oos_partition": {"far_oos": 16, "near_oos": 16},
        "required_tag_counts": {
            "brandless_xhs": 8,
            "context_distractor": 24,
            "hard_negative": 40,
            "multi_turn": 44,
            "rest_eat_confusion": 16,
            "weak_model_trap": 40,
        },
    }


def test_candidates_are_pending_drafts_not_formal_cases() -> None:
    cases = load_candidate_cases(CASES_PATH)
    assert all(case.source == "llm_assisted_pending_human_review" for case in cases)
    assert all(case.adjudication_status == "draft" for case in cases)
    with pytest.raises(ValidationError):
        Case.model_validate(cases[0].model_dump(mode="json"))


def test_committed_candidates_match_deterministic_builder(tmp_path: Path) -> None:
    generated_path = tmp_path / "candidates.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_ie12_candidates.py"),
            "--output",
            str(generated_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated_path.read_bytes() == CASES_PATH.read_bytes()


def test_candidate_validator_rejects_distribution_drift() -> None:
    cases = load_candidate_cases(CASES_PATH)
    with pytest.raises(CandidateDatasetError, match="exactly 160"):
        validate_candidate_cases(cases[:-1], load_taxonomy(TAXONOMY_PATH))


def test_candidate_validator_rejects_duplicate_context() -> None:
    payloads = candidate_payloads()
    source = payloads[0]
    duplicate = payloads[1]
    duplicate["context"] = source["context"]
    duplicate_gold = duplicate["gold"]
    source_gold = source["gold"]
    assert isinstance(duplicate_gold, dict)
    assert isinstance(source_gold, dict)
    duplicate_gold["evidence_quote"] = source_gold["evidence_quote"]
    with pytest.raises(CandidateDatasetError, match="contexts must be unique"):
        validate_candidate_cases(parse_candidates(payloads), load_taxonomy(TAXONOMY_PATH))


def test_candidate_validator_rejects_invalid_brandless_case() -> None:
    payloads = candidate_payloads()
    payload = next(item for item in payloads if "brandless_xhs" in item["tags"])
    context = payload["context"]
    assert isinstance(context, dict)
    context["state_summary"] = f"{context['state_summary']} 小红书"
    with pytest.raises(CandidateDatasetError, match="invalid brandless_xhs"):
        validate_candidate_cases(parse_candidates(payloads), load_taxonomy(TAXONOMY_PATH))


def test_candidate_validator_rejects_unlinked_near_oos() -> None:
    payloads = candidate_payloads()
    near_oos = next(item for item in payloads if "near_oos" in item["tags"])
    near_oos["contrast_group_id"] = "contrast-unlinked-near-oos"
    with pytest.raises(CandidateDatasetError, match="lacks contrast pair"):
        validate_candidate_cases(parse_candidates(payloads), load_taxonomy(TAXONOMY_PATH))


def test_candidate_validator_rejects_oversized_near_oos_contrast() -> None:
    payloads = candidate_payloads()
    near_oos = next(item for item in payloads if "near_oos" in item["tags"])
    unrelated = payloads[2]
    unrelated["contrast_group_id"] = near_oos["contrast_group_id"]
    with pytest.raises(CandidateDatasetError, match="exact pair"):
        validate_candidate_cases(parse_candidates(payloads), load_taxonomy(TAXONOMY_PATH))


def test_candidate_validator_rejects_paraphrase_label_drift() -> None:
    payloads = candidate_payloads()
    create_gold = payloads[8]["gold"]
    rest_gold = payloads[58]["gold"]
    assert isinstance(create_gold, dict)
    assert isinstance(rest_gold, dict)
    create_gold["target_intent"] = "rest"
    rest_gold["target_intent"] = "create_picture"
    with pytest.raises(CandidateDatasetError, match=r"paraphrase cluster.*crosses Gold labels"):
        validate_candidate_cases(parse_candidates(payloads), load_taxonomy(TAXONOMY_PATH))


def test_candidate_loader_rejects_formal_split_field(tmp_path: Path) -> None:
    payload = candidate_payloads()[0]
    payload["split"] = "dev"
    path = tmp_path / "candidate.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CandidateDatasetError, match="invalid candidate case"):
        load_candidate_cases(path)


def test_dataset_candidates_validate_cli() -> None:
    result = CliRunner().invoke(
        main,
        [
            "dataset",
            "candidates",
            "validate",
            "--cases",
            str(CASES_PATH),
            "--taxonomy",
            str(TAXONOMY_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "valid_pending_human_review"
