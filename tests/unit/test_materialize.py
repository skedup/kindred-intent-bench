from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from intentbench.dataset import load_candidate_cases
from intentbench.materialize import (
    MaterializationContractError,
    load_adjudicated_cases,
    load_policy_resolution_plan,
    materialize_adjudicated_cases,
)

ROOT = Path(__file__).parents[2]
CASES_PATH = ROOT / "data/kir-pilot-v2/candidates.jsonl"
TAXONOMY_PATH = ROOT / "configs/kindred-activity-intents-v2.yaml"
REVIEW_DIR = ROOT / "data/kir-pilot-v2/reviews/initial"
PACKET_DIR = REVIEW_DIR / "adjudication"
POLICY_PATH = PACKET_DIR / "policy-resolution.json"


def materialize(output_dir: Path, *, policy_path: Path = POLICY_PATH) -> dict[str, object]:
    return materialize_adjudicated_cases(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=REVIEW_DIR,
        packet_dir=PACKET_DIR,
        policy_resolution_path=policy_path,
        output_dir=output_dir,
    )


def test_dual_channel_policy_is_bound_to_all_three_impacts() -> None:
    plan = load_policy_resolution_plan(POLICY_PATH)
    assert plan.strategy == "routing-plus-open-intent-v1"
    assert [resolution.candidate_id for resolution in plan.resolutions] == [
        "draft-kir-pilot-v2-0094",
        "draft-kir-pilot-v2-0131",
        "draft-kir-pilot-v2-0151",
    ]
    assert [resolution.open_intent_candidate.name for resolution in plan.resolutions] == [
        "slow_run",
        "learn_music",
        "go_out_unspecified",
    ]
    assert all(
        resolution.open_intent_candidate.routing_effect == "none" for resolution in plan.resolutions
    )


def test_materialization_preserves_routing_contract_and_adds_open_intents(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "adjudicated"
    report = materialize(output_dir)
    assert report["status"] == "materialized_adjudicated_pre_split_gold"
    assert report["decision_counts"] == {
        "in_scope": 80,
        "oos": 32,
        "no_intent": 24,
        "ambiguous": 24,
    }
    assert report["routing_label_source_counts"] == {
        "adjudicated_draft": 14,
        "blind_human_review": 143,
        "policy_dual_channel_draft": 3,
    }
    assert report["open_intent_candidate_ids"] == [
        "draft-kir-pilot-v2-0094",
        "draft-kir-pilot-v2-0131",
        "draft-kir-pilot-v2-0151",
    ]
    assert report["semantic_tag_audit"] == "valid"
    relationship_audit = report["relationship_audit"]
    assert isinstance(relationship_audit, dict)
    assert relationship_audit["status"] == "valid"
    assert relationship_audit["connected_component_count"] == 112

    candidates = load_candidate_cases(CASES_PATH)
    cases = load_adjudicated_cases(output_dir / "cases.jsonl")
    assert len(cases) == 160
    assert [case.gold for case in cases] == [candidate.gold for candidate in candidates]
    assert all(case.source == "llm_assisted_human_reviewed" for case in cases)
    assert all(case.adjudication_status == "adjudicated" for case in cases)
    assert Counter(case.review_provenance.supporting_fields_source for case in cases) == Counter(
        {"llm_assisted_draft": 160}
    )
    open_names = {
        case.id: case.open_intent_candidate.name
        for case in cases
        if case.open_intent_candidate is not None
    }
    assert open_names == {
        "draft-kir-pilot-v2-0094": "slow_run",
        "draft-kir-pilot-v2-0131": "learn_music",
        "draft-kir-pilot-v2-0151": "go_out_unspecified",
    }

    repeated = materialize(output_dir)
    assert repeated["status"] == "already_materialized"


def test_materialization_rejects_policy_plan_that_omits_an_impact(tmp_path: Path) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["resolutions"].pop()
    policy_path = tmp_path / "incomplete-policy.json"
    policy_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(MaterializationContractError, match="every imported policy impact"):
        materialize(tmp_path / "output", policy_path=policy_path)


def test_materialization_rejects_policy_plan_bound_to_other_resolutions(tmp_path: Path) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["source_resolution_artifact_sha256"] = "0" * 64
    policy_path = tmp_path / "stale-policy.json"
    policy_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(MaterializationContractError, match="bound to different resolutions"):
        materialize(tmp_path / "output", policy_path=policy_path)
