from pathlib import Path

import pytest
import yaml

from intentbench.grounding import (
    GroundingContractError,
    load_grounding_snapshot,
    validate_grounding_alignment,
)
from intentbench.schemas import Taxonomy
from intentbench.taxonomy import load_taxonomy


def test_runtime_grounding_snapshot_matches_taxonomy_catalog() -> None:
    snapshot = load_grounding_snapshot(Path("configs/kindred-activity-grounding-v1.yaml"))
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v2.yaml"))

    report = validate_grounding_alignment(snapshot, taxonomy)

    assert report["status"] == "valid"
    assert report["activity_count"] == 8
    assert report["action_count"] == 15
    xhs = next(activity for activity in snapshot.activities if activity.name == "play_xiaohongshu")
    assert xhs.uses == ["use_xhs", "compose", "draw", "publish_xhs"]


def test_runtime_grounding_rejects_taxonomy_catalog_drift() -> None:
    snapshot = load_grounding_snapshot(Path("configs/kindred-activity-grounding-v1.yaml"))
    taxonomy = load_taxonomy(Path("configs/kindred-activity-intents-v2.yaml"))
    payload = taxonomy.model_dump(mode="json")
    for intent in payload["intents"]:
        intent["sibling_intents"] = []
    payload["intents"] = payload["intents"][:-1]

    with pytest.raises(GroundingContractError, match="Activity mismatch"):
        validate_grounding_alignment(snapshot, Taxonomy.model_validate(payload))


def test_grounding_impact_resolves_every_near_oos_boundary() -> None:
    payload = yaml.safe_load(
        Path("data/kir-pilot-v1/grounding-impact.yaml").read_text(encoding="utf-8")
    )
    findings = payload["near_oos_findings"]

    assert len(findings) == 16
    assert len({finding["candidate_id"] for finding in findings}) == 16
    assert sum(finding["draft_resolution"] == "keep" for finding in findings) == 15
    assert payload["summary"]["product_policy_required"] == 0
