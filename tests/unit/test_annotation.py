from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from intentbench.annotation import (
    AnnotationPack,
    CaseAuthoringTemplate,
    load_annotation_pack,
    load_case_authoring_template,
    validate_annotation_artifacts,
)
from intentbench.cli import main
from intentbench.schemas import Decision
from intentbench.taxonomy import load_taxonomy

ROOT = Path(__file__).parents[2]
PACK_PATH = ROOT / "configs/kir-pilot-v1-annotation-pack.yaml"
TAXONOMY_PATH = ROOT / "configs/kindred-activity-intents-v1.yaml"
TEMPLATE_PATH = ROOT / "templates/kir-pilot-v1-case.template.json"
GUIDELINE_PATH = ROOT / "docs/annotation-guideline.md"


def test_annotation_artifacts_are_complete_and_taxonomy_aligned() -> None:
    report = validate_annotation_artifacts(PACK_PATH, TAXONOMY_PATH, TEMPLATE_PATH)
    assert report == {
        "status": "valid",
        "pack_id": "kir-pilot-annotation-v1",
        "decision_example_count": 36,
        "intent_boundary_count": 8,
        "near_oos_pair_count": 8,
        "required_boundary_tags": [
            "brandless_xhs",
            "context_distractor",
            "multi_turn",
            "rest_eat_confusion",
        ],
        "template_case_id": "draft-template-001",
    }

    pack = load_annotation_pack(PACK_PATH)
    coverage = Counter(
        (example.boundary_decision, example.kind) for example in pack.decision_examples
    )
    assert coverage == Counter(
        {
            (decision, kind): 3
            for decision in Decision
            for kind in ("positive", "negative", "adjudication")
        }
    )


def test_guide_pack_and_template_cannot_masquerade_as_dataset_cases() -> None:
    pack = load_annotation_pack(PACK_PATH)
    example_fields = type(pack.decision_examples[0]).model_fields
    assert "id" not in example_fields
    assert "split" not in example_fields

    template = load_case_authoring_template(TEMPLATE_PATH)
    assert template.template_kind == "non_dataset_case_authoring_template"
    assert template.case.id.startswith("draft-template-")
    assert template.case.adjudication_status == "draft"
    assert template.case.split_group_id is None
    assert template.case.bootstrap_cluster_id is None


def test_pack_rejects_missing_decision_coverage() -> None:
    payload = load_annotation_pack(PACK_PATH).model_dump(mode="json")
    payload["decision_examples"].pop()
    with pytest.raises(ValidationError, match="at least 36 items"):
        AnnotationPack.model_validate(payload)


def test_pack_rejects_branded_text_in_brandless_fixture() -> None:
    payload = load_annotation_pack(PACK_PATH).model_dump(mode="json")
    example = next(item for item in payload["decision_examples"] if "brandless_xhs" in item["tags"])
    example["context"]["state_summary"] += " 我想打开小红书。"
    with pytest.raises(ValidationError, match="brandless_xhs"):
        AnnotationPack.model_validate(payload)


def test_cross_artifact_validation_rejects_unknown_intent(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(PACK_PATH.read_text(encoding="utf-8"))
    payload["decision_examples"][0]["gold"]["target_intent"] = "unknown_activity"
    mutated_pack = tmp_path / "annotation-pack.yaml"
    mutated_pack.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown target intent"):
        validate_annotation_artifacts(mutated_pack, TAXONOMY_PATH, TEMPLATE_PATH)


def test_template_must_remain_an_unassigned_draft() -> None:
    payload = load_case_authoring_template(TEMPLATE_PATH).model_dump(mode="json")
    payload["case"]["adjudication_status"] = "reviewed"
    with pytest.raises(ValidationError, match="must remain draft"):
        CaseAuthoringTemplate.model_validate(payload)


def test_cross_artifact_validation_checks_template_intent(
    tmp_path: Path,
) -> None:
    payload = load_case_authoring_template(TEMPLATE_PATH).model_dump(mode="json")
    payload["case"]["gold"]["target_intent"] = "unknown_activity"
    mutated_template = tmp_path / "case.template.json"
    mutated_template.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="template has unknown target intent"):
        validate_annotation_artifacts(PACK_PATH, TAXONOMY_PATH, mutated_template)


def test_guideline_and_machine_pack_have_the_same_named_inventory() -> None:
    pack = load_annotation_pack(PACK_PATH)
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    guideline = GUIDELINE_PATH.read_text(encoding="utf-8")

    required_names = {
        *(example.example_id for example in pack.decision_examples),
        *(boundary.intent for boundary in pack.intent_boundaries),
        *(pair.pair_id for pair in pack.near_oos_pairs),
        *(intent.name for intent in taxonomy.intents),
        *pack.required_boundary_tags,
        *pack.dataset_card.required_disclosures,
    }
    assert not {name for name in required_names if name not in guideline}


def test_annotations_validate_cli_is_offline_and_machine_readable() -> None:
    result = CliRunner().invoke(
        main,
        [
            "annotations",
            "validate",
            "--pack",
            str(PACK_PATH),
            "--taxonomy",
            str(TAXONOMY_PATH),
            "--template",
            str(TEMPLATE_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["decision_example_count"] == 36
