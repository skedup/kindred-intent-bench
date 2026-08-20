from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from intentbench.annotation import (
    AdjudicationContract,
    AnnotationPack,
    CaseAuthoringTemplate,
    load_annotation_pack,
    load_case_authoring_template,
    validate_annotation_artifacts,
)
from intentbench.cli import main
from intentbench.schemas import Case, Decision
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
        "pack_id": "kir-pilot-annotation-v2",
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
    draft_fields = type(template.case).model_fields
    assert "split" not in draft_fields
    assert "split_group_id" not in draft_fields
    assert "bootstrap_cluster_id" not in draft_fields
    with pytest.raises(ValidationError):
        Case.model_validate(template.case.model_dump(mode="json"))


def test_pack_rejects_missing_decision_coverage() -> None:
    payload = load_annotation_pack(PACK_PATH).model_dump(mode="json")
    payload["decision_examples"].pop()
    with pytest.raises(ValidationError, match="at least 36 items"):
        AnnotationPack.model_validate(payload)


def test_pack_rejects_extra_decision_example() -> None:
    payload = load_annotation_pack(PACK_PATH).model_dump(mode="json")
    duplicate = dict(payload["decision_examples"][0])
    duplicate["example_id"] = "guide-extra-positive-01"
    payload["decision_examples"].append(duplicate)
    with pytest.raises(ValidationError, match="at most 36 items"):
        AnnotationPack.model_validate(payload)


def test_pack_rejects_duplicate_intent_comparison_text() -> None:
    payload = load_annotation_pack(PACK_PATH).model_dump(mode="json")
    boundary = payload["intent_boundaries"][0]
    boundary["inclusion_examples"][1] = dict(boundary["inclusion_examples"][0])
    with pytest.raises(ValidationError, match="comparison texts must be unique"):
        AnnotationPack.model_validate(payload)


def test_pack_rejects_duplicate_guide_context() -> None:
    payload = load_annotation_pack(PACK_PATH).model_dump(mode="json")
    source = payload["decision_examples"][0]
    duplicate = payload["decision_examples"][1]
    duplicate["context"] = source["context"]
    duplicate["gold"]["evidence_quote"] = source["gold"]["evidence_quote"]
    with pytest.raises(ValidationError, match="guide contexts must be unique"):
        AnnotationPack.model_validate(payload)


def test_pack_requires_all_six_adjudication_decision_pairs() -> None:
    payload = load_annotation_pack(PACK_PATH).model_dump(mode="json")
    example = next(
        item
        for item in payload["decision_examples"]
        if item["example_id"] == "guide-oos-adjudication-03"
    )
    example["competing_labels"][1] = {
        "decision": "in_scope",
        "target_intent": "visit_cultural_place",
    }
    with pytest.raises(ValidationError, match="cover all six decision pairs"):
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


@pytest.mark.parametrize("missing_field", ["evidence_quote", "adjudicator_id"])
def test_adjudication_contract_requires_complete_audit_chain(missing_field: str) -> None:
    payload = load_annotation_pack(PACK_PATH).adjudication.model_dump(mode="json")
    payload["disagreement_record_fields"].remove(missing_field)
    payload["disagreement_record_fields"].append("reviewer_note")
    with pytest.raises(ValidationError, match="lacks required fields"):
        AdjudicationContract.model_validate(payload)


def test_adjudication_can_express_same_decision_different_intents() -> None:
    pack = load_annotation_pack(PACK_PATH)
    example = next(
        item
        for item in pack.decision_examples
        if item.example_id == "guide-in-scope-adjudication-03"
    )
    competing_targets = {
        label.target_intent
        for label in example.competing_labels
        if label.decision is Decision.IN_SCOPE
    }
    assert competing_targets == {"create_picture", "reach_out_to_user"}


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
    assert "Gold 标注者、独立复核者和仲裁者" in guideline
    assert "能否从输入确定唯一的下一主要行动" in guideline


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


def test_annotations_validate_cli_reports_invalid_yaml(tmp_path: Path) -> None:
    invalid_pack = tmp_path / "invalid.yaml"
    invalid_pack.write_text("decision_examples: [", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            "annotations",
            "validate",
            "--pack",
            str(invalid_pack),
            "--taxonomy",
            str(TAXONOMY_PATH),
            "--template",
            str(TEMPLATE_PATH),
        ],
    )
    assert result.exit_code != 0
    assert "invalid annotation pack YAML" in result.output
