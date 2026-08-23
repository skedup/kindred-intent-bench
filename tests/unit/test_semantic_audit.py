from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from intentbench.schemas import PredictionStatus
from intentbench.semantic_audit import (
    AUDIT_HEADERS,
    CompletedSemanticAudit,
    SemanticAuditRow,
    load_completed_semantic_audit,
    semantic_audit_csv_bytes,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_audit() -> CompletedSemanticAudit:
    root = repository_root()
    return load_completed_semantic_audit(
        workbook_path=root / "outputs/2026-08-23-ie4-semantic-audit/semantic-audit.xlsx",
        process_path=root / "configs/kir-pilot-v2-ie4-semantic-audit-process.yaml",
        cases_path=root / "data/kir-pilot-v2/test.jsonl",
        dataset_manifest_path=root / "data/kir-pilot-v2/freeze-manifest.json",
        split_manifest_path=root / "data/kir-pilot-v2/split-manifest.json",
        primary_formed_path=root
        / "experiments/ie3-test/primary_decision/b3/formed-intentions.jsonl",
        weak_formed_path=root / "experiments/ie3-test/weak_decision/b3/formed-intentions.jsonl",
        repository_root=root,
    )


def test_completed_audit_is_hash_bound_and_summarized() -> None:
    audit = load_audit()

    assert len(audit.rows) == 48
    assert audit.reviewer_id == "skedushwang"
    assert audit.reviewed_at == date(2026, 8, 23)
    assert audit.process.independent_second_human_review is False
    assert audit.process.pre_ai_human_label_snapshot_retained is False
    assert audit.process.ai_assistance.score_authority is False
    assert audit.process.ai_assistance.scores_overwritten is False
    disagreements = audit.process.ai_assistance.nonbinding_disagreements
    assert sum(map(len, disagreements.values())) == 10

    primary, weak = audit.role_summaries
    assert primary.model_role == "primary_decision"
    assert primary.stage_a_failure_count == 0
    assert primary.all_fields_preserved_count == 17
    assert primary.fields["action"].model_dump() == {
        "preserved": 24,
        "partial": 0,
        "lost": 0,
        "total": 24,
    }
    assert primary.fields["object"].model_dump() == {
        "preserved": 18,
        "partial": 6,
        "lost": 0,
        "total": 24,
    }
    assert primary.fields["horizon"].model_dump() == {
        "preserved": 23,
        "partial": 1,
        "lost": 0,
        "total": 24,
    }

    assert weak.model_role == "weak_decision"
    assert weak.stage_a_failure_count == 17
    assert weak.all_fields_preserved_count == 4
    assert weak.fields["action"].model_dump() == {
        "preserved": 7,
        "partial": 0,
        "lost": 17,
        "total": 24,
    }
    assert weak.fields["object"].model_dump() == {
        "preserved": 4,
        "partial": 3,
        "lost": 17,
        "total": 24,
    }
    assert weak.fields["horizon"].model_dump() == {
        "preserved": 7,
        "partial": 0,
        "lost": 17,
        "total": 24,
    }


def test_semantic_audit_csv_is_full_fidelity_and_deterministic() -> None:
    audit = load_audit()
    payload = semantic_audit_csv_bytes(audit)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))

    assert tuple(rows[0]) == AUDIT_HEADERS
    assert len(rows) == 48
    assert rows[0]["audit_row_id"] == "semantic-audit-0001"
    assert rows[-1]["audit_row_id"] == "semantic-audit-0048"
    assert payload == semantic_audit_csv_bytes(audit)


def test_failed_stage_a_row_requires_registered_lost_ratings() -> None:
    row = load_audit().rows[1]
    payload = row.model_dump(mode="json")
    payload.update(
        stage_a_status=PredictionStatus.PROVIDER_FAILURE.value,
        action_rating="preserved",
        object_rating="lost",
        horizon_rating="lost",
        reviewer_note="stage_a_provider_failure",
    )

    with pytest.raises(ValidationError, match="three lost ratings"):
        SemanticAuditRow.model_validate(payload)
