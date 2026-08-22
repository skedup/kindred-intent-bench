from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from intentbench.cli import main
from intentbench.dataset import load_candidate_cases
from intentbench.review import (
    IndependentReviewLabel,
    ReviewContractError,
    ReviewWorksheetEntry,
    build_adjudication_packet,
    build_review_items,
    import_adjudication_workbook,
    import_review_workbook,
    init_review_workspace,
    inspect_review_workspace,
    load_blind_batch,
    load_response_batch,
    write_response_batch,
)
from intentbench.schemas import Decision, Horizon, Slots

ROOT = Path(__file__).parents[2]
CASES_PATH = ROOT / "data/kir-pilot-v1/candidates.jsonl"
TAXONOMY_PATH = ROOT / "configs/kindred-activity-intents-v1.yaml"
REVIEW_DIR = ROOT / "data/kir-pilot-v1/reviews/initial"
COMPLETED_WORKBOOK_PATH = (
    ROOT / "outputs/2026-08-21-ie12-blind-review/kir-pilot-v1-initial-blind-review.xlsx"
)
V2_CASES_PATH = ROOT / "data/kir-pilot-v2/candidates.jsonl"
V2_TAXONOMY_PATH = ROOT / "configs/kindred-activity-intents-v2.yaml"
V2_GROUNDING_PATH = ROOT / "configs/kindred-activity-grounding-v1.yaml"
V2_REVIEW_DIR = ROOT / "data/kir-pilot-v2/reviews/initial"
COMPLETED_ADJUDICATION_WORKBOOK_PATH = (
    ROOT
    / "outputs/2026-08-22-ie12-grounded-adjudication"
    / "kir-pilot-v2-grounded-adjudication.xlsx"
)


def initialize(path: Path, *, batch_size: int = 40) -> None:
    init_review_workspace(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        output_dir=path,
        batch_size=batch_size,
    )


def test_review_workspace_is_deterministic_and_hides_candidate_labels(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    initialize(first)
    initialize(second)

    first_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files
    assert len(list((first / "blind").glob("batch-*.jsonl"))) == 4
    assert len(list((first / "responses").glob("batch-*.jsonl"))) == 4

    item_payload = json.loads((first / "blind/batch-01.jsonl").read_text().splitlines()[0])
    assert set(item_payload) == {"schema_version", "review_item_id", "context"}
    assert not {"id", "gold", "tags", "hard_negative_against"} & set(item_payload)
    response_payload = json.loads((first / "responses/batch-01.jsonl").read_text().splitlines()[0])
    assert response_payload["reviewer_id"] is None
    assert response_payload["label"] is None


def test_review_order_changes_for_blind_retest() -> None:
    cases = load_candidate_cases(CASES_PATH)
    initial = [case.id for _, _, case in build_review_items(cases, "initial")]
    retest = [case.id for _, _, case in build_review_items(cases, "blind_retest")]
    assert initial != retest
    assert set(initial) == set(retest)


def test_committed_blind_packets_match_deterministic_builder(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    initialize(generated)
    assert (generated / "manifest.json").read_bytes() == (REVIEW_DIR / "manifest.json").read_bytes()
    for batch_number in range(1, 5):
        filename = f"batch-{batch_number:02d}.jsonl"
        assert (generated / "blind" / filename).read_bytes() == (
            REVIEW_DIR / "blind" / filename
        ).read_bytes()

    report = inspect_review_workspace(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=REVIEW_DIR,
    )
    assert report["item_count"] == 160
    assert report["completed_count"] == 160
    assert report["pending_count"] == 0
    agreement_counts = report["agreement_counts"]
    assert isinstance(agreement_counts, dict)
    assert agreement_counts["hierarchical"] == 143
    assert len(report["hierarchical_disagreement_item_ids"]) == 17


def test_review_workspace_status_and_complete_gate(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir)
    report = inspect_review_workspace(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=review_dir,
    )
    assert report["status"] == "pending_human_review"
    assert report["completed_count"] == 0
    assert report["pending_count"] == 160
    assert report["agreement_status"] == "locked_until_pass_complete"
    assert "agreement_counts" not in report
    assert "disagreement_item_ids" not in report
    assert report["pending_by_batch"] == {
        "batch-01": 40,
        "batch-02": 40,
        "batch-03": 40,
        "batch-04": 40,
    }
    with pytest.raises(ReviewContractError, match="160 pending"):
        inspect_review_workspace(
            cases_path=CASES_PATH,
            taxonomy_path=TAXONOMY_PATH,
            review_dir=review_dir,
            require_complete=True,
        )


def test_review_workspace_rejects_candidate_hash_drift(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir)
    drifted_candidates = tmp_path / "candidates.jsonl"
    drifted_candidates.write_bytes(CASES_PATH.read_bytes() + b"\n")
    with pytest.raises(ReviewContractError, match="candidate artifact hash"):
        inspect_review_workspace(
            cases_path=drifted_candidates,
            taxonomy_path=TAXONOMY_PATH,
            review_dir=review_dir,
        )


def test_completed_workbook_import_is_validated_before_atomic_batch_replace(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir)
    entries = load_response_batch(review_dir, 1)
    entries[0] = load_response_batch(REVIEW_DIR, 1)[0]
    write_response_batch(review_dir, 1, entries)
    original = {path.name: path.read_bytes() for path in (review_dir / "responses").glob("*.jsonl")}

    dry_run = import_review_workbook(
        workbook_path=COMPLETED_WORKBOOK_PATH,
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=review_dir,
        dry_run=True,
    )
    assert dry_run["status"] == "valid_complete_workbook"
    assert dry_run["imported_count"] == 159
    assert dry_run["preserved_existing_count"] == 1
    assert original == {
        path.name: path.read_bytes() for path in (review_dir / "responses").glob("*.jsonl")
    }

    imported = import_review_workbook(
        workbook_path=COMPLETED_WORKBOOK_PATH,
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=review_dir,
    )
    assert imported["status"] == "imported_complete_workbook"
    workspace = imported["workspace"]
    assert isinstance(workspace, dict)
    assert workspace["completed_count"] == 160
    assert workspace["pending_count"] == 0
    receipt_path = Path(str(imported["receipt_path"]))
    assert receipt_path.is_file()

    repeated = import_review_workbook(
        workbook_path=COMPLETED_WORKBOOK_PATH,
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=review_dir,
    )
    assert repeated["status"] == "already_imported"


def test_completed_workbook_import_rejects_existing_response_conflict(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir)
    entries = load_response_batch(review_dir, 1)
    payload = load_response_batch(REVIEW_DIR, 1)[0].model_dump(mode="json")
    assert payload["label"] is not None
    payload["label"]["decision"] = "ambiguous"
    entries[0] = ReviewWorksheetEntry.model_validate(payload)
    write_response_batch(review_dir, 1, entries)
    before = {path.name: path.read_bytes() for path in (review_dir / "responses").glob("*.jsonl")}

    with pytest.raises(ReviewContractError, match="conflicts with an existing response"):
        import_review_workbook(
            workbook_path=COMPLETED_WORKBOOK_PATH,
            cases_path=CASES_PATH,
            taxonomy_path=TAXONOMY_PATH,
            review_dir=review_dir,
        )
    assert before == {
        path.name: path.read_bytes() for path in (review_dir / "responses").glob("*.jsonl")
    }


def test_adjudication_packet_is_deterministic_and_covers_disagreements(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = build_adjudication_packet(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=REVIEW_DIR,
        output_dir=first,
    )
    build_adjudication_packet(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=REVIEW_DIR,
        output_dir=second,
    )
    assert first_report["hierarchical_disagreement_count"] == 17
    assert first_report["agreement_audit_count"] == 15
    assert first_report["grounding_evidence_required_count"] == 7
    assert (first / "packet.jsonl").read_bytes() == (second / "packet.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()

    rows = [json.loads(line) for line in (first / "packet.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    assert sum(row["requires_user_decision"] for row in rows) == 17
    assert sum(row["sample_kind"] == "agreement_audit" for row in rows) == 15
    assert sum(row["recommendation"] == "grounding_evidence_required" for row in rows) == 7
    assert sum(row["review_focus"] == "near_oos_grounding_boundary" for row in rows) == 7


def test_completed_adjudication_workbook_import_is_hash_bound_and_flags_policy_impact(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    shutil.copytree(
        V2_REVIEW_DIR,
        review_dir,
        ignore=shutil.ignore_patterns("imports", "resolutions.jsonl", "policy-impact.md"),
    )
    packet_dir = review_dir / "adjudication"

    dry_run = import_adjudication_workbook(
        workbook_path=COMPLETED_ADJUDICATION_WORKBOOK_PATH,
        cases_path=V2_CASES_PATH,
        taxonomy_path=V2_TAXONOMY_PATH,
        grounding_path=V2_GROUNDING_PATH,
        packet_dir=packet_dir,
        adjudicator_id="skedushwang-adjudicator",
        dry_run=True,
    )
    assert dry_run["status"] == "valid_complete_adjudication_workbook"
    assert dry_run["item_count"] == 32
    assert dry_run["action_counts"] == {
        "保留草稿标签": 14,
        "接受人工标签": 3,
        "确认一致标签": 15,
    }
    assert dry_run["policy_review_candidate_ids"] == [
        "draft-kir-pilot-v2-0094",
        "draft-kir-pilot-v2-0131",
        "draft-kir-pilot-v2-0151",
    ]
    assert not (packet_dir / "resolutions.jsonl").exists()

    imported = import_adjudication_workbook(
        workbook_path=COMPLETED_ADJUDICATION_WORKBOOK_PATH,
        cases_path=V2_CASES_PATH,
        taxonomy_path=V2_TAXONOMY_PATH,
        grounding_path=V2_GROUNDING_PATH,
        packet_dir=packet_dir,
        adjudicator_id="skedushwang-adjudicator",
    )
    assert imported["status"] == "imported_complete_adjudication_workbook"
    resolution_rows = [
        json.loads(line)
        for line in (packet_dir / "resolutions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(resolution_rows) == 32
    assert sum(row["policy_impact"] != "none" for row in resolution_rows) == 3
    receipt_path = Path(str(imported["receipt_path"]))
    assert receipt_path.is_file()

    repeated = import_adjudication_workbook(
        workbook_path=COMPLETED_ADJUDICATION_WORKBOOK_PATH,
        cases_path=V2_CASES_PATH,
        taxonomy_path=V2_TAXONOMY_PATH,
        grounding_path=V2_GROUNDING_PATH,
        packet_dir=packet_dir,
        adjudicator_id="skedushwang-adjudicator",
    )
    assert repeated["status"] == "already_imported"


def test_complete_review_pass_reveals_field_agreement(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir)
    cases = load_candidate_cases(CASES_PATH)
    expected = build_review_items(cases, "initial")
    for batch_number in range(1, 5):
        start = (batch_number - 1) * 40
        expected_batch = expected[start : start + 40]
        entries = load_response_batch(review_dir, batch_number)
        completed: list[ReviewWorksheetEntry] = []
        for entry, (_, _, case) in zip(entries, expected_batch, strict=True):
            payload = entry.model_dump(mode="json")
            payload.update(
                {
                    "reviewer_id": "reviewer-1",
                    "reviewed_at": "2026-08-21T00:00:00Z",
                    "label": {
                        "decision": case.gold.decision.value,
                        "target_intent": case.gold.target_intent,
                        "evidence_quote": case.gold.evidence_quote,
                        "slots": case.gold.slots.model_dump(mode="json"),
                    },
                }
            )
            completed.append(ReviewWorksheetEntry.model_validate(payload))
        write_response_batch(review_dir, batch_number, completed)

    report = inspect_review_workspace(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=review_dir,
        require_complete=True,
    )
    assert report["status"] == "complete"
    assert report["agreement_status"] == "revealed_after_pass_complete"
    assert report["agreement_counts"] == {
        "decision": 160,
        "target_intent": 160,
        "horizon": 160,
        "hierarchical": 160,
    }
    assert report["free_text_exact_match_counts"] == {
        "evidence_quote": 160,
        "desired_experience": 160,
        "object": 160,
    }
    assert report["reference_label_kind"] == "llm_assisted_draft_label"
    assert report["comparison_purpose"] == "dataset_qa_not_model_accuracy"
    assert report["disagreement_item_ids"] == []


def test_review_workspace_rejects_non_context_evidence(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir, batch_size=160)
    entries = load_response_batch(review_dir, 1)
    payload = entries[0].model_dump(mode="json")
    payload.update(
        {
            "reviewer_id": "reviewer-1",
            "reviewed_at": "2026-08-21T00:00:00Z",
            "label": IndependentReviewLabel(
                decision=Decision.NO_INTENT,
                target_intent=None,
                evidence_quote="输入里不存在的证据",
                slots=Slots(
                    desired_experience=None,
                    object=None,
                    horizon=Horizon.NOW,
                ),
            ).model_dump(mode="json"),
        }
    )
    entries[0] = ReviewWorksheetEntry.model_validate(payload)
    write_response_batch(review_dir, 1, entries)
    with pytest.raises(ReviewContractError, match="contiguous context substring"):
        inspect_review_workspace(
            cases_path=CASES_PATH,
            taxonomy_path=TAXONOMY_PATH,
            review_dir=review_dir,
        )


def test_review_entry_rejects_partial_attribution() -> None:
    with pytest.raises(ValidationError, match="requires reviewer, timestamp, and label"):
        ReviewWorksheetEntry(
            review_item_id="review-initial-0001",
            review_pass="initial",
            reviewer_id="reviewer-1",
        )


def test_review_run_cli_records_only_blind_response(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir, batch_size=160)
    first_item = load_blind_batch(review_dir, 1)[0]
    prompt_input = "\n".join(
        (
            "3",
            first_item.context.state_summary,
            "",
            "",
            "now",
            "",
            "",
        )
    )
    result = CliRunner().invoke(
        main,
        [
            "dataset",
            "reviews",
            "run",
            "--cases",
            str(CASES_PATH),
            "--taxonomy",
            str(TAXONOMY_PATH),
            "--review-dir",
            str(review_dir),
            "--batch",
            "1",
            "--reviewer-id",
            "reviewer-1",
            "--limit",
            "1",
        ],
        input=prompt_input,
    )
    assert result.exit_code == 0, result.output
    assert "3=no_intent" in result.output
    saved_entry = load_response_batch(review_dir, 1)[0]
    assert saved_entry.label is not None
    assert saved_entry.label.decision is Decision.NO_INTENT
    report = inspect_review_workspace(
        cases_path=CASES_PATH,
        taxonomy_path=TAXONOMY_PATH,
        review_dir=review_dir,
    )
    assert report["completed_count"] == 1
    assert report["pending_count"] == 159
    assert report["reviewer_ids"] == ["reviewer-1"]
    assert report["agreement_status"] == "locked_until_pass_complete"
    assert "agreement_counts" not in report

    validate_result = CliRunner().invoke(
        main,
        [
            "dataset",
            "reviews",
            "validate",
            "--cases",
            str(CASES_PATH),
            "--taxonomy",
            str(TAXONOMY_PATH),
            "--review-dir",
            str(review_dir),
        ],
    )
    assert validate_result.exit_code != 0
    assert "159 pending" in validate_result.output


def test_review_revise_replaces_only_after_confirmation(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    initialize(review_dir, batch_size=160)
    first_item = load_blind_batch(review_dir, 1)[0]
    entries = load_response_batch(review_dir, 1)
    payload = entries[0].model_dump(mode="json")
    payload.update(
        {
            "reviewer_id": "reviewer-1",
            "reviewed_at": "2026-08-21T00:00:00Z",
            "label": IndependentReviewLabel(
                decision=Decision.NO_INTENT,
                target_intent=None,
                evidence_quote=first_item.context.state_summary,
                slots=Slots(
                    desired_experience=None,
                    object=None,
                    horizon=Horizon.NOW,
                ),
            ).model_dump(mode="json"),
        }
    )
    entries[0] = ReviewWorksheetEntry.model_validate(payload)
    write_response_batch(review_dir, 1, entries)
    command = [
        "dataset",
        "reviews",
        "revise",
        "--cases",
        str(CASES_PATH),
        "--taxonomy",
        str(TAXONOMY_PATH),
        "--review-dir",
        str(review_dir),
        "--review-item-id",
        first_item.review_item_id,
        "--reviewer-id",
        "reviewer-1",
    ]

    cancelled = CliRunner().invoke(main, command, input="n\n")
    assert cancelled.exit_code != 0
    unchanged = load_response_batch(review_dir, 1)[0]
    assert unchanged.label is not None
    assert unchanged.label.decision is Decision.NO_INTENT

    revise_input = "\n".join(
        (
            "y",
            "4",
            first_item.context.state_summary,
            "",
            "",
            "now",
            "corrected after self-review",
            "",
        )
    )
    revised = CliRunner().invoke(main, command, input=revise_input)
    assert revised.exit_code == 0, revised.output
    saved = load_response_batch(review_dir, 1)[0]
    assert saved.label is not None
    assert saved.label.decision is Decision.AMBIGUOUS
    assert saved.review_note == "corrected after self-review"
    assert '"revised_item_id": "review-initial-0001"' in revised.output
