"""Carry exact-context human labels into the grounded IE1.2 v2 review workspace."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from intentbench.dataset import load_candidate_cases
from intentbench.review import (
    ReviewContractError,
    ReviewWorksheetEntry,
    build_review_items,
    inspect_review_workspace,
    load_response_batch,
    load_review_manifest,
    validate_review_label,
    write_response_batch,
)
from intentbench.taxonomy import load_taxonomy

ROOT = Path(__file__).parents[1]
SOURCE_CASES = ROOT / "data/kir-pilot-v1/candidates.jsonl"
SOURCE_TAXONOMY = ROOT / "configs/kindred-activity-intents-v1.yaml"
SOURCE_REVIEW = ROOT / "data/kir-pilot-v1/reviews/initial"
TARGET_CASES = ROOT / "data/kir-pilot-v2/candidates.jsonl"
TARGET_TAXONOMY = ROOT / "configs/kindred-activity-intents-v2.yaml"
TARGET_REVIEW = ROOT / "data/kir-pilot-v2/reviews/initial"


def context_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    source_status = inspect_review_workspace(
        cases_path=SOURCE_CASES,
        taxonomy_path=SOURCE_TAXONOMY,
        review_dir=SOURCE_REVIEW,
        require_complete=True,
    )
    target_status = inspect_review_workspace(
        cases_path=TARGET_CASES,
        taxonomy_path=TARGET_TAXONOMY,
        review_dir=TARGET_REVIEW,
    )
    if target_status["completed_count"] != 0:
        raise ReviewContractError("target review workspace already contains completed labels")

    source_manifest = load_review_manifest(SOURCE_REVIEW)
    target_manifest = load_review_manifest(TARGET_REVIEW)
    source_cases = load_candidate_cases(SOURCE_CASES)
    target_cases = load_candidate_cases(TARGET_CASES)
    source_items = build_review_items(source_cases, source_manifest.review_pass)
    target_items = build_review_items(target_cases, target_manifest.review_pass)
    target_taxonomy = load_taxonomy(TARGET_TAXONOMY)

    source_entries = {
        entry.review_item_id: entry
        for batch_number in range(1, source_manifest.batch_count + 1)
        for entry in load_response_batch(SOURCE_REVIEW, batch_number)
    }
    source_by_context: dict[str, tuple[ReviewWorksheetEntry, str, str]] = {}
    for item, _, case in source_items:
        entry = source_entries[item.review_item_id]
        if not entry.is_complete:
            raise ReviewContractError("source review workspace contains a pending response")
        key = context_key(item.context.model_dump(mode="json"))
        if key in source_by_context:
            raise ReviewContractError("source review contexts are not unique")
        source_by_context[key] = (entry, item.review_item_id, case.id)

    migrated_by_batch: list[list[ReviewWorksheetEntry]] = [
        [] for _ in range(target_manifest.batch_count)
    ]
    mappings: list[dict[str, str]] = []
    pending: list[dict[str, object]] = []
    for index, (item, pending_entry, case) in enumerate(target_items):
        key = context_key(item.context.model_dump(mode="json"))
        source = source_by_context.get(key)
        if source is None:
            entry = pending_entry
            pending.append(
                {
                    "review_item_id": item.review_item_id,
                    "candidate_id": case.id,
                    "context": item.context.model_dump(mode="json"),
                }
            )
        else:
            source_entry, source_review_item_id, source_candidate_id = source
            assert source_entry.label is not None
            validate_review_label(source_entry.label, item, target_taxonomy)
            payload = source_entry.model_dump(mode="json")
            payload.update(
                {
                    "review_item_id": item.review_item_id,
                    "review_pass": target_manifest.review_pass,
                }
            )
            entry = ReviewWorksheetEntry.model_validate(payload)
            mappings.append(
                {
                    "source_review_item_id": source_review_item_id,
                    "source_candidate_id": source_candidate_id,
                    "target_review_item_id": item.review_item_id,
                    "target_candidate_id": case.id,
                }
            )
        batch_index = index // target_manifest.batch_size
        migrated_by_batch[batch_index].append(entry)

    for batch_number, entries in enumerate(migrated_by_batch, 1):
        write_response_batch(TARGET_REVIEW, batch_number, entries)

    receipt = {
        "schema_version": 1,
        "migration_id": "kir-pilot-v1-to-v2-grounding-v1",
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "carry_forward_only_when_normalized_context_is_byte_equivalent",
        "source": {
            "dataset_id": source_manifest.dataset_id,
            "candidate_artifact_sha256": source_manifest.candidate_artifact_sha256,
            "taxonomy_version": source_manifest.taxonomy_version,
            "taxonomy_artifact_sha256": source_manifest.taxonomy_artifact_sha256,
            "completed_count": source_status["completed_count"],
        },
        "target": {
            "dataset_id": target_manifest.dataset_id,
            "candidate_artifact_sha256": target_manifest.candidate_artifact_sha256,
            "taxonomy_version": target_manifest.taxonomy_version,
            "taxonomy_artifact_sha256": target_manifest.taxonomy_artifact_sha256,
        },
        "migrated_count": len(mappings),
        "pending_count": len(pending),
        "mappings": mappings,
        "pending": pending,
    }
    receipt_path = TARGET_REVIEW / "migrations" / "kir-pilot-v1-to-v2-grounding-v1.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        f"{json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n",
        encoding="utf-8",
    )
    report = inspect_review_workspace(
        cases_path=TARGET_CASES,
        taxonomy_path=TARGET_TAXONOMY,
        review_dir=TARGET_REVIEW,
    )
    print(
        json.dumps(
            {
                "status": "migrated_pending_delta_review",
                "migrated_count": len(mappings),
                "pending_count": len(pending),
                "pending": pending,
                "receipt_path": str(receipt_path),
                "workspace": report,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
