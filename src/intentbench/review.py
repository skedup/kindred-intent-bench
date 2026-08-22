"""Label-blind human review workspace for IE1.2 candidate adjudication."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

from pydantic import Field, ValidationError, model_validator

from intentbench.annotation import DraftCase
from intentbench.dataset import load_candidate_cases, validate_candidate_cases
from intentbench.freeze import sha256_file
from intentbench.grounding import load_grounding_snapshot
from intentbench.schemas import (
    Context,
    Decision,
    Gold,
    Horizon,
    Slots,
    StrictModel,
    Taxonomy,
    normalize_evidence_text,
)
from intentbench.taxonomy import load_taxonomy

ReviewPass = Literal["initial", "blind_retest"]
AdjudicationSampleKind = Literal["hierarchical_disagreement", "agreement_audit"]
AdjudicationReviewFocus = Literal[
    "near_oos_grounding_boundary",
    "target_selection",
    "decision_rule_boundary",
    "agreement_audit",
]
AdjudicationRecommendation = Literal[
    "grounding_evidence_required",
    "likely_reviewer_input_error",
    "current_rule_lean_keep_draft",
    "confirm_agreement",
]
AdjudicationAction = Literal[
    "确认一致标签",
    "接受人工标签",
    "保留草稿标签",
    "修改分类规则",
    "重写或删除样本",
]
AdjudicationPolicyImpact = Literal[
    "none",
    "taxonomy_or_decision_rule_review_required",
]
REVIEW_SHUFFLE_METHOD: Literal["sha256-review-pass-case-id-v1"] = "sha256-review-pass-case-id-v1"
REVIEW_WORKBOOK_SCHEMA_VERSION: Literal[1] = 1
REVIEW_WORKBOOK_HEADERS = (
    "batch",
    "review_item_id",
    "state_summary",
    "conversation",
    "recent_activities",
    "decision",
    "target_intent",
    "evidence_quote",
    "desired_experience",
    "object",
    "horizon",
    "review_note",
    "status（自动）",  # noqa: RUF001 - exact workbook contract header
)
REVIEW_WORKBOOK_DECISIONS: dict[str, Decision] = {
    "1": Decision.IN_SCOPE,
    "1 - in_scope": Decision.IN_SCOPE,
    "in_scope": Decision.IN_SCOPE,
    "2": Decision.OOS,
    "2 - oos": Decision.OOS,
    "oos": Decision.OOS,
    "3": Decision.NO_INTENT,
    "3 - no_intent": Decision.NO_INTENT,
    "no_intent": Decision.NO_INTENT,
    "4": Decision.AMBIGUOUS,
    "4 - ambiguous": Decision.AMBIGUOUS,
    "ambiguous": Decision.AMBIGUOUS,
}
ADJUDICATION_WORKBOOK_SCHEMA_VERSION: Literal[1] = 1
ADJUDICATION_WORKBOOK_HEADERS = (
    "packet_item_id",
    "sample_kind",
    "review_focus",
    "review_item_id",
    "candidate_id",
    "state_summary",
    "conversation",
    "recent_activities",
    "人工 decision",
    "人工 target",
    "人工 horizon",
    "人工 evidence",
    "人工 slots",
    "草稿 decision",
    "草稿 target",
    "草稿 horizon",
    "草稿 evidence",
    "草稿 slots",
    "tags",
    "hard_negative_against",
    "建议",
    "建议依据",
    "仲裁动作（填写）",  # noqa: RUF001 - exact workbook contract header
    "final_decision（填写）",  # noqa: RUF001 - exact workbook contract header
    "final_target（填写）",  # noqa: RUF001 - exact workbook contract header
    "final_horizon（填写）",  # noqa: RUF001 - exact workbook contract header
    "仲裁理由（填写）",  # noqa: RUF001 - exact workbook contract header
    "taxonomy 修改说明（如需）",  # noqa: RUF001 - exact workbook contract header
    "status（自动）",  # noqa: RUF001 - exact workbook contract header
    "complete_flag",
    "pending_flag",
)
ADJUDICATION_ACTIONS: frozenset[str] = frozenset(
    {
        "确认一致标签",
        "接受人工标签",
        "保留草稿标签",
        "修改分类规则",
        "重写或删除样本",
    }
)

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REFERENCE = re.compile(r"^([A-Z]{1,3})([1-9][0-9]*)$")


class ReviewContractError(ValueError):
    """A human-review workspace is incomplete, stale, or internally inconsistent."""


class BlindReviewItem(StrictModel):
    """Input shown to a reviewer without candidate Gold, tags, or source IDs."""

    schema_version: Literal[1] = 1
    review_item_id: str = Field(pattern=r"^review-(initial|blind-retest)-[0-9]{4}$")
    context: Context


class IndependentReviewLabel(StrictModel):
    """A label committed before candidate Gold and diagnostic metadata are revealed."""

    decision: Decision
    target_intent: str | None
    evidence_quote: str = Field(min_length=1)
    slots: Slots

    @model_validator(mode="after")
    def validate_decision_target(self) -> IndependentReviewLabel:
        if self.decision is Decision.IN_SCOPE:
            if self.target_intent is None:
                raise ValueError("in_scope review label requires target_intent")
        elif self.target_intent is not None:
            raise ValueError("only in_scope review labels may carry target_intent")
        return self


class ReviewWorksheetEntry(StrictModel):
    """One resumable response row; pending rows contain no reviewer attribution."""

    schema_version: Literal[1] = 1
    review_item_id: str = Field(pattern=r"^review-(initial|blind-retest)-[0-9]{4}$")
    review_pass: ReviewPass
    reviewer_id: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None
    label: IndependentReviewLabel | None = None
    review_note: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_completion_state(self) -> ReviewWorksheetEntry:
        completed_fields = (self.reviewer_id, self.reviewed_at, self.label)
        if all(value is None for value in completed_fields):
            if self.review_note is not None:
                raise ValueError("pending review entry cannot carry a note")
            return self
        if any(value is None for value in completed_fields):
            raise ValueError("completed review entry requires reviewer, timestamp, and label")
        assert self.reviewed_at is not None
        if self.reviewed_at.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        return self

    @property
    def is_complete(self) -> bool:
        return self.label is not None


class ReviewWorkspaceManifest(StrictModel):
    """Binds a blind review pass to exact candidate and taxonomy artifacts."""

    schema_version: Literal[1] = 1
    dataset_id: str = Field(default="kir-pilot-v1", pattern=r"^kir-pilot-v[0-9]+$")
    review_pass: ReviewPass
    candidate_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_version: str
    shuffle_method: Literal["sha256-review-pass-case-id-v1"] = REVIEW_SHUFFLE_METHOD
    item_count: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    batch_count: int = Field(ge=1)
    status: Literal["pending_human_review"] = "pending_human_review"


class ReviewWorkbookImportReceipt(StrictModel):
    """Immutable provenance for one validated workbook-to-JSONL import."""

    schema_version: Literal[1] = 1
    dataset_id: str = Field(default="kir-pilot-v1", pattern=r"^kir-pilot-v[0-9]+$")
    review_pass: ReviewPass
    workbook_schema_version: Literal[1] = REVIEW_WORKBOOK_SCHEMA_VERSION
    workbook_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_workbook_filename: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    reviewed_at: datetime
    reviewed_at_source: Literal["workbook_file_mtime_utc"] = "workbook_file_mtime_utc"
    imported_at: datetime
    item_count: int = Field(ge=1)
    imported_count: int = Field(ge=0)
    preserved_existing_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_timestamps_and_counts(self) -> ReviewWorkbookImportReceipt:
        if self.reviewed_at.tzinfo is None or self.imported_at.tzinfo is None:
            raise ValueError("workbook import timestamps must include a timezone")
        if self.imported_count + self.preserved_existing_count != self.item_count:
            raise ValueError("workbook import counts must cover every item")
        return self


class AdjudicationPacketItem(StrictModel):
    """One disclosed comparison row selected for explicit post-blind adjudication."""

    schema_version: Literal[1] = 1
    packet_item_id: str = Field(pattern=r"^adjudication-[0-9]{4}$")
    sample_kind: AdjudicationSampleKind
    review_item_id: str = Field(pattern=r"^review-(initial|blind-retest)-[0-9]{4}$")
    candidate_id: str
    context: Context
    human_label: IndependentReviewLabel
    human_review_note: str | None = None
    draft_label: Gold
    draft_annotation_note: str
    tags: list[str]
    hard_negative_against: list[str]
    scenario_family_id: str
    contrast_group_id: str
    decision_matches: bool
    target_matches: bool
    horizon_matches: bool
    hierarchical_matches: bool
    review_focus: AdjudicationReviewFocus
    recommendation: AdjudicationRecommendation
    recommendation_reason: str
    requires_user_decision: bool

    @model_validator(mode="after")
    def validate_sample_kind(self) -> AdjudicationPacketItem:
        is_disagreement = self.sample_kind == "hierarchical_disagreement"
        if is_disagreement == self.hierarchical_matches:
            raise ValueError("adjudication sample kind disagrees with controlled labels")
        if self.requires_user_decision != is_disagreement:
            raise ValueError("only hierarchical disagreements require adjudication")
        return self


class AdjudicationPacketManifest(StrictModel):
    """Binds an adjudication packet to the completed blind review and draft artifacts."""

    schema_version: Literal[1] = 1
    dataset_id: str = Field(default="kir-pilot-v1", pattern=r"^kir-pilot-v[0-9]+$")
    review_pass: ReviewPass
    candidate_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_method: Literal["hierarchical-disagreements-plus-stratified-agreements-v1"] = (
        "hierarchical-disagreements-plus-stratified-agreements-v1"
    )
    hierarchical_disagreement_count: int = Field(ge=1)
    agreement_audit_count: int = Field(ge=1)
    item_count: int = Field(ge=1)
    status: Literal["pending_adjudication"] = "pending_adjudication"

    @model_validator(mode="after")
    def validate_counts(self) -> AdjudicationPacketManifest:
        if self.hierarchical_disagreement_count + self.agreement_audit_count != self.item_count:
            raise ValueError("adjudication packet counts must add up to item_count")
        return self


class AdjudicationResolution(StrictModel):
    """One hash-attributed adjudication decision imported from the completed workbook."""

    schema_version: Literal[1] = 1
    packet_item_id: str = Field(pattern=r"^adjudication-[0-9]{4}$")
    candidate_id: str
    action: AdjudicationAction
    final_decision: Decision | None = None
    final_target_intent: str | None = None
    final_horizon: Horizon | None = None
    rationale: str = Field(min_length=1)
    taxonomy_change_note: str | None = Field(default=None, min_length=1)
    adjudicator_id: str = Field(min_length=1)
    adjudicated_at: datetime
    source_workbook_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_impact: AdjudicationPolicyImpact = "none"
    policy_impact_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_resolution(self) -> AdjudicationResolution:
        if self.adjudicated_at.tzinfo is None:
            raise ValueError("adjudicated_at must include a timezone")
        if self.action == "重写或删除样本":
            if any(
                value is not None
                for value in (self.final_decision, self.final_target_intent, self.final_horizon)
            ):
                raise ValueError("rewrite/delete resolutions cannot carry final controlled labels")
        else:
            if self.final_decision is None or self.final_horizon is None:
                raise ValueError("label resolutions require final decision and horizon")
            if self.final_decision is Decision.IN_SCOPE:
                if self.final_target_intent is None:
                    raise ValueError("in_scope adjudication requires final_target_intent")
            elif self.final_target_intent is not None:
                raise ValueError("only in_scope adjudication may carry final_target_intent")
        if self.action == "修改分类规则" and self.taxonomy_change_note is None:
            raise ValueError("taxonomy modification requires a change note")
        impact_requires_reason = self.policy_impact != "none"
        if impact_requires_reason != (self.policy_impact_reason is not None):
            raise ValueError("policy impact and reason must be present together")
        return self


class AdjudicationWorkbookImportReceipt(StrictModel):
    """Immutable provenance for one validated adjudication-workbook import."""

    schema_version: Literal[1] = 1
    dataset_id: str = Field(pattern=r"^kir-pilot-v[0-9]+$")
    review_pass: ReviewPass
    workbook_schema_version: Literal[1] = ADJUDICATION_WORKBOOK_SCHEMA_VERSION
    workbook_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_workbook_filename: str = Field(min_length=1)
    packet_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grounding_version: str = Field(min_length=1)
    grounding_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjudicator_id: str = Field(min_length=1)
    adjudicated_at: datetime
    adjudicated_at_source: Literal["workbook_file_mtime_utc"] = "workbook_file_mtime_utc"
    imported_at: datetime
    item_count: int = Field(ge=1)
    action_counts: dict[str, int]
    resolution_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_review_candidate_ids: list[str]

    @model_validator(mode="after")
    def validate_receipt(self) -> AdjudicationWorkbookImportReceipt:
        if self.adjudicated_at.tzinfo is None or self.imported_at.tzinfo is None:
            raise ValueError("adjudication import timestamps must include a timezone")
        if sum(self.action_counts.values()) != self.item_count:
            raise ValueError("adjudication action counts must cover every item")
        if len(self.policy_review_candidate_ids) != len(set(self.policy_review_candidate_ids)):
            raise ValueError("policy-review candidate IDs must be unique")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(f"{_canonical_json(value)}\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[object]) -> None:
    serialized = "\n".join(_canonical_json(value) for value in values)
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _jsonl_bytes(values: Sequence[object]) -> bytes:
    serialized = "\n".join(_canonical_json(value) for value in values)
    return f"{serialized}\n".encode()


def _nonblank(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_conversation(item: BlindReviewItem) -> str:
    return "\n".join(
        f"{'Kindred' if turn.role == 'assistant' else 'User'}: {turn.content}"
        for turn in item.context.conversation
    )


def _xlsx_column_number(name: str) -> int:
    value = 0
    for character in name:
        value = value * 26 + ord(character) - ord("A") + 1
    return value


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{_SPREADSHEET_NS}}}t"))
        for item in root.findall(f"{{{_SPREADSHEET_NS}}}si")
    ]


def _xlsx_sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships_root.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
    }
    paths: dict[str, str] = {}
    for sheet in workbook_root.findall(f"{{{_SPREADSHEET_NS}}}sheets/{{{_SPREADSHEET_NS}}}sheet"):
        relationship_id = sheet.attrib[f"{{{_OFFICE_REL_NS}}}id"]
        target = targets.get(relationship_id)
        if target is None:
            raise ReviewContractError(f"workbook sheet relationship missing: {relationship_id}")
        normalized = posixpath.normpath(posixpath.join("xl", target))
        if not normalized.startswith("xl/"):
            raise ReviewContractError("workbook sheet relationship escapes xl directory")
        paths[sheet.attrib["name"]] = normalized
    return paths


def _xlsx_cell_value(
    cell: ElementTree.Element, shared_strings: Sequence[str]
) -> str | int | float | bool | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        value = "".join(node.text or "" for node in cell.iter(f"{{{_SPREADSHEET_NS}}}t"))
        return value or None
    value_node = cell.find(f"{{{_SPREADSHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw_value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (IndexError, ValueError) as exc:
            raise ReviewContractError("workbook contains an invalid shared-string index") from exc
    if cell_type in {"str", "e"}:
        return raw_value
    if cell_type == "b":
        return raw_value == "1"
    try:
        if re.fullmatch(r"-?[0-9]+", raw_value):
            return int(raw_value)
        return float(raw_value)
    except ValueError:
        return raw_value


def _load_xlsx_cells(path: Path, required_sheets: Sequence[str]) -> dict[str, dict[str, object]]:
    if not zipfile.is_zipfile(path):
        raise ReviewContractError(f"not a valid .xlsx archive: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _xlsx_shared_strings(archive)
            sheet_paths = _xlsx_sheet_paths(archive)
            missing = set(required_sheets) - set(sheet_paths)
            if missing:
                raise ReviewContractError(f"workbook sheets missing: {sorted(missing)}")
            result: dict[str, dict[str, object]] = {}
            for sheet_name in required_sheets:
                root = ElementTree.fromstring(archive.read(sheet_paths[sheet_name]))
                cells: dict[str, object] = {}
                for cell in root.iter(f"{{{_SPREADSHEET_NS}}}c"):
                    reference = cell.attrib.get("r")
                    if reference is None or _CELL_REFERENCE.fullmatch(reference) is None:
                        raise ReviewContractError(
                            f"{sheet_name}: invalid or missing cell reference"
                        )
                    cells[reference] = _xlsx_cell_value(cell, shared_strings)
                result[sheet_name] = cells
            return result
    except (KeyError, ElementTree.ParseError, zipfile.BadZipFile) as exc:
        raise ReviewContractError(f"invalid .xlsx workbook structure: {path}") from exc


def _load_jsonl(
    path: Path, model: type[BlindReviewItem] | type[ReviewWorksheetEntry]
) -> list[object]:
    values: list[object] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise ReviewContractError(f"{path}:{line_number}: blank JSONL lines are forbidden")
        try:
            values.append(model.model_validate_json(raw_line))
        except ValidationError as exc:
            raise ReviewContractError(f"{path}:{line_number}: invalid review row") from exc
    return values


def _ordered_candidates(cases: Sequence[DraftCase], review_pass: ReviewPass) -> list[DraftCase]:
    def ordering_key(case: DraftCase) -> str:
        payload = f"{REVIEW_SHUFFLE_METHOD}|{review_pass}|{case.id}".encode()
        return hashlib.sha256(payload).hexdigest()

    return sorted(cases, key=ordering_key)


def build_review_items(
    cases: Sequence[DraftCase], review_pass: ReviewPass
) -> list[tuple[BlindReviewItem, ReviewWorksheetEntry, DraftCase]]:
    pass_slug = review_pass.replace("_", "-")
    items: list[tuple[BlindReviewItem, ReviewWorksheetEntry, DraftCase]] = []
    for index, case in enumerate(_ordered_candidates(cases, review_pass), 1):
        review_item_id = f"review-{pass_slug}-{index:04d}"
        items.append(
            (
                BlindReviewItem(review_item_id=review_item_id, context=case.context),
                ReviewWorksheetEntry(review_item_id=review_item_id, review_pass=review_pass),
                case,
            )
        )
    return items


def candidate_dataset_id(cases: Sequence[DraftCase]) -> str:
    if not cases:
        raise ReviewContractError("cannot infer dataset id from an empty candidate set")
    first_id = cases[0].id
    versioned = re.fullmatch(r"draft-(kir-pilot-v[0-9]+)-[0-9]{4}", first_id)
    if versioned is not None:
        dataset_id = versioned.group(1)
        expected_prefix = f"draft-{dataset_id}-"
    elif re.fullmatch(r"draft-kir-pilot-[0-9]{4}", first_id):
        dataset_id = "kir-pilot-v1"
        expected_prefix = "draft-kir-pilot-"
    else:
        raise ReviewContractError(f"unsupported candidate id for dataset inference: {first_id}")
    if any(not case.id.startswith(expected_prefix) for case in cases):
        raise ReviewContractError("candidate set mixes dataset id prefixes")
    return dataset_id


def init_review_workspace(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    review_pass: ReviewPass = "initial",
    batch_size: int = 40,
) -> dict[str, object]:
    if batch_size < 1:
        raise ReviewContractError("review batch size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReviewContractError(f"review workspace is not empty: {output_dir}")

    taxonomy = load_taxonomy(taxonomy_path)
    cases = load_candidate_cases(cases_path)
    validate_candidate_cases(cases, taxonomy)
    items = build_review_items(cases, review_pass)
    batch_count = (len(items) + batch_size - 1) // batch_size
    manifest = ReviewWorkspaceManifest(
        dataset_id=candidate_dataset_id(cases),
        review_pass=review_pass,
        candidate_artifact_sha256=sha256_file(cases_path),
        taxonomy_artifact_sha256=sha256_file(taxonomy_path),
        taxonomy_version=taxonomy.taxonomy_version,
        item_count=len(items),
        batch_size=batch_size,
        batch_count=batch_count,
    )

    blind_dir = output_dir / "blind"
    responses_dir = output_dir / "responses"
    blind_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))
    for batch_index in range(batch_count):
        start = batch_index * batch_size
        batch = items[start : start + batch_size]
        filename = f"batch-{batch_index + 1:02d}.jsonl"
        _write_jsonl(
            blind_dir / filename,
            [item.model_dump(mode="json") for item, _, _ in batch],
        )
        _write_jsonl(
            responses_dir / filename,
            [entry.model_dump(mode="json") for _, entry, _ in batch],
        )
    return {
        "status": "initialized_pending_human_review",
        "review_pass": review_pass,
        "item_count": len(items),
        "batch_size": batch_size,
        "batch_count": batch_count,
        "output_dir": str(output_dir),
    }


def load_review_manifest(review_dir: Path) -> ReviewWorkspaceManifest:
    path = review_dir / "manifest.json"
    if not path.is_file():
        raise ReviewContractError(f"review manifest missing: {path}")
    try:
        return ReviewWorkspaceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ReviewContractError(f"invalid review manifest: {path}") from exc


def load_blind_batch(review_dir: Path, batch_number: int) -> list[BlindReviewItem]:
    path = review_dir / "blind" / f"batch-{batch_number:02d}.jsonl"
    if not path.is_file():
        raise ReviewContractError(f"blind review batch missing: {path}")
    return [
        item for item in _load_jsonl(path, BlindReviewItem) if isinstance(item, BlindReviewItem)
    ]


def load_response_batch(review_dir: Path, batch_number: int) -> list[ReviewWorksheetEntry]:
    path = review_dir / "responses" / f"batch-{batch_number:02d}.jsonl"
    if not path.is_file():
        raise ReviewContractError(f"review response batch missing: {path}")
    return [
        item
        for item in _load_jsonl(path, ReviewWorksheetEntry)
        if isinstance(item, ReviewWorksheetEntry)
    ]


def write_response_batch(
    review_dir: Path, batch_number: int, entries: Sequence[ReviewWorksheetEntry]
) -> None:
    path = review_dir / "responses" / f"batch-{batch_number:02d}.jsonl"
    _write_jsonl(path, [entry.model_dump(mode="json") for entry in entries])


def validate_review_label(
    label: IndependentReviewLabel, item: BlindReviewItem, taxonomy: Taxonomy
) -> None:
    known_intents = {intent.name for intent in taxonomy.intents}
    if label.target_intent is not None and label.target_intent not in known_intents:
        raise ReviewContractError(f"unknown review target intent: {label.target_intent}")
    quote = normalize_evidence_text(label.evidence_quote)
    carriers = [item.context.state_summary]
    carriers.extend(turn.content for turn in item.context.conversation)
    if not any(quote in normalize_evidence_text(carrier) for carrier in carriers):
        raise ReviewContractError(
            f"{item.review_item_id}: evidence quote is not a contiguous context substring"
        )


def _replace_response_batches_with_rollback(
    *,
    review_dir: Path,
    batches: Sequence[Sequence[ReviewWorksheetEntry]],
    receipt_path: Path,
    receipt: ReviewWorkbookImportReceipt,
) -> None:
    response_paths = [
        review_dir / "responses" / f"batch-{batch_number:02d}.jsonl"
        for batch_number in range(1, len(batches) + 1)
    ]
    backups = {path: path.read_bytes() for path in response_paths}
    previous_receipt = receipt_path.read_bytes() if receipt_path.is_file() else None
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    replaced: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".workbook-import-", dir=review_dir) as raw_temp:
        temp_dir = Path(raw_temp)
        staged_paths: list[Path] = []
        for batch_number, entries in enumerate(batches, 1):
            staged = temp_dir / f"batch-{batch_number:02d}.jsonl"
            staged.write_bytes(_jsonl_bytes([entry.model_dump(mode="json") for entry in entries]))
            staged_paths.append(staged)
        staged_receipt = temp_dir / "receipt.json"
        staged_receipt.write_text(
            f"{_canonical_json(receipt.model_dump(mode='json'))}\n", encoding="utf-8"
        )
        try:
            for staged, target in zip(staged_paths, response_paths, strict=True):
                os.replace(staged, target)
                replaced.append(target)
            os.replace(staged_receipt, receipt_path)
        except OSError:
            for target in replaced:
                rollback = temp_dir / f"rollback-{target.name}"
                rollback.write_bytes(backups[target])
                os.replace(rollback, target)
            if previous_receipt is None:
                receipt_path.unlink(missing_ok=True)
            else:
                rollback_receipt = temp_dir / "rollback-receipt.json"
                rollback_receipt.write_bytes(previous_receipt)
                os.replace(rollback_receipt, receipt_path)
            raise


def import_review_workbook(
    *,
    workbook_path: Path,
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    """Validate a completed blind-review workbook before replacing any response batch."""

    if workbook_path.suffix.casefold() != ".xlsx":
        raise ReviewContractError("review workbook must use the .xlsx format")
    inspect_review_workspace(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        review_dir=review_dir,
    )
    manifest = load_review_manifest(review_dir)
    taxonomy = load_taxonomy(taxonomy_path)
    cases = load_candidate_cases(cases_path)
    validate_candidate_cases(cases, taxonomy)
    expected = build_review_items(cases, manifest.review_pass)
    workbook = _load_xlsx_cells(workbook_path, ("盲审", "选项与回传"))
    review_cells = workbook["盲审"]
    metadata_cells = workbook["选项与回传"]

    for reference, value in review_cells.items():
        if _nonblank(value) is None:
            continue
        match = _CELL_REFERENCE.fullmatch(reference)
        assert match is not None
        column_name, row_text = match.groups()
        if int(row_text) > manifest.item_count + 1 or _xlsx_column_number(column_name) > 13:
            raise ReviewContractError("review workbook contains data outside the registered table")

    actual_headers = tuple(_nonblank(review_cells.get(f"{column}1")) for column in "ABCDEFGHIJKLM")
    if actual_headers != REVIEW_WORKBOOK_HEADERS:
        raise ReviewContractError("review workbook headers differ from the registered contract")
    metadata = {
        _nonblank(metadata_cells.get(f"E{row}")): metadata_cells.get(f"F{row}")
        for row in range(2, 11)
    }
    if None in metadata:
        raise ReviewContractError("review workbook contains a blank metadata key")
    expected_metadata: dict[str, object] = {
        "workbook_schema_version": REVIEW_WORKBOOK_SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "review_pass": manifest.review_pass,
        "item_count": manifest.item_count,
        "taxonomy_version": manifest.taxonomy_version,
        "candidate_artifact_sha256": manifest.candidate_artifact_sha256,
        "taxonomy_artifact_sha256": manifest.taxonomy_artifact_sha256,
        "shuffle_method": manifest.shuffle_method,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ReviewContractError(f"review workbook metadata differs for {key}")
    reviewer_id = _nonblank(metadata.get("reviewer_id"))
    if reviewer_id is None:
        raise ReviewContractError("review workbook metadata requires reviewer_id")

    workbook_reviewed_at = datetime.fromtimestamp(workbook_path.stat().st_mtime, tz=timezone.utc)
    entries_by_batch: list[list[ReviewWorksheetEntry]] = [[] for _ in range(manifest.batch_count)]
    imported_count = 0
    preserved_existing_count = 0
    existing_by_id = {
        entry.review_item_id: entry
        for batch_number in range(1, manifest.batch_count + 1)
        for entry in load_response_batch(review_dir, batch_number)
    }
    known_intents = {intent.name for intent in taxonomy.intents}
    for index, (item, _, _) in enumerate(expected, 1):
        row = index + 1
        expected_batch = (index - 1) // manifest.batch_size + 1
        batch_value = review_cells.get(f"A{row}")
        if batch_value != expected_batch:
            raise ReviewContractError(f"row {row}: batch number differs from review manifest")
        review_item_id = _nonblank(review_cells.get(f"B{row}"))
        if review_item_id != item.review_item_id:
            raise ReviewContractError(f"row {row}: review item order or coverage drift")
        if _nonblank(review_cells.get(f"C{row}")) != item.context.state_summary:
            raise ReviewContractError(f"{item.review_item_id}: state_summary drift")
        if (_nonblank(review_cells.get(f"D{row}")) or "") != _format_conversation(item):
            raise ReviewContractError(f"{item.review_item_id}: conversation drift")
        expected_recent = ", ".join(item.context.recent_activities)
        if (_nonblank(review_cells.get(f"E{row}")) or "") != expected_recent:
            raise ReviewContractError(f"{item.review_item_id}: recent_activities drift")

        decision_raw = _nonblank(review_cells.get(f"F{row}"))
        decision = REVIEW_WORKBOOK_DECISIONS.get(decision_raw or "")
        if decision is None:
            raise ReviewContractError(f"{item.review_item_id}: missing or invalid decision")
        target_intent = _nonblank(review_cells.get(f"G{row}"))
        if target_intent is not None and target_intent not in known_intents:
            raise ReviewContractError(
                f"{item.review_item_id}: unknown target intent {target_intent}"
            )
        evidence_quote = _nonblank(review_cells.get(f"H{row}"))
        if evidence_quote is None:
            raise ReviewContractError(f"{item.review_item_id}: missing evidence_quote")
        horizon_raw = _nonblank(review_cells.get(f"K{row}"))
        try:
            horizon = Horizon(horizon_raw)
        except ValueError as exc:
            raise ReviewContractError(f"{item.review_item_id}: missing or invalid horizon") from exc
        label = IndependentReviewLabel(
            decision=decision,
            target_intent=target_intent,
            evidence_quote=evidence_quote,
            slots=Slots(
                desired_experience=_nonblank(review_cells.get(f"I{row}")),
                object=_nonblank(review_cells.get(f"J{row}")),
                horizon=horizon,
            ),
        )
        validate_review_label(label, item, taxonomy)
        review_note = _nonblank(review_cells.get(f"L{row}"))
        status = _nonblank(review_cells.get(f"M{row}"))
        if status is not None and status != "完成":
            raise ReviewContractError(
                f"{item.review_item_id}: workbook status is not complete: {status}"
            )

        existing = existing_by_id[item.review_item_id]
        if existing.is_complete:
            if (
                existing.reviewer_id != reviewer_id
                or existing.label != label
                or existing.review_note != review_note
            ):
                raise ReviewContractError(
                    f"{item.review_item_id}: workbook conflicts with an existing response"
                )
            entry = existing
            preserved_existing_count += 1
        else:
            payload = existing.model_dump(mode="json")
            payload.update(
                {
                    "reviewer_id": reviewer_id,
                    "reviewed_at": workbook_reviewed_at.isoformat(),
                    "label": label.model_dump(mode="json"),
                    "review_note": review_note,
                }
            )
            entry = ReviewWorksheetEntry.model_validate(payload)
            imported_count += 1
        entries_by_batch[expected_batch - 1].append(entry)

    workbook_sha256 = sha256_file(workbook_path)
    receipt = ReviewWorkbookImportReceipt(
        dataset_id=manifest.dataset_id,
        review_pass=manifest.review_pass,
        workbook_sha256=workbook_sha256,
        source_workbook_filename=workbook_path.name,
        reviewer_id=reviewer_id,
        reviewed_at=workbook_reviewed_at,
        imported_at=datetime.now(timezone.utc),
        item_count=manifest.item_count,
        imported_count=imported_count,
        preserved_existing_count=preserved_existing_count,
    )
    receipt_path = review_dir / "imports" / f"{workbook_sha256}.json"
    base_report: dict[str, object] = {
        "status": "valid_complete_workbook" if dry_run else "imported_complete_workbook",
        "dry_run": dry_run,
        "workbook_sha256": workbook_sha256,
        "reviewer_id": reviewer_id,
        "reviewed_at": workbook_reviewed_at.isoformat(),
        "reviewed_at_source": receipt.reviewed_at_source,
        "item_count": manifest.item_count,
        "imported_count": imported_count,
        "preserved_existing_count": preserved_existing_count,
        "receipt_path": str(receipt_path),
    }
    if dry_run:
        return base_report

    if receipt_path.is_file():
        try:
            existing_receipt = ReviewWorkbookImportReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ReviewContractError(f"invalid workbook import receipt: {receipt_path}") from exc
        if existing_receipt.workbook_sha256 != workbook_sha256:
            raise ReviewContractError("workbook import receipt hash mismatch")
        completed_report = inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
            require_complete=True,
        )
        base_report["status"] = "already_imported"
        base_report["workspace"] = completed_report
        return base_report

    _replace_response_batches_with_rollback(
        review_dir=review_dir,
        batches=entries_by_batch,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    completed_report = inspect_review_workspace(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        review_dir=review_dir,
        require_complete=True,
    )
    base_report["workspace"] = completed_report
    return base_report


ReviewComparison = tuple[
    ReviewWorksheetEntry,
    BlindReviewItem,
    DraftCase,
    bool,
    bool,
    bool,
    bool,
]


def _completed_review_comparisons(
    *, cases_path: Path, taxonomy_path: Path, review_dir: Path
) -> list[ReviewComparison]:
    inspect_review_workspace(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        review_dir=review_dir,
        require_complete=True,
    )
    manifest = load_review_manifest(review_dir)
    cases = load_candidate_cases(cases_path)
    expected = build_review_items(cases, manifest.review_pass)
    entries = [
        entry
        for batch_number in range(1, manifest.batch_count + 1)
        for entry in load_response_batch(review_dir, batch_number)
    ]
    comparisons: list[ReviewComparison] = []
    for (item, _, case), entry in zip(expected, entries, strict=True):
        if item.review_item_id != entry.review_item_id or entry.label is None:
            raise ReviewContractError("completed review order or label coverage drift")
        decision_matches = entry.label.decision is case.gold.decision
        target_matches = entry.label.target_intent == case.gold.target_intent
        horizon_matches = entry.label.slots.horizon is case.gold.slots.horizon
        hierarchical_matches = decision_matches and (
            case.gold.decision is not Decision.IN_SCOPE or target_matches
        )
        comparisons.append(
            (
                entry,
                item,
                case,
                decision_matches,
                target_matches,
                horizon_matches,
                hierarchical_matches,
            )
        )
    return comparisons


def _adjudication_audit_sample(
    agreements: Sequence[ReviewComparison], count: int
) -> list[ReviewComparison]:
    if count < 1 or len(agreements) < count:
        raise ReviewContractError("invalid adjudication agreement-audit sample size")

    def stable_key(comparison: ReviewComparison) -> str:
        return hashlib.sha256(
            f"adjudication-audit-v1|{comparison[0].review_item_id}".encode()
        ).hexdigest()

    ordered = sorted(agreements, key=stable_key)
    selected: list[ReviewComparison] = []
    selected_ids: set[str] = set()

    def select_one(predicate: Callable[[ReviewComparison], bool]) -> None:
        for comparison in ordered:
            item_id = comparison[0].review_item_id
            if item_id not in selected_ids and predicate(comparison):
                selected.append(comparison)
                selected_ids.add(item_id)
                return

    for decision in Decision:

        def has_decision(
            comparison: ReviewComparison, expected_decision: Decision = decision
        ) -> bool:
            label = comparison[0].label
            return label is not None and label.decision is expected_decision

        select_one(has_decision)
    for tag in (
        "brandless_xhs",
        "rest_eat_confusion",
        "near_oos",
        "far_oos",
        "context_control",
        "context_distractor",
        "hard_negative",
        "quiet_control",
        "multi_turn",
        "single_turn",
        "slot_required",
    ):

        def has_tag(comparison: ReviewComparison, expected_tag: str = tag) -> bool:
            return expected_tag in comparison[2].tags

        select_one(has_tag)
    for comparison in ordered:
        if len(selected) == count:
            break
        if comparison[0].review_item_id not in selected_ids:
            selected.append(comparison)
            selected_ids.add(comparison[0].review_item_id)
    if len(selected) != count:
        raise ReviewContractError("could not construct the adjudication audit sample")
    return selected


def _adjudication_packet_item(
    *,
    packet_index: int,
    comparison: ReviewComparison,
    sample_kind: AdjudicationSampleKind,
) -> AdjudicationPacketItem:
    (
        entry,
        item,
        case,
        decision_matches,
        target_matches,
        horizon_matches,
        hierarchical_matches,
    ) = comparison
    assert entry.label is not None
    if sample_kind == "agreement_audit":
        review_focus: AdjudicationReviewFocus = "agreement_audit"
        recommendation: AdjudicationRecommendation = "confirm_agreement"
        reason = (
            "人工标签与草稿的 decision、in-scope target 和 horizon 一致; 用于抽样检查共同偏差。"
        )
    elif "near_oos" in case.tags:
        review_focus = "near_oos_grounding_boundary"
        recommendation = "grounding_evidence_required"
        reason = (
            "先依据绑定的 Kindred Activity/Action grounding 证据裁决; "
            "只有运行时合同缺失或互相冲突时才升级为产品策略问题。"
        )
    elif decision_matches and not target_matches:
        review_focus = "target_selection"
        recommendation = "likely_reviewer_input_error"
        reason = "decision 一致但 target 与上下文及当前 taxonomy 不一致, 优先检查是否为下拉误选。"
    else:
        review_focus = "decision_rule_boundary"
        recommendation = "current_rule_lean_keep_draft"
        reason = "依据当前书面 decision 规则倾向保留草稿, 但规则本身仍需人工确认。"
    return AdjudicationPacketItem(
        packet_item_id=f"adjudication-{packet_index:04d}",
        sample_kind=sample_kind,
        review_item_id=entry.review_item_id,
        candidate_id=case.id,
        context=item.context,
        human_label=entry.label,
        human_review_note=entry.review_note,
        draft_label=case.gold,
        draft_annotation_note=case.annotation_note,
        tags=case.tags,
        hard_negative_against=case.hard_negative_against,
        scenario_family_id=case.scenario_family_id,
        contrast_group_id=case.contrast_group_id,
        decision_matches=decision_matches,
        target_matches=target_matches,
        horizon_matches=horizon_matches,
        hierarchical_matches=hierarchical_matches,
        review_focus=review_focus,
        recommendation=recommendation,
        recommendation_reason=reason,
        requires_user_decision=sample_kind == "hierarchical_disagreement",
    )


def build_adjudication_packet(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    output_dir: Path,
    agreement_audit_count: int = 15,
) -> dict[str, object]:
    """Disclose only the completed-review comparisons selected for adjudication."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReviewContractError(f"adjudication output directory is not empty: {output_dir}")
    manifest = load_review_manifest(review_dir)
    comparisons = _completed_review_comparisons(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        review_dir=review_dir,
    )
    disagreements = [comparison for comparison in comparisons if not comparison[6]]
    agreements = [comparison for comparison in comparisons if comparison[6]]
    disagreements.sort(
        key=lambda comparison: (
            "near_oos" not in comparison[2].tags,
            comparison[0].review_item_id,
        )
    )
    audit_sample = _adjudication_audit_sample(agreements, agreement_audit_count)
    selected: list[tuple[ReviewComparison, AdjudicationSampleKind]] = [
        *((comparison, "hierarchical_disagreement") for comparison in disagreements),
        *((comparison, "agreement_audit") for comparison in audit_sample),
    ]
    packet = [
        _adjudication_packet_item(
            packet_index=index,
            comparison=comparison,
            sample_kind=sample_kind,
        )
        for index, (comparison, sample_kind) in enumerate(selected, 1)
    ]

    response_digest = hashlib.sha256()
    for batch_number in range(1, manifest.batch_count + 1):
        response_path = review_dir / "responses" / f"batch-{batch_number:02d}.jsonl"
        response_digest.update(response_path.name.encode())
        response_digest.update(b"\0")
        response_digest.update(response_path.read_bytes())
    packet_manifest = AdjudicationPacketManifest(
        dataset_id=manifest.dataset_id,
        review_pass=manifest.review_pass,
        candidate_artifact_sha256=manifest.candidate_artifact_sha256,
        taxonomy_artifact_sha256=manifest.taxonomy_artifact_sha256,
        response_artifact_sha256=response_digest.hexdigest(),
        hierarchical_disagreement_count=len(disagreements),
        agreement_audit_count=len(audit_sample),
        item_count=len(packet),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "packet.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(packet_path, [item.model_dump(mode="json") for item in packet])
    _write_json(manifest_path, packet_manifest.model_dump(mode="json"))
    return {
        "status": "pending_adjudication",
        "item_count": len(packet),
        "hierarchical_disagreement_count": len(disagreements),
        "agreement_audit_count": len(audit_sample),
        "grounding_evidence_required_count": sum(
            item.recommendation == "grounding_evidence_required" for item in packet
        ),
        "packet_path": str(packet_path),
        "manifest_path": str(manifest_path),
    }


def load_adjudication_manifest(packet_dir: Path) -> AdjudicationPacketManifest:
    path = packet_dir / "manifest.json"
    if not path.is_file():
        raise ReviewContractError(f"adjudication manifest missing: {path}")
    try:
        return AdjudicationPacketManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ReviewContractError(f"invalid adjudication manifest: {path}") from exc


def load_adjudication_packet(packet_dir: Path) -> list[AdjudicationPacketItem]:
    path = packet_dir / "packet.jsonl"
    if not path.is_file():
        raise ReviewContractError(f"adjudication packet missing: {path}")
    items: list[AdjudicationPacketItem] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise ReviewContractError(f"{path}:{line_number}: blank JSONL lines are forbidden")
        try:
            items.append(AdjudicationPacketItem.model_validate_json(raw_line))
        except ValidationError as exc:
            raise ReviewContractError(f"{path}:{line_number}: invalid adjudication row") from exc
    return items


def load_adjudication_resolutions(packet_dir: Path) -> list[AdjudicationResolution]:
    path = packet_dir / "resolutions.jsonl"
    if not path.is_file():
        raise ReviewContractError(f"adjudication resolutions missing: {path}")
    resolutions: list[AdjudicationResolution] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise ReviewContractError(f"{path}:{line_number}: blank JSONL lines are forbidden")
        try:
            resolutions.append(AdjudicationResolution.model_validate_json(raw_line))
        except ValidationError as exc:
            raise ReviewContractError(f"{path}:{line_number}: invalid resolution row") from exc
    return resolutions


def _review_response_artifact_sha256(review_dir: Path, batch_count: int) -> str:
    digest = hashlib.sha256()
    for batch_number in range(1, batch_count + 1):
        response_path = review_dir / "responses" / f"batch-{batch_number:02d}.jsonl"
        if not response_path.is_file():
            raise ReviewContractError(f"review response batch missing: {response_path}")
        digest.update(response_path.name.encode())
        digest.update(b"\0")
        digest.update(response_path.read_bytes())
    return digest.hexdigest()


def _format_adjudication_slots(slots: Slots) -> str:
    values = []
    if slots.desired_experience is not None:
        values.append(f"experience={slots.desired_experience}")
    if slots.object is not None:
        values.append(f"object={slots.object}")
    return "\n".join(values)


def _format_adjudication_conversation(context: Context) -> str:
    return "\n".join(
        f"{'Kindred' if turn.role == 'assistant' else 'User'}: {turn.content}"
        for turn in context.conversation
    )


def _controlled_label(label: IndependentReviewLabel | Gold) -> tuple[Decision, str | None, Horizon]:
    return label.decision, label.target_intent, label.slots.horizon


def _validate_adjudication_packet_alignment(
    *,
    packet: Sequence[AdjudicationPacketItem],
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
) -> None:
    comparisons = _completed_review_comparisons(
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        review_dir=review_dir,
    )
    by_review_id = {comparison[0].review_item_id: comparison for comparison in comparisons}
    for item in packet:
        comparison = by_review_id.get(item.review_item_id)
        if comparison is None:
            raise ReviewContractError(
                f"{item.packet_item_id}: review item is absent from the completed review"
            )
        entry, blind_item, case, *_ = comparison
        if entry.label is None:
            raise ReviewContractError(f"{item.packet_item_id}: review label is incomplete")
        expected_values = (
            case.id,
            blind_item.context,
            entry.label,
            entry.review_note,
            case.gold,
            case.annotation_note,
            case.tags,
            case.hard_negative_against,
            case.scenario_family_id,
            case.contrast_group_id,
        )
        actual_values = (
            item.candidate_id,
            item.context,
            item.human_label,
            item.human_review_note,
            item.draft_label,
            item.draft_annotation_note,
            item.tags,
            item.hard_negative_against,
            item.scenario_family_id,
            item.contrast_group_id,
        )
        if actual_values != expected_values:
            raise ReviewContractError(f"{item.packet_item_id}: packet differs from bound artifacts")


def _policy_impact_for_resolution(
    item: AdjudicationPacketItem, action: str
) -> tuple[AdjudicationPolicyImpact, str | None]:
    if action != "接受人工标签":
        return "none", None
    if item.recommendation == "grounding_evidence_required":
        return (
            "taxonomy_or_decision_rule_review_required",
            "final label rejects the packet's runtime-grounding recommendation",
        )
    if item.recommendation == "current_rule_lean_keep_draft":
        return (
            "taxonomy_or_decision_rule_review_required",
            "final label rejects the packet's current written decision-rule recommendation",
        )
    return "none", None


def _write_adjudication_import(
    *,
    packet_dir: Path,
    resolutions_path: Path,
    resolutions: Sequence[AdjudicationResolution],
    receipt_path: Path,
    receipt: AdjudicationWorkbookImportReceipt,
) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".adjudication-import-", dir=packet_dir) as raw_temp:
        temp_dir = Path(raw_temp)
        staged_resolutions = temp_dir / "resolutions.jsonl"
        staged_receipt = temp_dir / "receipt.json"
        staged_resolutions.write_bytes(
            _jsonl_bytes([resolution.model_dump(mode="json") for resolution in resolutions])
        )
        staged_receipt.write_text(
            f"{_canonical_json(receipt.model_dump(mode='json'))}\n", encoding="utf-8"
        )
        os.replace(staged_resolutions, resolutions_path)
        try:
            os.replace(staged_receipt, receipt_path)
        except OSError:
            resolutions_path.unlink(missing_ok=True)
            raise


def import_adjudication_workbook(
    *,
    workbook_path: Path,
    cases_path: Path,
    taxonomy_path: Path,
    grounding_path: Path,
    packet_dir: Path,
    adjudicator_id: str,
    dry_run: bool = False,
) -> dict[str, object]:
    """Validate and import a hash-bound, completed adjudication workbook."""

    if workbook_path.suffix.casefold() != ".xlsx":
        raise ReviewContractError("adjudication workbook must use the .xlsx format")
    if not adjudicator_id.strip():
        raise ReviewContractError("adjudicator_id must be non-blank")
    manifest = load_adjudication_manifest(packet_dir)
    packet = load_adjudication_packet(packet_dir)
    if len(packet) != manifest.item_count:
        raise ReviewContractError("adjudication packet count differs from manifest")
    if len({item.packet_item_id for item in packet}) != len(packet):
        raise ReviewContractError("adjudication packet contains duplicate packet item IDs")
    if len({item.candidate_id for item in packet}) != len(packet):
        raise ReviewContractError("adjudication packet contains duplicate candidate IDs")

    review_dir = packet_dir.parent
    review_manifest = load_review_manifest(review_dir)
    if review_manifest.dataset_id != manifest.dataset_id:
        raise ReviewContractError("adjudication and review dataset IDs differ")
    if review_manifest.review_pass != manifest.review_pass:
        raise ReviewContractError("adjudication and review passes differ")
    if sha256_file(cases_path) != manifest.candidate_artifact_sha256:
        raise ReviewContractError("candidate artifact hash differs from adjudication manifest")
    if sha256_file(taxonomy_path) != manifest.taxonomy_artifact_sha256:
        raise ReviewContractError("taxonomy artifact hash differs from adjudication manifest")
    response_sha256 = _review_response_artifact_sha256(review_dir, review_manifest.batch_count)
    if response_sha256 != manifest.response_artifact_sha256:
        raise ReviewContractError("review response hash differs from adjudication manifest")
    taxonomy = load_taxonomy(taxonomy_path)
    grounding = load_grounding_snapshot(grounding_path)
    grounding_sha256 = sha256_file(grounding_path)
    _validate_adjudication_packet_alignment(
        packet=packet,
        cases_path=cases_path,
        taxonomy_path=taxonomy_path,
        review_dir=review_dir,
    )

    workbook = _load_xlsx_cells(workbook_path, ("仲裁工作台", "选项与元数据"))
    workbench_cells = workbook["仲裁工作台"]
    metadata_cells = workbook["选项与元数据"]
    for reference, value in workbench_cells.items():
        if _nonblank(value) is None:
            continue
        match = _CELL_REFERENCE.fullmatch(reference)
        assert match is not None
        column_name, row_text = match.groups()
        if int(row_text) > manifest.item_count + 1 or _xlsx_column_number(column_name) > 31:
            raise ReviewContractError(
                "adjudication workbook contains data outside the registered table"
            )
    actual_headers = tuple(
        _nonblank(workbench_cells.get(f"{column}1"))
        for column in (*tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "AA", "AB", "AC", "AD", "AE")
    )
    if actual_headers != ADJUDICATION_WORKBOOK_HEADERS:
        raise ReviewContractError("adjudication workbook headers differ from the contract")

    metadata = {
        _nonblank(metadata_cells.get(f"F{row}")): metadata_cells.get(f"G{row}")
        for row in range(2, 15)
    }
    if None in metadata:
        raise ReviewContractError("adjudication workbook contains a blank metadata key")
    expected_metadata: dict[str, object] = {
        "workbook_schema_version": ADJUDICATION_WORKBOOK_SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "review_pass": manifest.review_pass,
        "selection_method": manifest.selection_method,
        "candidate_artifact_sha256": manifest.candidate_artifact_sha256,
        "taxonomy_artifact_sha256": manifest.taxonomy_artifact_sha256,
        "grounding_version": grounding.grounding_version,
        "grounding_artifact_sha256": grounding_sha256,
        "response_artifact_sha256": manifest.response_artifact_sha256,
        "hierarchical_disagreement_count": manifest.hierarchical_disagreement_count,
        "agreement_audit_count": manifest.agreement_audit_count,
        "item_count": manifest.item_count,
        "status": manifest.status,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ReviewContractError(f"adjudication workbook metadata differs for {key}")

    workbook_sha256 = sha256_file(workbook_path)
    adjudicated_at = datetime.fromtimestamp(workbook_path.stat().st_mtime, tz=timezone.utc)
    known_intents = {intent.name for intent in taxonomy.intents}
    resolutions: list[AdjudicationResolution] = []
    for index, item in enumerate(packet, 1):
        row = index + 1
        immutable_expected = (
            item.packet_item_id,
            item.sample_kind,
            item.review_focus,
            item.review_item_id,
            item.candidate_id,
            item.context.state_summary,
            _format_adjudication_conversation(item.context),
            ", ".join(item.context.recent_activities),
            item.human_label.decision.value,
            item.human_label.target_intent or "",
            item.human_label.slots.horizon.value,
            item.human_label.evidence_quote,
            _format_adjudication_slots(item.human_label.slots),
            item.draft_label.decision.value,
            item.draft_label.target_intent or "",
            item.draft_label.slots.horizon.value,
            item.draft_label.evidence_quote,
            _format_adjudication_slots(item.draft_label.slots),
            ", ".join(item.tags),
            ", ".join(item.hard_negative_against),
        )
        actual_immutable = tuple(
            _nonblank(workbench_cells.get(f"{column}{row}")) or ""
            for column in (*tuple("ABCDEFGHIJKLMNOPQRST"),)
        )
        if actual_immutable != immutable_expected:
            raise ReviewContractError(f"{item.packet_item_id}: immutable workbook fields drifted")
        if _nonblank(workbench_cells.get(f"U{row}")) is None:
            raise ReviewContractError(f"{item.packet_item_id}: recommendation is missing")
        if _nonblank(workbench_cells.get(f"V{row}")) is None:
            raise ReviewContractError(f"{item.packet_item_id}: recommendation reason is missing")

        action = _nonblank(workbench_cells.get(f"W{row}"))
        if action not in ADJUDICATION_ACTIONS:
            raise ReviewContractError(f"{item.packet_item_id}: missing or invalid action")
        rationale = _nonblank(workbench_cells.get(f"AA{row}"))
        if rationale is None:
            raise ReviewContractError(f"{item.packet_item_id}: missing adjudication rationale")
        taxonomy_change_note = _nonblank(workbench_cells.get(f"AB{row}"))
        final_decision_raw = _nonblank(workbench_cells.get(f"X{row}"))
        final_target = _nonblank(workbench_cells.get(f"Y{row}"))
        final_horizon_raw = _nonblank(workbench_cells.get(f"Z{row}"))
        final_decision: Decision | None = None
        final_horizon: Horizon | None = None
        if action == "重写或删除样本":
            if any(
                value is not None for value in (final_decision_raw, final_target, final_horizon_raw)
            ):
                raise ReviewContractError(
                    f"{item.packet_item_id}: rewrite/delete must clear final controlled labels"
                )
        else:
            if final_decision_raw is None:
                raise ReviewContractError(f"{item.packet_item_id}: missing final decision")
            try:
                final_decision = Decision(final_decision_raw)
            except ValueError as exc:
                raise ReviewContractError(
                    f"{item.packet_item_id}: missing or invalid final decision"
                ) from exc
            if final_horizon_raw is None:
                raise ReviewContractError(f"{item.packet_item_id}: missing final horizon")
            try:
                final_horizon = Horizon(final_horizon_raw)
            except ValueError as exc:
                raise ReviewContractError(
                    f"{item.packet_item_id}: missing or invalid final horizon"
                ) from exc
            if final_decision is Decision.IN_SCOPE:
                if final_target not in known_intents:
                    raise ReviewContractError(
                        f"{item.packet_item_id}: in_scope requires a known final target"
                    )
            elif final_target is not None:
                raise ReviewContractError(
                    f"{item.packet_item_id}: non-in_scope final label must clear target"
                )

        final_controlled = (final_decision, final_target, final_horizon)
        if action == "确认一致标签":
            if item.sample_kind != "agreement_audit":
                raise ReviewContractError(
                    f"{item.packet_item_id}: only agreement-audit rows may confirm agreement"
                )
            if _controlled_label(item.human_label) != _controlled_label(item.draft_label):
                raise ReviewContractError(
                    f"{item.packet_item_id}: labels are not fully equal, so cannot confirm"
                )
            if final_controlled != _controlled_label(item.human_label):
                raise ReviewContractError(
                    f"{item.packet_item_id}: confirmed final label differs from agreement"
                )
        elif item.sample_kind == "agreement_audit":
            raise ReviewContractError(
                f"{item.packet_item_id}: agreement-audit rows must confirm the common label"
            )
        elif action == "接受人工标签" and final_controlled != _controlled_label(item.human_label):
            raise ReviewContractError(
                f"{item.packet_item_id}: final label differs from the accepted human label"
            )
        elif action == "保留草稿标签" and final_controlled != _controlled_label(item.draft_label):
            raise ReviewContractError(
                f"{item.packet_item_id}: final label differs from the retained draft label"
            )
        if action == "修改分类规则" and taxonomy_change_note is None:
            raise ReviewContractError(
                f"{item.packet_item_id}: taxonomy modification requires an explanation"
            )

        cached_status = _nonblank(workbench_cells.get(f"AC{row}"))
        if cached_status is not None and cached_status != "完成":
            raise ReviewContractError(
                f"{item.packet_item_id}: workbook status cache is not complete: {cached_status}"
            )
        policy_impact, policy_impact_reason = _policy_impact_for_resolution(item, action)
        resolutions.append(
            AdjudicationResolution.model_validate(
                {
                    "packet_item_id": item.packet_item_id,
                    "candidate_id": item.candidate_id,
                    "action": action,
                    "final_decision": final_decision,
                    "final_target_intent": final_target,
                    "final_horizon": final_horizon,
                    "rationale": rationale,
                    "taxonomy_change_note": taxonomy_change_note,
                    "adjudicator_id": adjudicator_id.strip(),
                    "adjudicated_at": adjudicated_at,
                    "source_workbook_sha256": workbook_sha256,
                    "policy_impact": policy_impact,
                    "policy_impact_reason": policy_impact_reason,
                }
            )
        )

    resolutions_bytes = _jsonl_bytes(
        [resolution.model_dump(mode="json") for resolution in resolutions]
    )
    resolutions_sha256 = hashlib.sha256(resolutions_bytes).hexdigest()
    action_counts: dict[str, int] = {
        action: count
        for action, count in sorted(
            Counter(resolution.action for resolution in resolutions).items()
        )
    }
    policy_review_candidate_ids = [
        resolution.candidate_id for resolution in resolutions if resolution.policy_impact != "none"
    ]
    receipt = AdjudicationWorkbookImportReceipt(
        dataset_id=manifest.dataset_id,
        review_pass=manifest.review_pass,
        workbook_sha256=workbook_sha256,
        source_workbook_filename=workbook_path.name,
        packet_artifact_sha256=sha256_file(packet_dir / "packet.jsonl"),
        candidate_artifact_sha256=manifest.candidate_artifact_sha256,
        taxonomy_artifact_sha256=manifest.taxonomy_artifact_sha256,
        grounding_version=grounding.grounding_version,
        grounding_artifact_sha256=grounding_sha256,
        response_artifact_sha256=manifest.response_artifact_sha256,
        adjudicator_id=adjudicator_id.strip(),
        adjudicated_at=adjudicated_at,
        imported_at=datetime.now(timezone.utc),
        item_count=manifest.item_count,
        action_counts=action_counts,
        resolution_artifact_sha256=resolutions_sha256,
        policy_review_candidate_ids=policy_review_candidate_ids,
    )
    resolutions_path = packet_dir / "resolutions.jsonl"
    receipt_path = packet_dir / "imports" / f"{workbook_sha256}.json"
    report: dict[str, object] = {
        "status": (
            "valid_complete_adjudication_workbook"
            if dry_run
            else "imported_complete_adjudication_workbook"
        ),
        "dry_run": dry_run,
        "workbook_sha256": workbook_sha256,
        "adjudicator_id": adjudicator_id.strip(),
        "adjudicated_at": adjudicated_at.isoformat(),
        "adjudicated_at_source": receipt.adjudicated_at_source,
        "item_count": manifest.item_count,
        "action_counts": action_counts,
        "policy_review_candidate_ids": policy_review_candidate_ids,
        "resolutions_path": str(resolutions_path),
        "receipt_path": str(receipt_path),
    }
    if dry_run:
        return report

    if receipt_path.is_file():
        try:
            existing_receipt = AdjudicationWorkbookImportReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ReviewContractError(
                f"invalid adjudication import receipt: {receipt_path}"
            ) from exc
        if not resolutions_path.is_file():
            raise ReviewContractError("adjudication receipt exists without resolutions")
        if sha256_file(resolutions_path) != existing_receipt.resolution_artifact_sha256:
            raise ReviewContractError("adjudication resolutions differ from existing receipt")
        if existing_receipt.workbook_sha256 != workbook_sha256:
            raise ReviewContractError("adjudication receipt workbook hash mismatch")
        report["status"] = "already_imported"
        return report
    if resolutions_path.exists():
        raise ReviewContractError(
            "adjudication resolutions already exist for a different workbook; "
            "preserve and review them"
        )
    _write_adjudication_import(
        packet_dir=packet_dir,
        resolutions_path=resolutions_path,
        resolutions=resolutions,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    return report


def inspect_review_workspace(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    require_complete: bool = False,
) -> dict[str, object]:
    manifest = load_review_manifest(review_dir)
    if sha256_file(cases_path) != manifest.candidate_artifact_sha256:
        raise ReviewContractError("candidate artifact hash differs from review manifest")
    if sha256_file(taxonomy_path) != manifest.taxonomy_artifact_sha256:
        raise ReviewContractError("taxonomy artifact hash differs from review manifest")

    taxonomy = load_taxonomy(taxonomy_path)
    if taxonomy.taxonomy_version != manifest.taxonomy_version:
        raise ReviewContractError("taxonomy version differs from review manifest")
    cases = load_candidate_cases(cases_path)
    validate_candidate_cases(cases, taxonomy)
    expected = build_review_items(cases, manifest.review_pass)
    if len(expected) != manifest.item_count:
        raise ReviewContractError("review item count differs from manifest")

    completed_entries: list[tuple[ReviewWorksheetEntry, DraftCase]] = []
    pending_by_batch: dict[str, int] = {}
    seen_item_ids: set[str] = set()
    for batch_number in range(1, manifest.batch_count + 1):
        start = (batch_number - 1) * manifest.batch_size
        expected_batch = expected[start : start + manifest.batch_size]
        blind_items = load_blind_batch(review_dir, batch_number)
        response_entries = load_response_batch(review_dir, batch_number)
        expected_blind = [item for item, _, _ in expected_batch]
        expected_ids = [item.review_item_id for item in expected_blind]
        if blind_items != expected_blind:
            raise ReviewContractError(f"batch-{batch_number:02d}: blind packet drift")
        if [entry.review_item_id for entry in response_entries] != expected_ids:
            raise ReviewContractError(f"batch-{batch_number:02d}: response coverage drift")
        if any(entry.review_pass != manifest.review_pass for entry in response_entries):
            raise ReviewContractError(f"batch-{batch_number:02d}: review pass mismatch")
        if seen_item_ids & set(expected_ids):
            raise ReviewContractError("review item appears in multiple batches")
        seen_item_ids.update(expected_ids)

        case_by_item_id = {item.review_item_id: case for item, _, case in expected_batch}
        item_by_id = {item.review_item_id: item for item in blind_items}
        pending = 0
        for entry in response_entries:
            if not entry.is_complete:
                pending += 1
                continue
            assert entry.label is not None
            validate_review_label(entry.label, item_by_id[entry.review_item_id], taxonomy)
            completed_entries.append((entry, case_by_item_id[entry.review_item_id]))
        pending_by_batch[f"batch-{batch_number:02d}"] = pending

    if len(seen_item_ids) != manifest.item_count:
        raise ReviewContractError("review workspace does not cover the full candidate set")
    pending_count = manifest.item_count - len(completed_entries)
    if require_complete and pending_count:
        raise ReviewContractError(f"review pass still has {pending_count} pending items")

    reviewer_ids = sorted(
        {entry.reviewer_id for entry, _ in completed_entries if entry.reviewer_id is not None}
    )
    report: dict[str, object] = {
        "status": "complete" if pending_count == 0 else "pending_human_review",
        "review_pass": manifest.review_pass,
        "item_count": manifest.item_count,
        "completed_count": len(completed_entries),
        "pending_count": pending_count,
        "pending_by_batch": pending_by_batch,
        "reviewer_ids": reviewer_ids,
    }
    if pending_count:
        report["agreement_status"] = "locked_until_pass_complete"
        return report

    decision_agreement = 0
    target_agreement = 0
    horizon_agreement = 0
    hierarchical_agreement = 0
    evidence_exact_match = 0
    desired_experience_exact_match = 0
    object_exact_match = 0
    hierarchical_disagreement_item_ids: list[str] = []
    for entry, case in completed_entries:
        assert entry.label is not None
        decision_matches = entry.label.decision is case.gold.decision
        target_matches = entry.label.target_intent == case.gold.target_intent
        horizon_matches = entry.label.slots.horizon is case.gold.slots.horizon
        hierarchical_matches = decision_matches and (
            case.gold.decision is not Decision.IN_SCOPE or target_matches
        )
        evidence_matches = normalize_evidence_text(
            entry.label.evidence_quote
        ) == normalize_evidence_text(case.gold.evidence_quote)
        desired_experience_matches = (
            entry.label.slots.desired_experience == case.gold.slots.desired_experience
        )
        object_matches = entry.label.slots.object == case.gold.slots.object
        decision_agreement += decision_matches
        target_agreement += target_matches
        horizon_agreement += horizon_matches
        hierarchical_agreement += hierarchical_matches
        evidence_exact_match += evidence_matches
        desired_experience_exact_match += desired_experience_matches
        object_exact_match += object_matches
        if not hierarchical_matches:
            hierarchical_disagreement_item_ids.append(entry.review_item_id)

    report["agreement_status"] = "revealed_after_pass_complete"
    report["reference_label_kind"] = "llm_assisted_draft_label"
    report["comparison_purpose"] = "dataset_qa_not_model_accuracy"
    report["agreement_counts"] = {
        "decision": decision_agreement,
        "target_intent": target_agreement,
        "horizon": horizon_agreement,
        "hierarchical": hierarchical_agreement,
    }
    report["free_text_exact_match_counts"] = {
        "evidence_quote": evidence_exact_match,
        "desired_experience": desired_experience_exact_match,
        "object": object_exact_match,
    }
    report["hierarchical_disagreement_item_ids"] = hierarchical_disagreement_item_ids
    report["disagreement_item_ids"] = hierarchical_disagreement_item_ids
    return report


def review_horizon_values() -> tuple[str, ...]:
    return tuple(horizon.value for horizon in Horizon)
