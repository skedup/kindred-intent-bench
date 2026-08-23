"""Hash-bound import for the completed IE4 human semantic-preservation audit."""

from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from intentbench.freeze import DatasetFreezeManifest, sha256_file
from intentbench.reporting import load_formal_cases
from intentbench.schemas import PredictionStatus, StrictModel
from intentbench.two_stage import FormedIntentionRecord
from intentbench.xlsx import CELL_REFERENCE, XlsxReadError, load_xlsx_cells, xlsx_column_number

Rating = Literal["preserved", "partial", "lost"]
AuditRole = Literal["primary_decision", "weak_decision"]
AUDIT_HEADERS = (
    "audit_row_id",
    "case_id",
    "model_role",
    "model",
    "state_summary",
    "conversation",
    "recent_activities",
    "source_evidence_quote",
    "tags",
    "stage_a_status",
    "error_type",
    "evidence_status",
    "formed_action",
    "formed_object",
    "desired_experience",
    "qualifiers",
    "alternative_actions",
    "formed_horizon",
    "reason_short",
    "action_rating",
    "object_rating",
    "horizon_rating",
    "reviewer_note",
    "reviewer_id",
    "reviewed_at",
    "row_status",
)
IMMUTABLE_HEADER_COUNT = 19
AUDIT_LIST_SEPARATOR = "\uff1b"
MODEL_DISPLAY: dict[AuditRole, str] = {
    "primary_decision": "Gemini 3.6 Flash",
    "weak_decision": "DeepSeek V4 Flash",
}


class SemanticAuditError(ValueError):
    """The completed semantic-audit workbook violates its frozen contract."""


class AIAssistanceDisclosure(StrictModel):
    score_authority: Literal[False] = False
    scores_overwritten: Literal[False] = False
    allowed_effects: tuple[
        Literal[
            "completeness_check",
            "reviewed_at_date_type_fix",
            "row_status_formula_fill",
            "nonbinding_rubric_disagreement_logging",
        ],
        ...,
    ]
    nonbinding_disagreements: dict[
        Literal["object_slot_boundary", "horizon_reference_frame"], list[str]
    ]
    disposition: Literal["human_scores_preserved_as_submitted"]

    @model_validator(mode="after")
    def validate_disagreements(self) -> AIAssistanceDisclosure:
        expected_effects = (
            "completeness_check",
            "reviewed_at_date_type_fix",
            "row_status_formula_fill",
            "nonbinding_rubric_disagreement_logging",
        )
        if self.allowed_effects != expected_effects:
            raise ValueError("AI QA allowed effects drifted")
        expected = {"object_slot_boundary", "horizon_reference_frame"}
        if set(self.nonbinding_disagreements) != expected:
            raise ValueError("AI QA disagreement categories drifted")
        ids = [case_id for values in self.nonbinding_disagreements.values() for case_id in values]
        if len(ids) != 10 or len(set(ids)) != 10:
            raise ValueError("AI QA disclosure must contain ten unique audit row IDs")
        return self


class ReportingConstraints(StrictModel):
    affects_tri_state_verdict: Literal[False] = False
    claim_inter_annotator_agreement: Literal[False] = False
    claim_independent_adjudication: Literal[False] = False
    disclose_single_reviewer: Literal[True] = True
    disclose_post_label_ai_qa: Literal[True] = True


class SemanticAuditProcess(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["frozen"] = "frozen"
    audit_id: Literal["kir-pilot-v2-ie4-semantic-preservation"]
    label_authority: Literal["single_human_reviewer"]
    human_reviewer_id: str = Field(min_length=1)
    independent_second_human_review: Literal[False] = False
    sequence: Literal["human_ratings_then_post_label_ai_qa"]
    pre_ai_human_label_snapshot_retained: Literal[False] = False
    ai_assistance: AIAssistanceDisclosure
    reporting_constraints: ReportingConstraints


class SemanticAuditRow(StrictModel):
    audit_row_id: str = Field(pattern=r"^semantic-audit-[0-9]{4}$")
    case_id: str
    model_role: AuditRole
    model: str
    state_summary: str
    conversation: str
    recent_activities: str
    source_evidence_quote: str
    tags: str
    stage_a_status: PredictionStatus
    error_type: str
    evidence_status: str
    formed_action: str
    formed_object: str
    desired_experience: str
    qualifiers: str
    alternative_actions: str
    formed_horizon: str
    reason_short: str
    action_rating: Rating
    object_rating: Rating
    horizon_rating: Rating
    reviewer_note: str | None = None
    reviewer_id: str = Field(min_length=1)
    reviewed_at: date
    row_status: Literal["complete"] = "complete"

    @model_validator(mode="after")
    def validate_review(self) -> SemanticAuditRow:
        ratings = (self.action_rating, self.object_rating, self.horizon_rating)
        if any(rating != "preserved" for rating in ratings) and not self.reviewer_note:
            raise ValueError("partial/lost audit ratings require reviewer_note")
        if self.stage_a_status is not PredictionStatus.SUCCESS:
            if ratings != ("lost", "lost", "lost"):
                raise ValueError("failed Stage A rows require three lost ratings")
            if self.reviewer_note != "stage_a_provider_failure":
                raise ValueError("failed Stage A rows require the registered failure note")
        return self


class RatingCounts(StrictModel):
    preserved: int = Field(ge=0)
    partial: int = Field(ge=0)
    lost: int = Field(ge=0)
    total: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_total(self) -> RatingCounts:
        if self.preserved + self.partial + self.lost != self.total:
            raise ValueError("semantic-audit rating counts do not sum to total")
        return self


class SemanticAuditRoleSummary(StrictModel):
    model_role: AuditRole
    model: str
    row_count: Literal[24] = 24
    stage_a_failure_count: int = Field(ge=0, le=24)
    all_fields_preserved_count: int = Field(ge=0, le=24)
    fields: dict[Literal["action", "object", "horizon"], RatingCounts]


class CompletedSemanticAudit(StrictModel):
    rows: list[SemanticAuditRow]
    reviewer_id: str
    reviewed_at: date
    role_summaries: list[SemanticAuditRoleSummary]
    process: SemanticAuditProcess

    @model_validator(mode="after")
    def validate_universe(self) -> CompletedSemanticAudit:
        if len(self.rows) != 48:
            raise ValueError("semantic audit must contain exactly 48 rows")
        if [item.model_role for item in self.role_summaries] != [
            "primary_decision",
            "weak_decision",
        ]:
            raise ValueError("semantic-audit role summaries must use frozen order")
        if self.process.human_reviewer_id != self.reviewer_id:
            raise ValueError("process disclosure reviewer differs from workbook")
        return self


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()
    text = _cell_text(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise SemanticAuditError("reviewed_at must be an Excel or ISO date") from exc


def _load_process(path: Path) -> SemanticAuditProcess:
    try:
        return SemanticAuditProcess.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise SemanticAuditError(f"invalid semantic-audit process disclosure: {path}") from exc


def _load_formed(path: Path) -> dict[str, FormedIntentionRecord]:
    records: list[FormedIntentionRecord] = []
    try:
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                raise SemanticAuditError(f"{path}:{line_number}: blank formed-intention row")
            records.append(FormedIntentionRecord.model_validate_json(raw_line))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise SemanticAuditError(f"invalid formed-intention artifact: {path}") from exc
    by_id = {record.case_id: record for record in records}
    if len(by_id) != len(records):
        raise SemanticAuditError(f"duplicate formed-intention case IDs: {path}")
    return by_id


def _display(value: str | None) -> str:
    return value if value else "—"


def _expected_immutable_rows(
    *,
    cases_path: Path,
    dataset_manifest_path: Path,
    primary_formed_path: Path,
    weak_formed_path: Path,
) -> list[dict[str, str]]:
    try:
        freeze = DatasetFreezeManifest.model_validate_json(
            dataset_manifest_path.read_text(encoding="utf-8")
        )
        cases = {case.id: case for case in load_formal_cases(cases_path)}
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        raise SemanticAuditError("invalid semantic-audit dataset binding") from exc
    audit_ids = freeze.semantic_audit_case_ids
    if len(audit_ids) != 24 or len(set(audit_ids)) != 24:
        raise SemanticAuditError("dataset freeze lacks 24 unique semantic-audit IDs")
    if missing := set(audit_ids) - set(cases):
        raise SemanticAuditError(f"test set lacks frozen semantic-audit cases: {sorted(missing)}")
    formed_by_role = {
        "primary_decision": _load_formed(primary_formed_path),
        "weak_decision": _load_formed(weak_formed_path),
    }
    expected: list[dict[str, str]] = []
    row_number = 1
    for case_id in audit_ids:
        case = cases[case_id]
        conversation = "\n".join(
            f"{'Kindred' if turn.role == 'assistant' else 'User'}: {turn.content}"
            for turn in case.context.conversation
        )
        for model_role in ("primary_decision", "weak_decision"):
            record = formed_by_role[model_role].get(case_id)
            if record is None:
                raise SemanticAuditError(f"{model_role} lacks frozen audit case {case_id}")
            formed = record.formed_intention
            expected.append(
                {
                    "audit_row_id": f"semantic-audit-{row_number:04d}",
                    "case_id": case_id,
                    "model_role": model_role,
                    "model": MODEL_DISPLAY[model_role],
                    "state_summary": case.context.state_summary,
                    "conversation": _display(conversation),
                    "recent_activities": _display(
                        AUDIT_LIST_SEPARATOR.join(case.context.recent_activities)
                    ),
                    "source_evidence_quote": case.gold.evidence_quote,
                    "tags": AUDIT_LIST_SEPARATOR.join(case.tags),
                    "stage_a_status": record.status.value,
                    "error_type": _display(record.error_type),
                    "evidence_status": _display(formed.evidence_status if formed else None),
                    "formed_action": _display(formed.action if formed else None),
                    "formed_object": _display(formed.object if formed else None),
                    "desired_experience": _display(formed.desired_experience if formed else None),
                    "qualifiers": _display(
                        AUDIT_LIST_SEPARATOR.join(formed.qualifiers) if formed else None
                    ),
                    "alternative_actions": _display(
                        AUDIT_LIST_SEPARATOR.join(formed.alternative_actions) if formed else None
                    ),
                    "formed_horizon": _display(formed.horizon.value if formed else None),
                    "reason_short": _display(formed.reason_short if formed else None),
                }
            )
            row_number += 1
    return expected


def _validate_embedded_hashes(
    *,
    options: dict[str, object],
    source_paths: dict[str, Path],
    repository_root: Path,
) -> None:
    for row_number, source_name in zip(range(11, 16), source_paths, strict=True):
        embedded_name = _cell_text(options.get(f"A{row_number}"))
        embedded_path = _cell_text(options.get(f"B{row_number}"))
        embedded_hash = _cell_text(options.get(f"C{row_number}"))
        path = source_paths[source_name]
        try:
            relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError as exc:
            raise SemanticAuditError(f"audit source is outside repository: {path}") from exc
        if (embedded_name, embedded_path, embedded_hash) != (
            source_name,
            relative,
            sha256_file(path),
        ):
            raise SemanticAuditError(f"workbook source binding drifted for {source_name}")


def _rating_counts(rows: list[SemanticAuditRow], field: str) -> RatingCounts:
    counts = Counter(getattr(row, f"{field}_rating") for row in rows)
    return RatingCounts(
        preserved=counts["preserved"],
        partial=counts["partial"],
        lost=counts["lost"],
        total=len(rows),
    )


def _role_summary(role: AuditRole, rows: list[SemanticAuditRow]) -> SemanticAuditRoleSummary:
    selected = [row for row in rows if row.model_role == role]
    if len(selected) != 24:
        raise SemanticAuditError(f"{role} must contain 24 semantic-audit rows")
    return SemanticAuditRoleSummary(
        model_role=role,
        model=MODEL_DISPLAY[role],
        stage_a_failure_count=sum(
            row.stage_a_status is not PredictionStatus.SUCCESS for row in selected
        ),
        all_fields_preserved_count=sum(
            (row.action_rating, row.object_rating, row.horizon_rating)
            == ("preserved", "preserved", "preserved")
            for row in selected
        ),
        fields={
            "action": _rating_counts(selected, "action"),
            "object": _rating_counts(selected, "object"),
            "horizon": _rating_counts(selected, "horizon"),
        },
    )


def load_completed_semantic_audit(
    *,
    workbook_path: Path,
    process_path: Path,
    cases_path: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    primary_formed_path: Path,
    weak_formed_path: Path,
    repository_root: Path,
) -> CompletedSemanticAudit:
    """Validate the XLSX universe, immutable sources, human inputs, and disclosure."""

    try:
        workbook = load_xlsx_cells(workbook_path, ("Guide", "Audit", "Options"))
    except XlsxReadError as exc:
        raise SemanticAuditError(str(exc)) from exc
    audit = workbook["Audit"]
    headers = tuple(_cell_text(audit.get(f"{chr(65 + index)}4")) for index in range(26))
    if headers != AUDIT_HEADERS:
        raise SemanticAuditError("semantic-audit workbook headers differ from contract")
    expected_rows = _expected_immutable_rows(
        cases_path=cases_path,
        dataset_manifest_path=dataset_manifest_path,
        primary_formed_path=primary_formed_path,
        weak_formed_path=weak_formed_path,
    )
    rows: list[SemanticAuditRow] = []
    for index, expected in enumerate(expected_rows, 5):
        values: dict[str, object] = {
            header: _cell_text(audit.get(f"{chr(65 + column)}{index}"))
            for column, header in enumerate(AUDIT_HEADERS)
        }
        for header in AUDIT_HEADERS[:IMMUTABLE_HEADER_COUNT]:
            if _cell_text(values[header]) != expected[header]:
                raise SemanticAuditError(
                    f"{expected['audit_row_id']}: immutable field drifted: {header}"
                )
        values["reviewer_note"] = values["reviewer_note"] or None
        values["reviewed_at"] = _parse_date(audit.get(f"Y{index}"))
        try:
            rows.append(SemanticAuditRow.model_validate(values))
        except ValidationError as exc:
            raise SemanticAuditError(
                f"{expected['audit_row_id']}: invalid human audit row"
            ) from exc
    for reference, value in audit.items():
        match = CELL_REFERENCE.fullmatch(reference)
        assert match is not None
        if _cell_text(value) and (
            xlsx_column_number(match.group(1)) > len(AUDIT_HEADERS) or int(match.group(2)) > 52
        ):
            raise SemanticAuditError("semantic-audit workbook contains cells outside the contract")
    source_paths = {
        "freeze": dataset_manifest_path,
        "split": split_manifest_path,
        "cases": cases_path,
        "primary": primary_formed_path,
        "weak": weak_formed_path,
    }
    _validate_embedded_hashes(
        options=workbook["Options"],
        source_paths=source_paths,
        repository_root=repository_root,
    )
    reviewer_ids = {row.reviewer_id for row in rows}
    reviewed_dates = {row.reviewed_at for row in rows}
    if len(reviewer_ids) != 1 or len(reviewed_dates) != 1:
        raise SemanticAuditError("semantic-audit rows require one reviewer and one review date")
    process = _load_process(process_path)
    disclosed_ids = {
        item
        for values in process.ai_assistance.nonbinding_disagreements.values()
        for item in values
    }
    if not disclosed_ids <= {row.audit_row_id for row in rows}:
        raise SemanticAuditError("AI QA disclosure references unknown audit rows")
    reviewer_id = next(iter(reviewer_ids))
    reviewed_at = next(iter(reviewed_dates))
    guide = workbook["Guide"]
    if _cell_text(guide.get("B11")) != reviewer_id or _parse_date(guide.get("B12")) != reviewed_at:
        raise SemanticAuditError("Guide reviewer metadata differs from Audit rows")
    if tuple(_cell_text(guide.get(cell)) for cell in ("G6", "G7", "G8")) != (
        "48",
        "0",
        "1",
    ):
        raise SemanticAuditError("Guide completion counters do not confirm a complete audit")
    return CompletedSemanticAudit(
        rows=rows,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        role_summaries=[
            _role_summary("primary_decision", rows),
            _role_summary("weak_decision", rows),
        ],
        process=process,
    )


def semantic_audit_csv_bytes(audit: CompletedSemanticAudit) -> bytes:
    """Create the canonical portable audit table from validated human rows."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=AUDIT_HEADERS, lineterminator="\n")
    writer.writeheader()
    for row in audit.rows:
        payload = row.model_dump(mode="json")
        payload["reviewer_note"] = payload["reviewer_note"] or ""
        writer.writerow(payload)
    return output.getvalue().encode("utf-8")
