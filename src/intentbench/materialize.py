"""Hash-bound IE1.2 adjudication materialization and relationship audit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from intentbench.annotation import AdjudicatedCase, DraftCase
from intentbench.dataset import load_candidate_cases, validate_candidate_cases
from intentbench.freeze import sha256_file
from intentbench.review import (
    AdjudicationPacketItem,
    AdjudicationResolution,
    AdjudicationWorkbookImportReceipt,
    BlindReviewItem,
    ReviewContractError,
    ReviewWorksheetEntry,
    build_review_items,
    inspect_review_workspace,
    load_adjudication_manifest,
    load_adjudication_packet,
    load_adjudication_resolutions,
    load_response_batch,
    load_review_manifest,
)
from intentbench.schemas import (
    Decision,
    Gold,
    Horizon,
    OpenIntentCandidate,
    ReviewProvenance,
    Slots,
    StrictModel,
)
from intentbench.taxonomy import load_taxonomy


class MaterializationContractError(ValueError):
    """Adjudication or policy artifacts cannot safely produce pre-split Gold."""


class PolicyImpactResolution(StrictModel):
    """Approved dual-channel handling for one imported policy-impact decision."""

    policy_resolution_id: str = Field(pattern=r"^policy-[0-9]{4}$")
    packet_item_id: str = Field(pattern=r"^adjudication-[0-9]{4}$")
    candidate_id: str
    routing_action: Literal["retain_draft_label"]
    open_intent_candidate: OpenIntentCandidate
    rationale: str = Field(min_length=1)


class PolicyResolutionPlan(StrictModel):
    """Binds the user-approved dual-channel strategy to exact imported resolutions."""

    schema_version: Literal[1] = 1
    dataset_id: str = Field(pattern=r"^kir-pilot-v[0-9]+$")
    strategy: Literal["routing-plus-open-intent-v1"]
    source_resolution_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    resolutions: list[PolicyImpactResolution] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> PolicyResolutionPlan:
        if self.approved_at.tzinfo is None:
            raise ValueError("policy approval timestamp must include a timezone")
        ids = [resolution.policy_resolution_id for resolution in self.resolutions]
        candidates = [resolution.candidate_id for resolution in self.resolutions]
        packet_items = [resolution.packet_item_id for resolution in self.resolutions]
        if len(ids) != len(set(ids)):
            raise ValueError("policy resolution IDs must be unique")
        if len(candidates) != len(set(candidates)):
            raise ValueError("policy resolution candidate IDs must be unique")
        if len(packet_items) != len(set(packet_items)):
            raise ValueError("policy resolution packet IDs must be unique")
        return self


class RelationshipAudit(StrictModel):
    status: Literal["valid"] = "valid"
    scenario_family_count: int = Field(ge=1)
    contrast_group_count: int = Field(ge=1)
    paraphrase_cluster_count: int = Field(ge=1)
    connected_component_count: int = Field(ge=1)
    largest_component_size: int = Field(ge=1)
    near_oos_pair_count: int = Field(ge=0)
    context_pair_count: int = Field(ge=0)


class MaterializationReceipt(StrictModel):
    """Provenance and audit summary for one deterministic pre-split Gold artifact."""

    schema_version: Literal[1] = 1
    dataset_id: str = Field(pattern=r"^kir-pilot-v[0-9]+$")
    candidate_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjudication_resolution_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_resolution_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_at: datetime
    case_count: int = Field(ge=1)
    routing_label_source_counts: dict[str, int]
    decision_counts: dict[str, int]
    open_intent_candidate_count: int = Field(ge=0)
    open_intent_candidate_ids: list[str]
    semantic_tag_audit: Literal["valid"] = "valid"
    relationship_audit: RelationshipAudit

    @model_validator(mode="after")
    def validate_receipt(self) -> MaterializationReceipt:
        if self.materialized_at.tzinfo is None:
            raise ValueError("materialized_at must include a timezone")
        if sum(self.routing_label_source_counts.values()) != self.case_count:
            raise ValueError("routing-label source counts must cover every case")
        if sum(self.decision_counts.values()) != self.case_count:
            raise ValueError("decision counts must cover every case")
        if self.open_intent_candidate_count != len(self.open_intent_candidate_ids):
            raise ValueError("open-intent count and IDs differ")
        if len(self.open_intent_candidate_ids) != len(set(self.open_intent_candidate_ids)):
            raise ValueError("open-intent candidate IDs must be unique")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl_bytes(values: Sequence[object]) -> bytes:
    return ("\n".join(_canonical_json(value) for value in values) + "\n").encode()


def load_policy_resolution_plan(path: Path) -> PolicyResolutionPlan:
    if not path.is_file():
        raise MaterializationContractError(f"policy resolution plan missing: {path}")
    try:
        return PolicyResolutionPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise MaterializationContractError(f"invalid policy resolution plan: {path}") from exc


def load_adjudicated_cases(path: Path) -> list[AdjudicatedCase]:
    cases: list[AdjudicatedCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise MaterializationContractError(
                f"{path}:{line_number}: blank JSONL lines are forbidden"
            )
        try:
            cases.append(AdjudicatedCase.model_validate_json(raw_line))
        except ValidationError as exc:
            raise MaterializationContractError(
                f"{path}:{line_number}: invalid adjudicated case"
            ) from exc
    return cases


def _validate_policy_alignment(
    *,
    plan: PolicyResolutionPlan,
    resolutions: Sequence[AdjudicationResolution],
    packet: Sequence[AdjudicationPacketItem],
    cases: Sequence[DraftCase],
) -> dict[str, PolicyImpactResolution]:
    resolution_by_candidate = {resolution.candidate_id: resolution for resolution in resolutions}
    packet_by_candidate = {item.candidate_id: item for item in packet}
    case_by_id = {case.id: case for case in cases}
    expected_policy_ids = {
        resolution.candidate_id for resolution in resolutions if resolution.policy_impact != "none"
    }
    actual_policy_ids = {resolution.candidate_id for resolution in plan.resolutions}
    if actual_policy_ids != expected_policy_ids:
        raise MaterializationContractError(
            "policy resolutions must cover every imported policy impact exactly once"
        )
    for policy in plan.resolutions:
        resolution = resolution_by_candidate.get(policy.candidate_id)
        packet_item = packet_by_candidate.get(policy.candidate_id)
        case = case_by_id.get(policy.candidate_id)
        if resolution is None or packet_item is None or case is None:
            raise MaterializationContractError(
                f"{policy.policy_resolution_id}: candidate is absent from bound artifacts"
            )
        if policy.packet_item_id != resolution.packet_item_id:
            raise MaterializationContractError(
                f"{policy.policy_resolution_id}: packet item differs from imported resolution"
            )
        if policy.packet_item_id != packet_item.packet_item_id:
            raise MaterializationContractError(
                f"{policy.policy_resolution_id}: packet item differs from adjudication packet"
            )
        quote = policy.open_intent_candidate.evidence_quote
        carriers = [case.context.state_summary]
        carriers.extend(turn.content for turn in case.context.conversation)
        if not any(quote in carrier for carrier in carriers):
            raise MaterializationContractError(
                f"{policy.policy_resolution_id}: open-intent evidence is not in context"
            )
    return {resolution.candidate_id: resolution for resolution in plan.resolutions}


def _final_controlled_label(
    *,
    case: DraftCase,
    review_decision: Decision,
    review_target: str | None,
    review_horizon: Horizon,
    resolution: AdjudicationResolution | None,
    policy: PolicyImpactResolution | None,
) -> tuple[Decision, str | None, Horizon, str]:
    review_controlled = (review_decision, review_target, review_horizon)
    draft_controlled = (
        case.gold.decision,
        case.gold.target_intent,
        case.gold.slots.horizon,
    )
    if resolution is None:
        if review_controlled != draft_controlled:
            raise MaterializationContractError(
                f"{case.id}: unresolved review disagreement is absent from adjudication"
            )
        return (*review_controlled, "blind_human_review")
    if policy is not None:
        return (*draft_controlled, "policy_dual_channel_draft")
    if resolution.action == "保留草稿标签":
        return (*draft_controlled, "adjudicated_draft")
    if resolution.action in {"确认一致标签", "接受人工标签"}:
        if resolution.final_decision is None or resolution.final_horizon is None:
            raise MaterializationContractError(f"{case.id}: resolution lacks final label")
        source = (
            "blind_human_review" if resolution.action == "确认一致标签" else "adjudicated_human"
        )
        return (
            resolution.final_decision,
            resolution.final_target_intent,
            resolution.final_horizon,
            source,
        )
    raise MaterializationContractError(
        f"{case.id}: action {resolution.action!r} cannot be materialized automatically"
    )


def _relationship_audit(cases: Sequence[AdjudicatedCase]) -> RelationshipAudit:
    parent = {case.id: case.id for case in cases}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    relation_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in cases:
        relation_members[("scenario", case.scenario_family_id)].append(case.id)
        relation_members[("contrast", case.contrast_group_id)].append(case.id)
        relation_members[("paraphrase", case.paraphrase_cluster_id)].append(case.id)
    for members in relation_members.values():
        for member in members[1:]:
            union(members[0], member)
    component_sizes = Counter(find(case.id) for case in cases)
    return RelationshipAudit(
        scenario_family_count=len({case.scenario_family_id for case in cases}),
        contrast_group_count=len({case.contrast_group_id for case in cases}),
        paraphrase_cluster_count=len({case.paraphrase_cluster_id for case in cases}),
        connected_component_count=len(component_sizes),
        largest_component_size=max(component_sizes.values()),
        near_oos_pair_count=sum("near_oos" in case.tags for case in cases),
        context_pair_count=sum("context_control" in case.tags for case in cases),
    )


def _validate_materialized_structure(
    cases: Sequence[AdjudicatedCase], taxonomy_path: Path
) -> dict[str, object]:
    pending_equivalents: list[DraftCase] = []
    for case in cases:
        payload = case.model_dump(mode="json")
        payload.pop("open_intent_candidate")
        payload.pop("review_provenance")
        payload["source"] = "llm_assisted_pending_human_review"
        payload["adjudication_status"] = "draft"
        pending_equivalents.append(DraftCase.model_validate(payload))
    return validate_candidate_cases(pending_equivalents, load_taxonomy(taxonomy_path))


def _load_adjudication_import_receipt(
    packet_dir: Path, resolutions: Sequence[AdjudicationResolution]
) -> AdjudicationWorkbookImportReceipt:
    workbook_hashes = {resolution.source_workbook_sha256 for resolution in resolutions}
    if len(workbook_hashes) != 1:
        raise MaterializationContractError("resolutions reference multiple source workbooks")
    workbook_hash = next(iter(workbook_hashes))
    path = packet_dir / "imports" / f"{workbook_hash}.json"
    if not path.is_file():
        raise MaterializationContractError(f"adjudication import receipt missing: {path}")
    try:
        return AdjudicationWorkbookImportReceipt.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except ValidationError as exc:
        raise MaterializationContractError(f"invalid adjudication import receipt: {path}") from exc


def materialize_adjudicated_cases(
    *,
    cases_path: Path,
    taxonomy_path: Path,
    review_dir: Path,
    packet_dir: Path,
    policy_resolution_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Materialize reviewed pre-split Gold without changing the frozen routing distribution."""

    try:
        inspect_review_workspace(
            cases_path=cases_path,
            taxonomy_path=taxonomy_path,
            review_dir=review_dir,
            require_complete=True,
        )
        review_manifest = load_review_manifest(review_dir)
        packet_manifest = load_adjudication_manifest(packet_dir)
        packet = load_adjudication_packet(packet_dir)
        resolutions = load_adjudication_resolutions(packet_dir)
    except ReviewContractError as exc:
        raise MaterializationContractError(str(exc)) from exc
    if len(resolutions) != packet_manifest.item_count:
        raise MaterializationContractError("resolution count differs from adjudication manifest")
    if {resolution.packet_item_id for resolution in resolutions} != {
        item.packet_item_id for item in packet
    }:
        raise MaterializationContractError("resolutions do not cover the adjudication packet")

    resolution_sha256 = sha256_file(packet_dir / "resolutions.jsonl")
    adjudication_receipt = _load_adjudication_import_receipt(packet_dir, resolutions)
    if adjudication_receipt.resolution_artifact_sha256 != resolution_sha256:
        raise MaterializationContractError("resolution artifact differs from its import receipt")
    plan = load_policy_resolution_plan(policy_resolution_path)
    if plan.dataset_id != packet_manifest.dataset_id:
        raise MaterializationContractError("policy plan dataset differs from adjudication")
    if plan.source_resolution_artifact_sha256 != resolution_sha256:
        raise MaterializationContractError("policy plan is bound to different resolutions")

    taxonomy = load_taxonomy(taxonomy_path)
    cases = load_candidate_cases(cases_path)
    validate_candidate_cases(cases, taxonomy)
    policy_by_candidate = _validate_policy_alignment(
        plan=plan,
        resolutions=resolutions,
        packet=packet,
        cases=cases,
    )
    resolution_by_candidate = {resolution.candidate_id: resolution for resolution in resolutions}
    expected = build_review_items(cases, review_manifest.review_pass)
    review_entries = [
        entry
        for batch_number in range(1, review_manifest.batch_count + 1)
        for entry in load_response_batch(review_dir, batch_number)
    ]
    if len(review_entries) != len(expected):
        raise MaterializationContractError("review responses do not cover every candidate")

    review_by_candidate: dict[str, tuple[BlindReviewItem, ReviewWorksheetEntry]] = {}
    for (blind_item, _, reviewed_case), entry in zip(expected, review_entries, strict=True):
        if entry.review_item_id != blind_item.review_item_id or entry.label is None:
            raise MaterializationContractError(
                f"{reviewed_case.id}: review response is incomplete or reordered"
            )
        review_by_candidate[reviewed_case.id] = (blind_item, entry)

    materialized: list[AdjudicatedCase] = []
    for case in cases:
        _, entry = review_by_candidate[case.id]
        assert entry.label is not None
        resolution = resolution_by_candidate.get(case.id)
        policy = policy_by_candidate.get(case.id)
        decision, target, horizon, label_source = _final_controlled_label(
            case=case,
            review_decision=entry.label.decision,
            review_target=entry.label.target_intent,
            review_horizon=entry.label.slots.horizon,
            resolution=resolution,
            policy=policy,
        )
        near_oos_siblings = case.gold.near_oos_sibling_intents if decision is Decision.OOS else []
        gold = Gold(
            decision=decision,
            target_intent=target,
            evidence_quote=case.gold.evidence_quote,
            near_oos_sibling_intents=near_oos_siblings,
            slots=Slots(
                desired_experience=case.gold.slots.desired_experience,
                object=case.gold.slots.object,
                horizon=horizon,
            ),
        )
        provenance_payload: dict[str, object] = {
            "review_item_id": entry.review_item_id,
            "reviewer_id": entry.reviewer_id,
            "reviewed_at": entry.reviewed_at,
            "routing_label_source": label_source,
            "supporting_fields_source": "llm_assisted_draft",
        }
        if resolution is not None:
            provenance_payload.update(
                {
                    "adjudication_packet_item_id": resolution.packet_item_id,
                    "adjudication_action": resolution.action,
                    "adjudicator_id": resolution.adjudicator_id,
                    "adjudicated_at": resolution.adjudicated_at,
                }
            )
        if policy is not None:
            provenance_payload["policy_resolution_id"] = policy.policy_resolution_id
        payload = case.model_dump(mode="json")
        payload.update(
            {
                "gold": gold.model_dump(mode="json"),
                "open_intent_candidate": (
                    policy.open_intent_candidate.model_dump(mode="json")
                    if policy is not None
                    else None
                ),
                "review_provenance": ReviewProvenance.model_validate(provenance_payload).model_dump(
                    mode="json"
                ),
                "source": "llm_assisted_human_reviewed",
                "adjudication_status": "adjudicated",
            }
        )
        try:
            materialized.append(AdjudicatedCase.model_validate(payload))
        except ValidationError as exc:
            raise MaterializationContractError(f"{case.id}: materialized Gold is invalid") from exc

    structural_report = _validate_materialized_structure(materialized, taxonomy_path)
    relationship_audit = _relationship_audit(materialized)
    case_payloads = [case.model_dump(mode="json") for case in materialized]
    cases_bytes = _jsonl_bytes(case_payloads)
    cases_sha256 = hashlib.sha256(cases_bytes).hexdigest()
    label_source_counts: dict[str, int] = dict(
        sorted(
            Counter(case.review_provenance.routing_label_source for case in materialized).items()
        )
    )
    decision_counts: dict[str, int] = {
        decision.value: sum(case.gold.decision is decision for case in materialized)
        for decision in Decision
    }
    open_intent_ids = [case.id for case in materialized if case.open_intent_candidate is not None]
    policy_sha256 = sha256_file(policy_resolution_path)
    cases_output_path = output_dir / "cases.jsonl"
    receipt_path = output_dir / "materialization-receipt.json"
    report: dict[str, object] = {
        "status": "materialized_adjudicated_pre_split_gold",
        "case_count": len(materialized),
        "decision_counts": decision_counts,
        "routing_label_source_counts": label_source_counts,
        "open_intent_candidate_ids": open_intent_ids,
        "semantic_tag_audit": "valid",
        "relationship_audit": relationship_audit.model_dump(mode="json"),
        "cases_path": str(cases_output_path),
        "receipt_path": str(receipt_path),
        "cases_sha256": cases_sha256,
        "candidate_validation_status": structural_report["status"],
    }

    if cases_output_path.is_file() or receipt_path.is_file():
        if not cases_output_path.is_file() or not receipt_path.is_file():
            raise MaterializationContractError("materialization output is partial")
        try:
            existing_receipt = MaterializationReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise MaterializationContractError(
                f"invalid materialization receipt: {receipt_path}"
            ) from exc
        expected_hashes = (
            sha256_file(cases_path),
            sha256_file(taxonomy_path),
            resolution_sha256,
            policy_sha256,
            cases_sha256,
        )
        actual_hashes = (
            existing_receipt.candidate_artifact_sha256,
            existing_receipt.taxonomy_artifact_sha256,
            existing_receipt.adjudication_resolution_artifact_sha256,
            existing_receipt.policy_resolution_artifact_sha256,
            existing_receipt.materialized_cases_sha256,
        )
        if actual_hashes != expected_hashes or sha256_file(cases_output_path) != cases_sha256:
            raise MaterializationContractError(
                "existing materialization differs from current inputs"
            )
        report["status"] = "already_materialized"
        return report
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MaterializationContractError(
            f"materialization output directory is not empty: {output_dir}"
        )

    receipt = MaterializationReceipt(
        dataset_id=packet_manifest.dataset_id,
        candidate_artifact_sha256=sha256_file(cases_path),
        taxonomy_artifact_sha256=sha256_file(taxonomy_path),
        response_artifact_sha256=packet_manifest.response_artifact_sha256,
        adjudication_resolution_artifact_sha256=resolution_sha256,
        policy_resolution_artifact_sha256=policy_sha256,
        materialized_cases_sha256=cases_sha256,
        materialized_at=datetime.now(timezone.utc),
        case_count=len(materialized),
        routing_label_source_counts=label_source_counts,
        decision_counts=decision_counts,
        open_intent_candidate_count=len(open_intent_ids),
        open_intent_candidate_ids=open_intent_ids,
        relationship_audit=relationship_audit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".materialize-", dir=output_dir) as raw_temp:
        temp_dir = Path(raw_temp)
        staged_cases = temp_dir / "cases.jsonl"
        staged_receipt = temp_dir / "materialization-receipt.json"
        staged_cases.write_bytes(cases_bytes)
        staged_receipt.write_text(
            f"{_canonical_json(receipt.model_dump(mode='json'))}\n", encoding="utf-8"
        )
        os.replace(staged_cases, cases_output_path)
        try:
            os.replace(staged_receipt, receipt_path)
        except OSError:
            cases_output_path.unlink(missing_ok=True)
            raise
    return report


__all__ = [
    "MaterializationContractError",
    "PolicyResolutionPlan",
    "load_adjudicated_cases",
    "load_policy_resolution_plan",
    "materialize_adjudicated_cases",
]
