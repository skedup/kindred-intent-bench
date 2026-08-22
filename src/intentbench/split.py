"""Deterministic IE1.3 relationship-safe split and dataset freeze."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, ValidationError, model_validator

from intentbench.annotation import AdjudicatedCase
from intentbench.bootstrap import (
    MIN_CLUSTERS,
    assign_relationship_clusters,
    validate_split_integrity,
)
from intentbench.freeze import ArtifactDigest, DatasetFreezeManifest, sha256_file
from intentbench.materialize import MaterializationReceipt, load_adjudicated_cases
from intentbench.schemas import (
    DIAGNOSTIC_SLICE_TAGS,
    INTENT_NAMES,
    Case,
    Decision,
    Horizon,
    Split,
    StrictModel,
)
from intentbench.taxonomy import load_taxonomy, validate_cases

DATASET_VERSION: Final = "kir-pilot-v2"
DEV_CASE_COUNT: Final = 48
TEST_CASE_COUNT: Final = 112
STABLE_HASH_NAMESPACE: Final = "kir-pilot-v2-group-split-v1"
SEMANTIC_AUDIT_NAMESPACE: Final = "kir-pilot-v2-semantic-audit-v1"
DEV_DECISION_TARGETS: Final = {
    Decision.IN_SCOPE: 24,
    Decision.OOS: 10,
    Decision.NO_INTENT: 7,
    Decision.AMBIGUOUS: 7,
}
DEV_INTENT_TARGET: Final = 3
DEV_NEAR_OOS_TARGET: Final = 5
DIAGNOSTIC_FEATURES: Final = (
    *DIAGNOSTIC_SLICE_TAGS,
    "open_intent_candidate",
    "horizon_later",
    "horizon_unspecified",
)
OUTPUT_FILENAMES: Final = {
    "dev": "dev.jsonl",
    "test": "test.jsonl",
    "split": "split-manifest.json",
    "dataset_card": "dataset-card.md",
    "freeze": "freeze-manifest.json",
    "checksums": "SHA256SUMS",
}


class DatasetSplitError(ValueError):
    """The reviewed Gold cannot be split or frozen under the IE1.3 contract."""


class GroupAssignment(StrictModel):
    group_id: str = Field(min_length=1)
    split: Split
    case_ids: list[str] = Field(min_length=1)


class SplitStatistics(StrictModel):
    case_count: int = Field(ge=1)
    group_count: int = Field(ge=1)
    decision_counts: dict[str, int]
    intent_counts: dict[str, int]
    horizon_counts: dict[str, int]
    tag_case_counts: dict[str, int]
    tag_cluster_counts: dict[str, int]
    open_intent_candidate_count: int = Field(ge=0)


class SemanticAuditSelection(StrictModel):
    method: Literal["risk-ranked-distinct-cluster-v1"] = "risk-ranked-distinct-cluster-v1"
    case_count: Literal[24] = 24
    decision_targets: dict[str, int]
    in_scope_intent_minimum: Literal[1] = 1
    case_ids: list[str] = Field(min_length=24, max_length=24)

    @model_validator(mode="after")
    def validate_ids(self) -> SemanticAuditSelection:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("semantic-audit case IDs must be unique")
        if sum(self.decision_targets.values()) != self.case_count:
            raise ValueError("semantic-audit decision targets must total 24")
        return self


class SplitManifest(StrictModel):
    schema_version: Literal[1] = 1
    dataset_version: Literal["kir-pilot-v2"] = DATASET_VERSION
    taxonomy_version: str = Field(min_length=1)
    method: Literal["relationship-group-stratified-v1"] = "relationship-group-stratified-v1"
    stable_hash_namespace: Literal["kir-pilot-v2-group-split-v1"] = STABLE_HASH_NAMESPACE
    source_artifacts: dict[str, ArtifactDigest]
    case_targets: dict[str, int]
    dev_decision_targets: dict[str, int]
    dev_intent_target_each: Literal[3] = DEV_INTENT_TARGET
    dev_near_oos_target: Literal[5] = DEV_NEAR_OOS_TARGET
    diagnostic_target_counts: dict[str, int]
    diagnostic_dev_counts: dict[str, int]
    diagnostic_objective: float = Field(ge=0)
    statistics: dict[str, SplitStatistics]
    required_test_cluster_counts: dict[str, int]
    group_assignments: list[GroupAssignment]
    semantic_audit: SemanticAuditSelection

    @model_validator(mode="after")
    def validate_manifest(self) -> SplitManifest:
        if self.case_targets != {"dev": DEV_CASE_COUNT, "test": TEST_CASE_COUNT}:
            raise ValueError("split manifest uses unexpected case targets")
        group_ids = [assignment.group_id for assignment in self.group_assignments]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("split manifest group assignments must be unique")
        if any(count < MIN_CLUSTERS for count in self.required_test_cluster_counts.values()):
            raise ValueError("a required test slice has fewer than eight clusters")
        return self


@dataclass(frozen=True)
class _Group:
    group_id: str
    cases: tuple[Case, ...]

    def count(self, predicate: Callable[[Case], bool]) -> int:
        return sum(predicate(case) for case in self.cases)


@dataclass(frozen=True)
class _Block:
    name: str
    group_ids: tuple[str, ...]
    options: tuple[frozenset[str], ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode()


def _jsonl_bytes(values: Sequence[object]) -> bytes:
    return ("\n".join(_canonical_json(value) for value in values) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_key(value: str, namespace: str = STABLE_HASH_NAMESPACE) -> tuple[str, str]:
    digest = hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()
    return digest, value


def _formal_case(case: AdjudicatedCase) -> Case:
    if not case.id.startswith("draft-"):
        raise DatasetSplitError(f"unexpected adjudicated case ID: {case.id}")
    payload = case.model_dump(mode="json")
    payload["id"] = case.id.removeprefix("draft-")
    payload["split"] = Split.TEST.value
    return Case.model_validate(payload)


def _load_bound_gold(
    cases_path: Path, receipt_path: Path, taxonomy_path: Path
) -> tuple[list[Case], MaterializationReceipt, str]:
    try:
        receipt = MaterializationReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise DatasetSplitError("cannot load the bound adjudicated Gold artifacts") from exc
    cases_sha256 = sha256_file(cases_path)
    taxonomy_sha256 = sha256_file(taxonomy_path)
    if receipt.dataset_id != DATASET_VERSION:
        raise DatasetSplitError("materialization receipt belongs to a different dataset")
    if receipt.materialized_cases_sha256 != cases_sha256:
        raise DatasetSplitError("adjudicated Gold differs from its materialization receipt")
    if receipt.taxonomy_artifact_sha256 != taxonomy_sha256:
        raise DatasetSplitError("taxonomy differs from the materialization receipt")
    try:
        adjudicated = load_adjudicated_cases(cases_path)
    except (OSError, ValidationError, ValueError) as exc:
        raise DatasetSplitError("cannot parse the hash-bound adjudicated Gold") from exc
    if receipt.case_count != len(adjudicated):
        raise DatasetSplitError("adjudicated Gold count differs from its receipt")
    formal = assign_relationship_clusters([_formal_case(case) for case in adjudicated])
    return formal, receipt, taxonomy_sha256


def _build_groups(cases: Sequence[Case]) -> dict[str, _Group]:
    grouped: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        if case.split_group_id is None:
            raise DatasetSplitError(f"{case.id}: missing generated split group")
        grouped[case.split_group_id].append(case)
    return {
        group_id: _Group(group_id, tuple(sorted(members, key=lambda item: item.id)))
        for group_id, members in sorted(grouped.items())
    }


def _exact_options(
    candidates: Sequence[_Group],
    exact_counts: tuple[int, ...],
    counters: Sequence[Callable[[Case], bool]],
) -> tuple[frozenset[str], ...]:
    ordered = sorted(candidates, key=lambda group: _stable_key(group.group_id))
    vectors = [tuple(group.count(counter) for counter in counters) for group in ordered]
    options: list[frozenset[str]] = []

    def visit(index: int, remaining: tuple[int, ...], selected: tuple[str, ...]) -> None:
        if any(value < 0 for value in remaining):
            return
        if index == len(ordered):
            if not any(remaining):
                options.append(frozenset(selected))
            return
        visit(index + 1, remaining, selected)
        vector = vectors[index]
        visit(
            index + 1,
            tuple(left - right for left, right in zip(remaining, vector, strict=True)),
            (*selected, ordered[index].group_id),
        )

    visit(0, exact_counts, ())
    if not options:
        raise DatasetSplitError(f"no group-safe subset satisfies exact counts {exact_counts}")
    return tuple(sorted(options, key=_option_key))


def _option_key(option: frozenset[str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_stable_key(group_id) for group_id in option))


def _group_shape(group: _Group) -> Counter[Decision]:
    return Counter(case.gold.decision for case in group.cases)


def _intent_predicate(intent: str) -> Callable[[Case], bool]:
    return lambda case: (
        case.gold.decision is Decision.IN_SCOPE and case.gold.target_intent == intent
    )


def _decision_predicate(decision: Decision) -> Callable[[Case], bool]:
    return lambda case: case.gold.decision is decision


def _tag_predicate(tag: str) -> Callable[[Case], bool]:
    return lambda case: tag in case.tags


def _build_blocks(groups: dict[str, _Group]) -> list[_Block]:
    near_intents = sorted(INTENT_NAMES, key=_stable_key)[:DEV_NEAR_OOS_TARGET]
    blocks: list[_Block] = []
    owned: set[str] = set()
    for intent in INTENT_NAMES:
        candidates = [
            group
            for group in groups.values()
            if any(
                case.gold.decision is Decision.IN_SCOPE and case.gold.target_intent == intent
                for case in group.cases
            )
        ]
        required_oos = 1 if intent in near_intents else 0
        options = _exact_options(
            candidates,
            (DEV_INTENT_TARGET, required_oos),
            (
                _intent_predicate(intent),
                lambda case: case.gold.decision is Decision.OOS,
            ),
        )
        group_ids = tuple(group.group_id for group in candidates)
        blocks.append(_Block(f"intent:{intent}", group_ids, options))
        owned.update(group_ids)

    decision_specs = (
        (Decision.OOS, DEV_DECISION_TARGETS[Decision.OOS] - DEV_NEAR_OOS_TARGET),
        (Decision.NO_INTENT, DEV_DECISION_TARGETS[Decision.NO_INTENT]),
        (Decision.AMBIGUOUS, DEV_DECISION_TARGETS[Decision.AMBIGUOUS]),
    )
    for decision, target in decision_specs:
        candidates = [
            group
            for group in groups.values()
            if group.group_id not in owned
            and _group_shape(group) == Counter({decision: len(group.cases)})
        ]
        options = _exact_options(
            candidates,
            (target,),
            (_decision_predicate(decision),),
        )
        group_ids = tuple(group.group_id for group in candidates)
        blocks.append(_Block(f"decision:{decision.value}", group_ids, options))
        owned.update(group_ids)

    if owned != set(groups):
        unowned = sorted(set(groups) - owned)
        raise DatasetSplitError(f"unclassified relationship groups: {unowned}")
    return blocks


def _has_feature(case: Case, feature: str) -> bool:
    if feature == "open_intent_candidate":
        return case.open_intent_candidate is not None
    if feature == "horizon_later":
        return case.gold.slots.horizon is Horizon.LATER
    if feature == "horizon_unspecified":
        return case.gold.slots.horizon is Horizon.UNSPECIFIED
    return feature in case.tags


def _diagnostic_counts(selected: Iterable[str], groups: dict[str, _Group]) -> dict[str, int]:
    selected_ids = set(selected)
    return {
        feature: sum(
            _has_feature(case, feature)
            for group_id, group in groups.items()
            if group_id in selected_ids
            for case in group.cases
        )
        for feature in DIAGNOSTIC_FEATURES
    }


def _diagnostic_targets(groups: dict[str, _Group]) -> dict[str, int]:
    total = sum(len(group.cases) for group in groups.values())
    return {
        feature: (count * DEV_CASE_COUNT + total // 2) // total
        for feature, count in _diagnostic_counts(groups, groups).items()
    }


def _diagnostic_objective(actual: dict[str, int], target: dict[str, int]) -> float:
    return sum(
        ((actual[feature] - expected) / max(1, expected)) ** 2
        for feature, expected in target.items()
    )


def _add_vectors(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _diagnostic_vector_for_groups(
    group_ids: Iterable[str], group_vectors: dict[str, tuple[int, ...]]
) -> tuple[int, ...]:
    total = (0,) * len(DIAGNOSTIC_FEATURES)
    for group_id in group_ids:
        total = _add_vectors(total, group_vectors[group_id])
    return total


def _vector_objective(actual: tuple[int, ...], target: tuple[int, ...]) -> float:
    return sum(
        ((value - expected) / max(1, expected)) ** 2
        for value, expected in zip(actual, target, strict=True)
    )


def _select_dev_groups(groups: dict[str, _Group]) -> tuple[set[str], dict[str, int], float]:
    blocks = _build_blocks(groups)
    target = _diagnostic_targets(groups)
    target_vector = tuple(target[feature] for feature in DIAGNOSTIC_FEATURES)
    group_vectors = {
        group_id: tuple(
            sum(_has_feature(case, feature) for case in group.cases)
            for feature in DIAGNOSTIC_FEATURES
        )
        for group_id, group in groups.items()
    }
    option_vectors = {
        block.name: {
            option: _diagnostic_vector_for_groups(option, group_vectors) for option in block.options
        }
        for block in blocks
    }
    selected_by_block = {block.name: block.options[0] for block in blocks}
    for _ in range(20):
        changed = False
        for block in blocks:
            other = set().union(
                *(option for name, option in selected_by_block.items() if name != block.name)
            )
            other_vector = _diagnostic_vector_for_groups(other, group_vectors)
            best = min(
                block.options,
                key=lambda option: (
                    _vector_objective(
                        _add_vectors(other_vector, option_vectors[block.name][option]),
                        target_vector,
                    ),
                    _option_key(option),
                ),
            )
            if best != selected_by_block[block.name]:
                selected_by_block[block.name] = best
                changed = True
        if not changed:
            break
    selected = set().union(*selected_by_block.values())
    actual = _diagnostic_counts(selected, groups)
    return selected, target, _diagnostic_objective(actual, target)


def _with_split(case: Case, split: Split) -> Case:
    return Case.model_validate({**case.model_dump(mode="json"), "split": split.value})


def _split_cases(cases: Sequence[Case], dev_group_ids: set[str]) -> tuple[list[Case], list[Case]]:
    dev: list[Case] = []
    test: list[Case] = []
    for case in cases:
        if case.split_group_id in dev_group_ids:
            dev.append(_with_split(case, Split.DEV))
        else:
            test.append(_with_split(case, Split.TEST))
    return sorted(dev, key=lambda case: case.id), sorted(test, key=lambda case: case.id)


def _cluster_count(cases: Sequence[Case], predicate: Callable[[Case], bool]) -> int:
    return len(
        {
            case.bootstrap_cluster_id
            for case in cases
            if predicate(case) and case.bootstrap_cluster_id is not None
        }
    )


def _statistics(cases: Sequence[Case]) -> SplitStatistics:
    return SplitStatistics(
        case_count=len(cases),
        group_count=len({case.split_group_id for case in cases}),
        decision_counts=dict(sorted(Counter(case.gold.decision.value for case in cases).items())),
        intent_counts=dict(
            sorted(
                Counter(
                    case.gold.target_intent for case in cases if case.gold.target_intent is not None
                ).items()
            )
        ),
        horizon_counts=dict(
            sorted(Counter(case.gold.slots.horizon.value for case in cases).items())
        ),
        tag_case_counts={
            tag: sum(tag in case.tags for case in cases) for tag in DIAGNOSTIC_SLICE_TAGS
        },
        tag_cluster_counts={
            tag: _cluster_count(cases, _tag_predicate(tag)) for tag in DIAGNOSTIC_SLICE_TAGS
        },
        open_intent_candidate_count=sum(case.open_intent_candidate is not None for case in cases),
    )


def _required_test_clusters(test: Sequence[Case]) -> dict[str, int]:
    return {
        "full": _cluster_count(test, lambda case: True),
        "gold_in_scope": _cluster_count(test, lambda case: case.gold.decision is Decision.IN_SCOPE),
        "near_oos": _cluster_count(test, lambda case: "near_oos" in case.tags),
        "no_intent": _cluster_count(test, lambda case: case.gold.decision is Decision.NO_INTENT),
    }


def _audit_risk_key(case: Case) -> tuple[int, tuple[str, str]]:
    diagnostic_tags = set(case.tags) - {"single_turn", "multi_turn"}
    slot_count = sum(
        value is not None for value in (case.gold.slots.desired_experience, case.gold.slots.object)
    )
    risk = (
        len(diagnostic_tags)
        + slot_count
        + 2 * ("multi_turn" in case.tags)
        + 3 * (case.gold.slots.horizon is not Horizon.NOW)
        + 8 * (case.open_intent_candidate is not None)
    )
    return -risk, _stable_key(case.id, SEMANTIC_AUDIT_NAMESPACE)


def _select_semantic_audit(test: Sequence[Case]) -> SemanticAuditSelection:
    selected: list[Case] = []
    selected_ids: set[str] = set()
    selected_groups: set[str | None] = set()

    def add(candidates: Iterable[Case], count: int) -> None:
        available = sorted(
            (case for case in candidates if case.id not in selected_ids),
            key=_audit_risk_key,
        )
        chosen: list[Case] = []
        chosen_groups: set[str | None] = set()
        for case in available:
            group_id = case.bootstrap_cluster_id
            if group_id in selected_groups or group_id in chosen_groups:
                continue
            chosen.append(case)
            chosen_groups.add(group_id)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            raise DatasetSplitError("semantic audit lacks enough distinct test clusters")
        for case in chosen:
            selected.append(case)
            selected_ids.add(case.id)
            selected_groups.add(case.bootstrap_cluster_id)

    for intent in INTENT_NAMES:
        add((case for case in test if case.gold.target_intent == intent), 1)
    add((case for case in test if case.gold.decision is Decision.IN_SCOPE), 4)
    for decision in (Decision.OOS, Decision.NO_INTENT, Decision.AMBIGUOUS):
        add((case for case in test if case.gold.decision is decision), 4)
    ordered_ids = sorted(selected_ids)
    return SemanticAuditSelection(
        decision_targets={
            Decision.IN_SCOPE.value: 12,
            Decision.OOS.value: 4,
            Decision.NO_INTENT.value: 4,
            Decision.AMBIGUOUS.value: 4,
        },
        case_ids=ordered_ids,
    )


def _validate_result(
    *, dev: Sequence[Case], test: Sequence[Case], taxonomy_path: Path
) -> dict[str, int]:
    all_cases = [*dev, *test]
    if len(dev) != DEV_CASE_COUNT or len(test) != TEST_CASE_COUNT:
        raise DatasetSplitError("group-safe split did not produce exact 48/112 case counts")
    validate_split_integrity(all_cases)
    taxonomy = load_taxonomy(taxonomy_path)
    validate_cases(all_cases, taxonomy)
    decisions = Counter(case.gold.decision for case in dev)
    if decisions != Counter(DEV_DECISION_TARGETS):
        raise DatasetSplitError("dev decision distribution differs from its exact targets")
    intents = Counter(
        case.gold.target_intent for case in dev if case.gold.decision is Decision.IN_SCOPE
    )
    if intents != Counter({intent: DEV_INTENT_TARGET for intent in INTENT_NAMES}):
        raise DatasetSplitError("dev intent distribution differs from its exact targets")
    if sum("near_oos" in case.tags for case in dev) != DEV_NEAR_OOS_TARGET:
        raise DatasetSplitError("dev near-OOS count differs from its exact target")
    required_clusters = _required_test_clusters(test)
    too_small = {name: count for name, count in required_clusters.items() if count < MIN_CLUSTERS}
    if too_small:
        raise DatasetSplitError(f"required test slices lack bootstrap clusters: {too_small}")
    return required_clusters


def _relative_to_root(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    root = repository_root.resolve()
    if not resolved.is_relative_to(root):
        raise DatasetSplitError(f"artifact is outside repository root: {path}")
    return resolved.relative_to(root).as_posix()


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    heading = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join((heading, separator, *body))


def _dataset_card(
    *,
    manifest: SplitManifest,
    source_gold_sha256: str,
    dev_sha256: str,
    test_sha256: str,
    split_sha256: str,
) -> str:
    dev = manifest.statistics["dev"]
    test = manifest.statistics["test"]
    decision_rows = [
        (
            decision.value,
            dev.decision_counts.get(decision.value, 0),
            test.decision_counts.get(decision.value, 0),
        )
        for decision in Decision
    ]
    intent_rows = [
        (intent, dev.intent_counts.get(intent, 0), test.intent_counts.get(intent, 0))
        for intent in INTENT_NAMES
    ]
    slice_rows = [
        (
            tag,
            dev.tag_case_counts[tag],
            test.tag_case_counts[tag],
            dev.tag_cluster_counts[tag],
            test.tag_cluster_counts[tag],
        )
        for tag in DIAGNOSTIC_SLICE_TAGS
    ]
    return f"""# KIR Pilot v2 Dataset Card

## 概要

KIR Pilot v2 是 Kindred activity routing 的离线、开放集意图识别评测集。它包含 160 条中文合成场景：dev 48 条、frozen test 112 条。所有条目均经过人工盲审或仲裁，正式 Case 保留逐条 review provenance。

本数据集只用于离线求职作品与方法验证，不接入 Kindred 生产路径，也不代表线上自然流量分布。

## 来源与标注

- 数据由 LLM 辅助起草，再由一名人工 reviewer 完成标签盲审；有分歧的条目经过显式仲裁。
- routing Gold 为 `in_scope / oos / no_intent / ambiguous`；开放目录外想法另存为非路由 `open_intent_candidate`，不改写 routing Gold。
- 数据不含线上用户日志、真实用户 PII 或生产会话。
- 当前仅有一名人工标注者，不能把本数据集的一致性视为多人共识；主观边界是已知限制。

## Split 方法

共享 scenario、paraphrase 或 contrast 关系的条目先合并为连通分量。dev/test 只按完整分量分配，且 `bootstrap_cluster_id = split_group_id`。主标签采用精确配额；诊断 tags 仅做确定性的近似分层。稳定选择 namespace 为 `{manifest.stable_hash_namespace}`。

该 frozen test 是仓库内可见的 non-blind test。冻结后不得依据 test Gold、test 错误或 test prediction 调整 taxonomy、prompt、阈值、模型选择或数据；后续正式运行由独立 experiment lock 再绑定。冻结之前没有生成 test prediction。

## 分布

### Decision

{_format_table(("decision", "dev", "test"), decision_rows)}

### In-scope intent

{_format_table(("intent", "dev", "test"), intent_rows)}

### 诊断切片

{_format_table(("tag", "dev cases", "test cases", "dev clusters", "test clusters"), slice_rows)}

必需 bootstrap test cluster 数：`{_canonical_json(manifest.required_test_cluster_counts)}`。预登记下限为每个切片 8 个 cluster。

## Semantic preservation audit

冻结前按 `risk-ranked-distinct-cluster-v1` 固定 24 个 test Case：in_scope 12（八个 intent 各至少一条）、OOS 4、no_intent 4、ambiguous 4。它们只用于 B3 primary/weak 的 action、object、horizon 语义保真诊断，不进入主 verdict，也不得反向用于调参。精确 ID 记录在 split 与 freeze manifest。

## 完整性绑定

- adjudicated pre-split Gold: `{source_gold_sha256}`
- dev.jsonl: `{dev_sha256}`
- test.jsonl: `{test_sha256}`
- split-manifest.json: `{split_sha256}`

## 适用范围与限制

- 这是人为平衡的小型合成集，不估计生产 prevalence、真实准确率或商业收益。
- 同一套 taxonomy 和 authoring process 可能造成构造偏差；诊断 slice 结果只支持定位假设，不自动证明因果。
- near-OOS、开放意图和低性能模型退化是定向覆盖，不代表所有小众 activity 或自然语言变体。
- test 在私有仓库中对开发者可见，因此防泄漏依赖 hash freeze、操作纪律与完整实验记录，不是密码学盲测。
- dataset freeze 只承诺数据层不再变化；模型、prompt、provider 参数、预算和 verdict 必须在首次正式 test 前由 experiment lock 单独冻结。
"""


def _manifest(
    *,
    cases_path: Path,
    receipt_path: Path,
    taxonomy_path: Path,
    repository_root: Path,
    dev: Sequence[Case],
    test: Sequence[Case],
    groups: dict[str, _Group],
    diagnostic_targets: dict[str, int],
    diagnostic_objective: float,
    required_clusters: dict[str, int],
    semantic_audit: SemanticAuditSelection,
) -> SplitManifest:
    assignments = [
        GroupAssignment(
            group_id=group_id,
            split=(
                Split.DEV
                if group.cases[0].split_group_id in {case.split_group_id for case in dev}
                else Split.TEST
            ),
            case_ids=[case.id for case in group.cases],
        )
        for group_id, group in groups.items()
    ]
    return SplitManifest(
        taxonomy_version=load_taxonomy(taxonomy_path).taxonomy_version,
        source_artifacts={
            "adjudicated_gold": ArtifactDigest(
                path=_relative_to_root(cases_path, repository_root),
                sha256=sha256_file(cases_path),
            ),
            "materialization_receipt": ArtifactDigest(
                path=_relative_to_root(receipt_path, repository_root),
                sha256=sha256_file(receipt_path),
            ),
            "taxonomy": ArtifactDigest(
                path=_relative_to_root(taxonomy_path, repository_root),
                sha256=sha256_file(taxonomy_path),
            ),
        },
        case_targets={"dev": DEV_CASE_COUNT, "test": TEST_CASE_COUNT},
        dev_decision_targets={
            decision.value: count for decision, count in DEV_DECISION_TARGETS.items()
        },
        diagnostic_target_counts=diagnostic_targets,
        diagnostic_dev_counts=_diagnostic_counts(
            {case.split_group_id for case in dev if case.split_group_id is not None}, groups
        ),
        diagnostic_objective=diagnostic_objective,
        statistics={"dev": _statistics(dev), "test": _statistics(test)},
        required_test_cluster_counts=required_clusters,
        group_assignments=assignments,
        semantic_audit=semantic_audit,
    )


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}


def _write_outputs(
    paths: dict[str, Path], payloads: dict[str, bytes]
) -> Literal["created", "unchanged"]:
    existing = {name for name, path in paths.items() if path.exists()}
    if existing:
        if existing != set(paths):
            raise DatasetSplitError(
                f"partial split/freeze outputs already exist: {sorted(existing)}"
            )
        changed = [name for name, path in paths.items() if path.read_bytes() != payloads[name]]
        if changed:
            raise DatasetSplitError(f"frozen split outputs differ from regeneration: {changed}")
        return "unchanged"

    paths["dev"].parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".kir-pilot-v2-freeze-", dir=paths["dev"].parent
        ) as temporary:
            stage = Path(temporary)
            for name, payload in payloads.items():
                (stage / paths[name].name).write_bytes(payload)
            for _name, path in paths.items():
                os.replace(stage / path.name, path)
                created.append(path)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return "created"


def split_and_freeze_dataset(
    *,
    cases_path: Path,
    receipt_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Create the deterministic formal split and freeze all data-layer artifacts."""

    cases, receipt, taxonomy_sha256 = _load_bound_gold(cases_path, receipt_path, taxonomy_path)
    if len(cases) != DEV_CASE_COUNT + TEST_CASE_COUNT:
        raise DatasetSplitError("IE1.3 requires exactly 160 adjudicated Gold cases")
    groups = _build_groups(cases)
    if len(groups) != receipt.relationship_audit.connected_component_count:
        raise DatasetSplitError("generated relationship group count differs from IE1.2 audit")
    dev_group_ids, diagnostic_targets, objective = _select_dev_groups(groups)
    dev, test = _split_cases(cases, dev_group_ids)
    required_clusters = _validate_result(dev=dev, test=test, taxonomy_path=taxonomy_path)
    semantic_audit = _select_semantic_audit(test)

    manifest = _manifest(
        cases_path=cases_path,
        receipt_path=receipt_path,
        taxonomy_path=taxonomy_path,
        repository_root=repository_root,
        dev=dev,
        test=test,
        groups=groups,
        diagnostic_targets=diagnostic_targets,
        diagnostic_objective=objective,
        required_clusters=required_clusters,
        semantic_audit=semantic_audit,
    )
    dev_bytes = _jsonl_bytes([case.model_dump(mode="json") for case in dev])
    test_bytes = _jsonl_bytes([case.model_dump(mode="json") for case in test])
    split_bytes = _json_bytes(manifest.model_dump(mode="json"))
    paths = _output_paths(output_dir)
    card_bytes = _dataset_card(
        manifest=manifest,
        source_gold_sha256=receipt.materialized_cases_sha256,
        dev_sha256=_sha256_bytes(dev_bytes),
        test_sha256=_sha256_bytes(test_bytes),
        split_sha256=_sha256_bytes(split_bytes),
    ).encode()
    staged_artifacts = {
        "taxonomy": ArtifactDigest(
            path=_relative_to_root(taxonomy_path, repository_root),
            sha256=taxonomy_sha256,
        ),
        "dev": ArtifactDigest(
            path=_relative_to_root(paths["dev"], repository_root),
            sha256=_sha256_bytes(dev_bytes),
        ),
        "test": ArtifactDigest(
            path=_relative_to_root(paths["test"], repository_root),
            sha256=_sha256_bytes(test_bytes),
        ),
        "split": ArtifactDigest(
            path=_relative_to_root(paths["split"], repository_root),
            sha256=_sha256_bytes(split_bytes),
        ),
        "dataset_card": ArtifactDigest(
            path=_relative_to_root(paths["dataset_card"], repository_root),
            sha256=_sha256_bytes(card_bytes),
        ),
        "adjudicated_gold": manifest.source_artifacts["adjudicated_gold"],
        "materialization_receipt": manifest.source_artifacts["materialization_receipt"],
    }
    freeze = DatasetFreezeManifest(
        status="frozen",
        dataset_version=DATASET_VERSION,
        taxonomy_version=manifest.taxonomy_version,
        artifacts=staged_artifacts,
        semantic_audit_case_ids=semantic_audit.case_ids,
    )
    freeze_bytes = _json_bytes(freeze.model_dump(mode="json"))
    checksum_values = {
        paths["dev"].name: _sha256_bytes(dev_bytes),
        paths["test"].name: _sha256_bytes(test_bytes),
        paths["split"].name: _sha256_bytes(split_bytes),
        paths["dataset_card"].name: _sha256_bytes(card_bytes),
        paths["freeze"].name: _sha256_bytes(freeze_bytes),
    }
    checksums_bytes = "".join(
        f"{digest}  {filename}\n" for filename, digest in sorted(checksum_values.items())
    ).encode()
    payloads = {
        "dev": dev_bytes,
        "test": test_bytes,
        "split": split_bytes,
        "dataset_card": card_bytes,
        "freeze": freeze_bytes,
        "checksums": checksums_bytes,
    }
    status = _write_outputs(paths, payloads)
    return {
        "status": status,
        "dataset_version": DATASET_VERSION,
        "dev_case_count": len(dev),
        "test_case_count": len(test),
        "dev_group_count": len(dev_group_ids),
        "test_group_count": len(groups) - len(dev_group_ids),
        "required_test_cluster_counts": required_clusters,
        "semantic_audit_case_count": len(semantic_audit.case_ids),
        "freeze_manifest_sha256": _sha256_bytes(freeze_bytes),
        "test_sha256": _sha256_bytes(test_bytes),
        "test_predictions_generated": False,
    }
