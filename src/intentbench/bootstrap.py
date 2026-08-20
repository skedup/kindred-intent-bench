"""Leakage-safe grouping and paired cluster bootstrap."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from intentbench.evaluator import index_predictions
from intentbench.schemas import Case, Prediction

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20_260_820
CONFIDENCE_LEVEL = 0.95
MIN_CLUSTERS = 8


class InsufficientClustersError(ValueError):
    """A required slice cannot support its pre-registered CI."""


class SplitLeakageError(ValueError):
    """Related cases have crossed a dev/test boundary or use unstable group ids."""


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def assign_relationship_clusters(cases: Sequence[Case]) -> list[Case]:
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique before clustering")
    disjoint = _DisjointSet(ids)
    relation_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in cases:
        relation_members[("scenario", case.scenario_family_id)].append(case.id)
        relation_members[("paraphrase", case.paraphrase_cluster_id)].append(case.id)
        relation_members[("contrast", case.contrast_group_id)].append(case.id)
    for members in relation_members.values():
        for member in members[1:]:
            disjoint.union(members[0], member)

    components: dict[str, list[str]] = defaultdict(list)
    for case_id in ids:
        components[disjoint.find(case_id)].append(case_id)
    group_for_id = {
        case_id: f"sg-{min(component)}"
        for component in components.values()
        for case_id in component
    }
    return [
        Case.model_validate(
            {
                **case.model_dump(),
                "split_group_id": group_for_id[case.id],
                "bootstrap_cluster_id": group_for_id[case.id],
            }
        )
        for case in cases
    ]


def validate_split_integrity(cases: Sequence[Case]) -> None:
    """Require generated stable groups and keep every component in one split."""

    expected = assign_relationship_clusters(cases)
    expected_groups = {case.id: case.split_group_id for case in expected}
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        if case.split_group_id != expected_groups[case.id]:
            raise SplitLeakageError(f"{case.id} does not use its stable connected-component id")
        assert case.split_group_id is not None
        splits_by_group[case.split_group_id].add(case.split.value)
    leaked = {group: sorted(splits) for group, splits in splits_by_group.items() if len(splits) > 1}
    if leaked:
        raise SplitLeakageError(f"connected components cross splits: {leaked}")


def type7_quantile(values: Sequence[float], probability: float) -> float:
    """R/NumPy default quantile: h=(n-1)q with linear interpolation."""

    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    confidence_level: float
    iterations: int
    seed: int
    cluster_count: int


MetricFunction = Callable[[Sequence[Case], Mapping[str, Prediction]], float]


def paired_cluster_bootstrap(
    cases: Sequence[Case],
    left_predictions: Sequence[Prediction],
    right_predictions: Sequence[Prediction],
    metric: MetricFunction,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
    min_clusters: int = MIN_CLUSTERS,
) -> BootstrapInterval:
    """Estimate right-left using identical cluster draws over all Gold cases."""

    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    gold_ids = {case.id for case in cases}
    if len(gold_ids) != len(cases):
        raise ValueError("Gold case ids must be unique")
    left = index_predictions(left_predictions, gold_ids)
    right = index_predictions(right_predictions, gold_ids)
    clusters: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        if case.bootstrap_cluster_id is None:
            raise ValueError(f"{case.id} has no bootstrap_cluster_id")
        clusters[case.bootstrap_cluster_id].append(case)
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < min_clusters:
        raise InsufficientClustersError(
            f"required slice has {len(cluster_ids)} clusters; minimum is {min_clusters}"
        )

    point = metric(cases, right) - metric(cases, left)
    random_source = random.Random(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled_cases: list[Case] = []
        for _cluster_index in range(len(cluster_ids)):
            cluster_id = random_source.choice(cluster_ids)
            sampled_cases.extend(clusters[cluster_id])
        differences.append(metric(sampled_cases, right) - metric(sampled_cases, left))

    alpha = 1 - confidence_level
    return BootstrapInterval(
        point=point,
        lower=type7_quantile(differences, alpha / 2),
        upper=type7_quantile(differences, 1 - alpha / 2),
        confidence_level=confidence_level,
        iterations=iterations,
        seed=seed,
        cluster_count=len(cluster_ids),
    )
