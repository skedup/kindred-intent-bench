from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from click.testing import CliRunner

from intentbench.bootstrap import assign_relationship_clusters
from intentbench.cli import main
from intentbench.freeze import ArtifactDigest, DatasetFreezeManifest, sha256_file
from intentbench.reporting import (
    BootstrapComparisonArtifact,
    EvaluationArtifactError,
    EvaluationMetricsArtifact,
    compare_frozen_dev,
    evaluate_frozen_dev,
)
from intentbench.schemas import Case, Decision, Prediction, PredictionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, values: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in values
    ]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _as_dev(cases: Sequence[Case]) -> list[Case]:
    dev = [Case.model_validate({**case.model_dump(mode="json"), "split": "dev"}) for case in cases]
    return assign_relationship_clusters(dev)


def _prediction(case_id: str, decision: Decision, target: str | None = None) -> Prediction:
    return Prediction.model_validate(
        {
            "case_id": case_id,
            "decision": decision.value,
            "predicted_intent": target,
            "candidate_intents": [],
            "slots": {
                "desired_experience": None,
                "object": None,
                "horizon": "now",
            },
            "reason_short": "fixture",
        }
    )


def _seed_frozen_dev(root: Path, cases: Sequence[Case]) -> dict[str, Path]:
    dev_path = root / "data/kir-pilot-fixture/dev.jsonl"
    taxonomy_path = root / "configs/kindred-activity-intents-v1.yaml"
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_path.write_bytes(
        (PROJECT_ROOT / "configs/kindred-activity-intents-v1.yaml").read_bytes()
    )
    _write_jsonl(dev_path, [case.model_dump(mode="json") for case in cases])
    zero_digest = "0" * 64
    manifest = DatasetFreezeManifest(
        status="frozen",
        dataset_version="kir-pilot-fixture-v1",
        taxonomy_version="kindred-activity-intents-v1",
        artifacts={
            "dev": ArtifactDigest(
                path="data/kir-pilot-fixture/dev.jsonl",
                sha256=sha256_file(dev_path),
            ),
            "taxonomy": ArtifactDigest(
                path="configs/kindred-activity-intents-v1.yaml",
                sha256=sha256_file(taxonomy_path),
            ),
            "test": ArtifactDigest(path="data/kir-pilot-fixture/test.jsonl", sha256=zero_digest),
            "split": ArtifactDigest(path="data/kir-pilot-fixture/split.json", sha256=zero_digest),
        },
    )
    manifest_path = root / "data/kir-pilot-fixture/freeze-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    return {
        "cases": dev_path,
        "taxonomy": taxonomy_path,
        "manifest": manifest_path,
    }


def test_dev_evaluation_writes_stable_metrics_confusion_and_badcases(
    tmp_path: Path, make_case: Callable[..., Case]
) -> None:
    cases = _as_dev(
        [
            make_case("kir-eval-001", Decision.IN_SCOPE, target_intent="rest"),
            make_case("kir-eval-002", Decision.OOS),
            make_case("kir-eval-003", Decision.NO_INTENT),
            make_case("kir-eval-004", Decision.AMBIGUOUS),
            make_case("kir-eval-005", Decision.IN_SCOPE, target_intent="rest"),
            make_case("kir-eval-006", Decision.NO_INTENT),
        ]
    )
    paths = _seed_frozen_dev(tmp_path, cases)
    predictions = [
        _prediction("kir-eval-001", Decision.OOS),
        _prediction("kir-eval-002", Decision.IN_SCOPE, "rest"),
        Prediction(
            case_id="kir-eval-004",
            status=PredictionStatus.SCHEMA_INVALID,
            error_type="invalid_json",
        ),
        _prediction("kir-eval-005", Decision.IN_SCOPE, "take_a_walk"),
        _prediction("kir-eval-006", Decision.NO_INTENT),
    ]
    predictions_path = tmp_path / "experiments/run-a/predictions.jsonl"
    _write_jsonl(
        predictions_path, [prediction.model_dump(mode="json") for prediction in predictions]
    )
    output_dir = predictions_path.parent
    arguments = {
        "cases_path": paths["cases"],
        "predictions_path": predictions_path,
        "taxonomy_path": paths["taxonomy"],
        "dataset_manifest_path": paths["manifest"],
        "output_dir": output_dir,
        "repository_root": tmp_path,
    }
    created = evaluate_frozen_dev(**arguments)
    unchanged = evaluate_frozen_dev(**arguments)
    assert created["status"] == "created"
    assert unchanged == {**created, "status": "unchanged"}
    assert created["provider_calls"] == 0
    assert created["hierarchical_exact_match"] == pytest.approx(1 / 6)

    metrics = EvaluationMetricsArtifact.model_validate_json(
        (output_dir / "metrics.json").read_text()
    )
    assert metrics.metrics["failure_counts"] == {
        "missing": 1,
        "provider_failure": 0,
        "schema_invalid": 1,
    }
    confusion = (output_dir / "confusion.csv").read_text().splitlines()
    assert len(confusion) == 21
    assert confusion[0] == "gold_label,predicted_label,count,gold_count,row_fraction"
    assert "in_scope,oos,1,2,0.500000" in confusion
    assert "no_intent,invalid,1,2,0.500000" in confusion

    badcases = [
        json.loads(line) for line in (output_dir / "badcases.jsonl").read_text().splitlines()
    ]
    assert len(badcases) == 5
    assert badcases[0]["case_id"] == "kir-eval-003"
    by_id = {record["case_id"]: record for record in badcases}
    assert by_id["kir-eval-001"]["error_types"] == ["known_intent_false_reject"]
    assert by_id["kir-eval-002"]["error_types"] == ["oos_false_accept"]
    assert by_id["kir-eval-004"]["error_types"] == ["schema_invalid"]
    assert by_id["kir-eval-005"]["error_types"] == ["intent_misroute"]


def test_dev_evaluation_rejects_gold_hash_drift(
    tmp_path: Path, make_case: Callable[..., Case]
) -> None:
    cases = _as_dev([make_case("kir-eval-001")])
    paths = _seed_frozen_dev(tmp_path, cases)
    paths["cases"].write_bytes(paths["cases"].read_bytes() + b"\n")
    predictions_path = tmp_path / "experiments/run-a/predictions.jsonl"
    _write_jsonl(predictions_path, [])
    with pytest.raises(EvaluationArtifactError, match="hash differs"):
        evaluate_frozen_dev(
            cases_path=paths["cases"],
            predictions_path=predictions_path,
            taxonomy_path=paths["taxonomy"],
            dataset_manifest_path=paths["manifest"],
            output_dir=predictions_path.parent,
            repository_root=tmp_path,
        )


def test_dev_evaluation_refuses_frozen_test_cases(
    tmp_path: Path, make_case: Callable[..., Case]
) -> None:
    cases = assign_relationship_clusters([make_case("kir-eval-001")])
    paths = _seed_frozen_dev(tmp_path, cases)
    predictions_path = tmp_path / "experiments/run-a/predictions.jsonl"
    _write_jsonl(predictions_path, [])
    with pytest.raises(EvaluationArtifactError, match="dev Case records only"):
        evaluate_frozen_dev(
            cases_path=paths["cases"],
            predictions_path=predictions_path,
            taxonomy_path=paths["taxonomy"],
            dataset_manifest_path=paths["manifest"],
            output_dir=predictions_path.parent,
            repository_root=tmp_path,
        )


def test_evaluate_dev_cli_writes_offline_artifacts(
    tmp_path: Path, make_case: Callable[..., Case]
) -> None:
    cases = _as_dev([make_case("kir-eval-001")])
    paths = _seed_frozen_dev(tmp_path, cases)
    predictions_path = tmp_path / "experiments/run-a/predictions.jsonl"
    _write_jsonl(predictions_path, [])
    output_dir = predictions_path.parent
    result = CliRunner().invoke(
        main,
        [
            "evaluate",
            "dev",
            "--cases",
            str(paths["cases"]),
            "--predictions",
            str(predictions_path),
            "--taxonomy",
            str(paths["taxonomy"]),
            "--dataset-manifest",
            str(paths["manifest"]),
            "--repository-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["provider_calls"] == 0
    assert report["status"] == "created"
    assert {path.name for path in output_dir.iterdir()} == {
        "predictions.jsonl",
        "metrics.json",
        "confusion.csv",
        "badcases.jsonl",
    }


def test_paired_dev_comparison_records_estimable_and_small_domains(
    tmp_path: Path, make_case: Callable[..., Case]
) -> None:
    cases = _as_dev(
        [make_case(f"kir-compare-{index:03d}", Decision.NO_INTENT) for index in range(1, 9)]
    )
    paths = _seed_frozen_dev(tmp_path, cases)
    left = [_prediction(case.id, Decision.OOS) for case in cases]
    right = [_prediction(case.id, Decision.NO_INTENT) for case in cases]
    left_path = tmp_path / "experiments/left/predictions.jsonl"
    right_path = tmp_path / "experiments/right/predictions.jsonl"
    _write_jsonl(left_path, [prediction.model_dump(mode="json") for prediction in left])
    _write_jsonl(right_path, [prediction.model_dump(mode="json") for prediction in right])
    output_path = tmp_path / "comparisons/right-minus-left/bootstrap.json"
    arguments = {
        "cases_path": paths["cases"],
        "left_predictions_path": left_path,
        "right_predictions_path": right_path,
        "left_id": "left",
        "right_id": "right",
        "taxonomy_path": paths["taxonomy"],
        "dataset_manifest_path": paths["manifest"],
        "output_path": output_path,
        "repository_root": tmp_path,
        "iterations": 200,
    }
    created = compare_frozen_dev(**arguments)
    unchanged = compare_frozen_dev(**arguments)
    assert created["status"] == "created"
    assert unchanged == {**created, "status": "unchanged"}
    artifact = BootstrapComparisonArtifact.model_validate_json(output_path.read_text())
    full = artifact.required_domains["full_hem"]
    no_intent = artifact.required_domains["no_intent_recall"]
    assert full.estimable is True
    assert full.interval is not None
    assert (full.point, full.interval.lower, full.interval.upper) == (1.0, 1.0, 1.0)
    assert no_intent.estimable is True
    assert no_intent.point == 1.0
    for name in ("gold_in_scope_intent_macro_f1", "near_oos_recall"):
        domain = artifact.required_domains[name]
        assert domain.estimable is False
        assert domain.cluster_count == 0
        assert domain.unavailable_reason == "insufficient_clusters"
