"""Rebuild the deterministic post-freeze OpenAI recovery report artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from intentbench.matrix import build_seven_run_matrix
from intentbench.reporting import compare_frozen_split, evaluate_frozen_split
from intentbench.schemas import Split

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "experiments/ie3-test"
RECOVERY = ROOT / "experiments/ie3-openai-recovery"
CASES = ROOT / "data/kir-pilot-v2/test.jsonl"
TAXONOMY = ROOT / "configs/kindred-activity-intents-v2.yaml"
DATASET_MANIFEST = ROOT / "data/kir-pilot-v2/freeze-manifest.json"
EXPERIMENT = ROOT / "configs/kir-pilot-v2-ie3-experiment.yaml"


def main() -> None:
    reports: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="intentbench-ie3-openai-recovery-") as raw:
        runs = Path(raw)
        os.symlink(
            ORIGINAL / "primary_decision",
            runs / "primary_decision",
            target_is_directory=True,
        )
        os.symlink(
            ORIGINAL / "weak_decision",
            runs / "weak_decision",
            target_is_directory=True,
        )
        cross_provider = runs / "cross_provider_reference"
        cross_provider.mkdir()
        for arm in ("b2b", "b3"):
            os.symlink(
                RECOVERY / "cross_provider_reference" / arm,
                cross_provider / arm,
                target_is_directory=True,
            )
        reports["matrix"] = build_seven_run_matrix(
            runs_root=runs,
            cases_path=CASES,
            taxonomy_path=TAXONOMY,
            dataset_manifest_path=DATASET_MANIFEST,
            experiment_path=EXPERIMENT,
            output_path=RECOVERY / "seven-run-matrix.json",
            repository_root=ROOT,
        )

    for arm in ("b2b", "b3"):
        reports[arm] = evaluate_frozen_split(
            split=Split.TEST,
            cases_path=CASES,
            predictions_path=RECOVERY / "cross_provider_reference" / arm / "predictions.jsonl",
            taxonomy_path=TAXONOMY,
            dataset_manifest_path=DATASET_MANIFEST,
            output_dir=RECOVERY / "evaluation" / arm,
            repository_root=ROOT,
        )
    reports["comparison"] = compare_frozen_split(
        split=Split.TEST,
        cases_path=CASES,
        left_predictions_path=RECOVERY / "cross_provider_reference" / "b2b" / "predictions.jsonl",
        right_predictions_path=RECOVERY / "cross_provider_reference" / "b3" / "predictions.jsonl",
        left_id="openai-recovery-B2b",
        right_id="openai-recovery-B3",
        taxonomy_path=TAXONOMY,
        dataset_manifest_path=DATASET_MANIFEST,
        output_path=RECOVERY / "b2b-vs-b3-bootstrap.json",
        repository_root=ROOT,
    )
    print(json.dumps(reports, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
