from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from intentbench.cli import main
from intentbench.formal_runs import FormalRunError, run_b0_test
from intentbench.freeze import FreezeGuardError


def test_formal_baseline_refuses_draft_experiment_before_writing(tmp_path: Path) -> None:
    with pytest.raises(FreezeGuardError, match="experiment lock is not frozen"):
        run_b0_test(
            cases_path=Path("data/kir-pilot-v2/test.jsonl"),
            taxonomy_path=Path("configs/kindred-activity-intents-v2.yaml"),
            dataset_manifest_path=Path("data/kir-pilot-v2/freeze-manifest.json"),
            experiment_path=Path("configs/kir-pilot-v2-ie3-experiment.yaml"),
            output_dir=tmp_path / "b0",
            repository_root=Path("."),
        )
    assert not (tmp_path / "b0").exists()


def test_formal_run_error_is_a_baseline_contract_error() -> None:
    assert issubclass(FormalRunError, ValueError)


def test_test_cli_checks_freeze_before_provider_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = CliRunner().invoke(
        main,
        [
            "ie3",
            "one-stage-cache-test",
            "--role",
            "weak_decision",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 1
    assert "experiment lock is not frozen" in result.output
    assert "missing weak_decision credential" not in result.output
    assert not (tmp_path / "cache").exists()
