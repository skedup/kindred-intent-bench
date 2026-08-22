"""Capture one installed Kindred derived runtime as a static grounding snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from intentbench.grounding import ActivityGroundingSnapshot

ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "configs/kindred-activity-grounding-v1.yaml"

PROBE = r"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import kindred
from kindred import _derived_runtime
from kindred.activity.action import list_registered_actions, load_atomic_action
from kindred.activity.skill import list_registered_activities, load_activity_skill
from kindred.life_assets import ACTIONS_DIR, ACTIVITIES_DIR

package_root = Path(kindred.__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    return path.resolve().relative_to(package_root).as_posix()


activities = []
for name in list_registered_activities():
    activity = load_activity_skill(name)
    package = ACTIVITIES_DIR / name
    manifest = package / "manifest.yaml"
    skill = package / "SKILL.md"
    activities.append(
        {
            "name": name,
            "description": activity.description,
            "uses": [item.action for item in activity.uses],
            "terminal_when": activity.terminal_when,
            "duration_hint": activity.duration_hint,
            "produces": activity.produces,
            "manifest_path": source(manifest),
            "manifest_sha256": sha256(manifest),
            "skill_path": source(skill),
            "skill_sha256": sha256(skill),
        }
    )

actions = []
for name in list_registered_actions():
    action = load_atomic_action(name)
    package = ACTIONS_DIR / name
    manifest = package / "manifest.yaml"
    skill = package / "SKILL.md"
    actions.append(
        {
            "name": name,
            "description": action.desc,
            "applies_when": action.applies_when,
            "capabilities": sorted(action.capabilities),
            "manifest_path": source(manifest),
            "manifest_sha256": sha256(manifest),
            "skill_path": source(skill),
            "skill_sha256": sha256(skill),
        }
    )

print(
    json.dumps(
        {
            "distribution_version": version("kindred"),
            "internal_source_commit": _derived_runtime.INTERNAL_SOURCE_COMMIT,
            "private_overlay_sha256": _derived_runtime.PRIVATE_OVERLAY_SHA256,
            "activities": activities,
            "actions": actions,
        },
        ensure_ascii=False,
    )
)
"""


def run_probe(python: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(python), "-c", PROBE],
        check=True,
        capture_output=True,
        text=True,
    )
    payload: object = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Kindred grounding probe did not return an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kindred-python", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    probed = run_probe(args.kindred_python)
    snapshot = ActivityGroundingSnapshot.model_validate(
        {
            "schema_version": 1,
            "grounding_version": "kindred-activity-grounding-v1",
            "captured_at": datetime.now(timezone.utc),
            "authority": {
                "authority_kind": "installed_derived_runtime",
                "environment": args.environment,
                "distribution_name": "kindred",
                "distribution_version": probed["distribution_version"],
                "internal_source_commit": probed["internal_source_commit"],
                "private_overlay_sha256": probed["private_overlay_sha256"],
                "activity_loader": "kindred.activity.skill.list_registered_activities",
                "action_loader": "kindred.activity.action.list_registered_actions",
            },
            "activities": probed["activities"],
            "actions": probed["actions"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(
            snapshot.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "captured",
                "output": str(args.output),
                "activity_count": len(snapshot.activities),
                "action_count": len(snapshot.actions),
                "internal_source_commit": snapshot.authority.internal_source_commit,
                "private_overlay_sha256": snapshot.authority.private_overlay_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
