"""Write the build-status artifact from file presence and contract validation only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from validate_artifacts import validate_directory
except ModuleNotFoundError:  # importable as scripts.write_status in tests
    from scripts.validate_artifacts import validate_directory


COMPONENTS: dict[str, tuple[str, ...]] = {
    "vision": ("predictions.parquet", "calibration.json", "novelty.json", "knn_bank.npz"),
    "localization": ("localization.json",),
    "rootcause": ("evidence_cards.json", "rootcause_recovery.json"),
    "flow": ("flow_metrics.json", "twin_validation.json", "sim_results.parquet"),
    "profit": ("profit_model.json",),
    "recommendations": ("recommendations.json",),
}


def _component_status(name: str, files: tuple[str, ...], results: dict[str, tuple[str, list[str]]], generated_at: str) -> dict[str, Any]:
    states = [results.get(filename, ("missing", []))[0] for filename in files]
    if all(state == "ok" for state in states):
        status = "ok"
    elif any(state == "invalid" for state in states):
        status = "invalid"
    else:
        status = "missing"
    existing = [filename for filename in files if results.get(filename, ("missing", []))[0] == "ok"]
    return {
        "component": name,
        "status": status,
        "evidence_file": existing[0] if existing else files[0],
        "generated_at": generated_at,
    }


def build_status(artifacts_dir: str | Path = "artifacts") -> dict[str, Any]:
    artifacts = Path(artifacts_dir).resolve()
    validation = validate_directory(artifacts, require_all=False)
    by_name = {name: (status, errors) for name, status, errors in validation}
    generated_at = datetime.now(timezone.utc).isoformat()
    components = [_component_status(name, files, by_name, generated_at) for name, files in COMPONENTS.items()]
    invalid = [name for name, status, _ in validation if status == "invalid"]
    present = [name for name, status, _ in validation if status == "ok"]
    overall = "invalid" if invalid else "ok" if present else "missing"
    payload: dict[str, Any] = {
        "component": "pipeline",
        "status": overall,
        "evidence_file": "scripts/validate_artifacts.py",
        "generated_at": generated_at,
        "components": components,
        "validation": {
            "checked": len(validation),
            "ok": len(present),
            "invalid": invalid,
            "missing": [name for name, status, _ in validation if status == "missing"],
        },
        "meta": {
            "generated_at": generated_at,
            "source": "artifact existence and scripts/validate_artifacts.py results",
            "assumption_ids": [],
            "tag": "BUILD STATUS",
        },
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    target = artifacts / "status.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={overall} components={len(components)} checked={len(validation)} wrote={target}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args(argv)
    build_status(args.artifacts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
