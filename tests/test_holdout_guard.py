from __future__ import annotations

import json
from pathlib import Path

from scripts import eval_locked_holdout


ROOT = Path(__file__).resolve().parents[1]


def test_holdout_requires_confirmation(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    code = eval_locked_holdout.main(["--dry-run", "--artifacts-dir", str(artifacts)])
    assert code == 2
    assert not (artifacts / "holdout_result.json").exists()


def test_holdout_dry_run_consumes_one_counter_and_writes_metrics(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    code = eval_locked_holdout.main(["--confirm", "--dry-run", "--artifacts-dir", str(artifacts)])
    assert code == 0
    ledger = json.loads((artifacts / "holdout_runs.json").read_text(encoding="utf-8"))
    result = json.loads((artifacts / "holdout_result.json").read_text(encoding="utf-8"))
    assert ledger["count"] == 1
    assert result["n"] == 10
    assert {"accuracy", "macro_f1", "per_class", "confusion_matrix", "ece", "conformal_coverage"}.issubset(result)


def test_holdout_counter_blocks_without_override(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "holdout_runs.json").write_text(json.dumps({"count": 2}), encoding="utf-8")
    code = eval_locked_holdout.main(["--confirm", "--dry-run", "--artifacts-dir", str(artifacts)])
    assert code == 2
    assert not (artifacts / "holdout_result.json").exists()


def test_only_evaluator_contains_exact_holdout_comparison() -> None:
    pattern = "split == 'holdout'"
    for path in (ROOT / "scripts").glob("*.py"):
        if path.name == "eval_locked_holdout.py":
            continue
        assert pattern not in path.read_text(encoding="utf-8"), path
