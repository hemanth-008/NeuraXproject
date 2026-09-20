from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_freshness_requires_all_outputs_and_newer_inputs(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    source = tmp_path / "source.py"
    output.write_text("out", encoding="utf-8")
    source.write_text("source", encoding="utf-8")
    os.utime(source, (output.stat().st_mtime + 2, output.stat().st_mtime + 2))
    assert run_pipeline._fresh([output], [source]) is False
    os.utime(output, (source.stat().st_mtime + 2, source.stat().st_mtime + 2))
    assert run_pipeline._fresh([output], [source]) is True


def test_all_stage_uses_current_fixture_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for path in FIXTURES.iterdir():
        if path.is_file():
            shutil.copy2(path, artifacts / path.name)
    # Fixture artifacts are intentionally current, so this exercises the
    # idempotent/skipped path and the final validation without model weights or
    # any locked-holdout evaluation.
    code = run_pipeline.main(["--stage", "all", "--artifacts-dir", str(artifacts)])
    assert code == 0
    assert (artifacts / "status.json").is_file()
    status = json.loads((artifacts / "status.json").read_text(encoding="utf-8"))
    assert status["status"] in {"ok", "invalid", "missing"}
    assert (artifacts / "logs" / "validate.log").is_file()
