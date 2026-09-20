from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_fixtures import make_fixtures  # noqa: E402
from validate_artifacts import validate_artifact, validate_directory  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"
CONTRACTS = ROOT / "contracts"


def _ensure_fixtures() -> None:
    make_fixtures(FIXTURES)


def test_all_fixtures_validate() -> None:
    _ensure_fixtures()
    results = validate_directory(FIXTURES, require_all=True)
    invalid = [(name, errors) for name, status, errors in results if status != "ok"]
    assert not invalid, invalid


def test_mutated_fixture_missing_required_field_fails(tmp_path: Path) -> None:
    _ensure_fixtures()
    frame = pd.read_parquet(FIXTURES / "predictions.parquet").drop(columns=["unit_id"])
    mutated = tmp_path / "predictions.parquet"
    frame.to_parquet(mutated, index=False)
    errors = validate_artifact(mutated, CONTRACTS / "predictions.schema.json")
    assert any("unit_id" in error and "missing" in error for error in errors), errors


def test_mutated_fixture_wrong_dtype_fails(tmp_path: Path) -> None:
    _ensure_fixtures()
    frame = pd.read_parquet(FIXTURES / "predictions.parquet")
    frame["p_defect"] = ["not-a-number"] * len(frame)
    mutated = tmp_path / "predictions.parquet"
    frame.to_parquet(mutated, index=False)
    errors = validate_artifact(mutated, CONTRACTS / "predictions.schema.json")
    assert any("p_defect" in error and "expected number" in error for error in errors), errors


def test_current_predictions_validate() -> None:
    errors = validate_artifact(ROOT / "artifacts" / "predictions.parquet", CONTRACTS / "predictions.schema.json")
    assert not errors, errors


def test_economics_values_are_labelled() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from common.config import economics  # noqa: PLC0415

    values = economics()
    assert {"c_escape", "c_false_reject", "c_review"}.issubset(values)
    assert all("value" in record and "assumption_id" in record for record in values.values())

