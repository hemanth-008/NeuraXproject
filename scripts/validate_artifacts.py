"""Validate all known artifacts against the contract files.

The JSON schemas are standard Draft 2020-12 schemas.  Parquet, CSV and NPZ
schemas carry small ``x-*`` annotations because JSON Schema itself cannot
describe a dataframe dtype or an NPZ member list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from pandas.api import types as ptypes


ARTIFACTS: dict[str, tuple[str, str]] = {
    "predictions.parquet": ("predictions.schema.json", "parquet"),
    "batch_metrics.parquet": ("batch_metrics.schema.json", "parquet"),
    "replay_stream.parquet": ("replay_stream.schema.json", "parquet"),
    "sim_unit_map.parquet": ("sim_unit_map.schema.json", "parquet"),
    "sim_results.parquet": ("sim_results.schema.json", "parquet"),
    "sim_truth.json": ("sim_truth.schema.json", "json"),
    "evidence_cards.json": ("evidence_cards.schema.json", "json"),
    "rootcause_recovery.json": ("rootcause_recovery.schema.json", "json"),
    "flow_metrics.json": ("flow_metrics.schema.json", "json"),
    "twin_validation.json": ("twin_validation.schema.json", "json"),
    "profit_model.json": ("profit_model.schema.json", "json"),
    "recommendations.json": ("recommendations.schema.json", "json"),
    "calibration.json": ("calibration.schema.json", "json"),
    "novelty.json": ("novelty.schema.json", "json"),
    "localization.json": ("localization.schema.json", "json"),
    "status.json": ("status.schema.json", "json"),
    "holdout_result.json": ("holdout_result.schema.json", "json"),
    "holdout_runs.json": ("holdout_runs.schema.json", "json"),
    "flow_v0.json": ("flow_v0.schema.json", "json"),
    "robustness_orient.csv": ("robustness_orient.schema.json", "csv"),
    "robustness.csv": ("robustness_orient.schema.json", "csv"),
    "knn_bank.npz": ("knn_bank.schema.json", "npz"),
}


def _location(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return path or "$"


def _json_errors(path: Path, schema: dict[str, Any]) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive I/O path
        return [f"cannot parse JSON: {exc}"]
    validator = Draft202012Validator(schema)
    return [f"{_location(error)}: {error.message}" for error in sorted(validator.iter_errors(payload), key=_location)]


def _string_like(series: pd.Series) -> bool:
    if ptypes.is_string_dtype(series.dtype):
        return True
    values = series.dropna()
    return ptypes.is_object_dtype(series.dtype) and bool(values.empty or values.map(lambda value: isinstance(value, str)).all())


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "object":
        return True
    if expected == "string":
        return _string_like(series)
    if expected == "number":
        if ptypes.is_object_dtype(series.dtype) and series.dropna().empty:
            return True
        return ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(series.dtype)
    if expected == "integer":
        if ptypes.is_object_dtype(series.dtype) and series.dropna().empty:
            return True
        return ptypes.is_integer_dtype(series.dtype) and not ptypes.is_bool_dtype(series.dtype)
    if expected == "boolean":
        if ptypes.is_object_dtype(series.dtype) and series.dropna().empty:
            return True
        return ptypes.is_bool_dtype(series.dtype)
    raise ValueError(f"unknown contract dtype {expected!r}")


def _table_errors(path: Path, schema: dict[str, Any], kind: str) -> list[str]:
    try:
        if kind == "parquet":
            table = pd.read_parquet(path)
        else:
            table = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O path
        return [f"cannot read {kind}: {exc}"]
    errors: list[str] = []
    columns = schema.get("x-columns", {})
    for column, spec in columns.items():
        required = bool(spec.get("required", False))
        if column not in table.columns:
            if required:
                errors.append(f"missing required column {column!r}")
            continue
        expected = spec.get("dtype", "object")
        try:
            matches = _dtype_matches(table[column], expected)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not matches:
            errors.append(f"column {column!r} has dtype {table[column].dtype!s}, expected {expected}")
    for group in schema.get("x-required-any-of", []):
        if not any(column in table.columns for column in group):
            errors.append(f"missing one of required columns: {', '.join(group)}")
    if "source_split" in table.columns:
        # The validator must never inspect or score locked-holdout rows.
        locked = table["source_split"].astype("string").eq("holdout")
        if bool(locked.any()):
            errors.append("locked-holdout rows are prohibited in artifacts")
    return errors


def _npz_errors(path: Path, schema: dict[str, Any]) -> list[str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
    except Exception as exc:  # pragma: no cover - defensive I/O path
        return [f"cannot read NPZ: {exc}"]
    errors: list[str] = []
    for requirement in schema.get("x-npz-keys", []):
        if isinstance(requirement, list):
            if not any(key in keys for key in requirement):
                errors.append(f"missing one of required NPZ keys: {', '.join(requirement)}")
        elif requirement not in keys:
            errors.append(f"missing required NPZ key {requirement!r}")
    return errors


def validate_artifact(path: str | Path, schema_path: str | Path) -> list[str]:
    """Return validation errors for one artifact; an empty list means valid."""

    artifact_path = Path(path)
    with Path(schema_path).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    kind = schema.get("x-artifact-type")
    if kind is None and artifact_path.suffix.lower() == ".json":
        kind = "json"
    if kind == "json":
        return _json_errors(artifact_path, schema)
    if kind in {"parquet", "csv"}:
        return _table_errors(artifact_path, schema, kind)
    if kind == "npz":
        return _npz_errors(artifact_path, schema)
    return [f"schema has unknown x-artifact-type {kind!r}"]


def validate_directory(artifact_dir: str | Path, require_all: bool = False) -> list[tuple[str, str, list[str]]]:
    artifact_path = Path(artifact_dir)
    contracts_dir = Path(__file__).resolve().parents[1] / "contracts"
    results: list[tuple[str, str, list[str]]] = []
    for name, (schema_name, _kind) in ARTIFACTS.items():
        path = artifact_path / name
        if not path.exists():
            results.append((name, "missing", ["required by --require-all"] if require_all else []))
            continue
        schema_path = contracts_dir / schema_name
        if not schema_path.exists():
            results.append((name, "invalid", [f"schema not found: {schema_path}"]))
            continue
        errors = validate_artifact(path, schema_path)
        results.append((name, "ok" if not errors else "invalid", errors[:3]))
    return results


def _print_results(results: list[tuple[str, str, list[str]]]) -> None:
    print("artifact | status | first 3 errors")
    print("--- | --- | ---")
    for name, status, errors in results:
        print(f"{name} | {status} | {'; '.join(errors) if errors else '-'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true", help="treat absent known artifacts as missing failures")
    parser.add_argument("--dir", default="artifacts", dest="artifact_dir", help="artifact directory")
    args = parser.parse_args(argv)
    results = validate_directory(args.artifact_dir, require_all=args.require_all)
    _print_results(results)
    return 1 if any(status == "invalid" or (args.require_all and status == "missing") for _, status, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
