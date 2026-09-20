"""Join simulated metadata into predictions without changing other columns."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


SIM_COLUMNS = ["batch_id", "variant", "condition", "station", "ts"]


def merge_predictions(
    predictions_path: str | Path = "artifacts/predictions.parquet",
    sim_map_path: str | Path = "artifacts/sim_unit_map.parquet",
    *,
    backup: bool = True,
) -> Path:
    predictions = Path(predictions_path)
    sim_map = Path(sim_map_path)
    if not predictions.exists():
        raise FileNotFoundError(predictions)
    if not sim_map.exists():
        raise FileNotFoundError(sim_map)
    original = pd.read_parquet(predictions)
    mapping = pd.read_parquet(sim_map)
    required = {"unit_id", *SIM_COLUMNS}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"sim_unit_map missing columns: {sorted(missing)}")
    if mapping["unit_id"].duplicated().any():
        raise ValueError("sim_unit_map unit_id must be unique")
    if "unit_id" not in original.columns:
        raise ValueError("predictions must contain unit_id")

    if backup:
        shutil.copy2(predictions, predictions.with_name(predictions.name + ".bak"))
    by_unit = mapping.set_index("unit_id")
    result = original.copy()
    for column in SIM_COLUMNS:
        values = result["unit_id"].map(by_unit[column])
        if column in result.columns:
            # Fill only missing metadata; existing non-null values are retained.
            result[column] = result[column].where(result[column].notna(), values)
        else:
            result[column] = values

    # Verify the join did not mutate unrelated columns or row order.
    assert len(result) == len(original)
    assert result["unit_id"].equals(original["unit_id"])
    untouched = [column for column in original.columns if column not in SIM_COLUMNS]
    for column in untouched:
        assert result[column].equals(original[column]), column
    result.to_parquet(predictions, index=False)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="artifacts/predictions.parquet")
    parser.add_argument("--sim-map", default="artifacts/sim_unit_map.parquet")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    output = merge_predictions(args.predictions, args.sim_map, backup=not args.no_backup)
    print(f"merged: {output}")
    if not args.no_backup:
        print(f"backup: {output.with_name(output.name + '.bak')}")


if __name__ == "__main__":
    main()
