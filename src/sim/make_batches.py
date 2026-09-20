"""Build the labelled-simulated factory layer used by the process demo.

The organizer data has no batch, process, or timestamp fields (D14).  This
module therefore makes those fields explicit simulated assumptions (D24/D25).
It is deliberately deterministic and CPU/offline only.  ``sim_truth.json`` is
written for the planted-effect recovery test; the production root-cause path
must read only ``batch_metrics.parquet``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


TAG = "ADVISORY/SIMULATED"
ASSUMPTION_IDS = ["D14", "D24", "D25"]
FAMILIES = ["crack", "hole", "rust", "scratch"]
STATION_MAP = {
    "hole": "Drilling",
    "scratch": "Milling",
    "crack": "Assembly",
    "rust": "environment",
}
DEFAULTS: dict[str, Any] = {
    "seed": 20260920,
    "n_batches": 120,
    "days": 5,
    "units_per_batch": 30,
    "max_units": 3600,
    "start_ts": "2026-01-05T00:00:00",
}


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read the config, tolerating the currently empty ``sim:`` block."""

    config_path = Path(path)
    if not config_path.exists():
        return {"sim": dict(DEFAULTS), "output_folder": "artifacts"}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    sim = config.get("sim") or {}
    merged = dict(DEFAULTS)
    merged.update(sim)
    config["sim"] = merged
    return config


def _stem(value: str) -> str:
    return Path(str(value).replace("\\", "/")).stem


def _allowed_units_from_splits(splits_path: str | Path) -> dict[str, dict[str, str]]:
    """Read only eligible split records; excluded split rows are never stored.

    The guard is intentionally line-oriented: a locked split record is not
    loaded into a dataframe or used for scoring.  Only ``dev`` and
    ``cond_holdout`` are admitted to the simulation pool.
    """

    allowed: dict[str, dict[str, str]] = {}
    path = Path(splits_path)
    if not path.exists():
        return allowed
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("split", ""))
            if split not in {"dev", "cond_holdout"}:
                continue
            file_name = str(row.get("file", ""))
            unit_id = _stem(file_name)
            if unit_id:
                allowed[unit_id] = {
                    "image_path": file_name,
                    "true_label": str(row.get("cls", "")),
                    "source_split": split,
                }
    assert all(meta["source_split"] in {"dev", "cond_holdout"} for meta in allowed.values())
    return allowed


def load_unit_pool(
    predictions_path: str | Path | None = "artifacts/predictions.parquet",
    splits_path: str | Path = "artifacts/splits.csv",
    *,
    seed: int = DEFAULTS["seed"],
    max_units: int = DEFAULTS["max_units"],
) -> pd.DataFrame:
    """Load a scored, eligible unit pool without admitting locked records."""

    allowed = _allowed_units_from_splits(splits_path)
    prediction_path = Path(predictions_path) if predictions_path else None
    if prediction_path is not None and prediction_path.exists():
        scored = pd.read_parquet(prediction_path)
        if "unit_id" not in scored.columns:
            raise ValueError("predictions.parquet must contain unit_id")
        # The split map is the source of truth for eligibility.  This also
        # excludes any prediction row whose provenance is absent or unknown.
        if not allowed and not Path(splits_path).exists() and "source_split" in scored.columns:
            # Unit tests and standalone callers may provide a pre-filtered
            # scored pool without a split file.  Admit only the two allowed
            # provenance labels in that case.
            scored = scored[scored["source_split"].astype(str).isin({"dev", "cond_holdout"})].copy()
            allowed = {
                str(row["unit_id"]): {
                    "image_path": str(row.get("image_path", "")),
                    "true_label": str(row.get("true_label", "")),
                    "source_split": str(row["source_split"]),
                }
                for row in scored.to_dict("records")
            }
        else:
            scored = scored[scored["unit_id"].astype(str).isin(allowed)].copy()
        if scored.empty and allowed:
            # A concurrently produced predictions file may be a partial or
            # differently named fixture. Do not admit unmatched IDs; fall
            # back to the eligible split pool with prediction fields null.
            rows = [dict(unit_id=unit_id, **meta) for unit_id, meta in allowed.items()]
            pool = pd.DataFrame(rows)
            for column in ["pred_family", "decision", "condition"]:
                pool[column] = None
        else:
            scored["unit_id"] = scored["unit_id"].astype(str)
            if "source_split" in scored.columns:
                assert not scored["source_split"].astype(str).isin({"holdout"}).any()
            for column, default in {
                "image_path": None,
                "true_label": None,
                "pred_family": None,
                "decision": None,
                "condition": None,
            }.items():
                if column not in scored.columns:
                    scored[column] = default
            for idx, row in scored.iterrows():
                meta = allowed[row["unit_id"]]
                image_path = row["image_path"]
                true_label = row["true_label"]
                scored.at[idx, "image_path"] = meta["image_path"] if pd.isna(image_path) else image_path
                scored.at[idx, "true_label"] = meta["true_label"] if pd.isna(true_label) else true_label
                scored.at[idx, "source_split"] = meta["source_split"]
            pool = scored
    else:
        rows = [dict(unit_id=unit_id, **meta) for unit_id, meta in allowed.items()]
        pool = pd.DataFrame(rows)
        for column in ["pred_family", "decision", "condition"]:
            pool[column] = None

    if pool.empty:
        raise ValueError("No eligible dev/cond_holdout units were found")
    # Keep a stable, deterministic subset when a much larger scored pool is
    # supplied.  Never sample from an ineligible split.
    pool = pool.drop_duplicates("unit_id").sort_values("unit_id").reset_index(drop=True)
    if len(pool) > max_units:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(pool), size=max_units, replace=False))
        pool = pool.iloc[chosen].reset_index(drop=True)
    assert pool["unit_id"].is_unique
    assert not pool["source_split"].astype(str).isin({"holdout"}).any()
    return pool


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_parquet(df: pd.DataFrame, path: Path, *, artifact_name: str) -> None:
    """Write a parquet with both visible provenance columns and metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata.update(
            {
                b"meta.tag": TAG.encode(),
                b"meta.assumption_ids": _json(ASSUMPTION_IDS).encode(),
                b"meta.artifact": artifact_name.encode(),
            }
        )
        pq.write_table(table.replace_schema_metadata(metadata), path)
    except ImportError:
        df.to_parquet(path, index=False)


def _process_frame(batch_index: int, variant: str, rng: np.random.Generator) -> dict[str, Any]:
    """Create process readings for one batch.

    Variant and clock time intentionally explain part of ``decoy_parameter``;
    this is the confounding challenge for stratified recovery.
    """

    day_fraction = batch_index / 119.0
    hour = batch_index % 24
    shift = ("Shift-1", "Shift-2", "Shift-3")[(hour // 8) % 3]
    variant_offset = {"A": -0.16, "B": 0.0, "C": 0.16}[variant]
    changeovers = 1.0 + (0.7 if shift == "Shift-1" else 0.25) + (variant == "C") * 0.35
    tool_cycle = (batch_index % 18) / 17.0
    # The midpoint is a deliberate change point for humidity.
    humidity = 0.43 + 0.05 * day_fraction + (0.19 if batch_index >= 60 else 0.0)
    humidity += float(rng.normal(0, 0.012))
    fixture_wear = np.clip(0.08 + 0.78 * day_fraction + float(rng.normal(0, 0.025)), 0, 1)
    tool_wear = np.clip(tool_cycle + float(rng.normal(0, 0.025)), 0, 1)
    decoy = 0.95 + variant_offset + 0.22 * day_fraction + float(rng.normal(0, 0.035))
    return {
        "variant": variant,
        "shift": shift,
        "supplier_lot": f"LOT-{1 + (batch_index // 10):02d}",
        "coolant_humidity": float(humidity),
        "fixture_wear": float(fixture_wear),
        "tool_wear_hours": float(8 + 64 * tool_wear),
        "changeovers_per_shift": float(changeovers),
        "ambient_temp_c": float(22.0 + 2.0 * day_fraction + rng.normal(0, 0.25)),
        "coolant_temp_c": float(18.0 + 0.7 * humidity + rng.normal(0, 0.15)),
        "spindle_vibration": float(0.25 + 0.30 * fixture_wear + rng.normal(0, 0.025)),
        "line_speed_mpm": float(1.25 - 0.08 * fixture_wear + rng.normal(0, 0.025)),
        "feed_rate_mm_s": float(4.0 + 0.20 * variant_offset + rng.normal(0, 0.05)),
        "pressure_bar": float(4.8 + 0.3 * variant_offset + rng.normal(0, 0.06)),
        "decoy_parameter": float(decoy),
        "time_drift_index": float(day_fraction),
        "change_point_flag": bool(batch_index >= 60),
    }


def _assign_variant(batch_index: int) -> str:
    # Three variants with changing mix across the five-day clock.
    return ("A", "B", "C")[(batch_index // 8 + batch_index // 24) % 3]


def _family_probabilities(process: Mapping[str, Any], batch_index: int) -> dict[str, float]:
    humidity_risk = float(_sigmoid((float(process["coolant_humidity"]) - 0.56) / 0.045))
    rust = 0.012 + 0.045 * humidity_risk
    scratch = 0.012 + 0.035 * float(process["fixture_wear"])
    hole = 0.012 + 0.035 * (float(process["tool_wear_hours"]) - 8.0) / 64.0
    # Crack is intentionally constant with respect to all process variables.
    crack = 0.012
    return {"crack": crack, "hole": hole, "rust": rust, "scratch": scratch}


def _sample_batch_labels(size: int, process: Mapping[str, Any], batch_index: int, rng: np.random.Generator) -> list[str]:
    """Sample a batch while keeping the null-control crack rate process-free.

    A fixed one-unit crack quota per batch is used at this scale.  The quota
    prevents random low-count crack noise from appearing correlated with the
    deliberately drifting parameters; the unit chosen within the batch is
    still random and independent of process readings.
    """

    probs = _family_probabilities(process, batch_index)
    n_crack = max(1, int(round(size * probs["crack"])))
    labels = ["normal"] * size
    crack_positions = rng.choice(size, size=n_crack, replace=False)
    for position in crack_positions:
        labels[int(position)] = "crack"
    remaining = [index for index in range(size) if labels[index] != "crack"]
    categories = ["normal", "hole", "rust", "scratch"]
    non_crack = [1.0 - sum(probs.values()) + probs["crack"]] + [probs[family] for family in categories[1:]]
    non_crack = np.asarray(non_crack, dtype=float)
    non_crack /= non_crack.sum()
    sampled = rng.choice(categories, size=len(remaining), p=non_crack).tolist()
    for position, label in zip(remaining, sampled):
        labels[position] = label
    return labels


def _sample_truth(
    pool: pd.DataFrame,
    processes: list[dict[str, Any]],
    batch_indices: list[int],
    rng: np.random.Generator,
) -> tuple[list[str], list[int]]:
    labels: list[str] = []
    batch_for_unit: list[int] = []
    for batch_index, process in zip(batch_indices, processes):
        batch_size = batch_indices.count(batch_index)
        probs = _family_probabilities(process, batch_index)
        categories = ["normal"] + FAMILIES
        weights = [1.0 - sum(probs.values())] + [probs[family] for family in FAMILIES]
        labels.extend(rng.choice(categories, size=batch_size, p=weights).tolist())
        batch_for_unit.extend([batch_index] * batch_size)
    return labels, batch_for_unit


def generate_simulation(
    *,
    output_dir: str | Path = "artifacts",
    predictions_path: str | Path | None = "artifacts/predictions.parquet",
    splits_path: str | Path = "artifacts/splits.csv",
    config_path: str | Path = "config.yaml",
) -> dict[str, Path]:
    """Generate all four simulation artifacts and return their paths."""

    config = load_config(config_path)
    sim = config["sim"]
    seed = int(sim["seed"])
    n_batches = int(sim["n_batches"])
    output = Path(output_dir)
    pool = load_unit_pool(predictions_path, splits_path, seed=seed, max_units=int(sim["max_units"]))
    rng = np.random.default_rng(seed)

    variants = [_assign_variant(i) for i in range(n_batches)]
    process_rows = [_process_frame(i, variants[i], rng) for i in range(n_batches)]
    # Fill exactly the available scored units, spread across 120 batches.
    n_units = len(pool)
    batch_sizes = np.full(n_batches, n_units // n_batches, dtype=int)
    # Randomize which clock positions receive the remainder units.  Otherwise
    # a harmless 29-vs-30 size difference would become a time confounder for
    # the fixed-rate crack null control.
    extra_positions = np.random.default_rng(seed + 1).choice(
        n_batches, size=n_units % n_batches, replace=False
    )
    batch_sizes[extra_positions] += 1
    batch_for_unit = np.repeat(np.arange(n_batches), batch_sizes).tolist()
    labels: list[str] = []
    for batch_index, size in enumerate(batch_sizes):
        labels.extend(_sample_batch_labels(int(size), process_rows[batch_index], batch_index, rng))
    assert len(labels) == len(pool) == len(batch_for_unit)

    start = pd.Timestamp(sim["start_ts"])
    unit_rows: list[dict[str, Any]] = []
    for unit_index, (source_row, batch_index, true_family) in enumerate(
        zip(pool.to_dict("records"), batch_for_unit, labels)
    ):
        process = process_rows[batch_index]
        ts = start + pd.Timedelta(hours=int(batch_index)) + pd.Timedelta(minutes=unit_index % 60)
        station = STATION_MAP.get(true_family, "Inspection")
        condition = source_row.get("condition")
        if pd.isna(condition):
            condition = "simulated_nominal"
        unit_rows.append(
            {
                "unit_id": str(source_row["unit_id"]),
                "batch_id": f"B{batch_index + 1:03d}",
                "variant": process["variant"],
                "condition": str(condition),
                "station": station,
                "shift": process["shift"],
                "supplier_lot": process["supplier_lot"],
                "ts": ts,
                "true_family": true_family,
                "pred_family": source_row.get("pred_family"),
                "decision": source_row.get("decision"),
                "image_path": source_row.get("image_path"),
                "source_split": source_row.get("source_split"),
                "meta_tag": TAG,
                "assumption_ids": _json(ASSUMPTION_IDS),
            }
        )
    unit_map = pd.DataFrame(unit_rows)

    batch_rows: list[dict[str, Any]] = []
    for batch_index, process in enumerate(process_rows):
        in_batch = unit_map[unit_map["batch_id"] == f"B{batch_index + 1:03d}"]
        rates = {
            f"defect_rate_{family}": float((in_batch["true_family"] == family).mean())
            for family in FAMILIES
        }
        predicted = in_batch["pred_family"].fillna("").astype(str)
        pred_rates = {
            f"predicted_defect_rate_{family}": float((predicted == family).mean())
            for family in FAMILIES
        }
        batch_rows.append(
            {
                "batch_id": f"B{batch_index + 1:03d}",
                "ts": start + pd.Timedelta(hours=int(batch_index)),
                "variant": process["variant"],
                "shift": process["shift"],
                "supplier_lot": process["supplier_lot"],
                "n_units": int(len(in_batch)),
                **rates,
                "fpy": float((in_batch["true_family"] == "normal").mean()),
                **pred_rates,
                **{key: value for key, value in process.items() if key not in {"variant", "shift", "supplier_lot"}},
                "primary_station": "Inspection",
                "drift_flags": _json(["humidity_change_point"] if process["change_point_flag"] else []),
                "meta_tag": TAG,
                "assumption_ids": _json(ASSUMPTION_IDS),
            }
        )
    batch_metrics = pd.DataFrame(batch_rows)

    replay = unit_map[
        [
            "unit_id",
            "ts",
            "batch_id",
            "station",
            "decision",
            "variant",
            "shift",
            "supplier_lot",
            "condition",
            "meta_tag",
            "assumption_ids",
        ]
    ].sort_values(["ts", "unit_id"]).reset_index(drop=True)

    effects = [
        {
            "family": "rust",
            "parameter": "coolant_humidity",
            "form": "threshold_with_1_batch_lag",
            "size": 0.045,
            "station": STATION_MAP["rust"],
            "assumption_id": "D24",
        },
        {
            "family": "scratch",
            "parameter": "fixture_wear",
            "form": "monotone_gradual_drift",
            "size": 0.035,
            "station": STATION_MAP["scratch"],
            "assumption_id": "D24",
        },
        {
            "family": "hole",
            "parameter": "tool_wear_hours",
            "form": "monotone_sawtooth_reset_at_tool_change",
            "size": 0.035,
            "station": STATION_MAP["hole"],
            "assumption_id": "D24",
        },
        {
            "family": "crack",
            "parameter": None,
            "form": "no_link_control",
            "size": 0.0,
            "station": STATION_MAP["crack"],
            "assumption_id": "D24",
        },
    ]
    truth = {
        "meta": {
            "tag": TAG,
            "assumption_ids": ASSUMPTION_IDS,
            "read_only_by": "planted-effect recovery test",
            "caveat": "association, not proven causation; simulated linkage",
        },
        "effects": effects,
        "null_parameters": ["ambient_temp_c", "coolant_temp_c", "spindle_vibration", "decoy_parameter"],
        "family_station_map": STATION_MAP,
        "generator": {"seed": seed, "n_batches": n_batches, "days": int(sim["days"])},
    }

    _write_parquet(batch_metrics, output / "batch_metrics.parquet", artifact_name="batch_metrics")
    _write_parquet(replay, output / "replay_stream.parquet", artifact_name="replay_stream")
    _write_parquet(unit_map, output / "sim_unit_map.parquet", artifact_name="sim_unit_map")
    truth_path = output / "sim_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "batch_metrics": output / "batch_metrics.parquet",
        "replay_stream": output / "replay_stream.parquet",
        "sim_unit_map": output / "sim_unit_map.parquet",
        "sim_truth": truth_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--predictions", default="artifacts/predictions.parquet")
    parser.add_argument("--splits", default="artifacts/splits.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or config.get("output_folder", "artifacts")
    paths = generate_simulation(
        output_dir=output,
        predictions_path=args.predictions,
        splits_path=args.splits,
        config_path=args.config,
    )
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
