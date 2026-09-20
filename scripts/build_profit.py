"""Build the offline M5 accounting-identity profit artifact.

This script never reads the locked holdout. It uses only config economics,
calibration, and the simulated-results grid; missing upstream JSON/parquet files
fall back to tests/fixtures and are marked as fixture-derived.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.profit.model import (
    Economics,
    accounting_margin,
    economics_from_config,
    margin_interval,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_input(path: Path, fixture_name: str) -> tuple[Path, bool]:
    """Resolve a real input or its deterministic fixture."""

    if path.exists():
        return path, False
    fixture = REPO_ROOT / "tests" / "fixtures" / fixture_name
    if fixture.exists():
        return fixture, True
    raise FileNotFoundError(f"missing input {path} and fixture {fixture}")


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _interval(value: Any, mean: float) -> tuple[float, float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_numeric(value[0], mean), _numeric(value[1], mean))
    return (mean, mean)


def _scenario_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    if key in row.index:
        return _numeric(row[key], default)
    raw = row.get("parameters")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if isinstance(raw, dict):
        return _numeric(raw.get(key), default)
    return default


def _calibration_summary(calibration: dict[str, Any]) -> dict[str, Any]:
    """Normalize real and fixture calibration shapes without inventing metrics."""

    section = calibration.get("clean_oof")
    if not isinstance(section, dict):
        section = calibration
    curve = section.get("cost_curve") or calibration.get("cost_curve") or []
    normalized: list[dict[str, float]] = []
    for item in curve:
        if not isinstance(item, dict):
            continue
        if "threshold" not in item or "expected_cost_per_unit" not in item:
            continue
        normalized.append(
            {
                "threshold": _numeric(item["threshold"]),
                "expected_cost_per_unit": _numeric(item["expected_cost_per_unit"]),
            }
        )
    if not normalized:
        table = calibration.get("cost_table", {})
        if isinstance(table, dict):
            for threshold, cost in table.items():
                normalized.append(
                    {"threshold": _numeric(threshold), "expected_cost_per_unit": _numeric(cost)}
                )
    normalized.sort(key=lambda x: x["threshold"])
    threshold = section.get("operating_threshold", calibration.get("operating_threshold"))
    if threshold is None and normalized:
        threshold = min(normalized, key=lambda x: x["expected_cost_per_unit"])["threshold"]
    threshold = _numeric(threshold, 0.0)

    def cost_at(target: float) -> float:
        if not normalized:
            return 0.0
        nearest = min(normalized, key=lambda x: abs(x["threshold"] - target))
        return float(nearest["expected_cost_per_unit"])

    errors = section.get("error_rates", {})
    far = _numeric(errors.get("far"), 0.0) if isinstance(errors, dict) else 0.0
    return {
        "operating_threshold": threshold,
        "false_accept_rate": far,
        "curve": normalized,
        "chosen_cost_per_unit": cost_at(threshold),
        "accept_all_cost_per_unit": cost_at(1.0),
        "reject_all_cost_per_unit": cost_at(0.0),
        "source_section": "clean_oof" if section is calibration.get("clean_oof") else "fixture/root",
    }


def _row_margin(row: pd.Series, economics: Economics, false_accept_rate: float) -> tuple[float, list[float]]:
    mean = _numeric(row.get("throughput_mean"))
    low, high = _interval(row.get("throughput_ci"), mean)
    if "throughput_p05" in row.index:
        low = _numeric(row.get("throughput_p05"), low)
    if "throughput_p95" in row.index:
        high = _numeric(row.get("throughput_p95"), high)
    rework_rate = _scenario_value(row, "rework_rate", 0.0)
    scrap_rate = _scenario_value(row, "scrap_rate", 0.0)
    downtime_hours = _scenario_value(row, "downtime_hours", 0.0)
    center = accounting_margin(
        mean,
        economics,
        scrap_units_per_hour=mean * scrap_rate,
        rework_units_per_hour=mean * rework_rate,
        false_accept_rate=false_accept_rate,
        downtime_hours=downtime_hours,
    )
    interval = margin_interval(
        low,
        high,
        economics,
        rework_rate=rework_rate,
        scrap_rate=scrap_rate,
        false_accept_rate=false_accept_rate,
        downtime_hours=downtime_hours,
    )
    return center, [interval[0], interval[1]]


def _waterfall(df: pd.DataFrame, economics: Economics, far: float) -> dict[str, float]:
    if df.empty:
        return {"gross_contribution": 0.0, "scrap_cost": 0.0, "rework_cost": 0.0, "escape_cost": 0.0, "downtime_cost": 0.0, "margin_per_hour": 0.0}
    rows = []
    for _, row in df.iterrows():
        throughput = _numeric(row.get("throughput_mean"))
        rework_rate = _scenario_value(row, "rework_rate", 0.0)
        scrap_rate = _scenario_value(row, "scrap_rate", 0.0)
        downtime = _scenario_value(row, "downtime_hours", 0.0)
        rows.append(
            {
                "gross_contribution": throughput * (economics.price - economics.unit_cost),
                "scrap_cost": throughput * scrap_rate * economics.scrap_cost,
                "rework_cost": throughput * rework_rate * economics.rework_cost,
                "escape_cost": throughput * far * economics.c_escape,
                "downtime_cost": downtime * economics.downtime_cost_per_hour,
            }
        )
    result = {key: float(sum(item[key] for item in rows) / len(rows)) for key in rows[0]}
    result["margin_per_hour"] = float(
        result["gross_contribution"]
        - result["scrap_cost"]
        - result["rework_cost"]
        - result["escape_cost"]
        - result["downtime_cost"]
    )
    return result


def build_profit(
    *,
    config_path: Path = REPO_ROOT / "config.yaml",
    sim_results_path: Path = REPO_ROOT / "artifacts" / "sim_results.parquet",
    calibration_path: Path = REPO_ROOT / "artifacts" / "calibration.json",
    output_dir: Path = REPO_ROOT / "artifacts",
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sim_path, sim_fixture = resolve_input(sim_results_path, "sim_results.parquet")
    cal_path, cal_fixture = resolve_input(calibration_path, "calibration.json")
    sim = pd.read_parquet(sim_path)
    calibration = _load_json(cal_path)
    economics = economics_from_config(config)
    cal = _calibration_summary(calibration)

    out = sim.copy()
    margins: list[float] = []
    intervals: list[list[float]] = []
    for _, row in out.iterrows():
        mean, interval = _row_margin(row, economics, cal["false_accept_rate"])
        margins.append(mean)
        intervals.append(interval)
    out["margin_mean"] = margins
    out["margin_ci"] = intervals
    output_dir.mkdir(parents=True, exist_ok=True)
    target_sim = output_dir / "sim_results.parquet"
    temp_sim = output_dir / "sim_results.parquet.c7tmp"
    out.to_parquet(temp_sim, index=False)
    os.replace(temp_sim, target_sim)

    assumptions = ["D18", "D22", *economics.assumption_ids]
    if "rework_rate" in out.columns:
        assumptions.append("D14")
    assumptions = list(dict.fromkeys(assumptions))
    waterfall = _waterfall(out, economics, cal["false_accept_rate"])
    source = "fixture" if sim_fixture or cal_fixture else "artifacts"
    artifact = {
        "meta": {
            "generated_at": _now(),
            "source": source,
            "assumption_ids": assumptions,
            "tag": "ADVISORY / SIMULATED",
        },
        "model_type": "accounting_identity",
        "features": [
            "throughput_mean",
            "rework_rate",
            "scrap_rate (0 when unavailable)",
            "false_accept_rate",
            "downtime_hours (0 when unavailable)",
        ],
        "out_of_sample_metrics": {
            "rows_scored": int(len(out)),
            "margin_mean": float(out["margin_mean"].mean()) if len(out) else 0.0,
            "margin_min": float(out["margin_mean"].min()) if len(out) else 0.0,
            "margin_max": float(out["margin_mean"].max()) if len(out) else 0.0,
            "accounting_identity": "margin/h = good units/h x (price - variable cost) - scrap - rework - escape cost x false-accept rate - downtime cost",
        },
        "interval_method": "throughput_p05_p95 pushed through accounting identity",
        "economics": {
            name: {"value": float(getattr(economics, name)), "assumption_id": f"economics.{name}", "tag": "ASSUMED"}
            for name in [
                "price",
                "unit_cost",
                "scrap_cost",
                "rework_cost",
                "c_escape",
                "c_false_reject",
                "c_review",
                "downtime_cost_per_hour",
            ]
        },
        "calibration_policy": {
            "source_section": cal["source_section"],
            "operating_threshold": cal["operating_threshold"],
            "false_accept_rate": cal["false_accept_rate"],
            "expected_cost_per_unit": {
                "chosen_operating_point": cal["chosen_cost_per_unit"],
                "accept_all": cal["accept_all_cost_per_unit"],
                "reject_all": cal["reject_all_cost_per_unit"],
            },
            "cost_curve": cal["curve"],
        },
        "margin_waterfall": waterfall,
        "limitations": [
            "Organizers supplied no economics; every economic value is ASSUMED under D18/D22.",
            "Scrap rate and downtime hours were not present in sim_results and therefore contribute zero, not an invented estimate.",
            "Simulated flow/profit coupling is advisory; association, not proven causation.",
        ],
    }
    (output_dir / "profit_model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"build_profit: source={source} rows={len(out)} margin_mean={artifact['out_of_sample_metrics']['margin_mean']:.6f}")
    print(f"build_profit: inspection_cost chosen={cal['chosen_cost_per_unit']:.6f} accept_all={cal['accept_all_cost_per_unit']:.6f} reject_all={cal['reject_all_cost_per_unit']:.6f}")
    print(f"build_profit: wrote {target_sim} and {output_dir / 'profit_model.json'}")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--sim-results", type=Path, default=REPO_ROOT / "artifacts" / "sim_results.parquet")
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "artifacts" / "calibration.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    build_profit(
        config_path=args.config,
        sim_results_path=args.sim_results,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
