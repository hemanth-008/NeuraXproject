"""Build the M4 flow, twin-validation, and scenario artifacts.

Run from the repository root:

    .venv\\Scripts\\python.exe scripts\\build_flow.py

The script reads only the three production simulation CSVs.  It never opens
artifacts/splits.csv, so the locked-holdout rows are outside this analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.flow.engine import (  # noqa: E402
    SIM_REPLICATIONS,
    scenario_grid,
    summarize_model1,
    summarize_model2,
    summarize_model3,
    validate_twin,
)
from src.ingest.loaders import load_model1, load_model2  # noqa: E402


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _headline(summary: dict[str, Any]) -> str:
    b = summary["bottleneck"]
    ev = summary["bottleneck"]["evidence_values"]
    return (
        f"bottleneck={b['station']} | "
        f"highest_utilization={ev['highest_utilization']['rho']:.6f} | "
        f"queue_wip_pileup={ev['queue_wip_pileup']['station']} "
        f"wip={ev['queue_wip_pileup']['wip']:.6f} | "
        f"starvation_proxy={ev['starvation_blocking_proxy']['station']} "
        f"rho={ev['starvation_blocking_proxy']['rho']:.6f}"
    )


def build(data_dir: Path, artifacts_dir: Path, replications: int, seed: int) -> None:
    # Only production CSVs are loaded here.  In particular, never add a read
    # of artifacts/splits.csv: locked-holdout rows are forbidden to this lane.
    model1 = load_model1(data_dir)
    model2 = load_model2(data_dir)
    m1_summary = summarize_model1(model1)
    m2_summary = summarize_model2(model2)
    m3_summary = summarize_model3(data_dir)
    summaries = {"model1": m1_summary, "model2": m2_summary, "model3": m3_summary}

    generated_at = datetime.now(timezone.utc).isoformat()
    all_stations = []
    all_evidence = []
    all_waterfalls = []
    for model_name, summary in summaries.items():
        all_stations.extend([{**row, "model": model_name} for row in summary["station_metrics"]])
        all_evidence.append({"model": model_name, **summary["bottleneck"]})
        all_waterfalls.append({"model": model_name, **summary["loss_waterfall"]})

    flow_payload = {
        "artifact": "flow_metrics",
        "tag": "ADVISORY / SIMULATED",
        # Compatibility envelope for contracts/flow_metrics.schema.json.
        # The model-specific detail remains under `models` below.
        "meta": {
            "generated_at": generated_at,
            "source": "data/Model 1/Model_1.csv; data/Model 2/Model_2.csv; data/Model 3/Model_3.csv",
            "assumption_ids": ["D14"],
            "tag": "ADVISORY / SIMULATED",
        },
        "stations": all_stations,
        "bottleneck": {model_name: summary["bottleneck"] for model_name, summary in summaries.items()},
        "evidence": all_evidence,
        "loss_waterfall": all_waterfalls,
        "assumptions": {
            "D14": {
                "label": "ASSUMPTION",
                "formula": "rho_new = rho_baseline * (1 + rework_rate)",
                "caveat": "production files contain no rework or defect field; association, not proven causation",
            },
            "data_limits": {
                "downtime": "not observable",
                "changeover": "not observable",
                "scrap": "not observable",
                "rework": "not observable; D14 is an editable assumption",
            },
        },
        "models": summaries,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifacts_dir / "flow_metrics.json", flow_payload)

    twin = validate_twin({"model1": m1_summary, "model2": m2_summary}, replications, seed)
    final_mape = max(
        result["final_mape_percent"]["max"] for result in twin["models"].values()
    )
    twin["meta"] = {
        "generated_at": generated_at,
        "source": "SimPy initial validation with analytical Monte Carlo fallback",
        "assumption_ids": ["D14"],
        "tag": "ADVISORY / SIMULATED",
    }
    # Compatibility envelope for contracts/twin_validation.schema.json.
    twin["mape"] = float(final_mape)
    twin["pass"] = bool(twin["pass_at_25_percent"])
    _write_json(artifacts_dir / "twin_validation.json", twin)

    # The scenario artifact intentionally contains throughput and WIP only;
    # C7 owns any margin columns.
    scenarios = scenario_grid({"model1": m1_summary, "model2": m2_summary}, replications, seed)
    scenarios.to_parquet(artifacts_dir / "sim_results.parquet", index=False)

    print("=== M4 flow build ===")
    print(f"model1 rows={len(model1)} demand_levels={model1['Demand'].nunique()} {_headline(m1_summary)}")
    print(f"model2 rows={len(model2)} demand_levels={model2['Demand'].nunique()} {_headline(m2_summary)}")
    print(
        f"model2 disclosure: {m2_summary['required_disclosure']}"
    )
    print(
        f"model3 rows={m3_summary['n_rows']} n_unique_time_now={m3_summary['n_unique_time_now']} "
        f"semantics={m3_summary['data_semantics']} {_headline(m3_summary)}"
    )
    print(f"twin backend initial={twin['initial_backend']} selected={twin['selected_backend']} replications={replications}")
    for model_name, model_result in twin["models"].items():
        print(
            f"{model_name} initial_mape_max={model_result['initial_mape_percent']['max']:.6f}% "
            f"final_mape_max={model_result['final_mape_percent']['max']:.6f}% "
            f"pass_at_25_percent={model_result['pass_at_25_percent']} "
            f"fallback={model_result['fallback']}"
        )
    print(f"scenario_rows={len(scenarios)} scenario_replications={replications}")
    print(f"wrote={artifacts_dir / 'flow_metrics.json'}")
    print(f"wrote={artifacts_dir / 'twin_validation.json'}")
    print(f"wrote={artifacts_dir / 'sim_results.parquet'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--replications", type=int, default=SIM_REPLICATIONS)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.replications < 200:
        parser.error("--replications must be >= 200")
    build(args.data_dir, args.artifacts_dir, args.replications, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
