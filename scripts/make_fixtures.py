"""Create deterministic, small contract fixtures for local tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_ROWS = 50
FIXTURE_TIME = "2026-01-01T00:00:00Z"


def _meta(tag: str = "SIMULATED") -> dict[str, object]:
    return {
        "generated_at": FIXTURE_TIME,
        "source": "fixture",
        "assumption_ids": ["D24"] if tag == "SIMULATED" else ["economics.fixture"],
        "tag": tag,
    }


def _write_json(directory: Path, name: str, payload: dict[str, object]) -> None:
    (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_parquets(directory: Path) -> None:
    ids = [f"FIXTURE_{index:03d}" for index in range(N_ROWS)]
    decisions = ["ACCEPT" if index % 2 == 0 else "REJECT" for index in range(N_ROWS)]
    predictions = pd.DataFrame(
        {
            "unit_id": ids,
            "image_path": [f"fixture/{unit_id}.png" for unit_id in ids],
            "batch_id": [f"FIXTURE_BATCH_{index // 5:02d}" for index in range(N_ROWS)],
            "variant": ["A"] * N_ROWS,
            "condition": ["nominal"] * N_ROWS,
            "station": ["Assembly"] * N_ROWS,
            "ts": [FIXTURE_TIME] * N_ROWS,
            "p_defect": [0.1 + 0.8 * (index % 2) for index in range(N_ROWS)],
            "class_probs": [json.dumps({"crack": 0.8, "hole": 0.05, "normal": 0.1, "rust": 0.03, "scratch": 0.02})] * N_ROWS,
            "pred_family": ["normal" if index % 2 == 0 else "crack" for index in range(N_ROWS)],
            "decision": decisions,
            "flag": ["NONE"] * N_ROWS,
            "conformal_set": [json.dumps(["normal"]) if index % 2 == 0 else json.dumps(["crack"]) for index in range(N_ROWS)],
            "anomaly_score": [None] * N_ROWS,
            "heatmap_path": [None] * N_ROWS,
            "boxes_mask_ref": [None] * N_ROWS,
            "localization_basis": [None] * N_ROWS,
            "true_label": ["normal" if index % 2 == 0 else "crack" for index in range(N_ROWS)],
            "source_split": ["fixture"] * N_ROWS,
            "box_xyxy": [None] * N_ROWS,
            "novel_score": [None] * N_ROWS,
            "top2": [None] * N_ROWS,
            "tile_boxes": [None] * N_ROWS,
            "mixed_defect_flag": [False] * N_ROWS,
            "ensemble_agreement": [1.0] * N_ROWS,
        }
    )
    predictions.to_parquet(directory / "predictions.parquet", index=False)

    batch = pd.DataFrame(
        {
            "batch_id": [f"FIXTURE_BATCH_{index:02d}" for index in range(N_ROWS)],
            "ts": [FIXTURE_TIME] * N_ROWS,
            "variant": ["A"] * N_ROWS,
            "n_units": [10] * N_ROWS,
            "defect_rate_crack": [0.1] * N_ROWS,
            "defect_rate_hole": [0.0] * N_ROWS,
            "defect_rate_rust": [0.0] * N_ROWS,
            "defect_rate_scratch": [0.0] * N_ROWS,
            "fpy": [0.9] * N_ROWS,
            "coolant_humidity": [0.4] * N_ROWS,
            "drift_flags": [json.dumps([])] * N_ROWS,
        }
    )
    batch.to_parquet(directory / "batch_metrics.parquet", index=False)

    replay = pd.DataFrame(
        {
            "unit_id": ids,
            "ts": [FIXTURE_TIME] * N_ROWS,
            "batch_id": [f"FIXTURE_BATCH_{index // 5:02d}" for index in range(N_ROWS)],
            "station": ["Assembly"] * N_ROWS,
            "decision": decisions,
        }
    )
    replay.to_parquet(directory / "replay_stream.parquet", index=False)

    unit_map = pd.DataFrame(
        {
            "unit_id": ids,
            "batch_id": [f"FIXTURE_BATCH_{index // 5:02d}" for index in range(N_ROWS)],
            "variant": ["A"] * N_ROWS,
            "station": ["Assembly"] * N_ROWS,
            "condition": ["nominal"] * N_ROWS,
            "ts": [FIXTURE_TIME] * N_ROWS,
        }
    )
    unit_map.to_parquet(directory / "sim_unit_map.parquet", index=False)

    results = pd.DataFrame(
        {
            "scenario_id": [f"FIXTURE_SCENARIO_{index:03d}" for index in range(N_ROWS)],
            "parameters": [json.dumps({"demand": 10})] * N_ROWS,
            "throughput_mean": [100.0] * N_ROWS,
            "throughput_ci": [json.dumps([95.0, 105.0])] * N_ROWS,
            "wip": [12.0] * N_ROWS,
            "margin_mean": [500.0] * N_ROWS,
            "margin_ci": [json.dumps([450.0, 550.0])] * N_ROWS,
        }
    )
    results.to_parquet(directory / "sim_results.parquet", index=False)


def _write_csvs(directory: Path) -> None:
    robustness = pd.DataFrame(
        {
            "perturbation": ["rotate90" if index % 2 else "nominal" for index in range(N_ROWS)],
            "acc": [0.98] * N_ROWS,
            "macro_f1": [0.97] * N_ROWS,
            "n": [10] * N_ROWS,
            "fold": [0] * N_ROWS,
        }
    )
    robustness.to_csv(directory / "robustness_orient.csv", index=False)
    robustness.to_csv(directory / "robustness.csv", index=False)


def _write_npz(directory: Path) -> None:
    embeddings = np.arange(N_ROWS * 4, dtype=np.float32).reshape(N_ROWS, 4)
    labels = np.array(["normal" if index % 2 == 0 else "crack" for index in range(N_ROWS)])
    np.savez(directory / "knn_bank.npz", bank=embeddings)


def _write_jsons(directory: Path) -> None:
    effects = [{"family": "crack", "parameter": "temperature", "form": "monotone", "size": 0.2, "station": "Assembly"}]
    _write_json(directory, "sim_truth.json", {"meta": _meta(), "planted_effects": effects})
    _write_json(directory, "evidence_cards.json", {"meta": _meta(), "cards": [{"id": "FIXTURE_CARD_001", "family": "crack", "hypothesis": "fixture", "caveat": "association, not proven causation"}]})
    _write_json(directory, "rootcause_recovery.json", {"meta": _meta(), "planted_effects": effects, "found_effects": effects, "matches": ["crack"], "false_positives": [], "misses": [], "no_link_control_clean": True})
    _write_json(directory, "flow_metrics.json", {"meta": _meta(), "stations": [{"station": "Assembly", "rho": 0.7}], "bottleneck": {"station": "Assembly"}, "evidence": [], "loss_waterfall": []})
    _write_json(directory, "twin_validation.json", {"meta": _meta(), "mape": 0.1, "pass": True})
    _write_json(directory, "profit_model.json", {"meta": _meta("ADVISORY"), "model_type": "accounting_identity", "features": ["defect_rate"], "out_of_sample_metrics": {}, "interval_method": "fixture"})
    _write_json(directory, "recommendations.json", {"meta": _meta("ADVISORY"), "recommendations": [{"id": "FIXTURE_REC_001", "tag": "ADVISORY"}]})
    _write_json(directory, "calibration.json", {"meta": _meta(), "ece": 0.01, "reliability_bins": [], "risk_coverage": {}, "conformal": {"q_hat": 0.1}, "operating_threshold": 0.5, "cost_table": {}})
    _write_json(directory, "novelty.json", {"meta": _meta(), "knn": {"k": 5, "threshold": 0.2}})
    _write_json(directory, "localization.json", {"meta": _meta(), "n": N_ROWS, "method": "heatmap-derived", "caveats": ["pseudo-GT"]})
    _write_json(directory, "status.json", {"meta": _meta(), "status": "fixture"})
    _write_json(directory, "holdout_result.json", {"meta": _meta(), "n": 10, "accuracy": 0.9, "macro_f1": 0.9, "per_class": {}, "confusion_matrix": [[1]], "frr": 0.1, "far": 0.1, "expected_cost": 1.0, "ece": 0.01, "conformal_coverage": 0.9, "uncertain_rate": 0.1, "novel_rate": 0.0, "review_rate": 0.1})
    _write_json(directory, "holdout_runs.json", {"meta": _meta(), "count": 0})
    _write_json(directory, "flow_v0.json", {"meta": _meta(), "model1": {}, "model2": {}, "model3_time_check": {}})


def make_fixtures(directory: str | Path = Path("tests/fixtures")) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_parquets(output)
    _write_csvs(output)
    _write_npz(output)
    _write_jsons(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="tests/fixtures")
    args = parser.parse_args()
    print(f"wrote fixtures to {make_fixtures(args.dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
