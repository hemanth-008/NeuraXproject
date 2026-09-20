from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.sim.make_batches import generate_simulation


def _pool(n: int = 3600) -> pd.DataFrame:
    labels = ["normal", "crack", "hole", "rust", "scratch"]
    return pd.DataFrame(
        {
            "unit_id": [f"unit_{i:05d}" for i in range(n)],
            "image_path": [f"normal/u_{i:05d}.png" for i in range(n)],
            "true_label": [labels[i % len(labels)] for i in range(n)],
            "pred_family": [labels[(i + 1) % len(labels)] for i in range(n)],
            "decision": ["REJECT" if i % 3 else "ACCEPT" for i in range(n)],
            "source_split": ["dev" if i % 2 else "cond_holdout" for i in range(n)],
            "condition": [None] * n,
        }
    )


def _run(tmp_path: Path, seed_config: str | None = None) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pool_path = tmp_path / "predictions.parquet"
    _pool().to_parquet(pool_path, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(seed_config or "output_folder: artifacts\nsim:\n  seed: 17\n", encoding="utf-8")
    return generate_simulation(
        output_dir=tmp_path / "out",
        predictions_path=pool_path,
        splits_path=tmp_path / "missing-splits.csv",
        config_path=config,
    )


def test_deterministic_outputs(tmp_path: Path) -> None:
    first = _run(tmp_path / "one")
    second = _run(tmp_path / "two")
    for name in ("batch_metrics", "replay_stream", "sim_unit_map"):
        left = pd.read_parquet(first[name])
        right = pd.read_parquet(second[name])
        pd.testing.assert_frame_equal(left, right)
    assert json.loads(first["sim_truth"].read_text(encoding="utf-8")) == json.loads(
        second["sim_truth"].read_text(encoding="utf-8")
    )


def test_planted_parameters_shift_their_family_rates(tmp_path: Path) -> None:
    paths = _run(tmp_path)
    frame = pd.read_parquet(paths["batch_metrics"])
    for parameter, rate_column in (
        ("coolant_humidity", "defect_rate_rust"),
        ("fixture_wear", "defect_rate_scratch"),
        ("tool_wear_hours", "defect_rate_hole"),
    ):
        low = frame[parameter] <= frame[parameter].median()
        high = ~low
        assert frame.loc[high, rate_column].mean() > frame.loc[low, rate_column].mean()


def test_crack_is_independent_of_process_parameters(tmp_path: Path) -> None:
    frame = pd.read_parquet(_run(tmp_path)["batch_metrics"])
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["defect_rate_crack"])
    correlations = numeric.corrwith(frame["defect_rate_crack"]).abs().dropna()
    assert correlations.empty or float(correlations.max()) < 0.35


def test_variant_explains_part_of_decoy_parameter(tmp_path: Path) -> None:
    frame = pd.read_parquet(_run(tmp_path)["batch_metrics"])
    means = frame.groupby("variant")["decoy_parameter"].mean()
    assert float(means.max() - means.min()) > 0.15


def test_artifacts_have_simulation_provenance_and_no_excluded_ids(tmp_path: Path) -> None:
    paths = _run(tmp_path)
    for name in ("batch_metrics", "replay_stream", "sim_unit_map"):
        frame = pd.read_parquet(paths[name])
        assert set(frame["meta_tag"]) == {"ADVISORY/SIMULATED"}
        assert set(frame["assumption_ids"]) == {'["D14","D24","D25"]'}
        if "unit_id" in frame.columns:
            assert not frame["unit_id"].astype(str).str.contains("locked", case=False).any()
    truth = json.loads(paths["sim_truth"].read_text(encoding="utf-8"))
    assert truth["meta"]["tag"] == "ADVISORY/SIMULATED"
    assert truth["meta"]["assumption_ids"] == ["D14", "D24", "D25"]
