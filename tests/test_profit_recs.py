"""Focused C7 tests; these use only fixtures and never read locked holdout rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_profit import build_profit
from scripts.build_recs import build_recommendations
from src.profit.model import Economics, accounting_margin, margin_interval
from src.recs.engine import score_recommendation


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_accounting_identity_arithmetic() -> None:
    economics = Economics(
        price=100.0,
        unit_cost=20.0,
        scrap_cost=30.0,
        rework_cost=10.0,
        c_escape=50.0,
        downtime_cost_per_hour=250.0,
    )
    actual = accounting_margin(
        10.0,
        economics,
        scrap_units_per_hour=1.0,
        rework_units_per_hour=2.0,
        false_accept_rate=0.1,
        downtime_hours=0.5,
    )
    assert actual == 10.0 * 80.0 - 30.0 - 20.0 - 10.0 * 0.1 * 50.0 - 125.0


def test_margin_interval_is_pushed_through_identity() -> None:
    economics = Economics(100.0, 20.0, 30.0, 10.0, 50.0, 250.0)
    low, high = margin_interval(10.0, 12.0, economics, rework_rate=0.1, false_accept_rate=0.05)
    assert low <= high
    assert low == accounting_margin(10.0, economics, rework_units_per_hour=1.0, false_accept_rate=0.05)


def test_recommendation_score_is_monotonic() -> None:
    assert score_recommendation(20.0, "High") > score_recommendation(10.0, "High")
    assert score_recommendation(10.0, "High") > score_recommendation(10.0, "Low")


def test_fixture_builds_are_tagged_and_keep_sim_columns(tmp_path: Path) -> None:
    fixture_sim = FIXTURES / "sim_results.parquet"
    sim_copy = tmp_path / "sim_results.parquet"
    pd.read_parquet(fixture_sim).to_parquet(sim_copy, index=False)
    output = tmp_path / "artifacts"
    profit = build_profit(
        sim_results_path=sim_copy,
        calibration_path=FIXTURES / "calibration.json",
        output_dir=output,
    )
    recs = build_recommendations(
        sim_results_path=output / "sim_results.parquet",
        flow_path=FIXTURES / "flow_metrics.json",
        evidence_path=FIXTURES / "evidence_cards.json",
        calibration_path=FIXTURES / "calibration.json",
        output_dir=output,
    )
    assert profit["model_type"] == "accounting_identity"
    result = pd.read_parquet(output / "sim_results.parquet")
    assert set(pd.read_parquet(fixture_sim).columns).issubset(result.columns)
    assert {"margin_mean", "margin_ci"}.issubset(result.columns)
    assert recs["meta"]["tag"] == "ADVISORY / SIMULATED"
    for rec in recs["recommendations"]:
        assert rec["assumptions"]
        assert "ADVISORY" in rec["tag"]


def test_recommendation_ranking_matches_score(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    build_profit(
        sim_results_path=FIXTURES / "sim_results.parquet",
        calibration_path=FIXTURES / "calibration.json",
        output_dir=output,
    )
    artifact = build_recommendations(
        sim_results_path=output / "sim_results.parquet",
        flow_path=FIXTURES / "flow_metrics.json",
        evidence_path=FIXTURES / "evidence_cards.json",
        calibration_path=FIXTURES / "calibration.json",
        output_dir=output,
    )
    scores = [rec["score"] for rec in artifact["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    for rec in artifact["recommendations"]:
        assert rec["score"] == rec["expected_delta_margin"]["mean"] * rec["tier_weight"]
