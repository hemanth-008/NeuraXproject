"""Build deterministic M6 recommendations from computed artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_profit import REPO_ROOT, _calibration_summary, build_profit, resolve_input
from src.recs.engine import TIER_WEIGHTS, delta_summary, score_recommendation, tier_weight


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _margin_bounds(value: Any, mean: float) -> tuple[float, float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _num(value[0], mean), _num(value[1], mean)
    return mean, mean


def _scenario_pairs(df: pd.DataFrame, parameter: str, *, baseline: float, intervention: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Pair scenario rows by every non-intervened scenario dimension."""

    if parameter not in df.columns:
        return [], []
    base = df[df[parameter].map(lambda x: abs(_num(x) - baseline) < 1e-12)].copy()
    new = df[df[parameter].map(lambda x: abs(_num(x) - intervention) < 1e-12)].copy()
    if base.empty or new.empty:
        return [], []
    excluded = {parameter, "throughput_mean", "throughput_p05", "throughput_p95", "margin_mean", "margin_ci"}
    keys = [c for c in ["model", "demand_level", "rework_rate", "capacity_plus_at_bottleneck", "bottleneck_station"] if c in df.columns and c not in excluded]
    if not keys:
        keys = ["_pair"]
        base["_pair"] = 0
        new["_pair"] = 0
    # For an intervention comparison, remove the changed field from the keys.
    keys = [c for c in keys if c != parameter]
    if not keys:
        base["_pair"] = 0
        new["_pair"] = 0
        keys = ["_pair"]
    base_rows: list[dict[str, float]] = []
    new_rows: list[dict[str, float]] = []
    for _, b in base.iterrows():
        mask = pd.Series(True, index=new.index)
        for key in keys:
            mask &= new[key].astype(str).eq(str(b[key]))
        matches = new[mask]
        if matches.empty:
            continue
        n = matches.iloc[0]
        b_mean = _num(b.get("throughput_mean"))
        n_mean = _num(n.get("throughput_mean"))
        b_low = _num(b.get("throughput_p05"), b_mean)
        b_high = _num(b.get("throughput_p95"), b_mean)
        n_low = _num(n.get("throughput_p05"), n_mean)
        n_high = _num(n.get("throughput_p95"), n_mean)
        b_margin = _num(b.get("margin_mean"))
        n_margin = _num(n.get("margin_mean"))
        b_ml, b_mh = _margin_bounds(b.get("margin_ci"), b_margin)
        n_ml, n_mh = _margin_bounds(n.get("margin_ci"), n_margin)
        base_rows.append({"mean": b_mean, "low": b_low, "high": b_high, "margin": b_margin, "margin_low": b_ml, "margin_high": b_mh})
        new_rows.append({"mean": n_mean, "low": n_low, "high": n_high, "margin": n_margin, "margin_low": n_ml, "margin_high": n_mh})
    return base_rows, new_rows


def _delta_for_pairs(base: list[dict[str, float]], new: list[dict[str, float]]) -> tuple[dict[str, Any], dict[str, Any]]:
    throughput = delta_summary(base, new)
    margin_base = [{"mean": r["margin"], "low": r["margin_low"], "high": r["margin_high"]} for r in base]
    margin_new = [{"mean": r["margin"], "low": r["margin_low"], "high": r["margin_high"]} for r in new]
    margin = delta_summary(margin_base, margin_new)
    return throughput, margin


def _baseline_throughput(df: pd.DataFrame) -> float:
    subset = df.copy()
    if "rework_rate" in subset:
        subset = subset[subset["rework_rate"].map(lambda x: abs(_num(x)) < 1e-12)]
    if "capacity_plus_at_bottleneck" in subset:
        subset = subset[subset["capacity_plus_at_bottleneck"].map(lambda x: abs(_num(x)) < 1e-12)]
    if subset.empty:
        subset = df
    return _num(subset["throughput_mean"].mean()) if "throughput_mean" in subset else 0.0


def _flow_context(flow: dict[str, Any]) -> tuple[str, list[str], str]:
    bottleneck = flow.get("bottleneck", {})
    if isinstance(bottleneck, dict):
        for model in ["model1", "model2", "model3"]:
            item = bottleneck.get(model)
            if isinstance(item, dict) and item.get("station"):
                evidence = item.get("evidence", [])
                return str(item["station"]), [f"flow_metrics.bottleneck.{model}", *[str(x) for x in evidence[:3]]], str(item.get("caveat", "association, not proven causation"))
    stations = flow.get("stations", [])
    if stations:
        row = max(stations, key=lambda x: _num(x.get("utilization_mean", x.get("rho"))))
        return str(row.get("station", "unknown")), ["flow_metrics.stations"], "association, not proven causation"
    return "unknown", ["flow_metrics"], "association, not proven causation"


def _card_context(cards: list[dict[str, Any]]) -> tuple[str, list[str], str, str]:
    if not cards:
        return "the supported process setpoint", ["evidence_cards.none"], "Low", "No evidence card was available."
    card = cards[0]
    card_id = str(card.get("id", "evidence_cards.0"))
    subject = str(card.get("hypothesis") or card.get("family") or "the supported process setpoint")
    tier = str(card.get("confidence_tier") or card.get("tier") or "Low")
    return subject, [card_id], tier, str(card.get("caveat") or "association, not proven causation")


def _recommendation(
    *,
    rec_id: str,
    category: str,
    action: str,
    rationale: str,
    throughput: dict[str, Any],
    margin: dict[str, Any],
    evidence_ids: list[str],
    confidence_tier: str,
    assumptions: list[str],
    invalidation: str,
    verify: str,
) -> dict[str, Any]:
    rec = {
        "id": rec_id,
        "category": category,
        "action": action,
        "rationale": rationale,
        "expected_delta_throughput": {
            "mean": float(throughput["mean"]),
            "interval_5_95": [float(x) for x in throughput["interval_5_95"]],
            "units": "good units/hour",
            "n_pairs": int(throughput.get("n_pairs", 0)),
        },
        "expected_delta_margin": {
            "mean": float(margin["mean"]),
            "interval_5_95": [float(x) for x in margin["interval_5_95"]],
            "units": "margin/hour",
            "n_pairs": int(margin.get("n_pairs", 0)),
        },
        "delta_throughput": float(throughput["mean"]),
        "delta_margin": float(margin["mean"]),
        "evidence_ids": evidence_ids,
        "confidence_tier": confidence_tier,
        "tier_weight": tier_weight(confidence_tier),
        "assumptions": assumptions,
        "what_could_invalidate": invalidation,
        "how_to_verify": verify,
        "tag": "ADVISORY / SIMULATED",
    }
    rec["score"] = score_recommendation(rec["expected_delta_margin"]["mean"], confidence_tier)
    return rec


def build_recommendations(
    *,
    sim_results_path: Path = REPO_ROOT / "artifacts" / "sim_results.parquet",
    flow_path: Path = REPO_ROOT / "artifacts" / "flow_metrics.json",
    evidence_path: Path = REPO_ROOT / "artifacts" / "evidence_cards.json",
    calibration_path: Path = REPO_ROOT / "artifacts" / "calibration.json",
    profit_path: Path = REPO_ROOT / "artifacts" / "profit_model.json",
    output_dir: Path = REPO_ROOT / "artifacts",
) -> dict[str, Any]:
    sim_path, sim_fixture = resolve_input(sim_results_path, "sim_results.parquet")
    flow_resolved, flow_fixture = resolve_input(flow_path, "flow_metrics.json")
    evidence_resolved, evidence_fixture = resolve_input(evidence_path, "evidence_cards.json")
    cal_resolved, cal_fixture = resolve_input(calibration_path, "calibration.json")
    sim = pd.read_parquet(sim_path)
    if "margin_mean" not in sim.columns or "margin_ci" not in sim.columns:
        build_profit(sim_results_path=sim_path, calibration_path=cal_resolved, output_dir=output_dir)
        sim = pd.read_parquet(sim_path)
    flow = _json(flow_resolved)
    evidence = _json(evidence_resolved)
    calibration = _json(cal_resolved)
    cards = evidence.get("cards", evidence.get("evidence_cards", []))
    if not isinstance(cards, list):
        cards = []
    station, flow_evidence, flow_caveat = _flow_context(flow)
    subject, process_evidence, process_tier, process_caveat = _card_context(cards)
    cal = _calibration_summary(calibration)
    baseline_throughput = _baseline_throughput(sim)
    assumptions = ["D18", "D22", "D14", *[str(x) for x in evidence.get("meta", {}).get("assumption_ids", [])], "economics.price", "economics.unit_cost"]
    assumptions = list(dict.fromkeys(assumptions))

    # (a) Setpoint/process move: compare the highest and lowest simulated rework
    # rates when that scenario axis exists. No pairs means a computed zero delta.
    rework_values = sorted(_num(x) for x in sim.get("rework_rate", pd.Series(dtype=float)).dropna().unique())
    if len(rework_values) >= 2:
        b, n = _scenario_pairs(sim, "rework_rate", baseline=max(rework_values), intervention=min(rework_values))
        process_th, process_margin = _delta_for_pairs(b, n)
    else:
        process_th, process_margin = ({"mean": 0.0, "interval_5_95": [0.0, 0.0], "n_pairs": 0},) * 2
    recs = [
        _recommendation(
            rec_id="REC_PROCESS_SETPOINT",
            category="process_setpoint",
            action=f"Verify and correct {subject} at the process setpoint before changing inspection policy.",
            rationale=f"The evidence card points to {subject}; the paired simulated rework comparison estimates the operational delta. {process_caveat}",
            throughput=process_th,
            margin=process_margin,
            evidence_ids=process_evidence,
            confidence_tier=process_tier,
            assumptions=assumptions,
            invalidation="The evidence card is observational and the simulated rework coupling may not hold on the live line; association, not proven causation.",
            verify="Run a controlled setpoint trial, hold mix and demand constant, and compare defect/rework and margin against the pre-trial baseline.",
        )
    ]

    # (b) Capacity/flow move: compare the smallest and largest capacity uplift.
    cap_values = sorted(_num(x) for x in sim.get("capacity_plus_at_bottleneck", pd.Series(dtype=float)).dropna().unique())
    if len(cap_values) >= 2:
        b, n = _scenario_pairs(sim, "capacity_plus_at_bottleneck", baseline=min(cap_values), intervention=max(cap_values))
        capacity_th, capacity_margin = _delta_for_pairs(b, n)
    else:
        capacity_th, capacity_margin = ({"mean": 0.0, "interval_5_95": [0.0, 0.0], "n_pairs": 0},) * 2
    recs.append(
        _recommendation(
            rec_id="REC_CAPACITY_BOTTLENECK",
            category="capacity_flow",
            action=f"Pilot capacity relief at the {station} bottleneck, subject to the three-line flow check.",
            rationale=f"Flow evidence identifies {station}; paired capacity scenarios estimate the delta. {flow_caveat}",
            throughput=capacity_th,
            margin=capacity_margin,
            evidence_ids=flow_evidence,
            confidence_tier="High" if "triangulation_agreement: true" in flow_caveat else "Med",
            assumptions=assumptions,
            invalidation="The aggregate flow snapshot may not preserve queue direction or future mix; association, not proven causation.",
            verify="Add capacity in a reversible trial at the named station and compare throughput, WIP, utilization, and margin over matched demand windows.",
        )
    )

    # (c) Inspection-policy move: use the calibration cost-vs-threshold curve.
    policy_costs = cal["chosen_cost_per_unit"], cal["accept_all_cost_per_unit"], cal["reject_all_cost_per_unit"]
    best_policy_cost = min(policy_costs)
    policy_margin_delta = (cal["chosen_cost_per_unit"] - best_policy_cost) * baseline_throughput
    policy_margin = {"mean": float(policy_margin_delta), "interval_5_95": [float(policy_margin_delta), float(policy_margin_delta)], "n_pairs": 1}
    policy_th = {"mean": 0.0, "interval_5_95": [0.0, 0.0], "n_pairs": 1}
    recs.append(
        _recommendation(
            rec_id="REC_INSPECTION_POLICY",
            category="inspection_policy",
            action=f"Operate at the minimum observed expected-cost point on the calibration curve (threshold={cal['operating_threshold']:.2f}).",
            rationale=f"Calibration compares chosen={cal['chosen_cost_per_unit']:.6f}, accept-all={cal['accept_all_cost_per_unit']:.6f}, reject-all={cal['reject_all_cost_per_unit']:.6f} expected cost per unit.",
            throughput=policy_th,
            margin=policy_margin,
            evidence_ids=["calibration.cost_curve", "calibration.operating_point"],
            confidence_tier="Med" if cal["curve"] else "Low",
            assumptions=assumptions,
            invalidation="The calibration curve is from development data and assumes the configured economic costs; association, not proven causation.",
            verify="Recalibrate on a fresh non-holdout validation sample, then confirm expected cost and FAR/FRR before changing the live threshold.",
        )
    )

    # (d) Scheduling/mix move: explicitly quantify zero because no changeover or
    # mix scenario was supplied; this is safer than inventing a benefit.
    zero = {"mean": 0.0, "interval_5_95": [0.0, 0.0], "n_pairs": 0}
    recs.append(
        _recommendation(
            rec_id="REC_SCHEDULING_MIX",
            category="scheduling",
            action="Pilot a scheduling/mix change only after changeover and variant-mix measurements are available.",
            rationale="The available flow and simulation artifacts do not contain an observable changeover or variant-mix scenario, so no benefit is quantified.",
            throughput=zero,
            margin=zero,
            evidence_ids=["flow_metrics.changeover", "sim_results.variant_mix"],
            confidence_tier="Low",
            assumptions=assumptions,
            invalidation="A measured changeover or mix effect could change the ranking; association, not proven causation.",
            verify="Collect matched variant sequence, changeover duration, throughput, WIP, and margin data, then rerun the scenario grid.",
        )
    )

    recs.sort(key=lambda r: (-float(r["score"]), str(r["id"])))
    source = "fixture" if sim_fixture or flow_fixture or evidence_fixture or cal_fixture else "artifacts"
    artifact = {
        "meta": {
            "generated_at": _now(),
            "source": source,
            "assumption_ids": assumptions,
            "tag": "ADVISORY / SIMULATED",
        },
        "recommendations": recs,
        "scoring": {"formula": "expected delta margin x evidence-confidence tier weight", "tier_weights": TIER_WEIGHTS},
        "limitations": [
            "Recommendations are advisory/simulated templates; association, not proven causation.",
            "No LLM or network call is used.",
            "Scheduling impact is zero until changeover/mix data exists; this is not a claim of no effect.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recommendations.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"build_recs: source={source} recommendations={len(recs)} top={recs[0]['id']} score={recs[0]['score']:.6f}")
    for rec in recs:
        print(f"build_recs: {rec['id']} delta_throughput={rec['delta_throughput']:.6f} delta_margin={rec['delta_margin']:.6f} score={rec['score']:.6f}")
    print(f"build_recs: wrote {output_dir / 'recommendations.json'}")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-results", type=Path, default=REPO_ROOT / "artifacts" / "sim_results.parquet")
    parser.add_argument("--flow", type=Path, default=REPO_ROOT / "artifacts" / "flow_metrics.json")
    parser.add_argument("--evidence", type=Path, default=REPO_ROOT / "artifacts" / "evidence_cards.json")
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "artifacts" / "calibration.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    build_recommendations(
        sim_results_path=args.sim_results,
        flow_path=args.flow,
        evidence_path=args.evidence,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
