"""Offline, CPU-only root-cause analysis for batch quality metrics.

The engine deliberately has no knowledge of ``sim_truth.json``.  Recovery
comparison is kept in the command-line wrapper so the analysis can be run on
real (or simulated) batch data without leaking the planted answers.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


FAMILIES = ("crack", "hole", "rust", "scratch")
RATE_PREFIX = "defect_rate_"
DEFAULT_ASSUMPTIONS = ["D24"]


@dataclass(frozen=True)
class AnalysisConfig:
    """Small, explicit configuration surface for deterministic analysis."""

    alpha: float = 0.05
    max_lag: int = 8
    n_splits: int = 5
    random_state: int = 2026


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def meta(source: str, tag: str = "ADVISORY/SIMULATED", assumptions: Sequence[str] = DEFAULT_ASSUMPTIONS) -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "source": source,
        "assumption_ids": list(assumptions),
        "tag": tag,
    }


def _jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
    return value


def _rate_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {family: f"{RATE_PREFIX}{family}" for family in FAMILIES if f"{RATE_PREFIX}{family}" in frame}


def _process_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "batch_id", "ts", "variant", "shift", "supplier_lot", "station", "primary_station",
        "drift_flags", "meta_tag", "assumption_ids", "n_units", "fpy", "change_point_flag",
    }
    result: list[str] = []
    for col in frame.columns:
        if col in excluded or col.startswith(RATE_PREFIX) or col.startswith("predicted_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            result.append(col)
    return result


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "ts" in result:
        result["_order_ts"] = pd.to_datetime(result["ts"], errors="coerce", utc=True)
    else:
        result["_order_ts"] = pd.NaT
    if "batch_id" not in result:
        result["batch_id"] = np.arange(len(result)).astype(str)
    result = result.sort_values(["_order_ts", "batch_id"], kind="mergesort", na_position="last").reset_index(drop=True)
    return result


def _assert_no_holdout(frame: pd.DataFrame, context: str) -> None:
    """Defensive guard: this module must never inspect locked-holdout rows."""
    for col in ("split", "source_split"):
        if col in frame.columns:
            values = frame[col].astype(str).str.lower()
            if values.eq("holdout").any():
                raise ValueError(f"locked holdout rows are forbidden in {context}")


def _standardized(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.zeros_like(x, dtype=float)
    mean = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1))
    return (x - mean) / sd if sd > 0 else np.zeros_like(x, dtype=float)


def cusum(values: Sequence[float], reference: float | None = None, threshold: float = 4.0, slack: float = 0.5) -> dict[str, Any]:
    """Two-sided standardized CUSUM, returning candidate change indices."""
    x = _standardized(values)
    ref = 0.0 if reference is None else float(reference)
    pos = neg = 0.0
    changes: list[int] = []
    for i, value in enumerate(x):
        pos = max(0.0, pos + value - ref - slack)
        neg = min(0.0, neg + value - ref + slack)
        if pos >= threshold or neg <= -threshold:
            changes.append(i)
            pos = neg = 0.0
    return {"changes": changes, "n_changes": len(changes), "threshold": threshold, "slack": slack}


def p_chart(values: Sequence[float], n_units: Sequence[float] | None = None) -> dict[str, Any]:
    """Compute p-chart center/control limits and out-of-control indices."""
    p = np.asarray(values, dtype=float)
    n = np.ones(len(p), dtype=float) if n_units is None else np.maximum(np.asarray(n_units, dtype=float), 1.0)
    center = float(np.average(p, weights=n)) if len(p) else 0.0
    limits = 3.0 * np.sqrt(np.maximum(center * (1.0 - center) / n, 0.0))
    lower = np.maximum(0.0, center - limits)
    upper = np.minimum(1.0, center + limits)
    outside = np.flatnonzero((p < lower) | (p > upper)).astype(int).tolist()
    return {"center": center, "lower": lower.tolist(), "upper": upper.tolist(), "out_of_control": outside}


def change_point(values: Sequence[float]) -> dict[str, Any]:
    """Use ruptures when installed, otherwise use the local CUSUM fallback."""
    x = np.asarray(values, dtype=float)
    if len(x) < 8 or np.nanstd(x) == 0:
        return {"method": "cusum", "changes": [], "n_changes": 0}
    try:
        import ruptures as rpt  # type: ignore

        finite = np.nan_to_num(x, nan=float(np.nanmean(x)))
        points = rpt.Pelt(model="rbf").fit(finite.reshape(-1, 1)).predict(pen=3.0)
        changes = [int(point) for point in points if 0 < point < len(x)]
        return {"method": "ruptures.Pelt", "changes": changes, "n_changes": len(changes)}
    except ImportError:
        return {"method": "cusum", **cusum(x)}
    except Exception as exc:  # pragma: no cover - defensive fallback for version differences
        return {"method": "cusum", "warning": str(exc), **cusum(x)}


def cross_correlation_lag(parameter: Sequence[float], rate: Sequence[float], max_lag: int = 8) -> dict[str, Any]:
    """Return the lag with largest absolute correlation.

    Positive lag means a parameter at time ``t`` is compared with rate at
    ``t + lag`` (parameter change precedes the quality shift).
    """
    x = np.asarray(parameter, dtype=float)
    y = np.asarray(rate, dtype=float)
    candidates: list[tuple[int, float]] = []
    for lag in range(-min(max_lag, len(x) - 2), min(max_lag, len(x) - 2) + 1):
        if lag >= 0:
            a, b = x[: len(x) - lag], y[lag:]
        else:
            a, b = x[-lag:], y[: len(y) + lag]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 4 or np.nanstd(a[valid]) == 0 or np.nanstd(b[valid]) == 0:
            continue
        candidates.append((lag, float(np.corrcoef(a[valid], b[valid])[0, 1])))
    if not candidates:
        return {"lag": None, "correlation": None, "tested_lags": []}
    lag, corr = max(candidates, key=lambda pair: abs(pair[1]))
    return {"lag": int(lag), "correlation": corr, "tested_lags": [{"lag": int(k), "correlation": v} for k, v in candidates]}


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta, with ties contributing zero."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return 0.0
    diffs = a[:, None] - b[None, :]
    return float((np.sign(diffs).sum()) / diffs.size)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if not len(p):
        return p
    order = np.argsort(np.nan_to_num(p, nan=1.0))
    ranked = np.nan_to_num(p, nan=1.0)[order]
    adjusted = np.minimum(1.0, ranked * len(p) / np.arange(1, len(p) + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def _stratified_association(frame: pd.DataFrame, family: str, parameter: str) -> dict[str, Any]:
    y_col = f"{RATE_PREFIX}{family}"
    groups = frame.groupby("variant", dropna=False) if "variant" in frame else [("all", frame)]
    deltas: list[float] = []
    pvals: list[float] = []
    n = 0
    for _, group in groups:
        x = pd.to_numeric(group[parameter], errors="coerce")
        y = pd.to_numeric(group[y_col], errors="coerce")
        valid = x.notna() & y.notna()
        x, y = x[valid].to_numpy(float), y[valid].to_numpy(float)
        n += len(x)
        if len(x) < 6 or np.std(x) == 0 or np.std(y) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            rho, pval = stats.spearmanr(x, y)
        if not np.isfinite(rho) or not np.isfinite(pval):
            continue
        deltas.append(float(rho))
        pvals.append(float(pval))
    return {
        "effect_size": float(np.mean(deltas)) if deltas else 0.0,
        "p_value": float(min(pvals)) if pvals else 1.0,
        "n": n,
        "test": "stratified Spearman correlation",
    }


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    return float(stats.rankdata(score)[y.astype(bool)].mean() - stats.rankdata(score)[~y.astype(bool)].mean()) / max(1.0, len(score)) + 0.5


def _model_signal(frame: pd.DataFrame, family: str, parameters: list[str], config: AnalysisConfig) -> dict[str, Any]:
    """Batch-grouped CV model with a dependency-light fallback.

    Rates are continuous; the model predicts rates.  If sklearn is available,
    GradientBoostingRegressor and permutation importance are used.  Otherwise
    rank correlations provide a conservative, deterministic diagnostic.
    """
    y = pd.to_numeric(frame[f"{RATE_PREFIX}{family}"], errors="coerce").to_numpy(float)
    valid = np.isfinite(y)
    x = frame[parameters].apply(pd.to_numeric, errors="coerce").fillna(frame[parameters].median(numeric_only=True)).fillna(0.0).to_numpy(float)
    x, y = x[valid], y[valid]
    groups = frame.loc[valid, "batch_id"].astype(str).to_numpy() if "batch_id" in frame else np.arange(len(y)).astype(str)
    if len(y) < max(12, config.n_splits * 2) or not parameters:
        return {"method": "insufficient_data", "cv_lift": 0.0, "permutation_importance": {}, "baseline": None}
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n_splits = min(config.n_splits, len(np.unique(groups)))
        if n_splits < 2:
            raise ValueError("fewer than two batch groups")
        cv = GroupKFold(n_splits=n_splits)
        model = GradientBoostingRegressor(random_state=config.random_state, n_estimators=80, max_depth=2, loss="huber")
        scores = cross_val_score(model, x, y, groups=groups, cv=cv, scoring="neg_mean_absolute_error")
        shuffled_groups = np.random.default_rng(config.random_state).permutation(groups)
        baseline = cross_val_score(model, x, y, groups=shuffled_groups, cv=cv, scoring="neg_mean_absolute_error")
        model.fit(x, y)
        importance = permutation_importance(model, x, y, n_repeats=8, random_state=config.random_state, scoring="neg_mean_absolute_error")
        return {
            "method": "sklearn.GradientBoostingRegressor",
            "cv_mae": float(-np.mean(scores)),
            "shuffled_batch_cv_mae": float(-np.mean(baseline)),
            "cv_lift": float(np.mean(baseline) - np.mean(scores)),
            "permutation_importance": {p: float(v) for p, v in zip(parameters, importance.importances_mean)},
            "baseline": "shuffled_batch_labels",
        }
    except (ImportError, ValueError, TypeError) as exc:
        effects = {p: _stratified_association(frame, family, p)["effect_size"] for p in parameters}
        return {"method": "rank-correlation_fallback", "cv_lift": 0.0, "permutation_importance": effects, "baseline": "unavailable", "warning": str(exc)}


def _partial_dependence_statement(frame: pd.DataFrame, family: str, parameter: str) -> str:
    x = pd.to_numeric(frame[parameter], errors="coerce")
    y = pd.to_numeric(frame[f"{RATE_PREFIX}{family}"], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 8 or x[valid].nunique() < 4:
        return "threshold statement unavailable: insufficient variation"
    threshold = float(x[valid].median())
    high, low = y[valid & (x >= threshold)], y[valid & (x < threshold)]
    direction = "rises" if high.mean() > low.mean() else "falls"
    return f"risk {direction} above {threshold:.6g} (association; not proven causation)"


def _stability(frame: pd.DataFrame, family: str, parameter: str) -> dict[str, Any]:
    ordered = _ordered(frame)
    half = max(1, len(ordered) // 2)
    values: dict[str, float] = {}
    for label, part in (("first_half", ordered.iloc[:half]), ("second_half", ordered.iloc[half:])):
        values[label] = _stratified_association(part, family, parameter)["effect_size"]
    variant_values: dict[str, float] = {}
    if "variant" in ordered:
        for variant, part in ordered.groupby("variant", dropna=False):
            variant_values[str(variant)] = _stratified_association(part, family, parameter)["effect_size"]
    nonzero = [v for v in values.values() if abs(v) > 1e-12]
    stable_halves = len(nonzero) == 2 and np.sign(nonzero[0]) == np.sign(nonzero[1])
    return {"first_half": values["first_half"], "second_half": values["second_half"], "across_variants": variant_values, "stable": bool(stable_halves)}


def analyze(frame: pd.DataFrame, source: str = "batch_metrics.parquet", config: AnalysisConfig | None = None) -> dict[str, Any]:
    """Run M3 analysis and return the evidence artifact payload."""
    config = config or AnalysisConfig()
    _assert_no_holdout(frame, source)
    ordered = _ordered(frame)
    rates = _rate_columns(ordered)
    parameters = _process_columns(ordered)
    model_by_family = {family: _model_signal(ordered, family, parameters, config) for family in rates}
    associations: list[dict[str, Any]] = []
    for family, rate_col in rates.items():
        chart = p_chart(ordered[rate_col].to_numpy(float), ordered.get("n_units"))
        drift = cusum(ordered[rate_col].to_numpy(float))
        family_change = change_point(ordered[rate_col].to_numpy(float))
        for parameter in parameters:
            assoc = _stratified_association(ordered, family, parameter)
            lags = cross_correlation_lag(ordered[parameter].to_numpy(float), ordered[rate_col].to_numpy(float), config.max_lag)
            associations.append({"family": family, "parameter": parameter, **assoc, "lag": lags["lag"], "lag_correlation": lags["correlation"], "lag_detail": lags})
        # Keep the chart in a family-level record used by the card generator.
        for row in associations:
            if row["family"] == family:
                row["drift"] = {"p_chart": chart, "cusum": drift, "change_point": family_change}
    adjusted = benjamini_hochberg([row["p_value"] for row in associations])
    for row, p_adj in zip(associations, adjusted):
        row["p_adj"] = float(p_adj)
        row["model"] = model_by_family[row["family"]]
        row["threshold_statement"] = _partial_dependence_statement(ordered, row["family"], row["parameter"])
        row["stability"] = _stability(ordered, row["family"], row["parameter"])
    return {"source": source, "rows": len(ordered), "families": list(rates), "parameters": parameters, "associations": associations}


def _confidence(row: Mapping[str, Any], alpha: float) -> str:
    effect = abs(float(row.get("effect_size", 0.0)))
    p_adj = float(row.get("p_adj", 1.0))
    lift = float(row.get("model", {}).get("cv_lift", 0.0) or 0.0)
    stable = bool(row.get("stability", {}).get("stable", False))
    if p_adj <= alpha and effect >= 0.35 and stable and lift > 0:
        return "High"
    if p_adj <= 0.10 and effect >= 0.20:
        return "Med"
    return "Low"


def evidence_cards(result: Mapping[str, Any], source: str = "batch_metrics.parquet") -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    grouped = {family: [] for family in result.get("families", [])}
    for row in result.get("associations", []):
        grouped.setdefault(str(row["family"]), []).append(row)
    for family in result.get("families", []):
        rows = grouped.get(family, [])
        # Keep the strongest interpretable lead per family.  BH-FDR is still
        # computed over every family x parameter test above; this presentation
        # rule prevents several correlated/confounded parameters from becoming
        # duplicate root-cause cards.  A raw p <= .10 exploratory lead is
        # retained when no FDR-significant candidate exists, with Low tier.
        ranked = sorted(rows, key=lambda row: abs(float(row.get("effect_size", 0.0))), reverse=True)
        stable = [row for row in ranked if bool(row.get("stability", {}).get("stable", False))]
        significant = [row for row in ranked if float(row.get("p_adj", 1.0)) <= 0.10 and abs(float(row.get("effect_size", 0.0))) >= 0.20]
        exploratory = [row for row in ranked if float(row.get("p_value", 1.0)) <= 0.10 and abs(float(row.get("effect_size", 0.0))) >= 0.20]
        stable_significant = [row for row in stable if row in significant]
        stable_exploratory = [row for row in stable if row in exploratory]
        linked = (stable_significant or stable_exploratory or significant or exploratory)[:1]
        if not linked:
            cards.append({
                "id": f"RC-{family}-NO-LINK",
                "family": family,
                "hypothesis": f"no process link found for family {family}",
                "variables": [],
                "effect_size": 0.0,
                "p_adj": 1.0,
                "cv_lift": 0.0,
                "lag": None,
                "stability": {"stable": False},
                "confidence_tier": "Low",
                "caveat": "association, not proven causation; labeled-simulated layer D24",
                "verify_action": "collect more batch-level process and defect observations before changing a setpoint",
            })
            continue
        for row in linked:
            effect = float(row.get("effect_size", 0.0))
            direction = "higher" if effect >= 0 else "lower"
            cards.append({
                "id": f"RC-{family}-{row['parameter']}",
                "family": family,
                "hypothesis": f"{family} defect rate is {direction} as {row['parameter']} increases",
                "variables": [row["parameter"], f"{RATE_PREFIX}{family}"],
                "effect_size": effect,
                "p_adj": float(row.get("p_adj", 1.0)),
                "cv_lift": float(row.get("model", {}).get("cv_lift", 0.0) or 0.0),
                "lag": row.get("lag"),
                "stability": row.get("stability", {}),
                "confidence_tier": _confidence(row, 0.05),
                "caveat": "association, not proven causation; labeled-simulated layer D24",
                "verify_action": f"hold {row['parameter']} at a controlled setpoint and compare {family} rate on a future batch",
                "drift": row.get("drift", {}),
                "threshold_statement": row.get("threshold_statement"),
            })
    return {"meta": meta(source), "cards": cards}


def load_input(root: str | Path) -> tuple[pd.DataFrame, str]:
    """Load real artifacts, falling back to fixtures, with holdout guard."""
    root = Path(root)
    artifact = root / "artifacts" / "batch_metrics.parquet"
    fixture = root / "tests" / "fixtures" / "batch_metrics.parquet"
    path = artifact if artifact.exists() else fixture
    frame = pd.read_parquet(path)
    _assert_no_holdout(frame, str(path))
    return frame, str(path)
