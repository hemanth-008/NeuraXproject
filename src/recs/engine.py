"""Deterministic, offline recommendation scoring and scenario helpers."""

from __future__ import annotations

from typing import Any, Iterable


TIER_WEIGHTS = {"high": 1.0, "med": 0.75, "medium": 0.75, "low": 0.5, "unknown": 0.25}


def tier_weight(tier: str | None) -> float:
    return TIER_WEIGHTS.get(str(tier or "unknown").strip().lower(), 0.25)


def score_recommendation(expected_delta_margin: float, confidence_tier: str | None) -> float:
    """M6 score: expected delta margin multiplied by tier weight."""

    return float(expected_delta_margin) * tier_weight(confidence_tier)


def interval_from_values(values: Iterable[float]) -> list[float]:
    """Return a deterministic 5--95% interval, or [0, 0] for no evidence."""

    vals = sorted(float(v) for v in values)
    if not vals:
        return [0.0, 0.0]
    n = len(vals)

    def quantile(p: float) -> float:
        if n == 1:
            return vals[0]
        pos = (n - 1) * p
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return vals[lo] + (vals[hi] - vals[lo]) * frac

    return [float(quantile(0.05)), float(quantile(0.95))]


def delta_summary(
    baseline: Iterable[dict[str, float]],
    intervention: Iterable[dict[str, float]],
    *,
    mean_key: str = "mean",
    low_key: str = "low",
    high_key: str = "high",
) -> dict[str, Any]:
    """Pair scenarios and summarize intervention minus baseline deltas."""

    base = list(baseline)
    new = list(intervention)
    n = min(len(base), len(new))
    if n == 0:
        return {"mean": 0.0, "interval_5_95": [0.0, 0.0], "n_pairs": 0}
    means = [float(new[i][mean_key]) - float(base[i][mean_key]) for i in range(n)]
    # A conservative interval for a difference uses the intervention low/high
    # against the baseline high/low, then summarizes paired rows.
    lows = [float(new[i][low_key]) - float(base[i][high_key]) for i in range(n)]
    highs = [float(new[i][high_key]) - float(base[i][low_key]) for i in range(n)]
    return {
        "mean": float(sum(means) / len(means)),
        "interval_5_95": [
            float(interval_from_values(lows)[0]),
            float(interval_from_values(highs)[1]),
        ],
        "n_pairs": n,
    }
