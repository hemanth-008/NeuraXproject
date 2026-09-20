"""CPU-only calibration, conformal, decision, and novelty primitives.

The functions in this module deliberately have no project-data dependencies so
they can be tested on small synthetic logits.  The builder in
``scripts/build_calibration.py`` is the only module that knows the Kaggle file
layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll: float
    raw_nll: float
    t_bound_hit: bool
    surface_flat: bool


def _as_logits(logits: Any) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        raise ValueError("logits must have shape (n_samples, n_classes), n>0")
    if not np.isfinite(arr).all():
        raise ValueError("logits contain non-finite values")
    return arr


def _as_labels(labels: Any, n: int, n_classes: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(y) != n or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("labels do not match logits")
    return y


def softmax(logits: Any, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax with temperature scaling."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    x = _as_logits(logits) / float(temperature)
    x = x - x.max(axis=1, keepdims=True)
    out = np.exp(x)
    return out / np.maximum(out.sum(axis=1, keepdims=True), EPS)


def nll(logits: Any, labels: Any, temperature: float = 1.0) -> float:
    arr = _as_logits(logits)
    y = _as_labels(labels, len(arr), arr.shape[1])
    p = softmax(arr, temperature)
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], EPS)).mean())


def fit_temperature(
    logits: Any,
    labels: Any,
    lower: float = 0.05,
    upper: float = 10.0,
    grid_size: int = 241,
) -> TemperatureFit:
    """Fit T by a log grid followed by bounded golden-section refinement.

    ``t_bound_hit`` is true for an optimum at either search bound or for a
    numerically flat NLL surface.  We expose both conditions because a flat
    surface is not evidence that the apparent optimum is meaningful.
    """
    if not (0 < lower < upper) or grid_size < 3:
        raise ValueError("invalid temperature search bounds")
    arr = _as_logits(logits)
    y = _as_labels(labels, len(arr), arr.shape[1])
    grid = np.geomspace(lower, upper, int(grid_size))
    values = np.asarray([nll(arr, y, float(t)) for t in grid])
    best_i = int(np.argmin(values))
    lo_i = max(0, best_i - 1)
    hi_i = min(len(grid) - 1, best_i + 1)
    lo, hi = float(grid[lo_i]), float(grid[hi_i])

    # Refine in log(T) to avoid a large-T bias.  If the grid winner is on a
    # boundary there is no interior interval to refine.
    if 0 < best_i < len(grid) - 1:
        a, b = np.log(lo), np.log(hi)
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        c, d = b - (b - a) / phi, a + (b - a) / phi
        fc, fd = nll(arr, y, float(np.exp(c))), nll(arr, y, float(np.exp(d)))
        for _ in range(80):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b - a) / phi
                fc = nll(arr, y, float(np.exp(c)))
            else:
                a, c, fc = c, d, fd
                d = a + (b - a) / phi
                fd = nll(arr, y, float(np.exp(d)))
        candidate = float(np.exp((a + b) / 2.0))
    else:
        candidate = float(grid[best_i])
    candidate = float(np.clip(candidate, lower, upper))
    fitted_nll = nll(arr, y, candidate)
    raw = nll(arr, y, 1.0)
    surface_scale = max(1.0, abs(float(values.mean())))
    flat = bool(float(values.max() - values.min()) <= 1e-6 * surface_scale)
    bound = bool(candidate <= lower * (1.0 + 1e-8) or candidate >= upper * (1.0 - 1e-8))
    return TemperatureFit(candidate, fitted_nll, raw, bool(bound or flat), flat)


def calibration_metrics(logits: Any, labels: Any, temperature: float = 1.0, bins: int = 15) -> dict[str, Any]:
    """Return NLL, multiclass Brier, ECE, and reliability bins."""
    arr = _as_logits(logits)
    y = _as_labels(labels, len(arr), arr.shape[1])
    p = softmax(arr, temperature)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y).astype(float)
    one_hot = np.eye(arr.shape[1], dtype=np.float64)[y]
    reliability: list[dict[str, float | int]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for i in range(int(bins)):
        mask = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= edges[i + 1]))
        count = int(mask.sum())
        if count:
            acc = float(correct[mask].mean())
            mean_conf = float(conf[mask].mean())
            ece += count / len(y) * abs(acc - mean_conf)
        else:
            acc = None
            mean_conf = None
        reliability.append({"bin": i, "lower": float(edges[i]), "upper": float(edges[i + 1]), "n": count, "accuracy": acc, "confidence": mean_conf})
    return {
        "nll": nll(arr, y, temperature),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "ece": float(ece),
        "reliability": reliability,
        "probs": p,
    }


def conformal_quantile(scores: Any, alpha: float = 0.10) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(s) or not (0 < alpha < 1):
        raise ValueError("conformal calibration needs scores and 0<alpha<1")
    k = int(np.ceil((len(s) + 1) * (1.0 - alpha)))
    k = min(max(k, 1), len(s))
    return float(np.sort(s)[k - 1])


def conformal_sets(probs: Any, q_hat: float) -> list[list[int]]:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError("probs must be finite 2-D array")
    sets = [(np.flatnonzero(row >= 1.0 - float(q_hat)).astype(int).tolist()) for row in p]
    return sets


def conformal_summary(probs: Any, labels: Any, q_hat: float, normal_index: int = 2) -> dict[str, Any]:
    p = np.asarray(probs, dtype=np.float64)
    y = _as_labels(labels, len(p), p.shape[1])
    sets = conformal_sets(p, q_hat)
    covered = np.asarray([int(int(label) in s) for label, s in zip(y, sets)], dtype=float)
    uncertain = np.asarray([(not s) or (normal_index in s and any(i != normal_index for i in s)) for s in sets])
    sizes = {str(size): int(sum(len(s) == size for s in sets)) for size in sorted({len(s) for s in sets})}
    return {"q_hat": float(q_hat), "coverage": float(covered.mean()), "n": len(y), "set_size_histogram": sizes, "uncertain_rate": float(uncertain.mean()), "sets": sets, "uncertain": uncertain.tolist()}


def three_way_decision(p_defect: Any, c_escape: float, c_false_reject: float, c_review: float) -> np.ndarray:
    """Minimum expected cost action for each calibrated defect probability."""
    p = np.asarray(p_defect, dtype=np.float64).reshape(-1)
    if min(c_escape, c_false_reject, c_review) < 0:
        raise ValueError("decision costs must be non-negative")
    costs = np.stack([p * c_escape, (1.0 - p) * c_false_reject, np.full(len(p), c_review)], axis=1)
    return np.asarray([("ACCEPT", "REJECT", "REVIEW")[int(i)] for i in costs.argmin(axis=1)])


def realized_cost(actions: Sequence[str], labels: Any, c_escape: float, c_false_reject: float, c_review: float) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    a = np.asarray(actions, dtype=object).reshape(-1)
    if len(y) != len(a):
        raise ValueError("actions and labels have different lengths")
    cost = np.where((a == "ACCEPT") & (y != 2), c_escape, 0.0)
    cost += np.where((a == "REJECT") & (y == 2), c_false_reject, 0.0)
    cost += np.where(a == "REVIEW", c_review, 0.0)
    return float(np.mean(cost))


def error_rates(actions: Sequence[str], labels: Any, normal_index: int = 2) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    a = np.asarray(actions, dtype=object).reshape(-1)
    normal = y == normal_index
    defect = ~normal
    return {
        "frr": float(np.sum((a == "REJECT") & normal) / max(1, int(normal.sum()))),
        "far": float(np.sum((a == "ACCEPT") & defect) / max(1, int(defect.sum()))),
        "review_rate": float(np.mean(a == "REVIEW")),
    }


def cost_curve(p_defect: Any, labels: Any, c_escape: float, c_false_reject: float, c_review: float, points: int = 101) -> tuple[list[dict[str, float]], float]:
    p = np.asarray(p_defect, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    thresholds = np.linspace(0.0, 1.0, int(points))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        # Threshold curve is the operational reject threshold.  REVIEW is
        # represented by the fixed review-cost comparator in the main rule.
        actions = np.where(p > threshold, "REJECT", "ACCEPT")
        cost = realized_cost(actions, y, c_escape, c_false_reject, c_review)
        rows.append({"threshold": float(threshold), "expected_cost_per_unit": float(cost)})
    best = min(rows, key=lambda row: row["expected_cost_per_unit"])
    return rows, float(best["threshold"])


def risk_coverage(probs: Any, labels: Any) -> dict[str, Any]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    confidence = p.max(axis=1)
    order = np.argsort(-confidence, kind="mergesort")
    err = (p.argmax(axis=1) != y).astype(float)[order]
    coverage = np.arange(1, len(y) + 1, dtype=float) / len(y)
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    # AURC is the rectangle integral over the retained sample sequence.
    aurc = float(np.sum(risk / len(y)))
    return {"coverage": coverage.tolist(), "risk": risk.tolist(), "aurc": aurc}


def cosine_knn_scores(embeddings: Any, bank: Any, k: int = 5) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    b = np.asarray(bank, dtype=np.float32)
    if x.ndim != 2 or b.ndim != 2 or x.shape[1] != b.shape[1] or not len(b):
        raise ValueError("embedding and bank shapes are incompatible")
    if k < 1:
        raise ValueError("k must be positive")
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), EPS)
    # Chunking keeps peak memory bounded on a CPU demo machine.
    out = np.empty(len(xn), dtype=np.float32)
    kk = min(int(k), len(bn))
    for start in range(0, len(xn), 1024):
        sim = xn[start:start + 1024] @ bn.T
        nearest = np.partition(sim, sim.shape[1] - kk, axis=1)[:, -kk:]
        out[start:start + len(nearest)] = 1.0 - nearest.mean(axis=1)
    return out


def deterministic_coreset(bank: Any, limit: int = 8000) -> np.ndarray:
    arr = np.asarray(bank)
    if arr.ndim != 2 or not len(arr):
        raise ValueError("bank must be a non-empty 2-D array")
    if len(arr) <= limit:
        return arr.astype(np.float16, copy=False)
    # A deterministic farthest-point coreset is more useful than a raw prefix
    # while staying cheap for the <=~10k banks produced by the trainer.
    x = arr.astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    chosen = np.empty(int(limit), dtype=np.int64)
    chosen[0] = 0
    best = 1.0 - x @ x[0]
    for i in range(1, int(limit)):
        idx = int(np.argmax(best))
        chosen[i] = idx
        best = np.minimum(best, 1.0 - x @ x[idx])
    return arr[chosen].astype(np.float16, copy=False)


def group_split(groups: Sequence[Any], test_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split rows by group, preserving no group overlap."""
    g = np.asarray([str(x) for x in groups])
    unique = np.asarray(sorted(set(g.tolist())))
    if len(unique) < 2:
        idx = np.arange(len(g))
        return idx, idx
    n_test = max(1, min(len(unique) - 1, int(round(len(unique) * test_fraction))))
    # Hash-independent lexical split makes synthetic and real runs reproducible.
    test_groups = set(unique[:: max(1, len(unique) // n_test)][:n_test].tolist())
    test = np.asarray([item in test_groups for item in g], dtype=bool)
    if test.all() or (~test).all():
        test[: max(1, len(test) // 2)] = False
        test[max(1, len(test) // 2):] = True
    return np.flatnonzero(~test), np.flatnonzero(test)


__all__ = [
    "TemperatureFit", "softmax", "nll", "fit_temperature", "calibration_metrics",
    "conformal_quantile", "conformal_sets", "conformal_summary", "three_way_decision",
    "realized_cost", "error_rates", "cost_curve", "risk_coverage", "cosine_knn_scores",
    "deterministic_coreset", "group_split",
]
