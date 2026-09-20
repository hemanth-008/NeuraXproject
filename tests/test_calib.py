"""Unit tests for calibration primitives; no project data is loaded here."""

from __future__ import annotations

import numpy as np

from src.calib.core import (
    conformal_quantile,
    conformal_sets,
    fit_temperature,
    three_way_decision,
)


def test_temperature_recovers_known_temperature() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(size=(5000, 5))
    labels = np.argmax(base, axis=1)
    # Labels are drawn from a deliberately softened model.  Fitting the
    # original logits therefore recovers the generating temperature.
    generating_t = 2.25
    probs = np.exp(base / generating_t - np.max(base / generating_t, axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.asarray([rng.choice(5, p=row) for row in probs], dtype=np.int64)
    result = fit_temperature(base, labels)
    assert abs(result.temperature - generating_t) < 0.15


def test_split_conformal_coverage() -> None:
    rng = np.random.default_rng(12)
    probs = rng.dirichlet(np.full(5, 0.3), size=3000)
    labels = np.asarray([rng.choice(5, p=row) for row in probs], dtype=np.int64)
    calibration = 1.0 - probs[np.arange(1500), labels[:1500]]
    q_hat = conformal_quantile(calibration, alpha=0.10)
    sets = conformal_sets(probs[1500:], q_hat)
    coverage = np.mean([labels[i + 1500] in s for i, s in enumerate(sets)])
    assert coverage >= 0.88


def test_three_way_decision_matches_master_plan() -> None:
    p = np.asarray([0.0, 0.05, 0.10, 0.50, 0.95, 1.0])
    actions = three_way_decision(p, c_escape=50.0, c_false_reject=5.0, c_review=1.0)
    # ACCEPT = 50p, REJECT = 5(1-p), REVIEW = 1.
    assert actions.tolist() == ["ACCEPT", "REVIEW", "REVIEW", "REVIEW", "REJECT", "REJECT"]
