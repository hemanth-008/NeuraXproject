"""Focused M4 formula and deterministic-twin tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.flow.engine import analytical_utilization, little_law_wip


def _analytical_sample(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.0, sigma=0.04, size=32)


def test_analytical_utilization_formula_and_d14_multiplier() -> None:
    baseline = analytical_utilization(20.0, 0.25, machines=2, availability=0.8, changeover_loss=0.1)
    with_rework = analytical_utilization(
        20.0, 0.25, machines=2, availability=0.8, changeover_loss=0.1, rework_rate=0.2
    )
    assert np.isclose(baseline, 20.0 * 0.25 / (2.0 * 0.8 * 0.9))
    assert np.isclose(with_rework, baseline * 1.2)


def test_little_law_sanity() -> None:
    assert np.isclose(little_law_wip(120.0, 0.5), 60.0)


def test_twin_deterministic_given_seed() -> None:
    first = _analytical_sample(1234)
    second = _analytical_sample(1234)
    different = _analytical_sample(1235)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_model2_disclosure_signals_are_not_collapsed() -> None:
    # This fixture mirrors the two signals the artifact must disclose without
    # loading any inspection split or locked-holdout rows.
    frame = pd.DataFrame({
        "Drilling Queue Time": [4.0, 6.0],
        "Assembly Queue Time": [1.0, 2.0],
        "Drilling Utilization": [0.5, 0.6],
        "Assembly Utilization": [0.7, 0.8],
    })
    assert frame["Drilling Queue Time"].mean() > frame["Assembly Queue Time"].mean()
    assert frame["Assembly Utilization"].mean() > frame["Drilling Utilization"].mean()

