from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rootcause.engine import analyze, evidence_cards


def _synthetic_batches(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    temperature = np.linspace(0.0, 1.0, n)
    humidity = rng.normal(0.5, 0.08, n)
    effect = 0.04 + 0.45 * temperature + rng.normal(0.0, 0.01, n)
    return pd.DataFrame(
        {
            "batch_id": [f"B{i:03d}" for i in range(n)],
            "ts": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC").astype(str),
            "variant": ["A", "B"] * (n // 2),
            "n_units": [100] * n,
            "defect_rate_crack": [0.10] * n,
            "defect_rate_hole": np.clip(effect, 0.0, 1.0),
            "defect_rate_rust": [0.06] * n,
            "defect_rate_scratch": [0.08] * n,
            "fpy": 1.0 - np.clip(effect, 0.0, 1.0),
            "temperature": temperature,
            "coolant_humidity": humidity,
            "drift_flags": ["[]"] * n,
        }
    )


def test_planted_effect_is_recovered_and_null_family_is_clean() -> None:
    result = analyze(_synthetic_batches())
    payload = evidence_cards(result, source="synthetic")
    hole = [card for card in payload["cards"] if card["family"] == "hole"]
    crack = [card for card in payload["cards"] if card["family"] == "crack"]
    assert any(card["variables"] and card["variables"][0] == "temperature" for card in hole)
    assert any(card["id"] == "RC-crack-NO-LINK" for card in crack)
    assert all(card["caveat"].startswith("association, not proven causation") for card in payload["cards"])


def test_locked_holdout_is_rejected() -> None:
    frame = _synthetic_batches(12)
    frame["split"] = "dev"
    frame.loc[0, "split"] = "holdout"
    with pytest.raises(ValueError, match="locked holdout"):
        analyze(frame)

