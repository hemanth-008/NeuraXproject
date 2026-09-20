"""Load the repository configuration and expose assumption-labelled economics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REQUIRED_ECONOMICS = ("c_escape", "c_false_reject", "c_review")
_OPTIONAL_ECONOMICS = (
    "price",
    "unit_cost",
    "scrap_cost",
    "rework_cost",
    "downtime_cost_per_hour",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``config.yaml`` without applying hidden defaults.

    The explicit defaults below are only used when a caller points at a config
    file that omits the optional economics block; the repository config keeps
    all economics visible and labelled as assumptions.
    """

    config_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return loaded


def _labelled_value(name: str, raw: Any) -> dict[str, Any]:
    """Normalize a scalar or mapping into a value carrying provenance."""

    if isinstance(raw, dict):
        if "value" not in raw:
            raise ValueError(f"economics.{name} must contain a value")
        value = raw["value"]
        assumption_id = raw.get("assumption_id", f"economics.{name}")
        tag = raw.get("tag", "ASSUMED")
    else:
        value = raw
        assumption_id = f"economics.{name}"
        tag = "ASSUMED"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"economics.{name}.value must be numeric")
    return {"value": float(value), "assumption_id": str(assumption_id), "tag": str(tag)}


def economics(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return economic inputs as ``{value, assumption_id, tag}`` records.

    The project has no organizer-supplied economics.  Keeping the assumption
    id beside every value prevents simulated/advisory numbers from becoming
    indistinguishable from observed data in downstream artifacts.
    """

    cfg = config if config is not None else load_config()
    raw_economics = cfg.get("economics") or {}
    if not isinstance(raw_economics, dict):
        raise ValueError("config.economics must be a mapping")
    missing = [name for name in _REQUIRED_ECONOMICS if name not in raw_economics]
    if missing:
        raise KeyError(f"missing required economics keys: {', '.join(missing)}")
    names = _REQUIRED_ECONOMICS + tuple(name for name in _OPTIONAL_ECONOMICS if name in raw_economics)
    return {name: _labelled_value(name, raw_economics[name]) for name in names}

