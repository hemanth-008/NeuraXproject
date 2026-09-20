"""Offline accounting-identity profit calculations for QualityTwin."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Economics:
    """Economic inputs loaded from config.yaml and explicitly marked assumed."""

    price: float
    unit_cost: float
    scrap_cost: float
    rework_cost: float
    c_escape: float
    downtime_cost_per_hour: float
    c_false_reject: float = 0.0
    c_review: float = 0.0
    assumption_ids: tuple[str, ...] = ()


def accounting_margin(
    good_units_per_hour: float,
    economics: Economics,
    *,
    scrap_units_per_hour: float = 0.0,
    rework_units_per_hour: float = 0.0,
    false_accept_rate: float = 0.0,
    downtime_hours: float = 0.0,
) -> float:
    """Return the M5 accounting identity margin per hour.

    The identity is intentionally explicit: good-unit contribution less scrap,
    rework, escape and downtime costs. Missing operational quantities are passed
    as zero by callers and recorded as unavailable rather than invented.
    """

    contribution = good_units_per_hour * (economics.price - economics.unit_cost)
    scrap = scrap_units_per_hour * economics.scrap_cost
    rework = rework_units_per_hour * economics.rework_cost
    escape = good_units_per_hour * false_accept_rate * economics.c_escape
    downtime = downtime_hours * economics.downtime_cost_per_hour
    return float(contribution - scrap - rework - escape - downtime)


def margin_interval(
    throughput_low: float,
    throughput_high: float,
    economics: Economics,
    *,
    rework_rate: float = 0.0,
    scrap_rate: float = 0.0,
    false_accept_rate: float = 0.0,
    downtime_hours: float = 0.0,
) -> tuple[float, float]:
    """Push a throughput 5--95% interval through the accounting identity."""

    low = accounting_margin(
        throughput_low,
        economics,
        scrap_units_per_hour=throughput_low * scrap_rate,
        rework_units_per_hour=throughput_low * rework_rate,
        false_accept_rate=false_accept_rate,
        downtime_hours=downtime_hours,
    )
    high = accounting_margin(
        throughput_high,
        economics,
        scrap_units_per_hour=throughput_high * scrap_rate,
        rework_units_per_hour=throughput_high * rework_rate,
        false_accept_rate=false_accept_rate,
        downtime_hours=downtime_hours,
    )
    return (float(min(low, high)), float(max(low, high)))


def economics_from_config(config: dict) -> Economics:
    """Read every economic parameter from the config economics block."""

    block = config.get("economics", {})

    def value(name: str, default: float = 0.0) -> float:
        raw = block.get(name, default)
        if isinstance(raw, dict):
            raw = raw.get("value", default)
        return float(raw)

    def assumption(name: str) -> str:
        raw = block.get(name, {})
        if isinstance(raw, dict):
            return str(raw.get("assumption_id", f"economics.{name}"))
        return f"economics.{name}"

    names = (
        "price",
        "unit_cost",
        "scrap_cost",
        "rework_cost",
        "c_escape",
        "downtime_cost_per_hour",
        "c_false_reject",
        "c_review",
    )
    return Economics(
        price=value("price"),
        unit_cost=value("unit_cost"),
        scrap_cost=value("scrap_cost"),
        rework_cost=value("rework_cost"),
        c_escape=value("c_escape"),
        downtime_cost_per_hour=value("downtime_cost_per_hour"),
        c_false_reject=value("c_false_reject"),
        c_review=value("c_review"),
        assumption_ids=tuple(assumption(name) for name in names),
    )
