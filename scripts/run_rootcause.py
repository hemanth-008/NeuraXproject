"""Run the offline root-cause engine and optional planted-effect recovery.

The normal analysis path never opens ``sim_truth.json``.  With ``--recovery``
the evidence cards are written first, then the truth file is read solely to
compare planted effects with the cards produced by the engine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rootcause.engine import (  # noqa: E402
    AnalysisConfig,
    _assert_no_holdout,
    analyze,
    evidence_cards,
    load_input,
    meta,
)


def _read_optional(root: Path, relative: str, fixture_relative: str) -> tuple[pd.DataFrame | None, str | None]:
    path = root / relative
    if not path.exists():
        path = root / fixture_relative
    if not path.exists():
        return None, None
    frame = pd.read_parquet(path)
    _assert_no_holdout(frame, str(path))
    return frame, str(path)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _normalise_effect(effect: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": str(effect.get("family", "")),
        "parameter": str(effect.get("parameter", "")),
        "direction": str(effect.get("direction", effect.get("form", "unknown"))),
        "form": effect.get("form"),
        "size": effect.get("size"),
        "station": effect.get("station"),
    }


def recover(root: Path, cards_payload: dict[str, Any]) -> dict[str, Any]:
    """Compare cards to truth, called only after cards have been persisted."""
    truth_path = root / "artifacts" / "sim_truth.json"
    if not truth_path.exists():
        truth_path = root / "tests" / "fixtures" / "sim_truth.json"
    # This is intentionally the only sim_truth read in this task.
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    raw_effects = truth.get("planted_effects", truth.get("effects", []))
    all_effects = [_normalise_effect(item) for item in raw_effects]
    no_link_effects = [effect for effect in all_effects if str(effect.get("form", "")).lower() in {"no_link_control", "no-link-control"}]
    planted = [effect for effect in all_effects if effect not in no_link_effects]
    cards = cards_payload.get("cards", [])
    found: list[dict[str, Any]] = []
    for card in cards:
        if str(card.get("id", "")).endswith("NO-LINK"):
            continue
        variables = [str(value) for value in card.get("variables", [])]
        if len(variables) < 2:
            continue
        found.append({
            "family": str(card.get("family", "")),
            "parameter": variables[0],
            "direction": "positive" if float(card.get("effect_size", 0.0) or 0.0) >= 0 else "negative",
            "effect_size": float(card.get("effect_size", 0.0) or 0.0),
            "p_adj": float(card.get("p_adj", 1.0) or 1.0),
        })

    def matches(effect: dict[str, Any], candidate: dict[str, Any]) -> bool:
        if effect["family"] != candidate["family"] or effect["parameter"] != candidate["parameter"]:
            return False
        form = str(effect.get("form") or "").lower()
        if form in {"positive", "increasing", "monotone"} or "monotone" in form or "threshold" in form:
            return candidate["direction"] == "positive"
        if form in {"negative", "decreasing"}:
            return candidate["direction"] == "negative"
        return True

    matched: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    used: set[int] = set()
    for effect in planted:
        hit = next((candidate for index, candidate in enumerate(found) if index not in used and matches(effect, candidate)), None)
        if hit is None:
            misses.append(effect)
        else:
            used.add(found.index(hit))
            matched.append({"planted": effect, "found": hit})
    false_positives = [candidate for index, candidate in enumerate(found) if index not in used]
    no_link_family = str(no_link_effects[0]["family"]) if no_link_effects else "crack"
    control_clean = not any(str(card.get("family")) == no_link_family and not str(card.get("id", "")).endswith("NO-LINK") for card in cards)
    return {
        "meta": meta(str(truth_path), tag="ADVISORY/SIMULATED"),
        "planted_effects": planted,
        "found_effects": found,
        "matches": matched,
        "false_positives": false_positives,
        "misses": misses,
        "no_link_control_clean": bool(control_clean),
    }


def run(root: Path, recovery: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    frame, source = load_input(root)
    # Load optional inputs to enforce the same holdout boundary and to make
    # their provenance visible in the run log.  Batch metrics remain the
    # canonical table because they already contain the predicted rate fields.
    unit_map, unit_source = _read_optional(root, "artifacts/sim_unit_map.parquet", "tests/fixtures/sim_unit_map.parquet")
    predictions, prediction_source = _read_optional(root, "artifacts/predictions.parquet", "tests/fixtures/predictions.parquet")
    result = analyze(frame, source=source, config=AnalysisConfig())
    cards = evidence_cards(result, source=source)
    evidence_path = root / "artifacts" / "evidence_cards.json"
    write_json(evidence_path, cards)

    recovery_payload = None
    recovery_path = root / "artifacts" / "rootcause_recovery.json"
    if recovery:
        recovery_payload = recover(root, cards)
        write_json(recovery_path, recovery_payload)
    elapsed = time.perf_counter() - started
    summary = {
        "source": source,
        "unit_map": unit_source,
        "predictions": prediction_source,
        "rows": len(frame),
        "cards": len(cards.get("cards", [])),
        "evidence_cards": str(evidence_path),
        "recovery": str(recovery_path) if recovery_payload is not None else None,
        "runtime_seconds": elapsed,
    }
    if recovery_payload is not None:
        summary.update({
            "planted_count": len(recovery_payload["planted_effects"]),
            "found_count": len(recovery_payload["found_effects"]),
            "match_count": len(recovery_payload["matches"]),
            "false_positive_count": len(recovery_payload["false_positives"]),
            "miss_count": len(recovery_payload["misses"]),
            "no_link_control_clean": recovery_payload["no_link_control_clean"],
        })
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    parser.add_argument("--recovery", action="store_true", help="run post-card planted-effect recovery")
    args = parser.parse_args(argv)
    run(Path(args.root).resolve(), recovery=args.recovery)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
