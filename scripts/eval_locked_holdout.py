"""Evaluate the locked holdout exactly once under an explicit human gate.

The real path is intentionally narrow: this is the only script that selects
rows whose split is the locked holdout.  It uses ``InferenceEngine`` directly,
so temperature, conformal, novelty and cost decisions are the shipped infer
path rather than a separately tuned evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASSES = ("crack", "hole", "normal", "rust", "scratch")
DEFECTS = set(CLASSES) - {"normal"}
LOCKED_SPLIT_NAMES = {"holdout", "locked", "locked_holdout", "locked-holdout"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meta(source: str) -> dict[str, Any]:
    return {
        "generated_at": _now(),
        "source": source,
        "assumption_ids": ["economics.c_escape", "economics.c_false_reject", "economics.c_review"],
        "tag": "LOCKED HOLDOUT EVALUATION",
    }


def _load_counter(path: Path) -> tuple[int, dict[str, Any]]:
    if not path.exists():
        return 0, {"meta": _meta(str(path)), "count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = int(payload.get("count", 0))
    if count < 0:
        raise ValueError("holdout run counter cannot be negative")
    return count, payload


def _reserve_run(path: Path) -> int:
    count, payload = _load_counter(path)
    next_count = count + 1
    payload["count"] = next_count
    payload["meta"] = {**payload.get("meta", {}), **_meta(str(path))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return next_count


def _guard(confirm: bool, override: bool, ledger_path: Path) -> int:
    if not confirm:
        raise PermissionError("refusing locked-holdout evaluation: pass --confirm")
    count, _ = _load_counter(ledger_path)
    if count >= 2 and not override:
        raise PermissionError(f"refusing locked-holdout evaluation: counter is already {count}; pass --override")
    return count


def _costs(repo_root: Path) -> dict[str, float]:
    values = {"c_escape": 50.0, "c_false_reject": 5.0, "c_review": 1.0}
    config = repo_root / "config.yaml"
    if not config.exists():
        return values
    try:
        import yaml

        loaded = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        economics = loaded.get("economics", {}) if isinstance(loaded, dict) else {}
        for key in values:
            raw = economics.get(key)
            raw = raw.get("value") if isinstance(raw, dict) else raw
            values[key] = float(raw)
    except (ImportError, TypeError, ValueError, OSError):
        pass
    return values


def _ece(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if not len(confidences):
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (confidences >= edges[index]) & (confidences <= edges[index + 1] if index == bins - 1 else confidences < edges[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(confidences[mask].mean()) - float(correct[mask].mean()))
    return total


def summarize(records: list[dict[str, Any]], costs: dict[str, float]) -> dict[str, Any]:
    y_true = [str(row["true_label"]) for row in records]
    y_pred = [str(row["family"]) for row in records]
    true_indices = np.asarray([CLASSES.index(value) for value in y_true], dtype=int)
    pred_indices = np.asarray([CLASSES.index(value) if value in CLASSES else 2 for value in y_pred], dtype=int)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for actual, predicted in zip(true_indices, pred_indices):
        matrix[actual, predicted] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index, :].sum())}

    actions = [str(row.get("decision", "REVIEW")).upper() for row in records]
    flags = [str(row.get("flag", "NONE")).upper() for row in records]
    normal = np.asarray([value == "normal" for value in y_true], dtype=bool)
    rejected = np.asarray([value == "REJECT" for value in actions], dtype=bool)
    accepted = np.asarray([value == "ACCEPT" for value in actions], dtype=bool)
    defect = ~normal
    frr = float(np.mean(rejected[normal])) if normal.any() else 0.0
    far = float(np.mean(accepted[defect])) if defect.any() else 0.0
    expected_cost = 0.0
    for actual, action in zip(y_true, actions):
        if action == "REVIEW":
            expected_cost += costs["c_review"]
        elif actual == "normal" and action == "REJECT":
            expected_cost += costs["c_false_reject"]
        elif actual != "normal" and action == "ACCEPT":
            expected_cost += costs["c_escape"]
    expected_cost /= max(1, len(records))
    confidences = np.asarray([float(row.get("confidence", 0.0)) for row in records], dtype=float)
    correct = true_indices == pred_indices
    conformal_hits = [value in set(row.get("conformal_set", [])) for value, row in zip(y_true, records)]
    return {
        "n": int(len(records)),
        "accuracy": float(np.mean(correct)) if len(records) else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "frr": frr,
        "far": far,
        "expected_cost": float(expected_cost),
        "ece": _ece(confidences, correct),
        "conformal_coverage": float(np.mean(conformal_hits)) if conformal_hits else 0.0,
        "uncertain_rate": float(np.mean(np.asarray(flags) == "UNCERTAIN")) if flags else 0.0,
        "novel_rate": float(np.mean(np.asarray(flags) == "NOVEL")) if flags else 0.0,
        "review_rate": float(np.mean(np.asarray(actions) == "REVIEW")) if actions else 0.0,
    }


def _dry_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, true_label in enumerate(CLASSES * 2):
        predicted = true_label if index % 5 else "normal"
        records.append({
            "true_label": true_label,
            "family": predicted,
            "confidence": 0.9 if predicted == true_label else 0.6,
            "decision": "ACCEPT" if predicted == "normal" else "REJECT",
            "flag": "NONE",
            "conformal_set": [predicted],
        })
    return records


def _locked_rows(splits_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(splits_path)
    if "split" not in frame.columns or "file" not in frame.columns or "cls" not in frame.columns:
        raise ValueError("splits CSV must contain file, cls and split columns")
    locked = frame[frame["split"].astype(str).str.lower().isin(LOCKED_SPLIT_NAMES)].copy()
    if locked.empty:
        raise ValueError("no locked-holdout rows found")
    return locked


def evaluate_real(*, splits_path: Path, data_root: Path, artifacts: Path, repo_root: Path) -> dict[str, Any]:
    rows = _locked_rows(splits_path)
    from src.vision.infer import InferenceEngine

    engine = InferenceEngine(artifacts, repo_root=repo_root)
    records: list[dict[str, Any]] = []
    for item in rows.to_dict("records"):
        image = Path(str(item["file"]))
        if not image.is_absolute():
            image = data_root / image
        result = engine.predict_file(image)
        records.append({"true_label": str(item["cls"]), **result})
    return summarize(records, _costs(repo_root))


def evaluate(*, confirm: bool, override: bool, dry_run: bool, splits_path: Path, data_root: Path, artifacts: Path, repo_root: Path) -> dict[str, Any]:
    ledger = artifacts / "holdout_runs.json"
    current = _guard(confirm, override, ledger)
    print("!!! WARNING: LOCKED HOLDOUT EVALUATION — this consumes one permitted run. !!!")
    print("!!! Do not use this result for tuning, threshold changes, or temperature refitting. !!!")
    result = summarize(_dry_records(), _costs(repo_root)) if dry_run else evaluate_real(splits_path=splits_path, data_root=data_root, artifacts=artifacts, repo_root=repo_root)
    run_number = _reserve_run(ledger)
    result["meta"] = _meta("dry-run fake split" if dry_run else str(splits_path))
    result["meta"]["run_number"] = run_number
    result["meta"]["previous_run_count"] = current
    result["meta"]["locked_split"] = True
    output = artifacts / "holdout_result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("n", "accuracy", "macro_f1", "frr", "far", "expected_cost", "ece", "conformal_coverage", "uncertain_rate", "novel_rate", "review_rate")}, sort_keys=True))
    print(f"wrote={output} run_number={run_number}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="explicitly authorize consuming one locked-holdout run")
    parser.add_argument("--override", action="store_true", help="allow a run when the ledger count is already >= 2")
    parser.add_argument("--dry-run", action="store_true", help="use a fake tiny split; never reads the real split")
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/images/train"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        evaluate(confirm=args.confirm, override=args.override, dry_run=args.dry_run, splits_path=args.splits, data_root=args.data_root, artifacts=args.artifacts_dir, repo_root=repo_root)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
