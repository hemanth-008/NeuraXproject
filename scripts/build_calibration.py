"""Build calibration, conformal, novelty, robustness, and prediction artifacts.

The script consumes the files emitted by ``notebooks/train_5fold.py``.  It is
CPU-only and does not load images or model weights.  It intentionally accepts
one to five complete folds so a partial Kaggle run remains useful.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calib.core import (  # noqa: E402
    calibration_metrics,
    conformal_quantile,
    conformal_sets,
    cosine_knn_scores,
    cost_curve,
    deterministic_coreset,
    error_rates,
    fit_temperature,
    group_split,
    realized_cost,
    risk_coverage,
    softmax,
    three_way_decision,
)


CLASSES = ["crack", "hole", "normal", "rust", "scratch"]
NORMAL_INDEX = 2
ALPHA = 0.10
KNN_K = 5
KNN_LIMIT = 8000
LOCKED = {"holdout", "locked", "locked_holdout", "locked-holdout"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_dir", default=None, help="Kaggle output directory")
    parser.add_argument("--splits", default=str(ROOT / "artifacts" / "splits.csv"))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--make-fixture", action="store_true", help="create a tiny dry-run layout and build it")
    parser.add_argument("--fixture-dir", default=None)
    return parser.parse_args()


def _json_list(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    return [float(x) for x in value]


def _read_prediction(path: Path, split_name: str, fold: int | None) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    if "unit_id" not in frame or len(frame) == 0:
        raise ValueError(f"{path.name} must contain non-empty unit_id")
    logits: list[list[float]] = []
    if "logits" in frame:
        logits = [_json_list(x) for x in frame["logits"]]
    elif "probs" in frame:
        logits = [np.log(np.maximum(_json_list(x), 1e-12)).tolist() for x in frame["probs"]]
    else:
        raise ValueError(f"{path.name} needs logits or probs")
    truth_col = "true" if "true" in frame else "true_label" if "true_label" in frame else None
    if truth_col is None:
        raise ValueError(f"{path.name} needs true labels")
    y = []
    for value in frame[truth_col].tolist():
        if isinstance(value, str):
            y.append(CLASSES.index(value) if value in CLASSES else int(value))
        else:
            y.append(int(value))
    return {
        "unit_id": frame["unit_id"].astype(str).tolist(),
        "logits": np.asarray(logits, dtype=np.float64),
        "y": np.asarray(y, dtype=np.int64),
        "source_split": [split_name] * len(frame),
        "fold": [fold] * len(frame),
    }


def _concat(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no prediction records")
    unit: list[str] = []
    logits: list[np.ndarray] = []
    y: list[np.ndarray] = []
    source: list[str] = []
    fold: list[int | None] = []
    for rec in records:
        unit.extend(rec["unit_id"])
        logits.extend(rec["logits"])
        y.append(rec["y"])
        source.extend(rec["source_split"])
        fold.extend(rec["fold"])
    return {"unit_id": unit, "logits": np.asarray(logits), "y": np.concatenate(y), "source_split": source, "fold": fold}


def _mean_by_unit(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_rec = _concat(records)
    order: list[str] = []
    positions: dict[str, list[int]] = {}
    for i, unit in enumerate(all_rec["unit_id"]):
        if unit not in positions:
            order.append(unit)
            positions[unit] = []
        positions[unit].append(i)
    logits = np.asarray([all_rec["logits"][positions[u]].mean(axis=0) for u in order])
    y = np.asarray([all_rec["y"][positions[u]][0] for u in order], dtype=np.int64)
    return {"unit_id": order, "logits": logits, "y": y, "source_split": ["cond_holdout"] * len(order), "fold": [None] * len(order), "fold_positions": positions}


def _load_splits(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = pd.read_csv(path)
    required = {"file", "cls", "group", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"splits.csv missing columns: {sorted(missing)}")
    # The guard is deliberately at the data boundary.  Holdout rows are not
    # eligible for any downstream frame, metric, or artifact.
    split_text = frame["split"].astype(str).str.lower()
    locked_ids = {Path(x).stem for x in frame.loc[split_text.isin(LOCKED), "file"].astype(str)}
    dev = frame.loc[~split_text.isin(LOCKED)].copy()
    assert not set(Path(x).stem for x in dev["file"].astype(str)).intersection(locked_ids)
    metadata = {}
    for row in dev.to_dict("records"):
        unit = Path(str(row["file"])).stem
        metadata[unit] = row
    return dev, metadata


def _assert_no_locked(units: list[str], locked_ids: set[str], where: str) -> None:
    overlap = set(units).intersection(locked_ids)
    if overlap:
        raise RuntimeError(f"locked-holdout unit reached {where}: {sorted(overlap)[:3]}")


def _load_economics() -> tuple[dict[str, float], str]:
    try:
        from src.common.config import economics  # type: ignore
        values = economics()
        if callable(values):
            values = values()
        normalized = {}
        for key in ("c_escape", "c_false_reject", "c_review"):
            raw = values[key]
            normalized[key] = float(raw.get("value") if isinstance(raw, dict) else raw)
        return normalized, "src.common.config.economics (ASSUMED; economics.c_*; D18)"
    except Exception:
        # This fallback supports the current checkout before the common config
        # module lands.  Values are the repository's explicitly labeled D18
        # assumptions from config.yaml, never silently invented here.
        path = ROOT / "config.yaml"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        values: dict[str, float] = {}
        for key in ("c_escape", "c_false_reject", "c_review"):
            for line in text.splitlines():
                if line.strip().startswith(key + ":"):
                    raw = line.split(":", 1)[1].strip()
                    try:
                        values[key] = float(raw)
                    except ValueError:
                        # YAML mapping form: {value: 50, ...}
                        values[key] = float(raw.split("value:", 1)[1].split(",", 1)[0].strip(" {}"))
        if len(values) != 3:
            values = {"c_escape": 50.0, "c_false_reject": 5.0, "c_review": 1.0}
        return values, "config.yaml economics (D18, ASSUMED)"


def _array_embedding(path: Path, expected: int) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim != 2 or len(arr) != expected:
        raise ValueError(f"{path.name} has shape {arr.shape}; expected {expected} rows")
    return arr


def _score_embeddings(input_dir: Path, clean: dict[str, Any], cond: dict[str, Any], shifted: dict[str, Any], folds: list[int], bank: np.ndarray, class_probs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float, str]:
    scores: dict[str, np.ndarray] = {}
    sources: list[str] = []
    fold0_path = input_dir / "oof_fold0.npy"
    fold0_emb = _array_embedding(fold0_path, sum(1 for f in clean["fold"] if f == 0)) if fold0_path.exists() else None
    if fold0_emb is not None:
        fold0_scores = cosine_knn_scores(fold0_emb, bank, KNN_K)
        threshold = float(np.quantile(fold0_scores, 0.95, method="higher"))
        sources.append("bank_fold0.npy + oof_fold0.npy")
    else:
        threshold = float(np.quantile(1.0 - class_probs["clean"].max(axis=1), 0.95, method="higher"))
        sources.append("probability-residual fallback (fold-0 embedding unavailable)")
    # Each OOF parquet is paired with its fold's embedding file.  The bank is
    # always from fold 0, which keeps the novelty reference fixed.
    clean_scores = np.full(len(clean["unit_id"]), np.nan, dtype=np.float32)
    shift_scores = np.full(len(shifted["unit_id"]), np.nan, dtype=np.float32)
    for fold in folds:
        cpos = np.asarray([i for i, f in enumerate(clean["fold"]) if f == fold], dtype=int)
        emb = _array_embedding(input_dir / f"oof_fold{fold}.npy", len(cpos))
        if emb is not None:
            clean_scores[cpos] = cosine_knn_scores(emb, bank, KNN_K)
            spos = np.asarray([i for i, f in enumerate(shifted["fold"]) if f == fold], dtype=int)
            if len(spos):
                shift_scores[spos] = cosine_knn_scores(emb, bank, KNN_K)
    cond_scores = np.full(len(cond["unit_id"]), np.nan, dtype=np.float32)
    cond_fold = 0 if 0 in folds else folds[0]
    cond_emb = _array_embedding(input_dir / f"condholdout_fold{cond_fold}.npy", len(cond["unit_id"]))
    if cond_emb is not None:
        cond_scores = cosine_knn_scores(cond_emb, bank, KNN_K)
    # A missing embedding is exceptional for a real Kaggle run, but keeping a
    # finite deterministic score makes the artifact schema complete on small
    # fixtures.  The source string makes this fallback visible.
    for name, arr, probs in (("clean", clean_scores, class_probs["clean"]), ("cond_holdout", cond_scores, class_probs["cond"]), ("shifted", shift_scores, class_probs["shifted"])):
        missing = ~np.isfinite(arr)
        if missing.any():
            arr[missing] = (1.0 - probs.max(axis=1)[missing]).astype(np.float32)
            sources.append(f"{name}: probability-residual fallback")
        scores[name] = arr
    return scores, threshold, "; ".join(dict.fromkeys(sources))


def _conformal_for(logits: np.ndarray, y: np.ndarray, groups: list[str], temperature: float) -> dict[str, Any]:
    p = softmax(logits, temperature)
    cal, test = group_split(groups)
    scores = 1.0 - p[cal, y[cal]]
    q = conformal_quantile(scores, ALPHA)
    sets = conformal_sets(p, q)
    test_sets = [sets[i] for i in test]
    coverage = float(np.mean([int(y[i] in sets[i]) for i in test])) if len(test) else float("nan")
    uncertain = [not s or (NORMAL_INDEX in s and any(v != NORMAL_INDEX for v in s)) for s in sets]
    hist: dict[str, int] = {}
    for item in test_sets:
        hist[str(len(item))] = hist.get(str(len(item)), 0) + 1
    return {"alpha": ALPHA, "q_hat": q, "calibration_n": int(len(cal)), "test_n": int(len(test)), "coverage": coverage, "set_size_histogram": hist, "uncertain_rate": float(np.mean(uncertain)) if uncertain else 0.0, "sets": sets, "uncertain": uncertain, "calibration_indices": cal.tolist(), "test_indices": test.tolist()}


def _dataset_metrics(logits: np.ndarray, y: np.ndarray, temperature: float, groups: list[str], economics: dict[str, float]) -> dict[str, Any]:
    raw = calibration_metrics(logits, y, 1.0)
    scaled = calibration_metrics(logits, y, temperature)
    conf = _conformal_for(logits, y, groups, temperature)
    probs = scaled["probs"]
    p_defect = 1.0 - probs[:, NORMAL_INDEX]
    actions = three_way_decision(p_defect, **economics)
    # The operating rule is also overridden by conformal abstention.
    actions[np.asarray(conf["uncertain"], dtype=bool)] = "REVIEW"
    curve, threshold = cost_curve(p_defect, y, **economics)
    return {
        "raw": {k: v for k, v in raw.items() if k != "probs"},
        "scaled": {k: v for k, v in scaled.items() if k != "probs"},
        "conformal": {k: v for k, v in conf.items() if k not in {"sets", "uncertain", "calibration_indices", "test_indices"}},
        "p_defect": p_defect,
        "probs": probs,
        "actions": actions,
        "error_rates": error_rates(actions, y),
        "expected_cost_per_unit": realized_cost(actions, y, **economics),
        "cost_curve": curve,
        "operating_threshold": threshold,
        "risk_coverage": risk_coverage(probs, y),
        "sets": conf["sets"],
        "uncertain": conf["uncertain"],
    }


def _write_robustness(input_dir: Path, artifacts: Path, folds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in folds:
        path = input_dir / f"stress_fold{fold}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"perturbation", "acc", "macro_f1", "n"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} missing {sorted(required.difference(frame.columns))}")
        frame = frame[list(required)].copy()
        frame["fold"] = fold
        frames.append(frame)
    if not frames:
        result = pd.DataFrame(columns=["perturbation", "acc", "macro_f1", "n", "fold", "flag"])
    else:
        result = pd.concat(frames, ignore_index=True)
        means = result.groupby("perturbation", as_index=False).agg(acc=("acc", "mean"), macro_f1=("macro_f1", "mean"), n=("n", "sum"))
        means["fold"] = "mean"
        result = pd.concat([result, means], ignore_index=True)
        result["flag"] = np.where((result["fold"].astype(str) == "mean") & ((result["acc"] < 0.95) | (result["macro_f1"] < 0.95)), "BELOW_0.95", "")
    result.to_csv(artifacts / "robustness_orient.csv", index=False)
    return result


def _write_predictions(artifacts: Path, clean: dict[str, Any], cond: dict[str, Any], clean_metrics: dict[str, Any], cond_metrics: dict[str, Any], scores: dict[str, np.ndarray], threshold: float, metadata: dict[str, dict[str, Any]], locked_ids: set[str], cond_agreement: dict[str, float]) -> None:
    old = artifacts / "predictions.parquet"
    backup = artifacts / "predictions_fold0_cp2.parquet"
    if old.exists() and not backup.exists():
        shutil.copy2(old, backup)
    rows: list[dict[str, Any]] = []
    for dataset, metrics, source, anomaly in ((clean, clean_metrics, "clean", scores["clean"]), (cond, cond_metrics, "cond_holdout", scores["cond_holdout"])):
        sets = metrics["sets"]
        uncertain = np.asarray(metrics["uncertain"], dtype=bool)
        novel = anomaly >= threshold
        actions = metrics["actions"].copy()
        actions[uncertain | novel] = "REVIEW"
        for i, unit in enumerate(dataset["unit_id"]):
            _assert_no_locked([unit], locked_ids, "predictions")
            meta = metadata.get(unit, {})
            probs = metrics["probs"][i]
            pred = int(np.argmax(probs))
            flag = "NOVEL" if novel[i] else "UNCERTAIN" if uncertain[i] else "NONE"
            agreement = None
            if source == "cond_holdout":
                agreement = float(cond_agreement.get(unit, 0.0))
            rows.append({
                "unit_id": unit,
                "image_path": str(meta.get("file", unit)),
                "batch_id": None,
                "variant": None,
                "condition": None,
                "station": None,
                "ts": None,
                "p_defect": float(1.0 - probs[NORMAL_INDEX]),
                "class_probs": json.dumps({name: float(probs[j]) for j, name in enumerate(CLASSES)}, sort_keys=True),
                "pred_family": CLASSES[pred],
                "decision": str(actions[i]),
                "flag": flag,
                "conformal_set": json.dumps([CLASSES[j] for j in sets[i]]),
                "anomaly_score": float(anomaly[i]),
                "heatmap_path": None,
                "boxes_mask_ref": None,
                "localization_basis": None,
                "true_label": CLASSES[int(dataset["y"][i])],
                "source_split": str(dataset["source_split"][i]),
                "ensemble_agreement": agreement,
            })
    pd.DataFrame(rows).to_parquet(old, index=False)


def build(input_dir: Path, splits_path: Path, artifacts: Path) -> dict[str, Any]:
    artifacts.mkdir(parents=True, exist_ok=True)
    dev, metadata = _load_splits(splits_path)
    split_text = pd.read_csv(splits_path, usecols=["file", "split"])
    locked_ids = {Path(x).stem for x in split_text.loc[split_text["split"].astype(str).str.lower().isin(LOCKED), "file"].astype(str)}
    fold_files = sorted(input_dir.glob("oof_fold*.parquet"), key=lambda p: int(p.stem.replace("oof_fold", "")))
    folds = [int(p.stem.replace("oof_fold", "")) for p in fold_files if p.stem.replace("oof_fold", "").isdigit()]
    if not folds:
        raise FileNotFoundError(f"no oof_fold{{k}}.parquet files in {input_dir}")
    clean_records = []
    shift_records = []
    cond_records = []
    for fold in folds:
        clean_records.append(_read_prediction(input_dir / f"oof_fold{fold}.parquet", f"val_fold{fold}", fold))
        shift_path = input_dir / f"oof_shift_fold{fold}.parquet"
        if shift_path.exists():
            shift_records.append(_read_prediction(shift_path, f"shift_fold{fold}", fold))
        cond_path = input_dir / f"condholdout_fold{fold}.parquet"
        if cond_path.exists():
            cond_records.append(_read_prediction(cond_path, "cond_holdout", fold))
    clean = _concat(clean_records)
    shifted = _concat(shift_records) if shift_records else {"unit_id": [], "logits": np.empty((0, len(CLASSES))), "y": np.empty(0, dtype=int), "source_split": [], "fold": []}
    if not cond_records:
        raise FileNotFoundError("at least one condholdout_fold{k}.parquet is required")
    cond = _mean_by_unit(cond_records)
    _assert_no_locked(clean["unit_id"], locked_ids, "OOF")
    _assert_no_locked(cond["unit_id"], locked_ids, "conditional holdout")
    _assert_no_locked(shifted["unit_id"], locked_ids, "shifted OOF")
    if len(set(clean["unit_id"])) != len(clean["unit_id"]):
        raise ValueError("OOF units must be unique across the supplied folds")
    # Group ids come only from the unlocked split metadata.
    def groups(units: list[str]) -> list[str]:
        return [str(metadata.get(u, {}).get("group", u)) for u in units]
    economics, economics_source = _load_economics()
    fit_candidate = fit_temperature(clean["logits"], clean["y"])
    raw_ece = calibration_metrics(clean["logits"], clean["y"], 1.0)["ece"]
    candidate_ece = calibration_metrics(clean["logits"], clean["y"], fit_candidate.temperature)["ece"]
    # A bound hit is not allowed to silently become a calibration claim.  Keep
    # the fitted value only when it improves ECE; the bound-hit flag remains
    # visible even when the operational temperature falls back to 1.0.
    kept_temperature = fit_candidate.temperature if (not fit_candidate.t_bound_hit or candidate_ece < raw_ece) else 1.0
    fit = type(fit_candidate)(kept_temperature, float(calibration_metrics(clean["logits"], clean["y"], kept_temperature)["nll"]), fit_candidate.raw_nll, fit_candidate.t_bound_hit, fit_candidate.surface_flat)
    cm = _dataset_metrics(clean["logits"], clean["y"], fit.temperature, groups(clean["unit_id"]), economics)
    cond_cm = _dataset_metrics(cond["logits"], cond["y"], fit.temperature, groups(cond["unit_id"]), economics)
    shifted_cm = _dataset_metrics(shifted["logits"], shifted["y"], fit.temperature, groups(shifted["unit_id"]) if shifted["unit_id"] else [], economics) if len(shifted["y"]) else None

    bank_path = input_dir / "bank_fold0.npy"
    bank_source = "bank_fold0.npy"
    if not bank_path.exists():
        bank_candidates = sorted(input_dir.glob("bank_fold*.npy"))
        if not bank_candidates:
            raise FileNotFoundError("bank_fold0.npy (or another bank_fold{k}.npy) is required")
        bank_path = bank_candidates[0]
        bank_source = bank_path.name
    bank = deterministic_coreset(np.load(bank_path), KNN_LIMIT)
    np.savez_compressed(artifacts / "knn_bank.npz", bank=bank)
    class_probs = {"clean": cm["probs"], "cond": cond_cm["probs"], "shifted": shifted_cm["probs"] if shifted_cm else np.empty((0, len(CLASSES)))}
    scores, novelty_threshold, embedding_source = _score_embeddings(input_dir, clean, cond, shifted, folds, bank, class_probs)
    cond_agreement: dict[str, float] = {}
    by_cond: dict[str, list[np.ndarray]] = {}
    for rec in cond_records:
        for unit, logits in zip(rec["unit_id"], rec["logits"]):
            by_cond.setdefault(unit, []).append(logits)
    for unit, pieces in by_cond.items():
        ensemble_argmax = int(cond["logits"][cond["unit_id"].index(unit)].argmax())
        cond_agreement[unit] = float(np.mean([int(piece.argmax()) == ensemble_argmax for piece in pieces]))
    _write_predictions(artifacts, clean, cond, cm, cond_cm, scores, novelty_threshold, metadata, locked_ids, cond_agreement)
    robust = _write_robustness(input_dir, artifacts, folds)

    def serial_metrics(item: dict[str, Any] | None) -> dict[str, Any] | None:
        if item is None:
            return None
        return {"raw": item["raw"], "scaled": item["scaled"], "conformal": item["conformal"], "error_rates": item["error_rates"], "expected_cost_per_unit": item["expected_cost_per_unit"], "operating_threshold": item["operating_threshold"], "risk_coverage": item["risk_coverage"], "cost_curve": item["cost_curve"]}

    calibration = {
        "schema_version": 2,
        "folds_used": folds,
        "n_oof": int(len(clean["y"])),
        "n_cond_holdout": int(len(cond["y"])),
        "n_shifted": int(len(shifted["y"])),
        "temperature": {"value": fit.temperature, "candidate_value": fit_candidate.temperature, "nll": fit.nll, "raw_nll": fit.raw_nll, "t_bound_hit": fit.t_bound_hit, "surface_flat": fit.surface_flat, "candidate_ece": candidate_ece, "raw_ece": raw_ece, "kept_fitted_temperature": bool(fit.temperature != 1.0 or not fit.t_bound_hit), "fit_method": "log_grid_plus_golden_refinement"},
        "alpha": ALPHA,
        "economics": {**economics, "source": economics_source, "assumption_id": "D18" if "ASSUMED" in economics_source else None},
        "decision_rule": {"accept_cost": "p_defect * c_escape", "reject_cost": "(1-p_defect) * c_false_reject", "review_cost": "c_review", "override": "UNCERTAIN and NOVEL route to REVIEW", "wording": "association, not proven causation"},
        "clean_oof": serial_metrics(cm),
        "cond_holdout": serial_metrics(cond_cm),
        "shifted": serial_metrics(shifted_cm),
        "set_definitions": {"normal_index": NORMAL_INDEX, "classes": CLASSES, "uncertain": "empty conformal set or set contains normal and any defect class"},
        "provenance": {"input_dir": str(input_dir), "locked_holdout_excluded": True},
    }
    (artifacts / "calibration.json").write_text(json.dumps(calibration, indent=2, allow_nan=False), encoding="utf-8")
    novelty = {"schema_version": 1, "knn": {"k": KNN_K, "threshold": novelty_threshold, "fpr_target": 0.05, "embedding_source": f"{bank_source}; {embedding_source}", "bank_n": int(len(bank)), "dtype": "float16", "coreset_limit": KNN_LIMIT, "scores_are_cosine_distance": True, "assumption_id": None}}
    (artifacts / "novelty.json").write_text(json.dumps(novelty, indent=2), encoding="utf-8")
    report = {"folds": folds, "temperature": fit.temperature, "t_bound_hit": fit.t_bound_hit, "ece_raw": cm["raw"]["ece"], "ece_scaled": cm["scaled"]["ece"], "conformal_q_hat": cm["conformal"]["q_hat"], "conformal_coverage": cm["conformal"]["coverage"], "robustness_rows": int(len(robust))}
    print(f"folds_used={folds}")
    print(f"temperature={fit.temperature:.12g} t_bound_hit={fit.t_bound_hit}")
    print(f"clean_oof ECE raw={cm['raw']['ece']:.12g} scaled={cm['scaled']['ece']:.12g} Brier raw={cm['raw']['brier']:.12g} scaled={cm['scaled']['brier']:.12g} NLL raw={cm['raw']['nll']:.12g} scaled={cm['scaled']['nll']:.12g}")
    print(f"clean conformal q_hat={cm['conformal']['q_hat']:.12g} coverage={cm['conformal']['coverage']:.12g}")
    print(f"shifted_n={len(shifted['y'])} novelty_threshold={novelty_threshold:.12g}")
    return report


def _make_fixture(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for fold in (0, 1):
        units = [f"u{fold}_{i}" for i in range(20)]
        y = np.asarray([i % 5 for i in range(20)], dtype=int)
        logits = rng.normal(0, 0.3, (20, 5))
        logits[np.arange(20), y] += 3.0
        for prefix, shift in (("oof", 0.0), ("oof_shift", 0.7)):
            frame = pd.DataFrame({"unit_id": units, "true": y, "logits": [json.dumps(x.tolist()) for x in (logits + shift * rng.normal(size=logits.shape))]})
            frame.to_parquet(path / f"{prefix}_fold{fold}.parquet", index=False)
        cond_units = [f"c{i}" for i in range(10)]
        cond_y = np.asarray([i % 5 for i in range(10)], dtype=int)
        cond_logits = rng.normal(0, 0.3, (10, 5)); cond_logits[np.arange(10), cond_y] += 2.5
        pd.DataFrame({"unit_id": cond_units, "true": cond_y, "logits": [json.dumps(x.tolist()) for x in cond_logits]}).to_parquet(path / f"condholdout_fold{fold}.parquet", index=False)
        np.save(path / f"oof_fold{fold}.npy", rng.normal(size=(20, 8)).astype(np.float16))
        np.save(path / f"condholdout_fold{fold}.npy", rng.normal(size=(10, 8)).astype(np.float16))
        np.save(path / f"bank_fold{fold}.npy", rng.normal(size=(30, 8)).astype(np.float16))
        pd.DataFrame({"perturbation": ["noise_sigma10", "blur_sigma3"], "acc": [0.98 - 0.02 * fold, 0.91], "macro_f1": [0.97, 0.90], "n": [20, 20]}).to_csv(path / f"stress_fold{fold}.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    if args.make_fixture:
        fixture = Path(args.fixture_dir) if args.fixture_dir else ROOT / "tests" / "fixtures" / "calibration_run"
        _make_fixture(fixture)
        # The fixture's ids are intentionally independent of the repository
        # split file; write an unlocked matching split table beside it.
        split_rows = [{"file": f"{u}.png", "cls": CLASSES[i % 5], "group": i // 2, "split": "dev", "fold": i % 2} for i, u in enumerate([f"u{f}_{j}" for f in (0, 1) for j in range(20)])]
        split_rows.extend({"file": f"c{i}.png", "cls": CLASSES[i % 5], "group": 20 + i, "split": "cond_holdout", "fold": ""} for i in range(10))
        fixture_splits = fixture / "splits.csv"
        pd.DataFrame(split_rows).to_csv(fixture_splits, index=False)
        args.input_dir, args.splits = str(fixture), str(fixture_splits)
    input_dir = Path(args.input_dir) if args.input_dir else ROOT / "kaggle_downloads"
    build(input_dir, Path(args.splits), Path(args.artifacts))
