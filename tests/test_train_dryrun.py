"""Integration test for the CPU-safe two-fold trainer dry-run."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAINER = ROOT / "notebooks" / "train_5fold.py"
SPLITS = ROOT / "artifacts" / "splits.csv"
DATA = ROOT / "data" / "images" / "train"


def _can_run() -> bool:
    if not PYTHON.exists() or not TRAINER.exists() or not SPLITS.exists() or not DATA.exists():
        return False
    probe = subprocess.run([str(PYTHON), "-c", "import torch, timm, pandas"], capture_output=True, text=True)
    return probe.returncode == 0


@pytest.mark.skipif(not _can_run(), reason="project .venv does not have torch/timm/pandas")
def test_train_dryrun_outputs_and_locked_holdout_exclusion() -> None:
    expected_suffixes = [
        "fold{fold}.pt", "oof_fold{fold}.parquet", "oof_fold{fold}.npy",
        "condholdout_fold{fold}.parquet", "condholdout_fold{fold}.npy",
        "bank_fold{fold}.npy", "oof_shift_fold{fold}.parquet",
        "stress_fold{fold}.csv", "log_fold{fold}.json",
    ]
    with tempfile.TemporaryDirectory(prefix="qualitytwin_dryrun_", dir=ROOT / "notebooks") as tmp:
        out = Path(tmp)
        result = subprocess.run(
            [str(PYTHON), str(TRAINER), "--dry-run", "--gpu", "cpu", "--folds", "0,1",
             "--data-root", str(DATA), "--splits-csv", str(SPLITS), "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        for fold in (0, 1):
            for suffix in expected_suffixes:
                assert (out / suffix.format(fold=fold)).exists(), suffix.format(fold=fold)

        locked_ids = set()
        with SPLITS.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split", "").strip().lower() in {"holdout", "locked_holdout", "locked"}:
                    locked_ids.add(Path(row["file"]).stem)
        assert locked_ids
        import pandas as pd
        for parquet in out.glob("*.parquet"):
            frame = pd.read_parquet(parquet)
            assert not locked_ids.intersection(set(frame["unit_id"].astype(str)))
