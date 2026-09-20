"""Tiny offline CPU smoke test for the leave-one-family-out novelty script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "lofo_novelty.py"
    spec = importlib.util.spec_from_file_location("lofo_novelty_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lofo_cpu_dryrun_outputs_and_locked_guard(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("pandas")
    pytest.importorskip("PIL")
    from PIL import Image

    script = _load_script()
    data_root = tmp_path / "images"
    data_root.mkdir()
    rows: list[dict[str, str | int]] = []
    classes = ["crack", "hole", "normal", "rust", "scratch"]
    rng = np.random.default_rng(7)
    for class_index, name in enumerate(classes):
        for fold in range(5):
            for replicate in range(2):
                filename = f"{name}_dev_f{fold}_{replicate}.png"
                array = np.full((64, 64), 25 + class_index * 40, dtype=np.uint8)
                array[(class_index + replicate + fold) % 8:32, 8 + class_index:40] = np.uint8(180 - class_index * 20)
                array = np.clip(array + rng.integers(0, 8, size=array.shape), 0, 255).astype(np.uint8)
                Image.fromarray(array, mode="L").save(data_root / filename)
                rows.append({"file": filename, "cls": name, "split": "dev", "fold": fold})
        for replicate in range(2):
            filename = f"{name}_cond_{replicate}.png"
            array = np.full((64, 64), 30 + class_index * 38, dtype=np.uint8)
            array[16:48, 16:48] = np.uint8(170 - class_index * 18)
            Image.fromarray(array, mode="L").save(data_root / filename)
            rows.append({"file": filename, "cls": name, "split": "cond_holdout", "fold": -1})
    # This row must be filtered before it reaches the DataFrame or image loader.
    rows.append({"file": "locked_secret_id.png", "cls": "normal", "split": "holdout", "fold": -1})
    splits = tmp_path / "splits.csv"
    with splits.open("w", encoding="utf-8", newline="") as handle:
        fields = ["file", "cls", "split", "fold"]
        handle.write(",".join(fields) + "\n")
        for row in rows:
            handle.write(",".join(str(row[field]) for field in fields) + "\n")
    output = tmp_path / "out"
    code = script.main(["--data-root", str(data_root), "--splits-csv", str(splits), "--out", str(output), "--families", "crack,hole,rust,scratch", "--gpu", "cpu", "--dry-run"])
    assert code == 0
    expected = [output / f"lofo_{family}.json" for family in ["crack", "hole", "rust", "scratch"]]
    expected.append(output / "novelty_lofo.json")
    assert all(path.is_file() for path in expected)
    for path in expected:
        payload = path.read_bytes()
        assert b"locked_secret_id" not in payload, path
    aggregate = json.loads((output / "novelty_lofo.json").read_text(encoding="utf-8"))
    assert aggregate["families_completed"] == 4
    assert set(aggregate["families"]) == {"crack", "hole", "rust", "scratch"}
    for family in aggregate["families"].values():
        assert {row["detector"] for row in family["detectors"]} == {"one_minus_max_softmax", "energy", "knn_cosine_distance_k5"}
        assert all(len(row["risk_coverage"]) == 20 for row in family["detectors"])
