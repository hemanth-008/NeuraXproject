"""CPU smoke tests for the offline inference contract.

The checkpoint is deliberately created in pytest's temporary directory and is
never written to artifacts/models.  Tests skip cleanly when the optional model
runtime is not installed on a developer machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _runtime_or_skip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("PIL")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    return torch


def _fixture(tmp_path: Path):
    torch = _runtime_or_skip()
    import timm
    from PIL import Image

    artifacts = tmp_path / "artifacts"
    (artifacts / "models").mkdir(parents=True)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
    state = {key: value.detach().cpu().half() if torch.is_floating_point(value) else value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, artifacts / "models" / "fold0.pt")
    (artifacts / "calibration.json").write_text(json.dumps({"temperature": 1.0, "q_hat": 0.1}), encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(images / "gray.png")
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8), mode="RGB").save(images / "rgb.jpg")
    Image.fromarray(np.full((300, 180), 180, dtype=np.uint8), mode="L").save(images / "large.bmp")
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, size=(70, 90), dtype=np.uint8), mode="L").save(images / "noisy.png")
    Image.fromarray(np.rot90(np.arange(80 * 55, dtype=np.uint8).reshape(80, 55)), mode="L").save(images / "rotated.png")
    return artifacts, images


def test_infer_shapes_and_outputs(tmp_path: Path):
    artifacts, images = _fixture(tmp_path)
    from src.vision.infer import run_inference

    output = tmp_path / "out"
    rows = run_inference(images, output, artifacts_dir=artifacts)
    assert len(rows) == 5
    assert {"file", "decision", "family", "confidence", "flag", "novel_score", "heatmap_path", "box_xyxy", "top2", "conformal_set"}.issubset(rows[0])
    assert (output / "predictions.csv").exists()
    assert (output / "predictions.parquet").exists()
    import pandas as pd

    frame = pd.read_parquet(output / "predictions.parquet")
    assert len(frame) == 5
    assert {"decision", "pred_family", "heatmap_path", "box_xyxy", "localization_basis"}.issubset(frame.columns)
    assert all(Path(path).exists() for path in frame["heatmap_path"])


def test_missing_weights_fail_loudly(tmp_path: Path):
    _runtime_or_skip()
    from src.vision.infer import InferenceError, InferenceEngine

    artifacts = tmp_path / "artifacts"
    (artifacts / "models").mkdir(parents=True)
    (artifacts / "calibration.json").write_text(json.dumps({"temperature": 1.0, "q_hat": 0.1}), encoding="utf-8")
    with pytest.raises(InferenceError, match="no model weights"):
        InferenceEngine(artifacts)
