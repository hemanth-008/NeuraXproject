"""Offline, CPU-only inference and heatmap generation for QualityTwin.

The training job saves a plain EfficientNet-B0 state dict per fold.  This
module deliberately reconstructs that model with ``pretrained=False`` so an
inference run can never reach the network.  The public ``run_inference``
function is also small enough for the demo scripts and tests to reuse.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CLASS_NAMES = ("crack", "hole", "normal", "rust", "scratch")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_SIZE = 256


class InferenceError(RuntimeError):
    """Raised when an inference prerequisite is missing or invalid."""


def _json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InferenceError(f"required calibration artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InferenceError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InferenceError(f"expected an object in {path}")
    return value


def _nested(value: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = value
        ok = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return current
    return None


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _load_torch_timm() -> tuple[Any, Any]:
    try:
        import torch
        import timm
    except Exception as exc:  # pragma: no cover - exercised in missing-dependency environments
        raise InferenceError("CPU inference requires importable torch and timm; no packages were installed") from exc
    if torch.cuda.is_available():
        # The process is intentionally pinned to CPU even on a GPU host.
        torch.set_default_device("cpu") if hasattr(torch, "set_default_device") else None
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch, timm


def discover_weights(artifacts_dir: Path) -> list[Path]:
    paths = []
    for path in artifacts_dir.joinpath("models").glob("fold*.pt"):
        match = re.fullmatch(r"fold(\d+)\.pt", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort(key=lambda item: item[0])
    if not paths:
        raise InferenceError(f"no model weights found in {artifacts_dir / 'models'}; expected fold{{k}}.pt")
    return [path for _, path in paths]


def _state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise InferenceError("model checkpoint is not a state_dict")
    result = {}
    for key, value in checkpoint.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[7:]
        result[clean_key] = value
    return result


def _make_model(torch: Any, timm: Any) -> Any:
    try:
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASS_NAMES))
    except Exception as exc:
        raise InferenceError("could not construct timm efficientnet_b0 without pretrained weights") from exc
    return model.to(device="cpu", dtype=torch.float32)


def load_models(artifacts_dir: Path) -> tuple[list[Any], Any, list[Path]]:
    """Load all available folds as float32 CPU models and return torch too."""
    torch, timm = _load_torch_timm()
    weights = discover_weights(artifacts_dir)
    models: list[Any] = []
    for path in weights:
        model = _make_model(torch, timm)
        try:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            state = _state_dict(checkpoint)
            state = {key: value.float() if torch.is_floating_point(value) else value for key, value in state.items()}
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise InferenceError(f"could not load checkpoint {path}: {exc}") from exc
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        models.append(model)
    return models, torch, weights


def preprocess_image(path: Path, size: int = IMAGE_SIZE) -> tuple[np.ndarray, tuple[int, int]]:
    """Convert any supported image mode to normalized 3-channel model input."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise InferenceError("Pillow is required for image inference") from exc
    try:
        with Image.open(path) as image:
            original_size = image.size
            resampling = getattr(Image, "Resampling", Image)
            gray = image.convert("L").resize((size, size), resampling.BILINEAR)
            array = np.asarray(gray, dtype=np.float32) / 255.0
    except Exception as exc:
        raise InferenceError(f"could not read image {path}: {exc}") from exc
    array = np.repeat(array[None, :, :], 3, axis=0)
    array = (array - 0.449) / 0.226
    return np.ascontiguousarray(array, dtype=np.float32), original_size


def _tensorize(torch: Any, array: np.ndarray) -> Any:
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(device="cpu", dtype=torch.float32)


def _tta_inputs(torch: Any, x: Any, flips: bool, rot90: bool) -> list[Any]:
    values = [x]
    if flips:
        values.extend((x.flip(-1), x.flip(-2)))
    if rot90:
        values.extend((torch.rot90(x, 1, (-2, -1)), torch.rot90(x, 2, (-2, -1)), torch.rot90(x, 3, (-2, -1))))
    return values


def _forward_logits(model: Any, x: Any) -> tuple[Any, Any]:
    features = model.forward_features(x)
    embedding = model.forward_head(features, pre_logits=True)
    if embedding.ndim > 2:
        embedding = embedding.flatten(1)
    logits = model.get_classifier()(embedding)
    return logits, embedding


def _softmax(torch: Any, logits: Any) -> np.ndarray:
    return torch.softmax(logits.float(), dim=1).detach().cpu().numpy()[0].astype(np.float64)


def _first_conv_module(model: Any) -> tuple[str, Any]:
    import torch.nn as nn

    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not candidates:
        raise InferenceError("the model has no convolutional layer for Grad-CAM++")
    return candidates[-1]


def gradcam_pp(model: Any, x: Any, class_index: int, torch: Any, target_layer: Any | None = None) -> np.ndarray:
    """Return a normalized 256x256 Grad-CAM++ heatmap for one input."""
    activations: list[Any] = []
    gradients: list[Any] = []
    layer = target_layer or _first_conv_module(model)[1]
    forward_handle = layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    try:
        if hasattr(layer, "register_full_backward_hook"):
            backward_handle = layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
        else:  # pragma: no cover - current torch has full backward hooks
            backward_handle = layer.register_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
        try:
            model.zero_grad(set_to_none=True)
            logits, _ = _forward_logits(model, x)
            logits[:, int(class_index)].sum().backward()
        finally:
            backward_handle.remove()
    finally:
        forward_handle.remove()
    if not activations or not gradients:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    activation = activations[-1]
    gradient = gradients[-1]
    if activation.ndim != 4 or gradient.ndim != 4:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    grad2 = gradient.pow(2)
    grad3 = gradient.pow(3)
    denominator = 2.0 * grad2 + (activation * grad3).sum(dim=(2, 3), keepdim=True)
    alpha = grad2 / (denominator + 1e-7)
    weights = (alpha * torch.relu(gradient)).sum(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    values = cam[0, 0].detach().cpu().numpy().astype(np.float32)
    values -= float(values.min())
    maximum = float(values.max())
    return values / maximum if maximum > 1e-8 else np.zeros_like(values)


def _otsu_threshold(values: np.ndarray) -> float:
    data = np.clip(values * 255.0, 0, 255).astype(np.uint8).ravel()
    histogram = np.bincount(data, minlength=256).astype(np.float64)
    total = data.size
    if total == 0 or histogram.sum() == 0:
        return 0.5
    cumulative = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))
    mean_total = cumulative_mean[-1]
    denominator = cumulative * (total - cumulative)
    numerator = (mean_total * cumulative - cumulative_mean) ** 2
    score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(np.argmax(score)) / 255.0


def _largest_component(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return x1,y1,x2,y2 for the largest 4-connected true component."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y, x in zip(*np.where(mask & ~visited)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        points: list[tuple[int, int]] = []
        while stack:
            yy, xx = stack.pop()
            points.append((yy, xx))
            for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(points) > len(best):
            best = points
    if len(best) < 2:
        return None
    ys = [p[0] for p in best]
    xs = [p[1] for p in best]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def heatmap_box(heatmap: np.ndarray, original_size: tuple[int, int], threshold: float | None = None) -> list[int] | None:
    cut = _otsu_threshold(heatmap) if threshold is None else float(threshold)
    if cut > 1.0:
        cut /= 255.0
    box = _largest_component(heatmap >= np.clip(cut, 0.0, 1.0))
    if box is None:
        return None
    width, height = original_size
    sx, sy = width / heatmap.shape[1], height / heatmap.shape[0]
    x1, y1, x2, y2 = box
    return [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]


def _overlay(path: Path, heatmap: np.ndarray, original_size: tuple[int, int]) -> None:
    from PIL import Image

    width, height = original_size
    with Image.open(path) as image:
        base = np.asarray(image.convert("L").resize((width, height)), dtype=np.float32) / 255.0
    hm = np.asarray(Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize((width, height)), dtype=np.float32) / 255.0
    # A dependency-free blue-to-red map is enough for a stable offline artifact.
    color = np.stack((hm, np.sqrt(hm), 1.0 - hm), axis=-1)
    out = (0.55 * np.repeat(base[..., None], 3, axis=2) + 0.45 * color) * 255.0
    Image.fromarray(np.uint8(np.clip(out, 0, 255))).save(path)


@dataclass
class RuntimeArtifacts:
    temperature: float
    q_hat: float
    localization_threshold: float | None
    novelty_threshold: float | None
    novelty_bank: np.ndarray | None
    novelty_k: int


def _load_runtime_artifacts(artifacts_dir: Path) -> RuntimeArtifacts:
    calibration = _json_load(artifacts_dir / "calibration.json")
    temperature = _as_float(_nested(calibration, "temperature", "T", "temperature_scale"), 1.0)
    q_hat = _as_float(_nested(calibration, "q_hat", "conformal.q_hat", "conformal_q_hat"), 1.0)
    if temperature is None or temperature <= 0:
        raise InferenceError("calibration.json has no positive temperature")
    if q_hat is None:
        raise InferenceError("calibration.json has no conformal q_hat")
    localization_threshold = None
    localization_path = artifacts_dir / "localization.json"
    if localization_path.exists():
        localization_threshold = _as_float(_nested(_json_load(localization_path), "threshold", "heatmap_threshold", "f_optimal_threshold"))
    novelty_threshold = None
    novelty_k = 5
    novelty_bank = None
    novelty_path = artifacts_dir / "novelty.json"
    bank_path = artifacts_dir / "knn_bank.npz"
    if novelty_path.exists() and bank_path.exists():
        novelty = _json_load(novelty_path)
        threshold = _nested(novelty, "knn.threshold", "threshold")
        novelty_threshold = _as_float(threshold)
        novelty_k = int(_as_float(_nested(novelty, "knn.k", "k"), 5) or 5)
        with np.load(bank_path, allow_pickle=False) as data:
            key = next((candidate for candidate in ("embeddings", "bank", "vectors", "arr_0") if candidate in data), None)
            if key is None:
                raise InferenceError(f"{bank_path} has no embeddings/bank/vectors array")
            novelty_bank = np.asarray(data[key], dtype=np.float32)
        if novelty_bank.ndim != 2 or novelty_bank.shape[0] == 0:
            raise InferenceError(f"invalid kNN bank shape {novelty_bank.shape}")
        norms = np.linalg.norm(novelty_bank, axis=1, keepdims=True)
        novelty_bank = novelty_bank / np.maximum(norms, 1e-8)
    return RuntimeArtifacts(float(temperature), float(q_hat), localization_threshold, novelty_threshold, novelty_bank, max(1, novelty_k))


def _load_costs(repo_root: Path) -> dict[str, float]:
    values = {"c_escape": 50.0, "c_false_reject": 5.0, "c_review": 1.0}
    path = repo_root / "config.yaml"
    if not path.exists():
        return values
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        economics = loaded.get("economics", {}) if isinstance(loaded, dict) else {}
        for key in values:
            parsed = _as_float(economics.get(key))
            if parsed is not None and parsed >= 0:
                values[key] = parsed
    except Exception:
        # Keep the documented config defaults if a YAML reader is unavailable.
        pass
    return values


class InferenceEngine:
    def __init__(self, artifacts_dir: Path, repo_root: Path | None = None, tta_flips: bool = False, tta_rot90: bool = False):
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.repo_root = Path(repo_root or self.artifacts_dir.parent).resolve()
        self.models, self.torch, self.weight_paths = load_models(self.artifacts_dir)
        self.runtime = _load_runtime_artifacts(self.artifacts_dir)
        self.tta_flips = bool(tta_flips)
        self.tta_rot90 = bool(tta_rot90)
        self.costs = _load_costs(self.repo_root)
        self.target_layer = _first_conv_module(self.models[0])[1]

    def _novelty(self, embedding: np.ndarray) -> float | None:
        bank = self.runtime.novelty_bank
        if bank is None:
            return None
        unit = embedding.astype(np.float32)
        unit /= max(float(np.linalg.norm(unit)), 1e-8)
        distances = 1.0 - bank @ unit
        k = min(self.runtime.novelty_k, len(distances))
        return float(np.sort(distances)[:k].mean())

    def _decision(self, probs: np.ndarray, uncertain: bool, novel: bool) -> tuple[str, str, float, list[str], str]:
        top = np.argsort(probs)[::-1]
        family_index = int(top[0])
        family = CLASS_NAMES[family_index]
        p_defect = float(1.0 - probs[2])
        expected = {
            "ACCEPT": p_defect * self.costs["c_escape"],
            "REJECT": (1.0 - p_defect) * self.costs["c_false_reject"],
            "REVIEW": self.costs["c_review"],
        }
        decision = min(expected, key=expected.get)
        flag = "NONE"
        if novel:
            flag = "NOVEL"
        elif uncertain:
            flag = "UNCERTAIN"
        if flag != "NONE":
            decision = "REVIEW"
        conformal = [CLASS_NAMES[i] for i, value in enumerate(probs) if value >= 1.0 - self.runtime.q_hat]
        if not conformal:
            conformal = []
        top2 = [CLASS_NAMES[int(i)] for i in top[:2]]
        return decision, family, p_defect, conformal, json.dumps(top2)

    def predict_array(self, array: np.ndarray, class_for_cam: int | None = None) -> dict[str, Any]:
        x = _tensorize(self.torch, array)
        with self.torch.no_grad():
            logits_values = []
            embedding = None
            for model in self.models:
                variants = _tta_inputs(self.torch, x, self.tta_flips, self.tta_rot90)
                variant_logits = [_forward_logits(model, item)[0] for item in variants]
                logits_values.append(self.torch.stack(variant_logits).mean(dim=0)[0])
                if embedding is None:
                    _, embedding_tensor = _forward_logits(model, x)
                    embedding = embedding_tensor[0].detach().cpu().numpy()
            logits = self.torch.stack(logits_values).mean(dim=0) / self.runtime.temperature
            probs = _softmax(self.torch, logits.unsqueeze(0))
        top = np.argsort(probs)[::-1]
        uncertain = len([p for p in probs if p >= 1.0 - self.runtime.q_hat]) == 0 or (probs[2] >= 1.0 - self.runtime.q_hat and any(probs[i] >= 1.0 - self.runtime.q_hat for i in (0, 1, 3, 4)))
        novel_score = self._novelty(embedding) if embedding is not None else None
        novel = novel_score is not None and self.runtime.novelty_threshold is not None and novel_score > self.runtime.novelty_threshold
        decision, family, p_defect, conformal, top2 = self._decision(probs, uncertain, novel)
        heatmap = None
        if class_for_cam is None:
            class_for_cam = int(top[0])
        heatmap = gradcam_pp(self.models[0], x, class_for_cam, self.torch, self.target_layer)
        return {
            "decision": decision,
            "family": family,
            "confidence": float(probs[int(top[0])]),
            "flag": "NOVEL" if novel else ("UNCERTAIN" if uncertain else "NONE"),
            "novel_score": novel_score,
            "p_defect": p_defect,
            "class_probs": probs.tolist(),
            "conformal_set": conformal,
            "top2": top2,
            "heatmap": heatmap,
            "embedding": embedding,
        }

    def predict_file(self, path: Path, heatmap_dir: Path | None = None) -> dict[str, Any]:
        array, original_size = preprocess_image(path)
        result = self.predict_array(array)
        heatmap = result.pop("heatmap")
        box = None if result["decision"] == "ACCEPT" else heatmap_box(heatmap, original_size, self.runtime.localization_threshold)
        heatmap_path = None
        if heatmap_dir is not None:
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem) + "_heatmap.png"
            heatmap_path = heatmap_dir / safe_name
            # Render into the output directory; never modify the user's source image.
            from PIL import Image

            width, height = original_size
            with Image.open(path) as image:
                base = np.asarray(image.convert("L").resize((width, height)), dtype=np.float32) / 255.0
            from PIL import Image as PILImage

            hm = np.asarray(PILImage.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize((width, height)), dtype=np.float32) / 255.0
            color = np.stack((hm, np.sqrt(hm), 1.0 - hm), axis=-1)
            overlay = (0.55 * np.repeat(base[..., None], 3, axis=2) + 0.45 * color) * 255.0
            PILImage.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(heatmap_path)
        result.update({"file": str(path), "heatmap_path": str(heatmap_path) if heatmap_path else None, "box_xyxy": box, "localization_basis": "heatmap+derived_box" if heatmap_path else None})
        result.pop("embedding", None)
        return result


def _image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise InferenceError(f"input image folder does not exist: {input_dir}")
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        records.append({
            "unit_id": Path(row["file"]).stem,
            "image_path": row["file"],
            "batch_id": None, "variant": None, "condition": None, "station": None, "ts": None,
            "p_defect": row["p_defect"], "class_probs": row["class_probs"], "pred_family": row["family"],
            "decision": row["decision"], "flag": row["flag"], "anomaly_score": row["novel_score"],
            "novel_score": row["novel_score"], "conformal_set": row["conformal_set"], "heatmap_path": row["heatmap_path"],
            "box_xyxy": row["box_xyxy"], "localization_basis": row["localization_basis"], "true_label": None,
            "source_split": "inference", "ensemble_agreement": None, "top2": row["top2"], "confidence": row["confidence"],
        })
    frame = pd.DataFrame(records)
    csv_frame = frame[["image_path", "decision", "pred_family", "confidence", "flag", "novel_score", "heatmap_path", "box_xyxy", "top2", "conformal_set"]].copy()
    for column in ("box_xyxy", "top2", "conformal_set"):
        csv_frame[column] = csv_frame[column].map(lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value)
    csv_frame.rename(columns={"image_path": "file", "pred_family": "family"}).to_csv(output_dir / "predictions.csv", index=False)
    frame.to_parquet(output_dir / "predictions.parquet", index=False)


def run_inference(input_dir: str | Path, output_dir: str | Path, artifacts_dir: str | Path | None = None, tta_flips: bool = False, tta_rot90: bool = False) -> list[dict[str, Any]]:
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    artifact_path = Path(artifacts_dir).resolve() if artifacts_dir else repo_root / "artifacts"
    engine = InferenceEngine(artifact_path, repo_root=repo_root, tta_flips=tta_flips, tta_rot90=tta_rot90)
    paths = _image_paths(input_path)
    if not paths:
        raise InferenceError(f"no PNG/JPG/BMP images found under {input_path}")
    heatmap_dir = output_path / "heatmaps"
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for path in paths:
        rows.append(engine.predict_file(path, heatmap_dir))
    elapsed = time.perf_counter() - started
    _write_outputs(rows, output_path)
    print(f"inference_cpu images={len(rows)} seconds={elapsed:.6f} seconds_per_image={elapsed / max(1, len(rows)):.6f}", flush=True)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline CPU inference for any image folder")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--tta-flips", action="store_true")
    parser.add_argument("--tta-rot90", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output = Path(args.output) if args.output else repo_root / "artifacts"
    try:
        run_inference(args.input, output, args.artifacts_dir, args.tta_flips, args.tta_rot90)
    except InferenceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
