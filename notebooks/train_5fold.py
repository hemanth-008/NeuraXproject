"""Self-contained five-fold QualityTwin trainer.

The reference notebook uses timm EfficientNet-B0, sqrt-inverse class weights,
AdamW, OneCycleLR, AMP, and a fixed twelve-epoch run.  This file keeps that
recipe while moving image batching and augmentation out of a DataLoader.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# Configuration block.  Kaggle paths are discovered at runtime; CLI flags and
# environment variables are intentionally supported for the local dry-run.
DATA_ROOT = os.environ.get("QUALITYTWIN_DATA_ROOT", "")
SPLITS_CSV = os.environ.get("QUALITYTWIN_SPLITS_CSV", "")
OUT = os.environ.get("QUALITYTWIN_OUT", "/kaggle/working")
FOLDS = [0, 1, 2, 3, 4]
GPU = os.environ.get("QUALITYTWIN_GPU", "auto")
EPOCHS = 12
IMG = 256
SEED = 0
DRY_RUN = False
ARCH = "efficientnet_b0"
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-2

LOCKED_WORDS = {"holdout", "locked", "locked_holdout", "locked-holdout"}
SHIFT_NAMES = [
    "noise_sigma10", "noise_sigma40", "jpeg_q20", "blur_sigma3",
    "lowres_4x", "motion_blur", "speckle", "rot90", "rot180", "rot270",
    "angle", "hflip", "vflip", "scale_0.5x", "scale_2x", "center_crop70",
    "perspective", "gamma", "shadow_gradient", "color_temperature",
]


def script_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return hashlib.sha256(b"embedded-train_5fold").hexdigest()


def version_of(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def print_banner(device: str) -> None:
    print("=== QualityTwin train_5fold startup ===", flush=True)
    print(f"python={platform.python_version()} torch={version_of('torch')} timm={version_of('timm')}", flush=True)
    try:
        import torch
        print(f"gpu_count={torch.cuda.device_count()} cuda_available={torch.cuda.is_available()} device={device}", flush=True)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"gpu[{i}]={torch.cuda.get_device_name(i)}", flush=True)
    except Exception as exc:
        print(f"gpu_count=unavailable ({type(exc).__name__}) device={device}", flush=True)
    print(f"script_sha256={script_sha256()}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--splits-csv", default=SPLITS_CSV)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--folds", default=None, help="comma-separated fold ids")
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--img", type=int, default=IMG)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    parser.add_argument("--launch-two-gpus", action="store_true")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def find_splits_csv(explicit: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for base in (Path("/kaggle/input"), Path.cwd(), Path(__file__).resolve().parents[1]):
        if base.exists():
            candidates.extend(sorted(base.rglob("splits.csv")))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Could not find splits.csv; pass --splits-csv or attach the dataset")


def find_data_root(explicit: str, split_path: Path, first_file: str) -> Path:
    if explicit:
        root = Path(explicit)
        if (root / first_file).exists():
            return root.resolve()
        raise FileNotFoundError(f"--data-root does not contain {first_file}: {root}")
    search_roots = [Path("/kaggle/input"), split_path.parent, Path.cwd(), Path(__file__).resolve().parents[1]]
    for base in search_roots:
        if not base.exists():
            continue
        direct = base / first_file
        if direct.exists():
            return base.resolve()
        for path in base.rglob(Path(first_file).name):
            candidate = path
            for _ in Path(first_file).parts:
                candidate = candidate.parent
            if (candidate / first_file).exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not locate image {first_file}; pass --data-root")


def locked_mask(df: Any) -> Any:
    """Find the frozen locked holdout without opening any image from it."""
    import pandas as pd
    mask = pd.Series(False, index=df.index)
    for column in df.columns:
        values = df[column].astype(str).str.strip().str.lower()
        exact = values.isin(LOCKED_WORDS)
        contains = values.str.contains("locked", regex=False) & ~values.str.contains("cond", regex=False)
        mask = mask | exact | contains
    if "split" in df.columns:
        split = df["split"].astype(str).str.strip().str.lower()
        # The frozen file calls this partition simply "holdout".
        mask = mask | (split == "holdout") | (split == "locked_holdout")
    return mask


def read_metadata(path: Path) -> Any:
    import pandas as pd
    df = pd.read_csv(path)
    required = {"file", "cls", "split", "fold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"splits.csv missing columns {sorted(missing)}; columns={list(df.columns)}")
    bad = locked_mask(df)
    locked_count = int(bad.sum())
    df = df.loc[~bad].copy().reset_index(drop=True)
    # This assertion is deliberately after the load-time filter: no downstream
    # operation may receive a locked row.
    assert not bool(locked_mask(df).any()), "locked-holdout rows reached the trainer"
    print(f"splits columns={list(df.columns)} rows_after_locked_filter={len(df)} dropped_locked={locked_count}", flush=True)
    if (df["split"].astype(str).str.lower() == "holdout").any():
        raise AssertionError("locked holdout rows were not removed")
    return df


def load_images(df: Any, root: Path, img: int, dry_run: bool) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    rs = getattr(Image, "Resampling", Image).BILINEAR

    def load_one(rel: str) -> np.ndarray:
        path = root / str(rel)
        with Image.open(path) as image:
            return np.asarray(image.convert("L").resize((img, img), rs), dtype=np.uint8)

    t0 = time.perf_counter()
    workers = min(8, max(1, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = np.stack(list(pool.map(load_one, df["file"].tolist())))
    print(f"preloaded {len(images)} images as uint8 tensors shape={images.shape} seconds={time.perf_counter()-t0:.2f}", flush=True)
    return images


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> float:
    scores = []
    for c in range(classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def finite_float_list(array: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(array).reshape(-1)]


def make_input(torch: Any, x_u8: np.ndarray, device: Any) -> Any:
    x = torch.from_numpy(np.ascontiguousarray(x_u8)).to(device=device, dtype=torch.float32).div_(255.0)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.449, 0.449, 0.449], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.226, 0.226, 0.226], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _affine(torch: Any, x: Any, angles: Any, scales: Any, device: Any) -> Any:
    n = x.shape[0]
    rad = angles * math.pi / 180.0
    c, s = torch.cos(rad) / scales, torch.sin(rad) / scales
    theta = torch.zeros((n, 2, 3), device=device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = c, s
    theta[:, 1, 0], theta[:, 1, 1] = -s, c
    grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def augment_gpu(torch: Any, x: Any, seed: int, img: int) -> Any:
    """Batched torch augmentation; approximately 40% of samples remain clean."""
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    n = x.shape[0]
    active = torch.rand(n, device=x.device, generator=g) >= 0.40
    if not bool(active.any()):
        return x
    z = x.clone()
    idx = active.nonzero(as_tuple=False).flatten()
    a = z[idx]
    choose = torch.rand(len(idx), device=x.device, generator=g)
    rot90 = choose < 0.16
    if bool(rot90.any()):
        a[rot90] = torch.rot90(a[rot90], k=1, dims=(-2, -1))
    choose = torch.rand(len(idx), device=x.device, generator=g)
    if bool((choose < 0.20).any()):
        q = choose < 0.20
        a[q] = _affine(torch, a[q], torch.empty(int(q.sum()), device=x.device).uniform_(-180, 180, generator=g), torch.ones(int(q.sum()), device=x.device), x.device)
    flip_h = torch.rand(len(idx), device=x.device, generator=g) < 0.25
    flip_v = torch.rand(len(idx), device=x.device, generator=g) < 0.25
    if bool(flip_h.any()): a[flip_h] = a[flip_h].flip(-1)
    if bool(flip_v.any()): a[flip_v] = a[flip_v].flip(-2)
    crop = torch.rand(len(idx), device=x.device, generator=g) < 0.18
    if bool(crop.any()):
        scale = torch.empty(int(crop.sum()), device=x.device).uniform_(0.6, 1.0, generator=g)
        a[crop] = _affine(torch, a[crop], torch.zeros_like(scale), scale, x.device)
    perspective = torch.rand(len(idx), device=x.device, generator=g) < 0.12
    if bool(perspective.any()):
        # A small differentiable projective-like warp made by perturbing the
        # sampling grid; it is intentionally cheap enough for the GPU path.
        q = a[perspective]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, img, device=x.device),
            torch.linspace(-1, 1, img, device=x.device), indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(len(q), 1, 1, 1)
        drift = torch.empty((len(q), 1, 1, 2), device=x.device).uniform_(-0.08, 0.08, generator=g)
        grid = grid + drift * (grid[..., 1:2].abs() + 0.25)
        a[perspective] = torch.nn.functional.grid_sample(q, grid, align_corners=False, padding_mode="reflection")
    b = torch.empty(len(idx), 1, 1, 1, device=x.device).uniform_(0.8, 1.2, generator=g)
    c = torch.empty(len(idx), 1, 1, 1, device=x.device).uniform_(0.8, 1.2, generator=g)
    gamma = torch.empty(len(idx), 1, 1, 1, device=x.device).uniform_(0.75, 1.35, generator=g)
    a = ((a - a.mean((-2, -1), keepdim=True)) * c + a.mean((-2, -1), keepdim=True)) * b
    a = a.clamp(0, 1).pow(gamma)
    shadow = torch.linspace(1.0, 0.45, img, device=x.device).view(1, 1, 1, img)
    shadow_mask = torch.rand(len(idx), 1, 1, 1, device=x.device, generator=g) < 0.20
    a = torch.where(shadow_mask, a * shadow, a)
    gains = torch.empty((len(idx), 3, 1, 1), device=x.device).uniform_(0.92, 1.08, generator=g)
    temperature = torch.rand(len(idx), 1, 1, 1, device=x.device, generator=g) < 0.18
    a = torch.where(temperature, a * gains, a)
    noise = torch.empty_like(a).normal_(0, 0.035, generator=g)
    a = torch.where((torch.rand(len(idx), 1, 1, 1, device=x.device, generator=g) < 0.28), a + noise, a)
    small = torch.rand(len(idx), device=x.device, generator=g) < 0.18
    if bool(small.any()):
        q = torch.nn.functional.interpolate(a[small], size=(max(8, img // 4), max(8, img // 4)), mode="area")
        a[small] = torch.nn.functional.interpolate(q, size=(img, img), mode="bilinear", align_corners=False)
    blur = torch.rand(len(idx), device=x.device, generator=g) < 0.14
    if bool(blur.any()):
        a[blur] = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(a[blur], (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    motion = torch.rand(len(idx), device=x.device, generator=g) < 0.10
    if bool(motion.any()):
        a[motion] = (a[motion] + a[motion].roll(3, -1) + a[motion].roll(-3, -1)) / 3.0
    cut = torch.rand(len(idx), device=x.device, generator=g) < 0.16
    if bool(cut.any()):
        h = max(4, img // 8)
        cut_positions = cut.nonzero(as_tuple=False).flatten()
        cy = torch.randint(0, img, (len(cut_positions),), device=x.device, generator=g)
        cx = torch.randint(0, img, (len(cut_positions),), device=x.device, generator=g)
        for j, (yy, xx) in enumerate(zip(cy.tolist(), cx.tolist())):
            a[cut_positions[j], :, max(0, yy-h):min(img, yy+h), max(0, xx-h):min(img, xx+h)] = 0
    z[idx] = a.clamp(0, 1)
    return z


def jpeg_cpu(batch: np.ndarray, seed: int, probability: float = 0.10) -> np.ndarray:
    if probability <= 0:
        return batch
    from PIL import Image
    rng = random.Random(seed)
    out = batch.copy()
    for i in range(len(out)):
        if rng.random() >= probability:
            continue
        buf = io.BytesIO()
        Image.fromarray(out[i]).save(buf, format="JPEG", quality=rng.randint(20, 65))
        buf.seek(0)
        with Image.open(buf) as im:
            out[i] = np.asarray(im.convert("L"), dtype=np.uint8)
    return out


def make_model(timm: Any, device: Any, pretrained: bool) -> Any:
    try:
        model = timm.create_model(ARCH, pretrained=pretrained, num_classes=5)
    except Exception:
        if not pretrained:
            raise
        print("pretrained weights unavailable; retrying dry-run model without download", flush=True)
        model = timm.create_model(ARCH, pretrained=False, num_classes=5)
    return model.to(device)


def amp_context(torch: Any, device: Any):
    if getattr(device, "type", str(device)) == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def forward_with_embedding(model: Any, x: Any) -> tuple[Any, Any]:
    features = model.forward_features(x)
    embedding = model.forward_head(features, pre_logits=True)
    if embedding.ndim > 2:
        embedding = embedding.flatten(1)
    logits = model.get_classifier()(embedding)
    return logits, embedding


def predict(model: Any, X: np.ndarray, indices: np.ndarray, device: Any, batch_size: int, img: int, torch: Any, return_embeddings: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    logits_parts: list[np.ndarray] = []
    emb_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            xb = make_input(torch, X[indices[start:start+batch_size]], device)
            with amp_context(torch, device):
                logits, emb = forward_with_embedding(model, xb)
            logits_parts.append(logits.float().cpu().numpy())
            if return_embeddings:
                emb_parts.append(emb.float().cpu().numpy())
    if not logits_parts:
        return np.empty((0, 5), np.float32), np.empty((0, 1280), np.float32) if return_embeddings else None
    return np.concatenate(logits_parts), np.concatenate(emb_parts) if return_embeddings else None


def stress_array(a: np.ndarray, name: str, img: int, seed: int) -> np.ndarray:
    from PIL import Image
    rng = np.random.default_rng(seed)
    f = a.astype(np.float32)
    if name == "noise_sigma10": return np.clip(f + rng.normal(0, 10, a.shape), 0, 255).astype(np.uint8)
    if name == "noise_sigma40": return np.clip(f + rng.normal(0, 40, a.shape), 0, 255).astype(np.uint8)
    if name == "speckle": return np.clip(f * (1 + rng.normal(0, 0.15, a.shape)), 0, 255).astype(np.uint8)
    if name == "rot90": return np.rot90(a, 1).copy()
    if name == "rot180": return np.rot90(a, 2).copy()
    if name == "rot270": return np.rot90(a, 3).copy()
    if name == "hflip": return np.flip(a, 1).copy()
    if name == "vflip": return np.flip(a, 0).copy()
    if name == "jpeg_q20":
        buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", quality=20); buf.seek(0)
        with Image.open(buf) as im: return np.asarray(im.convert("L"), dtype=np.uint8)
    if name == "lowres_4x":
        with Image.fromarray(a) as im: return np.asarray(im.resize((img//4, img//4)).resize((img, img)), dtype=np.uint8)
    if name == "blur_sigma3":
        from PIL import ImageFilter
        t = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(3)))
        return t.astype(np.uint8)
    if name == "motion_blur": return ((f + np.roll(f, 4, 1) + np.roll(f, -4, 1)) / 3).astype(np.uint8)
    if name == "angle":
        # PIL keeps this CPU-only stress path deterministic and isolated.
        with Image.fromarray(a) as im: return np.asarray(im.rotate(float(rng.uniform(-180, 180)), resample=Image.BILINEAR), dtype=np.uint8)
    if name == "scale_0.5x":
        with Image.fromarray(a) as im: return np.asarray(im.resize((img//2, img//2)).resize((img, img)), dtype=np.uint8)
    if name == "scale_2x":
        with Image.fromarray(a) as im: return np.asarray(im.resize((img*2, img*2)).crop((img//2, img//2, img//2+img, img//2+img)), dtype=np.uint8)
    if name == "center_crop70":
        n = max(1, int(img * 0.70)); off = (img-n)//2
        with Image.fromarray(a[off:off+n, off:off+n]) as im: return np.asarray(im.resize((img, img)), dtype=np.uint8)
    if name == "gamma": return np.clip(255 * (a / 255.0) ** float(rng.uniform(0.5, 2.0)), 0, 255).astype(np.uint8)
    if name == "shadow_gradient": return np.clip(f * np.linspace(1.0, 0.35, img)[None, :], 0, 255).astype(np.uint8)
    if name == "color_temperature":
        return np.clip(f * float(rng.uniform(0.85, 1.15)), 0, 255).astype(np.uint8)
    if name == "perspective":
        with Image.fromarray(a) as im: return np.asarray(im.transform((img, img), Image.Transform.QUAD, (4, 4, img-4, 0, img, img-4, 0, img), resample=Image.BILINEAR), dtype=np.uint8)
    return a.copy()


def stress_eval(model: Any, X: np.ndarray, indices: np.ndarray, y: np.ndarray, device: Any, batch_size: int, img: int, torch: Any, out_path: Path, seed: int) -> None:
    rows: list[dict[str, Any]] = []
    for name in SHIFT_NAMES:
        shifted = np.stack([stress_array(X[i], name, img, seed + j) for j, i in enumerate(indices)])
        logits, _ = predict(model, shifted, np.arange(len(shifted)), device, batch_size, img, torch)
        pred = logits.argmax(1)
        rows.append({"perturbation": name, "acc": float(np.mean(pred == y)), "macro_f1": macro_f1(y, pred, 5), "n": int(len(y))})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_prediction_table(path: Path, unit_ids: Sequence[str], true: np.ndarray, logits: np.ndarray) -> None:
    import pandas as pd
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    pd.DataFrame({"unit_id": list(unit_ids), "true": true.astype(int), "logits": [finite_float_list(x) for x in logits], "probs": [finite_float_list(x) for x in probs]}).to_parquet(path, index=False)


def expected_outputs(out: Path, fold: int) -> list[Path]:
    return [out/f"fold{fold}.pt", out/f"oof_fold{fold}.parquet", out/f"oof_fold{fold}.npy", out/f"condholdout_fold{fold}.parquet", out/f"condholdout_fold{fold}.npy", out/f"bank_fold{fold}.npy", out/f"oof_shift_fold{fold}.parquet", out/f"stress_fold{fold}.csv", out/f"log_fold{fold}.json"]


def run_fold(fold: int, df: Any, X: np.ndarray, out: Path, args: argparse.Namespace, device: Any, torch: Any, timm: Any, classes: list[str]) -> None:
    outputs = expected_outputs(out, fold)
    if all(p.exists() for p in outputs):
        print(f"SKIP fold {fold}: outputs already exist", flush=True)
        return
    start = time.perf_counter()
    dev_mask = df["split"].astype(str).str.lower() == "dev"
    val_idx = np.flatnonzero(dev_mask.to_numpy() & (df["fold"].astype(int).to_numpy() == fold))
    train_idx = np.flatnonzero(dev_mask.to_numpy() & (df["fold"].astype(int).to_numpy() != fold))
    cond_idx = np.flatnonzero(df["split"].astype(str).str.lower() == "cond_holdout")
    if not len(val_idx) or not len(train_idx) or not len(cond_idx):
        raise ValueError(f"fold {fold} has train={len(train_idx)} val={len(val_idx)} cond_holdout={len(cond_idx)}")
    y_all = df["cls"].map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
    counts = np.bincount(y_all[train_idx], minlength=len(classes)).astype(np.float64)
    weights = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    model = make_model(timm, device, pretrained=not args.dry_run)
    # A retry in make_model keeps dry-run offline-safe but the normal Kaggle
    # path remains pretrained EfficientNet-B0.
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = max(1, math.ceil(len(train_idx) / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=max(1, args.epochs * steps), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epoch_logs: list[dict[str, float]] = []
    train_rng = np.random.default_rng(args.seed + fold)
    for epoch in range(args.epochs):
        model.train(); order = train_rng.permutation(train_idx); loss_total = 0.0; seen = 0; t0 = time.perf_counter()
        for start_i in range(0, len(order), args.batch_size):
            ids = order[start_i:start_i+args.batch_size]
            raw = jpeg_cpu(X[ids], args.seed + fold*1000 + epoch*100 + start_i)
            xb = make_input(torch, raw, device)
            xb = augment_gpu(torch, xb, args.seed + fold*10000 + epoch*100 + start_i, args.img)
            yb = torch.from_numpy(y_all[ids]).to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(torch, device): loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_total += float(loss.detach().cpu()); seen += len(ids)
        img_s = seen / max(1e-9, time.perf_counter() - t0)
        print(f"fold={fold} epoch={epoch+1}/{args.epochs} loss={loss_total/max(1,steps):.5f} img/s={img_s:.2f}", flush=True)
        epoch_logs.append({"epoch": float(epoch + 1), "loss": float(loss_total/max(1,steps)), "img_s": float(img_s)})
    val_logits, val_emb = predict(model, X, val_idx, device, args.batch_size, args.img, torch, True)
    cond_logits, cond_emb = predict(model, X, cond_idx, device, args.batch_size, args.img, torch, True)
    y_val, y_cond = y_all[val_idx], y_all[cond_idx]
    val_pred = val_logits.argmax(1)
    print(f"fold={fold} val_acc={float(np.mean(val_pred==y_val)):.5f} val_macro_f1={macro_f1(y_val,val_pred,len(classes)):.5f}", flush=True)
    unit = [Path(str(x)).stem for x in df["file"].tolist()]
    save_prediction_table(out/f"oof_fold{fold}.parquet", [unit[i] for i in val_idx], y_val, val_logits)
    save_prediction_table(out/f"condholdout_fold{fold}.parquet", [unit[i] for i in cond_idx], y_cond, cond_logits)
    rng = np.random.default_rng(args.seed + 50000 + fold)
    shift_names = [SHIFT_NAMES[int(rng.integers(0, len(SHIFT_NAMES)))] for _ in val_idx]
    shifted = np.stack([stress_array(X[i], name, args.img, args.seed + j) for j, (i, name) in enumerate(zip(val_idx, shift_names))])
    shift_logits, _ = predict(model, shifted, np.arange(len(shifted)), device, args.batch_size, args.img, torch)
    save_prediction_table(out/f"oof_shift_fold{fold}.parquet", [unit[i] for i in val_idx], y_val, shift_logits)
    np.save(out/f"oof_fold{fold}.npy", np.asarray(val_emb, dtype=np.float16))
    np.save(out/f"condholdout_fold{fold}.npy", np.asarray(cond_emb, dtype=np.float16))
    bank_logits, bank_emb = predict(model, X, train_idx, device, args.batch_size, args.img, torch, True)
    del bank_logits
    np.save(out/f"bank_fold{fold}.npy", np.asarray(bank_emb, dtype=np.float16))
    state = {k: (v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()) for k, v in model.state_dict().items()}
    torch.save(state, out/f"fold{fold}.pt")
    stress_eval(model, X, val_idx, y_val, device, args.batch_size, args.img, torch, out/f"stress_fold{fold}.csv", args.seed + 70000 + fold)
    record = {"fold": fold, "wall_seconds": time.perf_counter()-start, "epochs": epoch_logs, "val_acc": float(np.mean(val_pred==y_val)), "val_macro_f1": macro_f1(y_val,val_pred,len(classes)), "train_n": int(len(train_idx)), "val_n": int(len(val_idx)), "cond_holdout_n": int(len(cond_idx)), "img_s": float(np.mean([x["img_s"] for x in epoch_logs])), "script_sha256": script_sha256()}
    (out/f"log_fold{fold}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"DONE fold {fold}", flush=True)
    print("files=" + ", ".join(p.name for p in outputs), flush=True)


def launch_two_gpus(args: argparse.Namespace) -> int:
    script = str(Path(__file__).resolve())
    common = [sys.executable, script, "--data-root", args.data_root, "--splits-csv", args.splits_csv, "--out", args.out, "--epochs", str(args.epochs), "--img", str(args.img), "--batch-size", str(args.batch_size)]
    if args.dry_run: common.append("--dry-run")
    jobs = [("gpu0", ["0,2,4"], "0"), ("gpu1", ["1,3"], "1")]
    processes = []
    for label, folds, gpu in jobs:
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
        command = common + ["--folds", folds[0], "--gpu", "0"]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        processes.append((label, proc))
    def tail(label: str, proc: Any) -> None:
        assert proc.stdout is not None
        for line in proc.stdout: print(f"[{label}] {line}", end="", flush=True)
    threads = [threading.Thread(target=tail, args=item, daemon=True) for item in processes]
    for thread in threads: thread.start()
    codes = [proc.wait() for _, proc in processes]
    for thread in threads: thread.join(timeout=2)
    return max(codes)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.launch_two_gpus:
        return launch_two_gpus(args)
    args.epochs = 1 if args.dry_run else args.epochs
    args.img = int(args.img)
    folds = [int(x) for x in args.folds.split(",") if x.strip()] if args.folds else ([0, 1] if args.dry_run else FOLDS)
    seed_everything(args.seed)
    try:
        import torch
        import timm
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("torch, timm, and pandas are required; no packages were installed by this script") from exc
    if args.gpu.lower() in {"cpu", "-1", "none"} or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if args.gpu.isdigit() else "cuda")
    print_banner(str(device))
    split_path = find_splits_csv(args.splits_csv)
    df = read_metadata(split_path)
    classes = sorted(df["cls"].astype(str).unique().tolist())
    if len(classes) != 5:
        raise ValueError(f"expected five classes, got {classes}")
    if args.dry_run:
        # Exactly 40 dev examples per class where available, with a small
        # balanced cond_holdout slice so the two-fold run stays CPU-friendly.
        dev = df["split"].astype(str).str.lower() == "dev"
        keep = np.zeros(len(df), dtype=bool)
        rng = np.random.default_rng(args.seed)
        for cls in classes:
            cls_mask = df["cls"].astype(str).to_numpy() == cls
            dev_candidates = np.flatnonzero(dev.to_numpy() & cls_mask)
            chosen_dev = []
            for fold_id in FOLDS:
                fold_candidates = dev_candidates[df.iloc[dev_candidates]["fold"].astype(int).to_numpy() == fold_id]
                chosen_dev.extend(rng.choice(fold_candidates, size=min(8, len(fold_candidates)), replace=False).tolist())
            if len(chosen_dev) < 40:
                remainder = np.setdiff1d(dev_candidates, np.asarray(chosen_dev, dtype=int), assume_unique=False)
                chosen_dev.extend(rng.choice(remainder, size=min(40-len(chosen_dev), len(remainder)), replace=False).tolist())
            keep[np.asarray(chosen_dev, dtype=int)] = True
            cond_candidates = np.flatnonzero(~dev.to_numpy() & (df["split"].astype(str).str.lower().to_numpy() == "cond_holdout") & cls_mask)
            keep[rng.choice(cond_candidates, size=min(8, len(cond_candidates)), replace=False)] = True
        df = df.loc[keep].reset_index(drop=True)
        print(f"DRY_RUN subset: {len(df)} rows ({sum(df.split.str.lower()=='cond_holdout')} cond_holdout)", flush=True)
    root = find_data_root(args.data_root, split_path, str(df["file"].iloc[0]))
    X = load_images(df, root, args.img, args.dry_run)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        try:
            run_fold(fold, df, X, out, args, device, torch, timm, classes)
        except Exception as exc:
            print(f"ERROR fold {fold}: {type(exc).__name__}: {exc}", flush=True)
            import traceback; traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
