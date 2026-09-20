"""Leave-one-family-out novelty evaluation for the QualityTwin image data.

The script is intentionally self-contained so it can be embedded in a Kaggle
notebook.  It trains one EfficientNet-B0 per held-out family on fold-0's
known-class training rows, then evaluates max-softmax, energy, and kNN cosine
distance novelty scores on fold-0 validation plus ``cond_holdout`` rows.
"""

from __future__ import annotations

import argparse
import csv
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
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DATA_ROOT = os.environ.get("QUALITYTWIN_DATA_ROOT", "")
SPLITS_CSV = os.environ.get("QUALITYTWIN_SPLITS_CSV", "")
OUT = os.environ.get("QUALITYTWIN_OUT", "/kaggle/working")
EPOCHS = 8
IMG = 256
SEED = 0
GPU = os.environ.get("QUALITYTWIN_GPU", "auto")
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-2
FAMILIES = ["crack", "hole", "rust", "scratch"]
ARCH = "efficientnet_b0"
LOCKED_WORDS = {"holdout", "locked", "locked_holdout", "locked-holdout"}


def script_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return hashlib.sha256(b"embedded-lofo_novelty").hexdigest()


def version_of(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def print_banner(device: str) -> None:
    print("=== QualityTwin lofo_novelty startup ===", flush=True)
    print(f"python={platform.python_version()} torch={version_of('torch')} timm={version_of('timm')}", flush=True)
    try:
        import torch
        print(f"gpu_count={torch.cuda.device_count()} cuda_available={torch.cuda.is_available()} device={device}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"gpu[{i}]={torch.cuda.get_device_name(i)}", flush=True)
    except Exception as exc:
        print(f"gpu_count=unavailable ({type(exc).__name__}) device={device}", flush=True)
    print(f"script_sha256={script_sha256()}", flush=True)
    print("data_policy=dev fold-0/other-dev folds + cond_holdout only; locked holdout excluded", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--splits-csv", default=SPLITS_CSV)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--families", default=",".join(FAMILIES), help="comma-separated held-out families")
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--img", type=int, default=IMG)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
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


def row_is_locked(row: dict[str, str]) -> bool:
    for value in row.values():
        text = str(value).strip().lower()
        if text in LOCKED_WORDS or ("locked" in text and "cond" not in text):
            return True
    return str(row.get("split", "")).strip().lower() in {"holdout", "locked_holdout"}


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


def read_metadata(path: Path) -> Any:
    """Read only safe metadata rows; locked rows never enter the DataFrame."""
    import pandas as pd

    safe_rows: list[dict[str, str]] = []
    locked_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            if row_is_locked(row):
                locked_count += 1
                continue
            safe_rows.append(row)
    required = {"file", "cls", "split", "fold"}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"splits.csv missing columns {sorted(missing)}; columns={fields}")
    df = pd.DataFrame.from_records(safe_rows, columns=fields)
    assert not bool(df.apply(lambda row: row_is_locked(row.to_dict()), axis=1).any()), "locked-holdout rows reached the trainer"
    assert not bool(df["split"].astype(str).str.strip().str.lower().isin({"holdout", "locked_holdout"}).any()), "locked holdout rows reached the trainer"
    print(f"splits columns={fields} rows_after_locked_filter={len(df)} dropped_locked={locked_count}", flush=True)
    return df.reset_index(drop=True)


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
        if (base / first_file).exists():
            return base.resolve()
        for path in base.rglob(Path(first_file).name):
            candidate = path
            for _ in Path(first_file).parts:
                candidate = candidate.parent
            if (candidate / first_file).exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not locate image {first_file}; pass --data-root")


def load_images(df: Any, root: Path, img: int) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    rs = getattr(Image, "Resampling", Image).BILINEAR

    def load_one(rel: str) -> np.ndarray:
        with Image.open(root / str(rel)) as image:
            return np.asarray(image.convert("L").resize((img, img), rs), dtype=np.uint8)

    t0 = time.perf_counter()
    workers = min(8, max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = np.stack(list(pool.map(load_one, df["file"].tolist())))
    print(f"preloaded {len(images)} images shape={images.shape} seconds={time.perf_counter()-t0:.2f}", flush=True)
    return images


def make_input(torch: Any, x_u8: np.ndarray, device: Any) -> Any:
    x = torch.from_numpy(np.ascontiguousarray(x_u8)).to(device=device, dtype=torch.float32).div_(255.0)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.449, 0.449, 0.449], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.226, 0.226, 0.226], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _affine(torch: Any, x: Any, angles: Any, scales: Any) -> Any:
    n = x.shape[0]
    rad = angles * math.pi / 180.0
    c, s = torch.cos(rad) / scales, torch.sin(rad) / scales
    theta = torch.zeros((n, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = c, s
    theta[:, 1, 0], theta[:, 1, 1] = -s, c
    grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def augment_gpu(torch: Any, x: Any, seed: int, img: int) -> Any:
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
        a[rot90] = torch.rot90(a[rot90], 1, (-2, -1))
    flip_h = torch.rand(len(idx), device=x.device, generator=g) < 0.25
    flip_v = torch.rand(len(idx), device=x.device, generator=g) < 0.25
    if bool(flip_h.any()):
        a[flip_h] = a[flip_h].flip(-1)
    if bool(flip_v.any()):
        a[flip_v] = a[flip_v].flip(-2)
    rotate = torch.rand(len(idx), device=x.device, generator=g) < 0.20
    if bool(rotate.any()):
        q = rotate.nonzero(as_tuple=False).flatten()
        angles = torch.empty(len(q), device=x.device).uniform_(-180, 180, generator=g)
        a[q] = _affine(torch, a[q], angles, torch.ones(len(q), device=x.device))
    crop = torch.rand(len(idx), device=x.device, generator=g) < 0.18
    if bool(crop.any()):
        q = crop.nonzero(as_tuple=False).flatten()
        scale = torch.empty(len(q), device=x.device).uniform_(0.6, 1.0, generator=g)
        a[q] = _affine(torch, a[q], torch.zeros(len(q), device=x.device), scale)
    b = torch.empty(len(idx), 1, 1, 1, device=x.device).uniform_(0.8, 1.2, generator=g)
    c = torch.empty(len(idx), 1, 1, 1, device=x.device).uniform_(0.8, 1.2, generator=g)
    gamma = torch.empty(len(idx), 1, 1, 1, device=x.device).uniform_(0.75, 1.35, generator=g)
    mean = a.mean((-2, -1), keepdim=True)
    a = ((a - mean) * c + mean) * b
    a = a.clamp(0, 1).pow(gamma)
    noise = torch.empty_like(a).normal_(0, 0.035, generator=g)
    a = torch.where(torch.rand(len(idx), 1, 1, 1, device=x.device, generator=g) < 0.28, a + noise, a)
    small = torch.rand(len(idx), device=x.device, generator=g) < 0.18
    if bool(small.any()):
        q = torch.nn.functional.interpolate(a[small], size=(max(8, img // 4), max(8, img // 4)), mode="area")
        a[small] = torch.nn.functional.interpolate(q, size=(img, img), mode="bilinear", align_corners=False)
    z[idx] = a.clamp(0, 1)
    return z


def amp_context(torch: Any, device: Any):
    return torch.autocast("cuda", dtype=torch.float16) if getattr(device, "type", str(device)) == "cuda" else torch.autocast("cpu", enabled=False)


def make_model(timm: Any, device: Any, n_classes: int, pretrained: bool) -> Any:
    try:
        model = timm.create_model(ARCH, pretrained=pretrained, num_classes=n_classes)
    except Exception:
        if not pretrained:
            raise
        print("pretrained weights unavailable; retrying without download", flush=True)
        model = timm.create_model(ARCH, pretrained=False, num_classes=n_classes)
    return model.to(device)


def forward_with_embedding(model: Any, x: Any) -> tuple[Any, Any]:
    features = model.forward_features(x)
    embedding = model.forward_head(features, pre_logits=True)
    if embedding.ndim > 2:
        embedding = embedding.flatten(1)
    logits = model.get_classifier()(embedding)
    return logits, embedding


def predict(model: Any, X: np.ndarray, indices: np.ndarray, device: Any, batch_size: int, torch: Any) -> tuple[np.ndarray, np.ndarray]:
    logits_parts: list[np.ndarray] = []
    emb_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            xb = make_input(torch, X[indices[start:start + batch_size]], device)
            with amp_context(torch, device):
                logits, emb = forward_with_embedding(model, xb)
            logits_parts.append(logits.float().cpu().numpy())
            emb_parts.append(emb.float().cpu().numpy())
    return np.concatenate(logits_parts), np.concatenate(emb_parts)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if not pos or not neg:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def threshold_at_fpr(scores: np.ndarray, y: np.ndarray, target: float = 0.05) -> tuple[float, float, float]:
    known = np.asarray(scores[y == 0], dtype=np.float64)
    positives = np.asarray(scores[y == 1], dtype=np.float64)
    if not len(known) or not len(positives):
        return float("inf"), float("nan"), float("nan")
    # Strict acceptance (< threshold) makes the measured FPR explicit and
    # avoids silently exceeding the requested operating point on ties.
    candidates = np.unique(np.concatenate(([known.max()], known)))
    best = (float("inf"), 0.0, float("inf"))
    for threshold in candidates:
        fpr = float(np.mean(known >= threshold))
        recall = float(np.mean(positives >= threshold))
        if fpr <= target + 1e-12 and (recall > best[1] or (recall == best[1] and threshold < best[2])):
            best = (float(threshold), recall, fpr)
    if not np.isfinite(best[2]):
        threshold = float(np.nextafter(known.max(), np.inf))
        return threshold, 0.0, 0.0
    return best


def classifier_accuracy(pred: np.ndarray, true_known_class: np.ndarray, y_novel: np.ndarray, accepted: np.ndarray) -> float:
    if not bool(accepted.any()):
        return 0.0
    correct = (y_novel[accepted] == 0) & (pred[accepted] == true_known_class[accepted])
    return float(np.mean(correct))


def risk_coverage(scores: np.ndarray, pred: np.ndarray, true_known_class: np.ndarray, y_novel: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for coverage_target in np.linspace(0.05, 1.0, 20):
        threshold = float(np.quantile(scores, coverage_target, method="linear"))
        accepted = scores < threshold
        if not bool(accepted.any()):
            accepted[np.argmin(scores)] = True
        accuracy = classifier_accuracy(pred, true_known_class, y_novel, accepted)
        rows.append({"threshold": threshold, "coverage": float(np.mean(accepted)), "risk": float(1.0 - accuracy) if np.isfinite(accuracy) else 1.0, "n_accepted": int(accepted.sum()), "accuracy": accuracy})
    return rows


def detector_metrics(name: str, scores: np.ndarray, y_novel: np.ndarray, pred: np.ndarray, true_known_class: np.ndarray) -> dict[str, Any]:
    threshold, recall, fpr = threshold_at_fpr(scores, y_novel)
    accepted = scores < threshold
    return {
        "detector": name,
        "auroc": auroc(y_novel, scores),
        "threshold_at_5pct_fpr": threshold,
        "measured_fpr": fpr,
        "recall_at_5pct_fpr": recall,
        "accepted_accuracy": classifier_accuracy(pred, true_known_class, y_novel, accepted),
        "accepted_coverage": float(np.mean(accepted)),
        "accepted_n": int(accepted.sum()),
        "risk_coverage": risk_coverage(scores, pred, true_known_class, y_novel),
    }


def train_family(family: str, df: Any, X: np.ndarray, out: Path, args: argparse.Namespace, device: Any, torch: Any, timm: Any) -> dict[str, Any]:
    result_path = out / f"lofo_{family}.json"
    if result_path.exists():
        print(f"SKIP {family}: {result_path.name} already exists", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    classes = sorted(str(x) for x in df["cls"].unique())
    known_classes = [c for c in classes if c != family]
    if len(known_classes) != 4:
        raise ValueError(f"expected five classes including {family}, got {classes}")
    split = df["split"].astype(str).str.lower().to_numpy()
    folds = df["fold"].astype(int).to_numpy()
    cls = df["cls"].astype(str).to_numpy()
    train_idx = np.flatnonzero((split == "dev") & (folds != 0) & (cls != family))
    eval_idx = np.flatnonzero(((split == "dev") & (folds == 0)) | (split == "cond_holdout"))
    if not len(train_idx) or not len(eval_idx):
        raise ValueError(f"{family} train/eval rows are empty: {len(train_idx)}/{len(eval_idx)}")
    class_to_id = {name: i for i, name in enumerate(known_classes)}
    y_train = np.asarray([class_to_id[str(cls[i])] for i in train_idx], dtype=np.int64)
    y_eval_known = np.asarray([class_to_id.get(str(cls[i]), -1) for i in eval_idx], dtype=np.int64)
    y_novel = (cls[eval_idx] == family).astype(np.int64)
    checkpoint_path = out / f"lofo_{family}.pt"
    model = make_model(timm, device, len(known_classes), pretrained=not args.dry_run)
    counts = np.bincount(y_train, minlength=len(known_classes)).astype(np.float64)
    weights = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = max(1, math.ceil(len(train_idx) / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=max(1, args.epochs * steps), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    epoch_logs: list[dict[str, float]] = []
    if checkpoint_path.exists():
        try:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state.get("scaler", {}))
            start_epoch = int(state.get("epoch", 0))
            epoch_logs = list(state.get("epoch_logs", []))
            print(f"RESUME {family} from epoch={start_epoch}", flush=True)
        except Exception as exc:
            print(f"checkpoint unreadable for {family}; restarting ({type(exc).__name__}: {exc})", flush=True)
    train_rng = np.random.default_rng(args.seed + FAMILIES.index(family))
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = train_rng.permutation(train_idx)
        loss_total = 0.0
        seen = 0
        t0 = time.perf_counter()
        for start_i in range(0, len(order), args.batch_size):
            ids = order[start_i:start_i + args.batch_size]
            xb = make_input(torch, X[ids], device)
            xb = augment_gpu(torch, xb, args.seed + epoch * 1000 + start_i, args.img)
            yb = torch.from_numpy(np.asarray([class_to_id[str(cls[i])] for i in ids], dtype=np.int64)).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(torch, device):
                logits, _ = forward_with_embedding(model, xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_total += float(loss.detach().cpu())
            seen += len(ids)
        img_s = seen / max(1e-9, time.perf_counter() - t0)
        log = {"epoch": float(epoch + 1), "loss": float(loss_total / max(1, steps)), "img_s": float(img_s)}
        epoch_logs.append(log)
        print(f"family={family} epoch={epoch + 1}/{args.epochs} loss={log['loss']:.5f} img/s={img_s:.2f}", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1, "epoch_logs": epoch_logs, "family": family, "classes": known_classes, "script_sha256": script_sha256()}, checkpoint_path)
    train_logits, train_emb = predict(model, X, train_idx, device, args.batch_size, torch)
    eval_logits, eval_emb = predict(model, X, eval_idx, device, args.batch_size, torch)
    del train_logits
    probs = np.exp(eval_logits - eval_logits.max(axis=1, keepdims=True))
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    max_softmax = 1.0 - probs.max(axis=1)
    energy = -np.logaddexp.reduce(eval_logits, axis=1)
    bank = normalize_rows(train_emb)
    query = normalize_rows(eval_emb)
    similarities = query @ bank.T
    k = min(5, similarities.shape[1])
    knn_cosine = 1.0 - np.mean(np.partition(similarities, -k, axis=1)[:, -k:], axis=1)
    pred = eval_logits.argmax(axis=1)
    detectors = [detector_metrics("one_minus_max_softmax", max_softmax, y_novel, pred, y_eval_known), detector_metrics("energy", energy, y_novel, pred, y_eval_known), detector_metrics("knn_cosine_distance_k5", knn_cosine, y_novel, pred, y_eval_known)]
    unit_ids = [Path(str(x)).stem for x in df.iloc[eval_idx]["file"].tolist()]
    split_counts = {str(key): int(value) for key, value in zip(*np.unique(split[eval_idx], return_counts=True))}
    result: dict[str, Any] = {"family": family, "held_out_family": family, "known_classes": known_classes, "detectors": detectors, "eval": {"n": int(len(eval_idx)), "known_n": int(np.sum(y_novel == 0)), "novel_n": int(np.sum(y_novel == 1)), "split_counts": split_counts, "unit_ids": unit_ids}, "train_n": int(len(train_idx)), "epochs": epoch_logs, "wall_seconds": float(time.perf_counter() - started), "img_s_mean": float(np.mean([item["img_s"] for item in epoch_logs])) if epoch_logs else float("nan"), "script_sha256": script_sha256(), "assumptions": {"novelty_label": "family F is treated as novel against known classes; association, not proven causation", "data_scope": "fold-0 val + cond_holdout, with locked holdout excluded"}}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"DONE {family}: train_n={len(train_idx)} eval_n={len(eval_idx)} wall_seconds={result['wall_seconds']:.2f}", flush=True)
    return result


def aggregate(out: Path, families: list[str]) -> dict[str, Any]:
    results = {family: json.loads((out / f"lofo_{family}.json").read_text(encoding="utf-8")) for family in families}
    summary: dict[str, Any] = {}
    for detector in ("one_minus_max_softmax", "energy", "knn_cosine_distance_k5"):
        values = [float(next(row for row in result["detectors"] if row["detector"] == detector)["auroc"]) for result in results.values()]
        summary[detector] = {"mean_auroc": float(np.mean(values)), "min_auroc": float(np.min(values)), "max_auroc": float(np.max(values)), "n_families": len(values)}
    payload = {"families": results, "summary": summary, "families_completed": len(results), "families_requested": families, "script_sha256": script_sha256(), "assumptions": {"interpretation": "association, not proven causation", "scope": "locked holdout excluded"}}
    (out / "novelty_lofo.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def launch_two_gpus(args: argparse.Namespace) -> int:
    script = str(Path(__file__).resolve())
    requested = [x.strip() for x in args.families.split(",") if x.strip()]
    groups = [requested[:2], requested[2:]]
    processes: list[tuple[str, Any]] = []
    for gpu, group in enumerate(groups):
        if not group:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [sys.executable, script, "--data-root", args.data_root, "--splits-csv", args.splits_csv, "--out", args.out, "--families", ",".join(group), "--epochs", str(args.epochs), "--img", str(args.img), "--batch-size", str(args.batch_size), "--gpu", "0"]
        if args.dry_run:
            command.append("--dry-run")
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        processes.append((f"gpu{gpu}", proc))

    def tail(label: str, proc: Any) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{label}] {line}", end="", flush=True)

    threads = [threading.Thread(target=tail, args=item, daemon=True) for item in processes]
    for thread in threads:
        thread.start()
    codes = [proc.wait() for _, proc in processes]
    for thread in threads:
        thread.join(timeout=2)
    if max(codes, default=0) == 0:
        aggregate(Path(args.out), requested)
    return max(codes, default=0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested = [x.strip().lower() for x in args.families.split(",") if x.strip()]
    unknown = sorted(set(requested).difference(FAMILIES))
    if unknown:
        raise ValueError(f"unknown family names: {unknown}")
    if args.launch_two_gpus:
        return launch_two_gpus(args)
    if args.dry_run:
        args.epochs = 1
        args.img = min(args.img, 64)
        args.batch_size = min(args.batch_size, 16)
    seed_everything(args.seed)
    try:
        import torch
        import timm
    except Exception as exc:
        raise RuntimeError("torch, timm, pandas, and Pillow are required; no packages were installed by this script") from exc
    if args.gpu.lower() in {"cpu", "-1", "none"} or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if args.gpu.isdigit() else "cuda")
    print_banner(str(device))
    split_path = find_splits_csv(args.splits_csv)
    df = read_metadata(split_path)
    classes = sorted(str(x) for x in df["cls"].unique())
    if sorted(classes) != sorted(["crack", "hole", "normal", "rust", "scratch"]):
        raise ValueError(f"expected crack/hole/normal/rust/scratch, got {classes}")
    if args.dry_run:
        # Keep all folds represented while making the local CPU test tiny.
        split = df["split"].astype(str).str.lower().to_numpy()
        folds = df["fold"].astype(int).to_numpy()
        cls = df["cls"].astype(str).to_numpy()
        rng = np.random.default_rng(args.seed)
        chosen: list[int] = []
        for name in classes:
            for fold in range(5):
                candidates = np.flatnonzero((split == "dev") & (folds == fold) & (cls == name))
                chosen.extend(rng.choice(candidates, size=min(2, len(candidates)), replace=False).tolist())
            candidates = np.flatnonzero((split == "cond_holdout") & (cls == name))
            chosen.extend(rng.choice(candidates, size=min(2, len(candidates)), replace=False).tolist())
        df = df.iloc[np.unique(chosen)].reset_index(drop=True)
        print(f"DRY_RUN subset: {len(df)} safe rows", flush=True)
    root = find_data_root(args.data_root, split_path, str(df["file"].iloc[0]))
    X = load_images(df, root, args.img)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for family in requested:
        train_family(family, df, X, out, args, device, torch, timm)
    aggregate(out, requested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
