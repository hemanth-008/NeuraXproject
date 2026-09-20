"""Populate heatmap fields for a curated, non-holdout demo subset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


LOCKED = {"holdout", "locked", "locked_holdout", "locked-holdout"}


def _assert_no_holdout(frame):
    if "source_split" in frame.columns:
        assert not bool(frame["source_split"].astype(str).str.lower().isin(LOCKED).any()), "holdout rows reached demo heatmap builder"
    return frame


def _resolve(value, root: Path) -> Path | None:
    path = Path(str(value))
    for candidate in (path, root / path, root / "data" / path):
        if candidate.exists():
            return candidate.resolve()
    return None


def build(predictions_path: Path, repo_root: Path, artifacts_dir: Path) -> int:
    import pandas as pd

    frame = _assert_no_holdout(pd.read_parquet(predictions_path))
    if "source_split" in frame.columns:
        frame = frame[~frame["source_split"].astype(str).str.lower().isin(LOCKED)]
    selected: list[int] = []
    for value in ("UNCERTAIN", "REVIEW", "NOVEL"):
        if "flag" in frame.columns:
            selected.extend(frame.index[frame["flag"].astype(str).str.upper() == value].tolist())
    if "pred_family" in frame.columns:
        for _, group in frame.groupby("pred_family", sort=True):
            selected.extend(group.head(20).index.tolist())
    for stem in ("crack_00011", "rust_00187", "rust_00891"):
        if "unit_id" in frame.columns:
            selected.extend(frame.index[frame["unit_id"].astype(str) == stem].tolist())
        if "image_path" in frame.columns:
            selected.extend(frame.index[frame["image_path"].astype(str).str.contains(stem, regex=False)].tolist())
    selected = list(dict.fromkeys(index for index in selected if index in frame.index))
    if not selected:
        raise RuntimeError("no curated non-holdout rows found")
    from src.vision.infer import InferenceEngine

    engine = InferenceEngine(artifacts_dir, repo_root=repo_root)
    heatmap_dir = artifacts_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    backup = predictions_path.with_suffix(predictions_path.suffix + ".bak")
    shutil.copy2(predictions_path, backup)
    # Only these three columns are changed below; every other parquet column
    # is copied from the original frame untouched.
    for index in selected:
        row = frame.loc[index]
        image = _resolve(row.get("image_path", row.get("file")), repo_root)
        if image is None:
            continue
        result = engine.predict_file(image, heatmap_dir)
        frame.loc[index, "heatmap_path"] = result["heatmap_path"]
        frame.loc[index, "box_xyxy"] = result["box_xyxy"]
        frame.loc[index, "localization_basis"] = result["localization_basis"]
    # Merge updates back into the complete original table, preserving rows and
    # all non-localization fields exactly.
    original = pd.read_parquet(backup)
    for column in ("heatmap_path", "box_xyxy", "localization_basis"):
        if column not in original.columns:
            original[column] = None
        for index in selected:
            if index in original.index and index in frame.index:
                original.loc[index, column] = frame.loc[index, column]
    original.to_parquet(predictions_path, index=False)
    print(f"demo_heatmaps rows={len(selected)} backup={backup}")
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/predictions.parquet")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    return 0 if build(Path(args.predictions), root, Path(args.artifacts_dir)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
