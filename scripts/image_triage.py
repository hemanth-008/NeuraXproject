r"""
scripts/image_triage.py

Run: .venv\Scripts\python.exe scripts\image_triage.py "C:\NeuraxHack\data\images\train"
Needs: pillow numpy pandas scipy scikit-learn  (all but pillow are already in .venv)

Read-only. Writes artifacts/image_features.csv, artifacts/class_grid.jpg and
artifacts/IMAGE_TRIAGE.md, and prints the same summary.

Checks: per-class brightness/contrast stats, whether trivial global stats can
predict the class (shortcut test), file-number gaps and drift, and
near-duplicate / shared-background structure.
"""
import re, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BOX = getattr(Image, "Resampling", Image).BOX
NUM = re.compile(r"_(\d+)\.png$", re.I)


def features(arr):
    lap = 4 * arr[1:-1, 1:-1] - arr[:-2, 1:-1] - arr[2:, 1:-1] - arr[1:-1, :-2] - arr[1:-1, 2:]
    return dict(mean=arr.mean(), std=arr.std(), p1=np.percentile(arr, 1),
                p99=np.percentile(arr, 99), lap_var=lap.var())


def main(root):
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    rows, thumbs = [], []
    for c in classes:
        for p in sorted((root / c).glob("*.png")):
            im = Image.open(p).convert("L")
            arr = np.asarray(im, dtype=np.float32)
            m = NUM.search(p.name)
            rows.append(dict(cls=c, file=p.name, idx=int(m.group(1)) if m else -1, **features(arr)))
            thumbs.append(np.asarray(im.resize((32, 32), BOX), dtype=np.float32).ravel())
    df = pd.DataFrame(rows)
    df.to_csv("artifacts/image_features.csv", index=False)
    out = [f"# IMAGE TRIAGE\n\n{len(df)} images, classes: {', '.join(classes)}\n"]

    # 1. per-class stats
    g = df.groupby("cls")[["mean", "std", "p1", "p99", "lap_var"]].mean()
    g["mean_sd"] = df.groupby("cls")["mean"].std()
    out.append("## Per-class averages (mean_sd = spread of image brightness inside the class)\n```\n"
               + g.to_string(float_format=lambda x: f"{x:.1f}") + "\n```\n")

    # 2. shortcut test: can trivial global stats predict the label?
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    out.append("## Shortcut test (5-fold CV, gradient boosting on global stats only)\n")
    is_def = (df["cls"] != "normal").astype(int).values if "normal" in classes else None
    for name, cols in [("brightness only", ["mean"]), ("all 5 global stats", ["mean", "std", "p1", "p99", "lap_var"])]:
        X = df[cols].values
        pr = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, df["cls"], cv=skf, method="predict_proba")
        acc = accuracy_score(df["cls"], np.array(sorted(classes))[pr.argmax(1)])
        line = f"- {name}: {len(classes)}-class accuracy {acc:.3f} (chance {1/len(classes):.3f})"
        if is_def is not None:
            pd_ = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, is_def, cv=skf, method="predict_proba")[:, 1]
            line += f"; normal-vs-defect AUROC {roc_auc_score(is_def, pd_):.3f} (chance 0.500)"
        out.append(line)
    out.append("")

    # 3. file numbering: gaps and drift
    out.append("## File numbering per class\n```")
    for c, d in df.groupby("cls"):
        lo, hi, n = d.idx.min(), d.idx.max(), len(d)
        rho, pv = spearmanr(d.idx, d["mean"])
        out.append(f"{c:8s} n={n} index {lo}..{hi} missing_in_range={hi - lo + 1 - n}  brightness-vs-index rho={rho:+.3f} (p={pv:.2g})")
    out.append("```\n")

    # 4. near-duplicates / shared backgrounds (cosine on mean-centred 32x32 thumbnails)
    T = np.stack(thumbs); T -= T.mean(1, keepdims=True)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)
    cls_arr = df["cls"].values
    best = np.zeros(len(T)); same = np.zeros(len(T), bool)
    for i in range(0, len(T), 1000):
        S = T[i:i + 1000] @ T.T
        S[np.arange(len(S)), np.arange(i, i + len(S))] = -1
        j = S.argmax(1); best[i:i + len(S)] = S.max(1); same[i:i + len(S)] = cls_arr[j] == cls_arr[i:i + len(S)]
    out.append("## Nearest-neighbour similarity (cosine, 32x32 thumbnails)\n```")
    out.append("class     n>=0.98  n>=0.995  of those, nearest is same class")
    for c in classes:
        m = cls_arr == c
        a, b = (best[m] >= 0.98), (best[m] >= 0.995)
        out.append(f"{c:8s} {a.sum():7d} {b.sum():9d}  {int(same[m][a].sum())}/{int(a.sum())}")
    out.append("```\n")

    # 5. grid: 10 random images per class, one row per class
    t = 160
    grid = Image.new("L", (10 * t, len(classes) * t), 255)
    rnd = random.Random(3)
    for r, c in enumerate(classes):
        for k, f in enumerate(rnd.sample(list(df[df.cls == c].file), 10)):
            im = Image.open(root / c / f).convert("L"); im.thumbnail((t, t))
            grid.paste(im, (k * t, r * t))
        d = ImageDraw.Draw(grid); d.rectangle((0, r * t, 70, r * t + 12), fill=0); d.text((2, r * t), c, fill=255)
    grid.save("artifacts/class_grid.jpg", quality=88)

    text = "\n".join(out)
    Path("artifacts/IMAGE_TRIAGE.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('usage: image_triage.py "<path to image dataset>"')
    Path("artifacts").mkdir(exist_ok=True)
    main(sys.argv[1])
