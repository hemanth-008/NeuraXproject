r"""
scripts/make_splits.py

Run: .venv\Scripts\python.exe scripts\make_splits.py "C:\NeuraxHack\data\images\train"
Writes artifacts/splits.csv. Deterministic (fixed seeds). Read-only on the images.

Split protocol (no batch / lighting / variant metadata exists, so we build groups ourselves):
- group:        images whose 32x32 thumbnails have cosine >= 0.995 share a group and are never split apart.
- cond_holdout: per class, the darkest 7.5% and brightest 7.5% images (by mean pixel value).
                Never used for training or model selection; tests unseen lighting conditions.
- holdout:      about 1/7 of the rest, stratified by class, grouped. LOCKED: at most 3 evaluations, ever.
- dev:          everything else, with fold 0-4 (stratified by class, grouped) for cross-validation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

BOX = getattr(Image, "Resampling", Image).BOX


def main(root):
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    recs, thumbs = [], []
    for c in classes:
        for p in sorted((root / c).glob("*.png")):
            im = Image.open(p).convert("L")
            recs.append((f"{c}/{p.name}", c, float(np.asarray(im, dtype=np.float32).mean())))
            thumbs.append(np.asarray(im.resize((32, 32), BOX), dtype=np.float32).ravel())
    df = pd.DataFrame(recs, columns=["file", "cls", "mean_brightness"])
    T = np.stack(thumbs)
    T -= T.mean(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-6

    # near-duplicate groups (union-find over pairs with cosine >= 0.995)
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(0, len(T), 1000):
        S = T[i:i + 1000] @ T.T
        ii, jj = np.where(S >= 0.995)
        for a, b in zip(ii + i, jj):
            if a < b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    df["group"] = [find(i) for i in range(len(df))]

    # condition holdout: brightness tails within each class
    g = df.groupby("cls")["mean_brightness"]
    tail = (df["mean_brightness"] <= g.transform(lambda s: s.quantile(0.075))) | \
           (df["mean_brightness"] >= g.transform(lambda s: s.quantile(0.925)))
    df["split"] = np.where(df["group"].isin(set(df.loc[tail, "group"])), "cond_holdout", "rest")

    # locked holdout from the rest
    rest = df[df.split == "rest"]
    _, ho = next(StratifiedGroupKFold(7, shuffle=True, random_state=0).split(rest, rest["cls"], rest["group"]))
    df.loc[rest.index, "split"] = "dev"
    df.loc[rest.index[ho], "split"] = "holdout"

    # CV folds inside dev
    df["fold"] = -1
    dev = df[df.split == "dev"]
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=1).split(dev, dev["cls"], dev["group"])):
        df.loc[dev.index[va], "fold"] = k

    # leakage checks
    assert df.groupby("group")["split"].nunique().max() == 1, "a group spans two splits"
    assert df[df.split == "dev"].groupby("group")["fold"].nunique().max() == 1, "a group spans two folds"

    Path("artifacts").mkdir(exist_ok=True)
    df.to_csv("artifacts/splits.csv", index=False)
    sizes = df.groupby("group").size()
    print(f"{len(df)} images; groups with >1 image: {(sizes > 1).sum()} (largest group: {sizes.max()})")
    if sizes.max() > 50:
        print("WARNING: a very large group formed; the 0.995 threshold may be chaining unrelated images.")
    print("\nImages per split x class:")
    print(pd.crosstab(df["split"], df["cls"], margins=True).to_string())
    print("\nMean brightness by split:")
    print(df.groupby("split")["mean_brightness"].agg(["mean", "min", "max"]).round(1).to_string())
    print("\nDev folds x class:")
    print(pd.crosstab(dev.assign(fold=df.loc[dev.index, "fold"])["fold"], dev["cls"]).to_string())
    print("\nWrote artifacts/splits.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('usage: make_splits.py "<path to image dataset>"')
    main(sys.argv[1])
