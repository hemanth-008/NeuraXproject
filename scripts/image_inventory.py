r"""
scripts/image_inventory.py

Run: .venv\Scripts\python.exe scripts\image_inventory.py "<path to image dataset>"
Needs: pillow  (.venv\Scripts\pip.exe install pillow)

Read-only. Writes artifacts/IMAGE_INVENTORY.md and artifacts/sample_grid.jpg
so we can see the label type, classes, volume, metadata and image sizes
before choosing the vision method.
"""
import sys, random, hashlib, collections
from pathlib import Path
from PIL import Image

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LAB = {".json", ".xml", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".parquet"}


def head(path, n_bytes=1500):
    try:
        return path.read_bytes()[:n_bytes].decode("utf-8", errors="replace")
    except Exception as e:
        return f"(could not read: {e})"


def main(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in IMG]
    labs = [p for p in files if p.suffix.lower() in LAB]
    out = [f"# IMAGE INVENTORY\n\nRoot: `{root}`\n",
           f"- files: {len(files)}, images: {len(imgs)}, label-like files: {len(labs)}\n"]

    ext = collections.Counter(p.suffix.lower() or "(none)" for p in files)
    out.append("## Extensions\n" + "\n".join(f"- `{k}`: {v}" for k, v in ext.most_common()) + "\n")

    # image counts per folder, depth <= 4 below root
    per_dir = collections.Counter()
    for p in imgs:
        rel = p.relative_to(root).parent.parts[:4]
        per_dir["/".join(rel) or "."] += 1
    out.append("## Images per folder (first 4 levels)\n")
    for k, v in sorted(per_dir.items())[:80]:
        out.append(f"- `{k}`: {v}")
    if len(per_dir) > 80:
        out.append(f"- ... {len(per_dir) - 80} more folders")
    out.append("")

    # example file names per folder (naming convention / possible batch or condition tokens)
    out.append("## Example file names (first 5 per top-level folder)\n")
    by_top = collections.defaultdict(list)
    for p in imgs:
        by_top[p.relative_to(root).parts[0]].append(p.name)
    for k, v in sorted(by_top.items())[:30]:
        out.append(f"- `{k}`: " + ", ".join(f"`{n}`" for n in sorted(v)[:5]))
    out.append("")

    # sizes / modes on a sample
    sample = random.Random(0).sample(imgs, min(300, len(imgs)))
    sizes = collections.Counter()
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes[(im.width, im.height, im.mode)] += 1
        except Exception:
            sizes[("unreadable", "", "")] += 1
    out.append(f"## Image sizes / modes (sample of {len(sample)})\n")
    for (w, h, m), c in sizes.most_common(15):
        out.append(f"- {w} x {h}, {m}: {c}")
    out.append("")

    # exact duplicates
    if imgs:
        pool = imgs if len(imgs) <= 20000 else random.Random(1).sample(imgs, 20000)
        seen = collections.defaultdict(list)
        for p in pool:
            seen[hashlib.md5(p.read_bytes()).hexdigest()].append(p)
        dups = [v for v in seen.values() if len(v) > 1]
        out.append(f"## Exact duplicates (checked {len(pool)} files)\n- duplicate groups: {len(dups)}\n")

    # label files
    out.append("## Label-like files (first 20; first ~1500 bytes each)\n")
    for p in sorted(labs, key=lambda x: -x.stat().st_size)[:20]:
        out.append(f"### `{p.relative_to(root)}` ({p.stat().st_size} bytes)\n```\n{head(p)}\n```\n")

    art = Path("artifacts"); art.mkdir(exist_ok=True)
    (art / "IMAGE_INVENTORY.md").write_text("\n".join(out), encoding="utf-8")

    # contact sheet of 50 random images
    pick = random.Random(2).sample(imgs, min(50, len(imgs)))
    if pick:
        t = 160; cols = 10; rows = (len(pick) + cols - 1) // cols
        grid = Image.new("RGB", (cols * t, rows * t), "white")
        for i, p in enumerate(pick):
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB"); im.thumbnail((t, t))
                    grid.paste(im, ((i % cols) * t, (i // cols) * t))
            except Exception:
                pass
        grid.save(art / "sample_grid.jpg", quality=85)
    print(f"Wrote {art / 'IMAGE_INVENTORY.md'} and {art / 'sample_grid.jpg'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('usage: image_inventory.py "<path to image dataset>"')
    main(sys.argv[1])
