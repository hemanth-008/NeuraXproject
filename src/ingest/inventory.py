#!/usr/bin/env python3
"""
inventory.py — Scan the data folder and produce DATA_INVENTORY.md.

Usage:
    .venv\\Scripts\\python.exe src/ingest/inventory.py [DATA_FOLDER] [OUTPUT_FILE]

Defaults come from config.yaml in the repo root.
"""

import os
import sys
import pathlib
import textwrap

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Allow overrides via CLI; otherwise read config.yaml
if len(sys.argv) >= 2:
    DATA_FOLDER = pathlib.Path(sys.argv[1])
else:
    import yaml
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    DATA_FOLDER = pathlib.Path(cfg["data_folder"])

OUTPUT_FILE = pathlib.Path(sys.argv[2]) if len(sys.argv) >= 3 else REPO_ROOT / "DATA_INVENTORY.md"

# We need pandas — import after path setup so script can fail fast
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TABULAR_EXT = {".csv", ".xlsx", ".xls", ".json", ".txt", ".dat", ".parquet", ".tsv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
BINARY_SPECIAL = {".mat", ".doe", ".mdb", ".accdb", ".zip", ".gz", ".7z", ".rar"}
MAX_PREVIEW_ROWS = 3
NROWS_LIMIT = 10000  # for big CSVs: read first N rows for stats


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def read_tabular(path: pathlib.Path, nrows=None):
    """Read a tabular file into a DataFrame. Returns (df, truncated_flag)."""
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path, nrows=nrows, encoding_errors="replace")
            truncated = nrows is not None
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, nrows=nrows)
            truncated = nrows is not None
        elif ext == ".parquet":
            df = pd.read_parquet(path)
            if nrows and len(df) > nrows:
                df = df.head(nrows)
                truncated = True
            else:
                truncated = False
        elif ext == ".json":
            df = pd.read_json(path, lines=False, nrows=nrows)
            truncated = nrows is not None
        elif ext in (".txt", ".tsv", ".dat"):
            # try tab-separated first, fallback to comma
            df = pd.read_csv(path, sep="\t", nrows=nrows, encoding_errors="replace")
            if len(df.columns) <= 1:
                df = pd.read_csv(path, nrows=nrows, encoding_errors="replace")
            truncated = nrows is not None
        else:
            return None, False
        # drop fully-empty columns (trailing commas)
        df = df.dropna(axis=1, how="all")
        return df, truncated
    except Exception as e:
        return None, str(e)


def col_summary(series: pd.Series, max_top=5) -> str:
    """One-line summary for a column."""
    dtype = str(series.dtype)
    nulls = int(series.isna().sum())
    nuniq = int(series.nunique())
    if pd.api.types.is_numeric_dtype(series):
        mn = series.min()
        mx = series.max()
        me = series.mean()
        stats = f"min={mn:.4g}, max={mx:.4g}, mean={me:.4g}"
    else:
        top = series.dropna().value_counts().head(max_top)
        vals = ", ".join(f"{v}({c})" for v, c in top.items())
        stats = f"top: {vals}"
    return f"{dtype} | nulls={nulls} | unique={nuniq} | {stats}"


def full_shape(path: pathlib.Path) -> tuple:
    """Get full (rows, cols) for a CSV without loading it fully."""
    ext = path.suffix.lower()
    if ext == ".csv":
        # Count lines cheaply
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            ncols = len(header.split(","))
            nrows = sum(1 for _ in f)
        return nrows, ncols
    return None, None


# ---------------------------------------------------------------------------
# Walk data folder
# ---------------------------------------------------------------------------

all_files = []
for root, dirs, files in os.walk(DATA_FOLDER):
    for fname in files:
        fp = pathlib.Path(root) / fname
        all_files.append(fp)

all_files.sort()

# ---------------------------------------------------------------------------
# Classify files
# ---------------------------------------------------------------------------

tabular_files = [f for f in all_files if f.suffix.lower() in TABULAR_EXT]
image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXT]
binary_files = [f for f in all_files if f.suffix.lower() in BINARY_SPECIAL]
readme_files = [f for f in all_files if f.name.lower().startswith("readme")]
pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
other_files = [f for f in all_files if f not in tabular_files + image_files + binary_files + pdf_files]

# Extensions summary
ext_counts = {}
for f in all_files:
    ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1

# ---------------------------------------------------------------------------
# Build interest-keyword flags
# ---------------------------------------------------------------------------
INTEREST_KEYWORDS = {
    "defect": [], "quality": [], "scrap": [], "rework": [], "yield": [],
    "cost": [], "price": [], "revenue": [], "profit": [],
    "batch": [], "lot": [], "timestamp": [], "time": [],
    "station": [], "facility": [], "sku": [], "product": [],
    "queue": [], "waiting": [], "wait": [], "utilization": [], "util": [],
    "breakdown": [], "downtime": [], "changeover": [],
}


def check_keywords(columns):
    hits = {}
    for kw in INTEREST_KEYWORDS:
        matched = [c for c in columns if kw in c.lower()]
        if matched:
            hits[kw] = matched
    return hits


# ---------------------------------------------------------------------------
# Analyze each tabular file
# ---------------------------------------------------------------------------
file_analyses = []
all_keyword_hits = {}

for fpath in tabular_files:
    size = fpath.stat().st_size
    # For very large files, limit rows
    use_nrows = NROWS_LIMIT if size > 10_000_000 else None
    df, trunc = read_tabular(fpath, nrows=use_nrows)
    if df is None:
        file_analyses.append({"path": fpath, "size": size, "error": str(trunc)})
        continue

    # Full shape for big CSVs
    if use_nrows and fpath.suffix.lower() == ".csv":
        full_r, full_c = full_shape(fpath)
        shape_str = f"({full_r}, {full_c}) [stats from first {NROWS_LIMIT} rows]"
    else:
        full_r = len(df)
        shape_str = str(df.shape)

    cols_info = {}
    for c in df.columns:
        cols_info[c] = col_summary(df[c])

    kw_hits = check_keywords(df.columns.tolist())
    for kw, cols in kw_hits.items():
        all_keyword_hits.setdefault(kw, []).extend(
            [(fpath.name, c) for c in cols]
        )

    first_rows = df.head(MAX_PREVIEW_ROWS)

    file_analyses.append({
        "path": fpath,
        "size": size,
        "shape": shape_str,
        "ncols": len(df.columns),
        "columns": list(df.columns),
        "cols_info": cols_info,
        "first_rows": first_rows,
        "kw_hits": kw_hits,
        "full_rows": full_r,
    })

# ---------------------------------------------------------------------------
# Read README content
# ---------------------------------------------------------------------------
readme_content = ""
for rf in readme_files:
    try:
        with open(rf, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[:60]
            readme_content = "".join(lines)
    except:
        pass

# ---------------------------------------------------------------------------
# Read first 60 lines from PDFs (text extraction attempt)
# ---------------------------------------------------------------------------
pdf_notes = []
for pf in pdf_files:
    pdf_notes.append(f"- `{pf.name}` ({human_size(pf.stat().st_size)}) — PDF process model diagram. Text extraction not attempted (binary).")


# ---------------------------------------------------------------------------
# Write DATA_INVENTORY.md
# ---------------------------------------------------------------------------

lines_out = []
w = lines_out.append

w("# DATA INVENTORY")
w("")
w("## A. Dataset Card")
w("")
w(f"- **Total files**: {len(all_files)} files — types: {', '.join(f'{e}({n})' for e, n in sorted(ext_counts.items()))}")
w(f"- **Image data**: {'Yes' if image_files else 'No'}")
w(f"- **README / data-dictionary**: {'Yes — `Readme.txt`' if readme_files else 'No'}")

if pdf_files:
    w(f"- **PDFs**: {', '.join(f'`{p.name}`' for p in pdf_files)} (process-model diagrams, not machine-readable text)")

w("")
if readme_content:
    w("**Readme.txt (first 60 lines):**")
    w("```")
    w(readme_content.rstrip())
    w("```")
    w("")

# Keyword presence
w("**Column keyword scan:**")
w("")
kw_yes = []
kw_no = []
for kw in sorted(INTEREST_KEYWORDS):
    if kw in all_keyword_hits:
        examples = list(set(c for _, c in all_keyword_hits[kw]))[:5]
        kw_yes.append(f"  - **{kw}**: YES — e.g. {', '.join(f'`{e}`' for e in examples)}")
    else:
        kw_no.append(kw)
for line in kw_yes:
    w(line)
if kw_no:
    w(f"  - **Not found**: {', '.join(kw_no)}")
w("")

# What one row represents
w("**What one row likely represents:**")
w("  - Model 1 & 2: one simulation *run* (experiment replicate) — GUESS")
w("  - Model 3: one *time snapshot* within a long simulation (Time_Now column present) — GUESS")
w("")

# Joinability
w("**Identifiers that could join files:**")
w("  - No shared key column across model files; they are independent simulation experiments. `Demand` appears in Models 1 & 2. — GUESS")
w("")

# Time span
w("**Time span & granularity:**")
for fa in file_analyses:
    if "error" in fa:
        continue
    name = fa["path"].name
    time_cols = [c for c in fa["columns"] if "time" in c.lower() or c == "Time_Now"]
    if time_cols:
        w(f"  - `{name}`: time-related columns: {', '.join(f'`{c}`' for c in time_cols)}")
w("")

# ---------------------------------------------------------------
w("## B. File Tree")
w("")
w("```")
for f in all_files:
    rel = f.relative_to(DATA_FOLDER)
    size = f.stat().st_size
    w(f"  {rel}  ({human_size(size)})")
w("```")
w("")

# ---------------------------------------------------------------
w("## C. Tabular File Details")
w("")

for fa in file_analyses:
    name = fa["path"].name
    w(f"### `{name}`")
    w("")
    if "error" in fa:
        w(f"⚠️ Could not read: {fa['error']}")
        w("")
        continue

    w(f"- **Size**: {human_size(fa['size'])}")
    w(f"- **Shape**: {fa['shape']}")
    w("")

    ncols = fa["ncols"]
    cols_info = fa["cols_info"]

    if ncols > 40:
        # Compact grouped listing
        w(f"**{ncols} columns (grouped by prefix):**")
        w("")
        prefixes = {}
        for c in fa["columns"]:
            prefix = c.split("_")[0] if "_" in c else c
            prefixes.setdefault(prefix, []).append(c)
        for prefix, cols in sorted(prefixes.items()):
            w(f"- `{prefix}` ({len(cols)}): {', '.join(f'`{c}`' for c in cols[:6])}{' ...' if len(cols) > 6 else ''}")
        w("")
        # Show 15 representative columns
        step = max(1, ncols // 15)
        rep_cols = [fa["columns"][i] for i in range(0, ncols, step)][:15]
        w("**15 representative columns (detailed):**")
        w("")
        for c in rep_cols:
            w(f"| `{c}` | {cols_info[c]} |")
        w("")
        # First 3 rows — only representative cols
        w("**First 3 rows (representative columns):**")
        w("")
        header = " | ".join(f"`{c}`" for c in rep_cols)
        w(f"| {header} |")
        w(f"| {' | '.join('---' for _ in rep_cols)} |")
        for _, row in fa["first_rows"][rep_cols].iterrows():
            vals = " | ".join(str(v)[:12] for v in row.values)
            w(f"| {vals} |")
        w("")
    else:
        w("| Column | dtype | nulls | unique | stats |")
        w("| --- | --- | --- | --- | --- |")
        for c in fa["columns"]:
            info = cols_info[c]
            parts = info.split(" | ")
            if len(parts) == 4:
                w(f"| `{c}` | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} |")
            else:
                w(f"| `{c}` | {info} |")
        w("")
        w("**First 3 rows:**")
        w("")
        header = " | ".join(f"`{c}`" for c in fa["columns"])
        w(f"| {header} |")
        w(f"| {' | '.join('---' for _ in fa['columns'])} |")
        for _, row in fa["first_rows"].iterrows():
            vals = " | ".join(str(v)[:15] for v in row.values)
            w(f"| {vals} |")
        w("")

# ---------------------------------------------------------------
w("## D. Column-Group Role Labels")
w("")
w("| Group | Columns (examples) | Likely Role |")
w("| --- | --- | --- |")
w("| Demand / Input params | `Demand` | GUESS: input parameter |")
w("| Utilization | `*_Util`, `*_Utilization` | GUESS: output metric |")
w("| Queue / Waiting | `*_Queue*`, `*Waiting*` | GUESS: output metric |")
w("| Part counters | `c_*`, `Entities*`, `Total parts`, `c_TotalProducts` | GUESS: output metric |")
w("| Time metrics | `VA Time`, `*_VA_Time`, `*_NVA_Time`, `*_Wait_Time`, `Parts per hour` | GUESS: output metric |")
w("| Storage | `*_Storage*`, `*_Stored`, `Warehouse*` | GUESS: output metric |")
w("| Quality | `Quality_Util`, `Quality_Queue` | GUESS: output metric (quality-check station) |")
w("| SKU / Product | `*_SKU*` | GUESS: identifier / categorical |")
w("| Time index | `Time_Now` | GUESS: identifier (simulation clock) |")
w("| Paint / Forklift | `Paint*_Util`, `Forklift_*` | GUESS: output metric |")
w("")

# ---------------------------------------------------------------
w("## E. Non-Tabular / Binary Files")
w("")
for bf in binary_files:
    ext = bf.suffix.lower()
    size = bf.stat().st_size
    if ext == ".mat":
        tool = "scipy.io.loadmat or h5py"
    elif ext == ".doe":
        tool = "Rockwell Arena Simulation v15 (proprietary)"
    elif ext == ".zip":
        tool = "zipfile / 7-Zip"
    else:
        tool = "UNKNOWN"
    w(f"- `{bf.name}` ({human_size(size)}) — `{ext}` file. Open with: {tool}")
w("")
for note in pdf_notes:
    w(note)
w("")

# ---------------------------------------------------------------
# Write output
md_text = "\n".join(lines_out)
OUTPUT_FILE.write_text(md_text, encoding="utf-8")
print(f"[OK] DATA_INVENTORY.md written to {OUTPUT_FILE} ({len(lines_out)} lines)")
