"""Build the private Kaggle dry-run notebook from the local trainer."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import zlib
from pathlib import Path

import nbformat as nbf


LOCKED_WORDS = {"holdout", "locked", "locked_holdout", "locked-holdout"}


def is_locked(row: dict[str, str]) -> bool:
    for value in row.values():
        text = str(value).strip().lower()
        if text in LOCKED_WORDS or ("locked" in text and "cond" not in text):
            return True
    split = str(row.get("split", "")).strip().lower()
    return split in {"holdout", "locked_holdout"}


def safe_frozen_splits(path: Path) -> tuple[str, int]:
    """Embed only dev/cond_holdout metadata; never transmit locked rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = rows[0].keys() if rows else []
    safe = [row for row in rows if not is_locked(row)]
    output_lines: list[str] = []
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(safe)
    return stream.getvalue(), len(rows) - len(safe)


def build(output: Path, source_path: Path, splits_path: Path) -> Path:
    source = source_path.read_text(encoding="utf-8")
    splits_text, locked_count = safe_frozen_splits(splits_path)
    encoded = base64.b64encode(zlib.compress(splits_text.encode("utf-8"), 9)).decode("ascii")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    env_cell = """import os, platform, sys
print('python', sys.version)
try:
    import torch
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpu_count', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print('gpu', i, torch.cuda.get_device_name(i))
except Exception as e:
    print('torch unavailable:', type(e).__name__, e)
try:
    import timm
    print('timm', timm.__version__)
except Exception as e:
    print('timm unavailable:', type(e).__name__, e)
print('script_sha256_expected', SCRIPT_SHA256)
print('locked_rows_excluded_from_embedded_splits', LOCKED_COUNT)
""".replace("SCRIPT_SHA256", repr(source_hash)).replace("LOCKED_COUNT", str(locked_count))

    split_cell = """from pathlib import Path
import base64, io, zlib
import pandas as pd

encoded = ENCODED_SPLITS
frozen_text = zlib.decompress(base64.b64decode(encoded)).decode('utf-8')
Path('/kaggle/working/splits.csv').write_text(frozen_text, encoding='utf-8')
frozen = pd.read_csv(io.StringIO(frozen_text))
print('wrote /kaggle/working/splits.csv rows', len(frozen), 'columns', list(frozen.columns))

def safe_rows(frame):
    split = frame['split'].astype(str).str.lower() if 'split' in frame.columns else ''
    return frame.loc[~split.isin(['holdout', 'locked_holdout', 'locked'])].copy()

for candidate in Path('/kaggle/input').rglob('splits.csv'):
    attached = safe_rows(pd.read_csv(candidate))
    left = safe_rows(frozen)
    if left.equals(attached):
        print('attached splits.csv matches embedded safe frozen metadata:', candidate)
    else:
        print('DIFF attached splits.csv vs embedded safe frozen metadata:', candidate)
        print('embedded_only_rows', len(left.merge(attached, how='outer', indicator=True).query("_merge == 'left_only'")))
        print('attached_only_rows', len(attached.merge(left, how='outer', indicator=True).query("_merge == 'left_only'")))
""".replace("ENCODED_SPLITS", repr(encoded))

    source_cell = f"from pathlib import Path\nSOURCE = {source!r}\nPath('/kaggle/working/train_5fold.py').write_text(SOURCE, encoding='utf-8')\nprint('wrote /kaggle/working/train_5fold.py')\nprint('script_sha256_actual', __import__('hashlib').sha256(SOURCE.encode('utf-8')).hexdigest())"
    launch_cell = """DRY_RUN = True
import sys
sys.path.insert(0, '/kaggle/working')
from train_5fold import main
args = ['--splits-csv', '/kaggle/working/splits.csv', '--out', '/kaggle/working', '--gpu', '0', '--folds', '0,1']
if DRY_RUN:
    args.append('--dry-run')
print('launching:', args, 'DRY_RUN=', DRY_RUN)
raise SystemExit(main(args))
"""

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}
    notebook.cells = [
        nbf.v4.new_code_cell(env_cell),
        nbf.v4.new_code_cell(split_cell),
        nbf.v4.new_code_cell(source_cell),
        nbf.v4.new_code_cell(launch_cell),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(f"wrote {output} cells={len(notebook.cells)} source_sha256={source_hash} locked_excluded={locked_count}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1] / "notebooks" / "train_5fold.py"))
    parser.add_argument("--splits", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "splits.csv"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "notebooks" / "train_5fold_kaggle.ipynb"))
    args = parser.parse_args()
    build(Path(args.output), Path(args.source), Path(args.splits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
