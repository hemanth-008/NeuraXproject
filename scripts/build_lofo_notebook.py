"""Build the self-contained Kaggle notebook for LOFO novelty evaluation."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import zlib
from pathlib import Path

import nbformat as nbf


LOCKED_WORDS = {"holdout", "locked", "locked_holdout", "locked-holdout"}


def is_locked(row: dict[str, str]) -> bool:
    for value in row.values():
        text = str(value).strip().lower()
        if text in LOCKED_WORDS or ("locked" in text and "cond" not in text):
            return True
    return str(row.get("split", "")).strip().lower() in {"holdout", "locked_holdout"}


def safe_frozen_splits(path: Path) -> tuple[str, int]:
    """Embed only dev/cond_holdout metadata; never transmit locked rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        safe: list[dict[str, str]] = []
        locked_count = 0
        for row in reader:
            if is_locked(row):
                locked_count += 1
            else:
                safe.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(safe)
    return stream.getvalue(), locked_count


def build(output: Path, source_path: Path, splits_path: Path) -> Path:
    source = source_path.read_text(encoding="utf-8")
    splits_text, locked_count = safe_frozen_splits(splits_path)
    encoded = base64.b64encode(zlib.compress(splits_text.encode("utf-8"), 9)).decode("ascii")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    env_cell = """import sys
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
split = frozen['split'].astype(str).str.lower()
assert not split.isin(['holdout', 'locked_holdout', 'locked']).any(), 'locked rows in embedded splits'
print('wrote /kaggle/working/splits.csv rows', len(frozen), 'columns', list(frozen.columns))
for candidate in Path('/kaggle/input').rglob('splits.csv'):
    attached = pd.read_csv(candidate)
    attached_split = attached['split'].astype(str).str.lower()
    attached = attached.loc[~attached_split.isin(['holdout', 'locked_holdout', 'locked'])].copy()
    if frozen.equals(attached):
        print('attached safe splits.csv matches embedded metadata:', candidate)
    else:
        print('attached splits.csv differs; using embedded safe metadata:', candidate)
""".replace("ENCODED_SPLITS", repr(encoded))

    source_cell = f"from pathlib import Path\nSOURCE = {source!r}\nPath('/kaggle/working/lofo_novelty.py').write_text(SOURCE, encoding='utf-8')\nprint('wrote /kaggle/working/lofo_novelty.py')\nprint('script_sha256_actual', __import__('hashlib').sha256(SOURCE.encode('utf-8')).hexdigest())"
    launch_cell = """DRY_RUN = True
import sys
sys.path.insert(0, '/kaggle/working')
from lofo_novelty import main
args = ['--splits-csv', '/kaggle/working/splits.csv', '--out', '/kaggle/working', '--families', 'crack,hole,rust,scratch', '--gpu', '0']
if DRY_RUN:
    args.append('--dry-run')
else:
    args.append('--launch-two-gpus')
print('launching:', args, 'DRY_RUN=', DRY_RUN)
raise SystemExit(main(args))
"""
    notes_cell = """# LOFO novelty launch

The default is a CPU-safe dry run. Set `DRY_RUN = False` after confirming the
environment cell and safe embedded split metadata. The full run trains two
families per GPU using `CUDA_VISIBLE_DEVICES`, then writes `lofo_{family}.json`
and `novelty_lofo.json` to `/kaggle/working`.
"""

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}
    notebook.cells = [
        nbf.v4.new_markdown_cell(notes_cell),
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
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", default=str(root / "notebooks" / "lofo_novelty.py"))
    parser.add_argument("--splits", default=str(root / "artifacts" / "splits.csv"))
    parser.add_argument("--output", default=str(root / "notebooks" / "lofo_novelty_kaggle.ipynb"))
    args = parser.parse_args()
    build(Path(args.output), Path(args.source), Path(args.splits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
