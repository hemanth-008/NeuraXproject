"""CLI entry point for the deterministic labelled-simulated factory layer."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.make_batches import main  # noqa: E402


if __name__ == "__main__":
    main()
