"""Small pipeline entrypoint; C9 may add the remaining stages around infer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QualityTwin offline pipeline")
    parser.add_argument("--stage", required=True, choices=("infer",))
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--tta-flips", action="store_true")
    parser.add_argument("--tta-rot90", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "infer" and not args.input:
        parser.error("--input is required for --stage infer")
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve() if args.output else root / "artifacts"
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        from src.vision.infer import InferenceError, run_inference

        rows = run_inference(
            args.input,
            output,
            artifacts_dir=args.artifacts_dir,
            tta_flips=args.tta_flips,
            tta_rot90=args.tta_rot90,
        )
        status = f"ok images={len(rows)} seconds={time.perf_counter() - started:.6f}"
        code = 0
    except Exception as exc:
        status = f"failed {type(exc).__name__}: {exc} seconds={time.perf_counter() - started:.6f}"
        code = 1
        print(status, file=sys.stderr)
    (log_dir / "infer.log").write_text(status + "\n", encoding="utf-8")
    if code == 0:
        print(status)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
