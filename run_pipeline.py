"""Offline, idempotent entrypoint for the QualityTwin artifact pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STAGES = ("ingest", "vision", "calib", "rootcause", "flow", "profit", "recs", "infer", "all")


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    seconds: float
    detail: str = ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--input", help="image folder for ingest or infer, or Kaggle output folder for calibration")
    parser.add_argument("--output", default=None, help="inference output folder (defaults to the artifact folder)")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--data-dir", default=None, help="production CSV folder for the flow stage")
    parser.add_argument("--kaggle-run", default=None, help="folder name below kaggle_downloads/")
    parser.add_argument("--force", action="store_true", help="run a stage even when all outputs are current")
    parser.add_argument("--tta-flips", action="store_true")
    parser.add_argument("--tta-rot90", action="store_true")
    return parser


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _fresh(outputs: list[Path], inputs: list[Path]) -> bool:
    """Return true only when every declared output is newer than every input."""

    if not outputs or not all(path.exists() for path in outputs):
        return False
    existing_inputs = _existing(inputs)
    if not existing_inputs:
        return True
    newest_input = max(path.stat().st_mtime for path in existing_inputs)
    return min(path.stat().st_mtime for path in outputs) >= newest_input


def _command(root: Path, args: list[str], log_path: Path) -> tuple[bool, str]:
    """Run one existing script offline and persist its complete output."""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {sys.executable} {' '.join(args)}\n{output}\nreturncode={completed.returncode} seconds={elapsed:.6f}\n",
            encoding="utf-8",
        )
        if output:
            print(output, end="")
        return completed.returncode == 0, output.strip()[-500:]
    except OSError as exc:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"command failed to start: {exc}\n", encoding="utf-8")
        return False, str(exc)


def _script(root: Path, name: str) -> Path:
    return root / "scripts" / name


def _stage_commands(
    stage: str,
    *,
    root: Path,
    artifacts: Path,
    args: argparse.Namespace,
    kaggle_input: Path | None,
) -> list[tuple[list[str], list[Path], list[Path], str]]:
    """Return (argv, outputs, inputs, label) entries for one stage."""

    config = root / "config.yaml"
    data_dir = Path(args.data_dir).resolve() if args.data_dir else root / "data"
    calibration_input = kaggle_input or (Path(args.input).resolve() if args.input and stage != "infer" else None)
    commands: list[tuple[list[str], list[Path], list[Path], str]] = []

    def add(script: str, argv: list[str], outputs: list[str], inputs: list[Path], label: str) -> None:
        script_path = _script(root, script)
        commands.append(([str(script_path), *argv], [artifacts / item for item in outputs], [script_path, *inputs], label))

    if stage == "ingest":
        if args.input:
            image_root = Path(args.input).resolve()
            add("image_inventory.py", [str(image_root)], ["IMAGE_INVENTORY.md", "sample_grid.jpg"], [image_root], "image inventory")
            add("image_triage.py", [str(image_root)], ["image_features.csv", "class_grid.jpg", "IMAGE_TRIAGE.md"], [image_root], "image triage")
            add("make_splits.py", [str(image_root)], ["splits.csv"], [image_root], "frozen splits")
        elif not (artifacts / "splits.csv").exists():
            raise ValueError("--input is required for --stage ingest when artifacts/splits.csv is absent")
    elif stage in {"vision", "calib"}:
        calibration_outputs = ["calibration.json", "novelty.json", "knn_bank.npz", "predictions.parquet", "robustness_orient.csv"]
        calibration_args = ["--splits", str(artifacts / "splits.csv"), "--artifacts", str(artifacts)]
        if calibration_input is not None:
            calibration_args[0:0] = ["--input", str(calibration_input)]
            calibration_inputs = [calibration_input, artifacts / "splits.csv"]
        else:
            calibration_args[0:0] = ["--input", str(root / "kaggle_downloads")]
            calibration_inputs = [artifacts / "splits.csv"]
        add("build_calibration.py", calibration_args, calibration_outputs, calibration_inputs, "calibration")
        if stage == "vision":
            add(
                "build_demo_heatmaps.py",
                ["--predictions", str(artifacts / "predictions.parquet"), "--artifacts-dir", str(artifacts)],
                ["predictions.parquet"],
                [artifacts / "predictions.parquet", artifacts / "calibration.json"],
                "demo heatmaps",
            )
    elif stage == "rootcause":
        # run_rootcause.py resolves its canonical artifacts directory from its
        # own repository root. Fixture tests use a temporary checkout.
        add("run_rootcause.py", ["--root", str(root)], ["evidence_cards.json"], [artifacts / "batch_metrics.parquet"], "root cause")
        add("run_rootcause.py", ["--root", str(root), "--recovery"], ["rootcause_recovery.json"], [artifacts / "evidence_cards.json", artifacts / "sim_truth.json"], "root-cause recovery")
    elif stage == "flow":
        add("build_sim_layer.py", ["--config", str(config), "--predictions", str(artifacts / "predictions.parquet"), "--splits", str(artifacts / "splits.csv"), "--output", str(artifacts)], ["batch_metrics.parquet", "replay_stream.parquet", "sim_unit_map.parquet", "sim_truth.json"], [config, artifacts / "predictions.parquet", artifacts / "splits.csv"], "simulated process layer")
        add("merge_sim_into_predictions.py", ["--predictions", str(artifacts / "predictions.parquet"), "--sim-map", str(artifacts / "sim_unit_map.parquet"), "--no-backup"], ["predictions.parquet"], [artifacts / "predictions.parquet", artifacts / "sim_unit_map.parquet"], "prediction metadata merge")
        add("build_flow.py", ["--data-dir", str(data_dir), "--artifacts-dir", str(artifacts)], ["flow_metrics.json", "twin_validation.json", "sim_results.parquet"], [data_dir], "flow and twin")
    elif stage == "profit":
        add("build_profit.py", ["--config", str(config), "--sim-results", str(artifacts / "sim_results.parquet"), "--calibration", str(artifacts / "calibration.json"), "--output-dir", str(artifacts)], ["profit_model.json", "sim_results.parquet"], [config, artifacts / "sim_results.parquet", artifacts / "calibration.json"], "profit")
    elif stage == "recs":
        add("build_recs.py", ["--sim-results", str(artifacts / "sim_results.parquet"), "--flow", str(artifacts / "flow_metrics.json"), "--evidence", str(artifacts / "evidence_cards.json"), "--calibration", str(artifacts / "calibration.json"), "--output-dir", str(artifacts)], ["recommendations.json"], [artifacts / "sim_results.parquet", artifacts / "flow_metrics.json", artifacts / "evidence_cards.json", artifacts / "calibration.json"], "recommendations")
    return commands


def _run_infer(root: Path, artifacts: Path, args: argparse.Namespace) -> StageResult:
    started = time.perf_counter()
    log_path = artifacts / "logs" / "infer.log"
    if not args.input:
        return StageResult("infer", "skipped", time.perf_counter() - started, "no --input")
    output = Path(args.output).resolve() if args.output else artifacts
    try:
        from src.vision.infer import run_inference

        rows = run_inference(args.input, output, artifacts_dir=artifacts, tta_flips=args.tta_flips, tta_rot90=args.tta_rot90)
        text = f"ok images={len(rows)} seconds={time.perf_counter() - started:.6f}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n", encoding="utf-8")
        print(text)
        return StageResult("infer", "ok", time.perf_counter() - started, text)
    except Exception as exc:
        text = f"failed {type(exc).__name__}: {exc}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n", encoding="utf-8")
        print(text, file=sys.stderr)
        return StageResult("infer", "failed", time.perf_counter() - started, text)


def _run_stage(stage: str, *, root: Path, artifacts: Path, args: argparse.Namespace, kaggle_input: Path | None) -> StageResult:
    started = time.perf_counter()
    if stage == "infer":
        return _run_infer(root, artifacts, args)
    try:
        commands = _stage_commands(stage, root=root, artifacts=artifacts, args=args, kaggle_input=kaggle_input)
    except ValueError as exc:
        return StageResult(stage, "failed", time.perf_counter() - started, str(exc))
    if not commands:
        return StageResult(stage, "skipped", time.perf_counter() - started, "no command configured")
    ran = False
    for index, (argv, outputs, inputs, label) in enumerate(commands, start=1):
        if not args.force and _fresh(outputs, inputs):
            print(f"{stage}: skipped {label} (outputs are current)")
            continue
        ran = True
        # The public contract is one stable log path per stage. A multi-step
        # stage updates the same log as each command completes; the final
        # command's output is the most useful summary for that stage.
        log_path = artifacts / "logs" / f"{stage}.log"
        ok, detail = _command(root, argv, log_path)
        if not ok:
            return StageResult(stage, "failed", time.perf_counter() - started, f"{label}: {detail}")
    return StageResult(stage, "ok" if ran else "skipped", time.perf_counter() - started, "completed" if ran else "outputs are current")


def _print_table(results: list[StageResult]) -> None:
    print("\nstage | status | seconds | detail")
    print("--- | --- | ---: | ---")
    for result in results:
        print(f"{result.stage} | {result.status} | {result.seconds:.3f} | {result.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.stage == "infer" and not args.input:
        parser.error("--input is required for --stage infer")
    root = Path(__file__).resolve().parent
    artifacts = Path(args.artifacts_dir).resolve() if args.artifacts_dir else root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    kaggle_input = None
    if args.kaggle_run:
        candidate = root / "kaggle_downloads" / args.kaggle_run
        if candidate.is_dir():
            kaggle_input = candidate
            print(f"kaggle input: {candidate}")

    requested = list(STAGES[:-1]) if args.stage == "all" else [args.stage]
    results: list[StageResult] = []
    for stage in requested:
        if stage == "ingest" and args.stage == "all" and not args.input and not (artifacts / "splits.csv").exists():
            results.append(StageResult(stage, "skipped", 0.0, "no image input; using existing artifacts"))
            continue
        # A Kaggle run is optional. When it is absent, --stage all is an
        # artifact-replay mode: do not invalidate a complete copied artifact
        # set merely because a source script has a newer filesystem timestamp.
        # This is also what makes a fixture checkout reproducible offline.
        if args.stage == "all" and not args.input and kaggle_input is None and not args.force:
            replay_outputs = {
                "vision": ("calibration.json", "novelty.json", "knn_bank.npz", "predictions.parquet", "robustness_orient.csv"),
                "calib": ("calibration.json", "novelty.json", "knn_bank.npz", "predictions.parquet", "robustness_orient.csv"),
                "rootcause": ("evidence_cards.json", "rootcause_recovery.json"),
                "flow": ("batch_metrics.parquet", "replay_stream.parquet", "sim_unit_map.parquet", "sim_truth.json", "flow_metrics.json", "twin_validation.json", "sim_results.parquet"),
                "profit": ("profit_model.json", "sim_results.parquet"),
                "recs": ("recommendations.json",),
            }.get(stage)
            if replay_outputs and all((artifacts / name).exists() for name in replay_outputs):
                results.append(StageResult(stage, "skipped", 0.0, "using existing artifacts"))
                continue
        result = _run_stage(stage, root=root, artifacts=artifacts, args=args, kaggle_input=kaggle_input)
        results.append(result)
        if result.status == "failed":
            _print_table(results)
            return 1

    if args.stage == "all":
        status_script = _script(root, "write_status.py")
        status_log = artifacts / "logs" / "status.log"
        ok, detail = _command(root, [str(status_script), "--artifacts-dir", str(artifacts)], status_log)
        results.append(StageResult("status", "ok" if ok else "failed", 0.0, detail))
        if not ok:
            _print_table(results)
            return 1
        validate_script = _script(root, "validate_artifacts.py")
        validate_log = artifacts / "logs" / "validate.log"
        ok, detail = _command(root, [str(validate_script), "--dir", str(artifacts)], validate_log)
        results.append(StageResult("validate", "ok" if ok else "failed", 0.0, detail))
        if not ok:
            _print_table(results)
            return 1
    _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
