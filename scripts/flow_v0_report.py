"""
scripts/flow_v0_report.py

Run: .venv\\Scripts\\python.exe scripts\\flow_v0_report.py
(from the C:\\NeuraxHack repo root, with data/ populated)

Prints station utilization/queue summaries and a naive bottleneck
candidate for Model 1 and Model 2, plus the Model 3 Time_Now check.
Also writes artifacts/flow_v0.json so the dashboard skeleton has
something real to read later.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.loaders import load_model1, load_model2, check_model3_time_structure
from src.flow.metrics import station_summary, bottleneck_v0


def main():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    results = {}

    for name, loader in [("model1", load_model1), ("model2", load_model2)]:
        print(f"\n=== {name} ===")
        df = loader()
        summary = station_summary(df)
        bottleneck = bottleneck_v0(df)
        print(summary.to_string(index=False))
        if "error" in bottleneck:
            print(f"Bottleneck v0 skipped: {bottleneck['error']}")
            print("Columns seen:", list(df.columns))
        else:
            print(f"Candidate bottleneck: {bottleneck['candidate_bottleneck']} "
                  f"(mean utilization {bottleneck['utilization']:.3f})")
        results[name] = {
            "summary": summary.to_dict(orient="records"),
            "bottleneck_v0": bottleneck,
        }

    print("\n=== model3 Time_Now structure ===")
    t3 = check_model3_time_structure()
    print(json.dumps(t3, indent=2, default=str))
    results["model3_time_check"] = t3

    out_path = artifacts_dir / "flow_v0.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
