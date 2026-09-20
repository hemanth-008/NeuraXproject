import subprocess
import pandas as pd
from pathlib import Path
import streamlit as st

def run_infer(input_folder: str) -> pd.DataFrame | None:
    if not Path("run_pipeline.py").exists() or not list(Path("artifacts/models").glob("*.pt")):
        return None
    try:
        subprocess.run(
            [".venv\\Scripts\\python.exe", "run_pipeline.py", "--stage", "infer", "--input", input_folder],
            check=True,
            timeout=120,
            capture_output=True
        )
        csv_path = Path("artifacts/predictions.csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
