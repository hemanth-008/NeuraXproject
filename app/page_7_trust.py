import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime, assumptions_expander
import pandas as pd
import plotly.express as px
from pathlib import Path

try:
    advisory_badge()
    st.title("Trust & Status")

    calib = load_artifact("artifacts/calibration.json", _mtime(Path("artifacts/calibration.json")))
    status = load_artifact("artifacts/status.json", _mtime(Path("artifacts/status.json")))

    if calib is None or status is None:
        raise FileNotFoundError("trust artifacts are unavailable in this build")

    st.table(pd.DataFrame(status))

    k1, k2, k3 = st.columns(3)
    k1.metric("ECE", calib.get("ece", "n/a"))
    k2.metric("Conformal coverage", calib.get("coverage", "n/a"))

    assumptions_expander([])
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
