import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime, assumptions_expander
import pandas as pd

try:
    advisory_badge()
    st.title("Trust & Status")

    calib_path = "artifacts/calibration.json"
    status_path = "artifacts/status.json"
    holdout_path = "artifacts/holdout_result.json"
    calib = load_artifact(calib_path, _mtime(calib_path))
    status = load_artifact(status_path, _mtime(status_path))
    holdout = load_artifact(holdout_path, _mtime(holdout_path))

    if calib is None or status is None:
        raise FileNotFoundError("trust artifacts are unavailable")

    st.caption("Calibration and locked-holdout values are read-only artifacts; the locked result was not used for tuning.")
    status_value = status.get("status", "n/a")
    if status_value != "ok":
        st.warning("Build validation is not fully green: some legacy or optional artifacts remain invalid/missing. This page shows only values present in artifacts.")
    st.metric("Build status", status_value)
    components = status.get("components", [])
    if components:
        st.dataframe(pd.DataFrame(components), width="stretch", hide_index=True)

    clean = calib.get("clean_oof", {})
    conformal = clean.get("conformal", {})
    k1, k2, k3 = st.columns(3)
    k1.metric("Clean OOF ECE", clean.get("scaled", {}).get("ece", "n/a"))
    k2.metric("Conformal coverage", conformal.get("coverage", "n/a"))
    k3.metric("Temperature", calib.get("temperature", {}).get("value", "n/a"))
    st.caption(f"Calibration source: {calib_path} · folds used: {calib.get('folds_used', 'n/a')}")

    if holdout:
        st.subheader("Locked holdout — final, not for tuning")
        st.dataframe(pd.DataFrame([{"n": holdout.get("n"), "accuracy": holdout.get("accuracy"), "macro_f1": holdout.get("macro_f1"), "ECE": holdout.get("ece"), "source": holdout_path}]), width="stretch", hide_index=True)

    assumptions_expander(["D14", "D18", "D22", "D24", "D25"])
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
