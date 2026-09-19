import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
import json

advisory_badge()
st.title("Inspect & Upload")

C_ESCAPE = 50.0
C_FALSE_REJECT = 5.0
C_REVIEW = 1.0

PRED_PATH = "artifacts/predictions.parquet"
preds = load_artifact(PRED_PATH, _mtime(PRED_PATH))

if preds is None:
    st.error("artifacts/predictions.parquet missing: run pipeline")
else:
    col_filters, col_view = st.columns([1, 2])
    with col_filters:
        decision_f = st.multiselect("Decision", sorted(preds["decision"].unique()), default=list(preds["decision"].unique()))
        flag_f = st.multiselect("Flag", sorted(preds["flag"].unique()), default=list(preds["flag"].unique()))
        family_f = st.multiselect("Predicted family", sorted(preds["pred_family"].unique()), default=list(preds["pred_family"].unique()))
        filtered = preds[
            preds["decision"].isin(decision_f)
            & preds["flag"].isin(flag_f)
            & preds["pred_family"].isin(family_f)
        ]
        st.write(f"{len(filtered)} of {len(preds)} units")
        idx = st.selectbox("Unit", filtered.index, format_func=lambda i: filtered.loc[i, "unit_id"]) if len(filtered) else None

    with col_view:
        if idx is not None:
            row = filtered.loc[idx]
            st.subheader(row["unit_id"])
