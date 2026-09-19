import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime
import pandas as pd
import plotly.express as px

advisory_badge()
st.title("Command Center")

PRED_PATH = "artifacts/predictions.parquet"
preds = load_artifact(PRED_PATH, _mtime(PRED_PATH))
if preds is None:
    st.error("artifacts/predictions.parquet missing: run pipeline")
else:
    n = len(preds)
    n_accept = int((preds["decision"] == "ACCEPT").sum())
    n_reject = int((preds["decision"] == "REJECT").sum())
    n_review = int((preds["decision"] == "REVIEW").sum())
    n_uncertain = int((preds["flag"] == "UNCERTAIN").sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Units scored", n)
    c2.metric("ACCEPT", n_accept, f"{n_accept/n:.1%}", delta_color="off")
    c3.metric("REJECT", n_reject, f"{n_reject/n:.1%}", delta_color="off", help="benchmark artifact, not factory yield")
    c4.metric("REVIEW", n_review, f"{n_review/n:.1%}", delta_color="off")
    c5.metric("UNCERTAIN flags", n_uncertain)
    by_family = preds["pred_family"].value_counts().reset_index()
    by_family.columns = ["family", "count"]
    st.plotly_chart(px.bar(by_family, x="family", y="count", title="Predicted family distribution"), use_container_width=True)
