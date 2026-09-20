import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime
import pandas as pd
import plotly.express as px
from pathlib import Path

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
    n_review = int((preds["decision"] == "REVIEW").sum()) if "REVIEW" in preds["decision"].values else 0
    n_uncertain = int((preds["flag"] == "UNCERTAIN").sum()) if "flag" in preds.columns else 0
    
    st.caption("Note: Numbers reflect benchmark class balance, not factory yield.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Units scored", n)
    c2.metric("ACCEPT", n_accept, f"{n_accept/n:.1%}" if n else "0%", delta_color="off")
    c3.metric("REJECT", n_reject, f"{n_reject/n:.1%}" if n else "0%", delta_color="off")
    c4.metric("REVIEW", n_review, f"{n_review/n:.1%}" if n else "0%", delta_color="off")
    c5.metric("UNCERTAIN flags", n_uncertain)

    by_family = preds["pred_family"].value_counts().reset_index()
    by_family.columns = ["family", "count"]
    st.plotly_chart(px.bar(by_family, x="family", y="count", title="Predicted family distribution"), use_container_width=True)

st.divider()
c_drift, c_replay = st.columns(2)
with c_drift:
    st.subheader("Drift Alert")
    BATCH_PATH = "artifacts/batch_metrics.parquet"
    batch_metrics = load_artifact(BATCH_PATH, _mtime(BATCH_PATH))
    if batch_metrics is None:
        st.info("Artifact missing: run simulated factory layer")
    else:
        st.dataframe(batch_metrics.head())

with c_replay:
    st.subheader("Live Replay")
    REPLAY_PATH = "artifacts/replay_stream.parquet"
    replay_stream = load_artifact(REPLAY_PATH, _mtime(REPLAY_PATH))
    if replay_stream is None:
        st.info("Artifact missing: run simulated factory layer")
    else:
        st.dataframe(replay_stream.head())
