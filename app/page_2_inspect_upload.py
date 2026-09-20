import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime, status_pill
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
import json
import yaml
from app.infer_client import run_infer
import tempfile
import os

advisory_badge()
st.title("Inspect & Upload")

def get_cost(val):
    if isinstance(val, dict):
        return val.get("value", 0.0)
    return float(val)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
costs = config.get("economics", {})
C_ESCAPE = get_cost(costs.get("c_escape", 50.0))
C_FALSE_REJECT = get_cost(costs.get("c_false_reject", 5.0))
C_REVIEW = get_cost(costs.get("c_review", 1.0))

tab_inspect, tab_upload = st.tabs(["Inspect", "Upload"])

with tab_inspect:
    PRED_PATH = "artifacts/predictions.parquet"
    preds = load_artifact(PRED_PATH, _mtime(PRED_PATH))

    if preds is None:
        st.error("artifacts/predictions.parquet missing: run pipeline")
    else:
        st.write("Quick Jump:")
        cols = st.columns(3)
        jump_id = None
        if cols[0].button("crack_00011"): jump_id = "crack_00011"
        if cols[1].button("rust_00187"): jump_id = "rust_00187"
        if cols[2].button("rust_00891"): jump_id = "rust_00891"

        col_filters, col_view = st.columns([1, 2])
        with col_filters:
            decision_f = st.multiselect("Decision", sorted(preds["decision"].unique()), default=list(preds["decision"].unique()))
            if "flag" in preds.columns:
                flag_f = st.multiselect("Flag", sorted(preds["flag"].dropna().unique()), default=list(preds["flag"].dropna().unique()))
            else:
                flag_f = []
            family_f = st.multiselect("Predicted family", sorted(preds["pred_family"].unique()), default=list(preds["pred_family"].unique()))
            
            mask = preds["decision"].isin(decision_f) & preds["pred_family"].isin(family_f)
            if flag_f and "flag" in preds.columns:
                mask = mask & preds["flag"].isin(flag_f)
            
            filtered = preds[mask].copy()
            st.write(f"{len(filtered)} of {len(preds)} units")
            
            idx = None
            if len(filtered):
                if jump_id and jump_id in filtered["unit_id"].values:
                    default_idx = int(filtered[filtered["unit_id"] == jump_id].index[0])
                else:
                    default_idx = int(filtered.index[0])
                idx = st.selectbox("Unit", filtered.index, index=filtered.index.get_loc(default_idx) if default_idx in filtered.index else 0, format_func=lambda i: filtered.loc[i, "unit_id"])

        with col_view:
            if idx is not None:
                row = filtered.loc[idx]
                st.subheader(row["unit_id"])
                
                c_pill1, c_pill2 = st.columns(2)
                with c_pill1:
                    status_pill(row["decision"])
                with c_pill2:
                    if "flag" in row and pd.notna(row["flag"]) and row["flag"] != "NONE":
                        status_pill(row["flag"])
                
                st.write(f"**Family:** {row['pred_family']}")
                if "p_defect" in row and pd.notna(row["p_defect"]):
                    st.write(f"**Confidence:** {row['p_defect']:.2%}")
                if "conformal_set" in row and pd.notna(row["conformal_set"]):
                    st.write(f"**Conformal set:** {row['conformal_set']}")

                img_path = Path("data") / row["image_path"] if "image_path" in row else None
                if img_path and img_path.exists():
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                
                if "heatmap_path" not in row or pd.isna(row["heatmap_path"]):
                    st.info("heatmap pending: comes from the infer stage")
                else:
                    hm_path = Path(row["heatmap_path"])
                    if hm_path.exists():
                        st.image(Image.open(hm_path), use_container_width=True, caption="Grad-CAM++ Heatmap")
                    else:
                        st.info("heatmap pending: comes from the infer stage")
                
                st.markdown("### Why this verdict?")
                p_def = float(row.get("p_defect", 0.0)) if pd.notna(row.get("p_defect")) else 0.0
                cost_accept = p_def * C_ESCAPE
                cost_reject = (1 - p_def) * C_FALSE_REJECT
                cost_review = C_REVIEW
                
                cost_df = pd.DataFrame({
                    "Action": ["ACCEPT", "REJECT", "REVIEW"],
                    "Expected Cost": [cost_accept, cost_reject, cost_review]
                })
                st.table(cost_df)
                st.caption("Costs computed from economics block (ASSUMED D18/D22).")

with tab_upload:
    st.subheader("Upload for Inference")
    if not Path("run_pipeline.py").exists() or not list(Path("artifacts/models").glob("*.pt")):
        st.error("Inference engine not installed yet (missing run_pipeline.py or weights in artifacts/models/).")
    else:
        uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files and st.button("Run Inference"):
            with tempfile.TemporaryDirectory() as tmpdir:
                for uf in uploaded_files:
                    p = Path(tmpdir) / uf.name
                    p.write_bytes(uf.getvalue())
                with st.spinner("Running inference..."):
                    res_df = run_infer(tmpdir)
                if res_df is not None:
                    st.success("Inference complete")
                    st.dataframe(res_df)
                else:
                    st.error("Inference failed.")
