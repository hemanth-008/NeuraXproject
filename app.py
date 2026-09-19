"""
QualityTwin CP2 vertical slice -- Streamlit dashboard (v0)

Reads only from artifacts/ (predictions.parquet, flow_v0.json). No GPU,
no live model, no internet needed at demo time (Wi-Fi-off survival is a
hard requirement per the master plan).

Run:
    .venv\\Scripts\\streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

ARTIFACTS = Path("artifacts")
IMAGE_ROOT = Path("data/images/train")  # predictions.image_path is relative to this, e.g. "train/crack/xxx.png"

st.set_page_config(page_title="QualityTwin", layout="wide")


@st.cache_data
def load_predictions():
    path = ARTIFACTS / "predictions.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["class_probs"] = df["class_probs"].apply(json.loads)
    return df


@st.cache_data
def load_flow():
    path = ARTIFACTS / "flow_v0.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


preds = load_predictions()
flow = load_flow()

st.title("QualityTwin")
st.caption("Software-only quality-aware digital twin. Everything here is ADVISORY / SIMULATED unless stated otherwise.")

tab_inspect, tab_health, tab_flow = st.tabs(["Inspect", "Line Health", "Flow & Bottleneck"])

# ---------------- Inspect ----------------
with tab_inspect:
    if preds is None:
        st.warning("No artifacts/predictions.parquet yet.")
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
                img_path = IMAGE_ROOT / row["image_path"]
                if img_path.exists():
                    st.image(Image.open(img_path), width=320)
                else:
                    st.info(f"Image not found locally at {img_path} -- adjust IMAGE_ROOT at the top of this file if your images live elsewhere.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Verdict", row["decision"])
                m2.metric("Predicted family", row["pred_family"])
                m3.metric("Flag", row["flag"])

                st.write("Class probabilities")
                probs_df = pd.DataFrame({"class": list(row["class_probs"].keys()), "p": list(row["class_probs"].values())})
                st.plotly_chart(px.bar(probs_df, x="class", y="p", range_y=[0, 1]), use_container_width=True)

                st.caption(
                    "No calibrated confidence, conformal set, heatmap or localization basis yet "
                    "(pending handoff steps 3 and 4). The decision above is a NAIVE normal-vs-not rule, "
                    "not the cost-based decision."
                )
                if row.get("true_label") is not None:
                    st.caption(f"Ground truth (dev/demo use only, not a live label): {row['true_label']}")
            else:
                st.info("Pick a unit from the filters on the left.")

# ---------------- Line Health ----------------
with tab_health:
    if preds is None:
        st.warning("No artifacts/predictions.parquet yet.")
    else:
        n = len(preds)
        n_accept = int((preds["decision"] == "ACCEPT").sum())
        n_reject = int((preds["decision"] == "REJECT").sum())
        n_uncertain = int((preds["flag"] == "UNCERTAIN").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Units scored", n)
        c2.metric("ACCEPT", n_accept, f"{n_accept/n:.1%}")
        c3.metric("REJECT", n_reject, f"{n_reject/n:.1%}")
        c4.metric("UNCERTAIN flags", n_uncertain)

        st.warning(
            f"This set is class-balanced (~20% per family), so the {n_reject/n:.0%} reject rate is NOT a "
            "real production defect rate -- it's an artifact of the labeled benchmark. Don't quote it as "
            "factory yield. The Flow tab uses a separate, adjustable assumed rate for the simulation."
        )

        by_family = preds["pred_family"].value_counts().reset_index()
        by_family.columns = ["family", "count"]
        st.plotly_chart(px.bar(by_family, x="family", y="count", title="Predicted family distribution"), use_container_width=True)

        st.caption(
            "A time-ordered live-replay ticker (FPY/OEE/margin over time) isn't built yet -- predictions.parquet "
            "has no real timestamps (ts is null per D17). This page is a static snapshot."
        )

# ---------------- Flow & Bottleneck ----------------
with tab_flow:
    if flow is None:
        st.warning("No artifacts/flow_v0.json yet -- run scripts/flow_v0_report.py")
    else:
        model_choice = st.radio("Production model", ["model1", "model2"], horizontal=True)
        m = flow[model_choice]
        util_rows = [r for r in m["summary"] if r["kind"] == "utilization"]
        util_df = pd.DataFrame(util_rows)
        fig = px.bar(util_df, x="column", y="mean", title=f"{model_choice}: mean station utilization")
        fig.add_hline(y=m["bottleneck_v0"]["utilization"], line_dash="dot", annotation_text="bottleneck")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Candidate bottleneck", m["bottleneck_v0"]["candidate_bottleneck"], f"{m['bottleneck_v0']['utilization']:.1%} utilization")

        if model_choice == "model2":
            drill_q = next(r for r in m["summary"] if r["column"] == "Drilling Queue Time")["mean"]
            asm_q = next(r for r in m["summary"] if r["column"] == "Assembly Queue Time")["mean"]
            if drill_q > asm_q:
                st.caption(
                    f"Note: Drilling's mean queue time ({drill_q:.2f}) exceeds Assembly's ({asm_q:.2f}) here, "
                    "even though Assembly has higher utilization. The bottleneck call isn't fully triangulated "
                    "on this model -- disclosed, not hidden."
                )

        st.divider()
        st.subheader("Simulated quality-to-flow coupling (ADVISORY / SIMULATED)")
        st.caption(
            "No batch, rework, scrap or cost fields exist in the production data (decision D14). This is an "
            "explicit, editable assumption, not a measurement: every rejected unit is assumed to cost the "
            "bottleneck station exactly one extra pass (a rework/re-inspection loop), holding cycle time, "
            "machine count, availability and changeover loss fixed. Utilization scales linearly with that "
            "extra-visits rate: rho_new = rho_baseline * (1 + rework_rate)."
        )
        rework_rate = st.slider("Assumed rework/defect rate at the bottleneck (%)", 0, 50, 3) / 100
        rho0 = m["bottleneck_v0"]["utilization"]
        rho_new = rho0 * (1 + rework_rate)
        c1, c2 = st.columns(2)
        c1.metric("Baseline bottleneck utilization", f"{rho0:.1%}")
        c2.metric("Simulated utilization at this rework rate", f"{rho_new:.1%}", f"{rho_new-rho0:+.1%}")
        if rho_new >= 1.0:
            st.error(
                "At this rework rate the bottleneck would exceed 100% utilization -- demand can't be "
                "sustained without dropping units or building backlog. This is where quality cost becomes "
                "a flow problem, not just a scrap-cost problem."
            )

st.divider()
st.caption("QualityTwin -- Neurax Hackathon 3.0. Software-only. No camera, PLC or robot. All recommendations advisory.")
