import json
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.theme import ACCEPT_COLOR, REJECT_COLOR, REVIEW_COLOR, NOVEL_COLOR

def advisory_badge():
    st.info("ADVISORY / SIMULATED: This system is a software-only digital twin. Recommendations and flow metrics are simulated and do not control hardware.")

def _mtime(path) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0

@st.cache_data
def load_artifact(path_str: str, _mtime_val: float):
    path = Path(path_str)
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if "class_probs" in df.columns:
            df["class_probs"] = df["class_probs"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
        if "conformal_set" in df.columns:
            df["conformal_set"] = df["conformal_set"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
        return df
    elif path.suffix == ".json":
        return json.loads(path.read_text())
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    return None

def kpi_strip(items):
    cols = st.columns(len(items))
    for i, col in enumerate(cols):
        item = items[i]
        label = item.get("label", "")
        value = item.get("value", "")
        delta = item.get("delta", None)
        delta_color = item.get("delta_color", "normal")
        help_text = item.get("help", "")
        col.metric(label, value, delta, delta_color=delta_color, help=help_text)

def status_pill(verdict_or_flag):
    color = "gray"
    if verdict_or_flag == "ACCEPT":
        color = ACCEPT_COLOR
    elif verdict_or_flag == "REJECT":
        color = REJECT_COLOR
    elif verdict_or_flag in ["REVIEW", "UNCERTAIN"]:
        color = REVIEW_COLOR
    elif verdict_or_flag == "NOVEL":
        color = NOVEL_COLOR
    st.markdown(f'<span style="background-color:{color}; padding: 2px 8px; border-radius: 12px; color: white; font-weight: bold; font-size: 0.8em;">{verdict_or_flag}</span>', unsafe_allow_html=True)

def evidence_card(data: dict):
    st.markdown(f"**Evidence:** {data.get('title', '')}")
    if "lines" in data:
        for line in data["lines"]:
            st.caption(f"- {line}")

def assumptions_expander(ids):
    with st.expander("Assumptions & limits"):
        st.markdown("- The organizers provided no economics sheet, hidden-test format, or checkpoint times, so costs and formats are stated assumptions.")
        st.markdown("- Inspection images and production data share no keys; the quality-to-flow link is an editable assumption, not a measured relationship.")
        st.markdown("- The production data has no batch, defect, scrap, rework or cost fields, so no real root cause can be mined from it; any root-cause output will be labeled simulated.")
        st.markdown("- The image set is class-balanced, so reject rates on it are not factory yield.")
        st.markdown("- Pretrained ImageNet weights are assumed to be allowed. All numbers are fold 0, single seed.")
