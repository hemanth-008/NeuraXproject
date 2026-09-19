import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime, evidence_card
import pandas as pd
import plotly.express as px

advisory_badge()
st.title("Flow & Bottleneck")

def _station(col: str) -> str:
    return col.replace(" Utilization", "").replace(" Util", "")

FLOW_PATH = "artifacts/flow_v0.json"
flow = load_artifact(FLOW_PATH, _mtime(FLOW_PATH))

if flow is None:
    st.error("artifacts/flow_v0.json missing: run scripts/flow_v0_report.py")
else:
    model_choice = st.radio("Production model", ["model1", "model2"], horizontal=True)
    m = flow[model_choice]
    util_rows = [r for r in m["summary"] if r["kind"] == "utilization"]
    util_df = pd.DataFrame(util_rows)
    util_df["station"] = util_df["column"].apply(_station)
    util_df = util_df.sort_values(by="mean", ascending=False)
    fig = px.bar(util_df, x="station", y="mean", title=f"{model_choice}: mean station utilization", text_auto=".2f")
    fig.add_hline(y=m["bottleneck_v0"]["utilization"], line_dash="dot", annotation_text="bottleneck")
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns([1, 2])
    c1.metric("Candidate bottleneck", _station(m["bottleneck_v0"]["candidate_bottleneck"]), f"{m['bottleneck_v0']['utilization']:.1%} utilization", delta_color="off")
    with c2:
        evidence_card({"title": "Triangulated Evidence (Placeholder)", "lines": ["Highest utilization", "Upstream queue build-up", "Downstream starvation"]})
