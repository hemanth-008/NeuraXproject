import os

pages = {
    "page_1_command_center.py": """import streamlit as st
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
""",
    "page_2_inspect_upload.py": """import streamlit as st
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
""",
    "page_3_root_cause.py": """import streamlit as st
from app.components import advisory_badge
advisory_badge()
st.title("Root Cause")
st.info("Artifact missing: run root-cause pipeline")
""",
    "page_4_flow_bottleneck.py": """import streamlit as st
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
""",
    "page_5_profit_whatif.py": """import streamlit as st
from app.components import advisory_badge
advisory_badge()
st.title("Profit & What-if")
st.info("Artifact missing: run profit pipeline")
""",
    "page_6_actions.py": """import streamlit as st
from app.components import advisory_badge
advisory_badge()
st.title("Actions")
st.info("Artifact missing: run recommendations pipeline")
""",
    "page_7_trust.py": """import streamlit as st
from app.components import advisory_badge, load_artifact, _mtime, assumptions_expander
import pandas as pd
import plotly.express as px
from pathlib import Path

advisory_badge()
st.title("Trust & Status")

calib = load_artifact("artifacts/calibration.json", _mtime(Path("artifacts/calibration.json")))
status = load_artifact("artifacts/status.json", _mtime(Path("artifacts/status.json")))

if status is None:
    st.info("artifacts/status.json missing: run pipeline")
else:
    st.table(pd.DataFrame(status))

k1, k2, k3 = st.columns(3)
if calib:
    k1.metric("ECE", calib.get("ece", "n/a"))
    k2.metric("Conformal coverage", calib.get("coverage", "n/a"))
else:
    k1.metric("ECE", "n/a: artifact missing")
    k2.metric("Conformal coverage", "n/a: artifact missing")

assumptions_expander([])
"""
}

for filename, content in pages.items():
    with open(f"app/{filename}", "w") as f:
        f.write(content)
