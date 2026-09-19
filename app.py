import streamlit as st
from app.theme import inject_theme
from app.components import advisory_badge

st.set_page_config(page_title="QualityTwin", layout="wide")
inject_theme()

pg = st.navigation([
    st.Page("app/page_1_command_center.py", title="Command Center"),
    st.Page("app/page_2_inspect_upload.py", title="Inspect & Upload"),
    st.Page("app/page_3_root_cause.py", title="Root Cause"),
    st.Page("app/page_4_flow_bottleneck.py", title="Flow & Bottleneck"),
    st.Page("app/page_5_profit_whatif.py", title="Profit & What-if"),
    st.Page("app/page_6_actions.py", title="Actions"),
    st.Page("app/page_7_trust.py", title="Trust")
])
pg.run()
