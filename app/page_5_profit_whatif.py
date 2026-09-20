import streamlit as st
from app.components import advisory_badge

try:
    advisory_badge()
    st.title("Profit & What-if")
    st.info("Artifact missing: run profit pipeline")
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
