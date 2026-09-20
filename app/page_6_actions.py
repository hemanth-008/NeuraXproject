import streamlit as st
from app.components import advisory_badge

try:
    advisory_badge()
    st.title("Actions")
    st.info("Artifact missing: run recommendations pipeline")
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
