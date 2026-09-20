import streamlit as st
from app.components import advisory_badge

try:
    advisory_badge()
    st.title("Root Cause")
    raise FileNotFoundError("root-cause artifact is unavailable in this build")
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
