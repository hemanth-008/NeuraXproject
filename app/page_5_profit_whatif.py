import streamlit as st
from app.components import advisory_badge

try:
    advisory_badge()
    st.title("Profit & What-if")
    raise FileNotFoundError("profit artifact is unavailable in this build")
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
