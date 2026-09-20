import streamlit as st
from app.components import advisory_badge, assumptions_expander, load_artifact, _mtime
import pandas as pd
import plotly.express as px

try:
    advisory_badge()
    st.title("Profit & What-if")
    st.caption("ADVISORY / SIMULATED — economic inputs are assumed (D18/D22); outputs are scenario estimates.")

    model_path = "artifacts/profit_model.json"
    results_path = "artifacts/sim_results.parquet"
    model = load_artifact(model_path, _mtime(model_path))
    results = load_artifact(results_path, _mtime(results_path))
    if model is None or results is None:
        raise FileNotFoundError("profit artifacts are unavailable")

    meta = model.get("meta", {})
    st.caption(f"Sources: {model_path}, {results_path} · tag: {meta.get('tag', 'ADVISORY / SIMULATED')}")
    metrics = model.get("out_of_sample_metrics", {})
    waterfall = model.get("margin_waterfall", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", metrics.get("rows_scored", "n/a"))
    c2.metric("Mean margin/hour", f"{metrics['margin_mean']:.2f}" if "margin_mean" in metrics else "n/a")
    c3.metric("Margin range", f"{metrics['margin_min']:.2f}–{metrics['margin_max']:.2f}" if "margin_min" in metrics and "margin_max" in metrics else "n/a")

    st.subheader("Margin waterfall")
    if waterfall:
        st.dataframe(pd.DataFrame([waterfall]), width="stretch", hide_index=True)

    st.subheader("Scenario grid")
    view = results.copy()
    models = sorted(view["model"].dropna().astype(str).unique().tolist()) if "model" in view else []
    if models:
        selected_model = st.selectbox("Production model", models)
        view = view.loc[view["model"].astype(str) == selected_model].copy()
    if not view.empty and {"demand_level", "throughput_mean", "margin_mean"}.issubset(view.columns):
        chart = px.line(view.sort_values("demand_level"), x="demand_level", y="margin_mean", color="capacity_plus_at_bottleneck", markers=True, title="Simulated margin by demand and bottleneck capacity")
        st.plotly_chart(chart, width="stretch")
    st.dataframe(view, width="stretch", hide_index=True)
    assumptions_expander(["D14", "D18", "D22"])
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
