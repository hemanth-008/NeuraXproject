import streamlit as st
from app.components import advisory_badge, assumptions_expander, load_artifact, _mtime
import pandas as pd

try:
    advisory_badge()
    st.title("Actions")
    st.caption("ADVISORY / SIMULATED — recommendations are ranked scenarios, not operating instructions.")

    path = "artifacts/recommendations.json"
    payload = load_artifact(path, _mtime(path))
    if payload is None:
        raise FileNotFoundError(f"missing {path}")

    recommendations = payload.get("recommendations", [])
    st.caption(f"Source: {path} · recommendations: {len(recommendations)}")
    rows = []
    for item in recommendations:
        delta = item.get("expected_delta_margin", {})
        rows.append({"id": item.get("id"), "category": item.get("category"), "confidence": item.get("confidence_tier"), "score": item.get("score"), "delta margin/hour": delta.get("mean"), "action": item.get("action"), "tag": item.get("tag", "ADVISORY / SIMULATED")})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        for item in recommendations:
            with st.expander(f"{item.get('id', 'Recommendation')} · {item.get('confidence_tier', 'n/a')}"):
                st.write(item.get("action", ""))
                st.caption(item.get("rationale", ""))
                st.write("Evidence:", ", ".join(map(str, item.get("evidence_ids", []))))
                st.write("Verify:", item.get("how_to_verify", "n/a"))
    else:
        st.info("No recommendations are available in this build.")
    assumptions_expander(["D14", "D18", "D22", "D24"])
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
