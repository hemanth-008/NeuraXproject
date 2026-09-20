import streamlit as st
from app.components import advisory_badge, assumptions_expander, load_artifact, _mtime
import pandas as pd

try:
    advisory_badge()
    st.title("Root Cause")
    st.caption("ADVISORY / SIMULATED — association, not proven causation; labeled-simulated layer D24.")

    path = "artifacts/evidence_cards.json"
    payload = load_artifact(path, _mtime(path))
    if payload is None:
        raise FileNotFoundError(f"missing {path}")

    cards = payload.get("cards", [])
    st.caption(f"Source: {path} · evidence cards: {len(cards)}")
    if not cards:
        st.info("No root-cause evidence cards are available in this build.")
    else:
        summary = pd.DataFrame([
            {"id": card.get("id"), "family": card.get("family"), "confidence": card.get("confidence_tier"), "effect size": card.get("effect_size"), "adjusted p": card.get("p_adj"), "hypothesis": card.get("hypothesis")}
            for card in cards
        ])
        st.dataframe(summary, width="stretch", hide_index=True)
        for card in cards:
            with st.expander(f"{card.get('id', 'Evidence card')} · {card.get('confidence_tier', 'n/a')}"):
                st.write(card.get("hypothesis", ""))
                st.caption(card.get("caveat", "association, not proven causation"))
                if card.get("variables"):
                    st.write("Variables:", ", ".join(map(str, card["variables"])))
                if card.get("verify_action"):
                    st.write("Verify:", card["verify_action"])

    assumptions_expander(["D24"])
except Exception:
    st.warning("This page is not available in this build — see README for known limitations.")
