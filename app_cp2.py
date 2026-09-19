"""
QualityTwin CP2 dashboard (v1) -- Streamlit

Reads only from artifacts/ (predictions.parquet, flow_v0.json, optional gradcam_overlay_grid.png).
No GPU, no live model, no internet needed at demo time (Wi-Fi-off survival is a hard requirement).

Run:
    .venv\\Scripts\\streamlit run app.py

Changes vs v0 (v1.1 adds: image lookup by file name, full-height tables, no delta badge at 0% rework):
  * Cache is keyed on the parquet/json modification time, so a rebuilt artifact shows up without
    restarting Streamlit (v0 kept serving the stale pre-rebuild predictions).
  * Header shows how many units are loaded and when the file was built (staleness check at a glance).
  * Inspect: "Why this verdict" expander shows the three expected costs behind ACCEPT/REJECT/REVIEW.
  * Line Health / Flow: neutral delta colours (a green up-arrow on REJECT count was misleading),
    clean station names, and the rework rate at which the bottleneck saturates.
  * New "Trust & Status" tab: novelty, localization faithfulness, calibration, robustness, limitations,
    build status and assumptions. Numbers there are hand-entered from the fold-0 runs (one seed).
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

ARTIFACTS = Path("artifacts")
IMAGE_ROOT = Path("data/images")  # predictions.image_path is relative to this, e.g. "train/crack/xxx.png"
GRADCAM_GRID = ARTIFACTS / "gradcam_overlay_grid.png"
# If an image isn't at IMAGE_ROOT / image_path, look it up by file name under these folders.
IMAGE_SEARCH_ROOTS = [IMAGE_ROOT, Path("data"), Path("kaggle_upload")]

# Assumed costs (decision D18: no economics sheet exists). Must match the values used to build predictions.parquet.
C_ESCAPE = 50.0
C_FALSE_REJECT = 5.0
C_REVIEW = 1.0

st.set_page_config(page_title="QualityTwin", layout="wide")


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _station(col: str) -> str:
    return col.replace(" Utilization", "").replace(" Util", "")


@st.cache_resource(show_spinner="Indexing local images (first time only)...")
def image_index():
    idx = {}
    for root in IMAGE_SEARCH_ROOTS:
        if root.exists():
            for p in root.rglob("*.png"):
                idx.setdefault(p.name, p)
    return idx


def find_image(rel_path: str):
    direct = IMAGE_ROOT / rel_path
    if direct.exists():
        return direct
    idx = image_index()
    if not idx:
        image_index.clear()  # don't cache an empty index
    return idx.get(Path(rel_path).name)


@st.cache_data
def load_predictions(mtime: float):
    path = ARTIFACTS / "predictions.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["class_probs"] = df["class_probs"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
    if "conformal_set" in df.columns:
        df["conformal_set"] = df["conformal_set"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
    return df


@st.cache_data
def load_flow(mtime: float):
    path = ARTIFACTS / "flow_v0.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


PRED_PATH = ARTIFACTS / "predictions.parquet"
FLOW_PATH = ARTIFACTS / "flow_v0.json"
preds = load_predictions(_mtime(PRED_PATH))
flow = load_flow(_mtime(FLOW_PATH))

st.title("QualityTwin")
st.caption("Software-only quality-aware digital twin. Everything here is ADVISORY / SIMULATED unless stated otherwise.")
if preds is not None:
    built = datetime.fromtimestamp(_mtime(PRED_PATH)).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Loaded artifacts/predictions.parquet: {len(preds)} units, file written {built}.")

tab_inspect, tab_health, tab_flow, tab_trust = st.tabs(["Inspect", "Line Health", "Flow & Bottleneck", "Trust & Status"])

# ---------------- Inspect ----------------
with tab_inspect:
    if preds is None:
        st.warning("No artifacts/predictions.parquet yet.")
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
                img_path = find_image(row["image_path"])
                if img_path is not None:
                    st.image(Image.open(img_path), width=320)
                else:
                    st.info(
                        f"Image {row['image_path']} not found under data/ or kaggle_upload/ -- "
                        "adjust IMAGE_ROOT / IMAGE_SEARCH_ROOTS at the top of this file if your images live elsewhere."
                    )

                m1, m2, m3 = st.columns(3)
                m1.metric("Verdict", row["decision"])
                m2.metric("Predicted family", row["pred_family"])
                m3.metric("Flag", row["flag"])

                st.write("Class probabilities (temperature-scaled)")
                probs_df = pd.DataFrame({"class": list(row["class_probs"].keys()), "p": list(row["class_probs"].values())})
                st.plotly_chart(px.bar(probs_df, x="class", y="p", range_y=[0, 1]), use_container_width=True)

                with st.expander("Why this verdict? (cost-based decision, assumed costs)"):
                    p_def = row.get("p_defect")
                    if p_def is None or pd.isna(p_def):
                        p_def = 1.0 - float(row["class_probs"].get("normal", 0.0))
                    p_def = float(p_def)
                    cost_df = pd.DataFrame(
                        {
                            "action": ["ACCEPT", "REJECT", "REVIEW"],
                            "expected cost": [round(p_def * C_ESCAPE, 3), round((1 - p_def) * C_FALSE_REJECT, 3), C_REVIEW],
                            "formula": ["p(defect) x c_escape", "(1 - p(defect)) x c_false_reject", "c_review"],
                        }
                    )
                    st.write(f"p(defect) = {p_def:.4f}. The rule picks the cheapest action.")
                    st.dataframe(cost_df, hide_index=True)
                    st.caption(
                        f"Assumed costs: escape = {C_ESCAPE:g}, false reject = {C_FALSE_REJECT:g}, review = {C_REVIEW:g} "
                        "(decision D18: no economics sheet was provided). These are placeholders to be replaced by "
                        "real costs, not measurements."
                    )

                conf_set = row.get("conformal_set")
                if conf_set is not None:
                    st.write(f"Conformal set (90% target coverage): {', '.join(conf_set) if len(conf_set) else 'empty'}")

                st.caption(
                    "Confidence is temperature-scaled and the decision above is the cost-based ACCEPT/REJECT/"
                    "REVIEW rule. The temperature fit hit its search bound and the conformal q_hat came out ~0, "
                    "because this benchmark has almost no calibration error to correct -- an honest result of how "
                    "clean the data is, not a bug, but it also means this set will rarely show UNCERTAIN. "
                    "Per-unit heatmaps are not wired in here yet; see the Trust & Status tab for the Grad-CAM++ overlay grid."
                )
                if row.get("true_label") is not None:
                    st.caption(f"Ground truth (dev/demo use only, not a live label): {row['true_label']}")
            else:
                st.info("Pick a unit from the filters on the left.")

# ---------------- Line Health ----------------
with tab_health:
    if preds is None:
        st.warning("No artifacts/predictions.parquet yet.")
    else:
        n = len(preds)
        n_accept = int((preds["decision"] == "ACCEPT").sum())
        n_reject = int((preds["decision"] == "REJECT").sum())
        n_review = int((preds["decision"] == "REVIEW").sum())
        n_uncertain = int((preds["flag"] == "UNCERTAIN").sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Units scored", n)
        c2.metric("ACCEPT", n_accept, f"{n_accept/n:.1%}", delta_color="off")
        c3.metric("REJECT", n_reject, f"{n_reject/n:.1%}", delta_color="off")
        c4.metric("REVIEW", n_review, f"{n_review/n:.1%}", delta_color="off")
        c5.metric("UNCERTAIN flags", n_uncertain)

        st.warning(
            f"This set is class-balanced (~20% per family), so the {n_reject/n:.0%} reject rate is NOT a "
            "real production defect rate -- it's an artifact of the labeled benchmark. Don't quote it as "
            "factory yield. The Flow tab uses a separate, adjustable assumed rate for the simulation."
        )

        by_family = preds["pred_family"].value_counts().reset_index()
        by_family.columns = ["family", "count"]
        st.plotly_chart(px.bar(by_family, x="family", y="count", title="Predicted family distribution"), use_container_width=True)

        st.caption(
            "A time-ordered live-replay ticker (FPY/OEE/margin over time) isn't built yet -- predictions.parquet "
            "has no real timestamps (ts is null per D17). This page is a static snapshot."
        )

# ---------------- Flow & Bottleneck ----------------
with tab_flow:
    if flow is None:
        st.warning("No artifacts/flow_v0.json yet -- run scripts/flow_v0_report.py")
    else:
        model_choice = st.radio("Production model", ["model1", "model2"], horizontal=True)
        m = flow[model_choice]
        util_rows = [r for r in m["summary"] if r["kind"] == "utilization"]
        util_df = pd.DataFrame(util_rows)
        util_df["station"] = util_df["column"].apply(_station)
        fig = px.bar(util_df, x="station", y="mean", title=f"{model_choice}: mean station utilization (averaged over the simulated demand sweep)")
        fig.add_hline(y=m["bottleneck_v0"]["utilization"], line_dash="dot", annotation_text="bottleneck")
        st.plotly_chart(fig, use_container_width=True)
        st.metric(
            "Candidate bottleneck",
            _station(m["bottleneck_v0"]["candidate_bottleneck"]),
            f"{m['bottleneck_v0']['utilization']:.1%} utilization",
            delta_color="off",
        )

        if model_choice == "model2":
            drill_q = next(r for r in m["summary"] if r["column"] == "Drilling Queue Time")["mean"]
            asm_q = next(r for r in m["summary"] if r["column"] == "Assembly Queue Time")["mean"]
            if drill_q > asm_q:
                st.caption(
                    f"Note: Drilling's mean queue time ({drill_q:.2f}) exceeds Assembly's ({asm_q:.2f}) here, "
                    "even though Assembly has higher utilization. The bottleneck call isn't fully triangulated "
                    "on this model -- disclosed, not hidden."
                )

        st.divider()
        st.subheader("Simulated quality-to-flow coupling (ADVISORY / SIMULATED)")
        st.caption(
            "No batch, rework, scrap or cost fields exist in the production data (decision D14). This is an "
            "explicit, editable assumption, not a measurement: every rejected unit is assumed to cost the "
            "bottleneck station exactly one extra pass (a rework/re-inspection loop), holding cycle time, "
            "machine count, availability and changeover loss fixed. Utilization scales linearly with that "
            "extra-visits rate: rho_new = rho_baseline * (1 + rework_rate)."
        )
        rework_rate = st.slider("Assumed rework/defect rate at the bottleneck (%)", 0, 50, 3) / 100
        rho0 = m["bottleneck_v0"]["utilization"]
        rho_new = rho0 * (1 + rework_rate)
        rate_at_saturation = (1.0 / rho0 - 1.0) if rho0 > 0 else float("inf")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline bottleneck utilization", f"{rho0:.1%}")
        c2.metric(
            "Simulated utilization at this rework rate",
            f"{rho_new:.1%}",
            f"{rho_new-rho0:+.1%}" if rework_rate > 0 else None,
            delta_color="inverse",
        )
        c3.metric("Rework rate that saturates the bottleneck", f"{rate_at_saturation:.1%}")
        if rho_new >= 1.0:
            st.error(
                "At this rework rate the bottleneck would exceed 100% utilization -- demand can't be "
                "sustained without dropping units or building backlog. This is where quality cost becomes "
                "a flow problem, not just a scrap-cost problem."
            )

# ---------------- Trust & Status ----------------
with tab_trust:
    st.caption(
        "Numbers on this page are hand-entered from the fold-0 runs (one seed, group-aware split). "
        "The locked final holdout (1,457 images) has not been touched and is not reported here."
    )

    st.subheader("Novelty: does it notice a defect family it never saw?")
    st.write("Leave-one-family-out: `scratch` hidden from training, then scored as unknown.")
    nov = pd.DataFrame(
        {
            "detector": ["Softmax (max probability)", "Energy score", "kNN embedding distance"],
            "AUROC (known vs hidden family)": [0.822, 0.791, 0.991],
        }
    )
    st.plotly_chart(px.bar(nov, x="detector", y="AUROC (known vs hidden family)", range_y=[0, 1]), use_container_width=True)
    st.write(
        "The kNN embedding-distance detector flagged 97.7% of the hidden-family images at a 5% false-positive "
        "threshold. The classifier's own confidence is a much weaker novelty signal."
    )
    st.caption("Only `scratch` has been held out so far; crack, hole and rust were not run.")

    st.subheader("Localization: is the heatmap looking at the defect?")
    l1, l2 = st.columns(2)
    l1.metric("Deletion AUC (lower is better)", "0.397")
    l2.metric("Insertion AUC (higher is better)", "0.826")
    if GRADCAM_GRID.exists():
        st.image(str(GRADCAM_GRID), width=900, caption="Grad-CAM++ overlays, 3 samples per class")
    else:
        st.info("Overlay grid not found at artifacts/gradcam_overlay_grid.png -- copy it from the Kaggle notebook output.")
    st.caption(
        "Crack, scratch and hole localize tightly on the defect. Normal is diffuse (nothing local to point at), "
        "and rust is noticeably more diffuse than the other defect classes -- a real limitation, not hidden. "
        "The data has image-level labels only, so no IoU/Dice/PRO is claimed; this is heatmap-based localization."
    )

    st.subheader("Calibration and decision")
    k1, k2, k3 = st.columns(3)
    k1.metric("ECE (raw and scaled)", "0.0006")
    k2.metric("Conformal coverage (target 90%)", "99.9%")
    k3.metric("Singleton prediction sets", "100%")
    st.caption(
        "The temperature fit hit its search-bound lower limit (T = 0.05) and split conformal gave q_hat = 0: the "
        "model is already almost perfectly calibrated on this benchmark, so conformal sets are technically valid "
        "but never abstain here. The abstention story on this dataset comes from the novelty detector above, not "
        "from conformal sets. The ACCEPT/REJECT/REVIEW decision uses assumed costs "
        f"(escape {C_ESCAPE:g}, false reject {C_FALSE_REJECT:g}, review {C_REVIEW:g})."
    )

    st.subheader("Robustness to corrupted images")
    rob = pd.DataFrame(
        {
            "corruption (validation fold)": ["Noise, sigma 10", "JPEG, quality 20", "Blur, sigma 3", "Low-res, 4x"],
            "baseline (no corruption aug)": ["0.345", "0.276", "0.282", "0.497"],
            "with corruption augmentation": [">= 0.98", ">= 0.98", ">= 0.98", ">= 0.98"],
        }
    )
    st.table(rob.set_index("corruption (validation fold)"))
    st.caption(
        "Corruption families not trained on also held up: motion blur 0.998, speckle 0.998, noise sigma 40 0.980. "
        "Clean accuracy on the unseen-brightness holdout: 0.9994. Baseline scores are from the stress suite as logged."
    )

    st.subheader("Build status")
    status = pd.DataFrame(
        {
            "component": [
                "Vision baseline + corruption augmentation",
                "Stress suite",
                "Novelty (open-set) detection",
                "Localization (Grad-CAM++ + faithfulness)",
                "Calibration, conformal, cost-based decision",
                "Flow / bottleneck v0 (Model 1, Model 2)",
                "Dashboard (Inspect, Line Health, Flow, Trust)",
                "Root-cause engine",
                "Recommendations engine",
                "One-command pipeline (run_pipeline.py)",
                "Folds 1-4, other hidden families, final locked-holdout run",
            ],
            "status": [
                "built (fold 0, one seed)",
                "built (fold 0)",
                "built (scratch only)",
                "built",
                "built (assumed costs)",
                "built (Model 2 bottleneck not fully triangulated)",
                "built (v1)",
                "not started",
                "not started",
                "not started",
                "not started (holdout runs exactly once, at the end)",
            ],
        }
    )
    st.table(status.set_index("component"))

    st.subheader("Assumptions and limits")
    st.markdown(
        "- The organizers provided no economics sheet, hidden-test format, or checkpoint times, so costs and formats are stated assumptions.\n"
        "- Inspection images and production data share no keys; the quality-to-flow link is an editable assumption, not a measured relationship.\n"
        "- The production data has no batch, defect, scrap, rework or cost fields, so no real root cause can be mined from it; "
        "any root-cause output will be labeled simulated.\n"
        "- The image set is class-balanced, so reject rates on it are not factory yield.\n"
        "- Pretrained ImageNet weights are assumed to be allowed. All numbers are fold 0, single seed."
    )

st.divider()
st.caption("QualityTwin -- Neurax Hackathon 3.0. Software-only. No camera, PLC or robot. All recommendations advisory.")
