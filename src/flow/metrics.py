"""
src/flow/metrics.py

Flow & bottleneck, v0. This is deliberately the crude first pass:
rank stations by utilization straight from the data, no triangulation
yet (that needs queue *and* starvation evidence together -- full M4).

Does NOT compute the full loss waterfall (downtime/changeover/scrap/
rework are not fields in this dataset -- see README Sec 4) and does
NOT need m_s/CT_s/A_s yet; those come once station parameters are
defined (see the note at the bottom of this file).
"""
import pandas as pd


def station_columns(df: pd.DataFrame) -> dict:
    """Split a model's columns into utilization-like and queue-like groups."""
    # Column naming differs across models: "Drilling Util" (M1, space),
    # "Drilling Utilization" (M2), "Cell2_Util" (M3). Match on "Util" alone.
    util_cols = [c for c in df.columns if "Util" in c]
    # Queue-like = queue columns plus M1's "* Waiting Time" columns.
    queue_cols = [
        c for c in df.columns
        if ("Queue" in c or "Waiting" in c) and c not in util_cols
    ]
    return {"util": util_cols, "queue": queue_cols}


def station_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / p95 / max for every utilization and queue column found."""
    cols = station_columns(df)
    rows = []
    for c in cols["util"]:
        s = df[c]
        rows.append({"column": c, "kind": "utilization",
                      "mean": s.mean(), "p95": s.quantile(0.95), "max": s.max()})
    for c in cols["queue"]:
        s = df[c]
        rows.append({"column": c, "kind": "queue",
                      "mean": s.mean(), "p95": s.quantile(0.95), "max": s.max()})
    out = pd.DataFrame(rows)
    return out.sort_values(["kind", "mean"], ascending=[True, False]).reset_index(drop=True)


def bottleneck_v0(df: pd.DataFrame) -> dict:
    """
    Naive single-signal bottleneck call: the station with the highest
    mean utilization. This is ONLY the utilization leg of the
    triangulation the README promises (utilization + upstream queue +
    downstream starvation) -- use this to sanity-check station names
    and ranges, not as the final bottleneck call.
    """
    util_cols = station_columns(df)["util"]
    if not util_cols:
        return {"error": "no utilization columns found in this frame"}
    means = df[util_cols].mean().sort_values(ascending=False)
    return {
        "candidate_bottleneck": means.index[0],
        "utilization": float(means.iloc[0]),
        "ranking": means.round(4).to_dict(),
    }


# --- Note on the full M4 formulas (not implemented here yet) ---
# Max good throughput = min_s [ m_s . A_s . (1 - CO_s) / (V_s . CT_s) ]
# rho_s = lambda . V_s . CT_s / (m_s . A_s . (1 - CO_s))
# These need m_s (machines), CT_s (cycle time), A_s (availability),
# CO_s (changeover loss) and V_s (visits per good unit) per station --
# none of these are columns in the data. They come either from
# ParametersFile.xls (Model 3 only, 50 rows of Arena config) or from
# explicit assumptions we state in config.yaml. Next step after this
# v0 report is deciding that station parameter table.
