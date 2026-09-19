"""
src/ingest/loaders.py

Loaders for the Manufacturing Data Shared Facility DES dataset
(Mendeley, UNSW/Arena, CC BY 4.0). Column names and shapes here are
taken directly from DATA_INVENTORY.md -- nothing invented.

Model 1: 3,000 x 10  -- one row per replicate, swept `Demand`, 3 stations
Model 2: 3,000 x 17  -- same design, 2 part types, 3 stations
Model 3: 605,620 x 78 -- ~20 stations, no swept per-row parameter;
         row semantics (replicates-at-a-snapshot vs. true time series)
         are unresolved -- see check_model3_time_structure() below.
"""
from pathlib import Path
import pandas as pd

DEFAULT_DATA_DIR = Path("data")


def _read_csv_clean(path: Path) -> pd.DataFrame:
    """
    Read a CSV and drop empty trailing 'Unnamed: N' columns. Model_1.csv on
    disk carries 12 of them (trailing commas), so pandas reports 22 columns
    while the 10 real columns match the data inventory exactly.
    """
    df = pd.read_csv(path)
    junk = [c for c in df.columns
            if str(c).startswith("Unnamed:") and df[c].isna().all()]
    return df.drop(columns=junk)


def load_model1(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = data_dir / "Model 1" / "Model_1.csv"
    df = _read_csv_clean(path)
    assert df.shape[1] == 10, f"Model 1 expected 10 cols, got {df.shape[1]}"
    df["source_model"] = "model1"
    return df


def load_model2(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = data_dir / "Model 2" / "Model_2.csv"
    df = _read_csv_clean(path)
    assert df.shape[1] == 17, f"Model 2 expected 17 cols, got {df.shape[1]}"
    df["source_model"] = "model2"
    return df


def load_model3(
    data_dir: Path = DEFAULT_DATA_DIR,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Model 3's CSV is ~297 MB / 605,620 rows / 78 cols. Always pass
    usecols or nrows unless you actually need the full frame in memory.
    """
    path = data_dir / "Model 3" / "Model_3.csv"
    df = pd.read_csv(path, usecols=usecols, nrows=nrows)
    return df


def check_model3_time_structure(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """
    Resolves the open question flagged in DATA_INVENTORY.md: in the
    first-10,000-row sample, Time_Now was constant at 24. Reading just
    this column over the full 605,620 rows is cheap and tells us
    whether Model 3 is a real time series or many replicates logged at
    fixed snapshots -- this decides how M4/M5 may use it.
    """
    df = load_model3(data_dir, usecols=["Time_Now"])
    vc = df["Time_Now"].value_counts().sort_index()
    return {
        "n_rows": len(df),
        "n_unique_time_now": int(df["Time_Now"].nunique()),
        "min": float(df["Time_Now"].min()),
        "max": float(df["Time_Now"].max()),
        "value_counts_head": vc.head(10).to_dict(),
        "interpretation": (
            "single snapshot across all replicates"
            if df["Time_Now"].nunique() == 1
            else "varies -- likely a real time series, confirm ordering"
        ),
    }


if __name__ == "__main__":
    import json

    m1 = load_model1()
    m2 = load_model2()
    print(f"Model 1: {m1.shape}, Demand range {m1['Demand'].min()}-{m1['Demand'].max()}")
    print(f"Model 2: {m2.shape}, Demand range {m2['Demand'].min()}-{m2['Demand'].max()}")
    print("Model 3 Time_Now check (full file, one column):")
    print(json.dumps(check_model3_time_structure(), indent=2, default=str))
