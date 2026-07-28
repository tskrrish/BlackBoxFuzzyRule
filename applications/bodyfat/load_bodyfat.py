"""
load_bodyfat.py — load the REAL StatLib body-fat dataset (Penrose 1985, 252 men).

--------------------------------------------------------------------------------
GETTING THE CSV 
--------------------------------------------------------------------------------
The canonical source is the StatLib "bodyfat" dataset, also mirrored on Kaggle
("fedesoriano/body-fat-prediction-dataset") and the UCI-style CMU StatLib page.

Any of these give the same 252 rows. Save the file as:

    applications/bodyfat/data/bodyfat.csv

Column names vary by mirror. This loader normalizes them. It expects (at least)
columns for chest, abdomen, hip circumference (cm) and body-fat percent. The
Kaggle CSV headers are: Density, BodyFat, Age, Weight, Height, Neck, Chest,
Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist.

If your mirror uses inches/pounds, the values still work for the geometry; only
the human-readable "high/low" labels are unit-agnostic (they use percentiles).
--------------------------------------------------------------------------------
"""
import os
import pandas as pd

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "data", "bodyfat.csv")

# what the rest of the pipeline expects, and the aliases we accept from mirrors
CANONICAL = {
    "bodyfat": ["bodyfat", "body_fat", "body fat", "pct_bf", "brozek", "siri"],
    "chest":   ["chest"],
    "abdomen": ["abdomen", "abdomen2", "waist"],
    "hip":     ["hip"],
    "thigh":   ["thigh"],
    "neck":    ["neck"],
}

FEATS = ["abdomen", "hip", "chest"]
TARGET = "bodyfat"
COLS = FEATS + [TARGET]


def _normalize_columns(df):
    """Rename whatever the mirror called things to our canonical lowercase names."""
    lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lower)
    rename = {}
    for canon, aliases in CANONICAL.items():
        for a in aliases:
            if a in df.columns:
                rename[a] = canon
                break
    return df.rename(columns=rename)


def load(csv_path=CSV_PATH, require=COLS):
    """Load the real body-fat CSV as a DataFrame with canonical column names.

    Raises a clear, actionable error if the file is missing so it's obvious the
    one manual step (download the CSV) hasn't been done yet.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Real body-fat CSV not found at:\n    {csv_path}\n\n"
            "Download it (see the header of load_bodyfat.py for sources) and save "
            "it there. Until then the application scripts cannot run on real data."
        )
    df = _normalize_columns(pd.read_csv(csv_path))
    missing = [c for c in require if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV loaded but missing expected column(s): {missing}. "
            f"Got columns: {list(df.columns)}. "
            "Edit the CANONICAL alias map in load_bodyfat.py to match your mirror."
        )
    # drop obvious bad rows (a couple of famous outliers/data-entry errors exist
    # in this dataset: bodyfat==0, or an implausible height/weight). We keep it
    # minimal and honest: only remove non-physical body-fat values.
    df = df[(df[TARGET] > 0) & (df[TARGET] < 60)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    try:
        df = load()
    except FileNotFoundError as e:
        print(e)
    else:
        print(f"Loaded {len(df)} real rows.")
        print("Columns:", list(df.columns))
        print("\nRealized correlations (the real thing, not reconstructed):")
        print(df[[c for c in COLS if c in df.columns]].corr().round(2).to_string())