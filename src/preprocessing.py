from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

# --- Paths (no src import needed, avoids ModuleNotFound issues) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "air_quality_uci"
RAW_CSV = RAW_DIR / "AirQualityUCI.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

# UCI dataset ZIP (contains AirQualityUCI.csv)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
UCI_ZIP_PATH = RAW_DIR / "air_quality.zip"


def _ensure_raw_dataset() -> None:
    """
    Ensure the AirQualityUCI.csv exists.
    In CI/GitHub Actions the dataset may not be committed to the repo,
    so we download it from UCI and extract it to the expected folder.
    """
    # If file exists and looks real, do nothing
    if RAW_CSV.exists():
        try:
            # quick sanity: should have many rows
            preview = pd.read_csv(RAW_CSV, sep=";", decimal=",", nrows=10)
            if len(preview) > 0:
                return
        except Exception:
            # will re-download below
            pass

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download zip
    urlretrieve(UCI_ZIP_URL, UCI_ZIP_PATH)

    # Extract AirQualityUCI.csv
    with ZipFile(UCI_ZIP_PATH, "r") as zf:
        # UCI zip usually contains "AirQualityUCI.csv"
        zf.extract("AirQualityUCI.csv", path=RAW_DIR)

    # Optional: cleanup zip to keep repo/runner tidy
    try:
        UCI_ZIP_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def load_raw() -> pd.DataFrame:
    _ensure_raw_dataset()

    df = pd.read_csv(RAW_CSV, sep=";", decimal=",")
    # drop empty/unnamed columns caused by trailing ';'
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(axis=1, how="all")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # -200 is a missing-value marker in this dataset
    df = df.replace(-200, np.nan)

    # Parse datetime
    dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    tm = pd.to_datetime(df["Time"], format="%H.%M.%S", errors="coerce").dt.time
    df["timestamp"] = pd.to_datetime(dt.astype(str) + " " + tm.astype(str), errors="coerce")

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Regression target
    if "C6H6(GT)" not in df.columns:
        raise KeyError("Expected target column 'C6H6(GT)' not found.")
    df = df.dropna(subset=["C6H6(GT)"]).copy()
    df["y_reg"] = df["C6H6(GT)"]

    # 3-class label from quantiles (0=low,1=mid,2=high)
    q1, q2 = df["y_reg"].quantile([0.33, 0.66]).tolist()

    def to_class(v: float) -> int:
        if v <= q1:
            return 0
        if v <= q2:
            return 1
        return 2

    df["y_cls"] = df["y_reg"].apply(to_class).astype(int)

    # Drop leak/non-usable cols
    df = df.drop(columns=[c for c in ["Date", "Time", "timestamp", "C6H6(GT)"] if c in df.columns])

    # Force numeric for feature columns
    for c in df.columns:
        if c not in ("y_reg", "y_cls"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in df.columns if c not in ("y_reg", "y_cls")]
    medians = df[features].median(numeric_only=True)
    df[features] = df[features].fillna(medians)
    return df


def main() -> None:
    print(f"Reading: {RAW_CSV}")
    df = load_raw()
    print("Raw shape:", df.shape)

    df = clean_and_engineer(df)
    print("After cleaning/targets:", df.shape)

    df = impute_missing(df)
    print("After imputation:", int(df.isna().sum().sum()), "total NaNs")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved processed dataset: {OUT_CSV}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()