from __future__ import annotations

import pandas as pd

from src.preprocessing import load_raw, clean_and_engineer, impute_missing


def test_load_raw_has_columns():
    df = load_raw()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 5000  # dataset requirement
    assert "Date" in df.columns
    assert "Time" in df.columns


def test_clean_and_engineer_creates_targets():
    raw = load_raw()
    df = clean_and_engineer(raw)

    assert "y_reg" in df.columns
    assert "y_cls" in df.columns
    assert df["y_cls"].dtype.kind in ("i", "u")  # int/uint
    assert df.isna().sum().sum() >= 0  # just sanity


def test_impute_removes_nans():
    raw = load_raw()
    df = clean_and_engineer(raw)
    df2 = impute_missing(df)
    assert df2.isna().sum().sum() == 0