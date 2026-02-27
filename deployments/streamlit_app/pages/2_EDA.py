import streamlit as st
import pandas as pd
from pathlib import Path

def find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(10):
        if (p / "data").exists():
            return p
        p = p.parent
    return start.parents[2]  # fallback

PROJECT_ROOT = find_repo_root(Path(__file__).resolve())
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

st.title("📊 EDA (Processed Dataset)")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at:\n{DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.write("Shape:", df.shape)
st.dataframe(df.head(20))

if "y_cls" in df.columns:
    st.subheader("Target distribution (y_cls)")
    st.bar_chart(df["y_cls"].value_counts().sort_index())
else:
    st.warning("Column y_cls not found in dataset.")

st.subheader("Correlation (quick view)")
st.dataframe(df.corr(numeric_only=True).round(3))