import streamlit as st
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

st.title("📊 EDA (Processed Dataset)")

df = pd.read_csv(DATA_PATH)
st.write("Shape:", df.shape)
st.dataframe(df.head(20))

st.subheader("Target distribution (y_cls)")
st.bar_chart(df["y_cls"].value_counts().sort_index())

st.subheader("Correlation (quick view)")
st.dataframe(df.corr(numeric_only=True).round(3))