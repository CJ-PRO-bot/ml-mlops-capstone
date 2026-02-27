import streamlit as st

st.set_page_config(page_title="EcoGuard Lite", layout="wide")
st.title("🌿 EcoGuard Lite")
st.write(
    "Demo dashboard for Air Quality classification/regression + MLflow + Postgres logging."
)
st.markdown("- Go to **Predict** to call the FastAPI `/predict` endpoint.")
st.markdown("- Go to **EDA** to view dataset charts.")
st.markdown(
    "- Go to **Monitoring** to view latest predictions from Postgres (optional)."
)
