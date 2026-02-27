import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("🔮 Predict (via FastAPI)")

# 1) Check API health safely (no crash)
with st.expander("API Status", expanded=True):
    try:
        r = requests.get(f"{API}/health", timeout=3)
        st.success(f"FastAPI is up ✅  {r.json()}")
    except Exception as e:
        st.error("FastAPI is NOT running ❌")
        st.code(str(e))
        st.info("Start FastAPI in another terminal:\nuvicorn deployments.fastapi_app.main:app --reload --port 8000")
        st.stop()

# 2) Inputs
default = {
    "CO(GT)": 2.6, "PT08.S1(CO)": 1360, "NMHC(GT)": 150, "PT08.S2(NMHC)": 1046,
    "NOx(GT)": 166, "PT08.S3(NOx)": 1056, "NO2(GT)": 113, "PT08.S4(NO2)": 1692,
    "PT08.S5(O3)": 1268, "T": 13.6, "RH": 48.9, "AH": 0.757,
    "hour": 18, "dayofweek": 3, "month": 3
}

st.subheader("Input features")
features = {}
cols = st.columns(3)
keys = list(default.keys())
for i, k in enumerate(keys):
    with cols[i % 3]:
        features[k] = st.number_input(k, value=float(default[k]))

# 3) Predict only when button pressed (important)
if st.button("Predict"):
    payload = {"features": features}
    try:
        r = requests.post(f"{API}/predict", json=payload, timeout=30)
        st.write("Status:", r.status_code)
        st.json(r.json())
    except Exception as e:
        st.error("Prediction request failed")
        st.code(str(e))