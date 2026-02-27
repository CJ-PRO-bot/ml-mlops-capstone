import os
import streamlit as st
import requests

# In docker, service name works: http://fastapi:8000
# Outside docker (local run), use http://127.0.0.1:8000
API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("🔮 Predict (via FastAPI)")

# 1) Health check (safe)
with st.expander("API Status", expanded=True):
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok:
            st.success("FastAPI is up ✅")
            try:
                st.json(r.json())
            except Exception:
                st.code(r.text)
        else:
            st.error(f"FastAPI responded but not OK ❌  (status={r.status_code})")
            st.code(r.text)
            st.stop()
    except Exception as e:
        st.error("FastAPI is NOT reachable ❌")
        st.code(str(e))
        st.info(
            "If running Streamlit locally (not docker), set API_URL=http://127.0.0.1:8000\n"
            "If running in docker, API_URL=http://fastapi:8000"
        )
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

# 3) Predict
if st.button("Predict"):
    payload = {"features": features}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        st.write("Status:", r.status_code)

        if r.ok:
            st.json(r.json())
        else:
            st.error("Prediction failed")
            # Show raw body for debugging (often stacktrace or message)
            st.code(r.text)

    except Exception as e:
        st.error("Prediction request crashed")
        st.code(str(e))