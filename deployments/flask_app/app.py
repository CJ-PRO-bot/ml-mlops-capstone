from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from flask import Flask, request, jsonify

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_classifier.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
FEATURES = [c for c in df.columns if c not in ("y_cls", "y_reg")]

app = Flask(__name__)

@app.get("/health")
def health() -> Dict[str, Any]:
    return jsonify({"status": "ok", "model": "best_classifier", "n_features": len(FEATURES)})

@app.get("/")
def root():
    return jsonify({"message": "EcoGuard Lite Flask API running"})

@app.post("/predict")
def predict():
    start = time.time()
    payload = request.get_json(silent=True) or {}
    feats = payload.get("features", {}) or {}

    row = {f: float(feats.get(f, 0.0)) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    y_cls = int(model.predict(X)[0])

    latency_ms = int((time.time() - start) * 1000)
    return jsonify({"y_cls": y_cls, "latency_ms": latency_ms})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8003, debug=True)