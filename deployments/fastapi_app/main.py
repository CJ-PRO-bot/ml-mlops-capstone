from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from deployments.fastapi_app.db import insert_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_classifier.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
FEATURES = [c for c in df.columns if c not in ("y_cls", "y_reg")]

app = FastAPI(title="EcoGuard Lite API", version="1.0")


class PredictRequest(BaseModel):
    features: Dict[str, float]


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "EcoGuard Lite API running. Visit /docs"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": "best_classifier", "n_features": len(FEATURES)}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    start = time.time()

    row = {f: float(req.features.get(f, 0.0)) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    y_cls = int(model.predict(X)[0])
    pred_obj = {"y_cls": y_cls}
    latency_ms = int((time.time() - start) * 1000)

    # In CI or when DB not available, skip DB logging
    if os.getenv("SKIP_DB", "0") == "1":
        return {"id": None, "y_cls": y_cls, "latency_ms": latency_ms}

    new_id = insert_prediction(
        model_name="best_classifier",
        model_version=None,
        input_obj=row,
        pred_obj=pred_obj,
        latency_ms=latency_ms,
    )

    return {"id": new_id, "y_cls": y_cls, "latency_ms": latency_ms}