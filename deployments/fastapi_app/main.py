from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from deployments.fastapi_app.db import insert_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_classifier.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

app = FastAPI(title="EcoGuard Lite API", version="1.0")


class PredictRequest(BaseModel):
    features: Dict[str, float]


# Load model once (this is fine in CI as long as artifact exists)
model = joblib.load(MODEL_PATH)

_FEATURES: Optional[List[str]] = None


def get_features() -> List[str]:
    """
    Determine feature columns without crashing CI.

    Priority:
    1) If dataset exists and SKIP_DATA != 1, read columns from processed CSV.
    2) Else, if model exposes feature_names_in_, use that (best for CI).
    3) Else, fallback to empty list (predict will use request keys).
    """
    global _FEATURES
    if _FEATURES is not None:
        return _FEATURES

    if os.getenv("SKIP_DATA", "0") != "1" and DATA_PATH.exists():
        # Read only header (fast)
        cols = list(pd.read_csv(DATA_PATH, nrows=0).columns)
        _FEATURES = [c for c in cols if c not in ("y_cls", "y_reg")]
        return _FEATURES

    # Use model's learned feature names if available (common in sklearn)
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        _FEATURES = [str(c) for c in list(model_features)]
        return _FEATURES

    _FEATURES = []
    return _FEATURES


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "EcoGuard Lite API running. Visit /docs"}


@app.get("/health")
def health() -> Dict[str, Any]:
    features = get_features()
    return {"status": "ok", "model": "best_classifier", "n_features": len(features)}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    start = time.time()

    features = get_features()

    # If we don't know expected feature list, use request keys (stable order)
    if not features:
        features = sorted(req.features.keys())

    row = {f: float(req.features.get(f, 0.0)) for f in features}
    X = pd.DataFrame([row], columns=features)

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