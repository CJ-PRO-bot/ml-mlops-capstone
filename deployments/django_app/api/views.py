from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Prediction

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_classifier.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
FEATURES = [c for c in df.columns if c not in ("y_cls", "y_reg")]


def health(request):
    return JsonResponse({"status": "ok", "model": "best_classifier", "n_features": len(FEATURES)})


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)

    start = time.time()

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    feats = payload.get("features", {}) or {}

    row = {f: float(feats.get(f, 0.0)) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    y_cls = int(model.predict(X)[0])

    pred_obj = {"y_cls": y_cls}
    latency_ms = int((time.time() - start) * 1000)

    rec = Prediction.objects.create(
        model_name="best_classifier",
        model_version=None,
        input_obj=row,
        pred_obj=pred_obj,
        latency_ms=latency_ms,
    )

    return JsonResponse({"id": rec.id, "y_cls": y_cls, "latency_ms": latency_ms})