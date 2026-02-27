from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_classifier.joblib"

DB_CFG = dict(
    host="127.0.0.1",
    port=5433,
    dbname="ecoguard_db",
    user="ecoguard",
    password="ecoguard_pw",
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_classical.py first."
        )
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run preprocessing first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["y_cls", "y_reg"])

    # pick one sample row
    x1 = X.iloc[[0]]

    model = joblib.load(MODEL_PATH)

    t0 = time.time()
    pred = model.predict(x1)[0]
    latency_ms = int((time.time() - t0) * 1000)

    input_json = json.dumps(x1.to_dict(orient="records")[0])
    prediction_json = json.dumps({"y_cls": int(pred)})

    conn = psycopg2.connect(**DB_CFG)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (model_name, model_version, input_json, prediction_json, latency_ms)
        VALUES (%s, %s, %s, %s, %s)
        """,
        ("best_classifier", "v1", input_json, prediction_json, latency_ms),
    )
    conn.commit()
    cur.close()
    conn.close()

    print("✅ Inserted one prediction into Postgres.")
    print("Prediction:", pred, "Latency(ms):", latency_ms)


if __name__ == "__main__":
    main()
