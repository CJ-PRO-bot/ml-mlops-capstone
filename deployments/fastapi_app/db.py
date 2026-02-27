from __future__ import annotations
import os
import json
import time
import psycopg2


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5433")),
        dbname=os.getenv("PGDATABASE", "ecoguard_db"),
        user=os.getenv("PGUSER", "ecoguard"),
        password=os.getenv("PGPASSWORD", "ecoguard_pw"),
    )


def insert_prediction(model_name: str, model_version: str | None, input_obj: dict, pred_obj: dict, latency_ms: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (model_name, model_version, input_json, prediction_json, latency_ms)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (
            model_name,
            model_version,
            json.dumps(input_obj),
            json.dumps(pred_obj),
            int(latency_ms),
        ),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return int(new_id)

def fetch_latest_predictions(limit: int = 10):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, model_name, latency_ms, input_json, prediction_json
                FROM predictions
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "created_at": str(r[1]),
                "model_name": r[2],
                "latency_ms": r[3],
                "input": r[4],
                "prediction": r[5],
            }
        )
    return out