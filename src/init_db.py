import psycopg2

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),

    model_name TEXT NOT NULL,
    model_version TEXT,

    input_json TEXT NOT NULL,
    prediction_json TEXT NOT NULL,

    latency_ms INTEGER
);
"""


def main():
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5433,
        dbname="ecoguard_db",
        user="ecoguard",
        password="ecoguard_pw",
    )
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    cur.close()
    conn.close()
    print("Table 'predictions' ready ✅")


if __name__ == "__main__":
    main()
