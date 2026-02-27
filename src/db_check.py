import psycopg2

def main():
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5433,
        dbname="ecoguard_db",
        user="ecoguard",
        password="ecoguard_pw",
    )
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print(cur.fetchone()[0])
    cur.close()
    conn.close()
    print("Postgres connection ✅")

if __name__ == "__main__":
    main()
