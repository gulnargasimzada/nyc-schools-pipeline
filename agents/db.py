"""
Database helper: read the cleaned table produced by the Airflow pipeline.
"""
import pandas as pd
import psycopg2

from config import DB_URL


def load_table(table_name: str) -> pd.DataFrame:
    """Read a whole table from PostgreSQL into a pandas DataFrame."""
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            # ORDER BY makes the row order (and therefore the train/test split) reproducible
            cur.execute(f"SELECT * FROM {table_name} ORDER BY dbn;")
            columns = [col[0] for col in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)