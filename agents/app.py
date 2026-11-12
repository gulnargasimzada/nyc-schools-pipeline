"""
Live agent graph UI for the agent-based ML workflow (Part 2).

Step P2.1: connection check only. The page reads the cleaned table
produced by the Airflow pipeline (Part 1) and shows a short summary.
"""
import os

import pandas as pd
import psycopg2
import streamlit as st

# Connection string to the schools database, set in docker-compose.yaml
DB_URL = os.environ["SCHOOLS_DB_URL"]
TABLE_NAME = "school_math_demographics"


@st.cache_data
def load_table() -> pd.DataFrame:
    """Read the cleaned table from PostgreSQL (cached between page refreshes)."""
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {TABLE_NAME};")
            columns = [col[0] for col in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


st.title("NYC Schools: Agent-Based ML Workflow")

df = load_table()

col1, col2, col3 = st.columns(3)
col1.metric("Schools", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Median proficiency", f"{df['pct_proficient'].median():.1%}")

st.subheader("Data from the Part 1 pipeline")
st.dataframe(df.head(20))