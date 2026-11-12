"""
Agents (graph nodes) of the ML workflow.

Each agent is a function: it reads the shared state and returns
only the keys it wants to update.
"""
from sklearn.model_selection import train_test_split

from config import BASE_FEATURES, MIN_ROWS, RANDOM_STATE, TABLE_NAME, TARGET_SOURCE, TEST_SIZE
from db import load_table
from state import AgentState


# ---------------------------------------------------------------------
# Data Agent: runs once, before the loop
# ---------------------------------------------------------------------
def data_agent(state: AgentState) -> dict:
    """Load the cleaned table, validate it against a data contract and split it."""
    df = load_table(TABLE_NAME)

    # 1. Data contract: the workflow only runs on data that passes these checks
    required = ["dbn", TARGET_SOURCE] + BASE_FEATURES
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    if len(df) < MIN_ROWS:
        raise ValueError(f"Only {len(df)} rows; at least {MIN_ROWS} expected")
    if df[required].isna().any().any():
        raise ValueError("Required columns contain missing values")
    if df["dbn"].duplicated().any():
        raise ValueError("Duplicate school IDs (dbn) found")
    if not df[TARGET_SOURCE].between(0, 1).all():
        raise ValueError(f"{TARGET_SOURCE} must be between 0 and 1")

    # 2. Target: 1 if the school's proficiency rate is above the median
    threshold = float(df[TARGET_SOURCE].median())
    df["target"] = (df[TARGET_SOURCE] > threshold).astype(int)

    # 3. Stratified split; the test set stays untouched until the final report
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["target"], random_state=RANDOM_STATE
    )

    summary = {
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "threshold": round(threshold, 4),
        "positive_share": round(float(df["target"].mean()), 3),
    }
    message = (f"Data Agent: loaded {summary['rows']:,} schools, passed the data contract, "
               f"split into {summary['train_rows']} train / {summary['test_rows']} test "
               f"(target: {TARGET_SOURCE} > {threshold:.3f})")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "data_summary": summary,
        "iteration": 0,
        "events": [message],
    }