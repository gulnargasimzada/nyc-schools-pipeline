"""
Configuration shared by all agents.
"""
import os
from pathlib import Path

# Connection to the schools database (set in docker-compose.yaml)
DB_URL = os.environ["SCHOOLS_DB_URL"]

# Table produced by the Airflow pipeline in Part 1
TABLE_NAME = "school_math_demographics"

# Target: a school is "high proficiency" (1) if its rate is above the median
TARGET_SOURCE = "pct_proficient"

# Same features and split settings as the Part 1 baseline model,
# so the agents start from a directly comparable point
BASE_FEATURES = [
    "economic_need_index",
    "pct_ell",
    "pct_students_with_disabilities",
    "pct_female",
    "total_enrollment",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5          # Folds for cross-validation inside the loop

# Data contract: the agents refuse to run on a table smaller than this
MIN_ROWS = 500

# The 3rd character of a DBN identifies the borough (e.g. 01M015 -> M -> Manhattan)
BOROUGHS = {"M": "Manhattan", "X": "Bronx", "K": "Brooklyn", "Q": "Queens", "R": "Staten Island"}

# ---------------------------------------------------------------------
# Search space explored by the Planner Agent
# ---------------------------------------------------------------------
# First experiment: reproduces the Part 1 baseline model
BASELINE_PLAN = {
    "name": "baseline",
    "feature_set": "base",
    "model": "logistic_regression",
    "params": {},
}

FEATURE_SETS = ["base", "base_plus_borough"]
MODEL_FAMILIES = ["logistic_regression", "random_forest", "gradient_boosting"]

# Parameter variants tried in the tuning stage (all lean towards less overfitting)
TUNING_GRID = {
    "logistic_regression": [
        {"C": 0.1},                                            # stronger regularization
        {"C": 10.0},                                           # weaker regularization
    ],
    "random_forest": [
        {"n_estimators": 300, "min_samples_leaf": 5},          # smoother trees
        {"n_estimators": 300, "max_depth": 6},                 # shallower trees
    ],
    "gradient_boosting": [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8},
    ],
}

# ---------------------------------------------------------------------
# Loop control
# ---------------------------------------------------------------------
MAX_ITERATIONS = 12       # Hard limit on the number of experiments
PATIENCE = 6              # Stop after this many experiments without a real improvement
MIN_IMPROVEMENT = 0.005   # A new best must beat the old one by at least this CV accuracy
OVERFIT_LIMIT = 0.05      # Training accuracy minus CV accuracy above this counts as overfitting
RECURSION_LIMIT = 100     # LangGraph safety limit on the number of agent steps

# Folder for the final report (inside agents/, so it is visible on the host)
RESULTS_DIR = Path(__file__).parent / "results"