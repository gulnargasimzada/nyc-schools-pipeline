"""
Configuration shared by all agents.
"""
import os

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

# First experiment: reproduces the Part 1 baseline model
BASELINE_PLAN = {
    "iteration": 1,
    "name": "baseline",
    "feature_set": "base",
    "model": "logistic_regression",
    "params": {},
}