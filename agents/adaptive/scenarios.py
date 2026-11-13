"""
Test scenarios for the Part 3 adaptive multi-agent system.

Each scenario starts from the clean table produced by the Airflow pipeline
and injects ONE realistic problem: missing values, a unit error, duplicate
rows, a leaky feature, a failing trainer, or too little data.
The evaluation suite runs every system on every scenario and compares
how each system reacts.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from config import BASE_FEATURES, RANDOM_STATE

# Features stored as shares (0-1). total_enrollment is a count, so it is excluded.
SHARE_FEATURES = [c for c in BASE_FEATURES if c != "total_enrollment"]


# ---------------------------------------------------------------------
# Perturbation functions: each one returns a NEW dataframe (never edits
# the input), so every scenario starts from the same clean data.
# ---------------------------------------------------------------------
def no_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def add_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # 5% of schools lose their Economic Need Index (e.g., a failed join upstream)
    out = df.copy()
    rows = out.sample(frac=0.05, random_state=RANDOM_STATE).index
    out.loc[rows, "economic_need_index"] = np.nan
    return out


def add_unit_error(df: pd.DataFrame) -> pd.DataFrame:
    # 3% of schools report pct_ell as 0-100 instead of 0-1 (a classic unit bug)
    out = df.copy()
    candidates = out[out["pct_ell"] > 0.01]
    rows = candidates.sample(frac=0.03, random_state=RANDOM_STATE).index
    out.loc[rows, "pct_ell"] = out.loc[rows, "pct_ell"] * 100
    return out


def add_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # 25 schools appear twice (e.g., an ingestion job ran twice)
    extra = df.sample(n=25, random_state=RANDOM_STATE)
    return pd.concat([df, extra], ignore_index=True)


def keep_few_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Only 100 schools arrive (e.g., a partial download)
    return df.sample(n=100, random_state=RANDOM_STATE).reset_index(drop=True)


# ---------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    perturb: Callable[[pd.DataFrame], pd.DataFrame] = no_change
    extra_features: tuple = ()      # features requested on top of BASE_FEATURES
    trainer_failures: int = 0       # first N training calls crash (transient error)
    failing_models: tuple = ()      # model families that ALWAYS crash


SCENARIOS = {
    s.name: s
    for s in [
        Scenario("clean", "Clean table from the Airflow pipeline"),
        Scenario("missing_values", "5% of economic_need_index values are missing",
                 perturb=add_missing_values),
        Scenario("unit_error", "3% of pct_ell values are on a 0-100 scale",
                 perturb=add_unit_error),
        Scenario("duplicates", "25 schools appear twice",
                 perturb=add_duplicates),
        Scenario("leaky_feature", "mean_scale_score (the exam result itself) is requested as a feature",
                 extra_features=("mean_scale_score",)),
        Scenario("transient_failure", "The first training call crashes",
                 trainer_failures=1),
        Scenario("model_crash", "Random Forest crashes on every attempt",
                 failing_models=("random_forest",)),
        Scenario("too_few_rows", "Only 100 schools are available",
                 perturb=keep_few_rows),
    ]
}


# ---------------------------------------------------------------------
# Quick check: python -m adaptive.scenarios
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from config import TABLE_NAME
    from db import load_table

    base = load_table(TABLE_NAME)
    print(f"Clean table: {len(base)} rows\n")
    print(f"{'scenario':<18} {'rows':>5} {'missing':>8} {'above_1':>8} {'dup_dbn':>8}  extra")
    for s in SCENARIOS.values():
        df = s.perturb(base)
        missing = int(df[SHARE_FEATURES].isna().sum().sum())
        above_1 = int((df[SHARE_FEATURES] > 1).sum().sum())
        dup_dbn = int(df["dbn"].duplicated().sum())
        notes = list(s.extra_features)
        if s.trainer_failures:
            notes.append(f"trainer_failures={s.trainer_failures}")
        notes += [f"failing={m}" for m in s.failing_models]
        print(f"{s.name:<18} {len(df):>5} {missing:>8} {above_1:>8} {dup_dbn:>8}  {', '.join(notes)}")