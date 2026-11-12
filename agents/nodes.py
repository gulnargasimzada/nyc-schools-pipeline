"""
Agents (graph nodes) of the ML workflow.

Each agent is a function: it reads the shared state and returns
only the keys it wants to update.
"""
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from config import (BASE_FEATURES, CV_FOLDS, MIN_ROWS, RANDOM_STATE, TABLE_NAME,
                    TARGET_SOURCE, TEST_SIZE)
from db import load_table
from ml import build_features, make_model
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


# ---------------------------------------------------------------------
# Feature Agent: prepares the features for the planned experiment
# ---------------------------------------------------------------------
def feature_agent(state: AgentState) -> dict:
    """Build the feature matrix requested by the current plan (training data only)."""
    plan = state["plan"]
    train_df = state["train_df"]

    X_train = build_features(train_df, plan["feature_set"])
    y_train = train_df["target"]

    message = (f"Feature Agent: built feature set '{plan['feature_set']}' "
               f"with {X_train.shape[1]} features")

    # Start a fresh record for this experiment
    return {
        "current": {
            "plan": plan,
            "X_train": X_train,
            "y_train": y_train,
            "feature_names": list(X_train.columns),
        },
        "events": [message],
    }


# ---------------------------------------------------------------------
# Trainer Agent: builds and fits the planned model
# ---------------------------------------------------------------------
def trainer_agent(state: AgentState) -> dict:
    """Create the model from the plan and fit it on the training data."""
    current = state["current"]
    plan = current["plan"]

    model = make_model(plan["model"], plan.get("params", {}))
    start = time.perf_counter()
    model.fit(current["X_train"], current["y_train"])
    fit_seconds = time.perf_counter() - start

    # Accuracy on the data the model was trained on (used to spot overfitting)
    train_accuracy = accuracy_score(current["y_train"], model.predict(current["X_train"]))

    message = (f"Trainer Agent: trained {plan['model']} {plan.get('params', {})} "
               f"in {fit_seconds:.2f}s (training accuracy {train_accuracy:.3f})")

    # Keep everything the Feature Agent stored and add the trained model
    return {
        "current": {**current, "model": model, "train_accuracy": train_accuracy},
        "events": [message],
    }


# ---------------------------------------------------------------------
# Evaluator Agent: scores the experiment with cross-validation
# ---------------------------------------------------------------------
def evaluator_agent(state: AgentState) -> dict:
    """Estimate performance with cross-validation on the training data only."""
    current = state["current"]
    plan = current["plan"]

    # Stratified folds keep the 50/50 class balance in every fold
    cv = StratifiedKFold(n_splits=CV_FOLDS)
    scores = cross_validate(
        current["model"], current["X_train"], current["y_train"],
        cv=cv, scoring=["accuracy", "roc_auc", "f1"],
    )

    cv_accuracy = float(scores["test_accuracy"].mean())
    record = {
        "iteration": plan.get("iteration", 1),
        "name": plan["name"],
        "model": plan["model"],
        "params": plan.get("params", {}),
        "feature_set": plan["feature_set"],
        "n_features": len(current["feature_names"]),
        "cv_accuracy": round(cv_accuracy, 4),
        "cv_accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
        "cv_f1": round(float(scores["test_f1"].mean()), 4),
        "train_accuracy": round(float(current["train_accuracy"]), 4),
        # A large gap means the model memorizes training data instead of generalizing
        "overfit_gap": round(float(current["train_accuracy"]) - cv_accuracy, 4),
    }

    message = (f"Evaluator Agent: '{record['name']}' -> CV accuracy "
               f"{record['cv_accuracy']:.3f} ± {record['cv_accuracy_std']:.3f}, "
               f"ROC-AUC {record['cv_roc_auc']:.3f}, overfit gap {record['overfit_gap']:.3f}")

    # "experiments" uses a reducer, so this record is appended to the history
    return {"experiments": [record], "events": [message]}