"""
Agents (graph nodes) of the ML workflow.

Each agent is a function: it reads the shared state and returns
only the keys it wants to update.
"""
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from config import (BASE_FEATURES, BASELINE_PLAN, CV_FOLDS, FEATURE_SETS, MAX_ITERATIONS,
                    MIN_IMPROVEMENT, MIN_ROWS, MODEL_FAMILIES, OVERFIT_LIMIT, PATIENCE,
                    RANDOM_STATE, TABLE_NAME, TARGET_SOURCE, TEST_SIZE, TUNING_GRID)
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
# Planner Agent: decides the next experiment or stops the loop
# ---------------------------------------------------------------------
def _experiment_key(model: str, feature_set: str, params: dict) -> tuple:
    """Identify an experiment so the Planner never repeats one."""
    return (model, feature_set, tuple(sorted(params.items())))


def _best_and_stale(history: list) -> tuple:
    """Return the best experiment so far and how many experiments ran since it was found.

    A new result only counts as better if it beats the current best by MIN_IMPROVEMENT;
    smaller gains are within cross-validation noise, so the earlier (simpler) model is kept.
    """
    best, stale = None, 0
    for exp in history:
        if best is None or exp["cv_accuracy"] > best["cv_accuracy"] + MIN_IMPROVEMENT:
            best, stale = exp, 0
        else:
            stale += 1
    return best, stale


def _family_best(history: list) -> dict:
    """Return the best experiment of every model family tried so far."""
    best_by_model = {}
    for exp in history:
        current = best_by_model.get(exp["model"])
        if current is None or exp["cv_accuracy"] > current["cv_accuracy"]:
            best_by_model[exp["model"]] = exp
    return best_by_model


def _next_plan(history: list, best: dict | None) -> tuple:
    """Pick the first untried experiment, stage by stage, built around the results so far."""
    if not history:
        return dict(BASELINE_PLAN), "Baseline: reproduce the Part 1 model"

    tried = {_experiment_key(e["model"], e["feature_set"], e["params"]) for e in history}

    candidates = []
    # Stage 2: does a different feature set help the best model?
    for feature_set in FEATURE_SETS:
        candidates.append((best["model"], feature_set, best["params"],
                           f"Feature exploration: try '{feature_set}' with {best['model']}"))

    # Stage 3: does another model family do better on the best feature set?
    for model in MODEL_FAMILIES:
        candidates.append((model, best["feature_set"], {},
                           f"Model exploration: try {model} on '{best['feature_set']}'"))

    # Stage 4: tune every promising model family.
    # "Promising" = its best CV accuracy is within one standard deviation of the
    # overall best, i.e. we cannot tell it apart from the best model yet.
    # Families that overfit most are tuned first, because the tuning variants
    # regularize the model and help an overfitting model the most.
    margin = best["cv_accuracy_std"]
    promising = [
        exp for exp in _family_best(history).values()
        if best["cv_accuracy"] - exp["cv_accuracy"] <= margin
    ]
    promising.sort(key=lambda exp: exp["overfit_gap"], reverse=True)

    for family in promising:
        reason = (f"Tuning: regularize {family['model']} "
                  f"(CV {family['cv_accuracy']:.3f}, within {margin:.3f} of the best")
        if family["overfit_gap"] > OVERFIT_LIMIT:
            reason += f"; overfits by {family['overfit_gap']:.3f}"
        reason += ")"
        for params in TUNING_GRID[family["model"]]:
            candidates.append((family["model"], family["feature_set"], params, reason))

    for model, feature_set, params, reason in candidates:
        if _experiment_key(model, feature_set, params) not in tried:
            name = f"{model} | {feature_set}" + (f" | {params}" if params else "")
            plan = {"name": name, "model": model, "feature_set": feature_set, "params": params}
            return plan, reason

    return None, "all planned experiments have been tried"


def planner_agent(state: AgentState) -> dict:
    """Look at the results so far and either plan the next experiment or stop."""
    history = state.get("experiments", [])
    iteration = state.get("iteration", 0)
    best, stale = _best_and_stale(history)

    # 1. Stopping rules
    stop_reason = None
    if iteration >= MAX_ITERATIONS:
        stop_reason = f"reached the limit of {MAX_ITERATIONS} experiments"
    elif stale >= PATIENCE:
        stop_reason = (f"no improvement of at least {MIN_IMPROVEMENT} "
                       f"in the last {PATIENCE} experiments")
    else:
        plan, reason = _next_plan(history, best)
        if plan is None:
            stop_reason = reason

    if stop_reason:
        message = (f"Planner Agent: STOP, {stop_reason}. Best so far: '{best['name']}' "
                   f"(CV accuracy {best['cv_accuracy']:.3f})")
        return {"done": True, "best": best, "events": [message]}

    # 2. Continue: hand the next plan to the Feature Agent
    plan = {**plan, "iteration": iteration + 1}
    best_text = f"current best CV {best['cv_accuracy']:.3f}" if best else "no results yet"
    message = f"Planner Agent: experiment {iteration + 1}, {reason} ({best_text})"
    return {"plan": plan, "iteration": iteration + 1, "done": False, "events": [message]}


def route_after_planner(state: AgentState) -> str:
    """Conditional edge: continue the loop or leave it."""
    return "stop" if state.get("done") else "continue"


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