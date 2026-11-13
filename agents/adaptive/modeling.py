"""
Shared ML helpers for Part 3: the adaptive agents AND both baselines use
exactly the same split, models and metrics, so the comparison is fair.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CV_FOLDS, RANDOM_STATE, TARGET_SOURCE, TEST_SIZE

# Experiments tried by the planner (Part 3) and the simple loop (Part 2 baseline)
DEFAULT_PLANS = [
    {"model": "logistic_regression", "params": {}},
    {"model": "random_forest", "params": {"n_estimators": 300}},
    {"model": "gradient_boosting", "params": {}},
]

# Stronger regularization used by the critic when a model overfits
REGULARIZED_PARAMS = {
    "random_forest": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5},
    "gradient_boosting": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
}


def make_model(name: str, params: dict):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **params))
    if name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unknown model: {name}")


def split_data(df: pd.DataFrame):
    # Binary target: is the school above the median share of proficient students?
    data = df.copy()
    data["target"] = (data[TARGET_SOURCE] > data[TARGET_SOURCE].median()).astype(int)
    return train_test_split(
        data, test_size=TEST_SIZE, stratify=data["target"], random_state=RANDOM_STATE
    )


def cross_val(train_df: pd.DataFrame, features: list, model_name: str, params: dict) -> dict:
    X, y = train_df[features], train_df["target"]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(make_model(model_name, params), X, y, cv=cv, scoring="accuracy")
    train_acc = make_model(model_name, params).fit(X, y).score(X, y)
    cv_acc = float(scores["test_score"].mean())
    return {
        "cv_accuracy": round(cv_acc, 4),
        "cv_std": round(float(scores["test_score"].std()), 4),
        "train_accuracy": round(float(train_acc), 4),
        "overfit_gap": round(float(train_acc) - cv_acc, 4),
    }


def test_eval(train_df, test_df, features, model_name, params) -> dict:
    # The test set is used only once, for the final chosen model
    model = make_model(model_name, params).fit(train_df[features], train_df["target"])
    proba = model.predict_proba(test_df[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": round(float(accuracy_score(test_df["target"], pred)), 4),
        "roc_auc": round(float(roc_auc_score(test_df["target"], proba)), 4),
    }


def pick_best(experiments: list, min_improvement: float = 0.005) -> dict:
    # Prefer the earliest (simplest) model unless a later one is clearly better
    best = experiments[0]
    for exp in experiments[1:]:
        if exp["cv_accuracy"] >= best["cv_accuracy"] + min_improvement:
            best = exp
    return best