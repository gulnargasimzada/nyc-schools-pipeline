"""
Reporter Agent: runs once after the loop.

It retrains the model chosen by the Planner on all training data, evaluates it
ONCE on the held-out test set and writes the final report. Test results are only
reported; they are never used to choose or change the model.
"""
import json
from datetime import datetime, timezone

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from config import BASELINE_PLAN, RESULTS_DIR
from ml import build_features, make_model
from state import AgentState


def _fit_on_train(spec: dict, train_df, test_df) -> tuple:
    """Train a model described by a plan/experiment on all training data.

    Returns the fitted model and the matching test feature matrix.
    """
    X_train = build_features(train_df, spec["feature_set"])
    X_test = build_features(test_df, spec["feature_set"])
    model = make_model(spec["model"], spec.get("params", {}))
    model.fit(X_train, train_df["target"])
    return model, X_train, X_test


def _top_features(model, feature_names: list, n: int = 5) -> list:
    """Most influential features: coefficients for linear models, importances for trees."""
    estimator = model[-1] if hasattr(model, "steps") else model   # unwrap the pipeline
    if hasattr(estimator, "coef_"):
        weights = estimator.coef_[0]
    else:
        weights = estimator.feature_importances_
    ranked = sorted(zip(feature_names, weights), key=lambda pair: abs(pair[1]), reverse=True)
    return [{"feature": name, "weight": round(float(w), 3)} for name, w in ranked[:n]]


def reporter_agent(state: AgentState) -> dict:
    """Evaluate the selected model on the test set and write the final report."""
    best = state["best"]
    train_df, test_df = state["train_df"], state["test_df"]
    y_train, y_test = train_df["target"], test_df["target"]

    # 1. Final model: the Planner's choice, retrained on all 862 training schools
    model, X_train, X_test = _fit_on_train(best, train_df, test_df)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }

    # 2. Reference points (reported only, never used to pick the model)
    baseline_model, _, X_test_baseline = _fit_on_train(BASELINE_PLAN, train_df, test_df)
    baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test_baseline))
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    majority_accuracy = majority.score(X_test, y_test)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": state["data_summary"],
        "n_experiments": len(state["experiments"]),
        "selected_experiment": best,
        "test_metrics": test_metrics,
        "baseline_test_accuracy": round(float(baseline_accuracy), 4),
        "majority_class_test_accuracy": round(float(majority_accuracy), 4),
        # Rows = actual class (0, 1); columns = predicted class (0, 1)
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "top_features": _top_features(model, list(X_train.columns)),
        "experiments": state["experiments"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    message = (f"Reporter Agent: '{best['name']}' selected after {report['n_experiments']} "
               f"experiments. Test accuracy {test_metrics['accuracy']:.3f} "
               f"(ROC-AUC {test_metrics['roc_auc']:.3f}) vs. majority class "
               f"{majority_accuracy:.3f}. Report saved to {report_path.name}")

    return {"report": report, "events": [message]}