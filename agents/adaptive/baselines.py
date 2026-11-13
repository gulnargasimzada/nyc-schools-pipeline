"""
The two baselines, re-run on the same scenarios as the adaptive system.

1. fixed_workflow - mirrors the Part 1 Airflow DAG: predefined steps,
   fail-fast quality checks, one model (Logistic Regression), and
   Airflow-style retries that re-run the SAME step unchanged (retries=3).
2. simple_loop - runs the REAL Part 2 LangGraph workflow (agents/graph.py)
   unchanged; only its data source and model factory are patched during the
   run so that the same scenarios and faults can be injected.

Both baselines see exactly the same perturbed data as the adaptive system.
"""
from unittest.mock import patch

from config import BASE_FEATURES
from adaptive.modeling import split_data, test_eval
from adaptive.scenarios import SCENARIOS, SHARE_FEATURES

AIRFLOW_RETRIES = 3


def _result(system, status, trace, reason=None, metrics=None, model=None,
            features=None, retries=0, experiments=0):
    metrics = metrics or {}
    return {
        "system": system, "status": status, "reason": reason,
        "test_accuracy": metrics.get("test_accuracy"), "roc_auc": metrics.get("roc_auc"),
        "model": model, "features": features, "repairs": 0, "retries": retries,
        "experiments": experiments, "agent_calls": len(trace), "trace": trace, "events": [],
    }


def fixed_workflow(scenario_name, base_df):
    s = SCENARIOS[scenario_name]
    df = s.perturb(base_df)
    features = list(BASE_FEATURES) + list(s.extra_features)   # uses whatever is configured
    trace, retries = [], 0

    # transform / merge checks from the DAG (fail fast, no repair)
    trace.append("quality_checks")
    if df["dbn"].duplicated().any():
        return _result("fixed_workflow", "failed", trace, "merge one_to_one check failed: duplicate dbn")
    for col in SHARE_FEATURES:
        if not df[col].between(0, 1).all():   # NaN also fails this check
            return _result("fixed_workflow", "failed", trace, f"range check failed for {col}")

    # train_model task: Airflow retries the same code up to 3 times
    train_df, test_df = split_data(df)
    calls = 0
    for attempt in range(1 + AIRFLOW_RETRIES):
        trace.append("train_model")
        calls += 1
        if calls <= s.trainer_failures:
            retries += 1
            continue
        metrics = test_eval(train_df, test_df, features, "logistic_regression", {})
        return _result("fixed_workflow", "completed", trace, metrics=metrics,
                       model="logistic_regression", features=features,
                       retries=retries, experiments=1)
    return _result("fixed_workflow", "failed", trace, "train_model failed after retries", retries=retries)


def simple_loop(scenario_name, base_df):
    """Run the REAL Part 2 LangGraph workflow (agents/graph.py) on a scenario.

    Part 2 code is not changed. Two functions are replaced only during this run:
    - nodes.load_table returns the scenario's (perturbed) table instead of the database table
    - nodes.make_model raises the scenario's injected errors
    """
    import graph as part2_graph      # Part 2 graph (agents/graph.py)
    import nodes as part2_nodes      # Part 2 agents (agents/nodes.py)
    from config import RECURSION_LIMIT, RESULTS_DIR
    from ml import build_features, make_model as real_make_model

    s = SCENARIOS[scenario_name]
    df = s.perturb(base_df)
    calls = {"n": 0}

    def fake_load_table(table_name):
        return df.copy()

    def faulty_make_model(name, params):
        calls["n"] += 1
        if calls["n"] <= s.trainer_failures:
            raise RuntimeError("injected transient failure")
        if name in s.failing_models:
            raise RuntimeError(f"injected crash in {name}")
        return real_make_model(name, params)

    # Part 2's reporter overwrites results/report.json: keep the real report safe
    report_path = RESULTS_DIR / "report.json"
    saved_report = report_path.read_bytes() if report_path.exists() else None

    trace, state = [], {}
    try:
        with patch.object(part2_nodes, "load_table", fake_load_table), \
                patch.object(part2_nodes, "make_model", faulty_make_model):
            workflow = part2_graph.build_graph()
            for chunk in workflow.stream({}, config={"recursion_limit": RECURSION_LIMIT},
                                         stream_mode="updates"):
                for node, update in chunk.items():
                    trace.append(node)
                    state.update(update or {})
    except Exception as exc:
        # An error inside the Data Agent is the data contract stopping the run on purpose
        status = "aborted" if not trace else "failed"
        return _result("simple_loop", status, trace, reason=str(exc),
                       experiments=trace.count("evaluator_agent"))
    finally:
        if saved_report is not None:
            report_path.write_bytes(saved_report)

    # Final test metrics exactly as computed by the Part 2 Reporter Agent
    best, test_metrics = state["best"], state["report"]["test_metrics"]
    metrics = {"test_accuracy": test_metrics["accuracy"], "roc_auc": test_metrics["roc_auc"]}
    features = list(build_features(state["train_df"].head(), best["feature_set"]).columns)
    return _result("simple_loop", "completed", trace, metrics=metrics, model=best["model"],
                   features=features, experiments=trace.count("evaluator_agent"))