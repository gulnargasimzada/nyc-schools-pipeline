"""
Agents of the Part 3 adaptive multi-agent system (rule-based, no LLM).

Every worker agent reports back to the SUPERVISOR, which reads the shared
state and decides which agent runs next. This is what makes the workflow
adaptive: the path through the graph depends on intermediate results.
"""
from config import BASE_FEATURES, MIN_ROWS, TARGET_SOURCE
from adaptive.modeling import (
    DEFAULT_PLANS, REGULARIZED_PARAMS, cross_val, pick_best, split_data, test_eval,
)
from adaptive.scenarios import SCENARIOS, SHARE_FEATURES

MAX_REPAIR_ROUNDS = 2       # data repair -> re-check cycles
MAX_ATTEMPTS = 2            # attempts per experiment before the model is skipped
MAX_MISSING_SHARE = 0.10    # more missing values than this cannot be repaired
LEAKAGE_CORR = 0.90         # |corr(feature, target)| above this = leakage
OVERFIT_LIMIT = 0.05        # train accuracy - CV accuracy
SUSPICIOUS_ACCURACY = 0.95  # CV accuracy above this is "too good to be true"


# ---------------------------------------------------------------------
# Supervisor: selects the next agent based on the current state
# ---------------------------------------------------------------------
def supervisor(state):
    update = {"trace": ["supervisor"]}

    if state.get("abort_reason") or state.get("done"):
        nxt = "reporter"
    elif not state.get("data_checked"):
        nxt = "data_quality"
    elif state.get("issues"):
        issues = state["issues"]
        if all(i["repairable"] for i in issues) and state.get("repairs_done", 0) < MAX_REPAIR_ROUNDS:
            nxt = "data_repair"
        else:
            blocking = [i["detail"] for i in issues if not i["repairable"]] or ["repair limit reached"]
            update["abort_reason"] = "; ".join(blocking)
            nxt = "reporter"
    elif not state.get("current"):
        nxt = "planner"
    elif state.get("result") is None:
        nxt = "trainer"
    elif state["result"]["status"] == "error":
        if state.get("attempts", 0) < MAX_ATTEMPTS:
            nxt = "trainer"     # retry the failed step
            update["events"] = [f"Supervisor: retrying {state['current']['model']} "
                                f"(attempt {state['attempts'] + 1}/{MAX_ATTEMPTS})"]
        else:
            model = state["current"]["model"]   # give up on this model family
            skipped = state.get("skipped_models", []) + [model]
            update.update({
                "skipped_models": skipped,
                "queue": [p for p in state.get("queue", []) if p["model"] != model],
                "current": None, "result": None,
                "events": [f"Supervisor: {model} failed {MAX_ATTEMPTS} times -> skipped, routing to planner"],
            })
            nxt = "planner"
    else:
        nxt = "critic"

    update["next"] = nxt
    return update


def route(state):
    return state["next"]


# ---------------------------------------------------------------------
# Data Quality: checks the table and the requested features
# ---------------------------------------------------------------------
def data_quality(state):
    df, features = state["df"], state["features"]
    issues = []

    if len(df) < MIN_ROWS:
        issues.append({"type": "too_few_rows", "repairable": False,
                       "detail": f"only {len(df)} rows (minimum {MIN_ROWS})"})

    dup = int(df["dbn"].duplicated().sum())
    if dup:
        issues.append({"type": "duplicates", "repairable": True, "detail": f"{dup} duplicate schools"})

    for col in features:
        if col == TARGET_SOURCE or col not in df:
            continue
        corr = df[col].corr(df[TARGET_SOURCE])
        if abs(corr) > LEAKAGE_CORR:
            issues.append({"type": "leakage", "column": col, "repairable": True,
                           "detail": f"{col} has |corr| {abs(corr):.2f} with the target"})

    for col in [c for c in features if c in df]:
        share = df[col].isna().mean()
        if share > 0:
            issues.append({"type": "missing", "column": col,
                           "repairable": share <= MAX_MISSING_SHARE,
                           "detail": f"{col}: {share:.1%} missing"})

    for col in [c for c in SHARE_FEATURES if c in features]:
        bad = int((df[col] > 1).sum())
        if bad:
            issues.append({"type": "out_of_range", "column": col, "repairable": True,
                           "detail": f"{col}: {bad} values above 1"})

    update = {"data_checked": True, "issues": issues, "trace": ["data_quality"]}
    if issues:
        update["events"] = ["Data Quality: " + "; ".join(i["detail"] for i in issues)]
    else:
        train_df, test_df = split_data(df)
        update.update({"train_df": train_df, "test_df": test_df,
                       "events": [f"Data Quality: passed ({len(df)} rows, "
                                  f"{len(features)} features) -> train {len(train_df)} / test {len(test_df)}"]})
    return update


# ---------------------------------------------------------------------
# Data Repair: fixes repairable issues, then the data is checked again
# ---------------------------------------------------------------------
def data_repair(state):
    df, features = state["df"].copy(), list(state["features"])
    actions = []
    for issue in state["issues"]:
        col = issue.get("column")
        if issue["type"] == "duplicates":
            df = df.drop_duplicates(subset="dbn").reset_index(drop=True)
            actions.append("dropped duplicate schools")
        elif issue["type"] == "leakage":
            features.remove(col)
            actions.append(f"removed leaky feature {col}")
        elif issue["type"] == "missing":
            df[col] = df[col].fillna(df[col].median())
            actions.append(f"imputed {col} with the median")
        elif issue["type"] == "out_of_range":
            mask = df[col] > 1
            df.loc[mask, col] = df.loc[mask, col] / 100   # 0-100 scale -> 0-1
            actions.append(f"rescaled {int(mask.sum())} {col} values from 0-100 to 0-1")
    return {
        "df": df, "features": features, "issues": [], "data_checked": False,
        "repairs_done": state.get("repairs_done", 0) + 1,
        "trace": ["data_repair"], "events": ["Data Repair: " + "; ".join(actions)],
    }


# ---------------------------------------------------------------------
# Planner: picks the next experiment from the queue
# ---------------------------------------------------------------------
def planner(state):
    queue = state.get("queue")
    if queue is None:   # first call: build the plan
        skipped = state.get("skipped_models", [])
        queue = [dict(p) for p in DEFAULT_PLANS if p["model"] not in skipped]
    if not queue:
        return {"queue": [], "done": True, "trace": ["planner"],
                "events": ["Planner: no experiments left -> stop"]}
    current, rest = queue[0], queue[1:]
    return {"queue": rest, "current": current, "result": None, "attempts": 0,
            "trace": ["planner"],
            "events": [f"Planner: next experiment = {current['model']} {current['params']}"]}


# ---------------------------------------------------------------------
# Trainer: trains + cross-validates; failures are caught, not raised
# ---------------------------------------------------------------------
def trainer(state):
    scenario = SCENARIOS[state["scenario"]]
    plan = state["current"]
    calls = state.get("trainer_calls", 0) + 1
    attempts = state.get("attempts", 0) + 1
    try:
        # Fault injection for the evaluation scenarios
        if calls <= scenario.trainer_failures:
            raise RuntimeError("injected transient failure")
        if plan["model"] in scenario.failing_models:
            raise RuntimeError(f"injected crash in {plan['model']}")
        metrics = cross_val(state["train_df"], state["features"], plan["model"], plan["params"])
        result = {"status": "ok", **metrics}
        msg = (f"Trainer: {plan['model']} CV {metrics['cv_accuracy']:.3f} "
               f"(gap {metrics['overfit_gap']:.3f})")
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
        msg = f"Trainer: {plan['model']} FAILED ({exc})"
    return {"result": result, "trainer_calls": calls, "attempts": attempts,
            "trace": ["trainer"], "events": [msg]}


# ---------------------------------------------------------------------
# Critic: reviews the intermediate result and may request a new attempt
# ---------------------------------------------------------------------
def critic(state):
    plan, result = state["current"], state["result"]
    record = {**plan, **{k: v for k, v in result.items() if k != "status"}, "accepted": True}
    update = {"current": None, "result": None, "trace": ["critic"]}

    if result["cv_accuracy"] > SUSPICIOUS_ACCURACY:
        record["accepted"] = False
        msg = f"Critic: CV {result['cv_accuracy']:.3f} is suspiciously high -> rejected (possible leakage)"
    elif (result["overfit_gap"] > OVERFIT_LIMIT and plan["model"] in REGULARIZED_PARAMS
          and not plan.get("regularized")):
        record["accepted"] = False
        retry = {"model": plan["model"], "params": REGULARIZED_PARAMS[plan["model"]], "regularized": True}
        update["queue"] = [retry] + state.get("queue", [])
        msg = (f"Critic: {plan['model']} overfits (gap {result['overfit_gap']:.3f}) "
               f"-> retry with regularization")
    else:
        msg = f"Critic: {plan['model']} accepted"

    update["experiments"] = [record]
    update["events"] = [msg]
    return update


# ---------------------------------------------------------------------
# Reporter: evaluates the chosen model once on the test set
# ---------------------------------------------------------------------
def reporter(state):
    accepted = [e for e in state.get("experiments", []) if e["accepted"]]
    if state.get("abort_reason") or not accepted:
        reason = state.get("abort_reason") or "no experiment was accepted"
        report = {"status": "aborted", "reason": reason}
        msg = f"Reporter: aborted - {reason}"
    else:
        best = pick_best(accepted)
        metrics = test_eval(state["train_df"], state["test_df"], state["features"],
                            best["model"], best["params"])
        report = {"status": "completed", "model": best["model"], "params": best["params"],
                  "features": state["features"], "cv_accuracy": best["cv_accuracy"], **metrics}
        msg = (f"Reporter: chose {best['model']} -> test accuracy {metrics['test_accuracy']:.3f}, "
               f"ROC-AUC {metrics['roc_auc']:.3f}")
    return {"report": report, "trace": ["reporter"], "events": [msg]}