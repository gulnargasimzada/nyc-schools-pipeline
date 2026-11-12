"""
Live agent graph UI for the agent-based ML workflow (Part 2).

Click "Run agent workflow" to watch every agent light up as it works.
The graph is read from the compiled LangGraph, so it always matches graph.py.
"""
import json
import time
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from config import RECURSION_LIMIT, RESULTS_DIR, TABLE_NAME
from graph import build_graph

# Short pause after each agent so every step is visible (the agents themselves are fast)
STEP_DELAY = 0.4

# Chart colors: one blue for data, grays for structure and text
BLUE, LIGHT_BLUE, INK, MUTED, LINE = "#2a78d6", "#dbe9fb", "#0b0b0b", "#52514e", "#b9b8b2"

# Display names and one-line roles for the agents ("\\n" is a line break in Graphviz)
AGENT_LABELS = {
    "data_agent": "Data Agent\\nload, validate, split",
    "planner_agent": "Planner Agent\\nnext experiment or stop",
    "feature_agent": "Feature Agent\\nbuild feature set",
    "trainer_agent": "Trainer Agent\\nfit the model",
    "evaluator_agent": "Evaluator Agent\\n5-fold cross-validation",
    "reporter_agent": "Reporter Agent\\ntest set + final report",
}

EXPERIMENT_COLUMNS = ["iteration", "model", "feature_set", "params", "n_features",
                      "cv_accuracy", "cv_accuracy_std", "cv_roc_auc", "overfit_gap"]


@st.cache_resource
def get_workflow():
    """Compile the agent graph once and reuse it across page refreshes."""
    return build_graph()


# ---------------------------------------------------------------------
# Agent graph (Graphviz DOT built from the real LangGraph structure)
# ---------------------------------------------------------------------
def graph_to_dot(workflow, active=None, counts=None, traversed=None) -> str:
    """Draw the agent graph and highlight the active agent and the path taken."""
    counts = counts or {}
    traversed = traversed or set()
    structure = workflow.get_graph()

    lines = [
        "digraph agents {",
        'rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;',
        f'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
        f'color="{LINE}", fillcolor="white", fontcolor="{INK}"];',
        f'edge [fontname="Helvetica", fontsize=10, color="{LINE}", fontcolor="{MUTED}"];',
    ]

    # Nodes: active agent = dark blue, agents that already ran = light blue
    for node_id in structure.nodes:
        if node_id in ("__start__", "__end__"):
            label = "start" if node_id == "__start__" else "end"
            lines.append(f'"{node_id}" [label="{label}", shape=circle, width=0.55, '
                         f'fixedsize=true, fontsize=9];')
            continue
        label = AGENT_LABELS.get(node_id, node_id)
        if counts.get(node_id):
            label += f"\\nruns: {counts[node_id]}"
        if node_id == active:
            style = f', fillcolor="{BLUE}", fontcolor="white", color="{BLUE}", penwidth=2'
        elif counts.get(node_id):
            style = f', fillcolor="{LIGHT_BLUE}", color="{BLUE}"'
        else:
            style = ""
        lines.append(f'"{node_id}" [label="{label}"{style}];')

    # Edges: dashed = conditional (Planner's decision), blue = already traversed
    for edge in structure.edges:
        attrs = []
        label = str(edge.data) if edge.data else ""
        if (edge.source, edge.target) == ("evaluator_agent", "planner_agent"):
            label = "loop"
        if label:
            attrs.append(f'label=" {label} "')
        if edge.conditional:
            attrs.append("style=dashed")
        if (edge.source, edge.target) in traversed:
            attrs.append(f'color="{BLUE}", fontcolor="{BLUE}", penwidth=2')
        lines.append(f'"{edge.source}" -> "{edge.target}" [{", ".join(attrs)}];')

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Experiment results
# ---------------------------------------------------------------------
def experiments_frame(experiments: list) -> pd.DataFrame:
    """Experiments as a table, with parameters shown as text."""
    df = pd.DataFrame(experiments)[EXPERIMENT_COLUMNS].copy()
    df["params"] = df["params"].astype(str)
    return df


def experiments_chart(experiments: list) -> alt.Chart:
    """CV accuracy per experiment (± one std) against the baseline."""
    df = pd.DataFrame(experiments)
    df["low"] = df["cv_accuracy"] - df["cv_accuracy_std"]
    df["high"] = df["cv_accuracy"] + df["cv_accuracy_std"]
    df["experiment"] = df["iteration"].astype(str) + ". " + df["model"].str.replace("_", " ")
    df["params"] = df["params"].astype(str)
    baseline = pd.DataFrame({"cv_accuracy": [df["cv_accuracy"].iloc[0]], "label": ["baseline"]})

    x = alt.X("iteration:O", title="Experiment", axis=alt.Axis(labelAngle=0))
    tooltip = ["experiment", "feature_set", "params", "cv_accuracy",
               "cv_accuracy_std", "overfit_gap"]

    error_bars = alt.Chart(df).mark_rule(color=BLUE, strokeWidth=2).encode(
        x=x, y=alt.Y("low:Q", title="CV accuracy", scale=alt.Scale(zero=False)), y2="high:Q"
    )
    points = alt.Chart(df).mark_circle(size=110, color=BLUE, opacity=1).encode(
        x=x, y="cv_accuracy:Q", tooltip=tooltip
    )
    baseline_rule = alt.Chart(baseline).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(
        y="cv_accuracy:Q"
    )
    baseline_text = alt.Chart(baseline).mark_text(
        align="left", dx=4, dy=-6, color=MUTED, fontSize=11
    ).encode(y="cv_accuracy:Q", text="label:N")

    return (error_bars + points + baseline_rule + baseline_text).properties(
        height=280, width="container"
    )


def show_report(report: dict) -> None:
    """Final evaluation of the selected model on the held-out test set."""
    st.subheader("Final result on the held-out test set")
    metrics = report["test_metrics"]
    majority = report["majority_class_test_accuracy"]

    cols = st.columns(4)
    cols[0].metric("Test accuracy", f"{metrics['accuracy']:.3f}",
                   f"{metrics['accuracy'] - majority:+.3f} vs. majority class")
    cols[1].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    cols[2].metric("F1", f"{metrics['f1']:.3f}")
    cols[3].metric("Experiments run", report["n_experiments"])

    selected = report["selected_experiment"]
    st.markdown(
        f"**Selected model:** `{selected['name']}` "
        f"(CV accuracy {selected['cv_accuracy']:.3f}). "
        f"Baseline test accuracy: {report['baseline_test_accuracy']:.3f}. "
        f"Majority-class accuracy: {majority:.3f}."
    )

    left, right = st.columns(2)
    confusion = pd.DataFrame(
        report["confusion_matrix"],
        index=["Actual: below median", "Actual: above median"],
        columns=["Predicted: below", "Predicted: above"],
    )
    left.markdown("**Confusion matrix (216 test schools)**")
    left.table(confusion)
    right.markdown("**Most influential features**")
    right.table(pd.DataFrame(report["top_features"]))


# ---------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------
st.set_page_config(page_title="NYC Schools Agents", layout="wide")
st.title("NYC Schools: Agent-Based ML Workflow")
st.caption(
    f"The agents read the table `{TABLE_NAME}` produced by the Airflow pipeline (Part 1) "
    "and search for the best model to predict whether a school's math proficiency "
    "is above the median."
)

workflow = get_workflow()
run_clicked = st.button("Run agent workflow", type="primary")

graph_col, log_col = st.columns([3, 2])
with graph_col:
    st.markdown("**Agent graph**: dashed arrows are the Planner's decisions")
    graph_slot = st.empty()
with log_col:
    st.markdown("**Agent activity**")
    log_slot = st.empty()

st.markdown("**Experiments**: cross-validation on training data, ± one standard deviation")
chart_slot = st.empty()
table_slot = st.empty()
report_area = st.container()

report_path = RESULTS_DIR / "report.json"

if run_clicked:
    # Live run: stream() returns {agent_name: update} after every agent finishes
    counts, traversed = Counter(), set()
    events, experiments = [], []
    previous, report = "__start__", None

    for chunk in workflow.stream({}, config={"recursion_limit": RECURSION_LIMIT},
                                 stream_mode="updates"):
        for node, update in chunk.items():
            update = update or {}
            counts[node] += 1
            traversed.add((previous, node))
            previous = node

            events.extend(update.get("events", []))
            experiments.extend(update.get("experiments", []))
            report = update.get("report", report)

            graph_slot.graphviz_chart(graph_to_dot(workflow, node, counts, traversed))
            log_slot.code("\n".join(events[-12:]), language=None, wrap_lines=True)
            if experiments:
                chart_slot.altair_chart(experiments_chart(experiments))
                table_slot.dataframe(experiments_frame(experiments), hide_index=True)
            time.sleep(STEP_DELAY)

    # Workflow finished: show the full path, then the final report
    traversed.add((previous, "__end__"))
    graph_slot.graphviz_chart(graph_to_dot(workflow, None, counts, traversed))
    with report_area:
        show_report(report)
        with st.expander("Full agent log"):
                st.code("\n".join(events), language=None, wrap_lines=True)

elif report_path.exists():
    # No run yet in this session: show the graph and the last saved report
    report = json.loads(report_path.read_text())
    graph_slot.graphviz_chart(graph_to_dot(workflow))
    log_slot.info("Click **Run agent workflow** to watch the agents work. "
                  "Below are the results of the last saved run.")
    chart_slot.altair_chart(experiments_chart(report["experiments"]))
    table_slot.dataframe(experiments_frame(report["experiments"]), hide_index=True)
    with report_area:
        show_report(report)

else:
    graph_slot.graphviz_chart(graph_to_dot(workflow))
    log_slot.info("Click **Run agent workflow** to start.")