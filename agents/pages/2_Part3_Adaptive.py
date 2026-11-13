"""
Part 3 dashboard: live adaptive multi-agent graph + evaluation suite results.
Streamlit shows this file as a second page next to app.py (Part 2).
"""
import json
import time

import pandas as pd
import streamlit as st

from config import RESULTS_DIR
from adaptive.evaluate import SYSTEMS, TEST_CASES, run_suite, summarize
from adaptive.graph import RECURSION_LIMIT, initial_state, workflow
from adaptive.scenarios import SCENARIOS

st.set_page_config(page_title="Part 3 - Adaptive Multi-Agent", layout="wide")
EVAL_FILE = RESULTS_DIR / "part3_eval.json"
STEP_DELAY = 0.15


def graph_dot(active=None, counts=None):
    counts = counts or {}
    g = workflow.get_graph()
    lines = ["digraph G {", "rankdir=TB; node [shape=box, style=\"rounded,filled\", fontname=Helvetica];"]
    nodes = {e.source for e in g.edges} | {e.target for e in g.edges}
    for n in nodes:
        if n.startswith("__"):
            lines.append(f'"{n}" [label="{n.strip("_")}", shape=oval, fillcolor="#eeeeee"];')
            continue
        color = "#2a78d6" if n == active else ("#cfe2fb" if counts.get(n) else "#ffffff")
        font = "white" if n == active else "black"
        label = f"{n}\\nruns: {counts[n]}" if counts.get(n) else n
        lines.append(f'"{n}" [label="{label}", fillcolor="{color}", fontcolor="{font}"];')
    for e in g.edges:
        style = "dashed" if e.conditional else "solid"
        lines.append(f'"{e.source}" -> "{e.target}" [style={style}];')
    lines.append("}")
    return "\n".join(lines)


st.title("Part 3 - Adaptive Multi-Agent System")
st.caption("A supervisor selects the next agent at run time; data issues are repaired, "
           "failed steps are retried, and overfitting models are re-trained with regularization.")

# ---------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------
st.header("Live run")
name = st.selectbox("Scenario", list(SCENARIOS),
                    format_func=lambda s: f"{s} - {SCENARIOS[s].description}")
left, right = st.columns([1, 1])
graph_slot = left.empty()
log_slot = right.empty()
report_slot = st.empty()
graph_slot.graphviz_chart(graph_dot())

if st.button("Run scenario", type="primary"):
    counts, events, report = {}, [], None
    stream = workflow.stream(initial_state(name), config={"recursion_limit": RECURSION_LIMIT},
                             stream_mode="updates")
    for chunk in stream:
        for node, update in chunk.items():
            counts[node] = counts.get(node, 0) + 1
            update = update or {}
            events += update.get("events", [])
            report = update.get("report", report)
            graph_slot.graphviz_chart(graph_dot(node, counts))
            log_slot.code("\n".join(events[-25:]) or "...", wrap_lines=True)
            time.sleep(STEP_DELAY)
    graph_slot.graphviz_chart(graph_dot(None, counts))
    if report:
        if report["status"] == "completed":
            c1, c2, c3 = report_slot.columns(3)
            c1.metric("Chosen model", report["model"])
            c2.metric("Test accuracy", f"{report['test_accuracy']:.3f}")
            c3.metric("ROC-AUC", f"{report['roc_auc']:.3f}")
        else:
            report_slot.warning(f"Run stopped safely: {report['reason']}")

# ---------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------
st.header("Evaluation suite: adaptive system vs. baselines")
st.caption("fixed_workflow = Part 1 Airflow logic, simple_loop = Part 2 agent loop, "
           "adaptive_multi_agent = Part 3")

if st.button("Re-run evaluation suite (~1 minute)"):
    with st.spinner("Running 8 test cases on 3 systems..."):
        rows = run_suite()
        RESULTS_DIR.mkdir(exist_ok=True)
        EVAL_FILE.write_text(json.dumps({"test_cases": rows, "summary": summarize(rows)},
                                        indent=2, default=str))

if EVAL_FILE.exists():
    data = json.loads(EVAL_FILE.read_text())
    rows, summary = data["test_cases"], data["summary"]

    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "system"
    st.subheader("Summary")
    st.dataframe(summary_df)
    st.bar_chart(summary_df["tests_passed"].astype(int), horizontal=True)

    st.subheader("Pass / fail by test case")
    matrix = pd.DataFrame(rows).assign(
        result=lambda d: d.apply(lambda r: f"{'PASS' if r['passed'] else 'FAIL'} ({r['score']:.2f})", axis=1)
    ).pivot(index=["test", "scenario"], columns="system", values="result")[list(SYSTEMS)]
    st.dataframe(matrix)

    st.subheader("Test case details")
    for case in TEST_CASES:
        with st.expander(f"{case['id']} - {case['scenario']}"):
            st.markdown(f"**Input:** {SCENARIOS[case['scenario']].description}  \n"
                        f"**Expected behavior:** {case['expected']}  \n"
                        f"**Scoring criteria:** {', '.join(case['checks'])}  \n"
                        f"**Pass threshold:** score >= {case['threshold']}")
            detail = pd.DataFrame([r for r in rows if r["test"] == case["id"]])[
                ["system", "status", "reason", "test_accuracy", "repairs", "retries",
                 "agent_calls", "score", "passed"]]
            st.dataframe(detail, hide_index=True)
else:
    st.info("No evaluation results yet. Click the button above or run "
            "`python -m adaptive.evaluate`.")