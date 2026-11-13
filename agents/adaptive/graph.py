"""LangGraph wiring for Part 3: a supervisor routes work to six agents."""
from langgraph.graph import END, START, StateGraph

from config import BASE_FEATURES, TABLE_NAME
from db import load_table
from adaptive import agents
from adaptive.scenarios import SCENARIOS
from adaptive.state import AdaptiveState

WORKERS = ["data_quality", "data_repair", "planner", "trainer", "critic"]
RECURSION_LIMIT = 150


def build_graph():
    g = StateGraph(AdaptiveState)
    g.add_node("supervisor", agents.supervisor)
    for name in WORKERS:
        g.add_node(name, getattr(agents, name))
    g.add_node("reporter", agents.reporter)

    g.add_edge(START, "supervisor")
    # The supervisor picks the next agent at run time
    g.add_conditional_edges("supervisor", agents.route,
                            {name: name for name in WORKERS + ["reporter"]})
    for name in WORKERS:            # every worker reports back to the supervisor
        g.add_edge(name, "supervisor")
    g.add_edge("reporter", END)
    return g.compile()


workflow = build_graph()


def initial_state(scenario_name: str, base_df=None) -> dict:
    scenario = SCENARIOS[scenario_name]
    base = load_table(TABLE_NAME) if base_df is None else base_df
    return {
        "scenario": scenario_name,
        "df": scenario.perturb(base),
        "features": list(BASE_FEATURES) + list(scenario.extra_features),
    }


def summarize(final: dict) -> dict:
    # Common result format shared with the baselines (used by the evaluation)
    report = final.get("report", {})
    events = final.get("events", [])
    return {
        "system": "adaptive_multi_agent",
        "status": report.get("status", "failed"),
        "reason": report.get("reason"),
        "test_accuracy": report.get("test_accuracy"),
        "roc_auc": report.get("roc_auc"),
        "model": report.get("model"),
        "features": report.get("features", final.get("features")),
        "repairs": final.get("repairs_done", 0),
        "retries": sum("retrying" in e for e in events),
        "experiments": len(final.get("experiments", [])),
        "agent_calls": len(final.get("trace", [])),
        "trace": final.get("trace", []),
        "events": events,
    }


def run_adaptive(scenario_name: str, base_df=None) -> dict:
    final = workflow.invoke(initial_state(scenario_name, base_df),
                            config={"recursion_limit": RECURSION_LIMIT})
    return summarize(final)