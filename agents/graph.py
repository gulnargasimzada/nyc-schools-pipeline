"""
Agent graph of the ML workflow, built with LangGraph.
"""
from langgraph.graph import END, START, StateGraph

from nodes import data_agent, evaluator_agent, feature_agent, trainer_agent
from state import AgentState


def build_graph():
    """Create and compile the agent graph."""
    builder = StateGraph(AgentState)

    # Agents (nodes)
    builder.add_node("data_agent", data_agent)
    builder.add_node("feature_agent", feature_agent)
    builder.add_node("trainer_agent", trainer_agent)
    builder.add_node("evaluator_agent", evaluator_agent)

    # Connections (edges): one experiment from start to finish
    builder.add_edge(START, "data_agent")
    builder.add_edge("data_agent", "feature_agent")
    builder.add_edge("feature_agent", "trainer_agent")
    builder.add_edge("trainer_agent", "evaluator_agent")
    builder.add_edge("evaluator_agent", END)

    return builder.compile()