"""
Agent graph of the ML workflow, built with LangGraph.
"""
from langgraph.graph import END, START, StateGraph

from nodes import data_agent
from state import AgentState


def build_graph():
    """Create and compile the agent graph."""
    builder = StateGraph(AgentState)

    # Agents (nodes)
    builder.add_node("data_agent", data_agent)

    # Connections (edges)
    builder.add_edge(START, "data_agent")
    builder.add_edge("data_agent", END)

    return builder.compile()