"""
Agent graph of the ML workflow, built with LangGraph.
"""
from langgraph.graph import END, START, StateGraph

from nodes import (data_agent, evaluator_agent, feature_agent, planner_agent,
                   route_after_planner, trainer_agent)
from state import AgentState


def build_graph():
    """Create and compile the agent graph."""
    builder = StateGraph(AgentState)

    # Agents (nodes)
    builder.add_node("data_agent", data_agent)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("feature_agent", feature_agent)
    builder.add_node("trainer_agent", trainer_agent)
    builder.add_node("evaluator_agent", evaluator_agent)

    # Data preparation runs once, then control passes to the Planner
    builder.add_edge(START, "data_agent")
    builder.add_edge("data_agent", "planner_agent")

    # Conditional edge: the Planner's decision chooses the next agent
    builder.add_conditional_edges(
        "planner_agent",
        route_after_planner,
        {"continue": "feature_agent", "stop": END},
    )

    # One experiment: features -> training -> evaluation
    builder.add_edge("feature_agent", "trainer_agent")
    builder.add_edge("trainer_agent", "evaluator_agent")

    # The loop: after every evaluation the Planner decides again
    builder.add_edge("evaluator_agent", "planner_agent")

    return builder.compile()