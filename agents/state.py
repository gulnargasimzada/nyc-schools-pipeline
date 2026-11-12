"""
Shared state of the agent workflow.

Every agent receives the current state and returns only the keys it changes.
LangGraph merges those updates into the state before calling the next agent.
"""
import operator
from typing import Annotated, Any, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    # Data Agent: prepared data
    train_df: pd.DataFrame          # 80% of schools, used inside the loop
    test_df: pd.DataFrame           # 20% held out, used only once by the Reporter
    data_summary: dict[str, Any]

    # Planner Agent: loop control
    iteration: int                  # Number of experiments run so far
    plan: dict[str, Any]            # The next experiment to run
    done: bool                      # True when the Planner decides to stop

    # Feature, Trainer and Evaluator Agents: results of the current experiment
    current: dict[str, Any]

    # History: lists marked with operator.add are appended to, not overwritten
    experiments: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[str], operator.add]

    # Reporter Agent: final result
    best: dict[str, Any]
    report: dict[str, Any]