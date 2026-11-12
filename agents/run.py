"""
Run the agent workflow from the command line (without the UI).

Usage (inside the agents container):  python run.py
"""
import pandas as pd

from config import BASELINE_PLAN
from graph import build_graph

graph = build_graph()

# Print the graph structure as a Mermaid diagram
print(graph.get_graph().draw_mermaid())

# Until the Planner Agent exists, the first plan is provided here
final_state = graph.invoke({"plan": BASELINE_PLAN})

print("\n".join(final_state["events"]))

print("\nExperiments:")
columns = ["iteration", "name", "model", "n_features", "cv_accuracy",
           "cv_accuracy_std", "cv_roc_auc", "train_accuracy", "overfit_gap"]
print(pd.DataFrame(final_state["experiments"])[columns].to_string(index=False))