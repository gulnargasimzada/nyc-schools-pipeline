"""
Run the agent workflow from the command line (without the UI).

Usage (inside the agents container):  python run.py
"""
import pandas as pd

from config import RECURSION_LIMIT
from graph import build_graph

graph = build_graph()

# Print the graph structure as a Mermaid diagram
print(graph.get_graph().draw_mermaid())

# Start with an empty state: the Planner Agent now chooses every experiment
final_state = graph.invoke({}, config={"recursion_limit": RECURSION_LIMIT})

print("\n".join(final_state["events"]))

print("\nExperiments:")
columns = ["iteration", "model", "feature_set", "params", "n_features",
           "cv_accuracy", "cv_accuracy_std", "cv_roc_auc", "overfit_gap"]
pd.set_option("display.width", 200)
print(pd.DataFrame(final_state["experiments"])[columns].to_string(index=False))

best = final_state["best"]
print(f"\nBest experiment: {best['name']} (CV accuracy {best['cv_accuracy']:.3f})")