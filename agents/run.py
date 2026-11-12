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

# Start with an empty state: the Planner Agent chooses every experiment
final_state = graph.invoke({}, config={"recursion_limit": RECURSION_LIMIT})

print("\n".join(final_state["events"]))

print("\nExperiments (cross-validation on training data):")
columns = ["iteration", "model", "feature_set", "params", "n_features",
           "cv_accuracy", "cv_accuracy_std", "cv_roc_auc", "overfit_gap"]
pd.set_option("display.width", 200)
print(pd.DataFrame(final_state["experiments"])[columns].to_string(index=False))

report = final_state["report"]
print(f"\nSelected model: {report['selected_experiment']['name']}")
print("Test metrics:", report["test_metrics"])
print(f"Baseline test accuracy: {report['baseline_test_accuracy']:.3f} | "
      f"Majority class: {report['majority_class_test_accuracy']:.3f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:", report["confusion_matrix"])
print("Top features:", report["top_features"])