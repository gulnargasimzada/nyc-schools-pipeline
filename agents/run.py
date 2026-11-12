"""
Run the agent workflow from the command line (without the UI).

Usage (inside the agents container):  python run.py
"""
from graph import build_graph

graph = build_graph()

# Print the graph structure as a Mermaid diagram
print(graph.get_graph().draw_mermaid())

# Start with an empty state; the agents fill it in as they run
final_state = graph.invoke({})

print("\n".join(final_state["events"]))
print("Data summary:", final_state["data_summary"])