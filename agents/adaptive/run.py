"""Run the adaptive system on one scenario: python -m adaptive.run <scenario>"""
import sys

from adaptive.graph import run_adaptive
from adaptive.scenarios import SCENARIOS

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "clean"
    if name not in SCENARIOS:
        sys.exit(f"Unknown scenario. Choose one of: {', '.join(SCENARIOS)}")
    result = run_adaptive(name)
    print(f"=== Scenario: {name} - {SCENARIOS[name].description} ===\n")
    for event in result["events"]:
        print(" ", event)
    print(f"\nStatus: {result['status']} | test accuracy: {result['test_accuracy']} | "
          f"repairs: {result['repairs']} | retries: {result['retries']} | "
          f"agent calls: {result['agent_calls']}")