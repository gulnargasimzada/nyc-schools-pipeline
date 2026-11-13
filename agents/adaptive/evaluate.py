"""
Evaluation suite: python -m adaptive.evaluate

Every test case defines its input (scenario), expected behavior, scoring
criteria (named checks) and a pass/fail threshold. Each test runs on all
three systems, and the results are saved to results/part3_eval.json.
"""
import json
import time

from config import RESULTS_DIR, TABLE_NAME
from db import load_table
from adaptive.baselines import fixed_workflow, simple_loop
from adaptive.graph import run_adaptive
from adaptive.scenarios import SCENARIOS

SYSTEMS = {
    "fixed_workflow": fixed_workflow,
    "simple_loop": simple_loop,
    "adaptive_multi_agent": run_adaptive,
}


# Reusable checks: each returns True/False for one system result
def completed(r):            return r["status"] == "completed"
def accuracy_at_least(x):    return lambda r: (r["test_accuracy"] or 0) >= x
def accuracy_below(x):       return lambda r: r["test_accuracy"] is not None and r["test_accuracy"] < x
def repaired(r):             return r["repairs"] >= 1
def no_repairs(r):           return r["repairs"] == 0
def retried(r):              return r["retries"] >= 1
def stopped_safely(r):       return r["status"] == "aborted" and r["test_accuracy"] is None
def reason_given(r):         return bool(r["reason"])
def feature_excluded(col):   return lambda r: col not in (r["features"] or [])


TEST_CASES = [
    {"id": "T1", "scenario": "clean",
     "expected": "Completes without repairs; final model is accurate",
     "checks": {"completed": completed, "accuracy >= 0.78": accuracy_at_least(0.78),
                "no unnecessary repair": no_repairs},
     "threshold": 1.0},
    {"id": "T2", "scenario": "missing_values",
     "expected": "Detects missing values, imputes them, completes",
     "checks": {"completed": completed, "repair performed": repaired,
                "accuracy >= 0.75": accuracy_at_least(0.75)},
     "threshold": 1.0},
    {"id": "T3", "scenario": "unit_error",
     "expected": "Detects 0-100 values, rescales them, completes",
     "checks": {"completed": completed, "repair performed": repaired,
                "accuracy >= 0.75": accuracy_at_least(0.75)},
     "threshold": 1.0},
    {"id": "T4", "scenario": "duplicates",
     "expected": "Removes duplicate schools, completes",
     "checks": {"completed": completed, "repair performed": repaired,
                "accuracy >= 0.75": accuracy_at_least(0.75)},
     "threshold": 1.0},
    {"id": "T5", "scenario": "leaky_feature",
     "expected": "Detects the leaky feature and trains without it",
     "checks": {"completed": completed, "leaky feature excluded": feature_excluded("mean_scale_score"),
                "accuracy is realistic (< 0.95)": accuracy_below(0.95)},
     "threshold": 1.0},
    {"id": "T6", "scenario": "transient_failure",
     "expected": "Retries the failed training step and completes",
     "checks": {"completed": completed, "retry performed": retried,
                "accuracy >= 0.78": accuracy_at_least(0.78)},
     "threshold": 1.0},
    {"id": "T7", "scenario": "model_crash",
     "expected": "Skips the crashing model family and completes with another model",
     "checks": {"completed": completed, "accuracy >= 0.78": accuracy_at_least(0.78)},
     "threshold": 1.0},
    {"id": "T8", "scenario": "too_few_rows",
     "expected": "Stops safely with a clear reason instead of training on too little data",
     "checks": {"stopped safely": stopped_safely, "reason recorded": reason_given},
     "threshold": 1.0},
]


def run_suite():
    base = load_table(TABLE_NAME)
    rows = []
    for case in TEST_CASES:
        for system, fn in SYSTEMS.items():
            start = time.perf_counter()
            try:
                result = fn(case["scenario"], base)
            except Exception as exc:   # a crash counts as a failed run
                result = {"system": system, "status": "crashed", "reason": str(exc),
                          "test_accuracy": None, "roc_auc": None, "model": None, "features": None,
                          "repairs": 0, "retries": 0, "experiments": 0, "agent_calls": 0}
            seconds = round(time.perf_counter() - start, 2)
            checks = {name: bool(check(result)) for name, check in case["checks"].items()}
            score = round(sum(checks.values()) / len(checks), 2)
            rows.append({
                "test": case["id"], "scenario": case["scenario"],
                "input": SCENARIOS[case["scenario"]].description,
                "expected": case["expected"], "threshold": case["threshold"],
                "system": system, "status": result["status"], "reason": result.get("reason"),
                "test_accuracy": result["test_accuracy"], "model": result["model"],
                "repairs": result["repairs"], "retries": result["retries"],
                "experiments": result["experiments"], "agent_calls": result["agent_calls"],
                "seconds": seconds, "checks": checks, "score": score,
                "passed": score >= case["threshold"],
            })
    return rows


def summarize(rows):
    summary = {}
    for system in SYSTEMS:
        mine = [r for r in rows if r["system"] == system]
        faulty = [r for r in mine if r["scenario"] not in ("clean", "too_few_rows")]
        summary[system] = {
            "tests_passed": sum(r["passed"] for r in mine),
            "tests_total": len(mine),
            "mean_score": round(sum(r["score"] for r in mine) / len(mine), 2),
            "recovery_rate": round(sum(r["status"] == "completed" for r in faulty) / len(faulty), 2),
            "avg_agent_calls": round(sum(r["agent_calls"] for r in mine) / len(mine), 1),
            "total_seconds": round(sum(r["seconds"] for r in mine), 1),
        }
    return summary


if __name__ == "__main__":
    rows = run_suite()
    summary = summarize(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "part3_eval.json"
    out.write_text(json.dumps({"test_cases": rows, "summary": summary}, indent=2, default=str))

    print(f"{'test':<5}{'scenario':<19}{'fixed_workflow':<17}{'simple_loop':<17}{'adaptive':<17}")
    for case in TEST_CASES:
        cells = []
        for system in SYSTEMS:
            r = next(x for x in rows if x["test"] == case["id"] and x["system"] == system)
            cells.append(f"{'PASS' if r['passed'] else 'FAIL'} ({r['score']:.2f})")
        print(f"{case['id']:<5}{case['scenario']:<19}" + "".join(f"{c:<17}" for c in cells))
    print("\nSummary:")
    for system, s in summary.items():
        print(f"  {system:<22} passed {s['tests_passed']}/{s['tests_total']}  "
              f"mean score {s['mean_score']:.2f}  recovery {s['recovery_rate']:.0%}  "
              f"avg calls {s['avg_agent_calls']}  time {s['total_seconds']}s")
    print(f"\nSaved to {out}")