"""Shared state of the Part 3 adaptive multi-agent graph."""
import operator
from typing import Annotated, Any, TypedDict


class AdaptiveState(TypedDict, total=False):
    # Input
    scenario: str
    df: Any                    # working copy of the table (repaired in place)
    features: list             # features the model may use

    # Data quality
    data_checked: bool
    issues: list               # open data issues found by the quality agent
    repairs_done: int
    abort_reason: str
    train_df: Any
    test_df: Any

    # Experiment loop
    queue: list                # planned experiments not yet run
    current: dict              # experiment being run right now
    result: dict               # output of the trainer for `current`
    attempts: int              # attempts for `current`
    trainer_calls: int         # global count, used for fault injection
    skipped_models: list

    # Routing
    next: str                  # agent chosen by the supervisor
    done: bool

    # History (reducers append instead of overwrite)
    experiments: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]
    events: Annotated[list, operator.add]

    report: dict