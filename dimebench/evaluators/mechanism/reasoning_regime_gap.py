"""Track 4 chain/planning macro averages and regime gap."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import fmean

from pydantic import Field

from dimebench.evaluators.base import EvaluationError
from dimebench.schemas.base import StrictSchema

CHAIN_TASKS = ("gsm8k_chain", "math500_chain")
PLANNING_TASKS = ("winogrande_planning", "pathstar_planning")


class ReasoningRegimeSummary(StrictSchema):
    """Coverage-aware Track 4 mechanism aggregates."""

    chain_average: float | None
    planning_average: float | None
    reasoning_regime_gap: float | None
    chain_coverage: int = Field(ge=0, le=2)
    planning_coverage: int = Field(ge=0, le=2)


def aggregate_reasoning_regimes(
    task_scores: Mapping[str, float | None],
) -> ReasoningRegimeSummary:
    """Compute complete-pair macro means and ``planning - chain`` gap."""
    for task_id, score in task_scores.items():
        if score is not None and not 0.0 <= score <= 1.0:
            raise EvaluationError(f"task score {task_id!r} is outside [0, 1]")

    chain_values = [task_scores.get(task_id) for task_id in CHAIN_TASKS]
    planning_values = [task_scores.get(task_id) for task_id in PLANNING_TASKS]
    chain_present = [value for value in chain_values if value is not None]
    planning_present = [value for value in planning_values if value is not None]
    chain_average = fmean(chain_present) if len(chain_present) == 2 else None
    planning_average = fmean(planning_present) if len(planning_present) == 2 else None
    gap = (
        planning_average - chain_average
        if planning_average is not None and chain_average is not None
        else None
    )
    return ReasoningRegimeSummary(
        chain_average=chain_average,
        planning_average=planning_average,
        reasoning_regime_gap=gap,
        chain_coverage=len(chain_present),
        planning_coverage=len(planning_present),
    )


__all__ = [
    "CHAIN_TASKS",
    "PLANNING_TASKS",
    "ReasoningRegimeSummary",
    "aggregate_reasoning_regimes",
]
