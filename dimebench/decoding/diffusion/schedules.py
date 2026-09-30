"""Diffusion schedule names and validation."""

from __future__ import annotations

from typing import Literal, cast

DiffusionSchedule = Literal["linear"]


def normalize_schedule(schedule: str) -> DiffusionSchedule:
    """Validate schedules implemented by the initial LLaDA sampler."""
    normalized = schedule.strip().lower().replace("-", "_")
    if normalized != "linear":
        raise ValueError(
            f"unsupported diffusion schedule {schedule!r}; supported: linear"
        )
    return cast(DiffusionSchedule, normalized)


def linear_remaining_ratio(step: int, total_steps: int) -> float:
    """Return the ideal remaining mask ratio after a completed step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0 or step > total_steps:
        raise ValueError("step must be in [0, total_steps]")
    return 1.0 - (step / total_steps)
