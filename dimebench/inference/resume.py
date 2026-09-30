"""Deterministic resume planning from terminal prediction records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from dimebench.artifacts import PredictionStore

T = TypeVar("T")


@dataclass(frozen=True)
class ResumePlan:
    """Ordered pending item indexes and already completed sample IDs."""

    pending_indexes: tuple[int, ...]
    completed_sample_ids: tuple[str, ...]


def plan_resume(sample_ids: tuple[str, ...], store: PredictionStore) -> ResumePlan:
    """Build an order-preserving plan and reject ambiguous input IDs."""
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("resume planning requires unique sample IDs")
    completed = store.completed_sample_ids
    return ResumePlan(
        pending_indexes=tuple(
            index
            for index, sample_id in enumerate(sample_ids)
            if sample_id not in completed
        ),
        completed_sample_ids=tuple(
            sample_id for sample_id in sample_ids if sample_id in completed
        ),
    )


__all__ = ["ResumePlan", "plan_resume"]
