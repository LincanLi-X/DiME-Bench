"""Stable cost-aware partitioning of isolated model × task jobs."""

from __future__ import annotations

from pydantic import Field

from dimebench.runners.base import BenchmarkJob
from dimebench.schemas.base import StrictSchema


class JobPartition(StrictSchema):
    """Ordered jobs assigned to one worker."""

    worker_id: int = Field(ge=0)
    estimated_cost: float = Field(ge=0.0)
    jobs: tuple[BenchmarkJob, ...]


def partition_jobs(
    jobs: tuple[BenchmarkJob, ...],
    worker_count: int,
) -> tuple[JobPartition, ...]:
    """Greedily balance jobs with deterministic tie-breaking."""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not jobs:
        raise ValueError("cannot partition an empty job sequence")
    loads = [0.0 for _ in range(worker_count)]
    assignments: list[list[BenchmarkJob]] = [[] for _ in range(worker_count)]
    ordered = sorted(
        jobs,
        key=lambda job: (-job.resources.estimated_cost, job.job_id),
    )
    for job in ordered:
        worker_id = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker_id].append(job)
        loads[worker_id] += job.resources.estimated_cost
    return tuple(
        JobPartition(
            worker_id=worker_id,
            estimated_cost=loads[worker_id],
            jobs=tuple(sorted(worker_jobs, key=lambda job: job.job_id)),
        )
        for worker_id, worker_jobs in enumerate(assignments)
    )


__all__ = ["JobPartition", "partition_jobs"]
