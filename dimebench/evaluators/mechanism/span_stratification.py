"""Track 2 span-length and boundary-density grouped aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Literal

from pydantic import Field

from dimebench.artifacts import SampleMetricRecord
from dimebench.datasets import InfillingRecord
from dimebench.evaluators.base import EvaluationError
from dimebench.schemas.base import StrictSchema

Stratifier = Literal["span_length_bin", "boundary_density"]


class StratumAggregate(StrictSchema):
    """One grouped score with explicit eligibility and failure coverage."""

    value: float | None
    eligible_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    failure_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)


class StratificationResult(StrictSchema):
    """Versioned grouped results for one metric and one frozen sample label."""

    metric_id: str = Field(min_length=1)
    group_by: Stratifier
    groups: dict[str, StratumAggregate]


def stratify_infilling_metric(
    records: Sequence[SampleMetricRecord],
    samples: Mapping[str, InfillingRecord],
    metric_id: str,
    *,
    group_by: Stratifier,
) -> StratificationResult:
    """Macro-average a metric by a label stored in the frozen dataset record."""
    scheduled: dict[str, list[SampleMetricRecord]] = defaultdict(list)
    for record in records:
        if metric_id not in record.metrics and (
            f"{metric_id}.version" not in record.evaluator_metadata
        ):
            continue
        try:
            sample = samples[record.sample_id]
        except KeyError as exc:
            raise EvaluationError(
                f"missing infilling sample {record.sample_id!r} for stratification"
            ) from exc
        label = getattr(sample, group_by)
        scheduled[label].append(record)

    groups: dict[str, StratumAggregate] = {}
    for label, group_records in sorted(scheduled.items()):
        values = [
            record.metrics[metric_id]
            for record in group_records
            if metric_id in record.metrics
        ]
        total = len(group_records)
        groups[label] = StratumAggregate(
            value=fmean(values) if values else None,
            eligible_count=len(values),
            total_sample_count=total,
            coverage=len(values) / total if total else 0.0,
            failure_count=sum(record.status == "failure" for record in group_records),
            skipped_count=sum(record.status == "skipped" for record in group_records),
        )
    return StratificationResult(
        metric_id=metric_id,
        group_by=group_by,
        groups=groups,
    )


__all__ = [
    "StratificationResult",
    "Stratifier",
    "StratumAggregate",
    "stratify_infilling_metric",
]
