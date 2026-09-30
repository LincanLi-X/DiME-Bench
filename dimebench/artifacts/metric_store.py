"""Sample metric JSONL storage and traceable summary generation."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import Field

from dimebench.artifacts.hashing import (
    canonical_json,
    hash_file,
    hash_json,
    hash_ordered_strings,
    write_json_atomic,
)
from dimebench.artifacts.prediction_store import (
    ArtifactFormatError,
    DuplicateSampleError,
)
from dimebench.artifacts.provenance import utc_now
from dimebench.schemas.base import StrictSchema


class SampleMetricRecord(StrictSchema):
    """Metrics derived from one immutable prediction record."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    prediction_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["success", "failure", "skipped"]
    metrics: dict[str, float] = Field(default_factory=dict)
    evaluator_metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def record_hash(self) -> str:
        """Hash this exact sample-metric record."""
        return hash_json(self)


class MetricAggregate(StrictSchema):
    """One aggregate with its exact contributing sample set."""

    value: float | None
    sample_count: int = Field(ge=0)
    sample_ids: tuple[str, ...] = ()
    sample_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_sample_count: int | None = Field(default=None, ge=0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)


class SummaryRecord(StrictSchema):
    """Run summary linked to sample metrics and the resolved config."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "completed"] = "pending"
    metrics: dict[str, MetricAggregate] = Field(default_factory=dict)
    sample_metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime = Field(default_factory=utc_now)


class MetricStore:
    """Single-writer sample metric store with deterministic mean summaries."""

    def __init__(self, path: str | Path, run_id: str, config_hash: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.config_hash = config_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._records = self._load_records()

    def _load_records(self) -> dict[str, SampleMetricRecord]:
        records: dict[str, SampleMetricRecord] = {}
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = SampleMetricRecord.model_validate_json(line)
                except ValueError as exc:
                    raise ArtifactFormatError(
                        f"invalid sample metric at {self.path}:{line_number}: {exc}"
                    ) from exc
                self._validate_identity(record, line_number)
                if record.sample_id in records:
                    raise ArtifactFormatError(
                        f"duplicate sample_id {record.sample_id!r} in {self.path}"
                    )
                records[record.sample_id] = record
        return records

    def _validate_identity(
        self,
        record: SampleMetricRecord,
        line_number: int | None = None,
    ) -> None:
        location = f" at line {line_number}" if line_number is not None else ""
        if record.run_id != self.run_id:
            raise ArtifactFormatError(
                f"sample metric run_id mismatch{location}: {record.run_id!r}"
            )
        if record.config_hash != self.config_hash:
            raise ArtifactFormatError(
                f"sample metric config_hash mismatch{location}: {record.config_hash!r}"
            )

    def append(self, record: SampleMetricRecord) -> None:
        """Durably append metrics for one sample."""
        self._validate_identity(record)
        if record.sample_id in self._records:
            raise DuplicateSampleError(
                f"metrics for sample {record.sample_id!r} already exist"
            )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._records[record.sample_id] = record

    @property
    def completed_sample_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def get(self, sample_id: str) -> SampleMetricRecord:
        """Return stored metrics by sample ID."""
        try:
            return self._records[sample_id]
        except KeyError as exc:
            raise KeyError(f"metrics for sample {sample_id!r} were not found") from exc

    def pending_sample_ids(self, sample_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sample_id for sample_id in sample_ids if sample_id not in self._records
        )

    def summarize(self) -> SummaryRecord:
        """Compute means over every sample carrying an explicit metric value.

        Primary evaluators write zero-valued metrics for failed and skipped
        samples. Including those records here enforces the benchmark's frozen
        no-silent-dropping policy.
        """
        contributions: dict[str, list[tuple[str, float]]] = defaultdict(list)
        scheduled: dict[str, set[str]] = defaultdict(set)
        for sample_id in sorted(self._records):
            record = self._records[sample_id]
            for metric_id, value in sorted(record.metrics.items()):
                contributions[metric_id].append((sample_id, value))
                scheduled[metric_id].add(sample_id)
            for key in record.evaluator_metadata:
                if key.endswith(".version"):
                    scheduled[key.removesuffix(".version")].add(sample_id)

        aggregates: dict[str, MetricAggregate] = {}
        for metric_id in sorted(set(contributions) | set(scheduled)):
            values = contributions[metric_id]
            sample_ids = tuple(sample_id for sample_id, _ in values)
            scheduled_ids = scheduled[metric_id]
            total = len(scheduled_ids)
            failure_count = sum(
                self._records[sample_id].status == "failure"
                for sample_id in scheduled_ids
            )
            skipped_count = sum(
                self._records[sample_id].status == "skipped"
                for sample_id in scheduled_ids
            )
            aggregates[metric_id] = MetricAggregate(
                value=fmean(value for _, value in values) if values else None,
                sample_count=len(values),
                sample_ids=sample_ids,
                sample_ids_hash=hash_ordered_strings(sample_ids),
                total_sample_count=total,
                coverage=len(values) / total if total else 0.0,
                failure_count=failure_count,
                skipped_count=skipped_count,
            )
        return SummaryRecord(
            run_id=self.run_id,
            config_hash=self.config_hash,
            status="completed",
            metrics=aggregates,
            sample_metrics_sha256=hash_file(self.path),
        )

    def write_summary(self, path: str | Path) -> SummaryRecord:
        """Write and return a traceable summary for current sample metrics."""
        summary = self.summarize()
        write_json_atomic(path, summary)
        return summary

    def __len__(self) -> int:
        return len(self._records)
