"""Append-only inference lifecycle tracing."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from dimebench.artifacts.hashing import canonical_json
from dimebench.artifacts.provenance import utc_now
from dimebench.schemas.base import StrictSchema

TraceEventType = Literal[
    "run_started",
    "run_resumed",
    "batch_started",
    "cache_hit",
    "retry",
    "prediction_written",
    "run_finished",
    "run_failed",
]


class InferenceTraceEvent(StrictSchema):
    """One durable operational event without parsed/scored model output."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    event: TraceEventType
    timestamp: datetime = Field(default_factory=utc_now)
    sample_ids: tuple[str, ...] = ()
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TraceWriter:
    """Durably append trace events as canonical JSONL."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(
        self,
        event: TraceEventType,
        *,
        sample_ids: tuple[str, ...] = (),
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        record = InferenceTraceEvent(
            run_id=self.run_id,
            event=event,
            sample_ids=sample_ids,
            details=details or {},
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


__all__ = ["InferenceTraceEvent", "TraceEventType", "TraceWriter"]
