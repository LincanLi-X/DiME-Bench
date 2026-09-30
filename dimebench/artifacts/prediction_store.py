"""Validated append-only storage for sample predictions."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, ValidationError, model_validator

from dimebench.artifacts.hashing import canonical_json, hash_json
from dimebench.artifacts.provenance import utc_now
from dimebench.schemas.base import RawText, StrictSchema
from dimebench.schemas.result import FailureClass


class ArtifactFormatError(ValueError):
    """Raised when an artifact is corrupt or violates its record contract."""


class DuplicateSampleError(ArtifactFormatError):
    """Raised when a store already contains a sample ID."""


class PredictionRecord(StrictSchema):
    """One model output and its generation metadata."""

    model_config = ConfigDict(str_strip_whitespace=False)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    status: Literal["success", "failure", "skipped"]
    request: dict[str, JsonValue] = Field(default_factory=dict)
    raw_output: RawText | None = None
    failure_class: FailureClass | None = None
    error_message: str | None = None
    decoding_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_status(self) -> PredictionRecord:
        if self.status == "success":
            if self.raw_output is None:
                raise ValueError("successful predictions require raw_output")
            if self.failure_class is not None or self.error_message is not None:
                raise ValueError(
                    "successful predictions cannot contain failure details"
                )
        elif self.status == "failure" and self.failure_class is None:
            raise ValueError("failed predictions require failure_class")
        return self

    @property
    def record_hash(self) -> str:
        """Hash the complete prediction record for downstream linkage."""
        return hash_json(self)


class PredictionStore:
    """Single-writer JSONL store with resume and duplicate protection."""

    def __init__(self, path: str | Path, run_id: str, config_hash: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.config_hash = config_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._records = self._load_records()

    def _load_records(self) -> dict[str, PredictionRecord]:
        records: dict[str, PredictionRecord] = {}
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = PredictionRecord.model_validate_json(line)
                except ValidationError as exc:
                    raise ArtifactFormatError(
                        f"invalid prediction at {self.path}:{line_number}: {exc}"
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
        record: PredictionRecord,
        line_number: int | None = None,
    ) -> None:
        location = f" at line {line_number}" if line_number is not None else ""
        if record.run_id != self.run_id:
            raise ArtifactFormatError(
                f"prediction run_id mismatch{location}: {record.run_id!r}"
            )
        if record.config_hash != self.config_hash:
            raise ArtifactFormatError(
                f"prediction config_hash mismatch{location}: {record.config_hash!r}"
            )

    def append(self, record: PredictionRecord) -> None:
        """Durably append one record; never replace an existing sample."""
        self._validate_identity(record)
        if record.sample_id in self._records:
            raise DuplicateSampleError(
                f"prediction for sample {record.sample_id!r} already exists"
            )
        payload = canonical_json(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._records[record.sample_id] = record

    def get(self, sample_id: str) -> PredictionRecord:
        """Return a stored prediction by sample ID."""
        try:
            return self._records[sample_id]
        except KeyError as exc:
            raise KeyError(
                f"prediction for sample {sample_id!r} was not found"
            ) from exc

    @property
    def completed_sample_ids(self) -> frozenset[str]:
        """Return all terminal sample IDs, including retained failures."""
        return frozenset(self._records)

    def pending_sample_ids(self, sample_ids: Iterable[str]) -> tuple[str, ...]:
        """Filter an ordered sample sequence for resumable execution."""
        return tuple(
            sample_id for sample_id in sample_ids if sample_id not in self._records
        )

    def __len__(self) -> int:
        return len(self._records)
