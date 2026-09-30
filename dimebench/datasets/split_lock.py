"""Deterministic sample selection and immutable split lock files."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from dimebench.artifacts.hashing import hash_json, write_json_atomic
from dimebench.datasets.base import DatasetError, NormalizedDataset
from dimebench.datasets.records import SampleRecord
from dimebench.schemas.base import StrictSchema


class SplitLock(StrictSchema):
    """Frozen ordered sample membership for one normalized split."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    split: str
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)
    shuffle: bool
    sample_limit: int | None = Field(default=None, gt=0)
    sample_ids: tuple[str, ...] = Field(min_length=1)
    record_hashes: tuple[str, ...] = Field(min_length=1)
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lock(self) -> SplitLock:
        if len(self.sample_ids) != len(self.record_hashes):
            raise ValueError("split lock sample_ids and record_hashes must align")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("split lock sample_ids must be unique")
        expected = hash_json(
            self.model_dump(mode="json", exclude={"lock_hash"}, exclude_none=False)
        )
        if self.lock_hash != expected:
            raise ValueError("split lock hash does not match its contents")
        return self


def freeze_split(
    dataset: NormalizedDataset,
    manifest_hash: str,
    *,
    seed: int = 0,
    shuffle: bool = False,
    sample_limit: int | None = None,
) -> tuple[NormalizedDataset, SplitLock]:
    """Select records from a canonical order and return a self-hashing lock."""
    records = sorted(dataset.records, key=lambda record: record.sample_id)
    if shuffle:
        random.Random(seed).shuffle(records)
    if sample_limit is not None:
        if sample_limit <= 0:
            raise DatasetError("sample_limit must be positive")
        records = records[:sample_limit]
    if not records:
        raise DatasetError("cannot freeze an empty split")
    frozen = NormalizedDataset(dataset.dataset_id, dataset.split, records)
    payload = {
        "schema_version": "1.0",
        "dataset_id": frozen.dataset_id,
        "split": frozen.split,
        "manifest_hash": manifest_hash,
        "seed": seed,
        "shuffle": shuffle,
        "sample_limit": sample_limit,
        "sample_ids": tuple(record.sample_id for record in frozen),
        "record_hashes": tuple(record.record_hash for record in frozen),
        "dataset_hash": frozen.dataset_hash,
    }
    lock = SplitLock.model_validate({**payload, "lock_hash": hash_json(payload)})
    return frozen, lock


def apply_split_lock(
    dataset: NormalizedDataset,
    lock: SplitLock,
) -> NormalizedDataset:
    """Reproduce a frozen order and reject missing or modified samples."""
    if dataset.dataset_id != lock.dataset_id or dataset.split != lock.split:
        raise DatasetError("split lock does not match dataset identity")
    by_id: dict[str, SampleRecord] = {
        record.sample_id: record for record in dataset.records
    }
    try:
        records = tuple(by_id[sample_id] for sample_id in lock.sample_ids)
    except KeyError as exc:
        raise DatasetError(
            f"split lock references missing sample {exc.args[0]!r}"
        ) from exc
    observed_hashes = tuple(record.record_hash for record in records)
    if observed_hashes != lock.record_hashes:
        raise DatasetError("split lock record hash mismatch")
    locked = NormalizedDataset(dataset.dataset_id, dataset.split, records)
    if locked.dataset_hash != lock.dataset_hash:
        raise DatasetError("split lock dataset hash mismatch")
    return locked


def write_split_lock(path: str | Path, lock: SplitLock) -> None:
    """Atomically write a split lock."""
    write_json_atomic(path, lock)


def load_split_lock(path: str | Path) -> SplitLock:
    """Load and self-validate a split lock."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return SplitLock.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetError(f"invalid split lock {source}: {exc}") from exc


__all__ = [
    "SplitLock",
    "apply_split_lock",
    "freeze_split",
    "load_split_lock",
    "write_split_lock",
]
