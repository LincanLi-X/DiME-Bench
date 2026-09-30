"""Prepared-data integrity validation and machine-readable reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from dimebench.artifacts.hashing import hash_json
from dimebench.datasets.base import DatasetError, NormalizedDataset
from dimebench.datasets.records import expected_sample_id
from dimebench.datasets.split_lock import apply_split_lock, load_split_lock
from dimebench.schemas.base import StrictSchema


class ValidationReport(StrictSchema):
    """Summary of all deterministic data integrity checks."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["passed", "failed"]
    suite: str
    dataset_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    dataset_hashes: dict[str, str]
    lock_hashes: dict[str, str]
    errors: tuple[str, ...] = ()


def validate_prepared_data(path: str | Path) -> ValidationReport:
    """Validate a prepared suite manifest, records, IDs, hashes, and locks."""
    root = Path(path).resolve()
    manifest_path = root / "suite-manifest.json"
    errors: list[str] = []
    dataset_hashes: dict[str, str] = {}
    lock_hashes: dict[str, str] = {}
    sample_count = 0
    try:
        suite_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationReport(
            status="failed",
            suite=root.name,
            dataset_count=0,
            sample_count=0,
            dataset_hashes={},
            lock_hashes={},
            errors=(f"cannot read suite manifest: {exc}",),
        )
    entries = suite_manifest.get("datasets")
    if not isinstance(entries, list):
        return ValidationReport(
            status="failed",
            suite=str(suite_manifest.get("suite", root.name)),
            dataset_count=0,
            sample_count=0,
            dataset_hashes={},
            lock_hashes={},
            errors=("suite manifest datasets must be a list",),
        )
    expected_suite_hash = suite_manifest.get("suite_hash")
    observed_suite_hash = hash_json(entries)
    if expected_suite_hash != observed_suite_hash:
        errors.append("suite manifest hash mismatch")
    seen_keys: set[str] = set()
    for entry in entries:
        try:
            if not isinstance(entry, dict):
                raise DatasetError("suite dataset entry must be an object")
            dataset_id = str(entry["dataset_id"])
            split = str(entry["split"])
            key = f"{dataset_id}:{split}"
            if key in seen_keys:
                raise DatasetError(f"duplicate suite entry {key!r}")
            seen_keys.add(key)
            dataset = NormalizedDataset.read_jsonl(
                root / str(entry["data_path"]),
                dataset_id=dataset_id,
                split=split,
            )
            unstable = [
                record.sample_id
                for record in dataset
                if expected_sample_id(record) != record.sample_id
            ]
            if unstable:
                raise DatasetError(
                    "unstable sample_id values: " + ", ".join(unstable[:3])
                )
            lock = load_split_lock(root / str(entry["lock_path"]))
            locked = apply_split_lock(dataset, lock)
            if locked.dataset_hash != entry["dataset_hash"]:
                raise DatasetError("suite manifest dataset hash mismatch")
            if lock.lock_hash != entry["lock_hash"]:
                raise DatasetError("suite manifest lock hash mismatch")
            if len(locked) != entry["sample_count"]:
                raise DatasetError("suite manifest sample count mismatch")
            dataset_hashes[key] = locked.dataset_hash
            lock_hashes[key] = lock.lock_hash
            sample_count += len(locked)
        except (DatasetError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{entry!r}: {exc}")
    return ValidationReport(
        status="failed" if errors else "passed",
        suite=str(suite_manifest.get("suite", root.name)),
        dataset_count=len(entries),
        sample_count=sample_count,
        dataset_hashes=dataset_hashes,
        lock_hashes=lock_hashes,
        errors=tuple(errors),
    )


__all__ = ["ValidationReport", "validate_prepared_data"]
