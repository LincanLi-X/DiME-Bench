"""Canonical JSON serialization and content hashing utilities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value deterministically."""
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_json(value: object) -> str:
    """Hash the canonical JSON representation of a value."""
    return hash_bytes(canonical_json(value).encode("utf-8"))


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into a SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_ordered_strings(values: Iterable[str]) -> str:
    """Hash an ordered sequence without separator ambiguity."""
    return hash_json(list(values))


def write_json_atomic(path: str | Path, value: object) -> None:
    """Atomically write pretty, deterministic JSON with a trailing newline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
