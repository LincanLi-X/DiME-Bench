"""Dataset source acquisition with checksum and optional-dependency guards."""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from dimebench.artifacts.hashing import hash_file
from dimebench.datasets.base import DatasetError, DatasetManifest
from dimebench.datasets.synthetic_repair import generate_synthetic_repair_rows


class DatasetDependencyError(DatasetError):
    """Raised when a requested source requires an uninstalled dependency."""


class ManualDownloadRequired(DatasetError):
    """Raised when licensing requires a user-supplied source file."""


def read_jsonl_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object rows with line-aware errors."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"invalid JSON at {source}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise DatasetError(
                        f"JSONL row at {source}:{line_number} must be an object"
                    )
                yield dict(payload)
    except OSError as exc:
        raise DatasetError(f"cannot read JSONL source {source}: {exc}") from exc


def load_bundled_sample_rows(
    manifest: DatasetManifest,
    data_root: str | Path,
    split: str,
) -> tuple[dict[str, Any], ...]:
    """Load the synthetic schema fixtures committed with the repository."""
    sample_path = Path(data_root) / manifest.sample_path
    rows = tuple(
        row
        for row in read_jsonl_rows(sample_path)
        if row.get("dataset_id") == manifest.id and row.get("split", split) == split
    )
    if not rows:
        raise DatasetError(
            f"no bundled samples for dataset={manifest.id!r}, split={split!r} "
            f"in {sample_path}"
        )
    return rows


def _write_rows(path: Path, rows: Iterator[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _download_url(manifest: DatasetManifest, destination: Path) -> Path:
    source = manifest.source
    assert source.url is not None
    assert source.sha256 is not None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(source.url, timeout=60) as response:  # noqa: S310
            with destination.open("wb") as stream:
                shutil.copyfileobj(response, stream)
    except (OSError, urllib.error.URLError) as exc:
        destination.unlink(missing_ok=True)
        raise DatasetError(f"cannot download {source.url}: {exc}") from exc
    observed = hash_file(destination)
    if observed != source.sha256:
        destination.unlink(missing_ok=True)
        raise DatasetError(
            f"checksum mismatch for {source.url}: expected {source.sha256}, "
            f"observed {observed}"
        )
    return destination


def _download_huggingface(
    manifest: DatasetManifest,
    split: str,
    destination: Path,
) -> Path:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DatasetDependencyError(
            "Hugging Face dataset downloads require `pip install dime-bench[hf]`"
        ) from exc
    source = manifest.source
    assert source.name_or_path is not None
    try:
        dataset = load_dataset(
            source.name_or_path,
            source.subset,
            revision=source.revision,
            split=split,
        )
    except Exception as exc:
        raise DatasetError(
            f"cannot load Hugging Face dataset {source.name_or_path!r} "
            f"split {split!r}: {exc}"
        ) from exc
    _write_rows(destination, (dict(row) for row in dataset))
    return destination


def acquire_full_source(
    manifest: DatasetManifest,
    split: str,
    cache_root: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Acquire one full raw split and return a local JSONL-compatible path."""
    cache = Path(cache_root) / manifest.id / manifest.version
    destination = cache / f"{split}.jsonl"
    if destination.exists() and not force:
        return destination
    source = manifest.source
    if source.type == "manual":
        raise ManualDownloadRequired(source.instructions or "manual download required")
    if source.type == "builtin":
        if manifest.id != "synthetic_repair":
            raise ManualDownloadRequired(
                f"no full-mode builder is registered for {manifest.id!r}"
            )
        _write_rows(destination, generate_synthetic_repair_rows())
        return destination
    if source.type == "local":
        assert source.name_or_path is not None
        local_path = Path(source.name_or_path).expanduser()
        if not local_path.is_file():
            raise DatasetError(f"local dataset source does not exist: {local_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)
        return destination
    if source.type == "url":
        return _download_url(manifest, destination)
    if source.type == "huggingface":
        return _download_huggingface(manifest, split, destination)
    raise DatasetError(f"unsupported dataset source type: {source.type}")


__all__ = [
    "DatasetDependencyError",
    "ManualDownloadRequired",
    "acquire_full_source",
    "load_bundled_sample_rows",
    "read_jsonl_rows",
]
