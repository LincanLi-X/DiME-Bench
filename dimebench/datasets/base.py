"""Dataset manifest schemas and normalized JSONL storage."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, JsonValue, ValidationError, model_validator

from dimebench.artifacts.hashing import hash_json
from dimebench.datasets.records import SampleKind, SampleRecord, parse_sample_record
from dimebench.schemas.base import StrictSchema


class DatasetError(ValueError):
    """Raised for invalid manifests, records, downloads, or prepared data."""


class DatasetSource(StrictSchema):
    """A reproducible upstream or manually supplied dataset source."""

    type: Literal["huggingface", "url", "local", "manual", "builtin"]
    name_or_path: str | None = None
    subset: str | None = None
    revision: str | None = None
    url: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    instructions: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> DatasetSource:
        if self.type in {"huggingface", "local"} and not self.name_or_path:
            raise ValueError(f"{self.type} source requires name_or_path")
        if self.type == "url" and (not self.url or not self.sha256):
            raise ValueError("url source requires url and sha256")
        if self.type == "manual" and not self.instructions:
            raise ValueError("manual source requires instructions")
        return self


class PreprocessingSpec(StrictSchema):
    """Name and parameters of one deterministic normalization strategy."""

    strategy: Literal[
        "normalized",
        "multiple_choice",
        "generation",
        "infilling",
        "editing",
        "reasoning",
    ]
    field_map: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class DatasetManifest(StrictSchema):
    """Repository-tracked preparation contract for one logical dataset."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    display_name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    track: Literal[
        "track1_general",
        "track2_infilling",
        "track3_editing",
        "track4_reasoning",
    ]
    record_kind: SampleKind
    license: str = Field(min_length=1)
    homepage: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    source: DatasetSource
    splits: tuple[str, ...] = Field(min_length=1)
    sample_path: str = Field(min_length=1)
    preprocessing: PreprocessingSpec
    notes: str | None = None

    @property
    def manifest_hash(self) -> str:
        return hash_json(self.model_dump(mode="json", exclude_none=False))


class SuiteSpec(StrictSchema):
    """An ordered set of dataset manifests."""

    datasets: tuple[str, ...] = Field(min_length=1)


class DataRegistry(StrictSchema):
    """Top-level suite and manifest index."""

    schema_version: Literal["1.0"] = "1.0"
    suites: dict[str, SuiteSpec]
    datasets: dict[str, str]

    @model_validator(mode="after")
    def validate_references(self) -> DataRegistry:
        missing = {
            dataset_id
            for suite in self.suites.values()
            for dataset_id in suite.datasets
            if dataset_id not in self.datasets
        }
        if missing:
            raise ValueError(
                "suite references unknown datasets: " + ", ".join(sorted(missing))
            )
        return self


class NormalizedDataset:
    """Immutable ordered records for exactly one dataset split."""

    def __init__(
        self,
        dataset_id: str,
        split: str,
        records: Iterable[SampleRecord],
    ) -> None:
        self.dataset_id = dataset_id
        self.split = split
        self.records = tuple(records)
        if any(record.dataset_id != dataset_id for record in self.records):
            raise DatasetError("normalized dataset contains a mismatched dataset_id")
        if any(record.split != split for record in self.records):
            raise DatasetError("normalized dataset contains a mismatched split")
        identifiers = [record.sample_id for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise DatasetError("normalized dataset contains duplicate sample_id values")

    def __iter__(self) -> Iterator[SampleRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def dataset_hash(self) -> str:
        """Hash the ordered normalized records."""
        return hash_json(
            [record.model_dump(mode="json", exclude_none=False) for record in self]
        )

    def write_jsonl(self, path: str | Path) -> None:
        """Atomically write deterministic normalized JSONL."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
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
                for record in self:
                    payload = json.dumps(
                        record.model_dump(mode="json", exclude_none=False),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    @classmethod
    def read_jsonl(
        cls,
        path: str | Path,
        *,
        dataset_id: str | None = None,
        split: str | None = None,
    ) -> NormalizedDataset:
        """Read and validate normalized JSONL with actionable line errors."""
        source = Path(path)
        records: list[SampleRecord] = []
        try:
            with source.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(parse_sample_record(json.loads(line)))
                    except (json.JSONDecodeError, ValidationError) as exc:
                        raise DatasetError(
                            f"invalid record at {source}:{line_number}: {exc}"
                        ) from exc
        except OSError as exc:
            raise DatasetError(f"cannot read dataset {source}: {exc}") from exc
        if not records:
            raise DatasetError(f"dataset {source} contains no records")
        resolved_dataset_id = dataset_id or records[0].dataset_id
        resolved_split = split or records[0].split
        return cls(resolved_dataset_id, resolved_split, records)


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise DatasetError(f"cannot read data config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DatasetError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DatasetError(f"data config {path} must contain a mapping")
    return payload


def load_data_registry(path: str | Path) -> DataRegistry:
    """Load the repository data registry."""
    source = Path(path)
    try:
        return DataRegistry.model_validate(_load_yaml_mapping(source))
    except ValidationError as exc:
        raise DatasetError(f"invalid data registry {source}: {exc}") from exc


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load one dataset manifest."""
    source = Path(path)
    try:
        return DatasetManifest.model_validate(_load_yaml_mapping(source))
    except ValidationError as exc:
        raise DatasetError(f"invalid dataset manifest {source}: {exc}") from exc


__all__ = [
    "DataRegistry",
    "DatasetError",
    "DatasetManifest",
    "DatasetSource",
    "NormalizedDataset",
    "PreprocessingSpec",
    "SuiteSpec",
    "load_data_registry",
    "load_dataset_manifest",
]
