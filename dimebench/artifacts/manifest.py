"""Run directory lifecycle and cross-artifact traceability validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isclose
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import Field, JsonValue, ValidationError

from dimebench.artifacts.hashing import (
    hash_file,
    hash_json,
    hash_ordered_strings,
    write_json_atomic,
)
from dimebench.artifacts.metric_store import MetricStore, SummaryRecord
from dimebench.artifacts.prediction_store import PredictionStore
from dimebench.artifacts.provenance import (
    EnvironmentInfo,
    collect_environment,
    utc_now,
)
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.run import RunSpec
from dimebench.version import __version__


class ArtifactError(ValueError):
    """Raised when a run directory is conflicting, incomplete, or corrupt."""


class ArtifactNames(StrictSchema):
    """Fixed filenames in every DiME-Bench run directory."""

    manifest: Literal["run_manifest.json"] = "run_manifest.json"
    environment: Literal["environment.json"] = "environment.json"
    predictions: Literal["predictions.jsonl"] = "predictions.jsonl"
    sample_metrics: Literal["sample_metrics.jsonl"] = "sample_metrics.jsonl"
    summary: Literal["summary.json"] = "summary.json"


class RunManifest(StrictSchema):
    """Authoritative identity, configuration, and provenance links for a run."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["dime_bench_v1"] = "dime_bench_v1"
    run_id: str = Field(min_length=1)
    status: Literal["running", "completed", "failed"] = "running"
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: dict[str, JsonValue]
    dimebench_version: str
    adapter_name: str
    adapter_version: str
    model_id: str
    model_family: Literal["autoregressive", "diffusion"]
    model_revision: str
    tokenizer_revision: str
    dataset_id: str
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(gt=0)
    seed: int = Field(ge=0)
    artifacts: ArtifactNames = Field(default_factory=ArtifactNames)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


@dataclass(frozen=True)
class RunPaths:
    """Resolved standard paths for one run directory."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def environment(self) -> Path:
        return self.root / "environment.json"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def sample_metrics(self) -> Path:
        return self.root / "sample_metrics.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def all_files(self) -> tuple[Path, ...]:
        return (
            self.manifest,
            self.environment,
            self.predictions,
            self.sample_metrics,
            self.summary,
        )


@dataclass(frozen=True)
class RunArtifacts:
    """Result of run initialization or compatible resumption."""

    paths: RunPaths
    manifest: RunManifest
    resumed: bool


def _load_model(path: Path, model_type: type[StrictSchema]) -> StrictSchema:
    try:
        payload = path.read_text(encoding="utf-8")
        return model_type.model_validate_json(payload)
    except (OSError, ValidationError, ValueError) as exc:
        raise ArtifactError(f"invalid artifact {path}: {exc}") from exc


def load_manifest(path: str | Path) -> RunManifest:
    """Load and validate a run manifest."""
    loaded = _load_model(Path(path), RunManifest)
    if not isinstance(loaded, RunManifest):
        raise ArtifactError(f"unexpected manifest type at {path}")
    return loaded


def load_environment(path: str | Path) -> EnvironmentInfo:
    """Load and validate an environment snapshot."""
    loaded = _load_model(Path(path), EnvironmentInfo)
    if not isinstance(loaded, EnvironmentInfo):
        raise ArtifactError(f"unexpected environment type at {path}")
    return loaded


def load_summary(path: str | Path) -> SummaryRecord:
    """Load and validate a run summary."""
    loaded = _load_model(Path(path), SummaryRecord)
    if not isinstance(loaded, SummaryRecord):
        raise ArtifactError(f"unexpected summary type at {path}")
    return loaded


def _resolve_dataset_hash(spec: RunSpec, dataset_hash: str | None) -> str:
    resolved = dataset_hash or spec.dataset.manifest_hash
    if resolved is None:
        raise ArtifactError(
            "dataset hash is required: pass dataset_hash or set dataset.manifest_hash"
        )
    if len(resolved) != 64 or any(
        character not in "0123456789abcdef" for character in resolved
    ):
        raise ArtifactError("dataset hash must be a lowercase SHA-256 digest")
    return resolved


def initialize_run(
    spec: RunSpec,
    sample_ids: list[str] | tuple[str, ...],
    *,
    dataset_hash: str | None = None,
    environment: EnvironmentInfo | None = None,
) -> RunArtifacts:
    """Create a standard run directory or safely resume an identical run."""
    if not sample_ids:
        raise ArtifactError("at least one sample_id is required")
    if len(set(sample_ids)) != len(sample_ids):
        raise ArtifactError("sample_ids must be unique")

    resolved_dataset_hash = _resolve_dataset_hash(spec, dataset_hash)
    sample_ids_hash = hash_ordered_strings(sample_ids)
    paths = RunPaths(Path(spec.output_dir) / spec.run_id)

    if paths.manifest.exists():
        manifest = load_manifest(paths.manifest)
        conflicts = []
        if manifest.config_hash != spec.config_hash:
            conflicts.append("config_hash")
        if manifest.dataset_manifest_hash != resolved_dataset_hash:
            conflicts.append("dataset_manifest_hash")
        if manifest.sample_ids_hash != sample_ids_hash:
            conflicts.append("sample_ids_hash")
        missing = [path.name for path in paths.all_files() if not path.exists()]
        if conflicts or missing:
            details = []
            if conflicts:
                details.append(f"conflicting fields: {', '.join(conflicts)}")
            if missing:
                details.append(f"missing artifacts: {', '.join(missing)}")
            raise ArtifactError(
                f"cannot resume run {spec.run_id!r}: {'; '.join(details)}"
            )
        return RunArtifacts(paths=paths, manifest=manifest, resumed=True)

    if paths.root.exists() and any(paths.root.iterdir()):
        raise ArtifactError(
            f"run directory {paths.root} is non-empty but has no manifest"
        )

    paths.root.mkdir(parents=True, exist_ok=True)
    captured_environment = environment or collect_environment()
    write_json_atomic(paths.environment, captured_environment)
    paths.predictions.touch(exist_ok=False)
    paths.sample_metrics.touch(exist_ok=False)
    pending_summary = SummaryRecord(
        run_id=spec.run_id,
        config_hash=spec.config_hash,
        sample_metrics_sha256=hash_file(paths.sample_metrics),
    )
    write_json_atomic(paths.summary, pending_summary)

    tokenizer_revision = spec.model.tokenizer_revision or spec.model.revision
    manifest = RunManifest(
        run_id=spec.run_id,
        config_hash=spec.config_hash,
        config=spec.model_dump(mode="json", exclude_none=False),
        dimebench_version=__version__,
        adapter_name=spec.model.adapter,
        adapter_version=spec.model.adapter_version,
        model_id=spec.model.id,
        model_family=spec.model.family,
        model_revision=spec.model.revision,
        tokenizer_revision=tokenizer_revision,
        dataset_id=spec.dataset.id,
        dataset_manifest_hash=resolved_dataset_hash,
        sample_ids_hash=sample_ids_hash,
        sample_count=len(sample_ids),
        seed=spec.seed,
    )
    write_json_atomic(paths.manifest, manifest)
    return RunArtifacts(paths=paths, manifest=manifest, resumed=False)


def finalize_run(
    paths: RunPaths, status: Literal["completed", "failed"]
) -> RunManifest:
    """Seal a run manifest with current artifact hashes and terminal status."""
    manifest = load_manifest(paths.manifest)
    if status == "completed":
        summary = load_summary(paths.summary)
        if summary.status != "completed":
            raise ArtifactError("cannot complete a run with a pending summary")
    hashes = {
        "environment.json": hash_file(paths.environment),
        "predictions.jsonl": hash_file(paths.predictions),
        "sample_metrics.jsonl": hash_file(paths.sample_metrics),
        "summary.json": hash_file(paths.summary),
    }
    updated = manifest.model_copy(
        update={
            "status": status,
            "artifact_hashes": hashes,
            "updated_at": utc_now(),
        }
    )
    write_json_atomic(paths.manifest, updated)
    return updated


def validate_run_artifacts(paths: RunPaths) -> None:
    """Validate the complete summary-to-prediction-to-config traceability chain."""
    manifest = load_manifest(paths.manifest)
    summary = load_summary(paths.summary)
    if hash_json(manifest.config) != manifest.config_hash:
        raise ArtifactError("manifest config does not match config_hash")
    if summary.config_hash != manifest.config_hash:
        raise ArtifactError("summary config_hash does not match manifest")
    if summary.sample_metrics_sha256 != hash_file(paths.sample_metrics):
        raise ArtifactError("summary does not match sample_metrics.jsonl")

    predictions = PredictionStore(
        paths.predictions,
        manifest.run_id,
        manifest.config_hash,
    )
    metrics = MetricStore(
        paths.sample_metrics,
        manifest.run_id,
        manifest.config_hash,
    )
    for sample_id in metrics.completed_sample_ids:
        metric_record = metrics.get(sample_id)
        try:
            prediction = predictions.get(sample_id)
        except KeyError as exc:
            raise ArtifactError(
                f"metrics for {sample_id!r} have no corresponding prediction"
            ) from exc
        if metric_record.prediction_hash != prediction.record_hash:
            raise ArtifactError(f"prediction hash mismatch for sample {sample_id!r}")

    for metric_id, aggregate in summary.metrics.items():
        if aggregate.sample_ids_hash != hash_ordered_strings(aggregate.sample_ids):
            raise ArtifactError(f"sample_ids_hash mismatch for metric {metric_id!r}")
        if aggregate.sample_count != len(aggregate.sample_ids):
            raise ArtifactError(f"sample_count mismatch for metric {metric_id!r}")
        scheduled_ids = {
            sample_id
            for sample_id in metrics.completed_sample_ids
            if metric_id in metrics.get(sample_id).metrics
            or f"{metric_id}.version" in metrics.get(sample_id).evaluator_metadata
        }
        if aggregate.total_sample_count is not None:
            if aggregate.total_sample_count != len(scheduled_ids):
                raise ArtifactError(
                    f"total_sample_count mismatch for metric {metric_id!r}"
                )
            expected_coverage = (
                aggregate.sample_count / aggregate.total_sample_count
                if aggregate.total_sample_count
                else 0.0
            )
            if aggregate.coverage is None or not isclose(
                aggregate.coverage,
                expected_coverage,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ArtifactError(f"coverage mismatch for metric {metric_id!r}")
            expected_failures = sum(
                metrics.get(sample_id).status == "failure"
                for sample_id in scheduled_ids
            )
            expected_skips = sum(
                metrics.get(sample_id).status == "skipped"
                for sample_id in scheduled_ids
            )
            if aggregate.failure_count != expected_failures:
                raise ArtifactError(f"failure_count mismatch for metric {metric_id!r}")
            if aggregate.skipped_count != expected_skips:
                raise ArtifactError(f"skipped_count mismatch for metric {metric_id!r}")
        values: list[float] = []
        for sample_id in aggregate.sample_ids:
            try:
                record = metrics.get(sample_id)
            except KeyError as exc:
                raise ArtifactError(
                    f"summary metric {metric_id!r} references missing sample "
                    f"{sample_id!r}"
                ) from exc
            if metric_id not in record.metrics:
                raise ArtifactError(
                    f"summary metric {metric_id!r} is absent from sample {sample_id!r}"
                )
            values.append(record.metrics[metric_id])
        expected_value = fmean(values) if values else None
        values_match = (
            aggregate.value is None
            if expected_value is None
            else aggregate.value is not None
            and isclose(
                aggregate.value,
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        if not values_match:
            raise ArtifactError(
                f"aggregate value mismatch for metric {metric_id!r}: "
                f"{aggregate.value} != {expected_value}"
            )

    if manifest.status in {"completed", "failed"}:
        expected_hashes = {
            "environment.json": hash_file(paths.environment),
            "predictions.jsonl": hash_file(paths.predictions),
            "sample_metrics.jsonl": hash_file(paths.sample_metrics),
            "summary.json": hash_file(paths.summary),
        }
        if manifest.artifact_hashes != expected_hashes:
            raise ArtifactError("terminal manifest artifact hashes do not match files")
