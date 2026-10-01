"""Dataset-to-track aggregation with complete source evidence."""

from __future__ import annotations

from math import isclose
from pathlib import Path
from statistics import fmean
from typing import Literal, cast

from pydantic import Field

from dimebench.artifacts import MetricStore, RunPaths, load_manifest, load_summary
from dimebench.artifacts.hashing import hash_file
from dimebench.artifacts.manifest import validate_run_artifacts
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.model import ModelFamily


class SummaryError(ValueError):
    """Raised when result aggregation cannot be performed safely."""


class MetricEvidence(StrictSchema):
    """One aggregate plus the exact samples and file that produced it."""

    metric_id: str = Field(min_length=1)
    value: float | None
    sample_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    failure_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    sample_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetResult(StrictSchema):
    """Validated result for one model, task, and dataset."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_family: ModelFamily
    dataset_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    track: str = Field(min_length=1)
    reasoning_type: str | None = None
    primary_metric: str = Field(min_length=1)
    primary_value: float | None
    sample_count: int = Field(gt=0)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, MetricEvidence]
    artifact_hashes: dict[str, str]
    source_path: str = Field(min_length=1)


class TrackResult(StrictSchema):
    """Macro-average of dataset primary metrics for one model and track."""

    schema_version: Literal["1.0"] = "1.0"
    track: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_family: ModelFamily
    macro_score: float | None
    dataset_count: int = Field(ge=0)
    source_run_ids: tuple[str, ...]


class RunEvaluation(StrictSchema):
    """Independent recomputation result for a sealed run directory."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["passed", "failed"]
    run_id: str
    run_dir: str
    sample_count: int = Field(ge=0)
    metric_count: int = Field(ge=0)
    stored_summary_sha256: str
    sample_metrics_sha256: str
    recomputation_matches: bool
    errors: tuple[str, ...] = ()


def _task_fields(config: dict[str, object]) -> tuple[str, str, str, str | None]:
    task = config.get("task")
    if not isinstance(task, dict):
        raise SummaryError("manifest config has no task object")
    values = tuple(task.get(key) for key in ("id", "track", "primary_metric"))
    if not all(isinstance(value, str) and value for value in values):
        raise SummaryError("manifest task requires id, track, and primary_metric")
    reasoning_type = task.get("reasoning_type")
    if reasoning_type is not None and not isinstance(reasoning_type, str):
        raise SummaryError("manifest task reasoning_type must be a string or null")
    task_id, track, primary_metric = cast(tuple[str, str, str], values)
    return task_id, track, primary_metric, reasoning_type


def discover_run_dirs(run_root: str | Path) -> tuple[Path, ...]:
    """Find sealed-run candidates recursively and deterministically."""
    root = Path(run_root)
    if not root.is_dir():
        raise SummaryError(f"run root does not exist: {root}")
    directories = tuple(sorted(path.parent for path in root.rglob("run_manifest.json")))
    if not directories:
        raise SummaryError(f"no run_manifest.json files found under {root}")
    return directories


def load_dataset_result(
    run_dir: str | Path,
    *,
    source_root: str | Path | None = None,
) -> DatasetResult:
    """Validate a completed run and expose its summary with provenance."""
    paths = RunPaths(Path(run_dir))
    validate_run_artifacts(paths)
    manifest = load_manifest(paths.manifest)
    summary = load_summary(paths.summary)
    if manifest.status != "completed" or summary.status != "completed":
        raise SummaryError(f"run {manifest.run_id!r} is not completed")
    task_id, track, primary_metric, reasoning_type = _task_fields(
        cast(dict[str, object], manifest.config)
    )
    if primary_metric not in summary.metrics:
        raise SummaryError(
            f"run {manifest.run_id!r} is missing primary metric {primary_metric!r}"
        )
    root = Path(source_root).resolve() if source_root is not None else None
    resolved = paths.root.resolve()
    try:
        source_path = (
            str(resolved.relative_to(root)) if root is not None else str(resolved)
        )
    except ValueError:
        source_path = str(resolved)
    summary_hash = hash_file(paths.summary)
    metrics = {
        metric_id: MetricEvidence(
            metric_id=metric_id,
            value=aggregate.value,
            sample_count=aggregate.sample_count,
            total_sample_count=aggregate.total_sample_count or 0,
            coverage=aggregate.coverage or 0.0,
            failure_count=aggregate.failure_count,
            skipped_count=aggregate.skipped_count,
            sample_ids_hash=aggregate.sample_ids_hash,
            source_run_id=manifest.run_id,
            source_path=f"{source_path}/summary.json",
            source_sha256=summary_hash,
        )
        for metric_id, aggregate in sorted(summary.metrics.items())
    }
    return DatasetResult(
        run_id=manifest.run_id,
        model_id=manifest.model_id,
        model_family=manifest.model_family,
        dataset_id=manifest.dataset_id,
        task_id=task_id,
        track=track,
        reasoning_type=reasoning_type,
        primary_metric=primary_metric,
        primary_value=metrics[primary_metric].value,
        sample_count=manifest.sample_count,
        config_hash=manifest.config_hash,
        metrics=metrics,
        artifact_hashes=dict(sorted(manifest.artifact_hashes.items())),
        source_path=source_path,
    )


def summarize_tracks(datasets: tuple[DatasetResult, ...]) -> tuple[TrackResult, ...]:
    """Macro-average available dataset primary scores within each track."""
    grouped: dict[tuple[str, str, ModelFamily], list[DatasetResult]] = {}
    for dataset in datasets:
        grouped.setdefault(
            (dataset.track, dataset.model_id, dataset.model_family), []
        ).append(dataset)
    results = []
    for (track, model_id, family), members in sorted(grouped.items()):
        values = [
            item.primary_value for item in members if item.primary_value is not None
        ]
        results.append(
            TrackResult(
                track=track,
                model_id=model_id,
                model_family=family,
                macro_score=fmean(values) if values else None,
                dataset_count=len(members),
                source_run_ids=tuple(sorted(item.run_id for item in members)),
            )
        )
    return tuple(results)


def evaluate_run(run_dir: str | Path) -> RunEvaluation:
    """Recompute a run summary independently and compare every aggregate."""
    paths = RunPaths(Path(run_dir))
    errors: list[str] = []
    try:
        validate_run_artifacts(paths)
        manifest = load_manifest(paths.manifest)
        stored = load_summary(paths.summary)
        recomputed = MetricStore(
            paths.sample_metrics, manifest.run_id, manifest.config_hash
        ).summarize()
        if stored.sample_metrics_sha256 != recomputed.sample_metrics_sha256:
            errors.append("sample_metrics_sha256 mismatch")
        if set(stored.metrics) != set(recomputed.metrics):
            errors.append("metric id set mismatch")
        for metric_id in sorted(set(stored.metrics) & set(recomputed.metrics)):
            left = stored.metrics[metric_id]
            right = recomputed.metrics[metric_id]
            if left.model_dump() != right.model_dump():
                value_match = (left.value is None and right.value is None) or (
                    left.value is not None
                    and right.value is not None
                    and isclose(left.value, right.value, rel_tol=1e-12, abs_tol=1e-12)
                )
                if (
                    not value_match
                    or left.model_copy(update={"value": right.value}) != right
                ):
                    errors.append(f"aggregate mismatch: {metric_id}")
        count = len(
            MetricStore(paths.sample_metrics, manifest.run_id, manifest.config_hash)
        )
        return RunEvaluation(
            status="failed" if errors else "passed",
            run_id=manifest.run_id,
            run_dir=str(paths.root.resolve()),
            sample_count=count,
            metric_count=len(stored.metrics),
            stored_summary_sha256=hash_file(paths.summary),
            sample_metrics_sha256=hash_file(paths.sample_metrics),
            recomputation_matches=not errors,
            errors=tuple(errors),
        )
    except Exception as exc:  # validation report must preserve the root cause
        return RunEvaluation(
            status="failed",
            run_id=paths.root.name,
            run_dir=str(paths.root.resolve()),
            sample_count=0,
            metric_count=0,
            stored_summary_sha256=(
                hash_file(paths.summary) if paths.summary.is_file() else ""
            ),
            sample_metrics_sha256=(
                hash_file(paths.sample_metrics)
                if paths.sample_metrics.is_file()
                else ""
            ),
            recomputation_matches=False,
            errors=(f"{type(exc).__name__}: {exc}",),
        )
