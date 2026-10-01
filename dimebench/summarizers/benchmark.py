"""Track-to-benchmark aggregation."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import Field

from dimebench.artifacts.hashing import hash_json, write_json_atomic
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.model import ModelFamily
from dimebench.summarizers.track import (
    DatasetResult,
    SummaryError,
    TrackResult,
    discover_run_dirs,
    load_dataset_result,
    summarize_tracks,
)


class ModelBenchmarkResult(StrictSchema):
    """Macro-average of available track scores for one model."""

    model_id: str = Field(min_length=1)
    model_family: ModelFamily
    benchmark_score: float | None
    track_scores: dict[str, float | None]
    source_run_ids: tuple[str, ...]


class BenchmarkSummary(StrictSchema):
    """Complete deterministic sample -> dataset -> track -> benchmark view."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["dime_bench_v1"] = "dime_bench_v1"
    run_root: str
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    datasets: tuple[DatasetResult, ...]
    tracks: tuple[TrackResult, ...]
    models: tuple[ModelBenchmarkResult, ...]


def _summarize_models(
    tracks: tuple[TrackResult, ...],
) -> tuple[ModelBenchmarkResult, ...]:
    grouped: dict[tuple[str, ModelFamily], list[TrackResult]] = {}
    for track in tracks:
        grouped.setdefault((track.model_id, track.model_family), []).append(track)
    models = []
    for (model_id, family), members in sorted(grouped.items()):
        scores = {
            item.track: item.macro_score
            for item in sorted(members, key=lambda x: x.track)
        }
        values = [value for value in scores.values() if value is not None]
        models.append(
            ModelBenchmarkResult(
                model_id=model_id,
                model_family=family,
                benchmark_score=fmean(values) if values else None,
                track_scores=scores,
                source_run_ids=tuple(
                    sorted(
                        {run_id for item in members for run_id in item.source_run_ids}
                    )
                ),
            )
        )
    return tuple(models)


def build_benchmark_summary(run_root: str | Path) -> BenchmarkSummary:
    """Validate all discovered runs and build deterministic aggregates."""
    root = Path(run_root).resolve()
    datasets = tuple(
        sorted(
            (
                load_dataset_result(path, source_root=root)
                for path in discover_run_dirs(root)
            ),
            key=lambda item: item.run_id,
        )
    )
    run_ids = [item.run_id for item in datasets]
    if len(run_ids) != len(set(run_ids)):
        raise SummaryError("duplicate run_id values were discovered")
    tracks = summarize_tracks(datasets)
    fingerprint_payload = [
        {
            "run_id": item.run_id,
            "config_hash": item.config_hash,
            "artifact_hashes": item.artifact_hashes,
        }
        for item in datasets
    ]
    return BenchmarkSummary(
        run_root=str(root),
        source_fingerprint=hash_json(fingerprint_payload),
        datasets=datasets,
        tracks=tracks,
        models=_summarize_models(tracks),
    )


def write_benchmark_summary(path: str | Path, summary: BenchmarkSummary) -> None:
    """Write the benchmark summary as deterministic JSON."""
    write_json_atomic(path, summary)
