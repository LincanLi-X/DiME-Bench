"""Long-form, plotting-library-neutral figure inputs."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from dimebench.artifacts.hashing import write_json_atomic
from dimebench.reporting._io import write_text_atomic
from dimebench.summarizers.benchmark import BenchmarkSummary
from dimebench.summarizers.comparison import ComparisonSummary


def build_figure_rows(
    summary: BenchmarkSummary, comparisons: ComparisonSummary
) -> tuple[dict[str, Any], ...]:
    """Produce tidy rows for dataset, track, benchmark, and comparison plots."""
    rows: list[dict[str, Any]] = []
    for dataset in summary.datasets:
        rows.append(
            {
                "kind": "dataset_score",
                "model_id": dataset.model_id,
                "model_family": dataset.model_family,
                "track": dataset.track,
                "dataset_id": dataset.dataset_id,
                "metric_id": dataset.primary_metric,
                "value": dataset.primary_value,
                "source_run_ids": [dataset.run_id],
            }
        )
    for track in summary.tracks:
        rows.append(
            {
                "kind": "track_score",
                "model_id": track.model_id,
                "model_family": track.model_family,
                "track": track.track,
                "dataset_id": None,
                "metric_id": "macro_score",
                "value": track.macro_score,
                "source_run_ids": list(track.source_run_ids),
            }
        )
    for model in summary.models:
        rows.append(
            {
                "kind": "benchmark_score",
                "model_id": model.model_id,
                "model_family": model.model_family,
                "track": None,
                "dataset_id": None,
                "metric_id": "benchmark_score",
                "value": model.benchmark_score,
                "source_run_ids": list(model.source_run_ids),
            }
        )
    for comparison in comparisons.comparisons:
        rows.append(
            {
                "kind": "ar_dllm_delta",
                "model_id": (f"{comparison.dllm_model_id} vs {comparison.ar_model_id}"),
                "model_family": "comparison",
                "track": comparison.track,
                "dataset_id": comparison.dataset_id,
                "metric_id": comparison.metric_id,
                "value": comparison.oriented_delta,
                "source_run_ids": [
                    comparison.ar_run_id,
                    comparison.dllm_run_id,
                ],
            }
        )
    return tuple(rows)


def write_figure_inputs(
    output_dir: str | Path, rows: tuple[dict[str, Any], ...]
) -> tuple[Path, Path]:
    """Write identical plotting data in JSON and CSV formats."""
    root = Path(output_dir)
    json_path = root / "figure-input.json"
    csv_path = root / "figure-input.csv"
    write_json_atomic(json_path, rows)
    stream = io.StringIO(newline="")
    fieldnames = (
        "kind",
        "model_id",
        "model_family",
        "track",
        "dataset_id",
        "metric_id",
        "value",
        "source_run_ids",
    )
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serializable = dict(row)
        serializable["source_run_ids"] = ";".join(row["source_run_ids"])
        writer.writerow(serializable)
    write_text_atomic(csv_path, stream.getvalue())
    return json_path, csv_path
