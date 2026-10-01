"""End-to-end, deterministic manuscript result reproduction."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from dimebench.artifacts.hashing import hash_file, write_json_atomic
from dimebench.config import load_yaml
from dimebench.reporting._io import write_text_atomic
from dimebench.reporting.figures import build_figure_rows, write_figure_inputs
from dimebench.reporting.tables import (
    PaperTable,
    build_paper_tables,
    write_paper_tables,
)
from dimebench.schemas.base import StrictSchema
from dimebench.summarizers import (
    BenchmarkSummary,
    ComparisonSummary,
    build_benchmark_summary,
    build_comparisons,
    write_benchmark_summary,
)


class PaperReproductionConfig(StrictSchema):
    """Paths and display policy for one paper reproduction."""

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str = Field(min_length=1)
    run_root: Path
    output_dir: Path
    specification: Path
    precision: int = Field(default=4, ge=0, le=12)
    score_scale: float = Field(default=100.0, gt=0.0)
    require_complete_tables: bool = True


class ReproductionReport(StrictSchema):
    """Machine-checkable outcome and hashes of generated deliverables."""

    schema_version: Literal["1.0"] = "1.0"
    step: Literal[14] = 14
    status: Literal["passed", "failed"]
    experiment_id: str
    run_root: str
    output_dir: str
    source_fingerprint: str
    run_count: int = Field(ge=0)
    model_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    missing_cell_count: int = Field(ge=0)
    missing_cells: tuple[str, ...]
    output_hashes: dict[str, str]
    errors: tuple[str, ...] = ()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_reproduction_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> PaperReproductionConfig:
    """Load a tracked config and resolve paths relative to the project root."""
    config_path = Path(path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd().resolve()
    )
    config = PaperReproductionConfig.model_validate(load_yaml(config_path))
    specification = _resolve(root, config.specification)
    if not specification.is_file() and not config.specification.is_absolute():
        packaged = Path(sys.prefix) / "share" / "dimebench" / config.specification
        if packaged.is_file():
            specification = packaged.resolve()
    return config.model_copy(
        update={
            "run_root": _resolve(root, config.run_root),
            "output_dir": _resolve(root, config.output_dir),
            "specification": specification,
        }
    )


def _write_summary_csvs(root: Path, summary: BenchmarkSummary) -> tuple[Path, ...]:
    outputs: list[Path] = []
    definitions = (
        (
            "dataset-scores.csv",
            (
                "run_id",
                "model_id",
                "model_family",
                "track",
                "dataset_id",
                "task_id",
                "metric_id",
                "value",
                "sample_count",
                "sample_ids_hash",
            ),
            (
                (
                    item.run_id,
                    item.model_id,
                    item.model_family,
                    item.track,
                    item.dataset_id,
                    item.task_id,
                    item.primary_metric,
                    item.primary_value,
                    item.sample_count,
                    item.metrics[item.primary_metric].sample_ids_hash,
                )
                for item in summary.datasets
            ),
        ),
        (
            "track-scores.csv",
            (
                "model_id",
                "model_family",
                "track",
                "macro_score",
                "dataset_count",
                "source_run_ids",
            ),
            (
                (
                    item.model_id,
                    item.model_family,
                    item.track,
                    item.macro_score,
                    item.dataset_count,
                    ";".join(item.source_run_ids),
                )
                for item in summary.tracks
            ),
        ),
        (
            "benchmark-scores.csv",
            ("model_id", "model_family", "benchmark_score", "source_run_ids"),
            (
                (
                    item.model_id,
                    item.model_family,
                    item.benchmark_score,
                    ";".join(item.source_run_ids),
                )
                for item in summary.models
            ),
        ),
    )
    for filename, header, rows in definitions:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        destination = root / filename
        write_text_atomic(destination, stream.getvalue())
        outputs.append(destination)
    return tuple(outputs)


def _write_comparisons(root: Path, comparisons: ComparisonSummary) -> tuple[Path, Path]:
    json_path = root / "ar-dllm-comparisons.json"
    csv_path = root / "ar-dllm-comparisons.csv"
    write_json_atomic(json_path, comparisons)
    stream = io.StringIO(newline="")
    header = (
        "track",
        "dataset_id",
        "task_id",
        "metric_id",
        "ar_model_id",
        "dllm_model_id",
        "ar_value",
        "dllm_value",
        "raw_delta",
        "oriented_delta",
        "higher_is_better",
        "ar_run_id",
        "dllm_run_id",
    )
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for item in comparisons.comparisons:
        payload = item.model_dump(mode="json")
        writer.writerow(tuple(payload[key] for key in header))
    write_text_atomic(csv_path, stream.getvalue())
    return json_path, csv_path


def _write_traceability(
    root: Path, summary: BenchmarkSummary, tables: tuple[PaperTable, ...]
) -> Path:
    destination = root / "traceability.json"
    write_json_atomic(
        destination,
        {
            "source_fingerprint": summary.source_fingerprint,
            "runs": {
                item.run_id: {
                    "config_hash": item.config_hash,
                    "source_path": item.source_path,
                    "artifact_hashes": item.artifact_hashes,
                    "metrics": item.metrics,
                }
                for item in summary.datasets
            },
            "paper_cells": {
                table.table_id: {row.row_id: row.cells for row in table.rows}
                for table in tables
            },
        },
    )
    return destination


def _missing_cells(tables: tuple[PaperTable, ...]) -> tuple[str, ...]:
    return tuple(
        f"{table.table_id}/{row.row_id}/{column}"
        for table in tables
        for row in table.rows
        for column in table.columns
        if row.cells[column].status == "missing"
    )


def reproduce_paper(
    config: PaperReproductionConfig,
    *,
    run_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    allow_incomplete: bool = False,
) -> ReproductionReport:
    """Generate all manuscript outputs from sealed run artifacts."""
    resolved_run_root = (
        Path(run_root).resolve() if run_root else config.run_root.resolve()
    )
    root = Path(output_dir).resolve() if output_dir else config.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = build_benchmark_summary(resolved_run_root)
    comparisons = build_comparisons(summary)
    tables = build_paper_tables(
        summary,
        config.specification,
        score_scale=config.score_scale,
    )
    outputs: list[Path] = []
    summary_path = root / "benchmark-summary.json"
    write_benchmark_summary(summary_path, summary)
    outputs.append(summary_path)
    outputs.extend(_write_summary_csvs(root, summary))
    outputs.extend(_write_comparisons(root, comparisons))
    outputs.extend(write_paper_tables(root, tables, precision=config.precision))
    outputs.extend(write_figure_inputs(root, build_figure_rows(summary, comparisons)))
    outputs.append(_write_traceability(root, summary, tables))
    missing = _missing_cells(tables)
    errors = []
    if missing and config.require_complete_tables and not allow_incomplete:
        errors.append(f"{len(missing)} mapped paper cells have no source result")
    hashes = {path.name: hash_file(path) for path in sorted(outputs)}
    report = ReproductionReport(
        status="failed" if errors else "passed",
        experiment_id=config.experiment_id,
        run_root=str(resolved_run_root),
        output_dir=str(root),
        source_fingerprint=summary.source_fingerprint,
        run_count=len(summary.datasets),
        model_count=len(summary.models),
        table_count=len(tables),
        missing_cell_count=len(missing),
        missing_cells=missing,
        output_hashes=hashes,
        errors=tuple(errors),
    )
    write_json_atomic(root / "reproduction-report.json", report)
    return report
