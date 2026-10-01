"""Build paper tables from benchmark summaries and frozen result mappings."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from statistics import fmean
from typing import Literal

from dimebench.artifacts.hashing import write_json_atomic
from dimebench.reporting._io import write_text_atomic
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.model import ModelFamily
from dimebench.summarizers.benchmark import BenchmarkSummary
from dimebench.summarizers.track import DatasetResult, MetricEvidence


class PaperColumn(StrictSchema):
    """One frozen manuscript column mapping."""

    table: str
    column: str
    track: str
    task: str
    metric: str


class PaperCell(StrictSchema):
    """A displayed score and its complete source trail."""

    status: Literal["present", "missing"]
    value: float | None
    scaled_value: float | None
    source_run_ids: tuple[str, ...] = ()
    source_sample_ids_hashes: tuple[str, ...] = ()
    source_summary_hashes: tuple[str, ...] = ()


class PaperRow(StrictSchema):
    """One model row, or model/dataset row for editing."""

    row_id: str
    model_id: str
    model_family: ModelFamily
    dataset_id: str | None = None
    cells: dict[str, PaperCell]


class PaperTable(StrictSchema):
    """Machine-readable paper table."""

    schema_version: Literal["1.0"] = "1.0"
    table_id: str
    columns: tuple[str, ...]
    rows: tuple[PaperRow, ...]


def load_paper_columns(specification: str | Path) -> tuple[PaperColumn, ...]:
    """Load the frozen paper-result mapping from the benchmark specification."""
    try:
        payload = json.loads(Path(specification).read_text(encoding="utf-8"))
        raw_columns = payload["paper_result_columns"]
        return tuple(PaperColumn.model_validate(item) for item in raw_columns)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid paper result mapping {specification}: {exc}"
        ) from exc


def _evidence_cell(evidence: list[MetricEvidence], score_scale: float) -> PaperCell:
    values = [item.value for item in evidence if item.value is not None]
    if not values:
        return PaperCell(status="missing", value=None, scaled_value=None)
    value = fmean(values)
    return PaperCell(
        status="present",
        value=value,
        scaled_value=value * score_scale,
        source_run_ids=tuple(sorted({item.source_run_id for item in evidence})),
        source_sample_ids_hashes=tuple(
            sorted({item.sample_ids_hash for item in evidence})
        ),
        source_summary_hashes=tuple(sorted({item.source_sha256 for item in evidence})),
    )


def _cell_for(
    datasets: list[DatasetResult],
    column: PaperColumn,
    score_scale: float,
) -> PaperCell:
    candidates = [item for item in datasets if item.track == column.track]
    evidence: list[MetricEvidence] = []
    if column.task == "__dataset_row__":
        for item in candidates:
            if column.metric in item.metrics:
                evidence.append(item.metrics[column.metric])
    elif column.task == "__track__":
        for item in candidates:
            if column.metric in item.metrics:
                evidence.append(item.metrics[column.metric])
        if not evidence and column.metric in {"chain_average", "planning_average"}:
            expected = "chain" if column.metric == "chain_average" else "planning"
            for item in candidates:
                if (
                    item.reasoning_type == expected
                    and item.primary_metric in item.metrics
                ):
                    evidence.append(item.metrics[item.primary_metric])
    else:
        for item in candidates:
            if item.task_id == column.task and column.metric in item.metrics:
                evidence.append(item.metrics[column.metric])
    return _evidence_cell(evidence, score_scale)


def build_paper_tables(
    summary: BenchmarkSummary,
    specification: str | Path,
    *,
    score_scale: float = 100.0,
) -> tuple[PaperTable, ...]:
    """Generate every mapped paper table directly from validated summaries."""
    mappings = load_paper_columns(specification)
    by_table: dict[str, list[PaperColumn]] = {}
    for column in mappings:
        by_table.setdefault(column.table, []).append(column)
    tables = []
    for table_id, columns in by_table.items():
        relevant_tracks = {column.track for column in columns}
        dataset_rows = any(column.task == "__dataset_row__" for column in columns)
        members = [item for item in summary.datasets if item.track in relevant_tracks]
        grouped: dict[tuple[str, ModelFamily, str | None], list[DatasetResult]] = {}
        for item in members:
            key = (
                item.model_id,
                item.model_family,
                item.dataset_id if dataset_rows else None,
            )
            grouped.setdefault(key, []).append(item)
        rows = []
        for (model_id, family, dataset_id), datasets in sorted(grouped.items()):
            row_id = f"{family}:{model_id}"
            if dataset_id is not None:
                row_id += f":{dataset_id}"
            rows.append(
                PaperRow(
                    row_id=row_id,
                    model_id=model_id,
                    model_family=family,
                    dataset_id=dataset_id,
                    cells={
                        column.column: _cell_for(datasets, column, score_scale)
                        for column in columns
                    },
                )
            )
        tables.append(
            PaperTable(
                table_id=table_id,
                columns=tuple(column.column for column in columns),
                rows=tuple(rows),
            )
        )
    return tuple(tables)


def tables_to_csv(tables: tuple[PaperTable, ...], precision: int = 4) -> str:
    """Render all paper cells as a tidy CSV."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "table",
            "row_id",
            "model_id",
            "model_family",
            "dataset_id",
            "column",
            "value",
            "scaled_value",
            "status",
            "source_run_ids",
        )
    )
    for table in tables:
        for row in table.rows:
            for column in table.columns:
                cell = row.cells[column]
                writer.writerow(
                    (
                        table.table_id,
                        row.row_id,
                        row.model_id,
                        row.model_family,
                        row.dataset_id or "",
                        column,
                        "" if cell.value is None else f"{cell.value:.{precision}f}",
                        ""
                        if cell.scaled_value is None
                        else f"{cell.scaled_value:.{precision}f}",
                        cell.status,
                        ";".join(cell.source_run_ids),
                    )
                )
    return stream.getvalue()


def tables_to_markdown(tables: tuple[PaperTable, ...], precision: int = 2) -> str:
    """Render human-readable manuscript table previews."""
    sections = []
    for table in tables:
        header = ["Model", "Family", *table.columns]
        lines = [
            f"## {table.table_id}",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---", "---", *["---:" for _ in table.columns]]) + " |",
        ]
        for row in table.rows:
            label = row.model_id
            if row.dataset_id is not None:
                label += f" / {row.dataset_id}"
            values = [
                "—"
                if row.cells[column].scaled_value is None
                else f"{row.cells[column].scaled_value:.{precision}f}"
                for column in table.columns
            ]
            lines.append("| " + " | ".join((label, row.model_family, *values)) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def write_paper_tables(
    output_dir: str | Path,
    tables: tuple[PaperTable, ...],
    *,
    precision: int = 4,
) -> tuple[Path, ...]:
    """Write JSON, tidy CSV, Markdown, and LaTeX paper table artifacts."""
    from dimebench.reporting.latex import render_latex_table

    root = Path(output_dir)
    json_path = root / "paper-tables.json"
    csv_path = root / "paper-tables.csv"
    markdown_path = root / "paper-tables.md"
    latex_path = root / "paper-tables.tex"
    write_json_atomic(json_path, {table.table_id: table for table in tables})
    write_text_atomic(csv_path, tables_to_csv(tables, precision))
    write_text_atomic(markdown_path, tables_to_markdown(tables, min(precision, 4)))
    write_text_atomic(
        latex_path,
        "\n\n".join(render_latex_table(table, min(precision, 4)) for table in tables),
    )
    return json_path, csv_path, markdown_path, latex_path
