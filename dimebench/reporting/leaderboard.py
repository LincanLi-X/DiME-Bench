"""Read-only leaderboard views over traceable benchmark summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from dimebench.schemas.base import StrictSchema
from dimebench.schemas.model import ModelFamily
from dimebench.summarizers import BenchmarkSummary, build_benchmark_summary

from .submission import ResultSubmission


class LeaderboardEntry(StrictSchema):
    """One ranked model with its source run identities."""

    rank: int = Field(gt=0)
    model_id: str
    model_family: ModelFamily
    benchmark_score: float | None
    track_scores: dict[str, float | None]
    source_run_ids: tuple[str, ...]


class Leaderboard(StrictSchema):
    """Versioned leaderboard derived from a benchmark summary."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["dime_bench_v1"] = "dime_bench_v1"
    benchmark_version: Literal["1.0.0"] = "1.0.0"
    submission_id: str | None = None
    source_fingerprint: str
    entries: tuple[LeaderboardEntry, ...]


def build_leaderboard(
    summary: BenchmarkSummary,
    *,
    submission_id: str | None = None,
) -> Leaderboard:
    """Rank available benchmark scores, retaining unscored models last."""
    ordered = sorted(
        summary.models,
        key=lambda item: (
            item.benchmark_score is None,
            -(item.benchmark_score or 0.0),
            item.model_id,
            item.model_family,
        ),
    )
    return Leaderboard(
        submission_id=submission_id,
        source_fingerprint=summary.source_fingerprint,
        entries=tuple(
            LeaderboardEntry(
                rank=index,
                model_id=item.model_id,
                model_family=item.model_family,
                benchmark_score=item.benchmark_score,
                track_scores=item.track_scores,
                source_run_ids=item.source_run_ids,
            )
            for index, item in enumerate(ordered, start=1)
        ),
    )


def leaderboard_from_runs(run_root: str | Path) -> Leaderboard:
    """Build a leaderboard directly from sealed run directories."""
    return build_leaderboard(build_benchmark_summary(run_root))


def load_leaderboard_summary(path: str | Path) -> Leaderboard:
    """Build a leaderboard only from a versioned, hash-checked submission."""
    submission = ResultSubmission.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    return build_leaderboard(
        submission.summary,
        submission_id=submission.submission_id,
    )


def render_leaderboard_markdown(leaderboard: Leaderboard) -> str:
    """Render a compact human-readable leaderboard."""
    tracks = sorted(
        {track for entry in leaderboard.entries for track in entry.track_scores}
    )
    lines = [
        "| Rank | Model | Family | Overall | " + " | ".join(tracks) + " |",
        "| ---: | --- | --- | ---: | " + " | ".join("---:" for _ in tracks) + " |",
    ]
    for entry in leaderboard.entries:
        overall = (
            "—"
            if entry.benchmark_score is None
            else f"{entry.benchmark_score * 100:.2f}"
        )
        track_values = []
        for track in tracks:
            score = entry.track_scores.get(track)
            track_values.append("—" if score is None else f"{score * 100:.2f}")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(entry.rank),
                    entry.model_id,
                    entry.model_family,
                    overall,
                    *track_values,
                )
            )
            + " |"
        )
    return "\n".join(lines)


def render_leaderboard_html(leaderboard: Leaderboard) -> str:
    """Render an accessible, dependency-free static leaderboard table."""
    from html import escape

    tracks = sorted(
        {track for entry in leaderboard.entries for track in entry.track_scores}
    )
    headers = "".join(f'<th scope="col">{escape(track)}</th>' for track in tracks)
    rows = []
    for entry in leaderboard.entries:
        overall = (
            "—"
            if entry.benchmark_score is None
            else f"{entry.benchmark_score * 100:.2f}"
        )
        track_values = []
        for track in tracks:
            score = entry.track_scores.get(track)
            track_values.append("—" if score is None else f"{score * 100:.2f}")
        track_cells = "".join(f"<td>{value}</td>" for value in track_values)
        rows.append(
            "<tr>"
            f"<td>{entry.rank}</td>"
            f'<th scope="row">{escape(entry.model_id)}</th>'
            f"<td>{escape(entry.model_family)}</td>"
            f"<td>{overall}</td>"
            f"{track_cells}</tr>"
        )
    return (
        '<table><thead><tr><th scope="col">Rank</th>'
        '<th scope="col">Model</th><th scope="col">Family</th>'
        f'<th scope="col">Overall</th>{headers}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


__all__ = [
    "Leaderboard",
    "LeaderboardEntry",
    "build_leaderboard",
    "leaderboard_from_runs",
    "load_leaderboard_summary",
    "render_leaderboard_html",
    "render_leaderboard_markdown",
]
