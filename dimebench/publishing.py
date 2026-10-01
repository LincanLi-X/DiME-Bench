"""Deterministic v1 release bundles and static leaderboard publication."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field

from dimebench.artifacts.hashing import hash_file, write_json_atomic
from dimebench.reporting import (
    ResultSubmission,
    build_leaderboard,
    create_result_submission,
    render_leaderboard_html,
    render_leaderboard_markdown,
)
from dimebench.reporting._io import write_text_atomic
from dimebench.schemas.base import StrictSchema
from dimebench.summarizers import BenchmarkSummary
from dimebench.version import __version__

RELEASE_VERSION = "v1.0.0"
BENCHMARK_VERSION = "1.0.0"


class PublishingError(ValueError):
    """Raised when immutable release or leaderboard inputs are invalid."""


class ReleaseManifest(StrictSchema):
    """Content-addressed inventory for one benchmark release."""

    schema_version: Literal["1.0"] = "1.0"
    release_version: Literal["v1.0.0"] = "v1.0.0"
    benchmark_id: Literal["dime_bench_v1"] = "dime_bench_v1"
    benchmark_version: Literal["1.0.0"] = "1.0.0"
    package_version: Literal["1.0.0"] = "1.0.0"
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, str]


class ReleaseBuildReport(StrictSchema):
    """Paths and counts returned after assembling a release."""

    status: Literal["passed"] = "passed"
    release_root: str
    release_manifest: str
    result_submission: str
    file_count: int = Field(ge=1)


class LeaderboardSiteReport(StrictSchema):
    """Static-site build result with release-summary identity evidence."""

    status: Literal["passed"] = "passed"
    output_dir: str
    submission_id: str
    entry_count: int = Field(ge=0)
    release_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    site_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_identical: Literal[True] = True
    files: dict[str, str]


_PAPER_FILES = (
    "ar-dllm-comparisons.csv",
    "ar-dllm-comparisons.json",
    "benchmark-scores.csv",
    "dataset-scores.csv",
    "figure-input.csv",
    "figure-input.json",
    "paper-tables.csv",
    "paper-tables.json",
    "paper-tables.md",
    "paper-tables.tex",
    "reproduction-report.json",
    "traceability.json",
    "track-scores.csv",
)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_files(
    project_root: Path,
    release_root: Path,
    pattern: str,
) -> None:
    for source in sorted(project_root.glob(pattern)):
        if source.is_file():
            _copy(source, release_root / source.relative_to(project_root))


def _file_inventory(root: Path, *, exclude: frozenset[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hash_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }


def _private_path_leaks(root: Path) -> tuple[str, ...]:
    """Return release files containing common private absolute-path prefixes."""
    prefixes = ("/Users/", "/home/", "/blue/")
    leaks = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(prefix in text for prefix in prefixes):
            leaks.append(str(path.relative_to(root)))
    return tuple(leaks)


def build_release(
    summary_path: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    submission_id: str = "dime-bench-v1-reference",
) -> ReleaseBuildReport:
    """Assemble the immutable v1 protocol, configs, results, and metadata."""
    root = Path(project_root).resolve()
    source_summary = Path(summary_path).resolve()
    destination = Path(output_dir).resolve()
    if __version__ != BENCHMARK_VERSION:
        raise PublishingError(
            f"package version must be {BENCHMARK_VERSION}, got {__version__}"
        )
    if destination.exists():
        raise PublishingError(f"release output already exists: {destination}")
    try:
        summary = BenchmarkSummary.model_validate_json(
            source_summary.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise PublishingError(f"cannot read benchmark summary: {exc}") from exc
    # Public releases must not expose a submitter's local checkout or account path.
    summary = summary.model_copy(update={"run_root": "reference-runs"})
    destination.mkdir(parents=True)

    release_summary = destination / "benchmark-summary.json"
    write_json_atomic(release_summary, summary)
    submission = create_result_submission(summary, submission_id=submission_id)
    submission_path = destination / "result-submission.json"
    write_json_atomic(submission_path, submission)

    required_project_files = (
        "specification/benchmark-v1.json",
        "specification/result-submission-v1.schema.json",
        "docs/reproduction/paper.md",
        "containers/Dockerfile",
    )
    for relative in required_project_files:
        source = root / relative
        if not source.is_file():
            raise PublishingError(f"required release input is missing: {relative}")
        _copy(source, destination / relative)

    for pattern in (
        "configs/**/*.yaml",
        "data/registry.yaml",
        "data/manifests/*.yaml",
        "data/cards/*.md",
        "data/samples/*.jsonl",
        "docs/specification/*.md",
    ):
        _copy_tree_files(root, destination, pattern)

    paper_root = source_summary.parent
    for filename in _PAPER_FILES:
        source = paper_root / filename
        if not source.is_file():
            raise PublishingError(f"paper reproduction artifact is missing: {source}")
        paper_destination = destination / "paper" / filename
        if filename == "reproduction-report.json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["run_root"] = "reference-runs"
            payload["output_dir"] = "paper"
            write_json_atomic(paper_destination, payload)
        else:
            _copy(source, paper_destination)

    write_text_atomic(
        destination / "README.md",
        """# DiME-Bench v1.0.0 release

This immutable bundle contains the frozen benchmark protocol, model/task/data
configuration, the versioned result-submission schema, reference evaluation
summary, paper-reproduction artifacts, and the container build recipe.

Validate `release-manifest.json` before consuming any score. The public
leaderboard accepts `result-submission.json`; it does not ingest ad-hoc metric
files or unversioned summaries.
""",
    )
    manifest_path = destination / "release-manifest.json"
    inventory = _file_inventory(
        destination,
        exclude=frozenset({"release-manifest.json"}),
    )
    manifest = ReleaseManifest(
        source_fingerprint=summary.source_fingerprint,
        summary_file_sha256=hash_file(release_summary),
        files=inventory,
    )
    write_json_atomic(manifest_path, manifest)
    validate_release(destination)
    return ReleaseBuildReport(
        release_root=str(destination),
        release_manifest=str(manifest_path),
        result_submission=str(submission_path),
        file_count=len(inventory) + 1,
    )


def validate_release(release_root: str | Path) -> ReleaseManifest:
    """Validate every release hash and summary/submission identity."""
    root = Path(release_root).resolve()
    manifest_path = root / "release-manifest.json"
    try:
        manifest = ReleaseManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise PublishingError(f"cannot read release manifest: {exc}") from exc
    actual = _file_inventory(root, exclude=frozenset({"release-manifest.json"}))
    if actual != manifest.files:
        missing = sorted(set(manifest.files) - set(actual))
        unexpected = sorted(set(actual) - set(manifest.files))
        changed = sorted(
            path
            for path in set(actual) & set(manifest.files)
            if actual[path] != manifest.files[path]
        )
        raise PublishingError(
            "release inventory mismatch; "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    summary_path = root / "benchmark-summary.json"
    if hash_file(summary_path) != manifest.summary_file_sha256:
        raise PublishingError("release benchmark summary hash does not match")
    summary = BenchmarkSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    submission = ResultSubmission.model_validate_json(
        (root / "result-submission.json").read_text(encoding="utf-8")
    )
    if submission.summary != summary:
        raise PublishingError("release summary and embedded submission differ")
    if summary.source_fingerprint != manifest.source_fingerprint:
        raise PublishingError("release source fingerprint does not match")
    leaks = _private_path_leaks(root)
    if leaks:
        raise PublishingError(
            "release contains private absolute paths: " + ", ".join(leaks)
        )
    return manifest


_STYLE = """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { max-width: 1180px; margin: 2rem auto; padding: 0 1rem; }
table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
th, td { border-bottom: 1px solid #8886; padding: .65rem; text-align: right; }
th[scope=row], th:nth-child(2) { text-align: left; }
.meta { color: #666; font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
"""


def build_leaderboard_site(
    submission_path: str | Path,
    output_dir: str | Path,
) -> LeaderboardSiteReport:
    """Build a static site from one validated, versioned result submission."""
    source = Path(submission_path).resolve()
    output = Path(output_dir).resolve()
    submission = ResultSubmission.model_validate_json(
        source.read_text(encoding="utf-8")
    )
    leaderboard = build_leaderboard(
        submission.summary,
        submission_id=submission.submission_id,
    )
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "benchmark-summary.json"
    submission_output = output / "result-submission.json"
    leaderboard_json = output / "leaderboard.json"
    leaderboard_markdown = output / "leaderboard.md"
    write_json_atomic(summary_path, submission.summary)
    write_json_atomic(submission_output, submission)
    write_json_atomic(leaderboard_json, leaderboard)
    write_text_atomic(leaderboard_markdown, render_leaderboard_markdown(leaderboard))
    write_text_atomic(output / "style.css", _STYLE)
    table = render_leaderboard_html(leaderboard)
    write_text_atomic(
        output / "index.html",
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>DiME-Bench v1.0.0 Leaderboard</title>"
        '<link rel="stylesheet" href="style.css"></head><body>'
        "<h1>DiME-Bench v1.0.0 Leaderboard</h1>"
        "<p>Scores are percentages. Published entries are derived only from a "
        "versioned, hash-checked result submission.</p>"
        f'<p class="meta">Submission: {submission.submission_id}<br>'
        f"Source fingerprint: {submission.summary.source_fingerprint}</p>"
        f"{table}</body></html>",
    )
    release_summary = source.parent / "benchmark-summary.json"
    if not release_summary.is_file():
        raise PublishingError(
            "the versioned submission must be adjacent to benchmark-summary.json"
        )
    release_hash = hash_file(release_summary)
    site_hash = hash_file(summary_path)
    if release_hash != site_hash:
        raise PublishingError("leaderboard and release summaries are not identical")
    inventory = _file_inventory(output, exclude=frozenset({"site-manifest.json"}))
    write_json_atomic(
        output / "site-manifest.json",
        {
            "schema_version": "1.0",
            "submission_id": submission.submission_id,
            "release_summary_sha256": release_hash,
            "files": inventory,
        },
    )
    return LeaderboardSiteReport(
        output_dir=str(output),
        submission_id=submission.submission_id,
        entry_count=len(leaderboard.entries),
        release_summary_sha256=release_hash,
        site_summary_sha256=site_hash,
        files=_file_inventory(output, exclude=frozenset()),
    )


__all__ = [
    "BENCHMARK_VERSION",
    "RELEASE_VERSION",
    "LeaderboardSiteReport",
    "PublishingError",
    "ReleaseBuildReport",
    "ReleaseManifest",
    "build_leaderboard_site",
    "build_release",
    "validate_release",
]
