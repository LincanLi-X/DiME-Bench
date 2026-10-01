"""Traceable dataset, track, benchmark, and AR/dLLM summaries."""

from dimebench.summarizers.benchmark import (
    BenchmarkSummary,
    ModelBenchmarkResult,
    build_benchmark_summary,
    write_benchmark_summary,
)
from dimebench.summarizers.comparison import (
    ComparisonRecord,
    ComparisonSummary,
    build_comparisons,
)
from dimebench.summarizers.track import (
    DatasetResult,
    MetricEvidence,
    RunEvaluation,
    SummaryError,
    TrackResult,
    discover_run_dirs,
    evaluate_run,
    load_dataset_result,
    summarize_tracks,
)

__all__ = [
    "BenchmarkSummary",
    "ComparisonRecord",
    "ComparisonSummary",
    "DatasetResult",
    "MetricEvidence",
    "ModelBenchmarkResult",
    "RunEvaluation",
    "SummaryError",
    "TrackResult",
    "build_benchmark_summary",
    "build_comparisons",
    "discover_run_dirs",
    "evaluate_run",
    "load_dataset_result",
    "summarize_tracks",
    "write_benchmark_summary",
]
