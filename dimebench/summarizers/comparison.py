"""Controlled AR versus dLLM comparisons over matched tasks and samples."""

from __future__ import annotations

from typing import Literal

from dimebench.schemas.base import StrictSchema
from dimebench.summarizers.benchmark import BenchmarkSummary
from dimebench.summarizers.track import DatasetResult

LOWER_IS_BETTER = frozenset({"contradiction_rate", "over_edit_rate"})


class ComparisonRecord(StrictSchema):
    """One matched AR/dLLM metric comparison with source evidence."""

    track: str
    dataset_id: str
    task_id: str
    metric_id: str
    ar_model_id: str
    dllm_model_id: str
    ar_value: float
    dllm_value: float
    raw_delta: float
    oriented_delta: float
    higher_is_better: bool
    ar_run_id: str
    dllm_run_id: str
    ar_sample_ids_hash: str
    dllm_sample_ids_hash: str


class ComparisonSummary(StrictSchema):
    """All available controlled model-family comparisons."""

    schema_version: Literal["1.0"] = "1.0"
    comparisons: tuple[ComparisonRecord, ...]


def _key(item: DatasetResult) -> tuple[str, str, str]:
    return item.track, item.dataset_id, item.task_id


def build_comparisons(summary: BenchmarkSummary) -> ComparisonSummary:
    """Pair AR and diffusion runs sharing track, dataset, task, and metric."""
    ar: dict[tuple[str, str, str], list[DatasetResult]] = {}
    dllm: dict[tuple[str, str, str], list[DatasetResult]] = {}
    for item in summary.datasets:
        target = ar if item.model_family == "autoregressive" else dllm
        target.setdefault(_key(item), []).append(item)
    records = []
    for key in sorted(set(ar) & set(dllm)):
        for left in sorted(ar[key], key=lambda item: item.model_id):
            for right in sorted(dllm[key], key=lambda item: item.model_id):
                for metric_id in sorted(set(left.metrics) & set(right.metrics)):
                    ar_metric = left.metrics[metric_id]
                    dllm_metric = right.metrics[metric_id]
                    if ar_metric.value is None or dllm_metric.value is None:
                        continue
                    raw_delta = dllm_metric.value - ar_metric.value
                    higher_is_better = metric_id not in LOWER_IS_BETTER
                    records.append(
                        ComparisonRecord(
                            track=key[0],
                            dataset_id=key[1],
                            task_id=key[2],
                            metric_id=metric_id,
                            ar_model_id=left.model_id,
                            dllm_model_id=right.model_id,
                            ar_value=ar_metric.value,
                            dllm_value=dllm_metric.value,
                            raw_delta=raw_delta,
                            oriented_delta=raw_delta
                            if higher_is_better
                            else -raw_delta,
                            higher_is_better=higher_is_better,
                            ar_run_id=left.run_id,
                            dllm_run_id=right.run_id,
                            ar_sample_ids_hash=ar_metric.sample_ids_hash,
                            dllm_sample_ids_hash=dllm_metric.sample_ids_hash,
                        )
                    )
    return ComparisonSummary(comparisons=tuple(records))
