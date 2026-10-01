"""Unified inference-to-evaluation workflow used by the public CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from dimebench.artifacts import (
    MetricStore,
    PredictionStore,
    RunPaths,
    SampleMetricRecord,
    finalize_run,
    load_manifest,
    validate_run_artifacts,
)
from dimebench.artifacts.hashing import hash_file, write_json_atomic
from dimebench.datasets import NormalizedDataset, SampleRecord
from dimebench.evaluators import StandardEvaluator, evaluate_sample
from dimebench.evaluators.mechanism import (
    BoundaryConsistencyEvaluator,
    EditSuccessEvaluator,
    OverEditRateEvaluator,
    PreservationScoreEvaluator,
)
from dimebench.evaluators.standard import (
    available_standard_evaluators,
    create_standard_evaluator,
)
from dimebench.inference import (
    InferenceEngine,
    RetryPolicy,
    load_inference_dataset,
)
from dimebench.models import create_adapter
from dimebench.postprocessors import (
    postprocess_predictions,
    write_postprocessed_results,
)
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.run import RunSpec


class WorkflowError(ValueError):
    """Raised when a configuration cannot complete the unified workflow."""


class EvaluationResult(StrictSchema):
    """Artifacts and counters produced by the evaluation stage."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["passed"] = "passed"
    run_id: str
    run_dir: str
    sample_count: int = Field(ge=0)
    successful_predictions: int = Field(ge=0)
    failed_predictions: int = Field(ge=0)
    metrics: dict[str, float | None]
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunResult(StrictSchema):
    """Top-level result returned by ``dimebench run``."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["passed"] = "passed"
    run_id: str
    run_dir: str
    total_samples: int = Field(ge=0)
    written_predictions: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    evaluation: EvaluationResult


class _SmokeBoundaryBackend:
    """Dependency-free boundary backend used only by the built-in mock gate."""

    backend_id = "builtin_mock_boundary"
    revision = "1.0.0"

    def configuration(self) -> dict[str, JsonValue]:
        return {
            "backend_id": self.backend_id,
            "revision": self.revision,
            "purpose": "cpu_only_smoke_validation",
        }

    def score_pair(self, left: str, right: str) -> float:
        return float(bool(left.strip()) and bool(right.strip()))


def _evaluators_for(spec: RunSpec) -> tuple[StandardEvaluator, ...]:
    standard = available_standard_evaluators()
    evaluators: list[StandardEvaluator] = []
    for metric_id in spec.task.metrics:
        if metric_id in standard:
            evaluators.append(create_standard_evaluator(metric_id))
        elif metric_id == "boundary_consistency" and spec.model.adapter == "mock":
            evaluators.append(BoundaryConsistencyEvaluator(_SmokeBoundaryBackend()))
        elif metric_id == "edit_success":
            evaluators.append(EditSuccessEvaluator())
        elif metric_id == "preservation_score":
            evaluators.append(PreservationScoreEvaluator())
        elif metric_id == "over_edit_rate":
            evaluators.append(OverEditRateEvaluator())
        else:
            raise WorkflowError(
                f"metric {metric_id!r} requires a track-specific evaluator "
                "configuration; use the tracked experiment workflow"
            )
    return tuple(evaluators)


def _same_metric_record(left: SampleMetricRecord, right: SampleMetricRecord) -> bool:
    left_dump = left.model_dump(mode="json", exclude={"created_at"})
    right_dump = right.model_dump(mode="json", exclude={"created_at"})
    return left_dump == right_dump


def evaluate_configured_run(
    spec: RunSpec,
    dataset: NormalizedDataset | None = None,
) -> EvaluationResult:
    """Parse, score, seal, and validate one inference-complete run."""
    selected = dataset or load_inference_dataset(spec)
    paths = RunPaths(Path(spec.output_dir) / spec.run_id)
    manifest = load_manifest(paths.manifest)
    if manifest.config_hash != spec.config_hash:
        raise WorkflowError("run manifest does not match the resolved configuration")
    predictions = PredictionStore(paths.predictions, spec.run_id, spec.config_hash)
    ordered = tuple(predictions.get(sample.sample_id) for sample in selected)
    if len(ordered) != len(selected):
        raise WorkflowError("inference has not produced every selected sample")
    sample_map: dict[str, SampleRecord] = {
        sample.sample_id: sample for sample in selected
    }
    processed = postprocess_predictions(ordered, sample_map, spec.task)
    write_postprocessed_results(paths.root / "postprocessed-results.jsonl", processed)
    evaluators = _evaluators_for(spec)
    calculated = tuple(
        evaluate_sample(result, sample_map[result.sample_id], evaluators)
        for result in processed
    )
    store = MetricStore(paths.sample_metrics, spec.run_id, spec.config_hash)
    for record in calculated:
        if record.sample_id in store.completed_sample_ids:
            if not _same_metric_record(store.get(record.sample_id), record):
                raise WorkflowError(
                    f"stored metrics differ for sample {record.sample_id!r}"
                )
        else:
            store.append(record)
    summary = store.write_summary(paths.summary)
    finalize_run(paths, "completed")
    validate_run_artifacts(paths)
    result = EvaluationResult(
        run_id=spec.run_id,
        run_dir=str(paths.root.resolve()),
        sample_count=len(calculated),
        successful_predictions=sum(item.status == "success" for item in ordered),
        failed_predictions=sum(item.status != "success" for item in ordered),
        metrics={
            metric_id: aggregate.value
            for metric_id, aggregate in sorted(summary.metrics.items())
        },
        predictions_sha256=hash_file(paths.predictions),
        sample_metrics_sha256=hash_file(paths.sample_metrics),
        summary_sha256=hash_file(paths.summary),
    )
    write_json_atomic(paths.root / "evaluation-report.json", result)
    return result


def run_configured_workflow(
    spec: RunSpec,
    *,
    prompt_root: str | Path,
    cache_root: str | Path | None = None,
    retry_policy: RetryPolicy | None = None,
) -> RunResult:
    """Execute inference followed by deterministic evaluation and sealing."""
    dataset = load_inference_dataset(spec)
    inference = InferenceEngine(
        spec,
        create_adapter(spec.model),
        prompt_root=prompt_root,
        cache_root=cache_root,
        retry_policy=retry_policy,
    ).run(dataset)
    evaluation = evaluate_configured_run(spec, dataset)
    return RunResult(
        run_id=spec.run_id,
        run_dir=str(inference.run_dir.resolve()),
        total_samples=inference.total_samples,
        written_predictions=inference.written_predictions,
        cache_hits=inference.cache_hits,
        evaluation=evaluation,
    )


__all__ = [
    "EvaluationResult",
    "RunResult",
    "WorkflowError",
    "evaluate_configured_run",
    "run_configured_workflow",
]
