"""Unified batching, caching, retry, resume, and raw-output inference engine."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from dimebench.artifacts import PredictionStore, initialize_run
from dimebench.datasets import (
    DatasetError,
    NormalizedDataset,
    SampleRecord,
)
from dimebench.datasets.records import expected_sample_id, parse_sample_record
from dimebench.inference.batcher import iter_batches
from dimebench.inference.cache import ResponseCache
from dimebench.inference.output_writer import PredictionWriter
from dimebench.inference.request_builder import RequestBuilder
from dimebench.inference.resume import plan_resume
from dimebench.inference.tracing import TraceWriter
from dimebench.models import Capability, ModelAdapter
from dimebench.models.diffusion import DiffusionModelAdapter
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse
from dimebench.schemas.run import RunSpec


class InferenceError(RuntimeError):
    """Raised for invalid inference inputs or incompatible execution routes."""


@dataclass(frozen=True)
class RetryPolicy:
    """Deterministic retry budget for adapter exceptions."""

    max_attempts: int = 3
    backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")


@dataclass(frozen=True)
class InferenceResult:
    """Inference-stage counters and artifact locations."""

    run_id: str
    run_dir: Path
    predictions_path: Path
    trace_path: Path
    total_samples: int
    previously_completed: int
    written_predictions: int
    cache_hits: int
    failed_predictions: int


def _stable_smoke_record(index: int) -> SampleRecord:
    payload = {
        "schema_version": "1.0",
        "sample_id": f"smoke_infilling.test.pending-{index}",
        "dataset_id": "smoke_infilling",
        "split": "test",
        "source_id": f"smoke-{index}",
        "kind": "infilling",
        "prefix": f"A deterministic prefix number {index} contains",
        "middle": f"a hidden middle span {index}",
        "suffix": "before the fixed suffix closes the example.",
        "domain": "smoke",
        "span_length": 5,
        "prefix_length": 6,
        "suffix_length": 7,
        "span_length_bin": "short",
        "boundary_density": "high",
        "metadata": {"fixture": True},
    }
    provisional = parse_sample_record(payload)
    payload["sample_id"] = expected_sample_id(provisional)
    return parse_sample_record(payload)


def load_inference_dataset(spec: RunSpec) -> NormalizedDataset:
    """Load already-normalized data or the built-in smoke fixture."""
    if spec.dataset.source == "builtin":
        if spec.dataset.name_or_path != "smoke/infilling":
            raise DatasetError(
                f"unknown built-in dataset {spec.dataset.name_or_path!r}"
            )
        records = [_stable_smoke_record(index) for index in range(1, 5)]
        dataset = NormalizedDataset(spec.dataset.id, spec.dataset.split, records)
    else:
        source = Path(spec.dataset.name_or_path)
        if not source.is_file():
            raise DatasetError(
                "inference consumes normalized JSONL; run `dimebench prepare` "
                "and set dataset.name_or_path to the prepared split file"
            )
        dataset = NormalizedDataset.read_jsonl(
            source,
            dataset_id=spec.dataset.id,
            split=spec.dataset.split,
        )
    selected = list(dataset.records)
    if spec.dataset.shuffle:
        random.Random(spec.dataset.seed).shuffle(selected)
    if spec.dataset.sample_limit is not None:
        selected = selected[: spec.dataset.sample_limit]
    if not selected:
        raise DatasetError("inference dataset selection contains no samples")
    return NormalizedDataset(spec.dataset.id, spec.dataset.split, selected)


class InferenceEngine:
    """Run one model-task configuration without parsing or aggregation."""

    def __init__(
        self,
        spec: RunSpec,
        adapter: ModelAdapter,
        *,
        prompt_root: str | Path,
        cache_root: str | Path | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.spec = spec
        self.adapter = adapter
        self.prompt_root = Path(prompt_root)
        self.cache = ResponseCache(
            cache_root or Path(spec.output_dir) / ".cache" / "inference"
        )
        self.retry_policy = retry_policy or RetryPolicy()

    def _invoke(self, requests: tuple[ModelRequest, ...]) -> tuple[ModelResponse, ...]:
        modes = {request.mode for request in requests}
        if len(modes) != 1:
            raise InferenceError("one inference batch cannot mix request modes")
        mode = requests[0].mode
        if mode == "score_options":
            return self.adapter.score_options(requests)
        if isinstance(self.adapter, DiffusionModelAdapter) and self.adapter.supports(
            Capability.DENOISE
        ):
            steps = self.spec.decoding.denoising_steps
            schedule = self.spec.decoding.schedule
            mask_policy = self.spec.decoding.mask_policy
            if steps is None or schedule is None or mask_policy is None:
                raise InferenceError("diffusion denoising controls are incomplete")
            return self.adapter.denoise(
                requests,
                steps=steps,
                schedule=schedule,
                mask_policy=mask_policy,
            )
        return self.adapter.generate(requests)

    def _execute_with_retry(
        self,
        requests: tuple[ModelRequest, ...],
        trace: TraceWriter,
    ) -> tuple[tuple[ModelResponse, ...], int]:
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                return self._invoke(requests), attempt
            except Exception as exc:  # adapters may raise backend-specific errors
                if attempt < self.retry_policy.max_attempts:
                    trace.append(
                        "retry",
                        sample_ids=tuple(request.sample_id for request in requests),
                        details={
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    if self.retry_policy.backoff_seconds:
                        time.sleep(self.retry_policy.backoff_seconds)
                    continue
                responses = tuple(
                    ModelResponse(
                        request_id=request.request_id,
                        sample_id=request.sample_id,
                        model_id=self.spec.model.id,
                        status="error",
                        finish_reason="error",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        decoding_metadata={"attempts": attempt},
                    )
                    for request in requests
                )
                return responses, attempt
        raise AssertionError("retry loop exhausted unexpectedly")

    def run(self, dataset: NormalizedDataset) -> InferenceResult:
        """Execute pending samples and durably retain every terminal response."""
        if dataset.dataset_id != self.spec.dataset.id:
            raise InferenceError("dataset ID does not match run configuration")
        if dataset.split != self.spec.dataset.split:
            raise InferenceError("dataset split does not match run configuration")
        sample_ids = tuple(record.sample_id for record in dataset)
        run = initialize_run(
            self.spec,
            sample_ids,
            dataset_hash=dataset.dataset_hash,
        )
        trace_path = run.paths.root / "inference-trace.jsonl"
        trace = TraceWriter(trace_path, self.spec.run_id)
        store = PredictionStore(
            run.paths.predictions,
            self.spec.run_id,
            self.spec.config_hash,
        )
        resume = plan_resume(sample_ids, store)
        trace.append(
            "run_resumed" if run.resumed else "run_started",
            sample_ids=sample_ids,
            details={
                "total_samples": len(sample_ids),
                "previously_completed": len(resume.completed_sample_ids),
            },
        )
        pending = tuple(dataset.records[index] for index in resume.pending_indexes)
        if not pending:
            trace.append("run_finished", details={"written_predictions": 0})
            return InferenceResult(
                run_id=self.spec.run_id,
                run_dir=run.paths.root,
                predictions_path=run.paths.predictions,
                trace_path=trace_path,
                total_samples=len(dataset),
                previously_completed=len(resume.completed_sample_ids),
                written_predictions=0,
                cache_hits=0,
                failed_predictions=0,
            )

        builder = RequestBuilder(self.spec, self.adapter, self.prompt_root)
        writer = PredictionWriter(self.spec, store)
        written = 0
        cache_hits = 0
        failures = 0
        try:
            with self.adapter:
                for batch_index, sample_batch in enumerate(
                    iter_batches(pending, self.spec.batch_size),
                    start=1,
                ):
                    requests = tuple(builder.build(sample) for sample in sample_batch)
                    trace.append(
                        "batch_started",
                        sample_ids=tuple(request.sample_id for request in requests),
                        details={"batch_index": batch_index},
                    )
                    responses: dict[str, tuple[ModelResponse, bool, int]] = {}
                    misses: list[ModelRequest] = []
                    for request in requests:
                        key = self.cache.key_for(
                            request, self.spec.model, self.spec.decoding
                        )
                        cached = self.cache.get(key)
                        if cached is None:
                            misses.append(request)
                        else:
                            responses[request.request_id] = (cached, True, 0)
                            cache_hits += 1
                            trace.append(
                                "cache_hit",
                                sample_ids=(request.sample_id,),
                                details={"cache_key": key},
                            )
                    if misses:
                        generated, attempts = self._execute_with_retry(
                            tuple(misses), trace
                        )
                        for request, response in zip(misses, generated, strict=True):
                            responses[request.request_id] = (response, False, attempts)
                            if response.status == "success":
                                key = self.cache.key_for(
                                    request, self.spec.model, self.spec.decoding
                                )
                                self.cache.put(
                                    key,
                                    request,
                                    self.spec.model,
                                    self.spec.decoding,
                                    response,
                                )
                    for request in requests:
                        response, cache_hit, attempts = responses[request.request_id]
                        record = writer.write(
                            request,
                            response,
                            cache_hit=cache_hit,
                            attempts=attempts,
                        )
                        written += 1
                        failures += record.status == "failure"
                        trace.append(
                            "prediction_written",
                            sample_ids=(request.sample_id,),
                            details={
                                "status": record.status,
                                "cache_hit": cache_hit,
                            },
                        )
        except Exception as exc:
            trace.append(
                "run_failed",
                details={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        trace.append(
            "run_finished",
            details={
                "written_predictions": written,
                "cache_hits": cache_hits,
                "failed_predictions": failures,
            },
        )
        return InferenceResult(
            run_id=self.spec.run_id,
            run_dir=run.paths.root,
            predictions_path=run.paths.predictions,
            trace_path=trace_path,
            total_samples=len(dataset),
            previously_completed=len(resume.completed_sample_ids),
            written_predictions=written,
            cache_hits=cache_hits,
            failed_predictions=failures,
        )


__all__ = [
    "InferenceEngine",
    "InferenceError",
    "InferenceResult",
    "RetryPolicy",
    "load_inference_dataset",
]
