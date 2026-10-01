"""Compact sealed-run fixtures for Step 14 tests."""

from __future__ import annotations

from pathlib import Path

from dimebench.artifacts import (
    MetricStore,
    PredictionRecord,
    PredictionStore,
    SampleMetricRecord,
    finalize_run,
    initialize_run,
)
from dimebench.schemas import DatasetSpec, DecodingSpec, ModelSpec, RunSpec, TaskSpec


def make_sealed_run(
    root: Path,
    *,
    run_id: str,
    family: str,
    value: float,
    model_id: str,
) -> Path:
    """Create a two-sample, fully hashed Track 1 result."""
    spec = RunSpec(
        run_id=run_id,
        output_dir=root,
        model=ModelSpec(
            id=model_id,
            family=family,
            adapter="mock",
            name_or_path=f"fixture/{model_id}",
            revision="fixture",
            dtype="float32",
            device="cpu",
            capabilities=("generate",),
        ),
        dataset=DatasetSpec(
            id="mmlu_pro",
            source="builtin",
            name_or_path="fixture/mmlu_pro",
            revision="fixture",
            split="test",
            manifest_hash="a" * 64,
            sample_limit=2,
        ),
        task=TaskSpec(
            id="mmlu_pro",
            track="track1_general",
            dataset_id="mmlu_pro",
            type="multiple_choice",
            prompt_id="track1.multiple_choice",
            parser_id="multiple_choice",
            metrics=("accuracy",),
            primary_metric="accuracy",
            max_output_tokens=8,
        ),
        decoding=DecodingSpec(max_new_tokens=8),
    )
    sample_ids = ("mmlu_pro.test.000", "mmlu_pro.test.001")
    run = initialize_run(spec, sample_ids, dataset_hash="a" * 64)
    predictions = PredictionStore(run.paths.predictions, run_id, spec.config_hash)
    metrics = MetricStore(run.paths.sample_metrics, run_id, spec.config_hash)
    for sample_id in sample_ids:
        prediction = PredictionRecord(
            run_id=run_id,
            config_hash=spec.config_hash,
            sample_id=sample_id,
            model_id=model_id,
            dataset_id="mmlu_pro",
            task_id="mmlu_pro",
            status="success",
            raw_output="A",
        )
        predictions.append(prediction)
        metrics.append(
            SampleMetricRecord(
                run_id=run_id,
                config_hash=spec.config_hash,
                sample_id=sample_id,
                model_id=model_id,
                dataset_id="mmlu_pro",
                task_id="mmlu_pro",
                prediction_hash=prediction.record_hash,
                status="success",
                metrics={"accuracy": value},
                evaluator_metadata={"accuracy.version": "fixture"},
            )
        )
    metrics.write_summary(run.paths.summary)
    finalize_run(run.paths, "completed")
    return run.paths.root
