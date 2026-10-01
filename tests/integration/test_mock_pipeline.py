from __future__ import annotations

from pathlib import Path

from dimebench.artifacts import (
    MetricStore,
    PredictionRecord,
    PredictionStore,
    SampleMetricRecord,
    finalize_run,
    initialize_run,
    validate_run_artifacts,
)
from dimebench.artifacts.provenance import EnvironmentInfo, GitInfo, HardwareInfo
from dimebench.config import load_config
from dimebench.models import create_adapter
from dimebench.models.diffusion import DiffusionModelAdapter
from dimebench.schemas import ModelRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mock_adapter_runs_smoke_task_without_gpu(tmp_path: Path) -> None:
    spec = load_config(
        PROJECT_ROOT / "configs" / "smoke.yaml",
        [f"output_dir={tmp_path.as_posix()}"],
    )
    environment = EnvironmentInfo(
        dimebench_version="0.1.0.dev0",
        python_version="3.12.0",
        python_implementation="CPython",
        platform="test",
        git=GitInfo(),
        hardware=HardwareInfo(machine="cpu", processor="mock", cpu_count=1),
        dependencies={},
        accelerator_environment={},
    )
    run = initialize_run(
        spec,
        ["smoke-1"],
        dataset_hash="b" * 64,
        environment=environment,
    )
    request = ModelRequest(
        request_id="smoke-request-1",
        sample_id="smoke-1",
        task_id=spec.task.id,
        mode="infill",
        prompt="Complete only the missing middle between the boundaries.",
        prefix="DiME-Bench evaluates",
        suffix="language models.",
        max_output_tokens=spec.task.max_output_tokens,
        seed=spec.seed,
        metadata={"mock_response": "discrete diffusion"},
    )

    adapter = create_adapter(spec.model)
    assert isinstance(adapter, DiffusionModelAdapter)
    with adapter:
        response = adapter.denoise(
            [request],
            steps=spec.decoding.denoising_steps or 1,
            schedule=spec.decoding.schedule or "linear",
            mask_policy=spec.decoding.mask_policy or "bounded_middle",
        )[0]

    assert response.text == "discrete diffusion"
    prediction = PredictionRecord(
        run_id=spec.run_id,
        config_hash=spec.config_hash,
        sample_id=request.sample_id,
        model_id=spec.model.id,
        dataset_id=spec.dataset.id,
        task_id=spec.task.id,
        status="success",
        request=request.model_dump(mode="json"),
        raw_output=response.text,
        decoding_metadata=response.decoding_metadata,
    )
    PredictionStore(
        run.paths.predictions,
        spec.run_id,
        spec.config_hash,
    ).append(prediction)
    metrics = MetricStore(
        run.paths.sample_metrics,
        spec.run_id,
        spec.config_hash,
    )
    metrics.append(
        SampleMetricRecord(
            run_id=spec.run_id,
            config_hash=spec.config_hash,
            sample_id=request.sample_id,
            model_id=spec.model.id,
            dataset_id=spec.dataset.id,
            task_id=spec.task.id,
            prediction_hash=prediction.record_hash,
            status="success",
            metrics={"token_f1": 1.0, "boundary_consistency": 1.0},
            evaluator_metadata={"source": "mock-smoke"},
        )
    )
    metrics.write_summary(run.paths.summary)
    finalize_run(run.paths, "completed")
    validate_run_artifacts(run.paths)
