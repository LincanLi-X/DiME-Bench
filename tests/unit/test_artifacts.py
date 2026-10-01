from __future__ import annotations

import json
from pathlib import Path

import pytest

from dimebench.artifacts import (
    ArtifactError,
    DuplicateSampleError,
    MetricStore,
    PredictionRecord,
    PredictionStore,
    SampleMetricRecord,
    finalize_run,
    initialize_run,
    load_environment,
    load_manifest,
    validate_run_artifacts,
)
from dimebench.artifacts.hashing import hash_file, hash_json, hash_ordered_strings
from dimebench.artifacts.provenance import (
    EnvironmentInfo,
    GitInfo,
    HardwareInfo,
    collect_environment,
)
from dimebench.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
DATASET_HASH = "d" * 64


def fake_environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        dimebench_version="0.1.0.dev0",
        python_version="3.12.0",
        python_implementation="CPython",
        platform="test-platform",
        git=GitInfo(commit="a" * 40, dirty=False, root="/repo"),
        hardware=HardwareInfo(
            machine="test-machine",
            processor="test-cpu",
            cpu_count=4,
            memory_bytes=1024,
        ),
        dependencies={"dime-bench": "0.1.0.dev0"},
        accelerator_environment={},
    )


def smoke_spec(tmp_path: Path, *extra_overrides: str):
    overrides = [f"output_dir={tmp_path.as_posix()}", *extra_overrides]
    return load_config(SMOKE_CONFIG, overrides)


def make_prediction(spec, sample_id: str, output: str) -> PredictionRecord:
    return PredictionRecord(
        run_id=spec.run_id,
        config_hash=spec.config_hash,
        sample_id=sample_id,
        model_id=spec.model.id,
        dataset_id=spec.dataset.id,
        task_id=spec.task.id,
        status="success",
        request={"prompt": f"prompt for {sample_id}"},
        raw_output=output,
        decoding_metadata={"steps": 8},
    )


def test_initialize_run_creates_fixed_artifact_layout(tmp_path: Path) -> None:
    spec = smoke_spec(tmp_path)
    run = initialize_run(
        spec,
        ["sample-1", "sample-2"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )

    assert run.resumed is False
    assert {path.name for path in run.paths.all_files()} == {
        "run_manifest.json",
        "environment.json",
        "predictions.jsonl",
        "sample_metrics.jsonl",
        "summary.json",
    }
    assert all(path.is_file() for path in run.paths.all_files())

    manifest = load_manifest(run.paths.manifest)
    assert manifest.config_hash == spec.config_hash
    assert hash_json(manifest.config) == spec.config_hash
    assert manifest.adapter_name == "mock"
    assert manifest.adapter_version == "1.0.0"
    assert manifest.model_revision == "v1"
    assert manifest.dataset_manifest_hash == DATASET_HASH
    assert manifest.sample_ids_hash == hash_ordered_strings(["sample-1", "sample-2"])
    assert manifest.seed == 7
    assert load_environment(run.paths.environment).git.commit == "a" * 40


def test_identical_run_resumes_and_conflicting_run_is_rejected(
    tmp_path: Path,
) -> None:
    spec = smoke_spec(tmp_path)
    first = initialize_run(
        spec,
        ["sample-1"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )
    second = initialize_run(
        spec,
        ["sample-1"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )
    assert first.paths == second.paths
    assert second.resumed is True

    conflicting = smoke_spec(tmp_path, "batch_size=3")
    with pytest.raises(ArtifactError, match="config_hash"):
        initialize_run(
            conflicting,
            ["sample-1"],
            dataset_hash=DATASET_HASH,
            environment=fake_environment(),
        )


def test_prediction_store_resumes_and_skips_terminal_samples(tmp_path: Path) -> None:
    spec = smoke_spec(tmp_path)
    run = initialize_run(
        spec,
        ["sample-1", "sample-2", "sample-3"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )
    store = PredictionStore(run.paths.predictions, spec.run_id, spec.config_hash)
    prediction = make_prediction(spec, "sample-1", "middle")
    store.append(prediction)
    store.append(
        PredictionRecord(
            run_id=spec.run_id,
            config_hash=spec.config_hash,
            sample_id="sample-2",
            model_id=spec.model.id,
            dataset_id=spec.dataset.id,
            task_id=spec.task.id,
            status="failure",
            failure_class="empty_output",
        )
    )

    resumed = PredictionStore(run.paths.predictions, spec.run_id, spec.config_hash)
    assert resumed.completed_sample_ids == {"sample-1", "sample-2"}
    assert resumed.pending_sample_ids(["sample-1", "sample-2", "sample-3"]) == (
        "sample-3",
    )
    assert resumed.get("sample-1").record_hash == prediction.record_hash
    with pytest.raises(DuplicateSampleError):
        resumed.append(prediction)


def test_summary_is_traceable_and_terminal_artifacts_are_sealed(
    tmp_path: Path,
) -> None:
    spec = smoke_spec(tmp_path)
    run = initialize_run(
        spec,
        ["sample-1", "sample-2"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )
    predictions = PredictionStore(
        run.paths.predictions,
        spec.run_id,
        spec.config_hash,
    )
    metrics = MetricStore(
        run.paths.sample_metrics,
        spec.run_id,
        spec.config_hash,
    )

    for sample_id, output, score in (
        ("sample-1", "first", 0.5),
        ("sample-2", "second", 1.0),
    ):
        prediction = make_prediction(spec, sample_id, output)
        predictions.append(prediction)
        metrics.append(
            SampleMetricRecord(
                run_id=spec.run_id,
                config_hash=spec.config_hash,
                sample_id=sample_id,
                model_id=spec.model.id,
                dataset_id=spec.dataset.id,
                task_id=spec.task.id,
                prediction_hash=prediction.record_hash,
                status="success",
                metrics={"token_f1": score},
                evaluator_metadata={"version": "1.0"},
            )
        )

    summary = metrics.write_summary(run.paths.summary)
    assert summary.metrics["token_f1"].value == pytest.approx(0.75)
    assert summary.metrics["token_f1"].sample_ids == (
        "sample-1",
        "sample-2",
    )
    manifest = finalize_run(run.paths, "completed")
    assert manifest.status == "completed"
    assert manifest.artifact_hashes["sample_metrics.jsonl"] == hash_file(
        run.paths.sample_metrics
    )
    validate_run_artifacts(run.paths)


def test_summary_includes_explicit_failure_zero_in_denominator(tmp_path: Path) -> None:
    """Failure-zero policy must not silently improve aggregate scores."""
    spec = smoke_spec(tmp_path)
    run = initialize_run(
        spec,
        ["sample-1", "sample-2"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )
    predictions = PredictionStore(run.paths.predictions, spec.run_id, spec.config_hash)
    metrics = MetricStore(run.paths.sample_metrics, spec.run_id, spec.config_hash)
    for sample_id, status, score in (
        ("sample-1", "success", 1.0),
        ("sample-2", "failure", 0.0),
    ):
        prediction = PredictionRecord(
            run_id=spec.run_id,
            config_hash=spec.config_hash,
            sample_id=sample_id,
            model_id=spec.model.id,
            dataset_id=spec.dataset.id,
            task_id=spec.task.id,
            status=status,
            raw_output="answer" if status == "success" else None,
            failure_class="parse_error" if status == "failure" else None,
        )
        predictions.append(prediction)
        metrics.append(
            SampleMetricRecord(
                run_id=spec.run_id,
                config_hash=spec.config_hash,
                sample_id=sample_id,
                model_id=spec.model.id,
                dataset_id=spec.dataset.id,
                task_id=spec.task.id,
                prediction_hash=prediction.record_hash,
                status=status,
                metrics={"accuracy": score},
            )
        )
    summary = metrics.write_summary(run.paths.summary)
    aggregate = summary.metrics["accuracy"]
    assert aggregate.value == pytest.approx(0.5)
    assert aggregate.sample_count == 2
    assert aggregate.sample_ids == ("sample-1", "sample-2")
    validate_run_artifacts(run.paths)


def test_traceability_validator_detects_post_finalize_tampering(
    tmp_path: Path,
) -> None:
    spec = smoke_spec(tmp_path)
    run = initialize_run(
        spec,
        ["sample-1"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )
    prediction = make_prediction(spec, "sample-1", "output")
    PredictionStore(
        run.paths.predictions,
        spec.run_id,
        spec.config_hash,
    ).append(prediction)
    metric_store = MetricStore(
        run.paths.sample_metrics,
        spec.run_id,
        spec.config_hash,
    )
    metric_store.append(
        SampleMetricRecord(
            run_id=spec.run_id,
            config_hash=spec.config_hash,
            sample_id="sample-1",
            model_id=spec.model.id,
            dataset_id=spec.dataset.id,
            task_id=spec.task.id,
            prediction_hash=prediction.record_hash,
            status="success",
            metrics={"token_f1": 1.0},
        )
    )
    metric_store.write_summary(run.paths.summary)
    finalize_run(run.paths, "completed")

    with run.paths.sample_metrics.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ArtifactError, match="summary does not match"):
        validate_run_artifacts(run.paths)


def test_traceability_validator_recomputes_summary_values(tmp_path: Path) -> None:
    spec = smoke_spec(tmp_path)
    run = initialize_run(
        spec,
        ["sample-1"],
        dataset_hash=DATASET_HASH,
        environment=fake_environment(),
    )
    prediction = make_prediction(spec, "sample-1", "output")
    PredictionStore(
        run.paths.predictions,
        spec.run_id,
        spec.config_hash,
    ).append(prediction)
    metric_store = MetricStore(
        run.paths.sample_metrics,
        spec.run_id,
        spec.config_hash,
    )
    metric_store.append(
        SampleMetricRecord(
            run_id=spec.run_id,
            config_hash=spec.config_hash,
            sample_id="sample-1",
            model_id=spec.model.id,
            dataset_id=spec.dataset.id,
            task_id=spec.task.id,
            prediction_hash=prediction.record_hash,
            status="success",
            metrics={"token_f1": 1.0},
        )
    )
    metric_store.write_summary(run.paths.summary)
    payload = json.loads(run.paths.summary.read_text(encoding="utf-8"))
    payload["metrics"]["token_f1"]["value"] = 0.0
    run.paths.summary.write_text(json.dumps(payload), encoding="utf-8")
    finalize_run(run.paths, "completed")

    with pytest.raises(ArtifactError, match="aggregate value mismatch"):
        validate_run_artifacts(run.paths)


def test_real_environment_capture_has_core_reproducibility_fields() -> None:
    environment = collect_environment(PROJECT_ROOT)
    assert environment.dimebench_version == "1.0.0"
    assert environment.python_version
    assert environment.hardware.machine
    assert "dime-bench" in environment.dependencies


def test_hash_json_is_independent_of_mapping_order(tmp_path: Path) -> None:
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"value": 1}), encoding="utf-8")
    assert len(hash_file(artifact)) == 64
