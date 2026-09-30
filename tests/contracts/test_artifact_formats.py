from __future__ import annotations

import json
from pathlib import Path

from dimebench.artifacts import (
    MetricStore,
    PredictionRecord,
    PredictionStore,
    SampleMetricRecord,
    initialize_run,
)
from dimebench.artifacts.provenance import EnvironmentInfo, GitInfo, HardwareInfo
from dimebench.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_artifact_files_are_standard_json_and_jsonl(tmp_path: Path) -> None:
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
        hardware=HardwareInfo(machine="test", processor="test", cpu_count=1),
        dependencies={},
        accelerator_environment={},
    )
    run = initialize_run(
        spec,
        ["sample-1"],
        dataset_hash="a" * 64,
        environment=environment,
    )
    prediction = PredictionRecord(
        run_id=spec.run_id,
        config_hash=spec.config_hash,
        sample_id="sample-1",
        model_id=spec.model.id,
        dataset_id=spec.dataset.id,
        task_id=spec.task.id,
        status="success",
        raw_output="answer",
    )
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

    for path in (run.paths.manifest, run.paths.environment, run.paths.summary):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "1.0"

    for path in (run.paths.predictions, run.paths.sample_metrics):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["sample_id"] == "sample-1"
