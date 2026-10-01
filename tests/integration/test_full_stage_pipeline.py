from __future__ import annotations

import json
from pathlib import Path

import yaml

from dimebench.cli.main import main
from dimebench.config import load_config
from dimebench.datasets import NormalizedDataset
from dimebench.schemas import DatasetSpec, DecodingSpec, RunSpec
from dimebench.tasks import load_task_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prepare_infer_evaluate_summarize_pipeline(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    assert (
        main(
            [
                "prepare",
                "--suite",
                "dime_bench_v1",
                "--output-dir",
                str(prepared_root),
                "--force",
            ]
        )
        == 0
    )
    split_path = prepared_root / "dime_bench_v1/mmlu_pro/test.jsonl"
    dataset = NormalizedDataset.read_jsonl(split_path)
    smoke = load_config(PROJECT_ROOT / "configs/smoke.yaml")
    spec = RunSpec(
        run_id="step16-stage-pipeline",
        seed=0,
        deterministic=True,
        batch_size=1,
        output_dir=tmp_path / "outputs",
        model=smoke.model,
        dataset=DatasetSpec(
            id="mmlu_pro",
            source="local",
            name_or_path=str(split_path),
            revision="bundled-fixture-v1",
            split="test",
            manifest_hash=dataset.dataset_hash,
            sample_limit=1,
        ),
        task=load_task_spec(
            PROJECT_ROOT / "configs/tasks/track1_general/mmlu_pro.yaml"
        ),
        decoding=DecodingSpec(
            max_new_tokens=8,
            temperature=0.0,
            do_sample=False,
            denoising_steps=8,
            unmasking_strategy="confidence_based",
            schedule="linear",
            mask_policy="bounded_middle",
        ),
        tags=("step16", "integration"),
    )
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    assert main(["infer", "--config", str(config_path)]) == 0
    run_dir = tmp_path / "outputs/step16-stage-pipeline"
    running = json.loads((run_dir / "run_manifest.json").read_text())
    assert running["status"] == "running"
    assert main(["evaluate", "--run-dir", str(run_dir)]) == 0
    assert (
        main(
            [
                "summarize",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(tmp_path / "summary"),
            ]
        )
        == 0
    )

    completed = json.loads((run_dir / "run_manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    report = json.loads((tmp_path / "summary/benchmark-summary.json").read_text())
    assert completed["status"] == "completed"
    assert set(summary["metrics"]) == {"accuracy"}
    assert summary["metrics"]["accuracy"]["total_sample_count"] == 1
    assert summary["metrics"]["accuracy"]["sample_count"] == 1
    assert len(report["datasets"]) == 1
    assert len(report["models"]) == 1
    assert report["datasets"][0]["run_id"] == "step16-stage-pipeline"
