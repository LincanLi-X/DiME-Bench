from __future__ import annotations

from pathlib import Path

from dimebench.artifacts import PredictionRecord, PredictionStore, initialize_run
from dimebench.datasets import GenerationRecord, NormalizedDataset
from dimebench.schemas import DatasetSpec, DecodingSpec, ModelSpec, RunSpec, TaskSpec
from dimebench.tracks import evaluate_step12_run


def test_step12_scores_recompute_from_saved_predictions(tmp_path: Path) -> None:
    records = tuple(
        GenerationRecord(
            sample_id=f"gsm8k_general.test.fixture-{index:02d}",
            dataset_id="gsm8k_general",
            split="test",
            source_id=f"fixture-{index:02d}",
            instruction=f"What is {index} plus 1?",
            references=(str(index + 1),),
        )
        for index in range(25)
    )
    dataset = NormalizedDataset("gsm8k_general", "test", records)
    data_path = tmp_path / "gsm8k.jsonl"
    dataset.write_jsonl(data_path)
    spec = RunSpec(
        run_id="step12-recompute-fixture",
        output_dir=tmp_path / "runs",
        model=ModelSpec(
            id="fixture-ar",
            family="autoregressive",
            adapter="mock",
            name_or_path="fixture/ar",
            revision="fixture",
            dtype="float32",
            device="cpu",
            capabilities=("generate",),
        ),
        dataset=DatasetSpec(
            id="gsm8k_general",
            source="local",
            name_or_path=str(data_path),
            revision="fixture",
            split="test",
            manifest_hash="a" * 64,
            sample_limit=25,
        ),
        task=TaskSpec(
            id="gsm8k_general",
            track="track1_general",
            dataset_id="gsm8k_general",
            type="generation",
            prompt_id="track1.numeric_answer",
            prompt_version="1.0.0",
            parser_id="numeric_answer",
            parser_version="1.0.0",
            metrics=("normalized_exact_match",),
            primary_metric="normalized_exact_match",
            max_output_tokens=64,
        ),
        decoding=DecodingSpec(max_new_tokens=64),
    )
    run = initialize_run(
        spec,
        [record.sample_id for record in records],
        dataset_hash=dataset.dataset_hash,
    )
    predictions = PredictionStore(run.paths.predictions, spec.run_id, spec.config_hash)
    for index, sample in enumerate(records):
        predictions.append(
            PredictionRecord(
                run_id=spec.run_id,
                config_hash=spec.config_hash,
                sample_id=sample.sample_id,
                model_id=spec.model.id,
                dataset_id=spec.dataset.id,
                task_id=spec.task.id,
                status="success",
                raw_output=str(index + 1),
            )
        )

    report = evaluate_step12_run(
        spec,
        nli_model_id="unused",
        nli_revision="unused",
        nli_device="cpu",
    )

    assert report["recomputation_matches"] is True
    assert report["sample_count"] == 25
    assert report["primary_metric_value"] == 1.0
    assert run.paths.summary.is_file()
    assert (run.paths.root / "recomputed/summary.json").is_file()
