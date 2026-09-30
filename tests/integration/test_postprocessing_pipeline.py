from __future__ import annotations

from pathlib import Path

from dimebench.artifacts import PredictionStore
from dimebench.config import load_config
from dimebench.inference import InferenceEngine, load_inference_dataset
from dimebench.models import create_adapter
from dimebench.postprocessors import (
    postprocess_predictions,
    write_postprocessed_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mock_inference_to_versioned_postprocessing(tmp_path: Path) -> None:
    spec = load_config(
        PROJECT_ROOT / "configs" / "smoke.yaml",
        [f"output_dir={tmp_path.as_posix()}"],
    )
    dataset = load_inference_dataset(spec)
    inference = InferenceEngine(
        spec,
        create_adapter(spec.model),
        prompt_root=PROJECT_ROOT / "dimebench" / "prompts",
    ).run(dataset)
    store = PredictionStore(inference.predictions_path, spec.run_id, spec.config_hash)
    predictions = [store.get(sample.sample_id) for sample in dataset]
    samples = {sample.sample_id: sample for sample in dataset}
    results = postprocess_predictions(predictions, samples, spec.task)

    assert len(results) == len(dataset) == 4
    assert all(result.status == "success" for result in results)
    assert all(result.raw_output for result in results)
    assert all(result.parsed_output for result in results)
    assert all(result.metadata["parser"]["version"] == "1.0.0" for result in results)

    output = tmp_path / "postprocessed.jsonl"
    write_postprocessed_results(output, results)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
