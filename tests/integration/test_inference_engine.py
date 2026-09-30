from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from dimebench.artifacts import PredictionStore
from dimebench.config import load_config
from dimebench.inference import (
    InferenceEngine,
    RetryPolicy,
    load_inference_dataset,
)
from dimebench.models import Capability, create_adapter
from dimebench.models.mock import MockModelAdapter
from dimebench.schemas import ModelRequest, ModelResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FlakyMockAdapter(MockModelAdapter):
    """Fail one denoising call so the engine retry path is observable."""

    CAPABILITIES = frozenset(
        {
            Capability.GENERATE,
            Capability.INFILL,
            Capability.DENOISE,
        }
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _denoise(
        self,
        requests: tuple[ModelRequest, ...],
        *,
        steps: int,
        schedule: str,
        mask_policy: str,
    ) -> Sequence[ModelResponse]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient fixture failure")
        return super()._denoise(
            requests,
            steps=steps,
            schedule=schedule,
            mask_policy=mask_policy,
        )


def _spec(tmp_path: Path):
    return load_config(
        PROJECT_ROOT / "configs" / "smoke.yaml",
        [f"output_dir={tmp_path.as_posix()}"],
    )


def test_infer_cli_writes_raw_predictions_and_resumes(tmp_path: Path, capsys) -> None:
    from dimebench.cli.main import main

    config = PROJECT_ROOT / "configs" / "smoke.yaml"
    override = f"output_dir={tmp_path.as_posix()}"
    assert main(["infer", "--config", str(config), "--set", override]) == 0
    first_output = json.loads(capsys.readouterr().out)
    assert first_output["written_predictions"] == 4
    assert first_output["failed_predictions"] == 0

    assert main(["infer", "--config", str(config), "--set", override]) == 0
    second_output = json.loads(capsys.readouterr().out)
    assert second_output["previously_completed"] == 4
    assert second_output["written_predictions"] == 0

    spec = _spec(tmp_path)
    store = PredictionStore(
        Path(first_output["predictions"]), spec.run_id, spec.config_hash
    )
    assert len(store) == 4
    record = store.get(next(iter(store.completed_sample_ids)))
    assert record.raw_output
    assert record.request["metadata"]["prompt_hash"]
    assert record.decoding_metadata["inference"]["usage"]
    assert record.decoding_metadata["denoising_steps"] == 8


def test_engine_retries_adapter_exception_and_retains_outputs(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    dataset = load_inference_dataset(spec)
    adapter = FlakyMockAdapter(spec.model)
    result = InferenceEngine(
        spec,
        adapter,
        prompt_root=PROJECT_ROOT / "dimebench" / "prompts",
        retry_policy=RetryPolicy(max_attempts=2),
    ).run(dataset)
    assert result.written_predictions == 4
    assert result.failed_predictions == 0
    assert adapter.calls == 3  # one failed batch call plus two successful batches
    trace = result.trace_path.read_text(encoding="utf-8")
    assert '"event":"retry"' in trace


def test_cache_can_rehydrate_an_interrupted_prediction_store(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    dataset = load_inference_dataset(spec)
    first = InferenceEngine(
        spec,
        create_adapter(spec.model),
        prompt_root=PROJECT_ROOT / "dimebench" / "prompts",
    ).run(dataset)
    first.predictions_path.write_text("", encoding="utf-8")

    second = InferenceEngine(
        spec,
        create_adapter(spec.model),
        prompt_root=PROJECT_ROOT / "dimebench" / "prompts",
    ).run(dataset)
    assert second.written_predictions == 4
    assert second.cache_hits == 4
    store = PredictionStore(second.predictions_path, spec.run_id, spec.config_hash)
    assert all(
        store.get(sample_id).decoding_metadata["inference"]["cache_hit"] is True
        for sample_id in store.completed_sample_ids
    )
