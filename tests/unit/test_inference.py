from __future__ import annotations

from pathlib import Path

import pytest

from dimebench.config import load_config
from dimebench.datasets.records import MultipleChoiceRecord
from dimebench.inference.batcher import iter_batches
from dimebench.inference.cache import CacheError, ResponseCache
from dimebench.inference.request_builder import RequestBuilder
from dimebench.models import create_adapter
from dimebench.schemas import ModelResponse
from dimebench.tasks import TaskCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ordered_batching_and_validation() -> None:
    assert list(iter_batches((1, 2, 3, 4, 5), 2)) == [
        (1, 2),
        (3, 4),
        (5,),
    ]
    with pytest.raises(ValueError, match="positive"):
        list(iter_batches((1,), 0))


def test_request_builder_is_stable_and_records_prompt_provenance() -> None:
    smoke = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    catalog = TaskCatalog.from_roots(
        PROJECT_ROOT / "configs" / "tasks",
        PROJECT_ROOT / "dimebench" / "prompts",
    )
    mmlu = catalog.get("mmlu_pro").spec
    spec = smoke.model_copy(
        update={
            "dataset": smoke.dataset.model_copy(update={"id": "mmlu_pro"}),
            "task": mmlu,
            "decoding": smoke.decoding.model_copy(update={"max_new_tokens": 8}),
        }
    )
    adapter = create_adapter(spec.model)
    sample = MultipleChoiceRecord(
        sample_id="mmlu_pro.test.example",
        dataset_id="mmlu_pro",
        split="test",
        source_id="example",
        question="Which option is first?",
        choices=("alpha", "beta"),
        answer_index=0,
    )
    builder = RequestBuilder(spec, adapter, PROJECT_ROOT / "dimebench" / "prompts")
    first = builder.build(sample)
    second = builder.build(sample)
    assert first == second
    assert first.mode == "score_options"
    assert first.options == sample.choices
    assert first.metadata["prompt_version"] == "1.0.0"
    assert len(str(first.metadata["prompt_hash"])) == 64


def test_response_cache_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    spec = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    adapter = create_adapter(spec.model)
    from dimebench.inference import load_inference_dataset

    sample = load_inference_dataset(spec).records[0]
    request = RequestBuilder(
        spec, adapter, PROJECT_ROOT / "dimebench" / "prompts"
    ).build(sample)
    response = ModelResponse(
        request_id=request.request_id,
        sample_id=request.sample_id,
        model_id=spec.model.id,
        status="success",
        text="cached",
        finish_reason="completed",
    )
    cache = ResponseCache(tmp_path)
    key = cache.key_for(request, spec.model, spec.decoding)
    assert cache.get(key) is None
    cache.put(key, request, spec.model, spec.decoding, response)
    assert cache.get(key) == response

    path = tmp_path / key[:2] / f"{key}.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CacheError, match="invalid"):
        cache.get(key)
