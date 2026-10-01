from __future__ import annotations

import os
from pathlib import Path

import pytest

from dimebench.config import load_model_config
from dimebench.models import create_adapter
from dimebench.models.diffusion import DiffusionModelAdapter
from dimebench.schemas import ModelRequest, ModelResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_GPU_TESTS = os.environ.get("DIMEBENCH_RUN_GPU_TESTS") == "1"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not RUN_GPU_TESTS,
        reason="set DIMEBENCH_RUN_GPU_TESTS=1 on a CUDA host",
    ),
]


def _write_response(name: str, response: ModelResponse) -> None:
    directory = os.environ.get("DIMEBENCH_GPU_ARTIFACT_DIR")
    if directory is None:
        return
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}.json").write_text(
        response.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _gpu_spec(config_path: Path, model_environment_key: str):
    spec = load_model_config(config_path)
    updates: dict[str, object] = {
        "device": os.environ.get("DIMEBENCH_GPU_DEVICE", "cuda"),
    }
    model_path = os.environ.get(model_environment_key)
    if model_path:
        updates["name_or_path"] = model_path
        updates["tokenizer_name_or_path"] = model_path
    revision = os.environ.get(f"{model_environment_key}_REVISION")
    if revision:
        updates["revision"] = revision
        updates["tokenizer_revision"] = revision
    return spec.model_copy(update=updates)


def _assert_cuda() -> None:
    import torch

    assert torch.cuda.is_available(), "Step 5 integration tests require CUDA"


def test_huggingface_ar_real_generation() -> None:
    _assert_cuda()
    spec = _gpu_spec(
        PROJECT_ROOT / "configs/models/autoregressive/llama3_8b_instruct.yaml",
        "DIMEBENCH_AR_MODEL",
    )
    adapter = create_adapter(spec)
    request = ModelRequest(
        request_id="gpu-ar-1",
        sample_id="gpu-ar-1",
        task_id="step5_ar_smoke",
        prompt="Reply with only the word Paris: What is the capital of France?",
        max_output_tokens=16,
        seed=7,
        metadata={"do_sample": False, "temperature": 0.0},
    )
    try:
        adapter.load()
        response = adapter.generate([request])[0]
    finally:
        adapter.close()

    assert response.status == "success"
    assert response.text is not None
    assert response.usage.output_tokens is not None
    assert response.usage.wall_time_seconds is not None
    assert response.usage.peak_memory_bytes is not None
    assert response.decoding_metadata["requested_model_revision"] == spec.revision
    assert response.decoding_metadata["model_revision"]
    _write_response("huggingface_ar_response", response)


def test_llada_real_denoising() -> None:
    _assert_cuda()
    spec = _gpu_spec(
        PROJECT_ROOT / "configs/models/diffusion/llada_8b_instruct.yaml",
        "DIMEBENCH_DLLM_MODEL",
    )
    adapter = create_adapter(spec)
    assert isinstance(adapter, DiffusionModelAdapter)
    request = ModelRequest(
        request_id="gpu-llada-1",
        sample_id="gpu-llada-1",
        task_id="step5_llada_smoke",
        prompt="Answer the question concisely: What is the capital of France?",
        max_output_tokens=32,
        seed=7,
        metadata={
            "temperature": 0.0,
            "block_length": 32,
            "unmasking_strategy": "confidence_based",
        },
    )
    try:
        adapter.load()
        response = adapter.denoise(
            [request],
            steps=32,
            schedule="linear",
            mask_policy="append",
        )[0]
    finally:
        adapter.close()

    assert response.status == "success"
    assert response.text is not None
    assert response.usage.denoising_steps == 32
    assert response.usage.forward_passes == 32
    assert response.usage.wall_time_seconds is not None
    assert response.usage.peak_memory_bytes is not None
    assert response.decoding_metadata["unmasking_strategy"] == "confidence_based"
    _write_response("llada_response", response)
