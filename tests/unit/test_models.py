from __future__ import annotations

from collections.abc import Sequence

import pytest

from dimebench.models import (
    AdapterContractError,
    AdapterState,
    AdapterStateError,
    Capability,
    UnsupportedCapabilityError,
    create_adapter,
)
from dimebench.models.autoregressive import AutoregressiveModelAdapter
from dimebench.models.diffusion import DiffusionModelAdapter
from dimebench.schemas import ModelRequest, ModelResponse, ModelSpec


class GenerateOnlyAdapter(AutoregressiveModelAdapter):
    ADAPTER_NAME = "generate-only"
    CAPABILITIES = frozenset({Capability.GENERATE})

    def _load(self) -> None:
        return None

    def _close(self) -> None:
        return None

    def _generate(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        return tuple(
            ModelResponse(
                request_id=request.request_id,
                sample_id=request.sample_id,
                model_id=self.spec.id,
                status="success",
                text="generated",
                finish_reason="completed",
            )
            for request in requests
        )


def ar_spec(**updates: object) -> ModelSpec:
    values = {
        "id": "mock-ar",
        "family": "autoregressive",
        "adapter": "generate-only",
        "name_or_path": "builtin/mock-ar",
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(updates)
    return ModelSpec.model_validate(values)


def generate_request(request_id: str = "request-1") -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        sample_id=request_id.replace("request", "sample"),
        task_id="task",
        prompt="Generate a response",
        max_output_tokens=8,
    )


def test_adapter_lifecycle_and_generation_contract() -> None:
    adapter = GenerateOnlyAdapter(ar_spec())
    assert adapter.state == AdapterState.CREATED
    assert adapter.supports("generate")

    with pytest.raises(AdapterStateError, match="must be loaded"):
        adapter.generate([generate_request()])

    adapter.load()
    adapter.load()
    response = adapter.generate([generate_request()])[0]
    assert response.text == "generated"
    adapter.close()
    adapter.close()
    assert adapter.state == AdapterState.CLOSED


def test_unsupported_capability_has_actionable_error() -> None:
    adapter = GenerateOnlyAdapter(ar_spec())
    adapter.load()
    request = ModelRequest(
        request_id="request-1",
        sample_id="sample-1",
        task_id="task",
        mode="score_options",
        prompt="Choose",
        max_output_tokens=1,
        options=("A", "B"),
    )
    with pytest.raises(UnsupportedCapabilityError, match="score_options"):
        adapter.score_options([request])


def test_declared_capability_must_be_implemented() -> None:
    with pytest.raises(AdapterContractError, match="unsupported by adapter"):
        GenerateOnlyAdapter(ar_spec(capabilities=("generate", "infill")))


def test_configured_adapter_version_must_match_implementation() -> None:
    with pytest.raises(AdapterContractError, match="does not match configured"):
        GenerateOnlyAdapter(ar_spec(adapter_version="2.0.0"))


def test_mock_factory_validates_model_family() -> None:
    with pytest.raises(AdapterContractError, match="requires model family"):
        create_adapter(ar_spec(adapter="mock"))


def test_mock_generation_scoring_and_denoising() -> None:
    spec = ModelSpec(
        id="mock-dllm",
        family="diffusion",
        adapter="mock",
        name_or_path="builtin/mock",
        dtype="float32",
        device="cpu",
        capabilities=("generate", "infill", "score_options", "denoise"),
    )
    adapter = create_adapter(spec)
    assert isinstance(adapter, DiffusionModelAdapter)
    adapter.load()

    infill = ModelRequest(
        request_id="infill-1",
        sample_id="sample-1",
        task_id="infilling",
        mode="infill",
        prompt="Prefix: left; suffix: right",
        prefix="left",
        suffix="right",
        max_output_tokens=8,
        metadata={"mock_response": "center"},
    )
    generated = adapter.generate([infill])[0]
    assert generated.text == "center"
    assert generated.usage.denoising_steps is None

    option_request = ModelRequest(
        request_id="choice-1",
        sample_id="sample-2",
        task_id="choice",
        mode="score_options",
        prompt="Choose",
        options=("A", "B", "C"),
        max_output_tokens=1,
    )
    scored = adapter.score_options([option_request])[0]
    assert scored.option_scores == {"A": 0.0, "B": -1.0, "C": -2.0}

    denoised = adapter.denoise(
        [infill],
        steps=8,
        schedule="linear",
        mask_policy="bounded_middle",
    )[0]
    assert denoised.usage.denoising_steps == 8
    assert denoised.decoding_metadata["schedule"] == "linear"
