"""Deterministic CPU-only mock adapter for smoke and contract tests."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import JsonValue

from dimebench.models.capabilities import Capability
from dimebench.models.diffusion.base import DiffusionModelAdapter
from dimebench.models.registry import register_adapter
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse, ModelUsage


@register_adapter("mock")
class MockModelAdapter(DiffusionModelAdapter):
    """A deterministic dLLM-shaped adapter that allocates no model resources."""

    ADAPTER_NAME = "mock"
    CAPABILITIES = frozenset(
        {
            Capability.GENERATE,
            Capability.SCORE_OPTIONS,
            Capability.INFILL,
            Capability.EDIT,
            Capability.REASONING,
            Capability.DENOISE,
        }
    )

    def _load(self) -> None:
        return None

    def _close(self) -> None:
        return None

    def _text_for(self, request: ModelRequest) -> str:
        configured = request.metadata.get("mock_response")
        if isinstance(configured, str):
            return configured
        if request.mode == "infill":
            return f"mock middle for {request.sample_id}"
        return f"mock response for {request.sample_id}"

    def _generate(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        return tuple(self._generation_response(request) for request in requests)

    def _generation_response(
        self,
        request: ModelRequest,
        *,
        denoising_steps: int | None = None,
        decoding_metadata: dict[str, JsonValue] | None = None,
    ) -> ModelResponse:
        text = self._text_for(request)
        return ModelResponse(
            request_id=request.request_id,
            sample_id=request.sample_id,
            model_id=self.spec.id,
            status="success",
            text=text,
            finish_reason="completed",
            usage=ModelUsage(
                input_tokens=len(request.prompt.split()),
                output_tokens=len(text.split()),
                forward_passes=denoising_steps or 1,
                denoising_steps=denoising_steps,
                wall_time_seconds=0.0,
                peak_memory_bytes=0,
            ),
            decoding_metadata=decoding_metadata
            or {
                "adapter": self.ADAPTER_NAME,
                "adapter_version": self.ADAPTER_VERSION,
            },
        )

    def _score_options(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        responses = []
        for request in requests:
            scores = {
                option: -float(index) for index, option in enumerate(request.options)
            }
            responses.append(
                ModelResponse(
                    request_id=request.request_id,
                    sample_id=request.sample_id,
                    model_id=self.spec.id,
                    status="success",
                    option_scores=scores,
                    finish_reason="completed",
                    usage=ModelUsage(
                        input_tokens=len(request.prompt.split()),
                        forward_passes=1,
                        wall_time_seconds=0.0,
                        peak_memory_bytes=0,
                    ),
                    decoding_metadata={
                        "adapter": self.ADAPTER_NAME,
                        "adapter_version": self.ADAPTER_VERSION,
                    },
                )
            )
        return tuple(responses)

    def _denoise(
        self,
        requests: tuple[ModelRequest, ...],
        *,
        steps: int,
        schedule: str,
        mask_policy: str,
    ) -> Sequence[ModelResponse]:
        metadata: dict[str, JsonValue] = {
            "adapter": self.ADAPTER_NAME,
            "adapter_version": self.ADAPTER_VERSION,
            "denoising_steps": steps,
            "schedule": schedule,
            "mask_policy": mask_policy,
        }
        return tuple(
            self._generation_response(
                request,
                denoising_steps=steps,
                decoding_metadata=metadata,
            )
            for request in requests
        )
