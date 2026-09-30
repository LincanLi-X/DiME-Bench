"""Base interface for discrete diffusion language model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from dimebench.models.base import ModelAdapter
from dimebench.models.capabilities import AdapterContractError, Capability
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse


class DiffusionModelAdapter(ModelAdapter, ABC):
    """Family-constrained adapter with validated denoising controls."""

    MODEL_FAMILY = "diffusion"

    def denoise(
        self,
        requests: Sequence[ModelRequest],
        *,
        steps: int,
        schedule: str,
        mask_policy: str,
    ) -> tuple[ModelResponse, ...]:
        """Run iterative denoising under an explicit fixed budget."""
        self.require(Capability.DENOISE)
        self._require_loaded()
        if steps <= 0:
            raise AdapterContractError("denoising steps must be positive")
        if not schedule:
            raise AdapterContractError("denoising schedule cannot be empty")
        if not mask_policy:
            raise AdapterContractError("mask_policy cannot be empty")
        batch = tuple(requests)
        self._validate_request_batch(batch, allow_score_options=False)
        for request in batch:
            if request.mode == "score_options":
                raise AdapterContractError(
                    "score_options requests cannot use denoise()"
                )
            required = {
                "infill": Capability.INFILL,
                "edit": Capability.EDIT,
                "reasoning": Capability.REASONING,
            }.get(request.mode)
            if required is not None:
                self.require(required)
        responses = tuple(
            self._denoise(
                batch,
                steps=steps,
                schedule=schedule,
                mask_policy=mask_policy,
            )
        )
        self._validate_response_batch(batch, responses, score_options=False)
        return responses

    @abstractmethod
    def _denoise(
        self,
        requests: tuple[ModelRequest, ...],
        *,
        steps: int,
        schedule: str,
        mask_policy: str,
    ) -> Sequence[ModelResponse]:
        """Implement mechanism-specific iterative denoising."""
