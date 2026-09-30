"""Lifecycle-safe, model-independent adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum

from dimebench.models.capabilities import (
    AdapterContractError,
    AdapterStateError,
    Capability,
    UnsupportedCapabilityError,
    parse_capability,
)
from dimebench.schemas.model import ModelFamily, ModelSpec
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse


class AdapterState(str, Enum):
    """Observable adapter lifecycle state."""

    CREATED = "created"
    LOADED = "loaded"
    CLOSED = "closed"


_MODE_CAPABILITIES = {
    "generate": Capability.GENERATE,
    "infill": Capability.INFILL,
    "edit": Capability.EDIT,
    "reasoning": Capability.REASONING,
}


class ModelAdapter(ABC):
    """Common adapter facade with validated lifecycle and batch contracts."""

    ADAPTER_NAME = "base"
    ADAPTER_VERSION = "1.0.0"
    MODEL_FAMILY: ModelFamily | None = None
    CAPABILITIES: frozenset[Capability] = frozenset()

    def __init__(self, spec: ModelSpec) -> None:
        if self.MODEL_FAMILY is not None and spec.family != self.MODEL_FAMILY:
            raise AdapterContractError(
                f"adapter {self.ADAPTER_NAME!r} requires model family "
                f"{self.MODEL_FAMILY!r}, got {spec.family!r}"
            )
        if spec.adapter_version != self.ADAPTER_VERSION:
            raise AdapterContractError(
                f"adapter {self.ADAPTER_NAME!r} implementation version "
                f"{self.ADAPTER_VERSION!r} does not match configured version "
                f"{spec.adapter_version!r}"
            )
        declared = frozenset(parse_capability(item) for item in spec.capabilities)
        undeclared = declared - self.CAPABILITIES
        if undeclared:
            names = ", ".join(sorted(item.value for item in undeclared))
            raise AdapterContractError(
                f"model spec declares capabilities unsupported by adapter "
                f"{self.ADAPTER_NAME!r}: {names}"
            )
        self.spec = spec
        self._state = AdapterState.CREATED

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def is_loaded(self) -> bool:
        return self._state == AdapterState.LOADED

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def supports(self, capability: Capability | str) -> bool:
        """Return whether the adapter implements a capability."""
        return parse_capability(capability) in self.CAPABILITIES

    def require(self, capability: Capability | str) -> None:
        """Raise a specific error when a capability is unavailable."""
        normalized = parse_capability(capability)
        if normalized not in self.CAPABILITIES:
            raise UnsupportedCapabilityError(
                self.ADAPTER_NAME,
                normalized,
                self.CAPABILITIES,
            )

    def load(self) -> None:
        """Load model resources once; repeated calls are idempotent."""
        if self._state == AdapterState.LOADED:
            return
        self._load()
        self._state = AdapterState.LOADED

    def close(self) -> None:
        """Release resources; repeated calls are idempotent."""
        if self._state == AdapterState.CLOSED:
            return
        if self._state == AdapterState.LOADED:
            self._close()
        self._state = AdapterState.CLOSED

    def generate(self, requests: Sequence[ModelRequest]) -> tuple[ModelResponse, ...]:
        """Generate ordered responses for a homogeneous request batch."""
        self.require(Capability.GENERATE)
        self._require_loaded()
        batch = tuple(requests)
        self._validate_request_batch(batch, allow_score_options=False)
        for request in batch:
            required = _MODE_CAPABILITIES.get(request.mode)
            if required is not None and required != Capability.GENERATE:
                self.require(required)
        responses = tuple(self._generate(batch))
        self._validate_response_batch(batch, responses, score_options=False)
        return responses

    def score_options(
        self,
        requests: Sequence[ModelRequest],
    ) -> tuple[ModelResponse, ...]:
        """Score answer options when explicitly supported by the adapter."""
        self.require(Capability.SCORE_OPTIONS)
        self._require_loaded()
        batch = tuple(requests)
        self._validate_request_batch(batch, allow_score_options=True)
        if any(request.mode != "score_options" for request in batch):
            raise AdapterContractError(
                "score_options accepts only requests with mode='score_options'"
            )
        responses = tuple(self._score_options(batch))
        self._validate_response_batch(batch, responses, score_options=True)
        return responses

    def _require_loaded(self) -> None:
        if self._state != AdapterState.LOADED:
            raise AdapterStateError(
                f"adapter {self.ADAPTER_NAME!r} must be loaded before inference; "
                f"current state: {self._state.value}"
            )

    def _validate_request_batch(
        self,
        requests: tuple[ModelRequest, ...],
        *,
        allow_score_options: bool,
    ) -> None:
        request_ids = [request.request_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise AdapterContractError(
                "request_id values must be unique within a batch"
            )
        if not allow_score_options and any(
            request.mode == "score_options" for request in requests
        ):
            raise AdapterContractError(
                "score_options requests must use adapter.score_options()"
            )

    def _validate_response_batch(
        self,
        requests: tuple[ModelRequest, ...],
        responses: tuple[ModelResponse, ...],
        *,
        score_options: bool,
    ) -> None:
        if len(responses) != len(requests):
            raise AdapterContractError(
                f"adapter returned {len(responses)} responses for "
                f"{len(requests)} requests"
            )
        for index, (request, response) in enumerate(
            zip(requests, responses, strict=True)
        ):
            if response.request_id != request.request_id:
                raise AdapterContractError(
                    f"response {index} request_id {response.request_id!r} does not "
                    f"match {request.request_id!r}"
                )
            if response.sample_id != request.sample_id:
                raise AdapterContractError(
                    f"response {index} sample_id {response.sample_id!r} does not "
                    f"match {request.sample_id!r}"
                )
            if response.model_id != self.spec.id:
                raise AdapterContractError(
                    f"response {index} model_id {response.model_id!r} does not "
                    f"match {self.spec.id!r}"
                )
            if score_options and response.status == "success":
                if response.option_scores is None:
                    raise AdapterContractError(
                        f"response {index} is missing option_scores"
                    )
                if set(response.option_scores) != set(request.options):
                    raise AdapterContractError(
                        f"response {index} option scores do not match request options"
                    )

    @abstractmethod
    def _load(self) -> None:
        """Allocate implementation-specific model resources."""

    @abstractmethod
    def _generate(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        """Implement generation without repeating facade validation."""

    def _score_options(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        raise UnsupportedCapabilityError(
            self.ADAPTER_NAME,
            Capability.SCORE_OPTIONS,
            self.CAPABILITIES,
        )

    @abstractmethod
    def _close(self) -> None:
        """Release implementation-specific resources."""

    def __enter__(self) -> ModelAdapter:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
