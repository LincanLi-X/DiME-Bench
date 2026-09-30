"""Adapter capability declarations and explicit capability errors."""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """Operations or task modes an adapter can execute."""

    GENERATE = "generate"
    SCORE_OPTIONS = "score_options"
    INFILL = "infill"
    EDIT = "edit"
    REASONING = "reasoning"
    DENOISE = "denoise"


class AdapterError(RuntimeError):
    """Base exception for adapter lifecycle and contract failures."""


class AdapterStateError(AdapterError):
    """Raised when an operation is called in the wrong lifecycle state."""


class AdapterDependencyError(AdapterError):
    """Raised when an optional runtime dependency is unavailable."""


class AdapterContractError(AdapterError):
    """Raised when an adapter violates the common request/response contract."""


class UnsupportedCapabilityError(AdapterError):
    """Raised when an adapter is asked to perform an unsupported operation."""

    def __init__(
        self,
        adapter_name: str,
        capability: Capability,
        supported: frozenset[Capability],
    ) -> None:
        available = ", ".join(sorted(item.value for item in supported)) or "<none>"
        super().__init__(
            f"adapter {adapter_name!r} does not support capability "
            f"{capability.value!r}; supported: {available}"
        )
        self.adapter_name = adapter_name
        self.capability = capability
        self.supported = supported


def parse_capability(value: Capability | str) -> Capability:
    """Normalize a public capability argument with an actionable error."""
    if isinstance(value, Capability):
        return value
    try:
        return Capability(value)
    except ValueError as exc:
        available = ", ".join(capability.value for capability in Capability)
        raise ValueError(
            f"unknown capability {value!r}; expected one of: {available}"
        ) from exc
